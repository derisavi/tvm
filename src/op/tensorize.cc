/*!
 *  Copyright (c) 2017 by Contributors
 * \brief Logics related to tensorize, used by ComputeOpNode.
 * \file tensorize.cc
 */
#include <tvm/ir.h>
#include <tvm/ir_mutator.h>
#include <tvm/ir_pass.h>
#include <tvm/api_registry.h>
#include "op_util.h"
#include "compute_op.h"
#include "../schedule/message_passing.h"

namespace tvm {

using namespace ir;
using namespace op;

// Detect the region of input and output to be tensorized.
// out_dom: the domain of root iter vars in output op
// in_region: region of each input tensor.
// return The location of the tensorized scope start.
size_t InferTensorizeRegion(
    const ComputeOpNode* self,
    const Stage& stage,
    const std::unordered_map<IterVar, Range>& dom_map,
    std::unordered_map<IterVar, Range>* out_dom,
    std::unordered_map<Tensor, Array<Range> >* in_region) {
  // Get the bound of the tensorized scope.
  bool found_point = false;
  size_t loc_scope = 0;
  std::unordered_map<IterVar, IntSet> up_state;
  // Loop over the leafs (iteration variables) starting from the innermost loop
  // to find the loopnest to be tensorized
  for (size_t i = stage->leaf_iter_vars.size(); i != 0; --i) {
    IterVar iv = stage->leaf_iter_vars[i - 1];
    CHECK(iv->iter_type == kDataPar ||
          iv->iter_type == kCommReduce);
    auto vit = dom_map.find(iv);
    CHECK(vit != dom_map.end());
    const Range& vrange = vit->second;
    if (is_one(vrange->extent)) {
      up_state[iv] = IntSet::single_point(vrange->min);
    } else if (found_point) {
      // all loops within the tensorized loopnest must be normalized
      CHECK(is_zero(vrange->min));
      up_state[iv] = IntSet::single_point(iv->var);
    } else {
      up_state[iv] = IntSet::range(vrange);
    }
    auto iit = stage->iter_var_attrs.find(iv);
    if (iit != stage->iter_var_attrs.end()) {
      const IterVarAttr& attr = (*iit).second;
      if (!found_point) {
        CHECK(!attr->bind_thread.defined())
            << "Do not allow thread in tensorize scope";
      }
      if (attr->iter_type == kTensorized) {
        CHECK(!found_point) << "Do not allow two tensorized point";
        found_point = true;
        loc_scope = i - 1;
      }
    }
  }
  CHECK(found_point);
  // Get domain of the tensorized scope.
  schedule::PassUpDomain(stage, dom_map, &up_state);
  for (auto& e : up_state) {
    std::cout << "iv: " << e.first << "  up_state: " << e.second << std::endl;
  }
  // Get domains of inputs
  std::unordered_map<Tensor, TensorDom> in_dom;
  std::unordered_map<const Variable*, IntSet> temp_dmap;
  Array<Tensor> inputs = self->InputTensors();
  for (Tensor t : inputs) {
    in_dom.emplace(t, TensorDom(t.ndim()));
  }
  for (IterVar iv : self->root_iter_vars()) {
    IntSet iset = up_state.at(iv);
    (*out_dom)[iv] = iset.cover_range(dom_map.at(iv));
    std::cout << "iv: " << iv << "  Range: " << (*out_dom)[iv] << std::endl;
    temp_dmap[iv->var.get()] = iset;
  }
  // Input domains
  self->PropBoundToInputs(stage->op, temp_dmap, &in_dom);
  Range none;
  for (const auto& kv : in_dom) {
    Array<Range> vec;
    const Tensor& t = kv.first;
    for (size_t i = 0; i < t.ndim(); ++i) {
      Range r = arith::Union(kv.second.data.at(i)).cover_range(none);
      CHECK(r.defined()) << "cannot deduce region of tensorized scope for input " << t;
      vec.push_back(std::move(r));
    }
    std::cout << "Tensor: " << t << std::endl;
    for (auto& e : vec)
      std::cout << "    " << e << std::endl;
    (*in_region)[t] = std::move(vec);
  }
  return loc_scope;
}

void VerifyTensorizeLoopNest2(const For* top_for_loop) {
  // collects predicates and iteration variables in the tensorized scope and
  // verifies that none of the variables aree used in the predicates
  CollectPredicatesAndIterVars collectPredicates;
  collectPredicates.Mutate(top_for_loop);
}

void VerifyTensorizeLoopNest(const ComputeOpNode* self,
                             const Stage& stage,
                             const ComputeLoopNest& n,
                             size_t tloc) {
  // Verification step.
  std::unordered_set<const Variable*> banned;
  // The following two conditions are always true by construction
  // and they are independent of the intrinsic pattern
  CHECK_EQ(n.main_nest.size(), stage->leaf_iter_vars.size() + 1);
  CHECK(n.init_nest.size() == stage->leaf_iter_vars.size() + 1 ||
        n.init_nest.size() == 0);

  // Collect the list of banned variables, i.e., all For iteration variables,
  // attribute variables, and Let variables in the tensorize scope
  auto f_push_banned = [&banned](const Stmt& s) {
    if (const For* op = s.as<For>()) {
        banned.insert(op->loop_var.get());
    } else if (const AttrStmt* op = s.as<AttrStmt>()) {
      if (const IterVarNode* iv = op->node.as<IterVarNode>()) {
        banned.insert(iv->var.get());
      }
    } else if (const LetStmt* op = s.as<LetStmt>()) {
      banned.insert(op->var.get());
    }
  };
  for (size_t i = tloc; i < stage->leaf_iter_vars.size(); ++i) {
    for (const Stmt& s : n.main_nest[i + 1]) {
      f_push_banned(s);
    }
    if (n.init_nest.size() != 0) {
      for (const Stmt& s : n.init_nest[i + 1]) {
        f_push_banned(s);
      }
    }
  }

  // Make sure banned variables do not appear in split conditions
  for (const Expr& pred : n.main_predicates) {
    if (ir::ExprUseVar(pred, banned)) {
      LOG(FATAL) << "Tensorize failed, split condition "
                 << pred << " relies on var defined inside tensorize scope";
    }
  }
  for (const Expr& pred : n.init_predicates) {
    if (ir::ExprUseVar(pred, banned)) {
      LOG(FATAL) << "Tensorize failed, split condition "
                 << pred << " relies on var defined inside tensorize scope";
    }
  }
}


// During initialization, computes a mapping from actual (tensorize) IVs
// to intrinsic IVs (both regular and reduction IVs)
// and a mapping from input tensors of actual to input tensor of intrin.
// During mutation, transforms, using the mappings above, an expression of the body of the actual
// to how it should look like in the intrinsic
class TensorIntrinMatcher final : public IRMutator {
 public:
  Expr Mutate_(const Call* op, const Expr& e) final {
    Expr expr = IRMutator::Mutate_(op, e);
    op = expr.as<Call>();
    if (op->call_type == Call::Halide) {
      Tensor t = Operation(op->func.node_).output(op->value_index);
      auto it = in_remap_.find(t);
      if (it != in_remap_.end()) {
        const InputEntry& e = it->second;
        CHECK_EQ(op->args.size(), e.region.size());
        Array<Expr> args;
        for (size_t i = e.start; i < e.region.size(); ++i) {
          args.push_back(op->args[i] - e.region[i]->min);
        }
        return Call::make(
            op->type, e.tensor->op->name, args,
            op->call_type, e.tensor->op, e.tensor->value_index);
      }
    }
    return expr;
  }

  Expr Mutate_(const Variable* op, const Expr& e) final {
    auto it = var_remap_.find(op);
    if (it != var_remap_.end()) {
      return it->second;
    } else {
      return e;
    }
  }

  Expr Mutate_(const Reduce* op, const Expr& e) final {
    Expr expr = IRMutator::Mutate_(op, e);
    op = expr.as<Reduce>();
    Array<IterVar> axis;
    for (size_t i = 0; i < op->axis.size(); ++i) {
      auto it = axis_remap_.find(op->axis[i]);
      if (it != axis_remap_.end()) {
        axis.push_back(it->second);
      }
    }
    return Reduce::make(
        op->combiner, op->source, axis, op->condition, op->value_index);
  }

  void Init(const ComputeOpNode* self,
            const Stage& stage,
            const std::unordered_map<IterVar, Range>& out_dom,
            const std::unordered_map<Tensor, Array<Range> >& in_region,
            const TensorIntrin& intrin,
            Map<Var, Range>* compute_intrin_iter_space) {
    CHECK(self == stage->op.get());
    // TODO: if we do this matching after loops with trip_count=1 are simplified
    // we don't need to match [1,m,n] with [m,n]

    // build the mapping from input tensors of the tensorize scope (i.e., actual)
    // to input tensors of the intrinsic
    Array<Tensor> inputs = self->InputTensors();
    CHECK_EQ(inputs.size(), intrin->inputs.size());
    for (size_t i = 0; i < inputs.size(); ++i) {
      InputEntry e;
      /* change InputEntry::tensor to InputEntry::intrin_tensor
       * change InputEntry::region to InputEntry::actual_region
       */
      e.tensor = intrin->inputs[i];
      e.region = Array<Range>(in_region.at(inputs[i]));
      CHECK_GE(e.region.size(), e.tensor.ndim());
      // Enable fuzzy matching, to match [1, n, m] to [n, m]
      e.start = e.region.size() - e.tensor.ndim();
      for (size_t i = 0; i < e.start; ++i) {
        CHECK(is_one(e.region[i]->extent))
            << "Tensorize " << intrin->name << ":"
            << " Input dimension mismatch with tensor intrin "
            << " expected shape=" << e.tensor->shape
            << ", given region=" << e.region;
      }
      // rename in_remap_ to actual_to_intrin_in_tensor_map_
      in_remap_[inputs[i]] = e;
    }

    // Ensure that the shapes of the output tensors of the stage to be matched
    // and the intrinsics are compatible
    const ComputeOpNode* intrin_compute = intrin->op.as<ComputeOpNode>();
    CHECK(intrin_compute) << "Only support compute intrinsic for now";
    CHECK_GE(self->axis.size(), intrin_compute->axis.size())
        << "Tensorize: Output mismatch with tensor intrin ";
    // Enable fuzzy matching, to match [1, n, m] to [n, m]
    size_t axis_start = self->axis.size() - intrin_compute->axis.size();
    for (size_t i = 0; i < axis_start; ++i) {
      Range r = out_dom.at(self->axis[i]);
      CHECK(is_one(r->extent))
          << "Tensorize: Output mismatch with tensor intrin "
          << " intrin-dim=" << intrin_compute->axis.size()
          << ", tensorize-dim=" << self->axis.size();
      var_remap_[self->axis[i]->var.get()] = r->min;
    }
    // Assume we tensorize at region axis i [min, min + extent)
    // The corresponding intrinsic axis is j [0, extent)
    // Remap index i to j + min
    for (size_t i = axis_start; i < self->axis.size(); ++i) {
      IterVar iv = self->axis[i];
      IterVar target_iv = intrin_compute->axis[i - axis_start];
      Range r = out_dom.at(iv);
      var_remap_[iv->var.get()] = target_iv->var + r->min;
      axis_remap_[iv] = target_iv;
      compute_intrin_iter_space->Set(target_iv->var, target_iv->dom);
    }
    // Remap reduction axis
    CHECK_GE(self->reduce_axis.size(), intrin_compute->reduce_axis.size())
        << "Tensorize: Reduction dimension mismatch with tensor intrin";
    axis_start = self->reduce_axis.size() - intrin_compute->reduce_axis.size();
    for (size_t i = 0; i < axis_start; ++i) {
      Range r = out_dom.at(self->reduce_axis[i]);
      CHECK(is_one(r->extent))
          << "Tensorize: Reduction mismatch with tensor intrin "
          << " intrin-dim=" << intrin_compute->reduce_axis.size()
          << ", tensorize-dim=" << self->reduce_axis.size();
      var_remap_[self->reduce_axis[i]->var.get()] = r->min;
    }
    for (size_t i = axis_start; i < self->reduce_axis.size(); ++i) {
      IterVar iv = self->reduce_axis[i];
      IterVar target_iv = intrin_compute->reduce_axis[i - axis_start];
      Range r = out_dom.at(iv);
      var_remap_[iv->var.get()] = target_iv->var + r->min;
      axis_remap_[iv] = target_iv;
      compute_intrin_iter_space->Set(target_iv->var, target_iv->dom);
    }
  }

 private:
  // Input entry
  struct InputEntry {
    Tensor tensor;
    size_t start;
    Array<Range> region;
  };
  // input data remap
  std::unordered_map<Tensor, InputEntry> in_remap_;
  // mapping from iteration variables of the tensorize (actual) scope to a linear
  // expression of
  // the corresponding iteration variable of the intrinsic (includes both regular
  // and reduction
  // iteration variables)
  std::unordered_map<const Variable*, Expr> var_remap_;
  // mapping from iteration variable of the tensorize (actual) scope to the corresponding
  // iteration variable of the intrinsic (includes both regular and reduction iteration
  // variables)
  std::unordered_map<IterVar, IterVar> axis_remap_;
};

// Try to match tensor dataflow of the stage with the intrinsic
Array<Expr> MatchTensorizeBody(
    const ComputeOpNode* self,
    const Stage& stage,
    const std::unordered_map<IterVar, Range>& out_dom,
    const std::unordered_map<Tensor, Array<Range> >& in_region,
    const TensorIntrin& intrin,
    Map<Var, Range>* compute_intrin_iter_space) {
  TensorIntrinMatcher matcher;
  matcher.Init(self, stage, out_dom, in_region, intrin, compute_intrin_iter_space);
  Array<Expr> ret;
  // Using the mapping from actual IVs to intrinsic IVs, and mapping from
  // input tensors of actual (tensorize) scope to input tensors of intrinsic,
  // transform each expression in the actual body to what it would look like
  // in the intrinsic
  for (Expr expr : self->body) {
    ret.push_back(matcher.Mutate(expr));
  }
  return ret;
}

void VerifyTensorizeBody2(
    const For* outermost_loop,
    const std::unordered_map<IterVar, Range>& out_dom,
    const std::unordered_map<Tensor, Array<Range> >& in_region,
    const TensorIntrin& intrin)
{
}

void VerifyTensorizeBody(
    const ComputeOpNode* self,
    const Stage& stage,
    const std::unordered_map<IterVar, Range>& out_dom,
    const std::unordered_map<Tensor, Array<Range> >& in_region,
    const TensorIntrin& intrin) {
  Map<Var, Range> compute_intrin_iter_space;
  Array<Expr> body = MatchTensorizeBody(self, stage, out_dom, in_region, intrin,
                                        &compute_intrin_iter_space);
  const ComputeOpNode* intrin_compute = intrin->op.as<ComputeOpNode>();
  CHECK(intrin_compute) << "Only support compute intrinsic for now";
  CHECK_EQ(body.size(), intrin_compute->body.size())
      << "Tensorize failed: body size mismatch";
  for (size_t i = 0; i < body.size(); ++i) {
    Expr lhs = Simplify(body[i], compute_intrin_iter_space);
    lhs = CanonicalSimplify(lhs, compute_intrin_iter_space);
    Expr rhs = Simplify(intrin_compute->body[i], compute_intrin_iter_space);
    rhs = CanonicalSimplify(rhs, compute_intrin_iter_space);
    if (lhs.type() != rhs.type()) {
      LOG(FATAL)
          << "Failed to match the data type with TensorIntrin "
          << intrin->name << "'s declaration "
          << " provided=" << lhs.type()
          << ", intrin=" << rhs.type();
    }
    CHECK(Equal(lhs, rhs))
        << "Failed to match the compute with TensorIntrin "
        << intrin->name << "'s declaration "
        << " provided= " << lhs
        << ", intrin=  " << rhs;
  }
}

Stmt MakeTensorize2(const For* outermost_loop, const TensorIntrin& intrin) {
  std::unordered_map<IterVar, Range> out_dom;
  std::unordered_map<Tensor, Array<Range> > in_region;
  InferTensorizeRegion2()
  //size_t tloc = InferTensorizeRegion(self, stage, dom_map, &out_dom, &in_region);
//  const IterVar& tensorize_outermost_loop = op->loop_var
  VerifyTensorizeLoopNest2(outermost_loop);
  VerifyTensorizeBody2(outermost_loop, out_dom, in_region, intrin);

}

Stmt MakeTensorize(const ComputeOpNode* self,
                   const Stage& stage,
                   const std::unordered_map<IterVar, Range>& dom_map,
                   bool debug_keep_trivial_loop) {
  std::unordered_map<IterVar, Range> out_dom;
  std::unordered_map<Tensor, Array<Range> > in_region;
  size_t tloc = InferTensorizeRegion(self, stage, dom_map, &out_dom, &in_region);
  const auto& tensorize_outermost_loop = stage->leaf_iter_vars[tloc];
  TensorIntrin intrin = stage->iter_var_attrs.at(
      tensorize_outermost_loop)->tensor_intrin;
  CHECK(intrin.defined());
  ComputeLoopNest n = ComputeLoopNest::make(self, stage, dom_map, debug_keep_trivial_loop);
  VerifyTensorizeLoopNest(self, stage, n, tloc);
  VerifyTensorizeBody(self, stage, out_dom, in_region, intrin);
  // Start bind data.
  Stmt nop = Evaluate::make(0);
  std::vector<Stmt> input_bind_nest, output_bind_nest;
  Array<Tensor> inputs = self->InputTensors();
  CHECK_EQ(inputs.size(), intrin->inputs.size())
      << "Tensorize failed: input size mismatch ";
  // input binding
  for (size_t i = 0; i < intrin->inputs.size(); ++i) {
    Tensor tensor = inputs[i];
    Buffer buffer = intrin->buffers[i];
    Array<NodeRef> bind_spec{buffer, tensor};
    auto it = in_region.find(tensor);
    CHECK(it != in_region.end());
    const Array<Range>& region = it->second;
    Array<Expr> tuple;
    for (const Range r : region) {
      tuple.push_back(r->min);
      tuple.push_back(r->extent);
    }
    input_bind_nest.emplace_back(AttrStmt::make(
        bind_spec, ir::attr::buffer_bind_scope,
        Call::make(Handle(), ir::intrinsic::tvm_tuple, tuple, Call::Intrinsic), nop));
  }
  // output binding
  const ComputeOpNode* intrin_compute = intrin->op.as<ComputeOpNode>();
  CHECK(intrin_compute) << "Only support compute intrinsic for now";
  CHECK_EQ(intrin->inputs.size() + intrin_compute->body.size(), intrin->buffers.size());
  CHECK_EQ(intrin_compute->body.size(), self->body.size());
  Array<Expr> tuple;
  for (IterVar iv : self->axis) {
    auto it = out_dom.find(iv);
    CHECK(it != out_dom.end());
    tuple.push_back(it->second->min);
    tuple.push_back(it->second->extent);
  }
  for (size_t i = intrin->inputs.size(); i < intrin->buffers.size(); ++i) {
    Tensor tensor = stage->op.output(i - intrin->inputs.size());
    Buffer buffer = intrin->buffers[i];
    Array<NodeRef> bind_spec{buffer, tensor};
    output_bind_nest.emplace_back(AttrStmt::make(
        bind_spec, ir::attr::buffer_bind_scope,
        Call::make(Handle(), ir::intrinsic::tvm_tuple, tuple, Call::Intrinsic), nop));
  }
  // Check variable remap
  std::unordered_map<const Variable*, Expr> vmap;
  ir::ArgBinder binder(&vmap);
  CHECK_GE(self->reduce_axis.size(), intrin_compute->reduce_axis.size())
      << "Tensorization fail: reduction axis size do not match";
  size_t start = self->reduce_axis.size() - intrin_compute->reduce_axis.size();
  for (size_t i = 0; i < start; ++i) {
    IterVar iv = self->reduce_axis[i];
    auto it = out_dom.find(iv);
    CHECK(it != out_dom.end());
    CHECK(is_one(it->second->extent))
        << "Tensorization fail: reduction axis size do not match";
  }
  for (size_t i = start; i < self->reduce_axis.size(); ++i) {
    IterVar iv = self->reduce_axis[i];
    IterVar target = intrin_compute->reduce_axis[i - start];
    auto it = out_dom.find(iv);
    CHECK(it != out_dom.end());
    binder.Bind(target->dom->min, make_const(iv->dom->min.type(), 0),
                "tensor_intrin.reduction.min");
    binder.Bind(target->dom->extent, it->second->extent,
                "tensor_intrin.reduction.extent");
  }
  if (tloc <= n.num_common_loop) {
    // Do no need to split reduction
    std::vector<std::vector<Stmt> > nest(
        n.main_nest.begin(), n.main_nest.begin() + tloc + 1);
    nest.emplace_back(op::MakeIfNest(n.main_predicates));
    CHECK_EQ(n.init_predicates.size(), 0U);
    CHECK(intrin->body.defined())
        << "Normal store op for intrin " << intrin << " is not defined";
    Stmt body = MergeNest(output_bind_nest, intrin->body);
    body = MergeNest(input_bind_nest, body);
    body = Substitute(body, vmap);
    body = MergeNest(binder.asserts(), body);
    body = Substitute(body, n.main_vmap);
    return MergeNest(nest, body);
  } else {
    // Need to split reduction
    CHECK(intrin->reduce_update.defined())
        << "Reduction update op for intrin " << intrin << " is not defined";
    // Need init and update steps
    CHECK_NE(self->reduce_axis.size(), 0U);
    std::vector<std::vector<Stmt> > common(
        n.main_nest.begin(), n.main_nest.begin() + n.num_common_loop + 1);
    std::vector<std::vector<Stmt> > update_nest(
        n.main_nest.begin() + n.num_common_loop + 1, n.main_nest.begin() + tloc + 1);
    update_nest.emplace_back(op::MakeIfNest(n.main_predicates));

    if (intrin->reduce_init.defined()) {
      // init nest
      std::vector<std::vector<Stmt> > init_nest(
          n.init_nest.begin(), n.init_nest.begin() + tloc + 1);
      init_nest.emplace_back(op::MakeIfNest(n.init_predicates));
      Stmt init = MergeNest(output_bind_nest, intrin->reduce_init);
      init = Substitute(init, n.init_vmap);
      init = MergeNest(init_nest, init);
      // The update
      Stmt update = MergeNest(output_bind_nest, intrin->reduce_update);
      update = MergeNest(input_bind_nest, update);
      update = Substitute(update, vmap);
      update = MergeNest(binder.asserts(), update);
      update = Substitute(update, n.main_vmap);
      update = MergeNest(update_nest, update);
      return MergeNest(common, Block::make(init, update));
    } else {
      // When init op is not available, use body op for reset in the first iter.
      CHECK(intrin->body.defined())
          << "Normal body op for intrin " << intrin << " is not defined";
      Stmt update = TransformUpdate(stage, dom_map, n,
                                    intrin->body,
                                    intrin->reduce_update);
      update = MergeNest(output_bind_nest, update);
      update = MergeNest(input_bind_nest, update);
      update = Substitute(update, vmap);
      update = MergeNest(binder.asserts(), update);
      update = Substitute(update, n.main_vmap);
      update = MergeNest(update_nest, update);
      return MergeNest(common, update);
    }
  }
}

class Tensorizer : public IRMutator {
public:
  Stmt Mutate_(const AttrStmt* op, const Stmt& s) final {
    if (op->attr_key == "intrin") {
      const TensorIntrin* intrin = op->node.as<TensorIntrin>();
      CHECK(intrin && intrin->defined());
      const For* forLoop = op->body.as<For>();
      CHECK(forLoop && forLoop->for_type == ForType::Tensorized);
      return MakeTensorize2(forLoop, *intrin);
    } else {
      return IRMutator::Mutate_(op, s);
    }
  }
};
namespace ir {
Stmt Tensorize(Stmt stmt)
{
  return Tensorizer().Mutate(stmt);
}
}
// Register functions for unittests
TVM_REGISTER_API("test.op.InferTensorizeRegion")
.set_body([](TVMArgs args, TVMRetValue* ret) {
    Stage stage = args[0];
    Map<IterVar, Range> dmap = args[1];
    std::unordered_map<IterVar, Range> out_dom;
    std::unordered_map<Tensor, Array<Range> > in_region;
    CHECK(stage->op.as<ComputeOpNode>());
    InferTensorizeRegion(stage->op.as<ComputeOpNode>(),
                         stage,
                         as_unordered_map(dmap),
                         &out_dom, &in_region);
    *ret = Array<NodeRef>{Map<IterVar, Range>(out_dom),
                          Map<Tensor, Array<Range> >(in_region)};
  });

TVM_REGISTER_API("test.op.MatchTensorizeBody")
.set_body([](TVMArgs args, TVMRetValue* ret) {
    Stage stage = args[0];
    Map<IterVar, Range> out_dom = args[1];
    Map<Tensor, Array<Range> > in_region = args[2];
    TensorIntrin intrin = args[3];
    Map<Var, Range> vrange;
    CHECK(stage->op.as<ComputeOpNode>());
    *ret = MatchTensorizeBody(stage->op.as<ComputeOpNode>(),
                              stage,
                              as_unordered_map(out_dom),
                              as_unordered_map(in_region),
                              intrin,
                              &vrange);
  });
}  // namespace tvm
