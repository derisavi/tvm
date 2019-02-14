/*!
 *  Copyright (c) 2017 by Contributors
 * \brief Tensor Compute Op.
 * \file tensor_compute_op.cc
 */
#include <tvm/operation.h>
#include <tvm/arithmetic.h>
#include <tvm/ir.h>
#include <tvm/ir_visitor.h>
#include <tvm/ir_pass.h>
#include <unordered_set>
#include "./op_util.h"
#include "./compute_op.h"
#include "../arithmetic/compute_expr.h"

namespace tvm {
using namespace ir;
// TensorComputeOpNode
TVM_STATIC_IR_FUNCTOR(IRPrinter, vtable)
.set_dispatch<TensorComputeOpNode>([](const TensorComputeOpNode *op,
                                      IRPrinter *p) {
    p->stream << "tensor_compute_op(" << op->name << ", " << op << ")";
  });

TVM_REGISTER_NODE_TYPE(TensorComputeOpNode);

int TensorComputeOpNode::num_outputs() const {
  return static_cast<int>(this->intrin->buffers.size() - this->inputs.size());
}

Type TensorComputeOpNode::output_dtype(size_t i) const {
  return this->intrin->buffers[this->inputs.size() + i]->dtype;
}

Operation TensorComputeOpNode::make(std::string name,
                                    std::string tag,
                                    Array<IterVar> axis,
                                    Array<IterVar> reduce_axis,
                                    int schedulable_ndim,
                                    TensorIntrin intrin,
                                    Array<Tensor> tensors,
                                    Array<Region> regions) {
  auto n = make_node<TensorComputeOpNode>();
  n->name = std::move(name);
  n->tag = std::move(tag);
  n->axis = std::move(axis);
  n->reduce_axis = std::move(reduce_axis);
  n->schedulable_ndim = std::move(schedulable_ndim);
  n->intrin = std::move(intrin);
  n->inputs = std::move(tensors);
  n->input_regions = std::move(regions);
  return Operation(n);
}

Array<Tensor> TensorComputeOpNode::InputTensors() const {
  return inputs;
}

Operation TensorComputeOpNode::ReplaceInputs(
    const Operation& self,
    const std::unordered_map<Tensor, Tensor>& rmap) const {
  CHECK_EQ(self.operator->(), this);
  auto n = make_node<TensorComputeOpNode>(*this);
  auto intrin = make_node<TensorIntrinNode>(*(this->intrin.operator->()));
  intrin->body = op::ReplaceTensor(this->intrin->body, rmap);
  if (intrin->reduce_init.defined()) {
    intrin->reduce_init = op::ReplaceTensor(this->intrin->reduce_init, rmap);
  }
  if (intrin->reduce_update.defined()) {
    intrin->reduce_update = op::ReplaceTensor(this->intrin->reduce_update, rmap);
  }
  for (size_t i = 0; i < n->inputs.size(); ++i) {
    Tensor t = n->inputs[i];
    if (rmap.count(t)) {
      n->inputs.Set(i, rmap.at(t));
    }
  }

  if (intrin->body.same_as(n->intrin->body) &&
      intrin->reduce_init.same_as(n->intrin->reduce_init) &&
      intrin->reduce_update.same_as(n->intrin->reduce_update) &&
      inputs.same_as(n->inputs)) {
    return self;
  } else {
    n->intrin = TensorIntrin(intrin);
    return Operation(n);
  }
}

void TensorComputeOpNode::PropBoundToInputs(
    const Operation& self,
    const std::unordered_map<const Variable*, IntSet>& dom_map,
    std::unordered_map<Tensor, TensorDom>* out_dom_map) const {
  for (size_t i = 0; i < this->inputs.size(); ++i) {
    Tensor t = this->inputs[i];
    Region region = input_regions[i];

    auto it = out_dom_map->find(t);
    if (it == out_dom_map->end()) continue;
    TensorDom& dom = it->second;
    for (size_t j = 0; j < t.ndim(); ++j) {
      dom.data[j].emplace_back(EvalSet(region[j], dom_map));
    }
  }
}

size_t TensorComputeOpNode::num_schedulable_dims() const {
  return schedulable_ndim;
}

Stmt TensorComputeOpNode::BuildProvide(
    const Stage& stage,
    const std::unordered_map<IterVar, Range>& dom_map,
    bool debug_keep_trivial_loop) const {
  CHECK_EQ(stage->op.operator->(), this);

  // Start bind data.
  Stmt nop = Evaluate::make(0);
  std::vector<Stmt> input_bind_nest, output_bind_nest;
  Array<Tensor> inputs = this->InputTensors();

  // input binding
  size_t num_inputs = inputs.size();
  for (size_t i = 0; i < num_inputs; ++i) {
    Tensor tensor = inputs[i];
    Region region = this->input_regions[i];
    Buffer buffer = this->intrin->buffers[i];
    BuildInputBinding(input_bind_nest, tensor, region, buffer);
  }

  // output binding
  for (int i = 0; i < this->num_outputs(); ++i) {
    Tensor tensor = stage->op.output(i);
    Buffer buffer = this->intrin->buffers[num_inputs + i];
    Array<NodeRef> bind_spec{buffer, tensor};

    Array<Expr> tuple;
    for (size_t i = 0; i < this->axis.size(); ++i) {
      auto ivar = this->axis[i];
      if (i < static_cast<size_t>(this->schedulable_ndim)) {
        tuple.push_back(ivar->var);
        tuple.push_back(1);
      } else {
        Range dom = ivar->dom;
        tuple.push_back(dom->min);
        tuple.push_back(dom->extent);
      }
    }

    output_bind_nest.emplace_back(AttrStmt::make(
        bind_spec, ir::attr::buffer_bind_scope,
        Call::make(Handle(), ir::intrinsic::tvm_tuple, tuple, Call::Intrinsic), nop));
  }

  // Check variable remap
  std::unordered_map<const Variable*, Expr> vmap;
  ir::ArgBinder binder(&vmap);

  size_t tloc = stage->leaf_iter_vars.size();
  ComputeLoopNest n = ComputeLoopNest::make(this, stage, dom_map, debug_keep_trivial_loop);

  if (this->reduce_axis.size() == 0) {
    std::vector<std::vector<Stmt> > nest(
        n.main_nest.begin(), n.main_nest.begin() + tloc + 1);
    nest.emplace_back(op::MakeIfNest(n.main_predicates));
    CHECK_EQ(n.init_predicates.size(), 0U);
    CHECK(this->intrin->body.defined())
        << "Normal store op for intrin " << this << " is not defined";
    Stmt body = MergeNests(this->intrin->body, output_bind_nest, input_bind_nest, binder.asserts(), nest);
    body = ir::Substitute(body, vmap);
    body = op::Substitute(body, n.main_vmap);
    return body;
  } else {
    // Need to split reduction
    CHECK(this->intrin->reduce_update.defined())
        << "Reduction update op is not defined";
    // Need init and update steps
    CHECK_NE(this->reduce_axis.size(), 0U);
    std::vector<std::vector<Stmt> > common(
        n.main_nest.begin(), n.main_nest.begin() + n.num_common_loop + 1);
    std::vector<std::vector<Stmt> > update_nest(
        n.main_nest.begin() + n.num_common_loop + 1, n.main_nest.begin() + tloc + 1);
    update_nest.emplace_back(op::MakeIfNest(n.main_predicates));

    if (this->intrin->reduce_init.defined()) {
      // init nest
      std::vector<std::vector<Stmt> > init_nest(
          n.init_nest.begin(), n.init_nest.begin() + tloc + 1);
      init_nest.emplace_back(op::MakeIfNest(n.init_predicates));
      Stmt init = MergeNests(this->intrin->reduce_init, output_bind_nest, init_nest);
      init = op::Substitute(init, n.init_vmap);
      // The update
      Stmt update = MergeNests(this->intrin->reduce_update, output_bind_nest, input_bind_nest,
                               binder.asserts(), update_nest);
      update = MergeNest(common, Block::make(init, update));
      update = ir::Substitute(update, vmap);
      update = op::Substitute(update, n.main_vmap);
      return update;
    } else {
      // When init op is not available, use body op for reset in the first iter.
      CHECK(this->intrin->body.defined())
          << "Normal body op is not defined";
      Stmt update = TransformUpdate(stage, dom_map, n,
                                    this->intrin->body,
                                    this->intrin->reduce_update);
      update = MergeNests(update, output_bind_nest, input_bind_nest,
                          binder.asserts(), update_nest, common);
      update = ir::Substitute(update, vmap);
      update = op::Substitute(update, n.main_vmap);
      return update;
    }
  }
}

}  // namespace tvm
