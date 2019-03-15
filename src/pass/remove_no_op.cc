/*!
 *  Copyright (c) 2017 by Contributors
 * \file remove_no_op.cc
 * \brief Remove no op from the stmt
 */
#include <tvm/ir.h>
#include <tvm/ir_pass.h>
#include <tvm/ir_mutator.h>
#include <unordered_map>

namespace tvm {
namespace ir {

// Mark the statment of each stage.
class NoOpRemover : public IRMutator {
 public:
  Stmt Mutate_(const LetStmt* op, const Stmt& s) final {
    Stmt stmt = IRMutator::Mutate_(op, s);
    op = stmt.as<LetStmt>();
    return is_no_op(op->body) ? MakeEvaluate(op->value) : stmt;
  }
  Stmt Mutate_(const AttrStmt* op, const Stmt& s) final {
    if (op->attr_key == "pragma_debug_skip_region") {
      return MakeEvaluate(0);
    }
    Stmt stmt = IRMutator::Mutate_(op, s);
    op = stmt.as<AttrStmt>();
    return is_no_op(op->body) ? MakeEvaluate(op->value) : stmt;
  }
  Stmt Mutate_(const IfThenElse* op, const Stmt& s) final {
    Stmt stmt = IRMutator::Mutate_(op, s);
    op = stmt.as<IfThenElse>();
    if (op->else_case.defined()) {
      if (is_no_op(op->else_case)) {
        if (is_no_op(op->then_case)) {
          return MakeEvaluate(op->condition);
        } else {
          return IfThenElse::make(op->condition, op->then_case);
        }
      } else {
        return stmt;
      }
    } else {
      if (is_no_op(op->then_case)) {
        return MakeEvaluate(op->condition);
      } else {
        return stmt;
      }
    }
  }
  Stmt Mutate_(const For* op, const Stmt& s) final {
    if (can_prove(op->extent <= 0)) return MakeEvaluate(0);
    Stmt stmt = IRMutator::Mutate_(op, s);
    op = stmt.as<For>();
    return is_no_op(op->body) ? MakeEvaluate({op->min, op->extent}) : stmt;
  }
  Stmt Mutate_(const Allocate* op, const Stmt& s) final {
    for (const auto &e : op->extents) {
      if (can_prove(e <= 0)) {
        return MakeEvaluate(0);
      }
    }
    Stmt stmt = IRMutator::Mutate_(op, s);
    op = stmt.as<Allocate>();
    return is_no_op(op->body) ? MakeEvaluate(op->extents) : stmt;
  }
  Stmt Mutate_(const ProducerConsumer* op, const Stmt& s) final {
    Stmt stmt = IRMutator::Mutate_(op, s);
    op = stmt.as<ProducerConsumer>();
    return is_no_op(op->body) ? op->body : stmt;
  }
  Stmt Mutate_(const Realize* op, const Stmt& s) final {
    Stmt stmt = IRMutator::Mutate_(op, s);
    op = stmt.as<Realize>();
    return is_no_op(op->body) ? op->body : stmt;
  }
  Stmt Mutate_(const Evaluate* op, const Stmt& s) final {
    if (HasSideEffect(op->value)) return s;
    return Evaluate::make(0);
  }
  Stmt Mutate_(const Block* op, const Stmt& s) final {
    Stmt stmt = IRMutator::Mutate_(op, s);
    op = stmt.as<Block>();
    if (is_no_op(op->first)) {
      return op->rest;
    } else if (is_no_op(op->rest)) {
      return op->first;
    } else {
      return stmt;
    }
  }

 private:
  Stmt MakeEvaluate(Expr value) {
    if (HasSideEffect(value)) {
      return Evaluate::make(value);
    } else {
      return Evaluate::make(0);
    }
  }
  Stmt MakeEvaluate(const Array<Expr>& values) {
    Stmt stmt;
    for (Expr e : values) {
      if (HasSideEffect(e)) {
        if (stmt.defined()) {
          stmt = Block::make(stmt, Evaluate::make(e));
        } else {
          stmt = Evaluate::make(e);
        }
      }
    }
    return stmt.defined() ? stmt : Evaluate::make(0);
  }
};

Stmt RemoveNoOp(Stmt stmt) {
  return NoOpRemover().Mutate(stmt);
}
}  // namespace ir
}  // namespace tvm
