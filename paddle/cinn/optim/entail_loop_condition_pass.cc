// Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "paddle/cinn/optim/entail_loop_condition_pass.h"
#include "paddle/cinn/common/ir_util.h"
#include "paddle/cinn/ir/ir_mutator.h"
#include "paddle/cinn/ir/stmt_visitors.h"
#include "paddle/cinn/ir/utils/ir_copy.h"

namespace cinn {
namespace optim {

using ir::stmt::BlockRef;
using ir::stmt::Evaluate;
using ir::stmt::For;
using ir::stmt::IfThenElse;
using ir::stmt::Let;
using ir::stmt::Schedule;
using ir::stmt::StmtRef;
using ir::stmt::Store;

namespace {

struct LoopCondLowerboundMutator : public ir::IRMutator<> {
  explicit LoopCondLowerboundMutator(const ir::Var& var, const ir::Expr& extent)
      : var_(var), extent_(extent) {}

  void operator()(ir::Expr* expr) { ir::IRMutator<>::Visit(expr, expr); }

 private:
  void Visit(const ir::_Var_* op, ir::Expr* expr) override {
    if (!op->is_symbolic_constant) {
      if (expr->as_var_ref() == var_) {
        *expr = extent_;
      } else {
        *expr = common::make_const(op->type(), 0);
      }
    }
  }

  ir::Var var_;
  ir::Expr extent_;
};

bool CanProveEntailment(const ir::Var& loop_var,
                        const ir::Expr& extent,
                        const ir::LT* if_cond) {
  // 1. For static shape, it's easy to prove by setting `loop_var = extent` and
  //    check whether the if-condition is violated
  if (extent.is_constant()) {
    ir::Expr lhs = ir::ir_utils::IRCopy(if_cond->a());
    ir::Expr rhs = if_cond->b();
    LoopCondLowerboundMutator(loop_var, extent)(&lhs);
    if (!lhs.is_index()) return false;
    lhs = lhs.as_index().Normalize();
    return lhs.is_constant() && rhs.is_constant() &&
           lhs.as_int64() >= rhs.as_int64();
  }

  // 2. For dynamic shape, as CINN is currently unable to prove equality with
  //    symbols, we just check their forms rather than values. We require that
  //    the extent be like `(<expr_a> / <expr_b>) + 1`, and the rhs of if_cond
  //    be `<expr_a>`.
  auto* add_node = extent.As<ir::Add>();
  if (!add_node) return false;
  if (add_node->b() != ir::Expr(1)) return false;
  auto* div_node = add_node->a().As<ir::Div>();
  if (!div_node) return false;
  return div_node->a() == if_cond->b();
}

bool CanEntailLoopCondition(const For& for_stmt) {
  // 1. The loop extent must be >32 or be dynamic, otherwise the loop will be
  //    completely unrolled and we have nothing to do.
  ir::Expr extent = for_stmt->extent();
  if (extent.is_constant() && extent.as_int64() <= 32) return false;

  // 2. The loop body must contains exactly one IfThenElse.
  if (for_stmt->body()->stmts().size() != 1) return false;
  StmtRef for_body = for_stmt->body()->stmts().front();
  if (!for_body.isa<IfThenElse>()) return false;

  // 3. The IfThenElse must be a leaf node, i.e., it only contains Schedule
  //    nodes.
  IfThenElse if_stmt = for_body.as<IfThenElse>();
  if (!if_stmt->condition().As<ir::LT>()) return false;
  if (!if_stmt->false_case()->stmts().empty()) return false;
  for (auto& inner_stmt : if_stmt->true_case()->stmts()) {
    if (!inner_stmt.isa<Schedule>()) return false;
  }

  // 4. Check whether the if-condition actually entails the loop extent.
  return CanProveEntailment(
      for_stmt->loop_var(), extent, if_stmt->condition().As<ir::LT>());
}

struct CommonFactorExtractor : public ir::IRMutator<> {
  explicit CommonFactorExtractor(const ir::Var& var) : var_(var) {}

  void operator()(ir::Expr* expr) { ir::IRMutator<>::Visit(expr, expr); }

  int64_t GetResult() { return common_factor_ > 0 ? common_factor_ : 1; }

 private:
  void Visit(const ir::Mul* op, ir::Expr* expr) override {
    if (op->a() != var_ && op->b() != var_) {
      auto* node = expr->As<ir::Mul>();
      ir::IRMutator<>::Visit(&node->a(), &node->a());
      ir::IRMutator<>::Visit(&node->b(), &node->b());
      return;
    }

    int64_t factor = 1;
    if (op->a() == var_ && op->b().is_constant()) {
      factor = op->b().as_int64();
    } else if (op->b() == var_ && op->a().is_constant()) {
      factor = op->a().as_int64();
    }

    if (common_factor_ == 0) {
      common_factor_ = factor;
    } else {
      common_factor_ = std::gcd(common_factor_, factor);
    }
  }

  void Visit(const ir::_Var_* op, ir::Expr* expr) override {
    // Note: if we visit a standalone `var` that is not an operand of a Mul op,
    // the common factor must be 1.
    if (*expr == var_) {
      common_factor_ = 1;
    }
  }

  ir::Expr var_;
  int64_t common_factor_{0};
};

int64_t ExtractCommonFactorOfLoopVar(const For& for_stmt) {
  CommonFactorExtractor extractor(for_stmt->loop_var());

  const auto VisitFn = [&](const StmtRef& stmt) {
    if (stmt.isa<IfThenElse>()) {
      IfThenElse if_stmt = stmt.as<IfThenElse>();
      ir::Expr condition = if_stmt->condition();
      extractor(&condition);
    } else if (stmt.isa<Store>()) {
      Store store_stmt = stmt.as<Store>();
      for (ir::Expr index : store_stmt->indices()) {
        extractor(&index);
      }
      ir::Expr value = store_stmt->value();
      extractor(&value);
    }
  };

  ir::stmt::Visit(for_stmt->body(), VisitFn, [](auto) {});
  return extractor.GetResult();
}

struct StridedLoopVarReplacer : public ir::IRMutator<> {
  explicit StridedLoopVarReplacer(const ir::Var& var,
                                  const ir::Var& new_var,
                                  int64_t common_factor)
      : var_(var), new_var_(new_var), common_factor_(common_factor) {}

  void operator()(ir::Expr* expr) { ir::IRMutator<>::Visit(expr, expr); }

 private:
  void Visit(const ir::Mul* op, ir::Expr* expr) override {
    auto* node = expr->As<ir::Mul>();
    ir::IRMutator<>::Visit(&node->a(), &node->a());
    ir::IRMutator<>::Visit(&node->b(), &node->b());

    if (op->a() == var_) {
      Replace(expr, &node->b());
    } else if (op->b() == var_) {
      Replace(expr, &node->a());
    }
  }

  void Replace(ir::Expr* expr, ir::Expr* scale) {
    if (*scale == ir::Expr(common_factor_)) {
      *expr = new_var_;
    } else {
      scale->As<ir::IntImm>()->value /= common_factor_;
    }
  }

  ir::Expr var_;
  ir::Expr new_var_;
  int64_t common_factor_;
};

void ReplaceWithStridedLoopVar(For for_stmt,
                               const ir::Var& strided_loop_var,
                               int64_t common_factor) {
  ir::Var loop_var = for_stmt->loop_var();
  StridedLoopVarReplacer replacer(loop_var, strided_loop_var, common_factor);

  const auto VisitFn = [&](StmtRef stmt) {
    if (stmt.isa<IfThenElse>()) {
      IfThenElse if_stmt = stmt.as<IfThenElse>();
      ir::Expr condition = if_stmt->condition();
      replacer(&condition);
    } else if (stmt.isa<Store>()) {
      Store store_stmt = stmt.as<Store>();
      for (ir::Expr index : store_stmt->indices()) {
        replacer(&index);
      }
      ir::Expr value = store_stmt->value();
      replacer(&value);
    }
  };

  ir::stmt::Mutate(for_stmt->body(), VisitFn, [](auto) {});
}

void EntailLoopCondition(For for_stmt) {
  ir::Var loop_var = for_stmt->loop_var();
  IfThenElse if_stmt = for_stmt->body()->stmts().front().as<IfThenElse>();
  std::vector<StmtRef> new_body_stmts;

  // Step 1. Try extract common factor of loop_var.
  // Note: type of common_factor should be strictly the same as loop_var. Don't
  //   mix int64 value in an int32 expression.
  int64_t common_factor = ExtractCommonFactorOfLoopVar(for_stmt);
  ir::Expr common_factor_expr =
      common::make_const(loop_var->type(), common_factor);

  // Step 2. If loop_var has a non-unit common factor, replace all occurrences
  //   of loop_var with strided_loop_var = loop_var * common_factor.
  if (common_factor > 1) {
    ir::Var strided_loop_var = loop_var->Copy();
    strided_loop_var->name += "_strided";
    new_body_stmts.push_back(
        Let(strided_loop_var, ir::Mul::Make(loop_var, common_factor_expr)));

    ReplaceWithStridedLoopVar(for_stmt, strided_loop_var, common_factor);
    loop_var = strided_loop_var;
  }

  // Step 3. Declare the entailment relation on the loop condition.
  ir::Expr entail_expr =
      ir::Call::Make(Void(),
                     "CINN_ENTAIL_LOOP_CONDITION",
                     {loop_var, if_stmt->condition(), common_factor_expr},
                     {},
                     ir::CallType::Intrinsic);
  new_body_stmts.push_back(Evaluate(entail_expr));

  new_body_stmts.push_back(if_stmt);
  for_stmt->set_body(BlockRef(new_body_stmts));
}

}  // namespace

LogicalResult EntailLoopConditionPass::Run(ir::stmt::BlockRef block) {
  std::vector<StmtRef> new_stmts;
  for (auto stmt : block->stmts()) {
    if (stmt.isa<For>()) {
      For for_stmt = stmt.as<For>();
      if (CanEntailLoopCondition(for_stmt)) {
        EntailLoopCondition(for_stmt);
      }
    }
    new_stmts.push_back(stmt);
  }
  block->set_stmts(new_stmts);
  return LogicalResult::success();
}

std::unique_ptr<BlockPass> CreateEntailLoopConditionPass() {
  return std::make_unique<EntailLoopConditionPass>();
}

}  // namespace optim
}  // namespace cinn
