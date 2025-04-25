// Copyright (c) 2024 CINN Authors. All Rights Reserved.
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

#include "paddle/cinn/optim/longlong2int_pass.h"
#include "paddle/cinn/common/ir_util.h"
#include "paddle/cinn/ir/ir_mutator.h"
#include "paddle/cinn/ir/ir_printer.h"
#include "paddle/cinn/ir/ir_utils.h"
#include "paddle/cinn/ir/ir_visitor.h"
#include "paddle/cinn/ir/stmt.h"
#include "paddle/cinn/ir/stmt_visitors.h"
#include "paddle/cinn/ir/utils/ir_copy.h"
#include "paddle/cinn/optim/simplify_util.h"
#include "paddle/cinn/pass/pass_manager.h"

namespace cinn {
namespace optim {
namespace {
using ir::stmt::BlockRef;
using ir::stmt::For;
using ir::stmt::IfThenElse;
using ir::stmt::Schedule;
using ir::stmt::StmtRef;
using ir::stmt::Store;

void CastVarWithBound(cinn::ir::Var& var) {  // NOLINT
  if (!var.defined()) return;
  if (var->is_symbolic_constant) return;
  var->convert_int64_to_int32();
  if (var->lower_bound.defined()) ir::ElevateInt64ToInt32_(var->lower_bound);
  if (var->lower_bound.defined()) ir::ElevateInt64ToInt32_(var->lower_bound);
}
void CastBufferMeta(cinn::ir::Buffer& bf) {  // NOLINT
  if (!bf.defined()) return;
  ir::ElevateInt64ToInt32_(bf->shape);
  ir::ElevateInt64ToInt32_(bf->strides);
  ir::ElevateInt64ToInt32_(bf->elem_offset);
}

class CheckOverflow : public ir::stmt::StmtVisitor<> {
 public:
  bool operator()(const StmtRef& stmt) {
    VisitStmt(stmt);
    return is_overflow_;
  }
  bool operator()(const BlockRef& block) {
    VisitBlock(block);
    return is_overflow_;
  }

 private:
  void VisitStmt(const StmtRef& stmt) override {
    if (is_overflow_) return;
    ir::stmt::StmtVisitor<>::VisitStmt(stmt);
  }

  void VisitStmt(const For& for_stmt) override {
    if (!for_stmt->extent().is_constant()) is_overflow_ = true;
    if (!for_stmt->extent().type().is_index_type()) is_overflow_ = true;
    if (curr_product_ > INT_MAX) is_overflow_ = true;

    if (is_overflow_) return;

    int64_t prev_product = curr_product_;
    curr_product_ *= for_stmt->extent().as_int64();
    VisitBlock(for_stmt->body());
    curr_product_ = prev_product;
  }

  void VisitStmt(const Schedule& schedule_stmt) override {
    VisitBlock(schedule_stmt->body());
  }

  void VisitStmt(const IfThenElse& stmt) override {
    VisitBlock(stmt->true_case());
    if (stmt->false_case().defined()) {
      VisitBlock(stmt->false_case());
    }
  }

  void VisitStmt(const ir::stmt::Let& stmt) override { return; }
  void VisitStmt(const ir::stmt::Store& stmt) override { return; }
  void VisitStmt(const ir::stmt::Alloc& stmt) override { return; }
  void VisitStmt(const ir::stmt::Free& stmt) override { return; }
  void VisitStmt(const ir::stmt::Evaluate& stmt) override { return; }

 private:
  int64_t curr_product_ = 1;
  bool is_overflow_ = false;
};

class CastLonglong2IntMutator : public ir::IRMutator<> {
 public:
  void operator()(Expr* expr) { ir::IRMutator<>::Visit(expr, expr); }

 private:
  void Visit(const ir::_Tensor_* op, Expr* expr) override {
    auto node = expr->As<ir::_Tensor_>();
    ir::ElevateInt64ToInt32_(node->shape);
    CastBufferMeta(node->buffer);
  }
  void Visit(const ir::Load* op, Expr* expr) override {
    auto node = expr->As<ir::Load>();
    ir::ElevateInt64ToInt32_(node->indices);
    ir::IRMutator<>::Visit(&node->tensor, &node->tensor);
  }
  void Visit(const ir::Select* op, Expr* expr) override {
    auto node = expr->As<ir::Select>();
    auto cond = node->condition;
    // select(bool(v[]), T, F)
    if (auto cond_cast_bool = cond.As<ir::Cast>()) {
      if (cond_cast_bool->type().is_bool()) {
        cond = cond_cast_bool->v();
      }
    }

    if (cond.is_index()) {  // select(v[], T, F)
      ir::ElevateInt64ToInt32_(node->condition);
    } else if (cond.is_cmp() && cond->operand(0).is_index() &&
               cond->operand(1).is_index()) {  // select(i < S0, T, F)
      ir::ElevateInt64ToInt32_(node->condition->operands);
    } else {  // select(v[] or v1[], T, F)
      ir::IRMutator<>::Visit(&node->condition, &node->condition);
    }
    ir::IRMutator<>::Visit(&node->true_value, &node->true_value);
    ir::IRMutator<>::Visit(&node->false_value, &node->false_value);
  }
  void Visit(const ir::Min* op, Expr* expr) override {
    auto node = expr->As<ir::Min>();
    // min(min(S0, 1ll), 1ll) ==> min(min(S0, 1), 1)
    // min(V[S0, S1], 1ll)    ==> min(V[S0, S1], 1ll)
    // min(S0 + 1ll, 1ll)     ==> max(S0 + 1, 1)
    // IndexType::kValid means expr only has +-*/%, Const, Symbol, Min, Max.
    // IsDynamic == true means expr has Symbol.
    if (optim::VerifyIndex(*expr) == ir::IndexExpr::IndexType::kValid &&
        expr->as_index().IsDynamic()) {
      ir::ElevateInt64ToInt32_((*expr)->operands);
    } else {
      ir::IRMutator<>::Visit(&node->a(), &node->a());
      ir::IRMutator<>::Visit(&node->b(), &node->b());
    }
  }
  void Visit(const ir::Max* op, Expr* expr) override {
    auto node = expr->As<ir::Max>();
    if (optim::VerifyIndex(*expr) == ir::IndexExpr::IndexType::kValid &&
        expr->as_index().IsDynamic()) {
      ir::ElevateInt64ToInt32_((*expr)->operands);
    } else {
      ir::IRMutator<>::Visit(&node->a(), &node->a());
      ir::IRMutator<>::Visit(&node->b(), &node->b());
    }
  }
};

class LongLong2IntStmtPass : public StmtPass {
 public:
  LongLong2IntStmtPass() : StmtPass("longlong2int_stmt") {}
  LogicalResult Run(ir::stmt::StmtRef stmt) override;
};

class LongLong2IntExprPass : public ExprPass {
 public:
  LongLong2IntExprPass() : ExprPass("longlong2int_expr") {}
  LogicalResult Run(ir::Expr* expr) override;
};
}  // namespace

LogicalResult LongLong2IntStmtPass::Run(ir::stmt::StmtRef stmt) {
  auto CastStore = [&](StmtRef stmt) {
    Store store_stmt = stmt.as<Store>();
    store_stmt->set_indices(
        std::move(ir::ElevateInt64ToInt32(store_stmt->indices())));
  };

  auto CastIfThenElse = [&](StmtRef stmt) {
    IfThenElse if_stmt = stmt.as<IfThenElse>();
    Expr cond = if_stmt->condition();
    // if(bool(v[]))
    if (auto cond_cast_bool = cond.As<ir::Cast>()) {
      if (cond_cast_bool->type().is_bool()) {
        cond = cond_cast_bool->v();
      }
    }

    if (cond.is_index()) {  // if(v[])
      if_stmt->set_condition(std::move(ir::ElevateInt64ToInt32(cond)));
    } else if (cond.is_cmp() && cond->operand(0).is_index() &&
               cond->operand(1).is_index()) {  // if(i < S0)
      ir::ElevateInt64ToInt32_(if_stmt->condition()->operands);
    } else {  // if(v[] or v1[])
      CastLonglong2IntMutator mutator;
      mutator(&cond);
    }
  };

  auto CastFor = [](StmtRef stmt) {
    For for_stmt = stmt.as<For>();
    ir::Var loop_var = for_stmt->loop_var();
    CastVarWithBound(loop_var);
    for_stmt->set_loop_var(std::move(loop_var));
    for_stmt->set_min(std::move(ir::ElevateInt64ToInt32(for_stmt->min())));
    for_stmt->set_extent(
        std::move(ir::ElevateInt64ToInt32(for_stmt->extent())));
  };

  auto CastSchedule = [](StmtRef stmt) {
    Schedule schedule_stmt = stmt.as<Schedule>();
    std::vector<Var> iter_vars = schedule_stmt->iter_vars();
    std::for_each(iter_vars.begin(), iter_vars.end(), [&](cinn::ir::Var& v) {
      CastVarWithBound(v);
    });

    std::vector<Expr> iter_values = schedule_stmt->iter_values();
    ir::ElevateInt64ToInt32_(iter_values);

    for (auto& buffer_range : schedule_stmt->read_buffers()) {
      if (auto range = buffer_range.As<ir::_BufferRange_>()) {
        std::vector<Var> ranges = range->ranges;
        std::for_each(ranges.begin(), ranges.end(), [&](cinn::ir::Var& v) {
          CastVarWithBound(v);
        });
        auto bf = range->buffer.as_buffer_ref();
        CastBufferMeta(bf);
      }
    }

    for (auto& buffer_range : schedule_stmt->write_buffers()) {
      if (auto range = buffer_range.As<ir::_BufferRange_>()) {
        std::vector<Var> ranges = range->ranges;

        std::for_each(ranges.begin(), ranges.end(), [&](cinn::ir::Var& v) {
          CastVarWithBound(v);
        });
        auto bf = range->buffer.as_buffer_ref();
        CastBufferMeta(bf);
      }
    }
  };

  switch (stmt->stmt_type()) {
    case ir::StmtNodeTy::Store:
      CastStore(stmt);
      break;

    case ir::StmtNodeTy::IfThenElse:
      CastIfThenElse(stmt);
      break;

    case ir::StmtNodeTy::For:
      CastFor(stmt);
      break;

    case ir::StmtNodeTy::Schedule:
      CastSchedule(stmt);
      break;
    default:
      break;
  }
  return LogicalResult::success();
}

LogicalResult LongLong2IntExprPass::Run(ir::Expr* expr) {
  CastLonglong2IntMutator narrow;
  narrow(expr);
  return LogicalResult::success();
}
std::unique_ptr<StmtPass> CreateLongLong2IntStmtPass() {
  return std::make_unique<LongLong2IntStmtPass>();
}

std::unique_ptr<ExprPass> CreateLongLong2IntExprPass() {
  return std::make_unique<LongLong2IntExprPass>();
}

// Check if the given block can be converted from long long to int,
// A.K.A. the product of the extents of all possible nested loops is within
// INT_MAX
bool CanApplyLongLong2Int(ir::stmt::BlockRef block) {
  CheckOverflow check_overflow;
  return !check_overflow(block);
}

bool TryCastLonglong2Int(ir::stmt::BlockRef block,
                         std::optional<bool> enforce_cast) {
  bool can_cast = enforce_cast.has_value() ? enforce_cast.value()
                                           : CanApplyLongLong2Int(block);
  if (can_cast) {
    StmtPassManager stmt_pass_manager;
    stmt_pass_manager.AddPass(CreateLongLong2IntStmtPass());
    ExprPassManager expr_pass_manager;
    expr_pass_manager.AddPass(CreateLongLong2IntExprPass());

    stmt_pass_manager.Run(block);
    expr_pass_manager.Run(block);
  }
  return can_cast;
}

bool TryCastLonglong2Int(ir::LoweredFunc& func,  // NOLINT
                         const std::unordered_set<std::string>& symbol_args_set,
                         std::optional<bool> enforce_cast) {
  // Set lowered_func's symbol args to int32 type, although the inputs and
  // outputs are static, symbols may still exist. we can change those type
  // safely. e.g. out = inp[S0, S0 + 2], D(out) = 2, D(inp) = 8
  auto deal_func_args =
      [](const std::unordered_set<std::string>& symbol_args_set,
         std::vector<cinn::ir::Argument>& args) {
        for (auto& arg : args) {
          if (arg.is_var() && symbol_args_set.count(arg.name()) != 0) {
            arg.set_var(ir::ir_utils::IRCopy(arg.var_arg()));
            arg.var_arg()->set_type(cinn::common::Int(32));
          }
        }
      };
  auto deal_func_axis_info = [](ir::CudaAxisInfo& axis_info) {
    std::vector<ir::Expr> block_dim = {
        ir::ir_utils::IRCopy(axis_info.block_dim(0)),
        ir::ir_utils::IRCopy(axis_info.block_dim(1)),
        ir::ir_utils::IRCopy(axis_info.block_dim(2))};
    std::vector<ir::Expr> grid_dim = {
        ir::ir_utils::IRCopy(axis_info.grid_dim(0)),
        ir::ir_utils::IRCopy(axis_info.grid_dim(1)),
        ir::ir_utils::IRCopy(axis_info.grid_dim(2))};

    ir::ElevateInt64ToInt32_(block_dim);
    ir::ElevateInt64ToInt32_(grid_dim);

    axis_info.set_block_dim(0, block_dim[0]);
    axis_info.set_block_dim(1, block_dim[1]);
    axis_info.set_block_dim(2, block_dim[2]);

    axis_info.set_grid_dim(0, grid_dim[0]);
    axis_info.set_grid_dim(1, grid_dim[1]);
    axis_info.set_grid_dim(2, grid_dim[2]);
  };

  ir::stmt::BlockRef block = ir::ConvertExprBlockToStmtBlock(func->body);
  bool cast = TryCastLonglong2Int(block, enforce_cast);
  if (cast) {
    deal_func_args(symbol_args_set, func->args);
    deal_func_axis_info(func->cuda_axis_info);
  }
  func->body = ir::ConvertStmtBlockToExprBlock(block);

  return cast;
}
}  // namespace optim
}  // namespace cinn
