// Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

#pragma once
#include "paddle/cinn/optim/if_fold_pass.h"
#include <vector>
#include "paddle/cinn/common/ir_util.h"
#include "paddle/cinn/common/simplify_special_pattern.h"
#include "paddle/cinn/ir/ir_printer.h"
#include "paddle/cinn/ir/op/ir_operators.h"
#include "paddle/cinn/ir/utils/ir_compare.h"
#include "paddle/cinn/ir/utils/ir_copy.h"

namespace cinn {
namespace optim {
using ir::stmt::IfThenElse;
using ir::stmt::StmtRef;

// Determine whether `IfThenElse` satisfies the following conditions:
// 1. The condition is an equality comparison
// 2. The right side of the equality comparison is `0`
// 3. There are no statements in the false branch
// 4. Only one statement in the true branch
bool JudgeIfStmt(const StmtRef& stmt) {
  if (!stmt.isa<IfThenElse>()) return false;
  auto if_stmt = stmt.as<IfThenElse>();
  auto cond = if_stmt->condition().As<ir::EQ>();
  if (if_stmt->false_case()->stmts().size() != 0) return false;
  if (if_stmt->true_case()->stmts().size() != 1) return false;
  if (!cond) return false;
  if (!cond->b().is_constant()) return false;
  if (cond->b().get_constant() != 0) return false;
  return true;
}

// Only judge condition of `IfThenElse` like `xxx == 0`
bool IsIfWithEqCond(const StmtRef& stmt) {
  if (stmt.isa<IfThenElse>()) {
    auto if_stmt = stmt.as<IfThenElse>();
    if (auto eq = if_stmt->condition().As<ir::EQ>()) {
      if (eq->b().is_constant() && eq->b().get_constant() == 0) {
        return true;
      }
    }
  }
  return false;
}

void AppendContinuousIfCond(const StmtRef& stmt,
                            std::vector<ir::IndexExpr>* cond_vec,
                            StmtRef* inner_op) {
  if (!JudgeIfStmt(stmt)) {
    // inner op is a `IfThenElse`, so we need to check its condition.
    if (IsIfWithEqCond(stmt)) {
      auto eq_lhs = stmt.as<IfThenElse>()->condition().As<ir::EQ>()->a();
      if (eq_lhs.is_index()) {
        cond_vec->push_back(common::ChangeSeqOfDivMod(
            ir::ir_utils::IRCopy(eq_lhs).as_index().Normalize()));
      }
    }
    // inner op is other op.
    *inner_op = stmt;
    return;
  }

  // Continuous `IfThenElse`, so we push its condition and recursively.
  auto if_stmt = stmt.as<IfThenElse>();
  auto eq_lhs = if_stmt->condition().As<ir::EQ>()->a();
  if (eq_lhs.is_index())
    cond_vec->push_back(common::ChangeSeqOfDivMod(
        ir::ir_utils::IRCopy(eq_lhs).as_index().Normalize()));
  AppendContinuousIfCond(
      if_stmt->true_case()->stmts().at(0), cond_vec, inner_op);
}

LogicalResult IfFoldPass::Run(StmtRef stmt) {
  if (!JudgeIfStmt(stmt)) return LogicalResult::success();

  std::vector<ir::IndexExpr> cond_vec;
  StmtRef inner_op;

  AppendContinuousIfCond(stmt, &cond_vec, &inner_op);

  ir::IndexExpr expr(0);
  int32_t min_len = INT32_MAX;

  VLOG(6) << "-------------cond_vec start--------------";
  for (auto v : cond_vec) {
    VLOG(6) << "v: " << v;
    // record min length of all conditions, because we want the simplified
    // result to be shorter
    min_len = std::min(v.length(), min_len);
    // For all normalized conditions, they have the form `a % b /c`，modulo
    // first and then divided. We transform it as follows:
    // origin:
    // ((256*j)+((1024*i)+k))/3136==0
    // ((256*j)+((1024*i)+k))%3136/56==0
    // ((256*j)+((1024*i)+k))%56==0
    // Mul and Sum:
    // ((256*j)+((1024*i)+k))/3136*3136
    // +((256*j)+((1024*i)+k))%3136/56*56
    // +((256*j)+((1024*i)+k))%56==0
    if (v.node_type() == ir::IrNodeTy::Div) {
      expr = expr + v * v.operand(1);
    } else {
      expr = expr + v;
    }
  }
  VLOG(6) << "-------------cond_vec end----------------";

  // Normalize expr to simplify the expr after Mul and Sum.
  expr = expr.Normalize(ir::IndexExpr::OptLevel::Level2);

  if (expr != ir::IndexExpr(0) && expr.length() < min_len &&
      inner_op.defined()) {
    VLOG(6) << "old stmt: " << stmt;
    auto stmt_if = stmt.as<IfThenElse>();
    stmt_if->set_condition(ir::EQ::Make(expr, ir::IndexExpr(0)));
    if (IsIfWithEqCond(inner_op)) {
      stmt_if->set_true_case(inner_op.as<IfThenElse>()->true_case());
      stmt_if->set_false_case(inner_op.as<IfThenElse>()->false_case());
    } else {
      stmt_if->set_true_case(
          ir::stmt::BlockRef(std::vector<StmtRef>{inner_op}));
    }
    VLOG(6) << "new stmt: " << stmt;
  }

  return LogicalResult::success();
}

std::unique_ptr<StmtPass> CreateIfFoldPass() {
  return std::make_unique<IfFoldPass>();
}
}  // namespace optim
}  // namespace cinn
