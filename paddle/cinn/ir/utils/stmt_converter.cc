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

#include "paddle/cinn/ir/utils/stmt_converter.h"
#include "paddle/cinn/ir/ir_printer.h"

#include "paddle/common/enforce.h"

namespace cinn {
namespace ir {

namespace {
stmt::StmtRef Convert(const Expr &expr);

stmt::Let Convert(const Let *expr) {
  return stmt::Let(expr->symbol, expr->body);
}

stmt::Store Convert(const Store *expr) {
  return stmt::Store(expr->tensor, expr->value, expr->indices);
}

stmt::Alloc Convert(const Alloc *expr) {
  return stmt::Alloc(expr->destination,
                     expr->type(),
                     expr->extents,
                     expr->condition,
                     expr->body);
}

stmt::Free Convert(const Free *expr) { return stmt::Free(expr->destination); }

stmt::IfThenElse Convert(const IfThenElse *expr) {
  stmt::BlockRef true_case = ConvertExprBlockToStmtBlock(expr->true_case);
  if (expr->false_case.defined()) {
    stmt::BlockRef false_case = ConvertExprBlockToStmtBlock(expr->false_case);
    return stmt::IfThenElse(expr->condition, true_case, false_case);
  }
  return stmt::IfThenElse(expr->condition, true_case);
}

stmt::For Convert(const For *expr) {
  stmt::BlockRef body = ConvertExprBlockToStmtBlock(expr->body);
  return stmt::For(expr->loop_var,
                   expr->min,
                   expr->extent,
                   expr->for_type(),
                   expr->device_api,
                   body,
                   expr->vectorize_info(),
                   expr->bind_info());
}

stmt::Schedule Convert(const ScheduleBlockRealize *expr) {
  const auto *schedule_block = expr->schedule_block.As<ScheduleBlock>();
  const Expr &expr_body = schedule_block->body;
  stmt::BlockRef new_body;
  if (const Block *block_of_stmt = expr_body.As<Block>()) {
    new_body = ConvertExprBlockToStmtBlock(expr_body);
  } else {
    new_body = stmt::BlockRef(std::vector<stmt::StmtRef>{Convert(expr_body)});
  }
  return stmt::Schedule(schedule_block->iter_vars,
                        expr->iter_values,
                        schedule_block->read_buffers,
                        schedule_block->write_buffers,
                        schedule_block->name,
                        new_body,
                        schedule_block->attrs,
                        schedule_block->reduce_method);
}

stmt::StmtRef Convert(const Expr &expr) {
  if (auto *let = expr.As<Let>()) {
    return Convert(let);
  }
  if (auto *store = expr.As<Store>()) {
    return Convert(store);
  }
  if (auto *alloc = expr.As<Alloc>()) {
    return Convert(alloc);
  }
  if (auto *free = expr.As<Free>()) {
    return Convert(free);
  }
  if (auto *if_then_else = expr.As<IfThenElse>()) {
    return Convert(if_then_else);
  }
  if (auto *for_loop = expr.As<For>()) {
    return Convert(for_loop);
  }
  if (auto *schedule_block_realize = expr.As<ScheduleBlockRealize>()) {
    return Convert(schedule_block_realize);
  }
  if (auto *call = expr.As<Call>()) {
    return stmt::Evaluate(expr);
  }
  PADDLE_THROW(::common::errors::Fatal(
      "Dead code. Expr type %d is not supported to convert to StmtRef.",
      expr.type()));
  return stmt::StmtRef();
}

}  // namespace

stmt::BlockRef ConvertExprBlockToStmtBlock(const Expr &expr_block) {
  std::vector<stmt::StmtRef> stmts;
  if (auto *block = expr_block.As<Block>()) {
    for (const Expr &stmt : block->stmts) {
      if (stmt.As<Block>()) {
        const auto &sub_stmts = ConvertExprBlockToStmtBlock(stmt);
        stmts.insert(
            stmts.end(), sub_stmts->stmts().begin(), sub_stmts->stmts().end());
      } else {
        stmts.emplace_back(Convert(stmt));
      }
    }
    return stmt::BlockRef(stmts);
  }
  PADDLE_THROW(
      ::common::errors::Fatal("Dead code. expr_block must be a Block when "
                              "converted to statement stmt::BlockRef."));
  return stmt::BlockRef(stmts);
}

namespace {
class StmtConverter : public stmt::StmtVisitor<Expr, Expr> {
 public:
  Expr VisitStmt(const stmt::StmtRef &stmt) override;
  Expr VisitBlock(const stmt::BlockRef &block) override {
    std::vector<Expr> expr_stmts;
    StmtConverter converter;
    for (const auto &stmt : block->stmts()) {
      expr_stmts.emplace_back(VisitStmt(stmt));
    }
    return Block::Make(expr_stmts);
  }

 private:
#define __(stmt__) Expr VisitStmt(const stmt::stmt__ &stmt) override;
  NODETY_FORALL_STMT(__)
#undef __
};

Expr StmtConverter::VisitStmt(const stmt::StmtRef &stmt) {
  return StmtVisitor<Expr, Expr>::VisitStmt(stmt);
}

Expr StmtConverter::VisitStmt(const stmt::Let &stmt) {
  return Let::Make(stmt->symbol(), stmt->body());
}

Expr StmtConverter::VisitStmt(const stmt::Store &stmt) {
  return Store::Make(stmt->tensor(), stmt->value(), stmt->indices());
}

Expr StmtConverter::VisitStmt(const stmt::Alloc &stmt) {
  return Alloc::Make(stmt->destination(),
                     stmt->type(),
                     stmt->extents(),
                     stmt->condition(),
                     stmt->body());
}

Expr StmtConverter::VisitStmt(const stmt::Free &stmt) {
  return Free::Make(stmt->destination());
}

Expr StmtConverter::VisitStmt(const stmt::IfThenElse &stmt) {
  Expr true_case = VisitBlock(stmt->true_case());
  if (!stmt->false_case()->stmts().empty()) {
    Expr false_case = VisitBlock(stmt->false_case());
    return IfThenElse::Make(stmt->condition(), true_case, false_case);
  }
  return IfThenElse::Make(stmt->condition(), true_case);
}

Expr StmtConverter::VisitStmt(const stmt::For &stmt) {
  Expr body = VisitBlock(stmt->body());
  return For::Make(stmt->loop_var(),
                   stmt->min(),
                   stmt->extent(),
                   stmt->for_type(),
                   stmt->device_api(),
                   body,
                   stmt->vectorize_info(),
                   stmt->bind_info());
}

Expr StmtConverter::VisitStmt(const stmt::Schedule &stmt) {
  Expr body = VisitBlock(stmt->body());
  Expr schedule_block = ir::ScheduleBlock::Make(stmt->iter_vars(),
                                                stmt->read_buffers(),
                                                stmt->write_buffers(),
                                                stmt->name(),
                                                body);
  schedule_block.As<ScheduleBlock>()->attrs = stmt->attrs();
  schedule_block.As<ScheduleBlock>()->reduce_method = stmt->reduce_method();
  return ScheduleBlockRealize::Make(stmt->iter_values(), schedule_block);
}

Expr StmtConverter::VisitStmt(const stmt::Evaluate &stmt) {
  return stmt->value();
}
};  // namespace

Expr ConvertStmtBlockToExprBlock(const stmt::BlockRef &stmt_block) {
  StmtConverter converter;
  return converter.VisitBlock(stmt_block);
}

}  // namespace ir
}  // namespace cinn
