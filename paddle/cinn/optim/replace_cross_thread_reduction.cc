// Copyright (c) 2021 CINN Authors. All Rights Reserved.
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

/**
 * This file implements the strategy to remove the unnecessary nested block.
 */
#pragma once
#include "paddle/cinn/optim/replace_cross_thread_reduction.h"
#include <vector>

#include "paddle/cinn/common/common.h"
#include "paddle/cinn/hlir/pe/reduction.h"
#include "paddle/cinn/ir/ir.h"
#include "paddle/cinn/ir/ir_mutator.h"
#include "paddle/cinn/ir/ir_printer.h"
#include "paddle/cinn/ir/schedule/ir_schedule_util.h"
#include "paddle/cinn/lang/compute.h"

namespace cinn {
namespace optim {

struct CrossThreadReductionReplacer : public ir::IRMutator<Expr*> {
  void operator()(ir::Expr* expr) { Visit(expr); }

 private:
  bool CanReplace(const ir::ScheduleBlockRealize* block_realize) {
    const ir::ScheduleBlock* schedule_block =
        block_realize->schedule_block.As<ir::ScheduleBlock>();
    CHECK_NOTNULL(schedule_block);

    if (block_realize->schedule_block.As<ir::ScheduleBlock>()->name.substr(
            0, 4) == "root") {
      return false;
    }

    const std::vector<ir::Expr>& iter_values = block_realize->iter_values;
    const std::vector<ir::Var>& iter_vars = schedule_block->iter_vars;
    ir::Expr body = schedule_block->body;

    std::unordered_set<std::string> reduce_var_names;
    for (int i = 0; i < iter_values.size(); ++i) {
      if (!iter_vars[i]->is_reduce_axis) {
        continue;
      }
      ir::ir_utils::CollectIRNodesWithoutTensor(
          iter_values[i], [&](const ir::Expr* x) {
            if (x->as_var()) {
              reduce_var_names.insert(x->as_var()->name);
            }
            return false;
          });
    }

    std::vector<int> thread_binded_reduce_loop_indices;
    for (int i = 0; i < cur_loops_.size(); ++i) {
      if (reduce_var_names.count(cur_loops_[i].As<ir::For>()->loop_var->name) >
          0) {
        if (cur_loops_[i].As<ir::For>()->is_gpu_thread_binded()) {
          if (ir::GetLoopExtent(cur_loops_[i]) > 1024) {
            return false;
          }
          thread_binded_reduce_loop_indices.push_back(i);
        }
      }
    }
    if (thread_binded_reduce_loop_indices.size() == 0 ||
        thread_binded_reduce_loop_indices.back() != cur_loops_.size() - 1) {
      return false;
    }
    for (int i = 1; i < thread_binded_reduce_loop_indices.size(); ++i) {
      if (thread_binded_reduce_loop_indices[i - 1] + 1 !=
          thread_binded_reduce_loop_indices[i]) {
        return false;
      }
    }

    return true;
  }

  void Visit(ir::Expr* expr) { ir::IRMutator<>::Visit(expr, expr); }

  void Visit(const ir::ScheduleBlockRealize* expr, ir::Expr* op) override {
    if (!CanReplace(expr)) {
      VLOG(6) << "Can't replace cross thread reduction: " << *op;
      IRMutator::Visit(expr, op);
      return;
    }
    VLOG(6) << "Can replace cross thread reduction: " << *op;

    const ir::ScheduleBlock* schedule_block =
        expr->schedule_block.As<ir::ScheduleBlock>();
    CHECK_NOTNULL(schedule_block);
    ir::Expr original_update_body = schedule_block->body;
    ir::Expr original_update_stmt;
    CHECK(original_update_body.As<ir::Block>() ||
          original_update_body.As<ir::Store>());
    if (original_update_body.As<ir::Block>()) {
      CHECK_EQ(original_update_body.As<ir::Block>()->stmts.size(), 1);
      original_update_stmt = original_update_body.As<ir::Block>()->stmts[0];
    } else if (original_update_body.As<ir::Store>()) {
      original_update_stmt = original_update_body;
    }

#define REPLACE_TO_EXTERNAL_CALL(Op)                                   \
  if (original_update_stmt.As<ir::Store>()->value.As<Op>()) {          \
    auto* node = original_update_stmt.As<ir::Store>()->value.As<Op>(); \
    CHECK(node);                                                       \
    auto& operand = node->b();                                         \
    std::string reduce_func_name =                                     \
        hlir::pe::CrossThreadReduceExternalFuncName(                   \
            original_update_stmt.As<ir::Store>()->value,               \
            operand.As<ir::Load>()->tensor);                           \
    original_update_stmt.As<ir::Store>()->value =                      \
        lang::CallExtern(reduce_func_name, {node->b()});               \
  }

    REPLACE_TO_EXTERNAL_CALL(ir::Add)
    REPLACE_TO_EXTERNAL_CALL(ir::Mul)
    REPLACE_TO_EXTERNAL_CALL(ir::Max)
    REPLACE_TO_EXTERNAL_CALL(ir::Min)
    REPLACE_TO_EXTERNAL_CALL(ir::And)
    REPLACE_TO_EXTERNAL_CALL(ir::Or)
#undef REPLACE_TO_EXTERNAL_CALL

    VLOG(6) << "Replace cross thread reduction: " << *op;

    IRMutator::Visit(expr, op);
  }

  void Visit(const ir::For* expr, ir::Expr* op) override {
    cur_loops_.push_back(*op);
    IRMutator::Visit(expr, op);
    cur_loops_.pop_back();
  }

 private:
  std::vector<ir::Expr> cur_loops_;
};

void ReplaceCrossThreadReduction(Expr* e) { CrossThreadReductionReplacer()(e); }

}  // namespace optim
}  // namespace cinn
