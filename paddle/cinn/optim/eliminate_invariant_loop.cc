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

#include "paddle/cinn/optim/eliminate_invariant_loop.h"

#include "paddle/cinn/common/ir_util.h"
#include "paddle/cinn/ir/ir_analyzer/ir_analyzer.h"
#include "paddle/cinn/ir/ir_mutator.h"
#include "paddle/cinn/ir/utils/ir_replace.h"
#include "paddle/cinn/optim/replace_var_with_expr.h"

namespace cinn {
namespace optim {

namespace {

// Check whether var is used in the block's store indices or value.
bool HasVarInIndicesOrValue(const ir::Expr& block, const ir::Var& var) {
  auto* block_realize = block.As<ir::ScheduleBlockRealize>();
  auto* schedule_block = block_realize->schedule_block.As<ir::ScheduleBlock>();

  // collect iter vars that refer to `var`
  std::set<std::string> ref_iter_vars;
  for (int i = 0; i < block_realize->iter_values.size(); ++i) {
    auto var_use = ir::ir_utils::CollectIRNodes(
        block_realize->iter_values[i],
        [&](const ir::Expr* x) {
          return x->is_var() && x->as_var_ref() == var;
        },
        /* uniq_target = */ true);
    if (var_use.size() > 0) {
      ref_iter_vars.insert(schedule_block->iter_vars[i]->name);
    }
  }

  ir::Expr store = ir::analyzer::GetStoreOfSBlock(block);
  auto var_use = ir::ir_utils::CollectIRNodes(
      store,
      [&](const ir::Expr* x) {
        if (x->is_var()) {
          if (x->as_var_ref() == var) return true;
          if (ref_iter_vars.count(x->as_var()->name) > 0) return true;
        }
        return false;
      },
      /* uniq_target = */ true);
  return var_use.size() > 0;
}

// Check whether the block is writing to a buffer whose scope is smaller than
// the For node's scope.
bool HasWriteToSmallerScope(const ir::Expr& block, const ir::For* for_node) {
  ir::Expr store_tensor = ir::analyzer::GetStoreTensorOfSBlock(block);
  ir::MemoryType memory_type = store_tensor.as_tensor()->buffer->memory_type;
  if (for_node->is_gpu_thread_binded()) {
    if (memory_type == ir::MemoryType::GPULocal) {
      return true;
    }
  } else if (for_node->is_gpu_block_binded()) {
    if (memory_type == ir::MemoryType::GPULocal ||
        memory_type == ir::MemoryType::GPUShared) {
      return true;
    }
  }
  return false;
}

struct InvariantLoopEliminator : public ir::IRMutator<> {
  using ir::IRMutator<>::Visit;
  void operator()(ir::Expr* expr) { ir::IRMutator<>::Visit(expr, expr); }

  void Visit(const ir::For* op, ir::Expr* expr) override {
    auto* node = expr->As<ir::For>();
    ir::IRMutator<>::Visit(&node->body, &node->body);

    // do nothing if we are outside the root schedule block
    if (!root_) return;

    // check that all child schedule blocks satisfy the four rules
    std::vector<ir::Expr> child_blocks = ir::analyzer::GetChildBlocks(*expr);
    ir::Var loop_var = node->loop_var;
    for (auto& block : child_blocks) {
      if (HasVarInIndicesOrValue(block, loop_var)) return;
      if (ir::analyzer::IsReductionSBlock(block)) return;
      if (node->is_binded()) {
        if (HasWriteToSmallerScope(block, node)) return;
        if (!ir::analyzer::GetConsumerSBlocks(block, *root_).empty()) return;
      }
    }

    // now we can eliminate this For node
    ir::Expr body = node->body;
    ir::ir_utils::IrReplaceVarBroadcast(&body, ir::Expr(loop_var), ir::Expr(0));

    if (node->is_binded()) {
      ir::Expr cond = ir::EQ::Make(loop_var, ir::Expr(0));
      ir::Expr wrapped_body = ir::IfThenElse::Make(cond, body);
      node->body = ir::Block::Make({wrapped_body});
    } else {
      *expr = body;
    }
  }

  void Visit(const ir::ScheduleBlockRealize* op, ir::Expr* expr) override {
    auto* schedule_block = op->schedule_block.As<ir::ScheduleBlock>();
    // find and set the root schedule block
    if (schedule_block->name.substr(0, 4) == "root") {
      root_ = expr;
      ir::IRMutator<>::Visit(op, expr);
      root_ = nullptr;
    }
  }

 private:
  const ir::Expr* root_{nullptr};
};

}  // namespace

void EliminateInvariantLoop(ir::Expr* expr) { InvariantLoopEliminator()(expr); }

}  // namespace optim
}  // namespace cinn
