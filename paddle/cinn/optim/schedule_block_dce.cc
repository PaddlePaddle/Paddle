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

#include "paddle/cinn/optim/schedule_block_dce.h"

#include "paddle/cinn/ir/ir_mutator.h"
#include "paddle/cinn/ir/ir_printer.h"
#include "paddle/cinn/optim/replace_var_with_expr.h"

namespace cinn {
namespace optim {

struct ScheduleBlockDCE : public ir::IRMutator<Expr*> {
  explicit ScheduleBlockDCE(const std::vector<std::string>& output_names)
      : output_names_(output_names.begin(), output_names.end()) {}

  void operator()(ir::Expr* expr) {
    UpdateDeadScheduleBlocks(*expr);
    while (!dead_schedule_block_names_.empty()) {
      Visit(expr);
      UpdateDeadScheduleBlocks(*expr);
    }
  }

 private:
  void Visit(ir::Expr* expr) { ir::IRMutator<>::Visit(expr, expr); }

  void Visit(const ir::Block* op, Expr* expr) override {
    auto* node = expr->As<ir::Block>();
    PADDLE_ENFORCE_NOT_NULL(node,
                            ::common::errors::InvalidArgument(
                                "Sorry, but expr->As node is nullptr"));
    for (auto& stmt : node->stmts) {
      IRMutator::Visit(&stmt, &stmt);
    }

    std::unordered_set<int> need_remove_ids;
    for (int i = 0; i < node->stmts.size(); ++i) {
      if (IsDeadScheduleBlock(node->stmts[i]) || IsEmptyBlock(node->stmts[i])) {
        need_remove_ids.insert(i);
      }
    }
    if (!need_remove_ids.empty()) {
      node->stmts = [&] {
        std::vector<ir::Expr> new_stmts;
        for (int i = 0; i < node->stmts.size(); ++i) {
          if (need_remove_ids.count(i) == 0) {
            new_stmts.push_back(node->stmts[i]);
          }
        }
        return new_stmts;
      }();
    }
  }

  void Visit(const ir::IfThenElse* op, Expr* expr) override {
    auto* node = expr->As<ir::IfThenElse>();
    PADDLE_ENFORCE_NOT_NULL(node,
                            ::common::errors::InvalidArgument(
                                "Sorry, but node expr->As is nullptr"));
    IRMutator::Visit(&node->true_case, &node->true_case);
    if (node->false_case.defined()) {
      IRMutator::Visit(&node->false_case, &node->false_case);
    }
    if (IsEmptyIf(op)) {
      *expr = ir::Block::Make({});
    }
  }

  void Visit(const ir::For* op, Expr* expr) override {
    auto* node = expr->As<ir::For>();
    PADDLE_ENFORCE_NOT_NULL(node,
                            ::common::errors::InvalidArgument(
                                "Sorry, but node expr->As is nullptr"));
    IRMutator::Visit(&(node->body), &(node->body));
    if (IsEmptyBlock(op->body)) {
      *expr = ir::Block::Make({});
    }
  }

  bool IsEmptyBlock(const ir::Expr& expr) {
    const auto* block_node = expr.As<ir::Block>();
    if (block_node == nullptr) return false;
    for (const auto& stmt : block_node->stmts) {
      if (!IsEmptyBlock(stmt)) return false;
    }
    return true;
  }

  bool IsEmptyIf(const ir::IfThenElse* node) {
    if (node->false_case.defined()) {
      return IsEmptyBlock(node->true_case) && IsEmptyBlock(node->false_case);
    }
    return IsEmptyBlock(node->true_case);
  }

  bool IsDeadScheduleBlock(const ir::Expr& expr) {
    const auto* sbr = expr.As<ir::ScheduleBlockRealize>();
    return sbr != nullptr &&
           dead_schedule_block_names_.count(
               sbr->schedule_block.As<ir::ScheduleBlock>()->name) > 0;
  }

  void UpdateDeadScheduleBlocks(const ir::Expr& expr) {
    dead_schedule_block_names_.clear();
    std::unordered_set<std::string> load_buffer_names;
    std::unordered_set<std::string> load_tensor_names;
    auto InsertLoadTensorAndBufferNames = [&](const ir::Expr* x) -> bool {
      if (const ir::Load* load = x->As<ir::Load>()) {
        load_buffer_names.insert(load->tensor.as_tensor()->buffer->name);
        load_tensor_names.insert(load->tensor.as_tensor()->name);
      }
      return false;
    };
    ir::ir_utils::CollectIRNodes(expr, InsertLoadTensorAndBufferNames);

    auto IsShareBufferWithLoadedTensor =
        [&](const ir::_Tensor_* tensor) -> bool {
      return load_buffer_names.count(tensor->buffer->name) > 0;
    };
    auto IsLoadedTensor = [&](const ir::_Tensor_* tensor) -> bool {
      return load_tensor_names.count(tensor->name) > 0;
    };
    auto IsOutputTensor = [&](const ir::_Tensor_* tensor) -> bool {
      return output_names_.count(tensor->name) > 0;
    };
    auto IsDeadStore = [&](const ir::Store* store) -> bool {
      const ir::_Tensor_* tensor = store->tensor.as_tensor();
      return !IsOutputTensor(tensor) && !IsLoadedTensor(tensor) &&
             !IsShareBufferWithLoadedTensor(tensor);
    };
    auto InsertDeadStoreName = [&](const ir::Expr* x) -> bool {
      const ir::Store* store = x->As<ir::Store>();
      if (store != nullptr && IsDeadStore(store)) {
        VLOG(6) << "Find dead schedule block: "
                << store->tensor.as_tensor()->name;
        dead_schedule_block_names_.insert(store->tensor.as_tensor()->name);
      }
      return false;
    };
    ir::ir_utils::CollectIRNodes(expr, InsertDeadStoreName);
  }

 private:
  std::unordered_set<std::string> dead_schedule_block_names_;
  std::unordered_set<std::string> output_names_;
};

void EliminateDeadScheduleBlock(Expr* e,
                                const std::vector<std::string>& output_names) {
  VLOG(6) << "Start EliminateDeadScheduleBlock" << *e;
  ScheduleBlockDCE eliminator(output_names);
  eliminator(e);
  VLOG(6) << "End EliminateDeadScheduleBlock: " << *e;
}

}  // namespace optim
}  // namespace cinn
