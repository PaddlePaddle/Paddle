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
    FindDeadScheduleBlocks(*expr);
    Visit(expr);
  }

 private:
  void Visit(ir::Expr* expr) { ir::IRMutator<>::Visit(expr, expr); }

  void Visit(const ir::Block* op, Expr* expr) override {
    auto* node = expr->As<ir::Block>();
    CHECK(node);

    std::unordered_set<int> need_remove_ids;
    for (int i = 0; i < node->stmts.size(); ++i) {
      if (IsDeadScheduleBlock(node->stmts[i])) {
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

    for (auto& stmt : node->stmts) {
      IRMutator::Visit(&stmt, &stmt);
    }
  }

  bool IsDeadScheduleBlock(const ir::Expr& expr) {
    const auto* sbr = expr.As<ir::ScheduleBlockRealize>();
    return sbr != nullptr &&
           dead_schedule_block_names_.count(
               sbr->schedule_block.As<ir::ScheduleBlock>()->name) > 0;
  }

  void FindDeadScheduleBlocks(const ir::Expr& expr) {
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
