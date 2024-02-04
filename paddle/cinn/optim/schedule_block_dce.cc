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
      VLOG(6) << "node->stmts.size() = " << node->stmts.size();
      VLOG(6) << "i = " << i << ", node->stmts[i] = " << node->stmts[i];
      if (const auto* sbr = node->stmts[i].As<ir::ScheduleBlockRealize>()) {
        std::string name = sbr->schedule_block.As<ir::ScheduleBlock>()->name;
        VLOG(6) << "visit schedule block: " << name;
        if (dead_schedule_block_names_.count(name) > 0) {
          need_remove_ids.insert(i);
        }
      }
    }
    if (!need_remove_ids.empty()) {
      std::vector<ir::Expr> new_stmts;
      for (int i = 0; i < node->stmts.size(); ++i) {
        if (need_remove_ids.count(i) == 0) {
          VLOG(6) << "i = " << i << ", node->stmts[i] = " << node->stmts[i];
          new_stmts.push_back(node->stmts[i]);
        }
      }
      node->stmts = new_stmts;
    }

    for (auto& stmt : node->stmts) {
      IRMutator::Visit(&stmt, &stmt);
    }
  }

  void FindDeadScheduleBlocks(const ir::Expr& expr) {
    ir::ir_utils::CollectIRNodes(expr, [&](const ir::Expr* x) {
      if (const ir::Store* store = x->As<ir::Store>()) {
        std::string store_name = store->tensor.as_tensor()->name;
        if (output_names_.count(store_name) == 0) {
          dead_schedule_block_names_.insert(store->tensor.as_tensor()->name);
          VLOG(6) << "dead_schedule_block_names_.insert: "
                  << store->tensor.as_tensor()->name;
        }
      }
      return false;
    });
    std::unordered_set<std::string> load_buffer_names;
    ir::ir_utils::CollectIRNodes(expr, [&](const ir::Expr* x) {
      if (const ir::Load* load = x->As<ir::Load>()) {
        std::string load_name = load->tensor.as_tensor()->name;
        if (dead_schedule_block_names_.count(load_name)) {
          VLOG(6) << "dead_schedule_block_names_.erase: " << load_name;
          dead_schedule_block_names_.erase(load_name);
        }
        load_buffer_names.insert(load->tensor.as_tensor()->buffer->name);
      }
      return false;
    });
    ir::ir_utils::CollectIRNodes(expr, [&](const ir::Expr* x) {
      if (const ir::Store* store = x->As<ir::Store>()) {
        std::string store_name = store->tensor.as_tensor()->name;
        if (dead_schedule_block_names_.count(store_name) > 0 &&
            load_buffer_names.count(store->tensor.as_tensor()->buffer->name) >
                0) {
          dead_schedule_block_names_.erase(store_name);
        }
      }
      return false;
    });
    for (auto s : dead_schedule_block_names_) {
      VLOG(6) << "dead_schedule_block_name: " << s;
    }
  }

 private:
  std::unordered_set<std::string> dead_schedule_block_names_;
  std::unordered_set<std::string> output_names_;
};

void EliminateDeadScheduleBlock(Expr* e,
                                const std::vector<std::string>& output_names) {
  VLOG(6) << "start EliminateDeadScheduleBlock";
  ScheduleBlockDCE eliminator(output_names);
  eliminator(e);
  VLOG(6) << "end EliminateDeadScheduleBlock";
}

}  // namespace optim
}  // namespace cinn
