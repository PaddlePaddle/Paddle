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

#include "paddle/cinn/hlir/dialect/operator/transforms/reorder_block_pass.h"

#include <queue>

#include "paddle/cinn/hlir/dialect/operator/ir/manual_op.h"
#include "paddle/fluid/pir/dialect/operator/ir/control_flow_op.h"
#include "paddle/pir/include/core/builtin_op.h"

namespace {

std::vector<::pir::Value> GetInputValues(const pir::Block* block) {
  std::vector<::pir::Value> inputs;
  std::unordered_set<::pir::Value> visited_values;
  std::unordered_set<::pir::Operation*> ops_set;
  // count all op's input Value
  for (const auto& op : *block) {
    for (auto& value : op.operands_source()) {
      if (!value || !value.type()) {
        continue;
      }
      if (!ops_set.count(value.defining_op()) && !visited_values.count(value)) {
        // if the input value owner op is not in ops_set, it's the block's input
        visited_values.insert(value);
        inputs.push_back(value);
      }
    }
  }
  return inputs;
}

std::vector<::pir::Value> GetInputValues(const pir::Operation* operation) {
  std::vector<::pir::Value> inputs;
  if (operation->num_regions() > 0 &&
      (operation->isa<paddle::dialect::IfOp>() ||
       operation->isa<cinn::dialect::GroupOp>())) {
    std::unordered_set<::pir::Value> visited_values;
    for (size_t i = 0; i < operation->num_regions(); ++i) {
      for (const auto& block : operation->region(i)) {
        auto block_inputs = GetInputValues(&block);
        for (auto& value : block_inputs) {
          if (!visited_values.count(value)) {
            inputs.push_back(value);
            visited_values.insert(value);
          }
        }
      }
    }
  } else {
    for (auto& value : operation->operands_source()) {
      if (value && value.type()) {
        inputs.push_back(value);
      }
    }
  }
  return inputs;
}

class ReorderBlockOpsPass : public pir::Pass {
 public:
  ReorderBlockOpsPass() : pir::Pass("ReorderBlockOpsPass", 0) {}

  void Run(pir::Operation* op) override {
    CHECK(op->num_regions() > 0)
        << "ReorderBlockOpsPass should run on Operation which regions "
           "number greater than 0.";
    for (size_t i = 0; i < op->num_regions(); ++i) {
      for (auto& block : op->region(i)) {
        std::list<pir::Operation*> res_op_list;
        std::unordered_map<pir::Operation*, int>
            reorder_op_dep_cnt;  // op -> dependent input count
        std::unordered_set<pir::Value> visited_values;
        std::queue<pir::Operation*> op_que;

        auto update_op_que = [&](pir::Operation* op) {
          for (size_t i = 0; i < op->results().size(); ++i) {
            auto result = op->result(i);
            visited_values.insert(result);
            for (auto it = result.use_begin(); it != result.use_end(); ++it) {
              if (reorder_op_dep_cnt.count(it->owner())) {
                reorder_op_dep_cnt[it->owner()]--;
                if (reorder_op_dep_cnt[it->owner()] == 0) {
                  op_que.push(it->owner());
                }
              }
            }
          }
        };

        for (auto& op : block) {
          bool has_dependency = false;
          const auto& op_inputs = GetInputValues(&op);
          if (!op_inputs.empty()) {
            for (const auto& operand : op_inputs) {
              if (operand && visited_values.count(operand) == 0) {
                reorder_op_dep_cnt[&op]++;
                has_dependency = true;
              }
            }
          }
          if (!has_dependency) {
            res_op_list.push_back(&op);
            update_op_que(&op);
          }
        }

        if (reorder_op_dep_cnt.empty()) {
          return;
        }

        while (!op_que.empty()) {
          auto* op = op_que.front();
          op_que.pop();
          res_op_list.push_back(op);
          update_op_que(op);
        }
        VLOG(4) << "ReorderBlockOpsPass is applied.";
        block.ResetOpListOrder(res_op_list);
      }
    }
  }

  bool CanApplyOn(pir::Operation* op) const override {
    return op->isa<pir::ModuleOp>() && op->num_regions() > 0;
  }
};

}  // namespace

namespace cinn {
namespace dialect {
namespace ir {

std::unique_ptr<pir::Pass> CreateReorderBlockPass() {
  return std::make_unique<ReorderBlockOpsPass>();
}

}  // namespace ir
}  // namespace dialect
}  // namespace cinn
