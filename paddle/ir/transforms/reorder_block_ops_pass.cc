// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/ir/transforms/reorder_block_ops_pass.h"

#include <queue>

#include "paddle/ir/core/builtin_op.h"
#include "paddle/ir/core/program.h"
#include "paddle/ir/pass/pass.h"

namespace {

// TODO(wilber): After support SideEffectTrait, Only NoSideEffectTrait op can be
// removed by dce pass.
// Now just a naive implementation.
class ReorderBlockOpsPass : public ir::Pass {
 public:
  ReorderBlockOpsPass() : ir::Pass("ReorderBlockOpsPass", 0) {}

  void Run(ir::Operation *op) override {
    IR_ENFORCE(op->num_regions() > 0,
               "ReorderBlockOpsPass should run on Operation which regions "
               "number greater than 0.");
    for (size_t i = 0; i < op->num_regions(); ++i) {
      for (auto *block : op->region(i)) {
        std::list<ir::Operation *> res_op_list;
        std::unordered_map<ir::Operation *, int>
            reorder_op_dep_cnt;  // op -> dependent input count
        std::unordered_set<ir::Value> visited_values;
        std::queue<ir::Operation *> op_que;

        auto update_op_que = [&](ir::Operation *op) {
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

        for (auto &op : *block) {
          bool has_dependency = false;
          if (op->num_operands() > 0) {
            for (size_t i = 0; i < op->num_operands(); ++i) {
              auto operand = op->operand_source(i);
              if (operand && visited_values.count(op->operand_source(i)) == 0) {
                reorder_op_dep_cnt[op]++;
                has_dependency = true;
              }
            }
          }
          if (!has_dependency) {
            res_op_list.push_back(op);
            update_op_que(op);
          }
        }

        if (reorder_op_dep_cnt.empty()) {
          return;
        }

        while (!op_que.empty()) {
          auto *op = op_que.front();
          op_que.pop();
          res_op_list.push_back(op);
          update_op_que(op);
        }
        VLOG(4) << "ReorderBlockOpsPass is applied.";
        block->ResetOpListOrder(res_op_list);
      }
    }
  }

  bool CanApplyOn(ir::Operation *op) const override {
    return op->num_regions() > 0;
  }
};

}  // namespace

namespace ir {

std::unique_ptr<Pass> CreateReorderBlockOpsPass() {
  return std::make_unique<ReorderBlockOpsPass>();
}

}  // namespace ir
