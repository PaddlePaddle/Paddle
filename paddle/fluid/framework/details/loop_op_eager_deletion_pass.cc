// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/framework/details/computation_op_handle.h"
#include "paddle/fluid/framework/details/multi_devices_helper.h"
#include "paddle/fluid/framework/ir/graph_helper.h"
#include "paddle/fluid/operators/controlflow/loop_op_helper.h"

namespace paddle {
namespace framework {
namespace details {

struct LoopOpList {
  std::vector<OperatorBase *> while_ops_;
  std::vector<OperatorBase *> while_grad_ops_;
  std::vector<OperatorBase *> recurrent_ops_;
  std::vector<OperatorBase *> recurrent_grad_ops_;
};

class LoopOpEagerDeletionPass : public ir::Pass {
 protected:
  std::unique_ptr<ir::Graph> ApplyImpl(
      std::unique_ptr<ir::Graph> graph) const override {
    auto all_ops = ir::FilterByNodeWrapper<OpHandleBase>(*graph);

    // Find all while_op and while_grad_op
    std::unordered_map<size_t, LoopOpList> target_ops;
    for (auto *op_handle : all_ops) {
      auto compute_op = dynamic_cast<ComputationOpHandle *>(op_handle);
      if (compute_op == nullptr) continue;

      auto *op = compute_op->GetOp();
      size_t idx = compute_op->GetScopeIdx();
      auto &op_name = op->Type();

      if (op_name == "while") {
        target_ops[idx].while_ops_.emplace_back(op);
      } else if (op_name == "while_grad") {
        target_ops[idx].while_grad_ops_.emplace_back(op);
      } else if (op_name == "recurrent") {
        target_ops[idx].recurrent_ops_.emplace_back(op);
      } else if (op_name == "recurrent_grad") {
        target_ops[idx].recurrent_grad_ops_.emplace_back(op);
      }
    }

    for (auto &ops_pair : target_ops) {
      auto &ops = ops_pair.second;
      operators::PrepareSafeEagerDeletionOnLoopOps(
          ops.while_ops_, ops.while_grad_ops_, ops.recurrent_ops_,
          ops.recurrent_grad_ops_);
    }
    return graph;
  }
};

}  // namespace details
}  // namespace framework
}  // namespace paddle

REGISTER_PASS(loop_op_eager_deletion_pass,
              paddle::framework::details::LoopOpEagerDeletionPass);
