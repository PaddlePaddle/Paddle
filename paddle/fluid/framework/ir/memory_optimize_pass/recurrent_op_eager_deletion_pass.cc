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

#include "paddle/fluid/framework/ir/memory_optimize_pass/recurrent_op_eager_deletion_pass.h"

#include <unordered_map>
#include <vector>

#include "paddle/fluid/framework/details/computation_op_handle.h"
#include "paddle/fluid/framework/details/multi_devices_helper.h"
#include "paddle/fluid/framework/ir/graph_helper.h"
#include "paddle/fluid/string/string_helper.h"

namespace paddle {
namespace framework {
namespace ir {

using paddle::operators::OpAndGradOpPair;
using paddle::operators::OpVariant;
using paddle::operators::OpVariantSet;

void RecurrentOpEagerDeletionPass::ApplyImpl(Graph *graph) const {
  // Find all recurrent_op and recurrent_grad_op in graph
  // Note the graph only contains ops and block 0
  std::unordered_map<size_t, OpAndGradOpPair> target_ops =
      DeviceIdToRecurrentAndRecurrentGradOp(*graph);

  for (auto &entry : target_ops) {
    // Prepare safe eager deletion on different devices because the garbage
    // collection may be different across devices
    OpAndGradOpPair &op_pair = entry.second;
    PrepareSafeEagerDeletionOnRecurrentOpAndRecurrentGradOp(
        graph->OriginProgram(), &op_pair);
  }

  auto all_ops = ir::FilterByNodeWrapper<details::OpHandleBase>(*graph);
  for (auto op_hander : all_ops) {
    auto *compute_op = dynamic_cast<details::ComputationOpHandle *>(op_hander);
    if (compute_op == nullptr) continue;
    if (compute_op->Name() == "recurrent" ||
        compute_op->Name() == "recurrent_grad") {
      ir::Node *op_node = op_hander->Node();
      auto *op_base = compute_op->GetOp();
      if (op_base->Attrs().count("skip_eager_deletion_vars")) {
        op_node->Op()->SetAttr("skip_eager_deletion_vars",
                               op_base->Attrs().at("skip_eager_deletion_vars"));
      }
    }
  }
}

// Returns a std::unordered_map mapping from the device id to recurrent op and
// grad op pair
std::unordered_map<size_t, OpAndGradOpPair>
RecurrentOpEagerDeletionPass::DeviceIdToRecurrentAndRecurrentGradOp(
    const Graph &graph) const {
  std::unordered_map<size_t, OpAndGradOpPair> ret;
  std::vector<details::OpHandleBase *> all_ops =
      FilterByNodeWrapper<details::OpHandleBase>(graph);

  for (auto *op : all_ops) {
    auto compute_op = dynamic_cast<details::ComputationOpHandle *>(op);
    if (compute_op == nullptr) continue;

    if (compute_op->Name() == "recurrent") {
      // GetScopeIdx() returns device/place id
      ret[compute_op->GetScopeIdx()].first.emplace(compute_op->GetOp());
    } else if (compute_op->Name() == "recurrent_grad") {
      // GetScopeIdx() returns device/place id
      ret[compute_op->GetScopeIdx()].second.emplace(compute_op->GetOp());
    }
  }
  return ret;
}

}  // namespace ir
}  // namespace framework
}  // namespace paddle

REGISTER_PASS(recurrent_op_eager_deletion_pass,
              paddle::framework::ir::RecurrentOpEagerDeletionPass);
