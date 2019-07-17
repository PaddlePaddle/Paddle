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

#include <algorithm>
#include "paddle/fluid/framework/details/computation_op_handle.h"
#include "paddle/fluid/framework/details/multi_devices_helper.h"
#include "paddle/fluid/framework/details/share_tensor_buffer_op_handle.h"
#include "paddle/fluid/framework/ir/graph.h"
#include "paddle/fluid/framework/ir/graph_helper.h"
#include "paddle/fluid/framework/ir/pass.h"

namespace paddle {
namespace framework {
namespace ir {

class MoveTensorSharingToComputeOpPass : public Pass {
 protected:
  void ApplyImpl(Graph *graph) const override;

 private:
  static void ClearShareOp(Graph *graph,
                           details::ShareTensorBufferOpHandle *share_op,
                           details::ComputationOpHandle *compute_op);
};

template <typename T>
void RemoveElementInVector(std::vector<T> *vec, const T &value) {
  vec->erase(std::remove(vec->begin(), vec->end(), value), vec->end());
}

void MoveTensorSharingToComputeOpPass::ApplyImpl(Graph *graph) const {
  auto all_ops = FilterByNodeWrapper<details::OpHandleBase>(*graph);
  std::vector<details::ShareTensorBufferOpHandle *> share_ops;
  for (auto *op : all_ops) {
    auto *share_buffer_op =
        dynamic_cast<details::ShareTensorBufferOpHandle *>(op);
    if (share_buffer_op) {
      share_ops.emplace_back(share_buffer_op);
    }
  }

  for (auto *share_op : share_ops) {
    auto *compute_op = GetUniquePendingComputationOpHandle(share_op);
    compute_op->SetShareTensorBufferFunctor(share_op->Functor());
    ClearShareOp(graph, share_op, compute_op);
  }
}

void MoveTensorSharingToComputeOpPass::ClearShareOp(
    Graph *graph, details::ShareTensorBufferOpHandle *share_op,
    details::ComputationOpHandle *compute_op) {
  // 1. remove all output var nodes
  auto out_var_nodes = share_op->Node()->outputs;
  for (auto *out_var_node : out_var_nodes) {
    auto *out_var_handle = &(out_var_node->Wrapper<details::VarHandleBase>());
    PADDLE_ENFORCE_NOT_NULL(
        dynamic_cast<details::DummyVarHandle *>(out_var_handle));

    RemoveElementInVector(compute_op->MutableInputs(), out_var_handle);
    RemoveElementInVector(&(compute_op->Node()->inputs), out_var_node);

    graph->RemoveNode(out_var_node);

    graph->Get<details::GraphDepVars>(details::kGraphDepVars)
        .erase(out_var_handle);
  }

  // 2. erase share_op in each input var nodes
  auto in_var_nodes = share_op->Node()->inputs;
  for (auto *in_var_node : in_var_nodes) {
    in_var_node->Wrapper<details::VarHandleBase>().RemoveOutput(
        share_op, share_op->Node());
  }

  // 3. Delete share_op
  graph->RemoveNode(share_op->Node());
}

}  // namespace ir
}  // namespace framework
}  // namespace paddle

REGISTER_PASS(move_tensor_sharing_to_compute_op_pass,
              paddle::framework::ir::MoveTensorSharingToComputeOpPass);
