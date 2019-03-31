//   Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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
#include <string>
#include <vector>

#include "paddle/fluid/framework/details/all_reduce_op_handle.h"
#include "paddle/fluid/framework/details/container_cast.h"
#include "paddle/fluid/framework/details/fused_all_reduce_op_handle.h"
#include "paddle/fluid/framework/details/multi_devices_helper.h"
#include "paddle/fluid/framework/ir/graph_helper.h"

namespace paddle {
namespace framework {
namespace details {

class FuseAllReduceOpPass : public ir::Pass {
 protected:
  void ApplyImpl(ir::Graph *graph) const override {
    ir::Graph &result = *graph;

    auto &places = Get<const std::vector<platform::Place>>(kPlaces);
    auto &local_scopes = Get<const std::vector<Scope *>>(kLocalScopes);
#if defined(PADDLE_WITH_CUDA) && !defined(_WIN32)
    auto *nccl_ctxs = &Get<platform::NCCLContextMap>(kNCCLCtxs);
#endif

    std::unordered_set<std::string> grads;
    auto &params_grads = result.Get<ParamsAndGrads>(kParamsAndGrads);
    size_t num_of_all_reduce = params_grads.size();
    grads.reserve(num_of_all_reduce);
    for (auto p_g : params_grads) {
      grads.insert(p_g.second);
    }

    size_t num_place = places.size();
    std::unordered_map<std::string, ir::Node *> all_reduce_ops;
    all_reduce_ops.reserve(grads.size());
    for (auto &node : result.Nodes()) {
      if (node->IsOp()) {
        PADDLE_ENFORCE(node->IsWrappedBy<OpHandleBase>());
        auto *all_reduce_op_handle =
            dynamic_cast<AllReduceOpHandle *>(&node->Wrapper<OpHandleBase>());
        if (all_reduce_op_handle) {
          auto inputs = DynamicCast<VarHandle>(all_reduce_op_handle->Inputs());
          PADDLE_ENFORCE_EQ(inputs.size(), num_place);
          // The inputs' name should be the same.
          auto &grad_name = inputs[0]->name();
          for (size_t i = 1; i < inputs.size(); ++i) {
            PADDLE_ENFORCE_EQ(inputs[i]->name(), grad_name,
                              "The input name should be the same.");
          }
          PADDLE_ENFORCE_NE(grads.count(grad_name), static_cast<size_t>(0));
          all_reduce_ops.emplace(grad_name, node);
        }
      }
    }

    VLOG(10) << "Find all_reduce_ops: " << all_reduce_ops.size();
    if (all_reduce_ops.size() == 0) {
      return;
    }

    PADDLE_ENFORCE_EQ(all_reduce_ops.size(), grads.size(),
                      "The number of all_reduce OpHandle is not equal to the "
                      "number of grads. Maybe some gradients are sparse type, "
                      "it is not supported currently.");
    VLOG(10) << "Insert fused_all_reduce";

    auto &group_grads_params =
        graph->Get<GroupGradsAndParams>(kGroupGradsAndParams);

    for (auto &group_g_p : group_grads_params) {
      size_t group_size = group_g_p.size();
      PADDLE_ENFORCE_GT(group_size, static_cast<size_t>(0));
      std::vector<ir::Node *> group_all_reduce_ops;
      group_all_reduce_ops.reserve(group_size);
      for (auto &g_p : group_g_p) {
        group_all_reduce_ops.emplace_back(all_reduce_ops.at(g_p.first));
      }
#if defined(PADDLE_WITH_CUDA) && !defined(_WIN32)
      InsertFusedAllReduce(places, local_scopes, group_size,
                           group_all_reduce_ops, nccl_ctxs, &result);
#else
      InsertFusedAllReduce(places, local_scopes, group_size,
                           group_all_reduce_ops, &result);
#endif
    }
  }

  void InsertFusedAllReduce(const std::vector<platform::Place> &places,
                            const std::vector<Scope *> &local_scopes,
                            const size_t num_of_all_reduce,
                            const std::vector<ir::Node *> &all_reduce_ops,
#if defined(PADDLE_WITH_CUDA) && !defined(_WIN32)
                            const platform::NCCLContextMap *nccl_ctxs,
#endif
                            ir::Graph *result) const {
    std::vector<VarHandleBase *> inputs;
    std::vector<VarHandleBase *> outputs;
    for (auto &op : all_reduce_ops) {
      auto &op_handle = op->Wrapper<OpHandleBase>();
      inputs.insert(inputs.end(), op_handle.Inputs().begin(),
                    op_handle.Inputs().end());
      // Remove output
      for_each(op_handle.Inputs().begin(), op_handle.Inputs().end(),
               [&op_handle](VarHandleBase *var_handle) {
                 var_handle->RemoveOutput(&op_handle, op_handle.Node());
               });

      outputs.insert(outputs.end(), op_handle.Outputs().begin(),
                     op_handle.Outputs().end());
      // Remove Input
      for_each(
          op_handle.Outputs().begin(), op_handle.Outputs().end(),
          [](VarHandleBase *var_handle) { var_handle->ClearGeneratedOp(); });

      result->RemoveNode(op_handle.Node());
    }

#if defined(PADDLE_WITH_CUDA) && !defined(_WIN32)
    CreateFusedAllReduceOp(inputs, outputs, num_of_all_reduce, places,
                           local_scopes, nccl_ctxs, result);
#else
    CreateFusedAllReduceOp(inputs, outputs, num_of_all_reduce, places,
                           local_scopes, result);
#endif
  }

 private:
  void CreateFusedAllReduceOp(const std::vector<VarHandleBase *> &inputs,
                              const std::vector<VarHandleBase *> &outputs,
                              const size_t num_of_all_reduce,
                              const std::vector<platform::Place> &places,
                              const std::vector<Scope *> &local_scopes,
#if defined(PADDLE_WITH_CUDA) && !defined(_WIN32)
                              const platform::NCCLContextMap *nccl_ctxs,
#endif
                              ir::Graph *result) const {
#if defined(PADDLE_WITH_CUDA) && !defined(_WIN32)
    auto *op_handle = new FusedAllReduceOpHandle(
        result->CreateEmptyNode("fused_all_reduce", ir::Node::Type::kOperation),
        local_scopes, places, num_of_all_reduce, nccl_ctxs);
#else
    auto *op_handle = new FusedAllReduceOpHandle(
        result->CreateEmptyNode("fused_all_reduce", ir::Node::Type::kOperation),
        local_scopes, places, num_of_all_reduce);
#endif

    for (auto in : inputs) {
      op_handle->AddInput(in);
    }

    for (auto out : outputs) {
      op_handle->AddOutput(out);
    }

#if defined(PADDLE_WITH_CUDA) && !defined(_WIN32)
    if (!nccl_ctxs) {
      SetCommunicationContext(places, op_handle);
    }
#else
    SetCommunicationContext(places, op_handle);
#endif
  }

  void SetCommunicationContext(const std::vector<platform::Place> &places,
                               FusedAllReduceOpHandle *op_handle) const {
    for (size_t i = 0; i < places.size(); ++i) {
      op_handle->SetDeviceContext(
          places[i], platform::DeviceContextPool::Instance().Get(places[i]));
    }
  }
};

}  // namespace details
}  // namespace framework
}  // namespace paddle

REGISTER_PASS(fuse_all_reduce_op_pass,
              paddle::framework::details::FuseAllReduceOpPass);
