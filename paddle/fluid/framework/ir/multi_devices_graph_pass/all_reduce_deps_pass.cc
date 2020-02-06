// Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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
#include <map>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "paddle/fluid/framework/details/all_reduce_op_handle.h"
#include "paddle/fluid/framework/details/container_cast.h"
#include "paddle/fluid/framework/details/fused_all_reduce_op_handle.h"
#include "paddle/fluid/framework/details/multi_devices_helper.h"
#include "paddle/fluid/framework/ir/graph.h"
#include "paddle/fluid/framework/ir/graph_helper.h"
#include "paddle/fluid/framework/ir/pass.h"
#include "paddle/fluid/framework/op_proto_maker.h"

namespace paddle {
namespace framework {
namespace ir {

class AllReduceDepsPass : public ir::Pass {
 protected:
  void ApplyImpl(ir::Graph* graph) const override {
    std::vector<details::OpHandleBase*> all_reduce_op_handles =
        GetSortedAllReduceOps(*graph);

#if defined(PADDLE_WITH_NCCL)
    auto use_hierarchical_allreduce =
        Get<bool>(details::kUseHierarchicalAllReduce);
    for (size_t i = 0; i < all_reduce_op_handles.size(); ++i) {
      auto op_handle =
          dynamic_cast<details::NCCLOpHandleBase*>(all_reduce_op_handles[i]);
      PADDLE_ENFORCE(op_handle, "op_handle must be NCCLOpHandleBase");
      op_handle->SetRunEnv(i, use_hierarchical_allreduce);
    }
#endif

    for (size_t i = 1; i < all_reduce_op_handles.size(); ++i) {
      auto* dep_var = new details::DummyVarHandle(graph->CreateControlDepVar());
      graph->Get<details::GraphDepVars>(details::kGraphDepVars)
          .emplace(dep_var);
      all_reduce_op_handles[i - 1]->AddOutput(dep_var);
      all_reduce_op_handles[i]->AddInput(dep_var);
    }

    if (VLOG_IS_ON(10)) {
      DebugString(*graph, all_reduce_op_handles);
    }
  }

  std::vector<details::OpHandleBase*> GetSortedAllReduceOps(
      const ir::Graph& graph) const {
    std::vector<details::OpHandleBase*> all_reduce_op_handles;
    std::unordered_map<details::OpHandleBase*, size_t> pending_ops;
    std::unordered_set<details::OpHandleBase*> ready_ops;
    std::unordered_set<details::OpHandleBase*> next_ready_ops;
    auto op_handles = ir::FilterByNodeWrapper<details::OpHandleBase>(graph);
    size_t num_of_ops = op_handles.size();
    for (details::OpHandleBase* op : op_handles) {
      size_t not_ready_vars = op->NotReadyInputSize();
      if (not_ready_vars) {
        pending_ops.insert({op, not_ready_vars});
      } else {
        ready_ops.insert(op);
      }
    }

    GetSortedAllReduceOps(ready_ops, &all_reduce_op_handles);

    size_t has_run_ops = ready_ops.size();
    while (has_run_ops != num_of_ops) {
      for (auto* op : ready_ops) {
        for (auto& ready_var : op->Outputs()) {
          for (auto* pend_op : ready_var->PendingOps()) {
            auto& deps = --pending_ops[pend_op];
            if (deps == 0) {
              next_ready_ops.insert(pend_op);
            }
          }
        }
      }

      PADDLE_ENFORCE_NE(next_ready_ops.size(), 0, "There maybe have a cycle.");
      ready_ops.clear();
      std::swap(ready_ops, next_ready_ops);
      GetSortedAllReduceOps(ready_ops, &all_reduce_op_handles);
      has_run_ops += ready_ops.size();
    }
    return all_reduce_op_handles;
  }

  void GetSortedAllReduceOps(
      const std::unordered_set<details::OpHandleBase*>& ready_ops,
      std::vector<details::OpHandleBase*>* all_reduce_op_handles) const {
    std::vector<details::OpHandleBase*> current_all_reduce_op_handles;
    for (auto& op_handle : ready_ops) {
      auto all_reduce_op_handle =
          dynamic_cast<details::AllReduceOpHandle*>(op_handle);
      auto fused_all_reduce_op_handle =
          dynamic_cast<details::FusedAllReduceOpHandle*>(op_handle);

      if (all_reduce_op_handle || fused_all_reduce_op_handle) {
        current_all_reduce_op_handles.emplace_back(op_handle);
      }
    }

    // NOTE(zcd): For distributed training, it is important to keep the order of
    // allReduce on each node consistent. Otherwise, hang may occur.
    // Sort the current_all_reduce_op_handles according to the name of input.
    sort(current_all_reduce_op_handles.begin(),
         current_all_reduce_op_handles.end(),
         [](const details::OpHandleBase* left,
            const details::OpHandleBase* right) -> bool {
           auto left_in_vars =
               details::DynamicCast<details::VarHandle>(left->Inputs());
           auto right_in_vars =
               details::DynamicCast<details::VarHandle>(right->Inputs());
           PADDLE_ENFORCE_GT(left_in_vars.size(), 0);
           PADDLE_ENFORCE_GT(right_in_vars.size(), 0);
           return left_in_vars[0]->Name() > right_in_vars[0]->Name();
         });

    all_reduce_op_handles->insert(all_reduce_op_handles->end(),
                                  current_all_reduce_op_handles.begin(),
                                  current_all_reduce_op_handles.end());
  }

  void DebugString(
      const ir::Graph& graph,
      const std::vector<details::OpHandleBase*>& all_reduce_op_handles) const {
    // get vars order
    std::map<int, std::vector<std::string>> vars =
        GetSoredGradientsFromStaleProgram(graph);
    std::stringstream out;
    size_t grads_of_stale_program = 0;
    out << "Get Order From details::kStaleProgramOpDescs: ";
    for (auto& var : vars) {
      out << "Order " << var.first << " [";
      for (auto& var_name : var.second) {
        out << var_name << ", ";
        ++grads_of_stale_program;
      }
      out << "], ";
    }
    VLOG(10) << out.str();

    std::stringstream out2;
    out2 << "Get Order From Topological order: ";
    for (auto& op : all_reduce_op_handles) {
      bool find_valid_input = false;
      for (auto& in_var : op->Inputs()) {
        if (dynamic_cast<details::VarHandle*>(in_var)) {
          out2 << in_var->Name() << ", ";
          find_valid_input = true;
          break;
        }
      }
      PADDLE_ENFORCE(find_valid_input, "Doesn't find valid input.");
    }
    VLOG(10) << out2.str();
    if (grads_of_stale_program != all_reduce_op_handles.size()) {
      VLOG(10)
          << "The gradients number of stale program and graph is not equal.";
    }
  }

  std::map<int, std::vector<std::string>> GetSoredGradientsFromStaleProgram(
      const ir::Graph& graph) const {
    std::map<int, std::vector<std::string>> vars;
    auto ops =
        graph.Get<const std::vector<OpDesc*>>(details::kStaleProgramOpDescs);
    int order = 0;
    for (auto* op_desc : ops) {
      bool is_bk_op = details::IsOpRole(*op_desc, OpRole::kBackward);
      if (!is_bk_op) continue;

      auto backward_vars = details::GetOpRoleVarsOrEmpty(*op_desc);
      if (backward_vars.empty()) continue;

      for (size_t i = 1; i < backward_vars.size(); i += 2) {
        vars[order].emplace_back(backward_vars[i]);
        VLOG(1) << "get parameter and gradient: " << backward_vars[i - 1]
                << ", " << backward_vars[i];
      }
      order++;
    }
    return vars;
  }
};
}  // namespace ir
}  // namespace framework
}  // namespace paddle

REGISTER_PASS(all_reduce_deps_pass, paddle::framework::ir::AllReduceDepsPass)
    .RequireGraphAttr(paddle::framework::details::kStaleProgramOpDescs);
