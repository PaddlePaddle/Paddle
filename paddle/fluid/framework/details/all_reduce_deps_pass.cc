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
#include "paddle/fluid/framework/details/multi_devices_helper.h"
#include "paddle/fluid/framework/details/op_graph_view.h"
#include "paddle/fluid/framework/ir/graph.h"
#include "paddle/fluid/framework/ir/graph_helper.h"
#include "paddle/fluid/framework/ir/pass.h"
#include "paddle/fluid/framework/op_proto_maker.h"

namespace paddle {
namespace framework {
namespace details {

class AllReduceDepsPass : public ir::Pass {
 protected:
  void ApplyImpl(ir::Graph* graph) const override {
    std::vector<AllReduceOpHandle*> all_reduce_op_handles =
        GetSortedAllReduceOps(*graph);

    for (size_t i = 1; i < all_reduce_op_handles.size(); ++i) {
      auto* dep_var = new DummyVarHandle(graph->CreateControlDepVar());
      graph->Get<GraphDepVars>(kGraphDepVars).emplace(dep_var);
      all_reduce_op_handles[i - 1]->AddOutput(dep_var);
      all_reduce_op_handles[i]->AddInput(dep_var);
    }

    if (VLOG_IS_ON(10)) {
      DebugString(*graph, all_reduce_op_handles);
    }
  }

  std::vector<AllReduceOpHandle*> GetSortedAllReduceOps(
      const ir::Graph& graph) const {
    std::vector<AllReduceOpHandle*> all_reduce_op_handles;
    std::unordered_map<OpHandleBase*, size_t> pending_ops;
    std::unordered_set<OpHandleBase*> ready_ops;
    std::unordered_set<OpHandleBase*> next_ready_ops;

    auto op_handles = ir::FilterByNodeWrapper<OpHandleBase>(graph);
    size_t num_of_ops = op_handles.size();
    for (OpHandleBase* op : op_handles) {
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
      const std::unordered_set<OpHandleBase*>& ready_ops,
      std::vector<AllReduceOpHandle*>* all_reduce_op_handles) const {
    std::vector<AllReduceOpHandle*> current_all_reduce_op_handles;
    for (auto& op_handle : ready_ops) {
      auto all_reduce_op_handle = dynamic_cast<AllReduceOpHandle*>(op_handle);
      if (all_reduce_op_handle) {
        current_all_reduce_op_handles.emplace_back(all_reduce_op_handle);
      }
    }

    // NOTE(zcd): For distributed training, it is important to keep the order of
    // allReduce on each node consistent. Otherwise, hang may occur.
    // Sort the current_all_reduce_op_handles according to the name of input.
    sort(current_all_reduce_op_handles.begin(),
         current_all_reduce_op_handles.end(),
         [](const AllReduceOpHandle* left,
            const AllReduceOpHandle* right) -> bool {
           auto left_in_vars = DynamicCast<VarHandle>(left->Inputs());
           auto right_in_vars = DynamicCast<VarHandle>(right->Inputs());
           PADDLE_ENFORCE_GT(left_in_vars.size(), 0);
           PADDLE_ENFORCE_EQ(left_in_vars.size(), right_in_vars.size());
           return left_in_vars[0]->Name() > right_in_vars[0]->Name();
         });

    all_reduce_op_handles->insert(all_reduce_op_handles->end(),
                                  current_all_reduce_op_handles.begin(),
                                  current_all_reduce_op_handles.end());
  }

  void DebugString(
      const ir::Graph& graph,
      const std::vector<AllReduceOpHandle*>& all_reduce_op_handles) const {
    // get vars order
    std::map<int, std::vector<std::string>> vars =
        GetSoredGradientsFromStaleProgram(graph);
    std::stringstream out;
    size_t grads_of_stale_program = 0;
    out << "Get Order From kStaleProgramOpDescs: ";
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
        if (dynamic_cast<VarHandle*>(in_var)) {
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
    auto ops = graph.Get<const std::vector<OpDesc*>>(kStaleProgramOpDescs);
    int order = 0;
    for (auto* op_desc : ops) {
      try {
        bool is_bk_op =
            static_cast<bool>(boost::get<int>(op_desc->GetAttr(
                                  OpProtoAndCheckerMaker::OpRoleAttrName())) &
                              static_cast<int>(OpRole::kBackward));
        if (!is_bk_op) continue;

        auto backward_vars =
            boost::get<std::vector<std::string>>(op_desc->GetNullableAttr(
                OpProtoAndCheckerMaker::OpRoleVarAttrName()));
        if (backward_vars.empty()) continue;

        PADDLE_ENFORCE_EQ(backward_vars.size() % 2, 0);
        for (size_t i = 1; i < backward_vars.size(); i += 2) {
          vars[order].emplace_back(backward_vars[i]);
          VLOG(1) << "get parameter and gradient: " << backward_vars[i - 1]
                  << ", " << backward_vars[i];
        }
        order++;
      } catch (boost::bad_get e) {
      }
    }
    return vars;
  }
};
}  // namespace details
}  // namespace framework
}  // namespace paddle

REGISTER_PASS(all_reduce_deps_pass,
              paddle::framework::details::AllReduceDepsPass)
    .RequireGraphAttr(paddle::framework::details::kStaleProgramOpDescs);
