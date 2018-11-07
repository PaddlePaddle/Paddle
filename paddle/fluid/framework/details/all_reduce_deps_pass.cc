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
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "paddle/fluid/framework/details/all_reduce_deps_pass.h"
#include "paddle/fluid/framework/details/all_reduce_op_handle.h"
#include "paddle/fluid/framework/details/multi_devices_helper.h"
#include "paddle/fluid/framework/details/op_graph_view.h"
#include "paddle/fluid/framework/details/var_handle.h"
#include "paddle/fluid/framework/op_proto_maker.h"

namespace paddle {
namespace framework {
namespace details {

inline bool less(const AllReduceOpHandle* a1, const AllReduceOpHandle* a2) {
  PADDLE_ENFORCE(a1->Inputs().size() > 0 && a2->Inputs().size() > 0,
                 "All reduce must have > 1 input vars.");

  VarHandle* i0 = dynamic_cast<VarHandle*>(a1->Inputs()[0]);
  VarHandle* i1 = dynamic_cast<VarHandle*>(a2->Inputs()[0]);

  PADDLE_ENFORCE(i0 != nullptr && i1 != nullptr, "Convert to VarHandle error");

  return i0->name_ < i1->name_;
}

static void delete_from_old_level(
    std::vector<std::unordered_set<AllReduceOpHandle*>> allreduce_ops,
    AllReduceOpHandle* op, int idx) {
  PADDLE_ENFORCE(idx < static_cast<int>(allreduce_ops.size()),
                 "idx:%d must < vector size:%d", idx,
                 static_cast<int>(allreduce_ops.size()));
  auto& s = allreduce_ops[idx];
  bool bfind = (s.find(op) != s.end());
  PADDLE_ENFORCE(bfind, "insert to container error!");
  s.erase(op);
}

std::unique_ptr<ir::Graph> AllReduceDepsPass::ApplyImpl(
    std::unique_ptr<ir::Graph> graph) const {
  auto& graph_ops = graph->Get<GraphOps>(kGraphOps);
  OpGraphView graph_view(graph_ops);
  std::unordered_set<OpHandleBase*> cur_level_ops;
  std::vector<std::unordered_set<AllReduceOpHandle*>> allreduce_ops;
  std::unordered_map<OpHandleBase*, int> visited;

  for (auto* op : graph_view.AllOps()) {
    if (graph_view.PrecedingOps(op).empty()) {
      cur_level_ops.insert(op);
    }
  }
  VLOG(11) << "cur_level_ops:" << cur_level_ops.size() << std::endl;

  int cur_idx = 0;
  while (!cur_level_ops.empty()) {
    std::unordered_set<OpHandleBase*> next_level_ops;
    std::unordered_set<AllReduceOpHandle*> next_level_allreduce_ops;

    for (auto* op : cur_level_ops) {
      auto* allreduce_op = dynamic_cast<AllReduceOpHandle*>(op);
      if (allreduce_op != nullptr) {
        // VLOG(11) << "all_reduce_op:" << allreduce_op->Name() << std::endl;
        auto it = visited.find(allreduce_op);
        if (it == visited.end()) {
          visited[allreduce_op] = cur_idx;
          next_level_allreduce_ops.insert(allreduce_op);
        } else {
          if (it->second != cur_idx) {
            delete_from_old_level(allreduce_ops, allreduce_op, it->second);
            next_level_allreduce_ops.insert(allreduce_op);
          }
        }
      }

      for (auto* pending_op : graph_view.PendingOps(op)) {
        next_level_ops.insert(pending_op);
      }
    }

    VLOG(11) << "next_level_allreduce_ops:" << next_level_allreduce_ops.size()
             << std::endl;
    if (!next_level_allreduce_ops.empty()) {
      allreduce_ops.emplace_back(std::move(next_level_allreduce_ops));
      cur_idx++;
    }

    cur_level_ops.swap(next_level_ops);
    VLOG(11) << "cur_level_ops:" << cur_level_ops.size() << std::endl;
  }

  VLOG(11) << "allreduce_ops size:" << allreduce_ops.size() << std::endl;

  // std::vector<AllReduceOpHandle*> sorted_ops;
  for (auto& s : allreduce_ops) {
    std::vector<AllReduceOpHandle*> op_list;
    for (auto* n : s) {
      op_list.emplace_back(n);
    }

    VLOG(11) << "op_list size:" << op_list.size() << std::endl;

    // sort cur_level_ops by inputs[0].name
    std::sort(op_list.begin(), op_list.end(), less);

    /*
    for (auto* op : op_list) {
      sorted_ops.emplace_back(op);
    }
    */

    auto& sorted_ops = op_list;
    VLOG(11) << "sorted_list size:" << sorted_ops.size() << std::endl;

    // Add dependency.
    for (size_t i = 1; i < sorted_ops.size(); ++i) {
      auto* dep_var = new DummyVarHandle(graph->CreateControlDepVar());
      sorted_ops[i - 1]->AddOutput(dep_var);
      sorted_ops[i]->AddInput(dep_var);

      graph->Get<GraphDepVars>(kGraphDepVars).emplace(dep_var);

      VLOG(10) << "Add all_reduce Sequential dependencies between "
               << sorted_ops[i - 1] << " and " << sorted_ops[i];
    }
  }

  return graph;
}

}  // namespace details
}  // namespace framework
}  // namespace paddle

REGISTER_PASS(all_reduce_deps_pass,
              paddle::framework::details::AllReduceDepsPass)
    .RequirePassAttr(paddle::framework::details::kAllOpDescs);
