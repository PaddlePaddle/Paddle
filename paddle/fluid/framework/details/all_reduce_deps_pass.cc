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
#include "paddle/fluid/framework/ir/graph_helper.h"
#include "paddle/fluid/framework/op_proto_maker.h"

namespace paddle {
namespace framework {
namespace details {

static constexpr char kAllOpDescs[] = "all_op_descs";

VarHandle* GetValidInput(const AllReduceOpHandle* a) {
  for (auto p : a->Inputs()) {
    VarHandle* b = dynamic_cast<VarHandle*>(p);
    if (b) {
      return b;
    }
  }

  return nullptr;
}

std::unique_ptr<ir::Graph> AllReduceDepsPass::ApplyImpl(
    std::unique_ptr<ir::Graph> graph) const {
  auto graph_ops = ir::FilterByNodeWrapper<OpHandleBase>(*graph);

  // get vars level
  int level = 0;
  std::unordered_map<std::string, int> vars;
  auto& ops = Get<const std::vector<OpDesc*>>(kAllOpDescs);
  for (auto* op_desc : ops) {
    auto outputs = op_desc->Outputs();
    for (auto& o_it : outputs) {
      for (auto& v : o_it.second) {  // values
        vars[v] = level;
        level++;
      }
    }
  }

  std::vector<AllReduceOpHandle*> allreduce_ops;
  // get allreduce ops.
  for (auto& op : graph_ops) {
    auto* allreduce_op = dynamic_cast<AllReduceOpHandle*>(op);
    if (allreduce_op == nullptr) {
      continue;
    }

    allreduce_ops.push_back(allreduce_op);
  }

  VLOG(10) << "allreduce_ops size:" << allreduce_ops.size() << std::endl;

  std::sort(allreduce_ops.begin(), allreduce_ops.end(),
            [&](AllReduceOpHandle* op1, AllReduceOpHandle* op2) {
              VarHandle* i0 = dynamic_cast<VarHandle*>(GetValidInput(op1));
              VarHandle* i1 = dynamic_cast<VarHandle*>(GetValidInput(op2));

              PADDLE_ENFORCE(i0 != nullptr && i1 != nullptr,
                             "%s convert to %s error", op1->DebugString(),
                             op2->DebugString());

              auto l_it = vars.find(i0->name_);
              auto r_it = vars.find(i1->name_);

              if (l_it->second < r_it->second) return true;

              return false;
            });

  // add dependency.
  auto& sorted_ops = allreduce_ops;
  for (size_t i = 1; i < sorted_ops.size(); ++i) {
    auto* dep_var = new DummyVarHandle(graph->CreateControlDepVar());

    auto* pre_op = sorted_ops[i - 1];
    auto* op = sorted_ops[i];

    pre_op->AddOutput(dep_var);
    op->AddInput(dep_var);
    graph->Get<GraphDepVars>(kGraphDepVars).emplace(dep_var);

    VLOG(10) << "add all_reduce sequential dependencies between " << pre_op
             << " and " << op;

    VLOG(10) << "pre_op:" << pre_op->DebugString()
             << ", op:" << op->DebugString();
  }

  return graph;
}

}  // namespace details
}  // namespace framework
}  // namespace paddle

REGISTER_PASS(all_reduce_deps_pass,
              paddle::framework::details::AllReduceDepsPass)
    .RequirePassAttr(paddle::framework::details::kAllOpDescs);
