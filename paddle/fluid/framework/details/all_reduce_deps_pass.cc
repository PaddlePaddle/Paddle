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
#include <memory>
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

VarHandle* GetValidInput(const OpHandleBase* a) {
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

  // get vars order
  int order = 0;
  std::unordered_map<std::string, int> vars;
  // TODO(gongwb): use graph topology sort to find the order of operators.
  //               Note that must assert topology sort is stable
  auto& ops = graph->Get<const std::vector<OpDesc*>>(kStaleProgramOpDescs);
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
      PADDLE_ENFORCE_EQ(backward_vars.size() % 2, 0);

      auto outputs = op_desc->Outputs();
      for (auto& o_it : outputs) {
        for (auto& v : o_it.second) {  // values
          vars[v] = order;
          VLOG(1) << "in all_reduce_deps_pass:" << v;
        }
      }
      order++;
    } catch (boost::bad_get e) {
    }
  }

  std::vector<OpHandleBase*> dist_ops;
  // get allreduce ops.
  for (auto& op : graph_ops) {
    // FIXME(gongwb):add broad cast.
    if (op->Name() == "all_reduce" || op->Name() == "reduce") {
      dist_ops.push_back(op);
    }
  }

  VLOG(10) << "dist_ops size:" << dist_ops.size()
           << ", outputs size:" << vars.size() << ", ops size:" << ops.size();

  std::sort(dist_ops.begin(), dist_ops.end(), [&](OpHandleBase* op1,
                                                  OpHandleBase* op2) {
    VarHandle* i0 = dynamic_cast<VarHandle*>(GetValidInput(op1));
    VarHandle* i1 = dynamic_cast<VarHandle*>(GetValidInput(op2));

    PADDLE_ENFORCE(i0 != nullptr && i1 != nullptr, "%s convert to %s error",
                   op1->DebugString(), op2->DebugString());

    auto l_it = vars.find(i0->name());
    auto r_it = vars.find(i1->name());

    PADDLE_ENFORCE(l_it != vars.end() && r_it != vars.end(),
                   "can't find var's name %s and %s in opdesc", i0->name(),
                   i1->name());

    if (l_it->second < r_it->second) return true;

    if (l_it->second == r_it->second) {
      return i0->name() < i1->name();
    }

    return false;
  });

  // add dependency.
  auto& sorted_ops = dist_ops;
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
    .RequireGraphAttr(paddle::framework::details::kStaleProgramOpDescs);
