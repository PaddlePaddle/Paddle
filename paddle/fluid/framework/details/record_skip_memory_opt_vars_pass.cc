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

#include <string>
#include <unordered_set>
#include <vector>
#include "paddle/fluid/framework/details/memory_optimize_helper.h"
#include "paddle/fluid/framework/ir/graph.h"
#include "paddle/fluid/framework/ir/graph_helper.h"
#include "paddle/fluid/framework/ir/pass.h"
#include "paddle/fluid/framework/op_proto_maker.h"
#include "paddle/fluid/framework/operator.h"

namespace paddle {
namespace framework {
namespace details {

class RecordSkipMemoryOptVarsPass : public ir::Pass {
 protected:
  void ApplyImpl(ir::Graph* graph) const override {
    PADDLE_ENFORCE(!graph->Has(kMemOptSkipVars));
    graph->Set(kMemOptSkipVars, new MemOptSkipVars);
    auto& skip_vars = graph->Get<MemOptSkipVars>(kMemOptSkipVars);

    std::vector<ir::Node*> op_nodes;
    for (auto& node : graph->Nodes()) {
      PADDLE_ENFORCE_NOT_NULL(node, "The node should not be nullptr.");
      if (node->IsOp() && node->Op()) {
        op_nodes.emplace_back(node);
      }
    }

    // Insert kEmptyVarName to avoid optimizing empty variable
    skip_vars.insert(framework::kEmptyVarName);

    // NOTE(zcd): Insert OpRoleVars to SkipVarSet to prevent the vars are rename
    // in memory optimize pass.
    InsertOpRoleVarsToSkipVarSet(op_nodes, &skip_vars);

    InsertSkipMemOptOpInOutToSkipVarSet(op_nodes, &skip_vars);
  }

 private:
  static void InsertOpRoleVarsToSkipVarSet(const std::vector<ir::Node*>& ops,
                                           MemOptSkipVars* skip_vars) {
    for (auto& node : ops) {
      try {
        auto op_role_vars =
            boost::get<std::vector<std::string>>(node->Op()->GetNullableAttr(
                OpProtoAndCheckerMaker::OpRoleVarAttrName()));
        PADDLE_ENFORCE_EQ(op_role_vars.size() % 2, 0);
        for (size_t i = 0; i < op_role_vars.size(); i += 2) {
          auto& g_name = op_role_vars[i + 1];
          skip_vars->insert(g_name);
        }
      } catch (boost::bad_get& e) {
      }
    }
  }

  static void UpdateSkipVarSet(
      MemOptSkipVars* skip_vars,
      const std::vector<std::vector<std::string>>& var_names) {
    for (auto& var_name : var_names) {
      skip_vars->insert(var_name.begin(), var_name.end());
    }
  }

  static std::vector<std::string> ToGradVarName(
      const std::vector<std::string>& names) {
    std::vector<std::string> ret;
    ret.reserve(names.size());
    for (auto& name : names) {
      if (name != framework::kEmptyVarName) {
        ret.emplace_back(framework::GradVarName(name));
      }
    }
    return ret;
  }

  static void InsertSkipMemOptOpInOutToSkipVarSet(
      const std::vector<ir::Node*>& ops, MemOptSkipVars* skip_vars) {
    static std::unordered_set<std::string> kSkipMemOptOps{
        "send", "recv", "prefetch", "send_barrier", "fetch_barrier"};

    for (auto& node : ops) {
      auto* op_desc = node->Op();
      if (OpHasSubBlock(op_desc)) {
        UpdateSkipVarSet(skip_vars, {op_desc->InputArgumentNames(),
                                     op_desc->OutputArgumentNames()});
      }

      if (kSkipMemOptOps.count(op_desc->Type()) > 0) {
        UpdateSkipVarSet(skip_vars, {op_desc->InputArgumentNames(),
                                     op_desc->OutputArgumentNames()});
      }

      // FIXME(zjl): some ops use variables that are not from their
      // inputs or outputs. We do not have a nice method to solve this
      // issue yet. Currently, we should skip these variables when
      // memory optimization is enabled.
      auto op_type = op_desc->Type();
      if (op_type == "while_grad") {
        UpdateSkipVarSet(skip_vars, {ToGradVarName(op_desc->Input("X"))});
      } else if (op_type == "conditional_block_grad") {
        UpdateSkipVarSet(skip_vars, {ToGradVarName(op_desc->Input("Input")),
                                     ToGradVarName(op_desc->Input("Cond"))});
      } else if (op_type == "recurrent" || op_type == "recurrent_grad") {
        auto& ex_states =
            boost::get<std::vector<std::string>>(op_desc->GetAttr("ex_states"));
        auto& states =
            boost::get<std::vector<std::string>>(op_desc->GetAttr("states"));
        if (op_type == "recurrent") {
          UpdateSkipVarSet(skip_vars, {ex_states, states});
        } else {
          UpdateSkipVarSet(
              skip_vars,
              {ToGradVarName(op_desc->Input("parameters")),
               ToGradVarName(op_desc->Input("input")), ex_states, states,
               ToGradVarName(ex_states), ToGradVarName(states)});
        }
      }
    }
  }
};

}  // namespace details
}  // namespace framework
}  // namespace paddle

REGISTER_PASS(record_skip_memory_opt_vars_pass,
              paddle::framework::details::RecordSkipMemoryOptVarsPass);
