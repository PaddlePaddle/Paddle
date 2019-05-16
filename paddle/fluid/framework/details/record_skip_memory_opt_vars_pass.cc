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
#include "paddle/fluid/framework/details/memory_optimize_helper.h"
#include "paddle/fluid/framework/ir/graph.h"
#include "paddle/fluid/framework/ir/graph_helper.h"
#include "paddle/fluid/framework/ir/pass.h"
#include "paddle/fluid/framework/op_proto_maker.h"

namespace paddle {
namespace framework {
namespace details {

class RecordSkipMemoryOptVarsPass : public ir::Pass {
 protected:
  void ApplyImpl(ir::Graph* graph) const override {
    PADDLE_ENFORCE(!graph->Has(kMemOptSkipVars));
    graph->Set(kMemOptSkipVars, new MemOptSkipVars);
    auto& skip_vars = graph->Get<MemOptSkipVars>(kMemOptSkipVars);

    // NOTE(zcd): Insert OpRoleVars to SkipVarSet to prevent the vars are rename
    // in memory optimize pass.
    InsertOpRoleVarsToSkipVarSet(graph, &skip_vars);
  }

  void InsertOpRoleVarsToSkipVarSet(const ir::Graph* graph,
                                    MemOptSkipVars* skip_vars) const {
    for (auto& node : graph->Nodes()) {
      PADDLE_ENFORCE_NOT_NULL(node, "The node should not be nullptr.");
      if (node->IsOp() && node->Op()) {
        try {
          auto op_role_vars =
              boost::get<std::vector<std::string>>(node->Op()->GetNullableAttr(
                  OpProtoAndCheckerMaker::OpRoleVarAttrName()));
          PADDLE_ENFORCE_EQ(op_role_vars.size() % 2, 0);
          for (size_t i = 0; i < op_role_vars.size(); i += 2) {
            auto& g_name = op_role_vars[i + 1];
            skip_vars->insert(g_name);
          }
        } catch (boost::bad_get e) {
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
