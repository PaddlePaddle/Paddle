/* Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/framework/ir/int8_placement_pass.h"
#include <string>

namespace paddle {
namespace framework {
namespace ir {

std::unique_ptr<ir::Graph> INT8PlacementPass::ApplyImpl(
    std::unique_ptr<ir::Graph> graph) const {
  VLOG(3) << "Aplies INT8 placement strategy.";
  const auto& op_types_list =
      Get<std::unordered_set<std::string>>("int8_enabled_op_types");
  for (const Node* n : graph->Nodes()) {
    if (n->IsOp()) {
      auto* op = n->Op();
      if (op->HasAttr("use_int8") || op->HasProtoAttr("use_int8")) {
        if (op_types_list.empty()) {
          op->SetAttr("use_int8", true);
        } else if (std::find(op_types_list.begin(), op_types_list.end(),
                             n->Name()) != op_types_list.end()) {
          op->SetAttr("use_int8", true);
        }
      }
    }
  }
  return graph;
}

}  // namespace ir
}  // namespace framework
}  // namespace paddle

REGISTER_PASS(int8_placement_pass, paddle::framework::ir::INT8PlacementPass)
    .RequirePassAttr("int8_enabled_op_types");
