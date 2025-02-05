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

#include "paddle/fluid/framework/ir/placement_pass_base.h"

#include <string>

#include "paddle/fluid/framework/operator.h"

namespace paddle::framework::ir {

void PlacementPassBase::ApplyImpl(ir::Graph* graph) const {
  VLOG(3) << "Applies " << GetPlacementName() << " placement strategy.";
  std::string attr_name = GetAttrName();
  const auto& op_types_list = GetOpTypesList();
  if (!graph->Has(attr_name)) {
    graph->Set<bool>(attr_name, new bool(true));
  }
  for (const Node* n : graph->Nodes()) {
    if (n->IsOp()) {
      auto* op = n->Op();
      if (IsSupport(n)) {
        if (op_types_list.empty() ||
            std::find(op_types_list.begin(), op_types_list.end(), n->Name()) !=
                op_types_list.end()) {
          op->SetAttr(attr_name, true);
        }
      }
    }
  }
}

}  // namespace paddle::framework::ir
