/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/framework/ir/mkldnn/mkldnn_placement_pass.h"
#include <string>

namespace paddle {
namespace framework {
namespace ir {

std::unique_ptr<ir::Graph> MKLDNNPlacementPass::ApplyImpl(
    std::unique_ptr<ir::Graph> graph) const {
  VLOG(3) << "Applies MKL-DNN placement strategy.";
  const auto& op_types_list =
      Get<std::unordered_set<std::string>>("mkldnn_enabled_op_types");
  for (const Node* n : graph->Nodes()) {
    if (n->IsOp()) {
      auto* op = n->Op();
      if (op->HasAttr("use_mkldnn") || op->HasProtoAttr("use_mkldnn")) {
        if (op_types_list.empty()) {
          op->SetAttr("use_mkldnn", true);
        } else if (std::find(op_types_list.begin(), op_types_list.end(),
                             n->Name()) != op_types_list.end()) {
          op->SetAttr("use_mkldnn", true);
        }
      }
    }
  }
  return graph;
}

}  // namespace ir
}  // namespace framework
}  // namespace paddle

REGISTER_PASS(mkldnn_placement_pass, paddle::framework::ir::MKLDNNPlacementPass)
    .RequirePassAttr("mkldnn_enabled_op_types");
