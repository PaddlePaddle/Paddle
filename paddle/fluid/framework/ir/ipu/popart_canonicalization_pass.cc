// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/framework/ir/ipu/popart_canonicalization_pass.h"

#include "paddle/fluid/framework/ir/graph_helper.h"
#include "paddle/fluid/framework/ir/pass_tester_helper.h"
#include "paddle/fluid/platform/device/ipu/popart_canonicalization/canonicalization_utils.h"

namespace paddle {
namespace framework {
namespace ir {

void PopartCanonicalizationPass::ApplyImpl(ir::Graph* graph) const {
  VLOG(10) << "enter PopartCanonicalizationPass::ApplyImpl";
  VLOG(10) << "Raw Graph: ";
  VLOG(10) << DebugString(graph);

  auto custom_ops = Get<std::unordered_set<std::string>>("custom_ops");
  std::vector<std::string> missing_ops;
  auto sorted_ops = TopologySortOperations(*graph);
  for (auto* node : sorted_ops) {
    auto* op = node->Op();
    auto op_type = op->Type();

    ir::Node* new_node = nullptr;
    platform::ipu::SymbolHandler handler = platform::ipu::GetHandler(op_type);
    if (!handler && !custom_ops.empty()) {
      if (custom_ops.count(op_type)) {
        VLOG(10) << "Found custom op: " << op_type;
        handler = platform::ipu::GetHandler("custom_op");
      }
    }

    if (handler) {
      VLOG(11) << "Raw Paddle Node:";
      VLOG(11) << node->Op()->Proto()->DebugString();
      new_node = handler(graph, node);
      if (new_node) {
        VLOG(11) << "Post Popart Node:";
        VLOG(11) << new_node->Op()->Proto()->DebugString();
        platform::ipu::ClearNode(node);
        graph->RemoveNode(node);
      }
    } else {
      missing_ops.push_back(op_type);
    }
  }

  if (!missing_ops.empty()) {
    LOG(ERROR) << "Can not find OpHandler for op_type: ";
    for (auto& op_type : missing_ops) {
      LOG(ERROR) << op_type;
    }
    PADDLE_THROW(platform::errors::Unimplemented(
        "Found unimplemented op_handler(s) for IPU"));
  }

  // post popart_canonicalization

  VLOG(10) << "Post Graph: ";
  VLOG(10) << DebugString(graph);
  VLOG(10) << "leave PopartCanonicalizationPass::ApplyImpl";
}

}  // namespace ir
}  // namespace framework
}  // namespace paddle

REGISTER_PASS(popart_canonicalization_pass,
              paddle::framework::ir::PopartCanonicalizationPass)
    .DefaultPassAttr("custom_ops", new std::unordered_set<std::string>{});
