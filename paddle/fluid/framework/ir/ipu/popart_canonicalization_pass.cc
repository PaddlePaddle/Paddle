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

#include "paddle/fluid/framework/ipu/popart_canonicalization/canonicalization_utils.h"
#include "paddle/fluid/framework/ipu/popart_canonicalization/post_canonicalization.h"
#include "paddle/fluid/framework/ir/pass_tester_helper.h"

namespace paddle {
namespace framework {
namespace ir {

void PopartCanonicalizationPass::ApplyImpl(ir::Graph* graph) const {
  // register op_handlers
  // call op_handler, get inputs and attrs
  // create new node in paddle, map attr to popart, mark raw node to remove
  // remove paddle raw node

  VLOG(10) << "enter PopartCanonicalizationPass::ApplyImpl";
  VLOG(10) << "Raw Graph: ";
  VLOG(10) << DebugString(graph);

  // // graph_viz_pass
  // auto graph_viz_pass = PassRegistry::Instance().Get("graph_viz_pass");
  // graph_viz_pass->Set("graph_viz_path",
  //                     new std::string("/home/Paddle/demos/before_pass.dot"));
  // graph_viz_pass->Apply(graph);

  auto nodes = graph->Nodes();
  for (auto* node : nodes) {
    if (!node->IsOp()) {
      continue;
    }
    auto* op = node->Op();
    auto op_type = op->Type();

    ir::Node* new_node = nullptr;
    ipu::SymbolHandler handler = ipu::GetHandler(op_type);
    if (handler) {
      VLOG(11) << "Raw Paddle Node:";
      VLOG(11) << node->Op()->Proto()->DebugString();
      new_node = handler(graph, node);
      VLOG(11) << "Post Popart Node:";
      VLOG(11) << new_node->Op()->Proto()->DebugString();
      ipu::ClearNode(node);
      graph->RemoveNode(node);
    } else {
      LOG(ERROR) << "Can not find OpHandler for op_type: " << op_type;
    }
  }

  // post popart_canonicalization

  // // graph_viz_pass
  // graph_viz_pass->Erase("graph_viz_path");
  // graph_viz_pass->Set("graph_viz_path",
  //                     new std::string("/home/Paddle/demos/after_pass.dot"));
  // graph_viz_pass->Apply(graph);

  VLOG(10) << "Post Graph: ";
  VLOG(10) << DebugString(graph);
  VLOG(10) << "leave PopartCanonicalizationPass::ApplyImpl";
}

}  // namespace ir
}  // namespace framework
}  // namespace paddle

REGISTER_PASS(popart_canonicalization_pass,
              paddle::framework::ir::PopartCanonicalizationPass);

// USE_PASS(graph_viz_pass);
