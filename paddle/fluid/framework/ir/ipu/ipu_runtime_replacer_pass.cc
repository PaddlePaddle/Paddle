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

#include "paddle/fluid/framework/ir/ipu/ipu_runtime_replacer_pass.h"

#include "paddle/fluid/framework/ir/graph_pattern_detector.h"
#include "paddle/fluid/framework/ir/pass_tester_helper.h"

namespace paddle {
namespace framework {
namespace ir {

void IpuRuntimeReplacerPass::ApplyImpl(ir::Graph* graph) const {
  VLOG(10) << "enter IpuRuntimeReplacerPass::ApplyImpl";
  VLOG(10) << "Raw Graph: ";
  VLOG(10) << DebugString(graph);

  std::vector<std::string> feed_list;
  feed_list = Get<std::vector<std::string>>("feed_list");

  std::vector<std::string> fetch_list;
  fetch_list = Get<std::vector<std::string>>("fetch_list");

  framework::OpDesc ipu_rt_op_desc;
  ipu_rt_op_desc.SetType("ipu_runtime");
  ipu_rt_op_desc.SetInput("FeedList", feed_list);
  ipu_rt_op_desc.SetOutput("FetchList", fetch_list);
  ipu_rt_op_desc.Flush();

  // Create a new node for the ipu_runtime_op.
  auto* ipu_rt_node = graph->CreateOpNode(&ipu_rt_op_desc);

  for (auto* node : graph->Nodes()) {
    if (node->IsVar()) {
      for (auto feed : feed_list) {
        if (node->Name() == feed) {
          IR_NODE_LINK_TO(node, ipu_rt_node);
        }
      }
      for (auto fetch : fetch_list) {
        if (node->Name() == fetch) {
          IR_NODE_LINK_TO(ipu_rt_node, node);
        }
      }
    }
  }

  // Remove unneeded nodes.
  std::unordered_set<const Node*> marked_nodes;
  for (auto* node : graph->Nodes()) {
    if (node->IsOp()) {
      auto* op_desc = node->Op();
      if (op_desc->Type() != "ipu_runtime") {
        marked_nodes.insert(node);
      }
    }
  }

  GraphSafeRemoveNodes(graph, marked_nodes);

  VLOG(10) << "Post Graph: ";
  VLOG(10) << DebugString(graph);
  VLOG(10) << "leave IpuRuntimeReplacerPass::ApplyImpl";
}

}  // namespace ir
}  // namespace framework
}  // namespace paddle

REGISTER_PASS(ipu_runtime_replacer_pass,
              paddle::framework::ir::IpuRuntimeReplacerPass)
    .RequirePassAttr("feed_list")
    .RequirePassAttr("fetch_list");
