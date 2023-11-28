// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/framework/ir/gcu/gcu_runtime_replacer_pass.h"
#include "paddle/fluid/framework/ir/graph_pattern_detector.h"
#include "paddle/fluid/framework/ir/pass_tester_helper.h"
#include "paddle/fluid/platform/device/gcu/utils/types.h"

namespace paddle {
namespace framework {
namespace ir {
using paddle::platform::gcu::kGcuProgramKey;

void GcuRuntimeReplacerPass::ApplyImpl(ir::Graph* graph) const {
  VLOG(10) << "enter GcuRuntimeReplacerPass::ApplyImpl";
  VLOG(10) << "Raw Graph: ";
  VLOG(10) << DebugString(graph);

  std::vector<std::string> feed_list;
  feed_list = Get<std::vector<std::string>>("feed_list");

  std::vector<std::string> fetch_list;
  fetch_list = Get<std::vector<std::string>>("fetch_list");

  std::string program_key = Get<std::string>("program_key");

  framework::OpDesc gcu_rt_op_desc;
  gcu_rt_op_desc.SetType("gcu_runtime");
  gcu_rt_op_desc.SetInput("FeedList", feed_list);
  gcu_rt_op_desc.SetOutput("FetchList", fetch_list);
  // set program_key
  gcu_rt_op_desc.SetAttr(kGcuProgramKey, program_key);
  gcu_rt_op_desc.Flush();

  // Create a new node for the gcu_runtime_op.
  auto* gcu_rt_node = graph->CreateOpNode(&gcu_rt_op_desc);

  for (auto* node : graph->Nodes()) {
    if (node->IsVar()) {
      for (auto feed : feed_list) {
        if (node->Name() == feed) {
          IR_NODE_LINK_TO(node, gcu_rt_node);
        }
      }
      // no need to connect to var
      // for (auto fetch : fetch_list) {
      // if (node->Name() == fetch) {
      // IR_NODE_LINK_TO(gcu_rt_node, node);
      // }
      // }
    }
  }

  // Remove unneeded nodes.
  std::unordered_set<const Node*> marked_nodes;
  for (auto* node : graph->Nodes()) {
    if (node->IsOp()) {
      auto* op_desc = node->Op();
      if (op_desc->Type() != "gcu_runtime") {
        marked_nodes.insert(node);
      }
    }
  }

  GraphSafeRemoveNodes(graph, marked_nodes);

  VLOG(10) << "Post Graph: ";
  VLOG(10) << DebugString(graph);
  VLOG(10) << "leave GcuRuntimeReplacerPass::ApplyImpl";
}

}  // namespace ir
}  // namespace framework
}  // namespace paddle

REGISTER_PASS(gcu_runtime_replacer_pass,
              paddle::framework::ir::GcuRuntimeReplacerPass)
    .RequirePassAttr("feed_list")
    .RequirePassAttr("fetch_list");
