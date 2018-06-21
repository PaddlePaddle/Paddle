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

#include "paddle/fluid/inference/analysis/dfg_graphviz_draw_pass.h"

namespace paddle {
namespace inference {
namespace analysis {

void DFG_GraphvizDrawPass::Run(DataFlowGraph *graph) {
  auto content = Draw(graph);
  std::ofstream file(GenDotPath());
  file.write(content.c_str(), content.size());
  file.close();
  LOG(INFO) << "draw dot to " << GenDotPath();
}

std::string DFG_GraphvizDrawPass::Draw(DataFlowGraph *graph) {
  Dot dot;
  // Add nodes
  for (size_t i = 0; i < graph->nodes.size(); i++) {
    const Node &node = graph->nodes.Get(i);
    if (config_.display_deleted_node || !node.deleted()) {
      dot.AddNode(node.repr(), node.dot_attrs());
    }
  }
  // Add edges
  for (size_t i = 0; i < graph->nodes.size(); i++) {
    const Node &node = graph->nodes.Get(i);
    if (!config_.display_deleted_node && node.deleted()) continue;
    for (auto &in : node.inlinks) {
      if (!config_.display_deleted_node && in->deleted()) continue;
      for (auto &in : node.inlinks) {
        dot.AddEdge(in->repr(), node.repr(), {});
      }
    }
  }
  return dot.Build();
}

}  // namespace analysis
}  // namespace inference
}  // namespace paddle
