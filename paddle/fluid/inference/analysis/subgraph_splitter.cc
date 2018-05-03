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

#include "paddle/fluid/inference/analysis/subgraph_splitter.h"

namespace paddle {
namespace inference {
namespace analysis {

SubGraphSplitter::SubGraphSplitter(
    const DataFlowGraph &graph,
    SubGraphSplitter::NodeInsideSubgraphTeller &&teller)
    : graph_(graph), node_inside_subgraph_teller_(std::move(teller)) {}

void SubGraphSplitter::MarkNodesInsideSubGraph() {
  auto trait = GraphTraits<DataFlowGraph>(graph_);
  auto nodes = trait.nodes();
  for (auto it = nodes.begin(); it != nodes.end(); it++) {
    if (node_inside_subgraph_teller_(*it)) {
      NodeAttr &attr = (*it).NewAttr<NodeAttr>(kMarkerAttrName);
      attr.is_in_subgraph = true;
    }
  }
}

std::vector<std::vector<const Node *>> SubGraphSplitter::ExtractSubGraphs() {
  std::vector<Node *> marked_nodes;
  auto trait = GraphTraits<DataFlowGraph>(graph_);
  auto nodes = trait.nodes();
  for (auto it = nodes.begin(); it != nodes.end(); it++) {
    auto &attr = it->NewAttr<NodeAttr>(kMarkerAttrName);
    if (attr.is_in_subgraph) {
      marked_nodes.push_back(it);
    }
  }
  // extract sub-graphs in the marked node set, use Union Find algorithm.

  return std::vector<std::vector<const Node *>>();
}

}  // namespace analysis
}  // namespace inference
}  // namespace paddle
