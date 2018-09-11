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

/*
 * This file defines the the class to partition a graph.
 */

#pragma once

#include <vector>

#include "paddle/fluid/inference/analysis/data_flow_graph.h"
#include "paddle/fluid/inference/analysis/node.h"

namespace paddle {
namespace inference {
namespace analysis {

/*
 * Detect the nodes in a sub-graph that meet some conditions. This class doesn't
 * modify the graph.
 */
class SubGraphSplitter {
 public:
  static const char *kMarkerAttrName;
  // Tell whether a node is inside a sub-graph.
  using NodeInsideSubgraphTeller = std::function<bool(const Node *)>;

  SubGraphSplitter(DataFlowGraph *graph, const NodeInsideSubgraphTeller &teller)
      : graph_(graph), node_inside_subgraph_teller_(teller) {}

  std::vector<std::vector<Node *>> operator()();

 protected:
  // Mark the nodes inside the accepted sub-graph using
  // node_inside_subgraph_teller.
  void MarkNodesInsideSubGraph();

  // Merge the marked nodes into sub-graphs and return the sub-graphs.
  std::vector<std::vector<Node *>> ExtractSubGraphs();

 private:
  DataFlowGraph *graph_;
  NodeInsideSubgraphTeller node_inside_subgraph_teller_;
};

/*
 * SubGraphFuse - Replace some nodes with the sub-graph node they are inside. To
 * some extent, the TensorRT engine is just a fusion op for a model.
 */
class SubGraphFuse {
 public:
  using NodeInsideSubgraphTeller = SubGraphSplitter::NodeInsideSubgraphTeller;

  SubGraphFuse(DataFlowGraph *graph, const NodeInsideSubgraphTeller &teller)
      : graph_(graph), node_inside_subgraph_teller_(teller) {}

  // The main method which run all the logic.
  void operator()();

 protected:
  // Remove the nodes inside sub-graphs and replace with the SubGraphNode.
  void ReplaceNodesWithSubGraphs();

 private:
  DataFlowGraph *graph_;
  NodeInsideSubgraphTeller node_inside_subgraph_teller_;
};

}  // namespace analysis
}  // namespace inference
}  // namespace paddle
