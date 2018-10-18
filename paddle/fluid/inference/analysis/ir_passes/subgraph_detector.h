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

#include "paddle/fluid/framework/ir/graph.h"
#include "paddle/fluid/framework/ir/node.h"
#include "paddle/fluid/inference/analysis/argument.h"

namespace paddle {
namespace inference {
namespace analysis {

using framework::ir::Graph;

const char kIsFunctionNode[] = "__is_function_node__";
const char kFunctionNodeSubGraph[] = "__function_node_sub_graph__";
const char kSubgraphSplitterMarkerAttrName[] =
    "_sub_graph_splitter_inside_sub_graph";

/*
 * Detect the nodes in a sub-graph that meet some conditions. This class doesn't
 * modify the graph.
 */
class SubgraphDetector {
 public:
  // Tell whether a node is inside a sub-graph.
  using NodeInsideSubgraphTeller =
      std::function<bool(const framework::ir::Node *)>;

  SubgraphDetector(Graph *graph, const NodeInsideSubgraphTeller &teller)
      : graph_(graph), node_inside_subgraph_teller_(teller) {}

  std::vector<std::vector<framework::ir::Node *>> operator()();

 protected:
  // Mark the nodes inside the accepted sub-graph using
  // node_inside_subgraph_teller.
  void MarkNodesInsideSubGraph();

  // Merge the marked nodes into sub-graphs and return the sub-graphs.
  std::vector<std::vector<framework::ir::Node *>> ExtractSubGraphs();

 private:
  Graph *graph_;
  NodeInsideSubgraphTeller node_inside_subgraph_teller_;
};

/*
 * SubGraphFuse - Replace some nodes with the sub-graph node they are inside. To
 * some extent, the TensorRT engine is just a fusion op for a model.
 */
/*
class SubGraphFuse {
 public:
  using NodeInsideSubgraphTeller = SubgraphDetector::NodeInsideSubgraphTeller;

  SubGraphFuse(Graph *graph, const NodeInsideSubgraphTeller &teller,
               Argument *argument)
      : graph_(graph),
        node_inside_subgraph_teller_(teller),
        argument_(argument) {}

  // The main method which run all the logic.
  void operator()();

 protected:
  // Remove the nodes inside sub-graphs and replace with the SubGraphNode.
  void ReplaceNodesWithSubGraphs();

 private:
  Graph *graph_;
  NodeInsideSubgraphTeller node_inside_subgraph_teller_;
  Argument *argument_;
};
 */

}  // namespace analysis
}  // namespace inference
}  // namespace paddle
