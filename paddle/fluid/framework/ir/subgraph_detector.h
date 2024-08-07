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

#pragma once

#include <string>
#include <vector>

#include "paddle/fluid/framework/ir/graph.h"
#include "paddle/fluid/framework/ir/graph_traits.h"
#include "paddle/fluid/framework/ir/node.h"

namespace paddle {
namespace framework {
namespace ir {

class Graph;
class Node;

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
  using NodeInsideSubgraphTeller = std::function<bool(const Node *)>;

  SubgraphDetector(Graph *graph, const NodeInsideSubgraphTeller &teller)
      : graph_(graph), node_inside_subgraph_teller_(teller) {}

  std::vector<std::vector<Node *>> operator()();

 protected:
  // Mark the nodes inside the accepted sub-graph using
  // node_inside_subgraph_teller.
  void MarkNodesInsideSubGraph();

  // Merge the marked nodes into sub-graphs and return the sub-graphs.
  std::vector<std::vector<Node *>> ExtractSubGraphs();

 private:
  Graph *graph_;
  NodeInsideSubgraphTeller node_inside_subgraph_teller_;
};

/*
 * SubGraphFuser - Replace some nodes with the sub-graph node they are inside.
 * To some extent, the TensorRT engine is just a fusion op for a model.
 */
class SubGraphFuser {
 public:
  using NodeInsideSubgraphTeller = SubgraphDetector::NodeInsideSubgraphTeller;

  SubGraphFuser(Graph *graph,
                const NodeInsideSubgraphTeller &teller,
                int min_subgraph_size,
                const std::vector<std::string> &trt_exclude_var_names = {},
                std::string name = "tensorrt_engine")
      : graph_(graph),
        node_inside_subgraph_teller_(teller),
        min_subgraph_size_{min_subgraph_size},
        trt_exclude_var_names_(trt_exclude_var_names),
        name_{name} {}

  // The main method which run all the logic.
  void operator()();

 protected:
  // Remove the nodes inside sub-graphs and replace with the SubGraphNode.
  void ReplaceNodesWithSubGraphs();

 private:
  Graph *graph_;
  NodeInsideSubgraphTeller node_inside_subgraph_teller_;
  int min_subgraph_size_;
  std::vector<std::string> trt_exclude_var_names_;
  const std::string name_;
};

struct NodeWrapper {
  bool deleted{false};
  bool marked{false};
  int union_find_parent{-1};
  std::vector<Node *> subgraph;
};

/*
 * ir::Node agent for subgraph detector.
 */
struct Agent {
  explicit Agent(Node *x) : x_(x) {}

  NodeWrapper &wrapper() {
    if (!x_->IsWrappedBy<NodeWrapper>()) {
      x_->WrappedBy<NodeWrapper>(new NodeWrapper);
    }
    return x_->template Wrapper<NodeWrapper>();
  }

  bool deleted() { return wrapper().deleted; }
  void set_deleted(bool x) { wrapper().deleted = x; }

  bool marked() { return wrapper().marked; }
  void set_marked(bool x) { wrapper().marked = x; }

  void set_subgraph(const std::vector<framework::ir::Node *> &x) {
    wrapper().subgraph = x;
  }

  int union_find_parent() { return wrapper().union_find_parent; }
  void set_union_find_parent(int v) { wrapper().union_find_parent = v; }

  std::vector<Node *> *subgraph() { return &wrapper().subgraph; }
  std::vector<Node *> &inputs() { return x_->inputs; }
  std::vector<Node *> &outputs() { return x_->outputs; }

 private:
  Node *x_;
};

// The nodes those have no input will be treated as start points.
static std::vector<Node *> ExtractStartPoints(const Graph &g) {
  std::vector<Node *> result;
  for (auto *node : g.Nodes()) {
    if (node->inputs.empty()) {
      result.push_back(node);
    }
  }
  return result;
}

static iterator_range<NodesTSIterator> TopologicalSort(const Graph &g) {
  auto start_points = ExtractStartPoints(g);
  PADDLE_ENFORCE_GT(
      start_points.size(),
      0U,
      common::errors::InvalidArgument(
          "Expected the number of graph's start points >= 1. Expected %d.",
          start_points.size()));
  NodesTSIterator x(start_points);
  return iterator_range<NodesTSIterator>(NodesTSIterator(start_points),
                                         NodesTSIterator());
}

}  // namespace ir
}  // namespace framework
}  // namespace paddle
