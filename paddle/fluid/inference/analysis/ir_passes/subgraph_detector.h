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
#include "paddle/fluid/framework/ir/graph_traits.h"
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
 * SubGraphFuser - Replace some nodes with the sub-graph node they are inside.
 * To some extent, the TensorRT engine is just a fusion op for a model.
 */
class SubGraphFuser {
 public:
  using NodeInsideSubgraphTeller = SubgraphDetector::NodeInsideSubgraphTeller;

  SubGraphFuser(Graph *graph, const NodeInsideSubgraphTeller &teller,
                int min_subgraph_size)
      : graph_(graph),
        node_inside_subgraph_teller_(teller),
        min_subgraph_size_{min_subgraph_size} {}

  // The main method which run all the logic.
  void operator()();

 protected:
  // Remove the nodes inside sub-graphs and replace with the SubGraphNode.
  void ReplaceNodesWithSubGraphs();

 private:
  Graph *graph_;
  NodeInsideSubgraphTeller node_inside_subgraph_teller_;
  int min_subgraph_size_;
};

const char kUnionFindParent[] = "_sub_graph_splitter_union_find_parent_";
const char kDetectedSubgraph[] = "_detected_sub_graph_";
const char kSubgraph[] = "_subgraph_";
/*
 * ir::Node agent for subgraph detector.
 */
struct Agent {
  Agent(framework::ir::Node *x) : x_(x) {}

  bool deleted() { return GetBool(framework::ir::kNodeDeleted); }
  void set_deleted(bool x) { SetBool(framework::ir::kNodeDeleted, x); }

  bool marked() { return GetBool(kSubgraphSplitterMarkerAttrName); }
  void set_marked(bool x) { SetBool(kSubgraphSplitterMarkerAttrName, x); }

  void set_subgraph(const std::vector<framework::ir::Node *> &x) {
    if (!x_->Has(kSubgraph)) {
      x_->Set(kSubgraph, new std::vector<framework::ir::Node *>(x));
    } else {
      x_->Get<std::vector<framework::ir::Node *>>(kSubgraph) = x;
    }
  }

  std::vector<framework::ir::Node *> *subgraph() {
    if (!x_->Has(kSubgraph)) return nullptr;
    return &x_->Get<std::vector<framework::ir::Node *>>(kSubgraph);
  }

  std::vector<framework::ir::Node *> &inputs() { return x_->inputs; }
  std::vector<framework::ir::Node *> &outputs() { return x_->outputs; }

 private:
  void SetBool(const std::string &key, bool x) {
    if (x_->Has(key)) {
      x_->Get<bool>(key) = x;
    } else {
      x_->Set(key, new bool(x));
    }
  }

  bool GetBool(const std::string &key) {
    return x_->Has(key) && x_->Get<bool>(key);
  }

 private:
  framework::ir::Node *x_;
};

}  // namespace analysis
}  // namespace inference
}  // namespace paddle
