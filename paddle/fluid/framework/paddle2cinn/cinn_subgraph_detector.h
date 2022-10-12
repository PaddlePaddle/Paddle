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

#include <queue>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "paddle/fluid/framework/ir/graph.h"
#include "paddle/fluid/framework/ir/graph_traits.h"
#include "paddle/fluid/framework/ir/node.h"

namespace paddle {
namespace framework {
namespace paddle2cinn {

using Node = ir::Node;
using Graph = ir::Graph;

/*
 *
 *
 */
struct CinnSubGraph;
using CinnSubGraphPtr = std::shared_ptr<CinnSubGraph>;

struct CinnSubGraph {
  // construct function
  CinnSubGraph() {}
  // construct function
  CinnSubGraph(Node *op, bool subst) : substitute(subst) { Insert(op); }
  void Insert(Node *op);

  int depth{0};
  int max_depth{0};
  int min_depth{INT_MAX};
  bool substitute{true};
  std::vector<Node *> nodes;
  std::unordered_set<Node *> node_set;
  std::unordered_set<Node *> input_nodes;

  std::unordered_set<CinnSubGraphPtr> producers;
  std::unordered_set<CinnSubGraphPtr> consumers;
};

/*
 * Detect the nodes in a subgraph that meet some conditions. This class doesn't
 * modify the graph.
 */
class CinnSubgraphDetector {
 public:
  // Tell whether a node is inside a sub-graph.
  using NodeClassifier = std::function<bool(const Node *)>;

  CinnSubgraphDetector(Graph *graph, const NodeClassifier &classifier)
      : graph_(graph), node_classifier_(classifier) {}

  std::vector<std::vector<Node *>> operator()();

 protected:
  // Do Op Fusion
  void DoOpFusion();
  void BuildSubGraph();
  // SubGraph Fusion
  void DoSubGraphFusion();
  bool FuseSubGraph(CinnSubGraphPtr *);
  // check exist depency.
  bool IsDependency(const CinnSubGraphPtr &,
                    const CinnSubGraphPtr &,
                    const std::unordered_set<CinnSubGraphPtr> &);
  bool IsDependencySimplify(const CinnSubGraphPtr &,
                            const CinnSubGraphPtr &,
                            const std::unordered_set<CinnSubGraphPtr> &);

 private:
  Graph *graph_;
  NodeClassifier node_classifier_;

  std::vector<Node *> nodes_;
  std::vector<CinnSubGraphPtr> subgraph_list_;
  std::unordered_map<Node *, CinnSubGraphPtr> subgraph_map_;
};

}  // namespace paddle2cinn
}  // namespace framework
}  // namespace paddle
