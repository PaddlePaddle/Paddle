// Copyright (c) 2021 CINN Authors. All Rights Reserved.
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

#include <gtest/gtest.h>

#include "paddle/cinn/common/common.h"

namespace cinn {
namespace common {

struct GraphNodeWithName : public GraphNode {
  explicit GraphNodeWithName(std::string name) : name(name) {}

  std::string id() const override { return name; }

  std::string name;
};

// A simple graph.
std::unique_ptr<Graph> CreateGraph0() {
  std::unique_ptr<Graph> graph(new Graph);

  auto* A = make_shared<GraphNodeWithName>("A");
  auto* B = make_shared<GraphNodeWithName>("B");
  auto* C = make_shared<GraphNodeWithName>("C");
  auto* D = make_shared<GraphNodeWithName>("D");
  auto* E = make_shared<GraphNodeWithName>("E");

  graph->RegisterNode("A", A);
  graph->RegisterNode("B", B);
  graph->RegisterNode("C", C);
  graph->RegisterNode("D", D);
  graph->RegisterNode("E", E);

  A->LinkTo(B);
  A->LinkTo(C);

  B->LinkTo(D);
  C->LinkTo(D);
  C->LinkTo(E);

  return graph;
}

std::unique_ptr<Graph> CreateGraph1() {
  std::unique_ptr<Graph> graph(new Graph);

  auto* A = make_shared<GraphNodeWithName>("A");
  auto* B = make_shared<GraphNodeWithName>("B");

  graph->RegisterNode("A", A);
  graph->RegisterNode("B", B);

  B->LinkTo(A);

  return graph;
}

TEST(Graph, Visualize) {
  auto graph = CreateGraph0();
  LOG(INFO) << "graph:\n" << graph->Visualize();
}

TEST(Graph, simple) {
  auto graph = CreateGraph1();
  Graph::node_order_t node_order;
  Graph::edge_order_t edge_order;
  std::tie(node_order, edge_order) = graph->topological_order();

  LOG(INFO) << "graph1 " << graph->Visualize();

  std::vector<GraphNode*> node_order_target(
      {graph->RetrieveNode("B"), graph->RetrieveNode("A")});

  ASSERT_EQ(node_order.size(), node_order_target.size());
  for (int i = 0; i < node_order.size(); i++) {
    EXPECT_EQ(node_order[i]->id(), node_order_target[i]->id());
  }
}

}  // namespace common
}  // namespace cinn
