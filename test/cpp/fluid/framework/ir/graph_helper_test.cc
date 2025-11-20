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

#include "paddle/fluid/framework/ir/graph_helper.h"

#include "gtest/gtest.h"
#include "paddle/fluid/framework/ir/graph.h"
#include "paddle/fluid/framework/program_desc.h"

namespace paddle::framework::ir {

void BuildCircleGraph(Graph* g) {
  ir::Node* o1 = g->CreateEmptyNode("op1", Node::Type::kOperation);
  ir::Node* v1 = g->CreateEmptyNode("var1", Node::Type::kVariable);

  o1->outputs.push_back(v1);
  o1->inputs.push_back(v1);
  v1->inputs.push_back(o1);
  v1->outputs.push_back(o1);
}

void BuildCircleGraph2(Graph* g) {
  ir::Node* o1 = g->CreateEmptyNode("op1", Node::Type::kOperation);
  ir::Node* o2 = g->CreateEmptyNode("op2", Node::Type::kOperation);
  ir::Node* v1 = g->CreateEmptyNode("var1", Node::Type::kVariable);
  ir::Node* v2 = g->CreateEmptyNode("var2", Node::Type::kVariable);

  o1->outputs.push_back(v1);
  o2->inputs.push_back(v1);
  v1->inputs.push_back(o1);
  v1->outputs.push_back(o2);

  o2->outputs.push_back(v2);
  o1->inputs.push_back(v2);
  v2->inputs.push_back(o2);
  v2->outputs.push_back(o1);
}

void BuildNoCircleGraph(Graph* g) {
  ir::Node* o1 = g->CreateEmptyNode("op1", Node::Type::kOperation);
  ir::Node* o2 = g->CreateEmptyNode("op2", Node::Type::kOperation);
  ir::Node* o3 = g->CreateEmptyNode("op3", Node::Type::kOperation);
  ir::Node* o4 = g->CreateEmptyNode("op4", Node::Type::kOperation);
  ir::Node* o5 = g->CreateEmptyNode("op5", Node::Type::kOperation);
  ir::Node* v1 = g->CreateEmptyNode("var1", Node::Type::kVariable);
  ir::Node* v2 = g->CreateEmptyNode("var2", Node::Type::kVariable);
  ir::Node* v3 = g->CreateEmptyNode("var3", Node::Type::kVariable);
  ir::Node* v4 = g->CreateEmptyNode("var4", Node::Type::kVariable);

  // o1->v1->o2
  o1->outputs.push_back(v1);
  o2->inputs.push_back(v1);
  v1->inputs.push_back(o1);
  v1->outputs.push_back(o2);
  // o2->v2->o3
  // o2->v2->o4
  o2->outputs.push_back(v2);
  o3->inputs.push_back(v2);
  o4->inputs.push_back(v2);
  v2->inputs.push_back(o2);
  v2->outputs.push_back(o3);
  v2->outputs.push_back(o4);
  // o2->v3->o5
  o2->outputs.push_back(v3);
  o5->inputs.push_back(v3);
  v3->inputs.push_back(o2);
  v3->outputs.push_back(o5);
  // o3-v4->o5
  o3->outputs.push_back(v4);
  o5->inputs.push_back(v4);
  v4->inputs.push_back(o3);
  v4->outputs.push_back(o5);
}

TEST(GraphHelperTest, Basic) {
  ProgramDesc prog;

  Graph g(prog);
  BuildCircleGraph(&g);
  ASSERT_TRUE(HasCircle(g));

  Graph g2(prog);
  BuildCircleGraph2(&g2);
  ASSERT_TRUE(HasCircle(g2));

  auto adj_list = BuildOperationAdjList(g2);
  for (auto& adj : adj_list) {
    auto& adj_set = adj.second;
    if (adj.first->Name() == "op1") {
      ASSERT_EQ((*adj_set.begin())->Name(), "op2");
    } else if (adj.first->Name() == "op2") {
      ASSERT_EQ((*adj_set.begin())->Name(), "op1");
    } else {
      ASSERT_TRUE(false);
    }
  }

  Graph g3(prog);
  BuildNoCircleGraph(&g3);
  ASSERT_FALSE(HasCircle(g3));
  auto sorted = TopologySortOperations(g3);
  std::map<std::string, size_t> node_map;
  for (size_t i = 0; i < sorted.size(); ++i) {
    node_map[sorted[i]->Name()] = i;
  }
  ASSERT_EQ(node_map.at("op1"), 0UL);
  ASSERT_EQ(node_map.at("op2"), 1UL);
  ASSERT_TRUE(node_map.at("op3") < node_map.at("op5"));
}

void BuildZeroGraph(Graph* g) {}

void BuildOneGraph(Graph* g) {
  ir::Node* o1 = g->CreateEmptyNode("op1", Node::Type::kOperation);
  ir::Node* o2 = g->CreateEmptyNode("op2", Node::Type::kOperation);
  ir::Node* o3 = g->CreateEmptyNode("op3", Node::Type::kOperation);
  ir::Node* o4 = g->CreateEmptyNode("op4", Node::Type::kOperation);
  ir::Node* o5 = g->CreateEmptyNode("op5", Node::Type::kOperation);
  ir::Node* v1 = g->CreateEmptyNode("var1", Node::Type::kVariable);
  ir::Node* v2 = g->CreateEmptyNode("var2", Node::Type::kVariable);
  ir::Node* v3 = g->CreateEmptyNode("var3", Node::Type::kVariable);
  ir::Node* v4 = g->CreateEmptyNode("var4", Node::Type::kVariable);

  // o1->v1->o2
  o1->outputs.push_back(v1);
  o2->inputs.push_back(v1);
  v1->inputs.push_back(o1);
  v1->outputs.push_back(o2);
  // o2->v2->o3
  // o2->v2->o4
  o2->outputs.push_back(v2);
  o3->inputs.push_back(v2);
  o4->inputs.push_back(v2);
  v2->inputs.push_back(o2);
  v2->outputs.push_back(o3);
  v2->outputs.push_back(o4);
  // o2->v3->o5
  o2->outputs.push_back(v3);
  o5->inputs.push_back(v3);
  v3->inputs.push_back(o2);
  v3->outputs.push_back(o5);
  // o3-v4->o5
  o3->outputs.push_back(v4);
  o5->inputs.push_back(v4);
  v4->inputs.push_back(o3);
  v4->outputs.push_back(o5);
}

void BuildTwoGraphs(Graph* g) {
  ir::Node* o1 = g->CreateEmptyNode("op1", Node::Type::kOperation);
  ir::Node* o2 = g->CreateEmptyNode("op2", Node::Type::kOperation);
  ir::Node* o3 = g->CreateEmptyNode("op3", Node::Type::kOperation);
  ir::Node* o4 = g->CreateEmptyNode("op4", Node::Type::kOperation);
  ir::Node* o5 = g->CreateEmptyNode("op5", Node::Type::kOperation);
  ir::Node* v1 = g->CreateEmptyNode("var1", Node::Type::kVariable);
  ir::Node* v2 = g->CreateEmptyNode("var2", Node::Type::kVariable);
  ir::Node* v3 = g->CreateEmptyNode("var3", Node::Type::kVariable);
  ir::Node* v4 = g->CreateEmptyNode("var4", Node::Type::kVariable);

  // o1->v1->o2
  o1->outputs.push_back(v1);
  o2->inputs.push_back(v1);
  v1->inputs.push_back(o1);
  v1->outputs.push_back(o2);
  // o2->v2->o3
  // o2->v2->o4
  o2->outputs.push_back(v2);
  o3->inputs.push_back(v2);
  o4->inputs.push_back(v2);
  v2->inputs.push_back(o2);
  v2->outputs.push_back(o3);
  v2->outputs.push_back(o4);
  // o2->v3->o5
  //  o2->outputs.push_back(v3);
  o5->inputs.push_back(v3);
  //  v3->inputs.push_back(o2);
  v3->outputs.push_back(o5);
  // o3-v4->o5
  o3->outputs.push_back(v4);
  //  o5->inputs.push_back(v4);
  v4->inputs.push_back(o3);
  //  v4->outputs.push_back(o5);
}

TEST(GraphHelperTest, Circles) {
  ProgramDesc prog;

  Graph g(prog);
  BuildCircleGraph(&g);

  std::vector<std::vector<ir::Node*>> circles;
  ASSERT_TRUE(FindCircleSubGraph(g, &circles));
  ASSERT_EQ(circles.size(), 1UL);
}

TEST(GraphHelperTest, GraphNum) {
  ProgramDesc prog;

  Graph g(prog);
  BuildZeroGraph(&g);
  ASSERT_EQ(GraphNum(g), 0UL);

  Graph g2(prog);
  BuildOneGraph(&g2);
  ASSERT_EQ(GraphNum(g2), 1UL);

  Graph g3(prog);
  BuildTwoGraphs(&g3);
  ASSERT_EQ(GraphNum(g3), 2UL);
}

}  // namespace paddle::framework::ir
