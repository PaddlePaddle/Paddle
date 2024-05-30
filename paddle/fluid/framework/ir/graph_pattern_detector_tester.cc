// Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/framework/ir/graph_pattern_detector.h"

namespace paddle::framework::ir {

class Node;

void BuildGraph(Graph* g) {
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

TEST(PDPattern, NewNode) {
  PDPattern x;
  auto* n = x.NewNode([](Node* x) { return true; });
  ASSERT_TRUE(n);
  ASSERT_EQ(x.nodes_.size(), 1UL);
}

TEST(PDPattern, AddEdge) {
  PDPattern x;
  auto* a = x.NewNode([](Node* x) { return true; });
  auto* b = x.NewNode([](Node* x) { return true; });
  ASSERT_TRUE(a);
  ASSERT_TRUE(b);
  x.AddEdge(a, b);
  ASSERT_EQ(x.nodes_.size(), 2UL);
  ASSERT_EQ(x.edges_.size(), 1UL);
  ASSERT_EQ(x.edges_.front().first, a);
  ASSERT_EQ(x.edges_.front().second, b);

  ASSERT_EQ(x.nodes().size(), 2UL);
  ASSERT_EQ(x.edges().size(), 1UL);
  ASSERT_EQ(x.edges().front().first, a);
  ASSERT_EQ(x.edges().front().second, b);
}

TEST(GraphPatternDetecter, MarkPDNodesInGraph) {
  GraphPatternDetector x;
  // mark o2, o3, v2

  // The pattern is a graph:
  //   o2(a node named o2) -> v2(a node named v2)
  //   v2 -> o3(a node named o3)
  auto* o2 = x.pattern_.NewNode([](Node* node) {
    // The teller can be any condition, such as op type, or variable's shape.
    return node && node->Name() == "op2" && node->IsOp();
  });
  auto* o3 = x.pattern_.NewNode([](Node* node) {
    // The teller can be any condition, such as op type, or variable's shape.
    return node && node->Name() == "op3" && node->IsOp();
  });
  auto* v2 = x.pattern_.NewNode([](Node* node) {
    // The teller can be any condition, such as op type, or variable's shape.
    return node && node->Name() == "var2" && node->IsVar();
  });

  ASSERT_FALSE(o2->Tell(nullptr));
  ASSERT_FALSE(o3->Tell(nullptr));
  ASSERT_FALSE(v2->Tell(nullptr));

  x.pattern_.AddEdge(o2, v2);
  x.pattern_.AddEdge(v2, o3);

  ASSERT_EQ(x.pattern_.edges().size(), 2UL);
  ASSERT_EQ(x.pattern_.edges()[0].first, o2);
  ASSERT_EQ(x.pattern_.edges()[0].second, v2);
  ASSERT_EQ(x.pattern_.edges()[1].first, v2);
  ASSERT_EQ(x.pattern_.edges()[1].second, o3);

  ProgramDesc program;
  Graph graph(program);
  BuildGraph(&graph);

  x.MarkPDNodesInGraph(graph);

  ASSERT_EQ(x.pdnodes2nodes_.size(), 3UL);

  auto subgraphs = x.DetectPatterns();
  ASSERT_EQ(subgraphs.size(), 1UL);
}

TEST(GraphPatternDetecter, MultiSubgraph) {
  ProgramDesc program;
  Graph graph(program);
  BuildGraph(&graph);

  GraphPatternDetector x;

  // The pattern is a graph:
  //   op -> var
  auto* any_op = x.mutable_pattern()->NewNode(
      [](Node* node) {
        return node->IsOp() && (node->Name() == "op2" || node->Name() == "op3");
      },
      "OP0");
  auto* any_var = x.mutable_pattern()
                      ->NewNode([](Node* node) { return node->IsVar(); }, "VAR")
                      ->AsIntermediate();
  auto* any_op1 = x.mutable_pattern()->NewNode(
      [](Node* node) { return node->IsOp(); }, "OP1");

  x.mutable_pattern()->AddEdge(any_op, any_var);
  x.mutable_pattern()->AddEdge(any_var, any_op1);

  int count = 0;
  GraphPatternDetector::handle_t handle =
      [&](const GraphPatternDetector::subgraph_t& s, Graph* g) {
        LOG(INFO) << "Detect " << s.at(any_op)->Name() << " -> "
                  << s.at(any_var)->Name() << " -> " << s.at(any_op1)->Name();
        count++;
      };

  x(&graph, handle);

  // 1. Detect op3 -> var4 -> op5
  // 2. Detect op2 -> var2 -> op3
  // 3. Detect op2 -> var2 -> op4
  // 4. Detect op2 -> var3 -> op5
  // But 2 and 3 and 4 overlapped, so keep 2, so the final choices are 1 and 2
  ASSERT_GE(count, 1);
  ASSERT_LE(count, 2);
}

TEST(GraphPatternDetector, IntermediateCheck) {
  ProgramDesc program;
  Graph graph(program);
  BuildGraph(&graph);

  // o2->v2->o3
  // o2->v2->o4
  // check o2+o3 fuse, should fail because v2 also link to o4.
  GraphPatternDetector detector;
  auto* op2 = detector.mutable_pattern()->NewNode(
      [](Node* x) { return x && x->IsOp() && x->Name() == "op2"; }, "op2");
  auto* op3 = detector.mutable_pattern()->NewNode(
      [](Node* x) { return x && x->IsOp() && x->Name() == "op3"; }, "op3");
  auto* v2 =
      detector.mutable_pattern()
          ->NewNode(
              [](Node* x) { return x && x->IsVar() && x->Name() == "var2"; },
              "var2")
          ->AsIntermediate();
  v2->LinksFrom({op2}).LinksTo({op3});

  int count = 0;
  detector(&graph,
           [&](const GraphPatternDetector::subgraph_t& g, Graph* graph) {
             ++count;
           });
  EXPECT_EQ(count, 0);

  count = 0;
  v2->AsInput();
  detector(&graph,
           [&](const GraphPatternDetector::subgraph_t& g, Graph* graph) {
             ++count;
           });
  ASSERT_EQ(count, 1);
}

}  // namespace paddle::framework::ir
