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

#include "paddle/fluid/lite/core/mir/pattern_matcher.h"

#include <gtest/gtest.h>

namespace paddle {
namespace lite {
namespace mir {

void BuildGraph(SSAGraph* g) {
  g->mutable_nodes().emplace_back();
  Node& o1 = g->mutable_nodes().back();
  o1.AsStmt().op_type = "op1";
  g->mutable_nodes().emplace_back();
  Node& o2 = g->mutable_nodes().back();
  o2.AsStmt().op_type = "op2";
  g->mutable_nodes().emplace_back();
  Node& o3 = g->mutable_nodes().back();
  o3.AsStmt().op_type = "op3";
  g->mutable_nodes().emplace_back();
  Node& o4 = g->mutable_nodes().back();
  o4.AsStmt().op_type = "op4";
  g->mutable_nodes().emplace_back();
  Node& o5 = g->mutable_nodes().back();
  o5.AsStmt().op_type = "op5";
  g->mutable_nodes().emplace_back();
  Node& v1 = g->mutable_nodes().back();
  v1.AsArg("var1");
  g->mutable_nodes().emplace_back();
  Node& v2 = g->mutable_nodes().back();
  v2.AsArg("var2");
  g->mutable_nodes().emplace_back();
  Node& v3 = g->mutable_nodes().back();
  v3.AsArg("var3");
  g->mutable_nodes().emplace_back();
  Node& v4 = g->mutable_nodes().back();
  v4.AsArg("var4");

  // o1->v1->o2
  o1.outlinks.push_back(&v1);
  o2.inlinks.push_back(&v1);
  v1.inlinks.push_back(&o1);
  v1.outlinks.push_back(&o2);
  // o2->v2->o3
  // o2->v2->o4
  o2.outlinks.push_back(&v2);
  o3.inlinks.push_back(&v2);
  o4.inlinks.push_back(&v2);
  v2.inlinks.push_back(&o2);
  v2.outlinks.push_back(&o3);
  v2.outlinks.push_back(&o4);
  // o2->v3->o5
  o2.outlinks.push_back(&v3);
  o5.inlinks.push_back(&v3);
  v3.inlinks.push_back(&o2);
  v3.outlinks.push_back(&o5);
  // o3-v4->o5
  o3.outlinks.push_back(&v4);
  o5.inlinks.push_back(&v4);
  v4.inlinks.push_back(&o3);
  v4.outlinks.push_back(&o5);
}

TEST(PMPattern, NewNode) {
  PMPattern x;
  auto* n = x.NewNode([](const Node* x) { return true; });
  ASSERT_TRUE(n);
  ASSERT_EQ(x.nodes_.size(), 1UL);
}

TEST(PMPattern, AddEdge) {
  PMPattern x;
  auto* a = x.NewNode([](const Node* x) { return true; });
  auto* b = x.NewNode([](const Node* x) { return true; });
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

TEST(PatternMatcher, MarkPMNodesInGraph) {
  PatternMatcher x;
  // mark o2, o3, v2

  // The pattern is a graph:
  //   o2(a node named o2) -> v2(a node named v2)
  //   v2 -> o3(a node named o3)
  auto* o2 = x.pattern_.NewNode([](const Node* node) {
    // The teller can be any condition, such as op type, or variable's shape.
    return node && node->IsStmt() && node->stmt()->op_type == "op2";
  });
  auto* o3 = x.pattern_.NewNode([](const Node* node) {
    // The teller can be any condition, such as op type, or variable's shape.
    return node && node->IsStmt() && node->stmt()->op_type == "op3";
  });
  auto* v2 = x.pattern_.NewNode([](const Node* node) {
    // The teller can be any condition, such as op type, or variable's shape.
    return node && node->IsArg() && node->arg()->name == "var2";
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

  SSAGraph graph;
  BuildGraph(&graph);

  x.MarkPMNodesInGraph(&graph);

  ASSERT_EQ(x.pmnodes2nodes_.size(), 3UL);

  auto subgraphs = x.DetectPatterns();
  ASSERT_EQ(subgraphs.size(), 1UL);
}

TEST(PatternMatcher, MultiSubgraph) {
  SSAGraph graph;
  BuildGraph(&graph);

  PatternMatcher x;

  // The pattern is a graph:
  //   op -> var
  auto* any_op = x.mutable_pattern()->NewNode(
      [](const Node* node) {
        return node->IsStmt() && (node->stmt()->op_type == "op2" ||
                                  node->stmt()->op_type == "op3");
      },
      "OP0");
  auto* any_var =
      x.mutable_pattern()
          ->NewNode([](const Node* node) { return node->IsArg(); }, "VAR")
          ->AsIntermediate();
  auto* any_op1 = x.mutable_pattern()->NewNode(
      [](const Node* node) { return node->IsStmt(); }, "OP1");

  x.mutable_pattern()->AddEdge(any_op, any_var);
  x.mutable_pattern()->AddEdge(any_var, any_op1);

  int count = 0;
  PatternMatcher::handle_t handle = [&](const PatternMatcher::subgraph_t& s,
                                        SSAGraph* g) {
    LOG(INFO) << "Detect " << s.at(any_op)->stmt()->op_type << " -> "
              << s.at(any_var)->arg()->name << " -> "
              << s.at(any_op1)->stmt()->op_type;
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

TEST(PatternMatcher, IntermediateCheck) {
  SSAGraph graph;
  BuildGraph(&graph);

  // o2->v2->o3
  // o2->v2->o4
  // check o2+o3 fuse, should fail because v2 also link to o4.
  PatternMatcher matcher;
  auto* op2 = matcher.mutable_pattern()->NewNode(
      [](const Node* x) {
        return x && x->IsStmt() && x->stmt()->op_type == "op2";
      },
      "op2");
  auto* op3 = matcher.mutable_pattern()->NewNode(
      [](const Node* x) {
        return x && x->IsStmt() && x->stmt()->op_type == "op3";
      },
      "op3");
  auto* v2 = matcher.mutable_pattern()
                 ->NewNode(
                     [](const Node* x) {
                       return x && x->IsArg() && x->arg()->name == "var2";
                     },
                     "var2")
                 ->AsIntermediate();
  v2->LinksFrom({op2}).LinksTo({op3});

  int count = 0;
  matcher(&graph, [&](const PatternMatcher::subgraph_t& g, SSAGraph* graph) {
    ++count;
  });
  EXPECT_EQ(count, 0);

  count = 0;
  v2->AsInput();
  matcher(&graph, [&](const PatternMatcher::subgraph_t& g, SSAGraph* graph) {
    ++count;
  });
  ASSERT_EQ(count, 1);
}

}  // namespace mir
}  // namespace lite
}  // namespace paddle
