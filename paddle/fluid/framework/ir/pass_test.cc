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

#include "paddle/fluid/framework/ir/pass.h"
#include <string>
#include "gtest/gtest.h"
#include "paddle/fluid/framework/ir/graph.h"

namespace paddle {
namespace framework {
namespace ir {
void BuildCircleGraph(Graph* g) {
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

class TestPass : public Pass {
 protected:
  ir::Graph* ApplyImpl(ir::Graph* graph) const {
    graph->Set<int>("copy_test_pass_attr", new int);
    graph->Set<int>("copy_test_graph_attr", new int);

    int test_pass_attr = this->Get<int>("test_pass_attr");
    graph->Get<int>("copy_test_pass_attr") = test_pass_attr + 1;

    int test_graph_attr = graph->Get<int>("test_graph_attr");
    graph->Get<int>("copy_test_graph_attr") = test_graph_attr + 1;
    return graph;
  }
};

TEST(PassTest, TestPassAttrCheck) {
  ProgramDesc prog;
  auto pass = PassRegistry::Instance().Get("test_pass");
  std::unique_ptr<Graph> graph(new Graph(prog));
  std::string exception;
  try {
    graph = pass->Apply(std::move(graph));
  } catch (paddle::platform::EnforceNotMet e) {
    exception = std::string(e.what());
  }
  ASSERT_TRUE(exception.find("test_pass_attr not set") != exception.npos);

  int val = 1;
  graph.reset(new Graph(prog));
  pass->SetNotOwned<int>("test_pass_attr", &val);

  try {
    graph = pass->Apply(std::move(graph));
  } catch (paddle::platform::EnforceNotMet e) {
    exception = std::string(e.what());
  }
  ASSERT_TRUE(exception.find("test_graph_attr not set") != exception.npos);

  graph.reset(new Graph(prog));
  graph->Set<int>("test_graph_attr", new int);
  graph->Get<int>("test_graph_attr") = 1;
  graph = pass->Apply(std::move(graph));
  ASSERT_EQ(graph->Get<int>("copy_test_pass_attr"), 2);
  ASSERT_EQ(graph->Get<int>("copy_test_graph_attr"), 2);

  // Allow apply more than once.
  graph.reset(new Graph(prog));
  graph->Set<int>("test_graph_attr", new int);
  graph = pass->Apply(std::move(graph));

  pass = PassRegistry::Instance().Get("test_pass");
  pass->SetNotOwned<int>("test_pass_attr", &val);
  graph.reset(new Graph(prog));
  BuildCircleGraph(graph.get());
  graph->Set<int>("test_graph_attr", new int);
  graph->Get<int>("test_graph_attr") = 2;
  try {
    auto tmp = pass->Apply(std::move(graph));
  } catch (paddle::platform::EnforceNotMet e) {
    exception = std::string(e.what());
  }
  ASSERT_TRUE(exception.find("shouldn't has cycle") != exception.npos);
}

}  // namespace ir
}  // namespace framework
}  // namespace paddle

REGISTER_PASS(test_pass, paddle::framework::ir::TestPass)
    .RequirePassAttr("test_pass_attr")
    .RequireGraphAttr("test_graph_attr");
