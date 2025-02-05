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

namespace paddle::framework::ir {
class Graph;
class Node;

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
  void ApplyImpl(ir::Graph* graph) const override {
    graph->Set<int>("copy_test_pass_attr", new int);
    graph->Set<int>("copy_test_graph_attr", new int);

    int test_pass_attr = this->Get<int>("test_pass_attr");
    graph->Get<int>("copy_test_pass_attr") = test_pass_attr + 1;

    int test_graph_attr = graph->Get<int>("test_graph_attr");
    graph->Get<int>("copy_test_graph_attr") = test_graph_attr + 1;
  }
};

TEST(PassTest, TestPassAttrCheck) {
  ProgramDesc prog;
  auto pass = PassRegistry::Instance().Get("test_pass");
  std::unique_ptr<Graph> graph(new Graph(prog));
  std::string exception;
  try {
    graph.reset(pass->Apply(graph.release()));
  } catch (paddle::platform::EnforceNotMet& e) {
    exception = std::string(e.what());
  }
  ASSERT_TRUE(exception.find("Required attribute test_pass_attr for pass < "
                             "test_pass > is not set") != exception.npos);

  int val = 1;
  graph = std::make_unique<Graph>(prog);
  pass->SetNotOwned<int>("test_pass_attr", &val);

  for (std::string try_type : {"bool", "const int", "std::string"}) {
    try {
      if (try_type == "bool") {
        pass->Get<bool>("test_pass_attr");
      } else if (try_type == "const int") {
        pass->Get<const int>("test_pass_attr");
      } else if (try_type == "std::string") {
        pass->Get<std::string>("test_pass_attr");
      }
    } catch (paddle::platform::EnforceNotMet& e) {
      exception = std::string(e.what());
    }
    std::string msg =
        "Invalid type for attribute test_pass_attr, expected: " + try_type +
        ", actual: int";
    ASSERT_TRUE(exception.find(msg) != exception.npos);
  }

  try {
    graph.reset(pass->Apply(graph.release()));
  } catch (paddle::platform::EnforceNotMet& e) {
    exception = std::string(e.what());
  }
  ASSERT_TRUE(exception.find(
                  "Required attribute test_graph_attr for graph is not set") !=
              exception.npos);

  graph = std::make_unique<Graph>(prog);
  graph->Set<int>("test_graph_attr", new int);
  graph->Get<int>("test_graph_attr") = 1;
  graph.reset(pass->Apply(graph.release()));
  ASSERT_EQ(graph->Get<int>("copy_test_pass_attr"), 2);
  ASSERT_EQ(graph->Get<int>("copy_test_graph_attr"), 2);

  // Allow apply more than once.
  graph = std::make_unique<Graph>(prog);
  graph->Set<int>("test_graph_attr", new int);
  graph.reset(pass->Apply(graph.release()));

  pass = PassRegistry::Instance().Get("test_pass");
  pass->SetNotOwned<int>("test_pass_attr", &val);
  graph = std::make_unique<Graph>(prog);
  BuildCircleGraph(graph.get());
  graph->Set<int>("test_graph_attr", new int);
  graph->Get<int>("test_graph_attr") = 2;
  try {
    pass->Apply(graph.release());
  } catch (paddle::platform::EnforceNotMet& e) {
    exception = std::string(e.what());
  }
  ASSERT_TRUE(exception.find("shouldn't contain cycle") != exception.npos);

  pass = PassRegistry::Instance().Get("test_pass");
  pass->Set<int>("test_pass_attr", new int);
  try {
    pass->Set<int>("test_pass_attr", new int);
  } catch (paddle::platform::EnforceNotMet& e) {
    exception = std::string(e.what());
  }
  ASSERT_TRUE(
      exception.find("Attribute test_pass_attr already set in the pass") !=
      exception.npos);
}

TEST(PassTest, TestPassAttrCheckConvertAllBlocks) {
  // Set FLAGS_convert_all_blocks to true to make sure this test works.
  bool flag_temp = FLAGS_convert_all_blocks;
  FLAGS_convert_all_blocks = true;

  ProgramDesc prog;
  auto pass = PassRegistry::Instance().Get("test_pass");
  std::unique_ptr<Graph> graph(new Graph(prog));
  std::string exception;
  try {
    graph.reset(pass->Apply(graph.release()));
  } catch (paddle::platform::EnforceNotMet& e) {
    exception = std::string(e.what());
  }
  ASSERT_TRUE(exception.find("Required attribute test_pass_attr for pass < "
                             "test_pass > is not set") != exception.npos);

  int val = 1;
  graph = std::make_unique<Graph>(prog);
  pass->SetNotOwned<int>("test_pass_attr", &val);

  for (std::string try_type : {"bool", "const int", "std::string"}) {
    try {
      if (try_type == "bool") {
        pass->Get<bool>("test_pass_attr");
      } else if (try_type == "const int") {
        pass->Get<const int>("test_pass_attr");
      } else if (try_type == "std::string") {
        pass->Get<std::string>("test_pass_attr");
      }
    } catch (paddle::platform::EnforceNotMet& e) {
      exception = std::string(e.what());
    }
    std::string msg =
        "Invalid type for attribute test_pass_attr, expected: " + try_type +
        ", actual: int";
    ASSERT_TRUE(exception.find(msg) != exception.npos);
  }

  try {
    graph.reset(pass->Apply(graph.release()));
  } catch (paddle::platform::EnforceNotMet& e) {
    exception = std::string(e.what());
  }
  ASSERT_TRUE(exception.find(
                  "Required attribute test_graph_attr for graph is not set") !=
              exception.npos);

  graph = std::make_unique<Graph>(prog);
  graph->Set<int>("test_graph_attr", new int);
  graph->Get<int>("test_graph_attr") = 1;
  graph.reset(pass->Apply(graph.release()));
  ASSERT_EQ(graph->Get<int>("copy_test_pass_attr"), 2);
  ASSERT_EQ(graph->Get<int>("copy_test_graph_attr"), 2);

  // Allow apply more than once.
  graph = std::make_unique<Graph>(prog);
  graph->Set<int>("test_graph_attr", new int);
  graph.reset(pass->Apply(graph.release()));

  pass = PassRegistry::Instance().Get("test_pass");
  pass->SetNotOwned<int>("test_pass_attr", &val);
  graph = std::make_unique<Graph>(prog);
  BuildCircleGraph(graph.get());
  graph->Set<int>("test_graph_attr", new int);
  graph->Get<int>("test_graph_attr") = 2;
  try {
    pass->Apply(graph.release());
  } catch (paddle::platform::EnforceNotMet& e) {
    exception = std::string(e.what());
  }
  ASSERT_TRUE(exception.find("shouldn't contain cycle") != exception.npos);

  pass = PassRegistry::Instance().Get("test_pass");
  pass->Set<int>("test_pass_attr", new int);
  try {
    pass->Set<int>("test_pass_attr", new int);
  } catch (paddle::platform::EnforceNotMet& e) {
    exception = std::string(e.what());
  }
  ASSERT_TRUE(
      exception.find("Attribute test_pass_attr already set in the pass") !=
      exception.npos);

  // Recover FLAGS_convert_all_blocks.
  FLAGS_convert_all_blocks = flag_temp;
}

class TestPassWithDefault : public Pass {
 protected:
  void ApplyImpl(ir::Graph* graph) const override {
    graph->Set<int>("copy_default_attr", new int);

    int test_pass_attr = this->Get<int>("default_attr");
    graph->Get<int>("copy_default_attr") = test_pass_attr + 1;
  }
};

TEST(PassTest, TestPassDefaultAttrCheck) {
  ProgramDesc prog;
  // check if default value is set
  auto pass = PassRegistry::Instance().Get("test_pass_default_attr");
  std::unique_ptr<Graph> graph(new Graph(prog));
  ASSERT_EQ(pass->Get<int>("default_attr"), 1);
  graph.reset(pass->Apply(graph.release()));
  ASSERT_EQ(graph->Get<int>("copy_default_attr"), 2);

  // check if new value overrides default value
  pass = PassRegistry::Instance().Get("test_pass_default_attr");
  pass->Set<int>("default_attr", new int{3});
  ASSERT_EQ(pass->Get<int>("default_attr"), 3);
}

TEST(PassTest, TestPassDefaultAttrCheckConvertAllBlocks) {
  // Set FLAGS_convert_all_blocks to true to make sure this test works.
  bool flag_temp = FLAGS_convert_all_blocks;
  FLAGS_convert_all_blocks = true;

  ProgramDesc prog;
  // check if default value is set
  auto pass = PassRegistry::Instance().Get("test_pass_default_attr");
  std::unique_ptr<Graph> graph(new Graph(prog));
  ASSERT_EQ(pass->Get<int>("default_attr"), 1);
  graph.reset(pass->Apply(graph.release()));
  ASSERT_EQ(graph->Get<int>("copy_default_attr"), 2);

  // check if new value overrides default value
  pass = PassRegistry::Instance().Get("test_pass_default_attr");
  pass->Set<int>("default_attr", new int{3});
  ASSERT_EQ(pass->Get<int>("default_attr"), 3);

  // Recover FLAGS_convert_all_blocks.
  FLAGS_convert_all_blocks = flag_temp;
}

TEST(PassTest, TestPassRegistrarDeconstructor) {
  auto pass_registrary =
      new PassRegistrar<paddle::framework::ir::TestPassWithDefault>(
          "test_deconstructor");
  pass_registrary->DefaultPassAttr("deconstructor_attr", new int{1});
  pass_registrary->~PassRegistrar();
}

}  // namespace paddle::framework::ir

REGISTER_PASS(test_pass, paddle::framework::ir::TestPass)
    .RequirePassAttr("test_pass_attr")
    .RequireGraphAttr("test_graph_attr");

REGISTER_PASS(test_pass_default_attr,
              paddle::framework::ir::TestPassWithDefault)
    .DefaultPassAttr("default_attr", new int{1});
