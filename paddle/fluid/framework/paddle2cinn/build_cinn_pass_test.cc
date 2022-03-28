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

#include "paddle/fluid/framework/paddle2cinn/build_cinn_pass.h"

#include <algorithm>
#include <memory>
#include <string>

#include "gtest/gtest.h"

#include "paddle/fluid/framework/details/build_strategy.h"
#include "paddle/fluid/framework/ir/graph.h"
#include "paddle/fluid/framework/ir/node.h"
#include "paddle/fluid/framework/op_desc.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/operator.h"
#include "paddle/fluid/framework/paddle2cinn/cinn_compiler.h"
#include "paddle/fluid/framework/program_desc.h"
#include "paddle/fluid/framework/var_desc.h"
#include "paddle/fluid/operators/cinn/cinn_launch_op.h"

namespace paddle {
namespace framework {
namespace paddle2cinn {

using framework::ir::Graph;
using framework::ir::Node;

inline bool CheckNodeExisted(const std::unordered_set<Node*>& nodes,
                             const std::string& op_name) {
  return std::find_if(nodes.begin(), nodes.end(), [&op_name](const Node* node) {
           return node->Name() == op_name;
         }) != nodes.end();
}

inline int CountNode(const std::unordered_set<Node*>& nodes,
                     const std::string& op_name) {
  return std::count_if(
      nodes.begin(), nodes.end(),
      [&op_name](const Node* node) { return node->Name() == op_name; });
}

inline Node* GetNode(const std::unordered_set<Node*>& nodes,
                     const std::string& op_name) {
  return *std::find_if(nodes.begin(), nodes.end(),
                       [&op_name](const Node* node) {
                         return node->Name().find(op_name) != std::string::npos;
                       });
}

inline bool CheckGraphIndependence(const std::unordered_set<Node*>& nodes) {
  auto check_node_ok = [&nodes](Node* n1, Node* n2) -> bool {
    if (n1->IsOp() && !n2->IsVar()) {
      return false;
    }
    if (n1->IsVar() && !n2->IsOp()) {
      return false;
    }
    if (nodes.count(n2) == 0) {
      return false;
    }
    return true;
  };

  for (auto node : nodes) {
    for (auto in : node->inputs) {
      if (!check_node_ok(node, in)) {
        return false;
      }
    }
    for (auto out : node->outputs) {
      if (!check_node_ok(node, out)) {
        return false;
      }
    }
  }
  return true;
}

// Get compilation_key values
std::vector<std::string> GetCompilationKeys(const Graph& graph) {
  std::vector<std::string> compilation_keys;
  for (auto& node : graph.Nodes()) {
    if (node->IsOp() && node->Name() == kCinnLaunchOp) {
      compilation_keys.emplace_back(BOOST_GET_CONST(
          std::string, node->Op()->GetAttr(operators::kCompilationKey)));
    }
  }
  return compilation_keys;
}

std::unique_ptr<Graph> BuildNoCinnSubgraph() {
  ProgramDesc prog;
  auto g = std::make_unique<Graph>(prog);
  // var1 --
  //        | --> fake1 --> var3 --> fake2 --> var4
  // var2 --
  OpDesc fake1_op;
  fake1_op.SetType("fake1");
  OpDesc fake2_op;
  fake2_op.SetType("fake2");

  VarDesc var1("var1");
  VarDesc var2("var2");
  var2.SetPersistable(true);
  var2.SetIsParameter(true);
  VarDesc var3("var3");
  VarDesc var4("var4");

  ir::Node* fake1 = g->CreateOpNode(&fake1_op);
  ir::Node* fake2 = g->CreateOpNode(&fake2_op);

  ir::Node* v1 = g->CreateVarNode(&var1);
  ir::Node* v2 = g->CreateVarNode(&var2);
  ir::Node* v3 = g->CreateVarNode(&var3);
  ir::Node* v4 = g->CreateVarNode(&var4);

  // fill op node
  fake1->inputs = {v1, v2};
  fake1->outputs = {v3};
  fake2->inputs = {v3};
  fake2->outputs = {v4};

  // fill variable node
  v1->outputs = {fake1};
  v2->outputs = {fake1};

  v3->inputs = {fake1};
  v3->outputs = {fake2};

  v4->inputs = {fake2};

  return g;
}

TEST(BuildCinnPassTest, NoCinnSubgraph) {
  auto g = BuildNoCinnSubgraph();
  auto previous_nodes = g->Nodes();

  auto pass =
      paddle::framework::ir::PassRegistry::Instance().Get("build_cinn_pass");
  pass->Apply(g.get());

  // After search, origin graph should no change
  ASSERT_EQ(previous_nodes, g->Nodes());
  ASSERT_TRUE(CheckGraphIndependence(g->Nodes()));

  // After search, there should be no cinn subgraph
  ASSERT_TRUE(GetCompilationKeys(*g).empty());
}

std::unique_ptr<Graph> BuildAllOpSupportCinnGraph() {
  ProgramDesc prog;
  auto g = std::make_unique<Graph>(prog);

  // v1 --
  //      | --> mul --> v3 --
  // v2 --                   | --> add --> v5 --> relu --> v6
  //                    v4 --

  OpDesc add_op;
  add_op.SetType("elementwise_add");
  OpDesc mul_op;
  mul_op.SetType("mul");
  OpDesc relu_op;
  relu_op.SetType("relu");

  VarDesc var1("var1");
  VarDesc var2("var2");
  var2.SetPersistable(true);
  var2.SetIsParameter(true);
  VarDesc var3("var3");
  VarDesc var4("var4");
  VarDesc var5("var5");
  VarDesc var6("var6");

  ir::Node* add = g->CreateOpNode(&add_op);
  ir::Node* mul = g->CreateOpNode(&mul_op);
  ir::Node* relu = g->CreateOpNode(&relu_op);

  ir::Node* v0 = g->CreateEmptyNode("var0", Node::Type::kVariable);
  ir::Node* v1 = g->CreateVarNode(&var1);
  ir::Node* v2 = g->CreateVarNode(&var2);
  ir::Node* v3 = g->CreateVarNode(&var3);
  ir::Node* v4 = g->CreateVarNode(&var4);
  ir::Node* v5 = g->CreateVarNode(&var5);
  ir::Node* v6 = g->CreateVarNode(&var6);
  ir::Node* v7 = g->CreateControlDepVar();

  // fill op node
  mul->inputs = {v0, v1, v2};
  mul->outputs = {v3};
  add->inputs = {v3, v4};
  add->outputs = {v5};
  relu->inputs = {v5};
  relu->outputs = {v6, v7};

  // fill variable node
  v0->outputs = {mul};
  v1->outputs = {mul};
  v2->outputs = {mul};

  v3->inputs = {mul};
  v3->outputs = {add};

  v4->outputs = {add};

  v5->inputs = {add};
  v5->outputs = {relu};

  v6->inputs = {relu};
  v7->inputs = {relu};

  return g;
}

TEST(BuildCinnPassTest, AllOpSupportCinn) {
  auto g = BuildAllOpSupportCinnGraph();

  auto pass =
      paddle::framework::ir::PassRegistry::Instance().Get("build_cinn_pass");
  pass->Apply(g.get());

  // After search, the graph should as following
  // v0 --|
  // v1 --|                   |--> v6
  // v2 --| --> kCinnLaunchOp |--> v7
  // v4 --|
  const auto& nodes = g->Nodes();
  ASSERT_EQ(nodes.size(), static_cast<size_t>(7));
  ASSERT_TRUE(CheckGraphIndependence(nodes));

  // A new op named kCinnLaunchOp should be added
  ASSERT_TRUE(CheckNodeExisted(nodes, kCinnLaunchOp));
  auto* cinn_op = GetNode(nodes, kCinnLaunchOp);
  auto* v0 = GetNode(nodes, "var0");
  auto* v1 = GetNode(nodes, "var1");
  auto* v2 = GetNode(nodes, "var2");
  auto* v4 = GetNode(nodes, "var4");
  auto* v6 = GetNode(nodes, "var6");
  auto* v7 = GetNode(nodes, Node::kControlDepVarName);

  ASSERT_EQ(
      std::unordered_set<Node*>(cinn_op->inputs.begin(), cinn_op->inputs.end()),
      std::unordered_set<Node*>({v0, v1, v2, v4}));
  ASSERT_EQ(std::unordered_set<Node*>(cinn_op->outputs.begin(),
                                      cinn_op->outputs.end()),
            std::unordered_set<Node*>({v6, v7}));
  ASSERT_EQ(v1->outputs, std::vector<Node*>({cinn_op}));
  ASSERT_EQ(v6->inputs, std::vector<Node*>({cinn_op}));

  // previous op (mul, add, relu) should all removed
  ASSERT_FALSE(CheckNodeExisted(nodes, "mul"));
  ASSERT_FALSE(CheckNodeExisted(nodes, "elementwise_add"));
  ASSERT_FALSE(CheckNodeExisted(nodes, "relu"));

  // After search, there should has just one cinn subgraph
  // feed --> v1 --
  //               | --> mul --> v3 --
  //          v2 --                   | --> add --> v5 --> relu --> v6 --> fetch
  //                    feed --> v4 --
  auto compilation_keys = GetCompilationKeys(*g);
  ASSERT_EQ(compilation_keys.size(), static_cast<size_t>(1));
  auto* cinn_compiler = CinnCompiler::GetInstance();
  const auto& subgraph = cinn_compiler->FindGraph(compilation_keys[0]);

  const auto& subnodes = subgraph.Nodes();
  ASSERT_EQ(subnodes.size(), static_cast<size_t>(12));
  ASSERT_TRUE(CheckGraphIndependence(subnodes));

  ASSERT_TRUE(CheckNodeExisted(subnodes, "mul"));
  ASSERT_TRUE(CheckNodeExisted(subnodes, "elementwise_add"));
  ASSERT_TRUE(CheckNodeExisted(subnodes, "relu"));
  ASSERT_EQ(CountNode(subnodes, "feed"), 2);
  ASSERT_EQ(CountNode(subnodes, "fetch"), 1);

  // No-parameter input should has feed op
  auto new_v1 = GetNode(subnodes, "var1");
  ASSERT_EQ(new_v1->inputs.size(), static_cast<size_t>(1));
  ASSERT_EQ(new_v1->outputs.size(), static_cast<size_t>(1));
  ASSERT_EQ(new_v1->inputs[0]->Name(), "feed");
  ASSERT_EQ(new_v1->outputs[0]->Name(), "mul");

  // Parameter input should not has feed op
  auto new_v2 = GetNode(subnodes, "var2");
  ASSERT_TRUE(new_v2->inputs.empty());
  ASSERT_EQ(new_v2->outputs.size(), static_cast<size_t>(1));
  ASSERT_EQ(new_v2->outputs[0]->Name(), "mul");

  // output should has fetch op
  auto new_v6 = GetNode(subnodes, "var6");
  ASSERT_EQ(new_v6->inputs.size(), static_cast<size_t>(1));
  ASSERT_EQ(new_v6->outputs.size(), static_cast<size_t>(1));
  ASSERT_EQ(new_v6->inputs[0]->Name(), "relu");
  ASSERT_EQ(new_v6->outputs[0]->Name(), "fetch");
}

std::unique_ptr<Graph> BuildGraphWithOneCinnSubgraph() {
  ProgramDesc prog;
  auto g = std::make_unique<Graph>(prog);

  // fake1 --> v1 --
  //                | --> mul --> v3 --> relu --> v4 --> fake2
  //           v2 --

  OpDesc fake1_op;
  fake1_op.SetType("fake1");
  OpDesc mul_op;
  mul_op.SetType("mul");
  OpDesc relu_op;
  relu_op.SetType("relu");
  OpDesc fake2_op;
  fake2_op.SetType("fake2");

  VarDesc var1("var1");
  VarDesc var2("var2");
  var2.SetPersistable(true);
  var2.SetIsParameter(true);
  VarDesc var3("var3");
  VarDesc var4("var4");

  ir::Node* fake1 = g->CreateOpNode(&fake1_op);
  ir::Node* mul = g->CreateOpNode(&mul_op);
  ir::Node* relu = g->CreateOpNode(&relu_op);
  ir::Node* fake2 = g->CreateOpNode(&fake2_op);

  ir::Node* v1 = g->CreateVarNode(&var1);
  ir::Node* v2 = g->CreateVarNode(&var2);
  ir::Node* v3 = g->CreateVarNode(&var3);
  ir::Node* v4 = g->CreateVarNode(&var4);

  // fill op node
  fake1->outputs = {v1};
  mul->inputs = {v2, v1};
  mul->outputs = {v3};
  relu->inputs = {v3};
  relu->outputs = {v4};
  fake2->inputs = {v4};

  // fill variable node
  v2->outputs = {mul};

  v1->inputs = {fake1};
  v1->outputs = {mul};

  v3->inputs = {mul};
  v3->outputs = {relu};

  v4->inputs = {relu};
  v4->outputs = {fake2};

  return g;
}

TEST(BuildCinnPassTest, OneCinnSubgraph) {
  auto g = BuildGraphWithOneCinnSubgraph();

  auto pass =
      paddle::framework::ir::PassRegistry::Instance().Get("build_cinn_pass");
  pass->Apply(g.get());

  // After search, the graph should as following
  // fake1 --> v1 --
  //                | --> kCinnLaunchOp --> v4 --> fake2
  //           v2 --
  const auto& nodes = g->Nodes();
  ASSERT_EQ(nodes.size(), static_cast<size_t>(6));
  ASSERT_TRUE(CheckGraphIndependence(nodes));

  // A new op named kCinnLaunchOp should be added
  ASSERT_TRUE(CheckNodeExisted(nodes, kCinnLaunchOp));

  // previous op (mul, add, relu) should be removed
  ASSERT_FALSE(CheckNodeExisted(nodes, "mul"));
  ASSERT_FALSE(CheckNodeExisted(nodes, "relu"));

  // previous op (fake1, fake2) should be preserved
  ASSERT_TRUE(CheckNodeExisted(nodes, "fake1"));
  ASSERT_TRUE(CheckNodeExisted(nodes, "fake2"));

  // After search, there should has just one cinn subgraph
  // feed --> v1 --
  //               | --> mul --> v3 --> relu --> v4 --> fetch
  //          v2 --
  auto compilation_keys = GetCompilationKeys(*g);
  ASSERT_EQ(compilation_keys.size(), static_cast<size_t>(1));
  auto* cinn_compiler = CinnCompiler::GetInstance();
  const auto& subgraph = cinn_compiler->FindGraph(compilation_keys[0]);

  const auto& subnodes = subgraph.Nodes();
  ASSERT_EQ(subnodes.size(), static_cast<size_t>(8));
  ASSERT_TRUE(CheckGraphIndependence(subnodes));

  ASSERT_TRUE(CheckNodeExisted(subnodes, "mul"));
  ASSERT_TRUE(CheckNodeExisted(subnodes, "relu"));
  ASSERT_EQ(CountNode(subnodes, "feed"), 1);
  ASSERT_EQ(CountNode(subnodes, "fetch"), 1);
}

std::unique_ptr<Graph> BuildGraphWithMultiCinnSubgraph() {
  ProgramDesc prog;
  auto g = std::make_unique<Graph>(prog);

  // fake1 --> v1 --
  //                | --> mul --> v3 --> fake2 --> v4 --> relu --> v5 --> fake3
  //           v2 --

  OpDesc fake1_op;
  fake1_op.SetType("fake1");
  OpDesc mul_op;
  mul_op.SetType("mul");
  OpDesc relu_op;
  relu_op.SetType("relu");
  OpDesc fake2_op;
  fake2_op.SetType("fake2");
  OpDesc fake3_op;
  fake3_op.SetType("fake3");

  VarDesc var1("var1");
  VarDesc var2("var2");
  var2.SetPersistable(true);
  var2.SetIsParameter(true);
  VarDesc var3("var3");
  VarDesc var4("var4");
  VarDesc var5("var5");

  ir::Node* fake1 = g->CreateOpNode(&fake1_op);
  ir::Node* mul = g->CreateOpNode(&mul_op);
  ir::Node* relu = g->CreateOpNode(&relu_op);
  ir::Node* fake2 = g->CreateOpNode(&fake2_op);
  ir::Node* fake3 = g->CreateOpNode(&fake3_op);

  ir::Node* v1 = g->CreateVarNode(&var1);
  ir::Node* v2 = g->CreateVarNode(&var2);
  ir::Node* v3 = g->CreateVarNode(&var3);
  ir::Node* v4 = g->CreateVarNode(&var4);
  ir::Node* v5 = g->CreateVarNode(&var5);

  // fill op node
  fake1->outputs = {v1};
  mul->inputs = {v2, v1};
  mul->outputs = {v3};
  fake2->inputs = {v3};
  fake2->outputs = {v4};
  relu->inputs = {v4};
  relu->outputs = {v5};
  fake3->inputs = {v5};

  // fill variable node
  v2->outputs = {mul};

  v1->inputs = {fake1};
  v1->outputs = {mul};

  v3->inputs = {mul};
  v3->outputs = {fake2};

  v4->inputs = {fake2};
  v4->outputs = {relu};

  v5->inputs = {relu};
  v5->outputs = {fake3};

  return g;
}

TEST(BuildCinnPassTest, MultiCinnSubgraph) {
  auto g = BuildGraphWithMultiCinnSubgraph();

  auto pass =
      paddle::framework::ir::PassRegistry::Instance().Get("build_cinn_pass");
  pass->Apply(g.get());

  // After search, the graph should as following
  // fake1 -> v1 -
  //              | -> CinnOp -> v3 -> fake2 -> v4 -> CinnOp ->v5 -> fake3
  //          v2 -
  const auto& nodes = g->Nodes();
  ASSERT_EQ(nodes.size(), static_cast<size_t>(10));
  ASSERT_TRUE(CheckGraphIndependence(nodes));

  // A new op named kCinnLaunchOp should be added
  ASSERT_TRUE(CheckNodeExisted(nodes, kCinnLaunchOp));
  ASSERT_EQ(CountNode(nodes, kCinnLaunchOp), 2);

  // previous op (mul, add, relu) should be removed
  ASSERT_FALSE(CheckNodeExisted(nodes, "mul"));
  ASSERT_FALSE(CheckNodeExisted(nodes, "relu"));

  // previous op (fake1, fake2) should be preserved
  ASSERT_TRUE(CheckNodeExisted(nodes, "fake1"));
  ASSERT_TRUE(CheckNodeExisted(nodes, "fake2"));
  ASSERT_TRUE(CheckNodeExisted(nodes, "fake3"));

  // After search, there should has two cinn subgraphs,
  // and each of subgraphs just has one node.
  auto compilation_keys = GetCompilationKeys(*g);
  ASSERT_EQ(compilation_keys.size(), static_cast<size_t>(2));

  // subgraph1:
  // feed --> v4 --> relu --> v5 --> fetch
  // subgraph2:
  // feed --> v1 --
  //               | --> mul --> v3 --> fetch
  //          v2 --
  auto* cinn_compiler = CinnCompiler::GetInstance();
  const auto& subgraph1 = cinn_compiler->FindGraph(compilation_keys[0]);
  const auto& subnodes1 = subgraph1.Nodes();
  ASSERT_TRUE(CheckGraphIndependence(subnodes1));

  const auto& subgraph2 = cinn_compiler->FindGraph(compilation_keys[1]);
  const auto& subnodes2 = subgraph2.Nodes();
  ASSERT_TRUE(CheckGraphIndependence(subnodes2));

  if (CheckNodeExisted(subnodes1, "relu")) {
    ASSERT_EQ(subnodes1.size(), static_cast<size_t>(5));
    ASSERT_EQ(subnodes2.size(), static_cast<size_t>(6));
  } else {
    ASSERT_EQ(subnodes2.size(), static_cast<size_t>(5));
    ASSERT_EQ(subnodes1.size(), static_cast<size_t>(6));
  }
}

std::unique_ptr<Graph> BuildGraphWithNoNeedBufferInput() {
  ProgramDesc prog;
  auto g = std::make_unique<Graph>(prog);

  // fake1 --> v1 --                 --> v4 --> relu_grad --> v6
  //           v2 -- | --> add_grad |
  //           v3 --                 --> v5 --> fake2

  OpDesc fake1_op;
  fake1_op.SetType("fake1");
  OpDesc add_grad_op;
  add_grad_op.SetType("elementwise_add_grad");
  add_grad_op.SetInput(::paddle::framework::GradVarName("Out"), {"var1"});
  add_grad_op.SetInput("X", {"var2"});
  add_grad_op.SetInput("Y", {"var3"});
  OpDesc relu_grad_op;
  relu_grad_op.SetType("relu_grad");
  OpDesc fake2_op;
  fake2_op.SetType("fake2");

  VarDesc var1("var1");
  VarDesc var2("var2");
  VarDesc var3("var3");
  VarDesc var4("var4");
  VarDesc var5("var5");
  VarDesc var6("var6");

  ir::Node* fake1 = g->CreateOpNode(&fake1_op);
  ir::Node* add_grad = g->CreateOpNode(&add_grad_op);
  ir::Node* relu_grad = g->CreateOpNode(&relu_grad_op);
  ir::Node* fake2 = g->CreateOpNode(&fake2_op);

  ir::Node* v1 = g->CreateVarNode(&var1);
  ir::Node* v2 = g->CreateVarNode(&var2);
  ir::Node* v3 = g->CreateVarNode(&var3);
  ir::Node* v4 = g->CreateVarNode(&var4);
  ir::Node* v5 = g->CreateVarNode(&var5);
  ir::Node* v6 = g->CreateVarNode(&var6);

  // fill op node
  fake1->outputs = {v1};
  add_grad->inputs = {v1, v2, v3};
  add_grad->outputs = {v4, v5};
  relu_grad->inputs = {v4};
  relu_grad->outputs = {v6};
  fake2->inputs = {v5};

  // fill variable node
  v1->inputs = {fake1};
  v1->outputs = {add_grad};

  v2->outputs = {add_grad};
  v3->outputs = {add_grad};

  v4->inputs = {add_grad};
  v4->outputs = {relu_grad};
  v5->inputs = {add_grad};
  v5->outputs = {fake2};

  v6->inputs = {relu_grad};

  return g;
}

TEST(BuildCinnPassTest, NoNeedBufferInput) {
  auto g = BuildGraphWithNoNeedBufferInput();

  auto pass =
      paddle::framework::ir::PassRegistry::Instance().Get("build_cinn_pass");
  pass->Apply(g.get());

  // After search, the graph should as following
  // fake1 --> v1 --                     --> v6
  //           v2 -- | -->kCinnLaunchOp |
  //           v3 --                     --> v5 --> fake2
  const auto& nodes = g->Nodes();
  ASSERT_EQ(nodes.size(), static_cast<size_t>(8));
  ASSERT_TRUE(CheckGraphIndependence(nodes));

  // A new op named kCinnLaunchOp should be added and
  // its input arguments are set correctly
  ASSERT_TRUE(CheckNodeExisted(nodes, kCinnLaunchOp));
  ASSERT_EQ(CountNode(nodes, kCinnLaunchOp), 1);
  auto* cinn_op_node = GetNode(nodes, kCinnLaunchOp);
  ASSERT_EQ(cinn_op_node->Op()->Input(operators::kX),
            std::vector<std::string>({"var1"}));
  auto& no_need_buffer_x = cinn_op_node->Op()->Input(operators::kNoNeedBufferX);
  ASSERT_EQ(std::unordered_set<std::string>(no_need_buffer_x.begin(),
                                            no_need_buffer_x.end()),
            std::unordered_set<std::string>({"var2", "var3"}));

  // previous op (add_grad, relu_grad) should be removed
  ASSERT_FALSE(CheckNodeExisted(nodes, "add_grad"));
  ASSERT_FALSE(CheckNodeExisted(nodes, "relu_grad"));

  // previous op (fake1, fake2) should be preserved
  ASSERT_TRUE(CheckNodeExisted(nodes, "fake1"));
  ASSERT_TRUE(CheckNodeExisted(nodes, "fake2"));

  // After search, there should has just one cinn subgraph
  // feed --> v1 --                                     --> v6 --> fetch
  // feed --> v2 -- | -->add_grad --> v4 --> relu_grad |
  // feed --> v3 --                                     --> v5 --> fetch
  auto compilation_keys = GetCompilationKeys(*g);
  ASSERT_EQ(compilation_keys.size(), static_cast<size_t>(1));
  auto* cinn_compiler = CinnCompiler::GetInstance();
  const auto& subgraph = cinn_compiler->FindGraph(compilation_keys[0]);

  const auto& subnodes = subgraph.Nodes();
  ASSERT_EQ(subnodes.size(), static_cast<size_t>(13));
  ASSERT_TRUE(CheckGraphIndependence(subnodes));

  ASSERT_TRUE(CheckNodeExisted(subnodes, "elementwise_add_grad"));
  ASSERT_TRUE(CheckNodeExisted(subnodes, "relu_grad"));
  ASSERT_EQ(CountNode(subnodes, "feed"), 3);
  ASSERT_EQ(CountNode(subnodes, "fetch"), 2);
  const auto& no_need_buffer_feeds =
      subgraph.Get<std::unordered_set<std::string>>(kNoNeedBufferFeeds);
  ASSERT_EQ(no_need_buffer_feeds.size(), 2);
  ASSERT_EQ(no_need_buffer_feeds,
            std::unordered_set<std::string>({"var2", "var3"}));

  // check the attributes of variable lists are saved correctly
  ASSERT_TRUE(subgraph.Has(kInputVars));
  EXPECT_EQ(subgraph.Get<std::vector<std::string>>(kInputVars),
            std::vector<std::string>({"var1"}));
  ASSERT_TRUE(subgraph.Has(kInternalVars));
  EXPECT_EQ(subgraph.Get<std::vector<std::string>>(kInternalVars),
            std::vector<std::string>({"var4"}));
  ASSERT_TRUE(subgraph.Has(kOutputVars));
  const auto& output_vars = subgraph.Get<std::vector<std::string>>(kOutputVars);
  EXPECT_EQ(
      std::unordered_set<std::string>(output_vars.begin(), output_vars.end()),
      std::unordered_set<std::string>({"var5", "var6"}));
}

}  // namespace paddle2cinn
}  // namespace framework
}  // namespace paddle

USE_PASS(build_cinn_pass);
USE_OP_ITSELF(mul);
USE_OP_ITSELF(relu);
USE_OP_ITSELF(elementwise_add);
USE_OP_ITSELF(relu_grad);
USE_OP_ITSELF(elementwise_add_grad);
