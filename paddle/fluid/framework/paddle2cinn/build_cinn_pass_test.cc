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

#include "gtest/gtest.h"

#include "paddle/fluid/framework/details/build_strategy.h"
#include "paddle/fluid/framework/ir/graph.h"
#include "paddle/fluid/framework/ir/node.h"
#include "paddle/fluid/framework/op_desc.h"
#include "paddle/fluid/framework/program_desc.h"
#include "paddle/fluid/framework/var_desc.h"

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
  return *std::find_if(
      nodes.begin(), nodes.end(),
      [&op_name](const Node* node) { return node->Name() == op_name; });
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
  std::vector<std::unique_ptr<Graph>> cinn_subgraphs;
  pass->SetNotOwned<std::vector<std::unique_ptr<Graph>>>("cinn_subgraphs",
                                                         &cinn_subgraphs);
  pass->Apply(g.get());

  // After search, origin graph should no change
  ASSERT_EQ(previous_nodes, g->Nodes());

  // After search, there should one cinn subgraph
  ASSERT_TRUE(cinn_subgraphs.empty());
}

std::unique_ptr<Graph> BuildAllOpSupportCinnGraph() {
  ProgramDesc prog;
  auto g = std::make_unique<Graph>(prog);

  // v1 --
  //      |
  //      | --> mul --> v3 --
  //      |                  |
  // v2 --                   | --> add --> v5 --> relu --> v6
  //                         |
  //                    v4 --

  OpDesc add_op;
  add_op.SetType("add");
  OpDesc mul_op;
  mul_op.SetType("mul");
  OpDesc relu_op;
  relu_op.SetType("relu");

  VarDesc var1("var1");
  VarDesc var2("var2");
  VarDesc var3("var3");
  VarDesc var4("var4");
  VarDesc var5("var5");
  VarDesc var6("var6");

  ir::Node* add = g->CreateOpNode(&add_op);
  ir::Node* mul = g->CreateOpNode(&mul_op);
  ir::Node* relu = g->CreateOpNode(&relu_op);

  ir::Node* v1 = g->CreateVarNode(&var1);
  ir::Node* v2 = g->CreateVarNode(&var2);
  ir::Node* v3 = g->CreateVarNode(&var3);
  ir::Node* v4 = g->CreateVarNode(&var4);
  ir::Node* v5 = g->CreateVarNode(&var5);
  ir::Node* v6 = g->CreateVarNode(&var6);

  // fill op node
  mul->inputs = {v1, v2};
  mul->outputs = {v3};
  add->inputs = {v3, v4};
  add->outputs = {v5};
  relu->inputs = {v5};
  relu->outputs = {v6};

  // fill variable node
  v1->outputs = {mul};
  v2->outputs = {mul};

  v3->inputs = {mul};
  v3->outputs = {add};

  v4->outputs = {add};

  v5->inputs = {add};
  v5->outputs = {relu};

  v6->inputs = {relu};

  return g;
}

TEST(BuildCinnPassTest, AllOpSupportCinn) {
  auto g = BuildAllOpSupportCinnGraph();

  auto pass =
      paddle::framework::ir::PassRegistry::Instance().Get("build_cinn_pass");
  std::vector<std::unique_ptr<Graph>> cinn_subgraphs;
  pass->SetNotOwned<std::vector<std::unique_ptr<Graph>>>("cinn_subgraphs",
                                                         &cinn_subgraphs);
  pass->Apply(g.get());

  // After search, the graph should as following
  // v1 --|
  // v2 --| --> kCinnLaunchOp --> v6
  // v4 --|
  const auto& nodes = g->Nodes();
  ASSERT_EQ(nodes.size(), static_cast<size_t>(5));

  // A new op named kCinnLaunchOp should be added
  ASSERT_TRUE(CheckNodeExisted(nodes, kCinnLaunchOp));
  auto* cinn_op = GetNode(nodes, kCinnLaunchOp);
  auto* v1 = GetNode(nodes, "var1");
  auto* v2 = GetNode(nodes, "var2");
  auto* v4 = GetNode(nodes, "var4");
  auto* v6 = GetNode(nodes, "var6");

  ASSERT_EQ(
      std::unordered_set<Node*>(cinn_op->inputs.begin(), cinn_op->inputs.end()),
      std::unordered_set<Node*>({v1, v2, v4}));
  ASSERT_EQ(cinn_op->outputs, std::vector<Node*>({v6}));
  ASSERT_EQ(v1->outputs, std::vector<Node*>({cinn_op}));
  ASSERT_EQ(v6->inputs, std::vector<Node*>({cinn_op}));

  // previous op (mul, add, relu) should all removed
  ASSERT_FALSE(CheckNodeExisted(nodes, "mul"));
  ASSERT_FALSE(CheckNodeExisted(nodes, "add"));
  ASSERT_FALSE(CheckNodeExisted(nodes, "relu"));

  // After search, there should has just one cinn subgraph
  // mul --> v3 --> add --> v5 --> relu
  ASSERT_EQ(cinn_subgraphs.size(), static_cast<size_t>(1));
  const auto& subgraph = cinn_subgraphs.back();

  const auto& subnodes = subgraph->Nodes();
  ASSERT_EQ(subnodes.size(), static_cast<size_t>(5));

  ASSERT_TRUE(CheckNodeExisted(subnodes, "mul"));
  ASSERT_TRUE(CheckNodeExisted(subnodes, "add"));
  ASSERT_TRUE(CheckNodeExisted(subnodes, "relu"));
}

std::unique_ptr<Graph> BuildGraphWithOneCinnSubgraph() {
  ProgramDesc prog;
  auto g = std::make_unique<Graph>(prog);

  // fake1 --> v1 --
  //                |
  //                | --> mul --> v3 --> relu --> v4 --> fake2
  //                |
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
  std::vector<std::unique_ptr<Graph>> cinn_subgraphs;
  pass->SetNotOwned<std::vector<std::unique_ptr<Graph>>>("cinn_subgraphs",
                                                         &cinn_subgraphs);
  pass->Apply(g.get());

  // After search, the graph should as following
  // fake1 --> v1 --
  //                | --> kCinnLaunchOp --> v4 --> fake2
  //           v2 --
  const auto& nodes = g->Nodes();
  ASSERT_EQ(nodes.size(), static_cast<size_t>(6));

  // A new op named kCinnLaunchOp should be added
  ASSERT_TRUE(CheckNodeExisted(nodes, kCinnLaunchOp));

  // previous op (mul, add, relu) should be removed
  ASSERT_FALSE(CheckNodeExisted(nodes, "mul"));
  ASSERT_FALSE(CheckNodeExisted(nodes, "relu"));

  // previous op (fake1, fake2) should be preserved
  ASSERT_TRUE(CheckNodeExisted(nodes, "fake1"));
  ASSERT_TRUE(CheckNodeExisted(nodes, "fake2"));

  // After search, there should has just one cinn subgraph
  // mul --> v3 --> relu
  ASSERT_EQ(cinn_subgraphs.size(), static_cast<size_t>(1));
  const auto& subgraph = cinn_subgraphs.back();

  const auto& subnodes = subgraph->Nodes();
  ASSERT_EQ(subnodes.size(), static_cast<size_t>(3));

  ASSERT_TRUE(CheckNodeExisted(subnodes, "mul"));
  ASSERT_TRUE(CheckNodeExisted(subnodes, "relu"));
}

std::unique_ptr<Graph> BuildGraphWithMultiCinnSubgraph() {
  ProgramDesc prog;
  auto g = std::make_unique<Graph>(prog);

  // fake1 --> v1 --
  //                |
  //                | --> mul --> v3 --> fake2 --> v4 --> relu --> v5 --> fake3
  //                |
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
  std::vector<std::unique_ptr<Graph>> cinn_subgraphs;
  pass->SetNotOwned<std::vector<std::unique_ptr<Graph>>>("cinn_subgraphs",
                                                         &cinn_subgraphs);
  pass->Apply(g.get());

  // After search, the graph should as following
  // fake1 -> v1 -
  //              | -> CinnOp -> v3 -> fake2 -> v4 -> CinnOp ->v5 -> fake3
  //          v2 -
  const auto& nodes = g->Nodes();
  ASSERT_EQ(nodes.size(), static_cast<size_t>(10));

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
  ASSERT_EQ(cinn_subgraphs.size(), static_cast<size_t>(2));

  // subgraph1: relu
  const auto& subgraph1 = cinn_subgraphs[0];
  const auto& subnodes1 = subgraph1->Nodes();
  ASSERT_EQ(subnodes1.size(), static_cast<size_t>(1));

  // subgraph2: mul
  const auto& subgraph2 = cinn_subgraphs[1];
  const auto& subnodes2 = subgraph2->Nodes();
  ASSERT_EQ(subnodes2.size(), static_cast<size_t>(1));
}

}  // namespace paddle2cinn
}  // namespace framework
}  // namespace paddle

USE_PASS(build_cinn_pass);
