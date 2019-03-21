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

#include "paddle/fluid/framework/details/memory_optimize_helper.h"
#include <algorithm>
#include <iostream>
#include <iterator>
#include <memory>
#include <sstream>
#include <string>
#include <utility>
#include <vector>
#include "glog/logging.h"
#include "gtest/gtest.h"
#include "paddle/fluid/framework/details/graph_test_base.h"
#include "paddle/fluid/framework/ir/graph.h"
#include "paddle/fluid/framework/ir/graph_helper.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/operator.h"
#include "paddle/fluid/framework/program_desc.h"

namespace paddle {
namespace framework {
namespace details {

TEST(OrderedSet, Normal) {
  OrderedSet pool;
  std::vector<std::unique_ptr<ir::Node>> nodes;

  // clang-format off
  std::vector<std::vector<int64_t>> shapes = {{-1, 10},
                                              {-1, 20},
                                              {1, 2},
                                              {5, 2},
                                              {10, 20},
                                              {-1, 2, 5},
                                              {-1, 1, 5},
                                              {-1, 1}};
  // clang-format on
  const int COUNT = shapes.size();
  ProgramDesc prog;
  BlockDesc* block_desc = prog.MutableBlock(0);
  auto* op_desc = block_desc->AppendOp();
  op_desc->SetType("dummy");
  std::unique_ptr<ir::Node> op = ir::CreateNodeForTest(op_desc);

  for (int i = 0; i < COUNT; ++i) {
    auto desc = block_desc->Var(std::to_string(i));
    desc->SetShape(shapes[i]);
    std::unique_ptr<ir::Node> node = ir::CreateNodeForTest(desc);
    node->inputs.emplace_back(op.get());
    nodes.emplace_back(std::move(node));
  }

  // Insert
  for (auto& node : nodes) {
    pool.Insert(node.get());
  }

  // Has/size
  ASSERT_EQ(pool.size(), shapes.size());
  for (auto& node : nodes) {
    ASSERT_TRUE(pool.Has(node.get()));
  }

  // assert its order and interface.
  std::cout << pool.ToString() << std::endl;
  pool.Erase(nodes.front().get());
  std::cout << pool.ToString() << std::endl;

  ASSERT_EQ(pool.size(), static_cast<size_t>(COUNT - 1));
  ASSERT_EQ(pool.GetNodeIndexInPool(nodes.back().get()), 0);

  {
    auto v1 = block_desc->Var("11");
    v1->SetShape({-1, 256, 56, 56});
    std::unique_ptr<ir::Node> node1 = ir::CreateNodeForTest(v1);
    node1->inputs.emplace_back(op.get());
    auto* cache = pool.FindBestFitNode(node1.get());
    ASSERT_EQ(cache, nullptr);
  }
  {
    auto v2 = block_desc->Var("12");
    v2->SetShape({-1, 2, 5});
    std::unique_ptr<ir::Node> node1 = ir::CreateNodeForTest(v2);
    node1->inputs.emplace_back(op.get());
    auto* cache = pool.FindBestFitNode(node1.get());
    ASSERT_EQ(pool.GetNodeIndexInPool(cache), 2);  // match 6:[-1,2,5]
  }
  {
    auto v3 = block_desc->Var("13");
    v3->SetShape({2, 5});
    std::unique_ptr<ir::Node> node1 = ir::CreateNodeForTest(v3);
    node1->inputs.emplace_back(op.get());
    auto* cache = pool.FindBestFitNode(node1.get());
    ASSERT_EQ(pool.GetNodeIndexInPool(cache), 5);  // match  4:[5,2]
  }
}

TEST(OrderedSet, FindBestFitNode) {
  OrderedSet pool;
  std::vector<std::unique_ptr<ir::Node>> nodes;
  ProgramDesc prog;
  BlockDesc* block_desc = prog.MutableBlock(0);
  auto* op_desc = block_desc->AppendOp();
  op_desc->SetType("dummy");
  std::unique_ptr<ir::Node> op = ir::CreateNodeForTest(op_desc);

  {
    auto desc = block_desc->Var("a");
    desc->SetShape({128, 128});
    std::unique_ptr<ir::Node> node = ir::CreateNodeForTest(desc);
    node->inputs.emplace_back(op.get());
    nodes.emplace_back(std::move(node));
  }
  {
    auto desc = block_desc->Var("b");
    desc->SetShape({128, 129});
    std::unique_ptr<ir::Node> node = ir::CreateNodeForTest(desc);
    node->inputs.emplace_back(op.get());
    nodes.emplace_back(std::move(node));
  }
  {
    auto desc = block_desc->Var("c");
    desc->SetShape({128, 128});
    std::unique_ptr<ir::Node> node = ir::CreateNodeForTest(desc);
    node->inputs.emplace_back(op.get());
    nodes.emplace_back(std::move(node));
  }

  for (auto& node : nodes) {
    pool.Insert(node.get());
  }

  // FindNextBestFitNode
  auto* n = nodes[0].get();
  auto* cache = pool.FindBestFitNode(n);
  PADDLE_ENFORCE(cache->Name() == "a");
  cache = pool.FindNextBestFitNode(n, cache);
  PADDLE_ENFORCE(cache->Name() == "c");
  cache = pool.FindNextBestFitNode(n, cache);
  PADDLE_ENFORCE(cache->Name() == "b");
}

}  // namespace details
}  // namespace framework
}  // namespace paddle

REGISTER_OPERATOR(sum, paddle::framework::DummyOp,
                  paddle::framework::SumOpMaker,
                  paddle::framework::DummyVarTypeInference);
REGISTER_OPERATOR(assign, paddle::framework::DummyOp,
                  paddle::framework::AssignOpMaker,
                  paddle::framework::DummyVarTypeInference);
REGISTER_OPERATOR(dummy, paddle::framework::DummyOp,
                  paddle::framework::SumOpMaker,
                  paddle::framework::DummyVarTypeInference);
/*
  https://en.wikipedia.org/wiki/Live_variable_analysis
  Create a customed classical dependency graph, left row is the instruction
  number.
  1. a = 1
  2. b = a
  3. c = a
  4. d = b + c
  5. e = d

  a--------+
  |        |
  b        c
  |        |
  d--------+
  |
  e
  Then analysis these variable's liveness range
 */

namespace paddle {
namespace framework {
namespace details {

inline static ProgramDesc FillProgramDesc() {
  ProgramDesc prog;
  prog.MutableBlock(0)->Var("a")->SetType(proto::VarType::LOD_TENSOR);
  prog.MutableBlock(0)->Var("b")->SetType(proto::VarType::LOD_TENSOR);
  prog.MutableBlock(0)->Var("c")->SetType(proto::VarType::LOD_TENSOR);
  prog.MutableBlock(0)->Var("d")->SetType(proto::VarType::LOD_TENSOR);
  prog.MutableBlock(0)->Var("e")->SetType(proto::VarType::LOD_TENSOR);
  {
    auto* op = prog.MutableBlock(0)->AppendOp();
    op->SetType("assign");
    op->SetInput("X", {"a"});
    op->SetOutput("Out", {"b"});
  }
  {
    auto* op = prog.MutableBlock(0)->AppendOp();
    op->SetType("assign");
    op->SetInput("X", {"a"});
    op->SetOutput("Out", {"c"});
  }
  {
    auto* op = prog.MutableBlock(0)->AppendOp();
    op->SetType("sum");
    op->SetInput("X", {"b", "c"});
    op->SetOutput("Out", {"d"});
  }
  {
    auto* op = prog.MutableBlock(0)->AppendOp();
    op->SetType("assign");
    op->SetInput("X", {"d"});
    op->SetOutput("Out", {"e"});
  }
  return prog;
}

TEST(CFGGraph, IRGraph) {
  // prepare ir graph
  auto prog = FillProgramDesc();
  ir::Graph graph(prog);

  ControlFlowGraph cfg(graph);
  cfg.LiveVariableAnalysis();

  // test assign op
  ASSERT_TRUE((std::set<std::string>{"a"} == cfg.LiveIn(cfg.Ops()[0])));
  ASSERT_TRUE((std::set<std::string>{"a", "b"} == cfg.LiveOut(cfg.Ops()[0])));

  // test assign op
  ASSERT_TRUE((std::set<std::string>{"a", "b"} == cfg.LiveIn(cfg.Ops()[1])));
  ASSERT_TRUE((std::set<std::string>{"b", "c"} == cfg.LiveOut(cfg.Ops()[1])));

  // test sum op
  ASSERT_TRUE((std::set<std::string>{"b", "c"} == cfg.LiveIn(cfg.Ops()[2])));
  ASSERT_TRUE((std::set<std::string>{"d"} == cfg.LiveOut(cfg.Ops()[2])));

  // test assign op
  ASSERT_TRUE((std::set<std::string>{"d"} == cfg.LiveIn(cfg.Ops()[3])));
  ASSERT_TRUE((std::set<std::string>{} == cfg.LiveOut(cfg.Ops()[3])));
}

// 1. normal test
TEST(SortOpLikeDescOrder, NormalTest) {
  auto prog = FillProgramDesc();
  ir::Graph graph(prog);

  auto nodes = SortOpLikeDescOrder(graph);
  auto op_descs = prog.Block(0).AllOps();
  for (size_t i = 0; i < nodes.size(); ++i) {
    auto node = nodes[i];
    auto op_desc = op_descs[i];
    ASSERT_TRUE(IsSameDesc(node->Op(), op_desc));
  }
}

// 2. remove some op_desc
TEST(SortOpLikeDescOrder, RemoveOpDesc) {
  auto prog = FillProgramDesc();
  ir::Graph graph(prog);
  auto nodes = graph.Nodes();
  auto op_descs = prog.Block(0).AllOps();
  ir::Node* found_node = nullptr;
  for (auto node : nodes) {
    if (node->IsOp() && node->outputs.back()->Name() == "e") {
      found_node = node;
      break;
    }
  }
  PADDLE_ENFORCE(found_node != nullptr);
  for (auto it = op_descs.begin(); it != op_descs.end();) {
    if (IsSameDesc(*it, found_node->Op())) {
      it = op_descs.erase(it);
    } else {
      ++it;
    }
  }

  auto find_node_in_graph = [&](std::string s) {
    ir::Node* ret = nullptr;
    for (auto n : graph.Nodes()) {
      if (n->Name() == s) {
        ret = n;
        break;
      }
    }
    PADDLE_ENFORCE(ret != nullptr);
    return ret;
  };

  ir::Node* e = find_node_in_graph("e");
  ir::Node* d = find_node_in_graph("d");
  std::remove(d->outputs.begin(), d->outputs.end(), found_node);
  graph.RemoveNode(found_node);
  graph.RemoveNode(e);

  // other node keeps the same order
  auto remain_nodes = SortOpLikeDescOrder(graph);
  for (size_t i = 0; i < remain_nodes.size(); ++i) {
    auto node = remain_nodes[i];
    auto op_desc = op_descs[i];
    ASSERT_TRUE(IsSameDesc(node->Op(), op_desc));
  }
}

// 3. add some op_desc
TEST(SortOpLikeDescOrder, AddOpDesc) {
  auto prog = FillProgramDesc();
  ir::Graph graph(prog);

  auto find_node_in_graph = [&](std::string s) {
    ir::Node* ret = nullptr;
    for (auto n : graph.Nodes()) {
      if (n->Name() == s) {
        ret = n;
        break;
      }
    }
    PADDLE_ENFORCE(ret != nullptr);
    return ret;
  };

  // cached desc different with real one
  // mimic the intermidiete pass modify the programdesc.
  std::vector<OpDesc*> op_descs = graph.OriginProgram().Block(0).AllOps();

  auto op = prog.MutableBlock(0)->AppendOp();
  prog.MutableBlock(0)->Var("d1")->SetType(proto::VarType::LOD_TENSOR);
  op->SetType("sum");
  op->SetInput("X", {"b", "c"});
  op->SetOutput("Out", {"d1"});
  ir::Node* node = graph.CreateOpNode(op);
  ir::Node* d1 = graph.CreateVarNode(prog.MutableBlock(0)->Var("d1"));
  ir::Node* b = find_node_in_graph("b");
  ir::Node* c = find_node_in_graph("c");
  node->outputs.emplace_back(d1);
  node->inputs.emplace_back(b);
  node->inputs.emplace_back(c);
  d1->inputs.emplace_back(node);
  b->outputs.emplace_back(node);
  c->outputs.emplace_back(node);
  op_descs.insert(op_descs.begin() + 4, op);

  auto nodes = SortOpLikeDescOrder(graph);

  for (size_t i = 0; i < nodes.size(); ++i) {
    auto node = nodes[i];
    auto op_desc = op_descs[i];
    ASSERT_TRUE(IsSameDesc(node->Op(), op_desc));
  }
}

// 4. add and delete some op_desc
TEST(SortOpLikeDescOrder, AddAndDeleteOpDesc) {
  auto prog = FillProgramDesc();
  ir::Graph graph(prog);

  auto find_node_in_graph = [&](std::string s) {
    ir::Node* ret = nullptr;
    for (auto n : graph.Nodes()) {
      if (n->Name() == s) {
        ret = n;
        break;
      }
    }
    PADDLE_ENFORCE(ret != nullptr);
    return ret;
  };

  std::vector<OpDesc*> op_descs = graph.OriginProgram().Block(0).AllOps();

  // remove sum node
  ir::Node* found_node = nullptr;
  auto nodes = graph.Nodes();
  for (auto node : nodes) {
    if (node->Name() == "sum") {
      found_node = node;
      break;
    }
  }
  PADDLE_ENFORCE(found_node != nullptr);
  for (auto it = op_descs.begin(); it != op_descs.end();) {
    if (IsSameDesc(*it, found_node->Op())) {
      it = op_descs.erase(it);
    } else {
      ++it;
    }
  }
  {
    ir::Node* d = find_node_in_graph("d");
    ir::Node* c = find_node_in_graph("c");
    ir::Node* e = find_node_in_graph("e");
    std::remove(d->outputs.begin(), d->outputs.end(), found_node);
    std::remove(c->outputs.begin(), c->outputs.end(), found_node);
    ir::Node* pending_op = found_node->outputs[0]->outputs[0];
    graph.RemoveNode(e);
    graph.RemoveNode(pending_op);
    graph.RemoveNode(found_node);
  }

  // add node
  auto op = prog.MutableBlock(0)->AppendOp();
  prog.MutableBlock(0)->Var("d1")->SetType(proto::VarType::LOD_TENSOR);
  op->SetType("sum");
  op->SetInput("X", {"b", "c"});
  op->SetOutput("Out", {"d1"});
  {
    ir::Node* node = graph.CreateOpNode(op);
    ir::Node* d1 = graph.CreateVarNode(prog.MutableBlock(0)->Var("d1"));
    ir::Node* b = find_node_in_graph("b");
    ir::Node* c = find_node_in_graph("c");
    node->outputs.emplace_back(d1);
    node->inputs.emplace_back(b);
    node->inputs.emplace_back(c);
    b->outputs.emplace_back(node);
    c->outputs.emplace_back(node);
  }
  op_descs.insert(op_descs.begin() + 2, op);

  // check the order
  auto mynodes = SortOpLikeDescOrder(graph);
  for (size_t i = 0; i < mynodes.size(); ++i) {
    auto node = mynodes[i];
    auto op_desc = op_descs[i];
    ASSERT_TRUE(IsSameDesc(node->Op(), op_desc));
  }
}

// 5. add and replace some op_desc inplace.
TEST(SortOpLikeDescOrder, AddAndReplaceOpDescInplace) {
  auto prog = FillProgramDesc();
  ir::Graph graph(prog);
  std::vector<OpDesc*> op_descs = graph.OriginProgram().Block(0).AllOps();

  auto find_node_in_graph = [&](std::string s) {
    ir::Node* ret = nullptr;
    for (auto n : graph.Nodes()) {
      if (n->Name() == s) {
        ret = n;
        break;
      }
    }
    PADDLE_ENFORCE(ret != nullptr);
    return ret;
  };

  // add node
  auto op = prog.MutableBlock(0)->AppendOp();
  prog.MutableBlock(0)->Var("d1")->SetType(proto::VarType::LOD_TENSOR);
  op->SetType("sum");
  op->SetInput("X", {"b", "c"});
  op->SetOutput("Out", {"d1"});
  {
    ir::Node* node = graph.CreateOpNode(op);
    ir::Node* d1 = graph.CreateVarNode(prog.MutableBlock(0)->Var("d1"));
    ir::Node* b = find_node_in_graph("b");
    ir::Node* c = find_node_in_graph("c");
    node->outputs.emplace_back(d1);
    node->inputs.emplace_back(b);
    node->inputs.emplace_back(c);
    d1->inputs.emplace_back(node);
    b->outputs.emplace_back(node);
    c->outputs.emplace_back(node);
  }

  op_descs.emplace_back(op);

  // replace op_desc inplace
  auto nodes = graph.Nodes();
  ir::Node* found_node = nullptr;
  for (auto node : nodes) {
    if (node->IsOp() && node->Op() && node->Name() == "assign") {
      if (node->outputs.size() == 1 && node->outputs[0]->Name() == "e") {
        found_node = node;
        break;
      }
    }
  }
  {
    ir::Node* d = find_node_in_graph("d");
    ir::Node* e = find_node_in_graph("e");
    std::remove(d->outputs.begin(), d->outputs.end(), found_node);
    std::remove(e->inputs.begin(), e->inputs.end(), found_node);
    graph.RemoveNode(found_node);
  }
  op_descs.erase(op_descs.begin() + 3);

  auto replace_op = prog.MutableBlock(0)->AppendOp();
  replace_op->SetType("sum");
  replace_op->SetInput("X", {"d", "d1"});
  replace_op->SetOutput("Out", {"e"});
  {
    ir::Node* sum2 = graph.CreateOpNode(replace_op);
    ir::Node* e = find_node_in_graph("e");
    ir::Node* d = find_node_in_graph("d");
    ir::Node* d1 = find_node_in_graph("d1");
    sum2->inputs.emplace_back(d);
    sum2->inputs.emplace_back(d1);
    sum2->outputs.emplace_back(e);
    e->inputs.emplace_back(sum2);
    d->outputs.emplace_back(sum2);
    d1->outputs.emplace_back(sum2);
  }

  op_descs.emplace_back(replace_op);
  // compare op order
  auto graph_nodes = SortOpLikeDescOrder(graph);
  for (size_t i = 0; i < graph_nodes.size(); ++i) {
    auto node = graph_nodes[i];
    auto op_desc = op_descs[i];
    ASSERT_TRUE(IsSameDesc(node->Op(), op_desc));
  }
}

}  // namespace details
}  // namespace framework
}  // namespace paddle
