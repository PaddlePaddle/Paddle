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

#include "paddle/fluid/framework/details/analysis_var_pass.h"
#include <algorithm>
#include <iostream>
#include <iterator>
#include "glog/logging.h"
#include "gtest/gtest.h"
#include "paddle/fluid/framework/ir/graph.h"
#include "paddle/fluid/framework/ir/graph_helper.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/operator.h"
#include "paddle/fluid/framework/program_desc.h"

namespace paddle {
namespace framework {

class DummyOp : public OperatorBase {
 public:
  DummyOp(const std::string& type, const VariableNameMap& inputs,
          const VariableNameMap& outputs, const AttributeMap& attrs)
      : OperatorBase(type, inputs, outputs, attrs) {}

 private:
  void RunImpl(const Scope& scope,
               const platform::Place& place) const override {}
};

class SumOpMaker : public OpProtoAndCheckerMaker {
 public:
  void Make() {
    AddInput("X", "").AsDuplicable();
    AddOutput("Out", "");
    AddComment("");
  }
};

class AssignOpMaker : public OpProtoAndCheckerMaker {
 public:
  void Make() {
    AddInput("X", "").AsDuplicable();
    AddOutput("Out", "");
    AddComment("");
  }
};

class DummyVarTypeInference : public VarTypeInference {
 public:
  void operator()(const OpDesc& op_desc, BlockDesc* block) const override {
    auto& inputs = op_desc.Input("X");
    auto type = block->Var(inputs.front())->GetType();
    auto out_var_name = op_desc.Output("Out").front();
    block->Var(out_var_name)->SetType(type);
  }
};

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

static inline bool IsSameDesc(OpDesc* op1, OpDesc* op2) {
  return op1->Type() == op2->Type() && op1->Inputs() == op2->Inputs() &&
         op1->Outputs() == op2->Outputs();
}

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

template <typename Container>
inline static std::string DebugString(const Container& c) {
  std::stringstream ss;
  for (auto& item : c) {
    ss << item << " ";
  }
  return ss.str();
}

TEST(CFGGraph, IRGraph) {
  // prepare ir graph
  auto prog = FillProgramDesc();
  ir::Graph graph(prog);
  const std::vector<OpDesc*>* all_op_descs =
      new std::vector<OpDesc*>(prog.Block(0).AllOps());
  graph.Set(details::kAllOpDescs, all_op_descs);  // take ownership

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
  const std::vector<OpDesc*>* all_op_descs =
      new std::vector<OpDesc*>(prog.Block(0).AllOps());
  graph.Set(details::kAllOpDescs, all_op_descs);  // take ownership

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
  const std::vector<OpDesc*>* all_op_descs =
      new std::vector<OpDesc*>(prog.Block(0).AllOps());
  graph.Set(details::kAllOpDescs, all_op_descs);  // take ownership
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
  const std::vector<OpDesc*>* all_op_descs =
      new std::vector<OpDesc*>(prog.Block(0).AllOps());
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
  graph.Set(details::kAllOpDescs, all_op_descs);  // take ownership

  auto op_descs = prog.Block(0).AllOps();

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
  const std::vector<OpDesc*>* all_op_descs =
      new std::vector<OpDesc*>(prog.Block(0).AllOps());
  graph.Set(details::kAllOpDescs, all_op_descs);  // take ownership

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

  // remove sum node
  auto op_descs = prog.Block(0).AllOps();
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
  const std::vector<OpDesc*>* all_op_descs =
      new std::vector<OpDesc*>(prog.Block(0).AllOps());
  graph.Set(details::kAllOpDescs, all_op_descs);  // take ownership

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

  auto op_descs = prog.Block(0).AllOps();
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
