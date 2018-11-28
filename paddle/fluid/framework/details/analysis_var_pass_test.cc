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
#include <iostream>
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
  std::vector<OpDesc*>* all_op_descs =
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

}  // namespace details
}  // namespace framework
}  // namespace paddle
