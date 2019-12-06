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

#include "paddle/fluid/framework/ir/graph.h"
#include "gtest/gtest.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/operator.h"
#include "paddle/fluid/framework/program_desc.h"

namespace paddle {
namespace framework {

class NOP : public OperatorBase {
 public:
  NOP(const std::string &type, const VariableNameMap &inputs,
      const VariableNameMap &outputs, const AttributeMap &attrs)
      : OperatorBase(type, inputs, outputs, attrs) {}

 private:
  void RunImpl(const Scope &scope,
               const platform::Place &place) const override {}
};

class SumOpMaker : public OpProtoAndCheckerMaker {
 public:
  void Make() {
    AddInput("X", "").AsDuplicable();
    AddOutput("Out", "").AsDuplicable();
    AddComment("");
  }
};

class SumOpVarTypeInference : public VarTypeInference {
 public:
  void operator()(InferVarTypeContext *ctx) const override {
    auto &inputs = ctx->Input("X");
    auto default_var_type = proto::VarType::SELECTED_ROWS;

    bool any_input_is_lod_tensor = std::any_of(
        inputs.begin(), inputs.end(), [&ctx](const std::string &name) {
          return ctx->GetType(name) == proto::VarType::LOD_TENSOR;
        });
    if (any_input_is_lod_tensor) {
      default_var_type = proto::VarType::LOD_TENSOR;
    }

    auto out_var_name = ctx->Output("Out").front();
    ctx->SetType(out_var_name, default_var_type);
  }
};

class DummyOpMaker : public OpProtoAndCheckerMaker {
 public:
  void Make() {
    AddInput("X", "").AsDuplicable();
    AddOutput("Out", "").AsDuplicable();
    AddComment("");
  }
};

class DummyOpVarTypeInference : public VarTypeInference {
 public:
  void operator()(framework::InferVarTypeContext *ctx) const override {}
};
}  // namespace framework
}  // namespace paddle

REGISTER_OPERATOR(sum, paddle::framework::NOP, paddle::framework::SumOpMaker,
                  paddle::framework::SumOpVarTypeInference);
REGISTER_OPERATOR(dummy, paddle::framework::NOP, paddle::framework::SumOpMaker,
                  paddle::framework::SumOpVarTypeInference);
REGISTER_OPERATOR(sum_without_infer_var_type, paddle::framework::NOP,
                  paddle::framework::SumOpMaker);

namespace paddle {
namespace framework {

TEST(GraphTest, Basic) {
  ProgramDesc prog;
  auto *op = prog.MutableBlock(0)->AppendOp();
  op->SetType("sum");
  op->SetInput("X", {"test_a", "test_b", "test_c"});
  op->SetOutput("Out", {"test_out"});
  op->SetAttr("op_role", 1);

  prog.MutableBlock(0)->Var("test_a")->SetType(proto::VarType::SELECTED_ROWS);
  prog.MutableBlock(0)->Var("test_b")->SetType(proto::VarType::SELECTED_ROWS);
  prog.MutableBlock(0)->Var("test_c")->SetType(proto::VarType::SELECTED_ROWS);
  prog.MutableBlock(0)->Var("test_out");

  op->InferVarType(prog.MutableBlock(0));

  ASSERT_EQ(proto::VarType::SELECTED_ROWS,
            prog.MutableBlock(0)->Var("test_out")->GetType());

  prog.MutableBlock(0)->Var("test_b")->SetType(proto::VarType::LOD_TENSOR);
  op->InferVarType(prog.MutableBlock(0));
  ASSERT_EQ(proto::VarType::LOD_TENSOR,
            prog.MutableBlock(0)->Var("test_out")->GetType());

  std::unique_ptr<ir::Graph> g(new ir::Graph(prog));
  std::vector<ir::Node *> nodes(g->Nodes().begin(), g->Nodes().end());
  for (ir::Node *n : nodes) {
    if (n->Name() == "sum") {
      ASSERT_EQ(n->inputs.size(), 3UL);
      ASSERT_EQ(n->outputs.size(), 1UL);
    } else if (n->Name() == "test_a" || n->Name() == "test_b" ||
               n->Name() == "test_c") {
      ASSERT_EQ(n->inputs.size(), 0UL);
      ASSERT_EQ(n->outputs.size(), 1UL);
    } else if (n->Name() == "test_out") {
      ASSERT_EQ(n->inputs.size(), 1UL);
      ASSERT_EQ(n->outputs.size(), 0UL);
    }
  }
  ASSERT_EQ(nodes.size(), 5UL);
}

TEST(GraphTest, WriteAfterRead) {
  // void Test() {
  ProgramDesc prog;
  auto *op = prog.MutableBlock(0)->AppendOp();
  op->SetType("sum");
  op->SetInput("X", {"a"});
  op->SetOutput("Out", {"b"});
  op->SetAttr("op_role", 1);

  op = prog.MutableBlock(0)->AppendOp();
  op->SetType("dummy");
  op->SetInput("X", {"c"});
  op->SetOutput("Out", {"a"});
  op->SetAttr("op_role", 1);

  prog.MutableBlock(0)->Var("a")->SetType(proto::VarType::LOD_TENSOR);
  prog.MutableBlock(0)->Var("b")->SetType(proto::VarType::LOD_TENSOR);
  prog.MutableBlock(0)->Var("c")->SetType(proto::VarType::LOD_TENSOR);

  std::unique_ptr<ir::Graph> g(new ir::Graph(prog));
  ir::Node *control_dep1 = nullptr;
  ir::Node *control_dep2 = nullptr;
  for (ir::Node *n : g->Nodes()) {
    if (n->Name() == "sum") {
      ASSERT_EQ(n->outputs[0]->Name(), "b");
      ASSERT_TRUE(ir::IsControlDepVar(*n->outputs[1]));
      control_dep1 = n->outputs[1];
      ASSERT_EQ(n->outputs.size(), 2UL);
    }
    if (n->Name() == "dummy") {
      ASSERT_EQ(n->inputs[0]->Name(), "c");
      ASSERT_TRUE(ir::IsControlDepVar(*n->inputs[1]));
      control_dep2 = n->inputs[1];
      ASSERT_EQ(n->inputs.size(), 2UL);
    }
  }
  ASSERT_EQ(control_dep1, control_dep2);
}

TEST(GraphTest, WriteAfterWrite) {
  // void Test() {
  ProgramDesc prog;
  auto *op = prog.MutableBlock(0)->AppendOp();
  op->SetType("sum");
  op->SetInput("X", {"a"});
  op->SetOutput("Out", {"b"});
  op->SetAttr("op_role", 1);

  op = prog.MutableBlock(0)->AppendOp();
  op->SetType("dummy");
  op->SetInput("X", {"c"});
  op->SetOutput("Out", {"b"});
  op->SetAttr("op_role", 1);

  prog.MutableBlock(0)->Var("a")->SetType(proto::VarType::LOD_TENSOR);
  prog.MutableBlock(0)->Var("b")->SetType(proto::VarType::LOD_TENSOR);
  prog.MutableBlock(0)->Var("c")->SetType(proto::VarType::LOD_TENSOR);

  std::unique_ptr<ir::Graph> g(new ir::Graph(prog));
  ir::Node *control_dep1 = nullptr;
  ir::Node *control_dep2 = nullptr;
  for (ir::Node *n : g->Nodes()) {
    if (n->Name() == "sum") {
      ASSERT_EQ(n->outputs[0]->Name(), "b");
      ASSERT_TRUE(ir::IsControlDepVar(*n->outputs[1]));
      ASSERT_EQ(n->outputs.size(), 2UL);
      control_dep1 = n->outputs[1];
    }
    if (n->Name() == "dummy") {
      ASSERT_EQ(n->inputs[0]->Name(), "c");
      ASSERT_TRUE(ir::IsControlDepVar(*n->inputs[1]));
      control_dep2 = n->inputs[1];
      ASSERT_EQ(n->inputs.size(), 2UL);
    }
  }
  ASSERT_NE(control_dep1, nullptr);
  ASSERT_NE(control_dep2, nullptr);
  ASSERT_EQ(control_dep1, control_dep2);
}

TEST(GraphTest, TestException) {
  ProgramDesc prog;
  std::unique_ptr<ir::Graph> g(new ir::Graph(prog));

  bool not_met_exception = false;
  try {
    g->Erase("no_attr");
  } catch (const platform::EnforceNotMet &e) {
    not_met_exception = true;
  }
  ASSERT_TRUE(not_met_exception);

  not_met_exception = false;
  try {
    g->CreateVarNode(nullptr);
  } catch (const platform::EnforceNotMet &e) {
    not_met_exception = true;
  }
  ASSERT_TRUE(not_met_exception);

  not_met_exception = false;
  try {
    g->CreateOpNode(nullptr);
  } catch (const platform::EnforceNotMet &e) {
    not_met_exception = true;
  }
  ASSERT_TRUE(not_met_exception);

  not_met_exception = false;
  try {
    g->RemoveNode(nullptr);
  } catch (const platform::EnforceNotMet &e) {
    not_met_exception = true;
  }
  ASSERT_TRUE(not_met_exception);

  not_met_exception = false;
  try {
    g->AddNode(nullptr);
    g->AddNode(nullptr);
  } catch (const platform::EnforceNotMet &e) {
    not_met_exception = true;
  }
  ASSERT_TRUE(not_met_exception);
}
}  // namespace framework
}  // namespace paddle
