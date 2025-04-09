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
#include "paddle/fluid/framework/details/multi_devices_helper.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/operator.h"
#include "paddle/fluid/framework/program_desc.h"

namespace paddle::framework {

class NOP : public OperatorBase {
 public:
  NOP(const std::string &type,
      const VariableNameMap &inputs,
      const VariableNameMap &outputs,
      const AttributeMap &attrs)
      : OperatorBase(type, inputs, outputs, attrs) {}

 private:
  void RunImpl(const Scope &scope, const phi::Place &place) const override {}
};

class SumOpMaker : public OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X", "").AsDuplicable();
    AddOutput("Out", "").AsDuplicable();
    AddComment("");
  }
};

class SumOpVarTypeInference : public VarTypeInference {
 public:
  void operator()(InferVarTypeContext *ctx) const override {
    auto default_var_type = proto::VarType::SELECTED_ROWS;

    if (ctx->InputTypeAnyOf("X", proto::VarType::DENSE_TENSOR)) {
      default_var_type = proto::VarType::DENSE_TENSOR;
    }

    ctx->SetOutputType("Out", default_var_type);
  }
};

class DummyOpMaker : public OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X", "").AsDuplicable();
    AddOutput("Out", "").AsDuplicable();
    AddComment("");
  }
};

class DummyOpVarTypeInference : public VarTypeInference {
 public:
  void operator()(framework::InferVarTypeContext *ctx) const override {}
};
}  // namespace paddle::framework

REGISTER_OPERATOR(fake_sum,
                  paddle::framework::NOP,
                  paddle::framework::SumOpMaker,
                  paddle::framework::SumOpVarTypeInference);
REGISTER_OPERATOR(dummy,
                  paddle::framework::NOP,
                  paddle::framework::SumOpMaker,
                  paddle::framework::SumOpVarTypeInference);
REGISTER_OPERATOR(sum_without_infer_var_type,
                  paddle::framework::NOP,
                  paddle::framework::SumOpMaker);

namespace paddle::framework {

TEST(GraphTest, Basic) {
  ProgramDesc prog;
  auto *op = prog.MutableBlock(0)->AppendOp();
  op->SetType("fake_sum");
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

  prog.MutableBlock(0)->Var("test_b")->SetType(proto::VarType::DENSE_TENSOR);
  op->InferVarType(prog.MutableBlock(0));
  ASSERT_EQ(proto::VarType::DENSE_TENSOR,
            prog.MutableBlock(0)->Var("test_out")->GetType());

  std::unique_ptr<ir::Graph> g(new ir::Graph(prog));
  std::vector<ir::Node *> nodes(g->Nodes().begin(), g->Nodes().end());
  for (ir::Node *n : nodes) {
    if (n->Name() == "fake_sum") {
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

TEST(GraphTest, TestInterfaceConvertAllBlocks) {
  // Set FLAGS_convert_all_blocks to true to make sure this test works.
  bool flag_temp = FLAGS_convert_all_blocks;
  FLAGS_convert_all_blocks = true;

  ProgramDesc prog;
  prog.MutableBlock(0)->Var("init_var")->SetType(proto::VarType::SELECTED_ROWS);
  ir::Graph g(prog);
  ASSERT_TRUE(g.IsMainGraph());

  const std::string kIntValue = "int_value";
  const int INT_VALUE = 3;
  g.Set<int>(kIntValue, new int(INT_VALUE));
  ASSERT_TRUE(g.Has(kIntValue));
  ASSERT_EQ(g.GetOrInit<int>(kIntValue), INT_VALUE);
  ASSERT_EQ(g.Get<int>(kIntValue), INT_VALUE);
  g.Erase(kIntValue);
  ASSERT_TRUE(!g.Has(kIntValue));
  g.SetNotOwned<int>(kIntValue, new int(INT_VALUE));
  ASSERT_TRUE(g.Has(kIntValue));
  g.Erase(kIntValue);

  g.ReleaseNodes();
  ASSERT_EQ(g.Nodes().size(), 0UL);
  g.CreateVarNode(new VarDesc("temp_var_desc_name"));
  g.CreateOpNode(prog.MutableBlock(0)->AppendOp());
  g.CreateControlDepVar();
  g.CreateEmptyNode("temp_empty_node_name", ir::Node::Type::kVariable);
  ASSERT_EQ(g.Nodes().size(), 4UL);
  g.RemoveNode(g.RetrieveNode(1));
  ASSERT_EQ(g.Nodes().size(), 3UL);

  // Recover FLAGS_convert_all_blocks.
  FLAGS_convert_all_blocks = flag_temp;
}

TEST(GraphTest, TestMultiBlock) {
  // Set FLAGS_convert_all_blocks to true to make sure this test works.
  bool flag_temp = FLAGS_convert_all_blocks;
  FLAGS_convert_all_blocks = true;

  // Step1: Build a program with 3 blocks.
  ProgramDesc prog;
  ASSERT_EQ(prog.Size(), 1UL);
  prog.AppendBlock(prog.Block(0));
  prog.AppendBlock(prog.Block(0));
  ASSERT_EQ(prog.Size(), 3UL);

  // Set contents in block_0.
  auto *op = prog.MutableBlock(0)->AppendOp();
  op->SetType("fake_sum");
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

  prog.MutableBlock(0)->Var("test_b")->SetType(proto::VarType::DENSE_TENSOR);
  op->InferVarType(prog.MutableBlock(0));
  ASSERT_EQ(proto::VarType::DENSE_TENSOR,
            prog.MutableBlock(0)->Var("test_out")->GetType());

  // Set contents in block_1.
  op = prog.MutableBlock(1)->AppendOp();
  op->SetType("fake_sum");
  op->SetInput("X", {"a"});
  op->SetOutput("Out", {"b"});
  op->SetAttr("op_role", 1);

  op = prog.MutableBlock(1)->AppendOp();
  op->SetType("dummy");
  op->SetInput("X", {"c"});
  op->SetOutput("Out", {"d"});
  op->SetAttr("op_role", 1);

  prog.MutableBlock(1)->Var("a")->SetType(proto::VarType::DENSE_TENSOR);
  prog.MutableBlock(1)->Var("b")->SetType(proto::VarType::DENSE_TENSOR);
  prog.MutableBlock(1)->Var("c")->SetType(proto::VarType::DENSE_TENSOR);
  prog.MutableBlock(1)->Var("d")->SetType(proto::VarType::DENSE_TENSOR);

  // Set contents in block_2.
  op = prog.MutableBlock(2)->AppendOp();
  op->SetType("fake_sum");
  op->SetInput("X", {"a"});
  op->SetOutput("Out", {"b"});
  op->SetAttr("op_role", 1);

  op = prog.MutableBlock(2)->AppendOp();
  op->SetType("dummy");
  op->SetInput("X", {"c"});
  op->SetOutput("Out", {"d"});
  op->SetAttr("op_role", 1);

  prog.MutableBlock(2)->Var("a")->SetType(proto::VarType::DENSE_TENSOR);
  prog.MutableBlock(2)->Var("b")->SetType(proto::VarType::DENSE_TENSOR);
  prog.MutableBlock(2)->Var("c")->SetType(proto::VarType::DENSE_TENSOR);
  prog.MutableBlock(1)->Var("d")->SetType(proto::VarType::DENSE_TENSOR);

  // Step2: Convert program into graph, 3 blocks corresponding 3 sub_graphs.
  std::unique_ptr<ir::Graph> g(new ir::Graph(prog));
  ASSERT_EQ(g->IsMainGraph(), true);
  ASSERT_EQ(g->SubGraphsSize(), 3UL);

  // Check contents in sub_graph_0.
  const ir::Graph *g0 = g->GetSubGraph(0);
  std::vector<ir::Node *> nodes(g0->Nodes().begin(), g0->Nodes().end());
  for (ir::Node *n : nodes) {
    if (n->Name() == "fake_sum") {
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

  // Check contents in sub_graph_1.
  const ir::Graph *g1 = g->GetSubGraph(1);
  for (ir::Node *n : g1->Nodes()) {
    if (n->Name() == "fake_sum") {
      ASSERT_EQ(n->outputs[0]->Name(), "b");
      ASSERT_EQ(n->outputs.size(), 1UL);
    }
    if (n->Name() == "dummy") {
      ASSERT_EQ(n->inputs[0]->Name(), "c");
      ASSERT_EQ(n->inputs.size(), 1UL);
    }
  }

  // Check contents in sub_graph_2.
  const ir::Graph *g2 = g->GetSubGraph(2);
  for (ir::Node *n : g2->Nodes()) {
    if (n->Name() == "fake_sum") {
      ASSERT_EQ(n->outputs[0]->Name(), "b");
      ASSERT_EQ(n->outputs.size(), 1UL);
    }
    if (n->Name() == "dummy") {
      ASSERT_EQ(n->inputs[0]->Name(), "c");
      ASSERT_EQ(n->inputs.size(), 1UL);
    }
  }

  // Step3: Clone graph.
  std::shared_ptr<ir::Graph> clone_g = g->Clone();
  ASSERT_EQ(clone_g->IsMainGraph(), true);
  ASSERT_EQ(clone_g->SubGraphsSize(), 3UL);

  // Recover FLAGS_convert_all_blocks.
  FLAGS_convert_all_blocks = flag_temp;
}

}  // namespace paddle::framework
