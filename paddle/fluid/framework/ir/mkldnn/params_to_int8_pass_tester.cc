// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/framework/ir/mkldnn/params_to_int8_pass.h"  // NOLINT
#include <gtest/gtest.h>

#include "paddle/fluid/imperative/type_defs.h"
#include "paddle/fluid/platform/place.h"

using LoDTensor = phi::DenseTensor;

namespace paddle {
namespace framework {
namespace ir {

struct TestScope {
  float* CreateTensorInScope(const std::string& var_name) {
    auto variable = scope.Var(var_name);
    auto tensor = variable->GetMutable<LoDTensor>();
    tensor->Resize({1});
    return tensor->mutable_data<float>(place, 1);
  }

  template <typename T>
  const T* GetTensorPtr(const std::string& input) const {
    Variable* var = scope.FindVar(input);
    auto tensor = var->Get<LoDTensor>();
    return tensor.data<T>();
  }

  std::unique_ptr<Graph> CreateGraphFromProgram(const ProgramDesc& program) {
    auto graph = std::make_unique<ir::Graph>(program);
    graph->SetNotOwned(kParamScopeAttr, &scope);
    return graph;
  }

 private:
  Scope scope;
  CPUPlace place;
};

struct ProgramStrategy {
  virtual ~ProgramStrategy() {}

  std::unique_ptr<Graph> CreateGraph() {
    CreateProgram();
    return test_scope.CreateGraphFromProgram(program);
  }

  void CheckGraph(const std::unique_ptr<ir::Graph>& graph) const {
    for (auto* node : graph->Nodes()) {
      if (node->IsOp()) {
        CheckOp(*node->Op());
      }
    }
  }

  virtual void CreateProgram() = 0;
  virtual void CheckOp(const OpDesc& op) const = 0;

 protected:
  TestScope test_scope;
  ProgramDesc program;
};

struct ConvProgramStrategy : public ProgramStrategy {
 protected:
  OpDesc* CreateBasicConvOp() {
    auto op = program.MutableBlock(0)->AppendOp();
    op->SetType("conv2d");
    op->SetAttr("use_mkldnn", true);
    op->SetAttr("name", std::string{"Conv1"});
    op->SetAttr("mkldnn_data_type", std::string{"int8"});
    op->SetAttr("data_format", std::string{"NCHW"});
    op->SetAttr("groups", 1);
    op->SetAttr("dilations", std::vector<int>({1, 1}));
    op->SetAttr("paddings", std::vector<int>({1, 1}));
    op->SetAttr("strides", std::vector<int>({1, 1}));
    op->SetAttr("Scale_weights", std::vector<float>{2});
    op->SetAttr("Scale_in", 1.0f);

    AddInputToOp(op, "Input");
    AddInputToOp(op, "Filter")->SetPersistable(true);
    SetOutput(op, "Output");
    return op;
  }

 protected:
  VarDesc* AddInputToOp(OpDesc* op, std::string input) {
    const std::string var_name = input + "_var";
    op->SetInput(input, {var_name});
    auto var = program.MutableBlock(0)->Var(var_name);
    test_scope.CreateTensorInScope(var_name)[0] = 1.5f;
    return var;
  }

  void SetOutput(OpDesc* op, std::string output) {
    const std::string var_name = output + "_var";
    op->SetOutput("Output", {var_name});
    program.MutableBlock(0)->Var(var_name);
    test_scope.CreateTensorInScope(var_name)[0] = 1.5f;
  }

  void CheckFilterScaleAndDType(const OpDesc& op) const {
    EXPECT_EQ(op.GetAttrIfExists<std::vector<float>>("Scale_weights"),
              std::vector<float>(1, 1));

    auto filter = op.Input("Filter");
    EXPECT_EQ(filter.size(), 1ul);
    const auto* filter_ptr = test_scope.GetTensorPtr<int8_t>(filter[0]);
    ASSERT_NE(filter_ptr, nullptr);
    EXPECT_EQ(filter_ptr[0], int8_t{3});
  }
};

struct ConvWithBiasProgram : public ConvProgramStrategy {
  void CreateProgram() override {
    OpDesc* op = CreateBasicConvOp();
    AddInputToOp(op, "Bias");
    op->SetAttr("Bias_scales", std::vector<float>{2});
  }

  void CheckOp(const OpDesc& op) const override {
    CheckFilterScaleAndDType(op);
    CheckBiasScaleAndDType(op);
  }

  void CheckBiasScaleAndDType(const OpDesc& op) const {
    EXPECT_EQ(op.GetAttrIfExists<std::vector<float>>("Bias_scales"),
              std::vector<float>(1, 1));
    auto bias = op.Input("Bias");
    EXPECT_EQ(bias.size(), 1ul);
    const auto* bias_ptr = test_scope.GetTensorPtr<int32_t>(bias[0]);
    EXPECT_NE(bias_ptr, nullptr);
    EXPECT_EQ(bias_ptr[0], 3);
  }
};

struct ConvWithoutBiasProgram : public ConvProgramStrategy {
  void CreateProgram() override { CreateBasicConvOp(); }

  void CheckOp(const OpDesc& op) const override {
    CheckFilterScaleAndDType(op);
  }
};

template <typename Program>
struct ParamsToInt8PassTest {
  void RunPassTest() {
    graph = program.CreateGraph();

    auto pass = PassRegistry::Instance().Get("params_to_int8_pass");
    graph.reset(pass->Apply(graph.release()));

    program.CheckGraph(graph);
  }

 private:
  std::unique_ptr<Graph> graph;
  Program program;
};

TEST(ParamToInt8Pass, conv_without_bias) {
  ParamsToInt8PassTest<ConvWithoutBiasProgram> test;
  test.RunPassTest();
}

TEST(ParamToInt8Pass, conv_with_bias) {
  ParamsToInt8PassTest<ConvWithBiasProgram> test;
  test.RunPassTest();
}

}  // namespace ir
}  // namespace framework
}  // namespace paddle

USE_PASS(params_to_int8_pass);
