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

#include <gtest/gtest.h>

#include "paddle/fluid/framework/ir/mkldnn/params_quantization_mkldnn_pass.h"  // NOLINT
#include "paddle/fluid/imperative/type_defs.h"
#include "paddle/phi/common/place.h"

namespace paddle {
namespace framework {
namespace ir {
namespace {
struct Data {
  Data() = default;

  Data(std::vector<int64_t>&& data_shape, std::vector<float>&& raw_data)
      : shape(std::move(data_shape)), data(std::move(raw_data)) {
    auto size_from_shape = std::accumulate(
        shape.begin(), shape.end(), 1, std::multiplies<int64_t>());
    PADDLE_ENFORCE_EQ(size_from_shape,
                      data.size(),
                      platform::errors::InvalidArgument(
                          "Shape size doesn't match data size."));
  }

  const std::vector<int64_t>& getShape() const { return shape; }
  const std::vector<float>& getData() const { return data; }

 private:
  const std::vector<int64_t> shape;
  const std::vector<float> data;
};

struct TestScope {
  void CreateTensor(const std::string& var_name, const Data& data) {
    auto variable = scope.Var(var_name);
    auto tensor = variable->GetMutable<phi::DenseTensor>();
    tensor->Resize(phi::make_ddim(data.getShape()));
    auto dptr = tensor->mutable_data<float>(place);
    std::copy(data.getData().begin(), data.getData().end(), dptr);
  }

  const phi::DenseTensor& GetTensor(const std::string& input) const {
    Variable* var = scope.FindVar(input);
    return var->Get<phi::DenseTensor>();
  }

  framework::Scope* Scope() { return &scope; }

 private:
  framework::Scope scope;
  CPUPlace place;
};

struct ProgramStrategy {
  virtual ~ProgramStrategy() {}

  std::unique_ptr<Graph> CreateGraph() {
    CreateProgram();
    auto graph = std::make_unique<ir::Graph>(program);
    graph->SetNotOwned(kParamScopeAttr, test_scope.Scope());
    return graph;
  }

  void CheckGraph(const std::unique_ptr<ir::Graph>& graph) const {
    for (auto* node : graph->Nodes()) {
      if (node->IsOp()) {
        CheckOp(*node->Op());
      }
    }
  }

 protected:
  virtual void CreateProgram() = 0;

  virtual void CheckOp(const OpDesc& op) const = 0;

  VarDesc* AddInput(OpDesc* op,
                    std::string input_name,
                    const Data& data,
                    const std::string user_var_name = "") {
    std::string var_name = user_var_name;
    if (var_name.empty()) {
      var_name = input_name + "_var";
    }
    op->SetInput(input_name, {var_name});
    auto var = program.MutableBlock(0)->Var(var_name);
    var->SetShape(data.getShape());
    test_scope.CreateTensor(var_name, data);
    return var;
  }

  void AddOutput(OpDesc* op,
                 std::string output_name,
                 const Data& data,
                 const std::string user_var_name = "") {
    std::string var_name = user_var_name;
    if (var_name.empty()) {
      var_name = output_name + "_var";
    }
    op->SetOutput(output_name, {var_name});
    program.MutableBlock(0)->Var(var_name);
    test_scope.CreateTensor(var_name, data);
  }

 protected:
  TestScope test_scope;
  ProgramDesc program;
};

struct ConvProgramStrategy : public ProgramStrategy {
  ConvProgramStrategy(Data&& input,
                      Data&& filter,
                      Data&& output,
                      std::vector<float>&& scale_weights,
                      int groups = 1,
                      Data&& bias = Data(),
                      std::vector<float>&& scale_bias = {},
                      bool share_weight = false)
      : input(std::move(input)),
        filter(std::move(filter)),
        output(std::move(output)),
        scale_weights(std::move(scale_weights)),
        groups(std::move(groups)),
        bias(std::move(bias)),
        scale_bias(std::move(scale_bias)),
        share_weight(std::move(share_weight)) {}

 protected:
  OpDesc* CreateBasicConvOp(const std::string conv_name = "Conv1") {
    auto op = program.MutableBlock(0)->AppendOp();
    op->SetType("conv2d");
    op->SetAttr("use_mkldnn", true);
    op->SetAttr("name", conv_name);
    op->SetAttr("mkldnn_data_type", std::string{"int8"});
    op->SetAttr("data_format", std::string{"NCHW"});
    op->SetAttr("dilations", std::vector<int>({1, 1}));
    op->SetAttr("paddings", std::vector<int>({1, 1}));
    op->SetAttr("strides", std::vector<int>({1, 1}));
    return op;
  }

 protected:
  void CreateProgram() override {
    OpDesc* op = CreateBasicConvOp();
    AddInput(op, "Input", input);
    AddInput(op, "Filter", filter)->SetPersistable(true);
    AddOutput(op, "Output", output);

    op->SetAttr("Scale_weights", scale_weights);
    op->SetAttr("Scale_in", 1.0f);
    op->SetAttr("groups", groups);

    if (HasBias()) {
      AddInput(op, "Bias", bias);
      op->SetAttr("Bias_scales", scale_bias);
    }

    if (share_weight) {
      OpDesc* op2 = CreateBasicConvOp("Conv2");
      AddInput(op2, "Input", input);
      AddInput(op2, "Filter", filter)->SetPersistable(true);
      AddOutput(op2, "Output", output, "output2");
      op2->SetAttr("Scale_weights", scale_weights);
      op2->SetAttr("Scale_in", 1.0f);
      op2->SetAttr("groups", groups);
      if (HasBias()) {
        AddInput(op2, "Bias", bias, "Bias2");
        op2->SetAttr("Bias_scales", scale_bias);
      }
    }
  }

  void CheckOp(const OpDesc& op) const override {
    CheckFilter(op);
    if (HasBias()) {
      CheckBias(op);
    }
  }

  bool HasBias() const { return !bias.getData().empty(); }

  void CheckFilter(const OpDesc& op) const {
    EXPECT_EQ(op.GetAttrIfExists<std::vector<float>>("Scale_weights"),
              std::vector<float>(1, 1));

    auto filter_inputs = op.Input("Filter");
    ASSERT_EQ(filter_inputs.size(), 1ul);

    auto tensor = test_scope.GetTensor(filter_inputs[0]);
    ASSERT_EQ(tensor.dtype(), phi::DataType::INT8);

    auto filter_ptr = tensor.data<int8_t>();
    ASSERT_NE(filter_ptr, nullptr);
    auto length = tensor.numel() / scale_weights.size();
    for (int64_t i = 0; i < tensor.numel(); i++) {
      EXPECT_EQ(filter_ptr[i],
                static_cast<int8_t>(std::round(filter.getData()[i] *
                                               scale_weights[i / length])));
    }
  }

  void CheckBias(const OpDesc& op) const {
    EXPECT_EQ(op.GetAttrIfExists<std::vector<float>>("Bias_scales"),
              std::vector<float>(1, 1));

    auto bias_inputs = op.Input("Bias");
    ASSERT_EQ(bias_inputs.size(), 1ul);

    auto tensor = test_scope.GetTensor(bias_inputs[0]);
    auto bias_ptr = tensor.data<int32_t>();
    ASSERT_NE(bias_ptr, nullptr);
    auto length = tensor.numel() / scale_bias.size();
    for (int64_t i = 0; i < tensor.numel(); i++) {
      EXPECT_EQ(bias_ptr[i],
                static_cast<int32_t>(
                    std::round(bias.getData()[i] * scale_bias[i / length])));
    }
  }

 private:
  const Data input;
  const Data filter;
  const Data output;
  const std::vector<float> scale_weights;
  const int groups;
  const Data bias;
  const std::vector<float> scale_bias;
  const bool share_weight;
};

struct ParamsQuantizationMkldnnPassTestFixture : public ::testing::Test {
  void RunPassTest(std::unique_ptr<ProgramStrategy> program) {
    auto graph = program->CreateGraph();

    auto pass = PassRegistry::Instance().Get("params_quantization_mkldnn_pass");
    graph.reset(pass->Apply(graph.release()));

    program->CheckGraph(graph);
  }
};

Data GenericInput() { return Data({1, 4, 1, 1}, {1.5f, 1.5f, 1.5f, 1.5f}); }
Data GenericOutput() { return GenericInput(); }

TEST_F(ParamsQuantizationMkldnnPassTestFixture, conv_without_bias_o1i1h1w1) {
  auto program =
      std::make_unique<ConvProgramStrategy>(GenericInput(),
                                            Data({1, 1, 1, 1}, {1.5f}),
                                            GenericOutput(),
                                            std::vector<float>{2.f});
  RunPassTest(std::move(program));
}

TEST_F(ParamsQuantizationMkldnnPassTestFixture, conv_without_bias_2o1i1h1w) {
  auto program =
      std::make_unique<ConvProgramStrategy>(GenericInput(),
                                            Data({2, 1, 1, 1}, {1.5f, 1.5f}),
                                            GenericOutput(),
                                            std::vector<float>{2.f, 4.f});
  RunPassTest(std::move(program));
}

TEST_F(ParamsQuantizationMkldnnPassTestFixture, conv_without_bias_2o2i2h2w) {
  auto program =
      std::make_unique<ConvProgramStrategy>(GenericInput(),
                                            Data({2, 2, 2, 2},
                                                 {1.5f,
                                                  1.5f,
                                                  1.5f,
                                                  1.5f,
                                                  1.5f,
                                                  1.5f,
                                                  1.5f,
                                                  1.5f,
                                                  1.5f,
                                                  1.5f,
                                                  1.5f,
                                                  1.5f,
                                                  1.5f,
                                                  1.5f,
                                                  1.5f,
                                                  1.5f}),
                                            GenericOutput(),
                                            std::vector<float>{2.f, 4.f});
  RunPassTest(std::move(program));
}

TEST_F(ParamsQuantizationMkldnnPassTestFixture, conv_without_bias_2g2o2i1h1w) {
  auto program = std::make_unique<ConvProgramStrategy>(
      GenericInput(),
      Data({2, 2, 2, 1, 1}, {1.5f, 1.5f, 1.5f, 1.5f, 1.5f, 1.5f, 1.5f, 1.5f}),
      GenericOutput(),
      std::vector<float>{2.f, 2.f, 2.f, 2.f},
      2);
  RunPassTest(std::move(program));
}

TEST_F(ParamsQuantizationMkldnnPassTestFixture, conv_without_bias_2g2o1i1h1w) {
  auto program = std::make_unique<ConvProgramStrategy>(
      GenericInput(),
      Data({2, 2, 1, 1, 1}, {1.5f, 1.5f, 1.5f, 1.5f}),
      GenericOutput(),
      std::vector<float>{2.f, 2.f, 2.f, 2.f},
      2);
  RunPassTest(std::move(program));
}

TEST_F(ParamsQuantizationMkldnnPassTestFixture, conv_with_bias_1o1i1h1w) {
  auto program =
      std::make_unique<ConvProgramStrategy>(GenericInput(),
                                            Data({1, 1, 1, 1}, {1.5f}),
                                            GenericOutput(),
                                            std::vector<float>{2.f},
                                            1,
                                            Data({1, 1, 1, 1}, {1.5f}),
                                            std::vector<float>{2.f});
  RunPassTest(std::move(program));
}

TEST_F(ParamsQuantizationMkldnnPassTestFixture, conv_with_bias_2o1i1h1w) {
  auto program =
      std::make_unique<ConvProgramStrategy>(GenericInput(),
                                            Data({2, 1, 1, 1}, {1.5f, 1.5f}),
                                            GenericOutput(),
                                            std::vector<float>{2.f, 4.f},
                                            1,
                                            Data({2, 1, 1, 1}, {1.5f, 1.5f}),
                                            std::vector<float>{2.f, 4.f});
  RunPassTest(std::move(program));
}

TEST_F(ParamsQuantizationMkldnnPassTestFixture, conv_with_bias_2g2o1i1h1w) {
  auto program = std::make_unique<ConvProgramStrategy>(
      GenericInput(),
      Data({4, 1, 1, 1}, {1.5f, 1.5f, 1.5f, 1.5f}),
      GenericOutput(),
      std::vector<float>{2.f, 2.f, 4.f, 4.f},
      2,
      Data({4, 1, 1, 1}, {1.5f, 1.5f, 1.5f, 1.5f}),
      std::vector<float>{2.f, 2.f, 4.f, 4.f});
  RunPassTest(std::move(program));
}

TEST_F(ParamsQuantizationMkldnnPassTestFixture, conv_with_bias_2g2o2i1h1w) {
  auto program = std::make_unique<ConvProgramStrategy>(
      GenericInput(),
      Data({2, 2, 2, 1, 1}, {1.5f, 1.5f, 1.5f, 1.5f, 1.5f, 1.5f, 1.5f, 1.5f}),
      GenericOutput(),
      std::vector<float>{2.f, 2.f, 4.f, 4.f},
      2,
      Data({2, 2, 1, 1, 1}, {1.5f, 1.5f, 1.5f, 1.5f}),
      std::vector<float>{2.f, 2.f, 4.f, 4.f});
  RunPassTest(std::move(program));
}

TEST_F(ParamsQuantizationMkldnnPassTestFixture, conv_with_bias_2g2o2i1h1ws) {
  auto program = std::make_unique<ConvProgramStrategy>(
      GenericInput(),
      Data({2, 2, 2, 1, 1}, {1.5f, 1.5f, 1.5f, 1.5f, 1.5f, 1.5f, 1.5f, 1.5f}),
      GenericOutput(),
      std::vector<float>{2.f, 2.f, 4.f, 4.f},
      2,
      Data({2, 2, 1, 1, 1}, {1.5f, 1.5f, 1.5f, 1.5f}),
      std::vector<float>{2.f, 2.f, 4.f, 4.f},
      true);
  RunPassTest(std::move(program));
}

}  // namespace
}  // namespace ir
}  // namespace framework
}  // namespace paddle

USE_PASS(params_quantization_mkldnn_pass);
