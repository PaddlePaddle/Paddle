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
#include <unordered_map>

#include "paddle/fluid/framework/ir/onednn/compute_propagate_scales_onednn_pass.h"
#include "paddle/fluid/framework/naive_executor.h"
#include "paddle/phi/common/place.h"

namespace paddle::framework::ir {

const std::array<float, 10> positive_and_negative_values = {-0.0482659,
                                                            -0.0102493,
                                                            -0.00794221,
                                                            -0.00387115,
                                                            -0.00674586,
                                                            -0.0495346,
                                                            0.0629528,
                                                            -0.00531285,
                                                            -0.0230353,
                                                            0.0269089};

const std::vector<std::vector<float>> wx = {
    {0.04347931, -0.5643393, 0.7551297, 0.26713502, 0.8055306, 0.91144973},
    {0.01707571, 0.12741385, 0.15419468, 0.66127586, 0.46821925, 0.9665961},
    {0.40393898, 0.884427, -0.5853097, 0.5840954, 0.9170512, 0.98245513}};
const std::vector<std::vector<float>> wh = {
    {0.42484227, -0.9025513, 0.17087583, 0.8403284, 0.03325734, 0.92331886},
    {0.32630175, 0.41691914, 0.99848574, 0.3504407, 0.06707559, 0.62239844}};

const std::vector<double> gru_scales = {
    2.35381475, 1.08304947, 1.32427582, 1.19001095, 1.00151656, 1.01785819};

const std::vector<double> lstm_scales = {
    2.35381475, 1.10797026, 1.00151656, 1.19001095, 1.09045166, 1.01785819};

static const std::initializer_list<std::string> conv_variable_names{
    "conv_in", "filter", "bias", "conv_out"};

static const std::initializer_list<std::string> rnn_variable_names{
    "x", "wx", "wh", "b", "h", "c"};

class ComputePropagateScalesMkldnnPassTest : public testing::Test {
 public:
  ComputePropagateScalesMkldnnPassTest() {  // NOLINT
    pass = std::make_unique<ComputePropagateScalesMkldnnPass>();
  }

  std::vector<float> GetScales(phi::DenseTensor* tensor, int axis) const {
    return pass->GetScales(tensor, axis);
  }

  void ComputeVarScales(ir::Graph* graph,
                        Scope* scope,
                        const std::unordered_set<std::string> ops,
                        const std::string& weight_name,
                        const int axis,
                        StringPairMap* var_quant_scales) const {
    pass->ComputeVarScales(
        graph, scope, ops, weight_name, axis, var_quant_scales);
  }

  void ComputeGruWeightScales(ir::Graph* graph,
                              Scope* scope,
                              const std::string& wx_name,
                              const std::string& wh_name,
                              StringPairMap* var_quant_scales) const {
    pass->ComputeGruWeightScales(
        graph, scope, wx_name, wh_name, var_quant_scales);
  }

  void ComputeLstmWeightScales(ir::Graph* graph,
                               Scope* scope,
                               std::string wx_name,
                               std::string wh_name,
                               StringPairMap* var_quant_scales) const {
    pass->ComputeLstmWeightScales(
        graph, scope, wx_name, wh_name, var_quant_scales);
  }

  void UpdateReluOutputScales(ir::Graph* graph,
                              StringPairMap* var_quant_scales) const {
    pass->UpdateReluOutputScales(graph, var_quant_scales);
  }

  void InitTensorHolder(Scope* scope,
                        const phi::Place& place,
                        const std::string& var_name) {
    auto x = scope->Var(var_name);
    auto tensor = x->GetMutable<phi::DenseTensor>();
    auto tensor_size = 1;
    if (var_name == "filter") {
      tensor_size = positive_and_negative_values.size();
    } else if (var_name == "wx") {
      tensor_size = wx.size();
    } else if (var_name == "wh") {
      tensor_size = wh.size();
    }
    tensor->mutable_data(
        place, phi::TransToPhiDataType(proto::VarType::FP32), tensor_size);
  }

  void PrepareGraph(ir::Graph* graph,
                    const ProgramDesc& prog,
                    Scope* scope,
                    const std::initializer_list<std::string>& variable_names) {
    auto place = phi::CPUPlace();
    NaiveExecutor exe{place};
    exe.CreateVariables(prog, 0, true, scope);

    for (auto& v : variable_names) {
      InitTensorHolder(scope, place, v.c_str());
    }
    graph->SetNotOwned(kParamScopeAttr, scope);
  }

  void ComputeRnnWeightScalesTest(const std::string& type,
                                  const framework::ProgramDesc& prog,
                                  std::vector<double> scales) {
    ir::Graph* graph(new ir::Graph(prog));
    Scope scope;

    PrepareGraph(graph, prog, &scope, rnn_variable_names);

    std::string wx_name = "WeightX";
    std::string wh_name = "WeightH";
    std::string wx_var_names = "wx";
    std::string wh_var_names = "wh";

    StringPairMap var_quant_scales;

    auto* wx_var = scope.FindVar(wx_var_names);
    auto* wx_tensor = wx_var->GetMutable<phi::DenseTensor>();
    wx_tensor->Resize(common::make_dim(wx.size(), wx[0].size()));
    for (size_t i = 0; i < wx.size(); i++)
      std::copy(
          begin(wx[i]),
          end(wx[i]),
          wx_tensor->mutable_data<float>(phi::CPUPlace()) + i * wx[0].size());

    auto* wh_var = scope.FindVar(wh_var_names);
    auto* wh_tensor = wh_var->GetMutable<phi::DenseTensor>();
    wh_tensor->Resize(common::make_dim(wh.size(), wh[0].size()));
    for (size_t i = 0; i < wh.size(); i++)
      std::copy(
          begin(wh[i]),
          end(wh[i]),
          wh_tensor->mutable_data<float>(phi::CPUPlace()) + i * wh[0].size());
    if (type == "gru") {  // NOLINT
      ComputeGruWeightScales(
          graph, &scope, wx_name, wh_name, &var_quant_scales);
    } else {
      ComputeLstmWeightScales(
          graph, &scope, wx_name, wh_name, &var_quant_scales);
    }
    bool is_unsigned;
    phi::DenseTensor wx_result_tensor;

    std::tie(is_unsigned, wx_result_tensor) = var_quant_scales[wx_var_names];
    ASSERT_EQ(is_unsigned, false);
    ASSERT_EQ(wx_result_tensor.numel(), static_cast<int64_t>(scales.size()));
    for (int64_t i = 0; i < wx_result_tensor.numel(); i++) {
      ASSERT_FLOAT_EQ(wx_result_tensor.data<float>()[i], scales[i]);
    }
  }

  void UpdateReluOutputScaleTest(
      const framework::ProgramDesc& prog,
      StringPairMap* var_quant_scales,
      const std::initializer_list<std::string>& variable_names) {
    ir::Graph* graph(new ir::Graph(prog));
    Scope scope;

    PrepareGraph(graph, prog, &scope, conv_variable_names);

    UpdateReluOutputScales(graph, var_quant_scales);

    for (auto& var_name : variable_names) {
      auto iter = var_quant_scales->find(var_name);
      ASSERT_NE(iter, var_quant_scales->end());
      ASSERT_EQ((*var_quant_scales)[var_name].first, true);
    }
  }

 private:
  std::unique_ptr<ComputePropagateScalesMkldnnPass> pass;
};

void SetOp(ProgramDesc* prog,
           const std::string& type,
           const std::string& name,
           const std::vector<std::string>& inputs,
           const std::vector<std::string>& outputs,
           const std::unordered_map<std::string, std::string>& attrs = {}) {
  auto* op = prog->MutableBlock(0)->AppendOp();
  op->SetType(type);
  op->SetAttr("use_mkldnn", true);
  op->SetAttr("name", name);
  if (!attrs.empty())
    for (auto& attr : attrs) op->SetAttr(attr.first, attr.second);

  if (type == "conv2d") {
    op->SetInput("Input", {inputs[0]});
    if (inputs.size() > 1) op->SetInput("Filter", {inputs[1]});
    if (inputs.size() > 2) op->SetInput("Bias", {inputs[2]});
    op->SetOutput("Output", {outputs[0]});
  } else if (type == "fusion_gru" || type == "fusion_lstm") {
    op->SetInput("X", {inputs[0]});
    op->SetInput("WeightX", {inputs[1]});
    op->SetInput("WeightH", {inputs[2]});
    op->SetOutput("Hidden", {outputs[0]});
    if (type == "fusion_lstm") op->SetOutput("Cell", {outputs[1]});
  }
}

ProgramDesc BuildConv2dProgramDesc() {
  ProgramDesc prog;
  for (auto& v : conv_variable_names) {
    prog.MutableBlock(0)->Var(v);
  }
  SetOp(&prog, "conv2d", "Conv2d", {"conv_in", "filter", "bias"}, {"conv_out"});

  return prog;
}

ProgramDesc BuildConv2dReluProgramDesc() {
  ProgramDesc prog;
  for (auto& v : conv_variable_names) {
    prog.MutableBlock(0)->Var(v);
  }
  std::unordered_map<std::string, std::string> attrs = {
      {"fuse_activation", "relu"}};
  SetOp(&prog,
        "conv2d",
        "Conv2d",
        {"conv_in", "filter", "bias"},
        {"conv_out"},
        attrs);

  return prog;
}

ProgramDesc BuildFusionGruProgramDesc() {
  ProgramDesc prog;
  for (auto& v : rnn_variable_names) {
    prog.MutableBlock(0)->Var(v);
  }
  SetOp(&prog, "fusion_gru", "Fusion_gru", {"x", "wx", "wh"}, {"h"});

  return prog;
}

ProgramDesc BuildFusionLstmProgramDesc() {
  ProgramDesc prog;
  for (auto& v : rnn_variable_names) {
    prog.MutableBlock(0)->Var(v);
  }
  SetOp(&prog, "fusion_lstm", "Fusion_lstm", {"x", "wx", "wh"}, {"h", "c"});

  return prog;
}

TEST_F(ComputePropagateScalesMkldnnPassTest, get_scales_function) {
  const auto& values = positive_and_negative_values;
  float max_val = *std::max_element(values.begin(), values.end());

  phi::DenseTensor var_tensor;
  var_tensor.Resize(common::make_dim(values.size(), 1));
  std::copy(begin(values),
            end(values),
            var_tensor.mutable_data<float>(phi::CPUPlace()));
  std::vector<float> results = GetScales(&var_tensor, 0);

  ASSERT_EQ(results.size(), std::size_t(1));
  ASSERT_EQ(results[0], (1.f / max_val));
}

TEST_F(ComputePropagateScalesMkldnnPassTest, compute_var_scales) {
  auto prog = BuildConv2dProgramDesc();
  const auto& values = positive_and_negative_values;
  ir::Graph* graph(new ir::Graph(prog));
  Scope scope;

  PrepareGraph(graph, prog, &scope, conv_variable_names);

  std::initializer_list<std::string> ops = {"conv2d", "depthwise_conv2d"};
  std::string weight_name = "Filter";
  std::string weight_var_name = "filter";

  auto axis = 1;
  StringPairMap var_quant_scales;

  auto* var = scope.FindVar(weight_var_name);
  auto* weight_tensor = var->GetMutable<phi::DenseTensor>();
  weight_tensor->Resize(common::make_dim(1, values.size()));
  std::copy(begin(values),
            end(values),
            weight_tensor->mutable_data<float>(phi::CPUPlace()));

  auto max_val = *std::max_element(values.begin(), values.end());

  ComputeVarScales(graph, &scope, ops, weight_name, axis, &var_quant_scales);

  bool is_unsigned;
  phi::DenseTensor result_tensor;

  std::tie(is_unsigned, result_tensor) = var_quant_scales[weight_var_name];

  ASSERT_EQ(is_unsigned, false);
  ASSERT_EQ(result_tensor.numel(), 1);
  ASSERT_FLOAT_EQ(result_tensor.data<float>()[0], (1.0 / max_val));
}

TEST_F(ComputePropagateScalesMkldnnPassTest, compute_gru_weight_scales) {
  ComputeRnnWeightScalesTest("gru", BuildFusionGruProgramDesc(), gru_scales);
}

TEST_F(ComputePropagateScalesMkldnnPassTest, compute_lstm_weight_scales) {
  ComputeRnnWeightScalesTest("lstm", BuildFusionLstmProgramDesc(), lstm_scales);
}

TEST_F(ComputePropagateScalesMkldnnPassTest, update_relu_output_scales) {
  StringPairMap var_quant_scales;
  for (auto& var_name : conv_variable_names) {
    phi::DenseTensor tensor;
    auto* data = tensor.mutable_data<float>({1}, phi::CPUPlace());
    data[0] = 10;
    auto pair = std::make_pair(false, tensor);
    var_quant_scales.insert(std::make_pair(var_name, pair));
  }
  UpdateReluOutputScaleTest(
      BuildConv2dReluProgramDesc(), &var_quant_scales, {"conv_out"});
}

}  // namespace paddle::framework::ir
