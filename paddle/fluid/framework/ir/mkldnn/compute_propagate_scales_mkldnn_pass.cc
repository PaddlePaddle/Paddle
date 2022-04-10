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

#include <float.h>
#include <algorithm>

#include "paddle/fluid/framework/ir/graph_helper.h"
#include "paddle/fluid/framework/ir/mkldnn/compute_propagate_scales_mkldnn_pass.h"
#include "paddle/fluid/framework/ir/mkldnn/mkldnn_pass_util.h"
#include "paddle/fluid/framework/op_version_registry.h"

namespace paddle {
namespace framework {
namespace ir {

void ComputePropagateScalesMkldnnPass::GetTensorFromVector(
    const std::vector<float>& data_v, Tensor* tensor) const {
  const int size = static_cast<int>(data_v.size());
  auto* data = tensor->mutable_data<float>({size}, platform::CPUPlace());
  for (int i = 0; i < size; i++) {
    data[i] = data_v[i];
  }
}

void ComputePropagateScalesMkldnnPass::GetQuantInfo(
    ir::Graph* graph, StringPairMap* var_quant_scales) const {
  std::unordered_map<std::string, std::vector<float>> info_map{};
  GetInfoFromTheFirstOp(graph, "has_quant_info", "var_quant_scales", &info_map);

  for (auto iter = info_map.begin(); iter != info_map.end(); iter++) {
    Tensor tensor;
    GetTensorFromVector(iter->second, &tensor);
    auto pair = std::make_pair(false, tensor);
    var_quant_scales->insert(std::make_pair(iter->first, pair));
  }
}

std::vector<float> ComputePropagateScalesMkldnnPass::GetScales(Tensor* tensor,
                                                               int axis) const {
  PADDLE_ENFORCE_LT(axis, 2,
                    platform::errors::InvalidArgument(
                        "The input axis is required to be less than 2."));
  auto* data = tensor->data<float>();
  const auto dims = tensor->dims();
  PADDLE_ENFORCE_EQ(dims.size(), 2,
                    platform::errors::InvalidArgument(
                        "The input tensor's rank is required to be 2."));

  const int rows = dims.at(0);
  const int columns = dims.at(1);
  std::vector<float> scales;
  if (axis == 0) {
    for (int i = 0; i < columns; i++) {
      float max_value = FLT_MIN;
      for (int j = 0; j < rows; j++) {
        max_value = std::max(max_value, std::abs(data[i + j * columns]));
      }
      max_value = 1.0 / max_value;
      if (std::isinf(max_value) || std::isnan(max_value)) {
        max_value = 0.0;
      }
      scales.push_back(max_value);
    }
  } else {
    for (int i = 0; i < rows; i++) {
      float max_value = FLT_MIN;
      for (int j = i * columns; j < (i + 1) * columns; j++) {
        max_value = std::max(max_value, std::abs(data[j]));
      }
      max_value = 1.0 / max_value;
      if (std::isinf(max_value) || std::isnan(max_value)) {
        max_value = 0.0;
      }
      scales.push_back(max_value);
    }
  }
  return scales;
}

void ComputePropagateScalesMkldnnPass::ComputeVarScales(
    ir::Graph* graph, Scope* scope, const std::unordered_set<std::string>& ops,
    const std::string& weight_name, const int axis,
    StringPairMap* var_quant_scales) const {
  for (auto* op_node :
       ir::TopologyVarientSort(*graph, static_cast<ir::SortKind>(0))) {
    if (!op_node->IsOp()) continue;

    auto* op_desc = op_node->Op();
    if (ops.count(op_desc->Type())) {
      auto var_name = op_desc->Input(weight_name)[0];
      auto* var = scope->FindVar(var_name);
      PADDLE_ENFORCE_NOT_NULL(
          var, platform::errors::NotFound(
                   "The input persistable var [%s] of [%s] op is not found.",
                   var_name, op_desc->Type()));
      auto* weight_tensor = var->GetMutable<LoDTensor>();
      const auto dims = weight_tensor->dims();
      int volume = 1;
      for (int i = 1; i < dims.size(); i++) {
        volume *= dims[i];
      }

      Tensor tmp_tensor;
      std::vector<int64_t> reshape_dims = {dims[0], volume};
      tmp_tensor.Resize(phi::make_ddim(reshape_dims));
      auto* weight_data = weight_tensor->data<float>();
      auto* tmp_data = tmp_tensor.mutable_data<float>(platform::CPUPlace());
      for (int i = 0; i < weight_tensor->numel(); i++) {
        tmp_data[i] = std::abs(weight_data[i]);
      }

      auto scales_v = GetScales(&tmp_tensor, axis);
      Tensor tensor;
      GetTensorFromVector(scales_v, &tensor);
      auto pair = std::make_pair(false, tensor);
      var_quant_scales->insert(std::make_pair(var_name, pair));
    }
  }
}

void ComputePropagateScalesMkldnnPass::ComputeSingleGruWeightScales(
    Scope* scope, const std::string& wx_var_name,
    const std::string& wh_var_name, Tensor* tensor) const {
  auto* wx_var = scope->FindVar(wx_var_name);
  PADDLE_ENFORCE_NOT_NULL(
      wx_var, platform::errors::NotFound(
                  "The input persistable var [%s] is not found.", wx_var_name));
  auto* wh_var = scope->FindVar(wh_var_name);
  PADDLE_ENFORCE_NOT_NULL(
      wh_var, platform::errors::NotFound(
                  "The input persistable var [%s] is not found.", wh_var_name));

  const auto* wx_tensor = wx_var->GetMutable<LoDTensor>();
  const auto* wh_tensor = wh_var->GetMutable<LoDTensor>();
  const int OC = wh_tensor->dims()[0];
  std::vector<float> scale_ur(2 * OC);
  std::vector<float> scale_o(OC);
  for (int row_id = 0; row_id < wx_tensor->dims()[0]; row_id++) {
    for (int col_id = 0; col_id < 2 * OC; col_id++) {
      int idx = (row_id * wx_tensor->dims()[1]) + col_id;
      auto abs_value = std::abs(wx_tensor->data<float>()[idx]);
      if (row_id == 0) {
        scale_ur[col_id] = abs_value;
      } else {
        if (abs_value > scale_ur[col_id]) scale_ur[col_id] = abs_value;
      }
    }
  }

  for (int i = 0; i < 2 * OC * OC; i++) {
    int col_id = i % (2 * OC);
    auto abs_value = std::abs(wh_tensor->data<float>()[i]);
    if (abs_value > scale_ur[col_id]) scale_ur[col_id] = abs_value;
  }

  for (int row_id = 0; row_id < wx_tensor->dims()[0]; row_id++) {
    for (int col_id = 2 * OC; col_id < wx_tensor->dims()[1]; col_id++) {
      int idx = (row_id * wx_tensor->dims()[1]) + col_id;
      auto abs_value = std::abs(wx_tensor->data<float>()[idx]);
      if (row_id == 0) {
        scale_o[col_id % OC] = abs_value;
      } else {
        if (abs_value > scale_o[col_id]) scale_o[col_id % OC] = abs_value;
      }
    }
  }

  for (int i = 2 * OC * OC; i < OC * wh_tensor->dims()[1]; i++) {
    int col_id = i % OC;
    auto abs_value = std::abs(wh_tensor->data<float>()[i]);
    if (abs_value > scale_o[col_id]) scale_o[col_id] = abs_value;
  }

  scale_ur.insert(scale_ur.end(), scale_o.begin(), scale_o.end());
  transform(scale_ur.begin(), scale_ur.end(), scale_ur.begin(),
            [](float c) { return 1 / c; });
  GetTensorFromVector(scale_ur, tensor);
}

void ComputePropagateScalesMkldnnPass::ComputeGruWeightScales(
    ir::Graph* graph, Scope* scope, const std::string& wx_name,
    const std::string& wh_name, StringPairMap* var_quant_scales) const {
  for (auto* op_node :
       ir::TopologyVarientSort(*graph, static_cast<ir::SortKind>(0))) {
    if (!op_node->IsOp()) continue;

    auto* op_desc = op_node->Op();
    if (op_desc->Type() == "fusion_gru" || op_desc->Type() == "multi_gru") {
      auto wx_var_names = op_desc->Input(wx_name);
      auto wh_var_names = op_desc->Input(wh_name);
      const int wx_names_size = static_cast<int>(wx_var_names.size());
      const int wh_names_size = static_cast<int>(wh_var_names.size());
      PADDLE_ENFORCE_EQ(
          wx_names_size, wh_names_size,
          platform::errors::Fatal("Mismatch in number of weights inputs (%d "
                                  "for WeightX vs. %d for WeightH).",
                                  wx_names_size, wh_names_size));
      for (int i = 0; i < wx_names_size; i++) {
        auto wh_var_name = wh_var_names[i];
        auto wx_var_name = wx_var_names[i];
        Tensor tensor;
        ComputeSingleGruWeightScales(scope, wx_var_name, wh_var_name, &tensor);
        auto pair = std::make_pair(false, tensor);
        var_quant_scales->insert(std::make_pair(wx_var_name, pair));
      }
    }
  }
}

void ComputePropagateScalesMkldnnPass::ComputeSingleLstmWeightScales(
    Scope* scope, const std::string& wx_var_name,
    const std::string& wh_var_name, Tensor* tensor) const {
  auto* wx_var = scope->FindVar(wx_var_name);
  PADDLE_ENFORCE_NOT_NULL(
      wx_var, platform::errors::NotFound(
                  "The input persistable var [%s] is not found.", wx_var_name));
  auto* wh_var = scope->FindVar(wh_var_name);
  PADDLE_ENFORCE_NOT_NULL(
      wh_var, platform::errors::NotFound(
                  "The input persistable var [%s] is not found.", wh_var_name));

  const auto* wx_tensor = wx_var->GetMutable<LoDTensor>();
  const auto* wh_tensor = wh_var->GetMutable<LoDTensor>();
  std::vector<float> scale(wx_tensor->dims()[1]);

  for (int row_id = 0; row_id < wx_tensor->dims()[0]; row_id++) {
    for (int col_id = 0; col_id < wx_tensor->dims()[1]; col_id++) {
      int idx = (row_id * wx_tensor->dims()[1]) + col_id;
      auto abs_value = std::abs(wx_tensor->data<float>()[idx]);
      if (row_id == 0) {
        scale[col_id] = abs_value;
      } else {
        if (abs_value > scale[col_id]) scale[col_id] = abs_value;
      }
    }
  }
  for (int row_id = 0; row_id < wh_tensor->dims()[0]; row_id++) {
    for (int col_id = 0; col_id < wh_tensor->dims()[1]; col_id++) {
      int idx = (row_id * wh_tensor->dims()[1]) + col_id;
      auto abs_value = std::abs(wh_tensor->data<float>()[idx]);
      if (abs_value > scale[col_id]) scale[col_id] = abs_value;
    }
  }
  transform(scale.begin(), scale.end(), scale.begin(),
            [](float c) { return 1 / c; });
  GetTensorFromVector(scale, tensor);
}

void ComputePropagateScalesMkldnnPass::ComputeLstmWeightScales(
    ir::Graph* graph, Scope* scope, const std::string& wx_name,
    const std::string& wh_name, StringPairMap* var_quant_scales) const {
  for (auto* op_node :
       ir::TopologyVarientSort(*graph, static_cast<ir::SortKind>(0))) {
    if (!op_node->IsOp()) continue;

    auto* op_desc = op_node->Op();
    if (op_desc->Type() == "fusion_lstm") {
      auto wx_var_names = op_desc->Input(wx_name);
      auto wh_var_names = op_desc->Input(wh_name);
      const int wx_names_size = static_cast<int>(wx_var_names.size());
      const int wh_names_size = static_cast<int>(wh_var_names.size());
      PADDLE_ENFORCE_EQ(
          wx_names_size, wh_names_size,
          platform::errors::Fatal("Mismatch in number of weights inputs (%d "
                                  "for WeightX vs. %d for WeightH).",
                                  wx_names_size, wh_names_size));

      for (int i = 0; i < wx_names_size; i++) {
        auto wh_var_name = wh_var_names[i];
        auto wx_var_name = wx_var_names[i];
        Tensor tensor;
        ComputeSingleLstmWeightScales(scope, wx_var_name, wh_var_name, &tensor);
        auto pair = std::make_pair(false, tensor);
        var_quant_scales->insert(std::make_pair(wx_var_name, pair));
      }
    }
  }
}

void ComputePropagateScalesMkldnnPass::ComputeWeightScales(
    ir::Graph* graph, Scope* scope, StringPairMap* var_quant_scales) const {
  ComputeVarScales(graph, scope, {"conv2d", "depthwise_conv2d"}, "Filter", 1,
                   var_quant_scales);
  ComputeVarScales(graph, scope, {"fc"}, "W", 0, var_quant_scales);
  ComputeVarScales(graph, scope, {"fusion_gru", "multi_gru"}, "WeightH", 0,
                   var_quant_scales);
  ComputeVarScales(graph, scope, {"fusion_lstm"}, "WeightH", 0,
                   var_quant_scales);
  ComputeGruWeightScales(graph, scope, "WeightX", "WeightH", var_quant_scales);
  ComputeLstmWeightScales(graph, scope, "WeightX", "WeightH", var_quant_scales);
}

void ComputePropagateScalesMkldnnPass::UpdateScaleOpInScale(
    Node* op_node, const std::string& input_name,
    const std::string& output_name, StringPairMap* var_quant_scales) const {
  auto iter = var_quant_scales->find(output_name);
  if (iter != var_quant_scales->end()) {
    auto pair = iter->second;
    const auto tensor = pair.second;

    const auto scale = BOOST_GET_CONST(float, op_node->Op()->GetAttr("scale"));
    Tensor tmp_tensor;
    tmp_tensor.Resize(tensor.dims());
    auto* data = tmp_tensor.mutable_data<float>(platform::CPUPlace());
    for (int i = 0; i < tensor.numel(); i++) {
      data[i] = data[i] * scale;
    }

    auto new_pair = std::make_pair(pair.first, tmp_tensor);
    var_quant_scales->insert(std::make_pair(input_name, new_pair));
  }
}

std::unordered_set<std::string> ComputePropagateScalesMkldnnPass::UpdateScales(
    ir::Graph* graph, StringPairMap* var_quant_scales,
    const std::unordered_set<std::string>& scale_immutable_ops) const {
  std::unordered_set<std::string> waiting_for_scale{};
  for (auto* op_node :
       ir::TopologyVarientSort(*graph, static_cast<ir::SortKind>(0))) {
    if (!op_node->IsOp()) continue;

    const auto op_name = op_node->Name();
    if (scale_immutable_ops.count(op_name)) {
      std::string input_name;
      if (op_name == "slice") {
        input_name = op_node->Op()->Input("Input")[0];
      } else {
        input_name = op_node->Op()->Input("X")[0];
      }

      const std::string output_name = op_node->Op()->Output("Out")[0];
      auto in_iter = var_quant_scales->find(input_name);
      auto out_iter = var_quant_scales->find(output_name);
      if (in_iter == var_quant_scales->end() &&
          out_iter == var_quant_scales->end()) {
        waiting_for_scale.insert(input_name);
        waiting_for_scale.insert(output_name);
      } else if (in_iter != var_quant_scales->end()) {
        out_iter->second = in_iter->second;
      } else if (out_iter != var_quant_scales->end()) {
        in_iter->second = out_iter->second;
      }
    } else if (op_name == "scale") {
      const std::string output_name = op_node->Op()->Output("Out")[0];
      auto out_iter = var_quant_scales->find(output_name);
      if (out_iter != var_quant_scales->end()) {
        const std::string input_name = op_node->Op()->Input("X")[0];
        UpdateScaleOpInScale(op_node, input_name, output_name,
                             var_quant_scales);
      }
    }
  }
  return waiting_for_scale;
}

void ComputePropagateScalesMkldnnPass::PropagateScales(
    ir::Graph* graph, StringPairMap* var_quant_scales,
    const std::unordered_set<std::string>& scale_immutable_ops) const {
  auto waiting_for_scale =
      UpdateScales(graph, var_quant_scales, scale_immutable_ops);
  std::unordered_set<std::string> waiting_for_scale_prev{};
  while (waiting_for_scale.size() != 0 &&
         waiting_for_scale != waiting_for_scale_prev) {
    waiting_for_scale_prev.clear();
    waiting_for_scale_prev.insert(waiting_for_scale.begin(),
                                  waiting_for_scale.end());
    waiting_for_scale =
        UpdateScales(graph, var_quant_scales, scale_immutable_ops);
  }
}

void ComputePropagateScalesMkldnnPass::ConvertStringPairMap(
    const StringPairMap& var_quant_scales,
    std::unordered_map<std::string, std::vector<float>>* info_map) const {
  for (auto iter = var_quant_scales.begin(); iter != var_quant_scales.end();
       iter++) {
    auto* data = iter->second.second.data<float>();
    std::vector<float> data_v;
    for (int i = 0; i < iter->second.second.numel(); i++) {
      data_v.push_back(data[i]);
    }

    info_map->insert(std::make_pair(iter->first, data_v));
  }
}

void ComputePropagateScalesMkldnnPass::ApplyImpl(ir::Graph* graph) const {
  VLOG(3) << "Convert paddle model to mkldnn quantized model.";
  const std::string pattern_name = "compute_propagate_scales_mkldnn_pass";
  FusePassBase::Init(pattern_name, graph);

  const std::unordered_set<std::string> scale_immutable_ops = {
      "transpose2", "reshape2",       "pool2d",
      "slice",      "nearest_interp", "nearest_interp_v2"};

  StringPairMap var_quant_scales{};

  auto* scope = param_scope();
  GetQuantInfo(graph, &var_quant_scales);
  ComputeWeightScales(graph, scope, &var_quant_scales);
  PropagateScales(graph, &var_quant_scales, scale_immutable_ops);

  // save var_quant_scales in the first op's attr
  // for cpu_quantize_pass
  std::unordered_map<std::string, std::vector<float>> info_map;
  ConvertStringPairMap(var_quant_scales, &info_map);
  SaveInfoInTheFirstOp(graph, "has_quant_info", "var_quant_scales", info_map);
}

}  // namespace ir
}  // namespace framework
}  // namespace paddle

REGISTER_PASS(compute_propagate_scales_mkldnn_pass,
              paddle::framework::ir::ComputePropagateScalesMkldnnPass);

REGISTER_PASS_CAPABILITY(compute_propagate_scales_mkldnn_pass)
    .AddCombination(
        paddle::framework::compatible::OpVersionComparatorCombination()
            .LE("conv2d", 1)
            .EQ("fc", 0)
            .LE("conv2d_transpose", 2)
            .EQ("fake_quantize_abs_max", 0)
            .EQ("fake_quantize_range_abs_max", 0)
            .EQ("fake_quantize_moving_average_abs_max", 0)
            .LE("fake_channel_wise_quantize_abs_max", 1)
            .EQ("fake_dequantize_max_abs", 0));
