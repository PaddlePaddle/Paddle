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

#include "paddle/fluid/framework/ir/onednn/compute_propagate_scales_onednn_pass.h"

#include <cfloat>

#include <algorithm>

#include "paddle/fluid/framework/ir/graph_helper.h"
#include "paddle/fluid/framework/op_version_registry.h"

namespace paddle::framework::ir {

void ComputePropagateScalesMkldnnPass::GetTensorFromVector(
    const std::vector<float>& data_v, phi::DenseTensor* tensor) const {
  const int size = static_cast<int>(data_v.size());
  auto* data = tensor->mutable_data<float>({size}, phi::CPUPlace());
  for (int i = 0; i < size; i++) {
    data[i] = data_v[i];
  }
}

void ComputePropagateScalesMkldnnPass::GetQuantInfo(
    ir::Graph* graph, StringPairMap* var_quant_scales) const {
  std::unordered_map<std::string, std::vector<float>> info_map{};
  GetInfoFromTheTmpOp(graph, "has_quant_info", "var_quant_scales", &info_map);

  for (auto& item : info_map) {
    phi::DenseTensor tensor;
    GetTensorFromVector(item.second, &tensor);
    auto pair = std::make_pair(false, tensor);
    var_quant_scales->insert(std::make_pair(item.first, pair));
  }
}

std::vector<float> ComputePropagateScalesMkldnnPass::GetScales(
    phi::DenseTensor* tensor, int axis) const {
  PADDLE_ENFORCE_LT(axis,
                    2,
                    common::errors::InvalidArgument(
                        "The input axis is required to be less than 2."));
  auto* data = tensor->data<float>();
  const auto dims = tensor->dims();
  PADDLE_ENFORCE_EQ(dims.size(),
                    2,
                    common::errors::InvalidArgument(
                        "The input tensor's rank is required to be 2."));

  const int rows = static_cast<int>(dims.at(0));
  const int columns = static_cast<int>(dims.at(1));
  std::vector<float> scales;
  if (axis == 0) {
    for (int i = 0; i < columns; i++) {
      float max_value = FLT_MIN;
      for (int j = 0; j < rows; j++) {
        max_value = std::max(max_value, std::abs(data[j + i * rows]));
      }
      max_value = static_cast<float>(1.0) / max_value;
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
      max_value = static_cast<float>(1.0) / max_value;
      if (std::isinf(max_value) || std::isnan(max_value)) {
        max_value = 0.0;
      }
      scales.push_back(max_value);
    }
  }
  return scales;
}

void ComputePropagateScalesMkldnnPass::ComputeVarScales(
    ir::Graph* graph,
    Scope* scope,
    const std::unordered_set<std::string>& ops,
    const std::string& weight_name,
    const int axis,
    StringPairMap* var_quant_scales) const {
  for (auto* op_node :
       ir::TopologyVariantSort(*graph, static_cast<ir::SortKind>(0))) {
    if (!op_node->IsOp()) continue;

    auto* op_desc = op_node->Op();
    if (ops.count(op_desc->Type())) {
      auto var_name = op_desc->Input(weight_name)[0];
      auto* var = scope->FindVar(var_name);
      PADDLE_ENFORCE_NOT_NULL(
          var,
          common::errors::NotFound(
              "The input persistable var [%s] of [%s] op is not found.",
              var_name,
              op_desc->Type()));
      auto* weight_tensor = var->GetMutable<phi::DenseTensor>();
      const auto dims = weight_tensor->dims();
      int volume = 1;
      for (int i = 1; i < dims.size(); i++) {
        volume *= dims[i];
      }

      phi::DenseTensor tmp_tensor;
      std::vector<int64_t> reshape_dims = {dims[0], volume};
      tmp_tensor.Resize(common::make_ddim(reshape_dims));
      auto* weight_data = weight_tensor->data<float>();
      auto* tmp_data = tmp_tensor.mutable_data<float>(phi::CPUPlace());
      for (int i = 0; i < weight_tensor->numel(); i++) {
        tmp_data[i] = std::abs(weight_data[i]);
      }

      auto scales_v = GetScales(&tmp_tensor, axis);
      phi::DenseTensor tensor;
      GetTensorFromVector(scales_v, &tensor);
      auto pair = std::make_pair(false, tensor);
      var_quant_scales->insert(std::make_pair(var_name, pair));
    }
  }
}

void ComputePropagateScalesMkldnnPass::ComputeSingleGruWeightScales(
    Scope* scope,
    const std::string& wx_var_name,
    const std::string& wh_var_name,
    phi::DenseTensor* tensor) const {
  auto* wx_var = scope->FindVar(wx_var_name);
  PADDLE_ENFORCE_NOT_NULL(
      wx_var,
      common::errors::NotFound("The input persistable var [%s] is not found.",
                               wx_var_name));
  auto* wh_var = scope->FindVar(wh_var_name);
  PADDLE_ENFORCE_NOT_NULL(
      wh_var,
      common::errors::NotFound("The input persistable var [%s] is not found.",
                               wh_var_name));

  const auto* wx_tensor = wx_var->GetMutable<phi::DenseTensor>();
  const auto* wh_tensor = wh_var->GetMutable<phi::DenseTensor>();
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
  transform(scale_ur.begin(), scale_ur.end(), scale_ur.begin(), [](float c) {
    return 1 / c;
  });
  GetTensorFromVector(scale_ur, tensor);
}

void ComputePropagateScalesMkldnnPass::ComputeGruWeightScales(
    ir::Graph* graph,
    Scope* scope,
    const std::string& wx_name,
    const std::string& wh_name,
    StringPairMap* var_quant_scales) const {
  for (auto* op_node :
       ir::TopologyVariantSort(*graph, static_cast<ir::SortKind>(0))) {
    if (!op_node->IsOp()) continue;

    auto* op_desc = op_node->Op();
    if (op_desc->Type() == "fusion_gru" || op_desc->Type() == "multi_gru") {
      auto wx_var_names = op_desc->Input(wx_name);
      auto wh_var_names = op_desc->Input(wh_name);
      const int wx_names_size = static_cast<int>(wx_var_names.size());
      const int wh_names_size = static_cast<int>(wh_var_names.size());
      PADDLE_ENFORCE_EQ(
          wx_names_size,
          wh_names_size,
          common::errors::Fatal("Mismatch in number of weights inputs (%d "
                                "for WeightX vs. %d for WeightH).",
                                wx_names_size,
                                wh_names_size));
      for (int i = 0; i < wx_names_size; i++) {
        auto wh_var_name = wh_var_names[i];
        auto wx_var_name = wx_var_names[i];
        phi::DenseTensor tensor;
        ComputeSingleGruWeightScales(scope, wx_var_name, wh_var_name, &tensor);
        auto pair = std::make_pair(false, tensor);
        var_quant_scales->insert(std::make_pair(wx_var_name, pair));
      }
    }
  }
}

void ComputePropagateScalesMkldnnPass::ComputeSingleLstmWeightScales(
    Scope* scope,
    const std::string& wx_var_name,
    const std::string& wh_var_name,
    phi::DenseTensor* tensor) const {
  auto* wx_var = scope->FindVar(wx_var_name);
  PADDLE_ENFORCE_NOT_NULL(
      wx_var,
      common::errors::NotFound("The input persistable var [%s] is not found.",
                               wx_var_name));
  auto* wh_var = scope->FindVar(wh_var_name);
  PADDLE_ENFORCE_NOT_NULL(
      wh_var,
      common::errors::NotFound("The input persistable var [%s] is not found.",
                               wh_var_name));

  const auto* wx_tensor = wx_var->GetMutable<phi::DenseTensor>();
  const auto* wh_tensor = wh_var->GetMutable<phi::DenseTensor>();
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
  transform(
      scale.begin(), scale.end(), scale.begin(), [](float c) { return 1 / c; });
  GetTensorFromVector(scale, tensor);
}

void ComputePropagateScalesMkldnnPass::ComputeLstmWeightScales(
    ir::Graph* graph,
    Scope* scope,
    const std::string& wx_name,
    const std::string& wh_name,
    StringPairMap* var_quant_scales) const {
  for (auto* op_node :
       ir::TopologyVariantSort(*graph, static_cast<ir::SortKind>(0))) {
    if (!op_node->IsOp()) continue;

    auto* op_desc = op_node->Op();
    if (op_desc->Type() == "fusion_lstm") {
      auto wx_var_names = op_desc->Input(wx_name);
      auto wh_var_names = op_desc->Input(wh_name);
      const int wx_names_size = static_cast<int>(wx_var_names.size());
      const int wh_names_size = static_cast<int>(wh_var_names.size());
      PADDLE_ENFORCE_EQ(
          wx_names_size,
          wh_names_size,
          common::errors::Fatal("Mismatch in number of weights inputs (%d "
                                "for WeightX vs. %d for WeightH).",
                                wx_names_size,
                                wh_names_size));

      for (int i = 0; i < wx_names_size; i++) {
        auto wh_var_name = wh_var_names[i];
        auto wx_var_name = wx_var_names[i];
        phi::DenseTensor tensor;
        ComputeSingleLstmWeightScales(scope, wx_var_name, wh_var_name, &tensor);
        auto pair = std::make_pair(false, tensor);
        var_quant_scales->insert(std::make_pair(wx_var_name, pair));
      }
    }
  }
}

void ComputePropagateScalesMkldnnPass::ComputeWeightScales(
    ir::Graph* graph, Scope* scope, StringPairMap* var_quant_scales) const {
  ComputeVarScales(graph,
                   scope,
                   {"conv2d", "depthwise_conv2d", "fused_conv2d"},
                   "Filter",
                   1,
                   var_quant_scales);
  ComputeVarScales(graph, scope, {"fc"}, "W", 0, var_quant_scales);
  ComputeVarScales(graph,
                   scope,
                   {"fusion_gru", "multi_gru"},
                   "WeightH",
                   0,
                   var_quant_scales);
  ComputeVarScales(
      graph, scope, {"fusion_lstm"}, "WeightH", 0, var_quant_scales);
  ComputeGruWeightScales(graph, scope, "WeightX", "WeightH", var_quant_scales);
  ComputeLstmWeightScales(graph, scope, "WeightX", "WeightH", var_quant_scales);
}

void ComputePropagateScalesMkldnnPass::UpdateScaleOpInOutScales(
    Node* op_node,
    const std::string& input_name,
    const std::string& output_name,
    StringPairMap* var_quant_scales) const {
  auto out_iter = var_quant_scales->find(output_name);
  auto input_iter = var_quant_scales->find(input_name);
  // All the input and output have scales
  if (out_iter != var_quant_scales->end() &&
      input_iter != var_quant_scales->end()) {
    return;
  }

  const auto scale = PADDLE_GET_CONST(float, op_node->Op()->GetAttr("scale"));
  if (std::abs(scale) < 1e-6 && out_iter != var_quant_scales->end()) {
    return;
  }

  std::string name = input_name;
  auto iter = out_iter;
  if (input_iter != var_quant_scales->end()) {
    iter = input_iter;
    name = output_name;
  }

  phi::DenseTensor tmp_tensor;
  auto pair = iter->second;
  const auto tensor = pair.second;
  tmp_tensor.Resize(tensor.dims());
  auto* data = tmp_tensor.mutable_data<float>(phi::CPUPlace());
  auto* src_data = tensor.data<float>();
  for (int i = 0; i < tensor.numel(); i++) {
    if (out_iter != var_quant_scales->end()) {
      data[i] = src_data[i] / scale;
    } else {
      data[i] = src_data[i] * scale;
    }
  }
  auto new_pair = std::make_pair(pair.first, tmp_tensor);
  var_quant_scales->insert(std::make_pair(name, new_pair));
}

std::unordered_set<std::string> ComputePropagateScalesMkldnnPass::UpdateScales(
    ir::Graph* graph,
    StringPairMap* var_quant_scales,
    const std::unordered_set<std::string>& scale_immutable_ops) const {
  std::unordered_set<std::string> waiting_for_scale{};
  for (auto* op_node :
       ir::TopologyVariantSort(*graph, static_cast<ir::SortKind>(0))) {
    if (!op_node->IsOp()) continue;

    const auto op_name = op_node->Name();
    if (scale_immutable_ops.count(op_name)) {
      std::string input_name;
      if (op_name == "slice" || op_name == "shape") {
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
        (*var_quant_scales)[output_name] = in_iter->second;
      } else if (out_iter != var_quant_scales->end()) {
        (*var_quant_scales)[input_name] = out_iter->second;
      }
    } else if (op_name == "concat") {
      auto out_iter = var_quant_scales->find(op_node->Op()->Output("Out")[0]);
      if (out_iter != var_quant_scales->end()) {
        std::vector<std::string> input_names = op_node->Op()->Input("X");
        for (auto const& input_name : input_names) {
          auto concat_in_iter = var_quant_scales->find(input_name);
          if (concat_in_iter == var_quant_scales->end())
            (*var_quant_scales)[input_name] = out_iter->second;
          else
            (*var_quant_scales)[input_name].second = out_iter->second.second;
        }
      }
    } else if (op_name == "scale") {
      const std::string output_name = op_node->Op()->Output("Out")[0];
      const std::string input_name = op_node->Op()->Input("X")[0];
      auto out_iter = var_quant_scales->find(output_name);
      auto input_iter = var_quant_scales->find(input_name);
      if (out_iter != var_quant_scales->end() ||
          input_iter != var_quant_scales->end()) {
        UpdateScaleOpInOutScales(
            op_node, input_name, output_name, var_quant_scales);
      }
    }
  }
  return waiting_for_scale;
}
void ComputePropagateScalesMkldnnPass::UpdateReluOutputScales(
    ir::Graph* graph, StringPairMap* var_quant_scales) const {
  for (auto* op_node :
       ir::TopologyVariantSort(*graph, static_cast<ir::SortKind>(0))) {
    if (!op_node->IsOp()) continue;
    auto op = op_node->Op();
    bool is_unsigned = false;
    std::string output_name = "Out";
    std::string act_name;
    if (op->Type() == "relu") {
      is_unsigned = true;
    } else {
      if (op->Type() == "conv2d" || op->Type() == "fused_conv2d") {
        act_name = "fuse_activation";
        output_name = "Output";
      } else if (op->Type() == "fc") {
        act_name = "activation_type";
      }
      if (!act_name.empty()) {
        auto act = op->GetAttrIfExists<std::string>(act_name);
        if (act == "relu" || act == "relu6") {
          is_unsigned = true;
        }
      }
    }
    if (is_unsigned) {
      std::string output_var_name = op->Output(output_name)[0];
      auto out_iter = var_quant_scales->find(output_var_name);
      if (out_iter != var_quant_scales->end()) {
        (*var_quant_scales)[output_var_name].first = true;
      }
    }
  }
}

void ComputePropagateScalesMkldnnPass::PropagateScales(
    ir::Graph* graph,
    StringPairMap* var_quant_scales,
    const std::unordered_set<std::string>& scale_immutable_ops) const {
  auto waiting_for_scale =
      UpdateScales(graph, var_quant_scales, scale_immutable_ops);
  std::unordered_set<std::string> waiting_for_scale_prev{};
  while (!waiting_for_scale.empty() &&
         waiting_for_scale != waiting_for_scale_prev) {
    waiting_for_scale_prev.clear();
    waiting_for_scale_prev.insert(waiting_for_scale.begin(),
                                  waiting_for_scale.end());
    waiting_for_scale =
        UpdateScales(graph, var_quant_scales, scale_immutable_ops);
  }
}

void ComputePropagateScalesMkldnnPass::ApplyImpl(ir::Graph* graph) const {
  VLOG(3) << "Convert paddle model to onednn quantized model.";
  const std::string pattern_name = "compute_propagate_scales_onednn_pass";
  FusePassBase::Init(pattern_name, graph);

  const std::unordered_set<std::string> scale_immutable_ops = {
      "fused_transpose",
      "transpose2",
      "reshape2",
      "pool2d",
      "slice",
      "shape",
      "nearest_interp",
      "nearest_interp_v2",
      "split"};

  StringPairMap var_quant_scales{};

  auto* scope = param_scope();
  GetQuantInfo(graph, &var_quant_scales);
  ComputeWeightScales(graph, scope, &var_quant_scales);
  UpdateReluOutputScales(graph, &var_quant_scales);
  PropagateScales(graph, &var_quant_scales, scale_immutable_ops);

  // save var_quant_scales in the temporary save op's attr
  // for cpu_quantize_pass
  SaveInfoInTheTmpOp(
      graph, "has_quant_info", "var_quant_scales", var_quant_scales);
}

}  // namespace paddle::framework::ir

REGISTER_PASS(compute_propagate_scales_onednn_pass,
              paddle::framework::ir::ComputePropagateScalesMkldnnPass);

REGISTER_PASS_CAPABILITY(compute_propagate_scales_onednn_pass)
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
