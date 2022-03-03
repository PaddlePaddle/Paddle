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

#include "paddle/fluid/framework/ir/mkldnn/requant_mkldnn_fuse_pass.h"
#include <float.h>
#include <algorithm>
#include "paddle/fluid/framework/ir/graph_helper.h"
#include "paddle/fluid/framework/op_version_registry.h"

namespace paddle {
namespace framework {
namespace ir {

Tensor* RequantMkldnnFusePass::GetTensorFromVector(
    const std::vector<float>& data_v) {
  Tensor tensor;
  tensor.mutable_data<float>({data_v.size()}, platform::CPUPlace());
  auto dev_ctx = paddle::platform::CPUDeviceContext();
  TensorFromVector(data_v, dev_ctx, &tensor);
  return &tensor;
}

void RequantMkldnnFusePass::GetQuantInfo(
    ir::Graph* graph, Scope* scope,
    std::unordered_map<std::string, Tensor*>& weight_thresholds,
    std::unordered_map<std::string, std::pair<bool, Tensor*>>& var_quant_scales)
    const {
  for (auto* op_node :
       ir::TopologyVarientSort(*graph, static_cast<ir::SortKind>(0))) {
    if (!op_node->IsOp()) continue;

    auto* op_desc = op_node->Op();
    if (op_desc->GetAttrIfExists<bool>("has_quant_info")) {
      op_desc->RemoveAttr("has_quant_info");
      std::vector<std::string> attr_names = op_desc->AttrNames();
      for (auto fake_name : attr_names) {
        if (fake_name.find("_weight_thresholds") != std::string::npos) {
          size_t pos = fake_name.find("_weight_thresholds");
          std::string name = fake_name.substr(0, pos);
          auto thresholds_vector =
              BOOST_GET_CONST(std::vector<float>, op_desc->GetAttr(fake_name));
          weight_thresholds[name] = GetTensorFromVector(thresholds_vector);
          op_desc->RemoveAttr(fake_name);
        }

        if (fake_name.find("_var_quant_scales") != std::string::npos) {
          size_t pos = fake_name.find("_var_quant_scales");
          std::string name = fake_name.substr(0, pos);
          auto scales_vector =
              BOOST_GET_CONST(std::vector<float>, op_desc->GetAttr(fake_name));
          auto* tensor = GetTensorFromVector(scales_vector);
          var_quant_scales[name] = std::make_pair(false, tensor);
          op_desc->RemoveAttr(fake_name);
        }
      }
      break;
    }
  }
}

void RequantMkldnnFusePass::ComputeWeightScales(
    ir::Graph* graph, Scope* scope,
    std::unordered_map<std::string, Tensor*>& weight_thresholds,
    std::unordered_map<std::string, std::pair<bool, Tensor*>>& var_quant_scales)
    const {
  auto get_scales = [&](Tensor* tensor, int axis) -> std::vector<float> {
    PADDLE_ENFORCE_LT(axis, 2, "The input axis is required to be less than 2.");
    auto data = tensor->mutable_data<float>(platform::CPUPlace());
    const auto dims = tensor->dims();
    PADDLE_ENFORCE_EQ(dims.size(), 2,
                      "The input tensor's rank is required to be 2.");

    const int rows = dims.at(0);
    const int columns = dims.at(1);
    std::vector<float> scales;
    if (axis == 0) {
      for (int i = 0; i < columns; i++) {
        float max_value = FLT_MIN;
        for (int j = 0; j < rows; j++) {
          max_value = std::max(max_value, data[i + j * columns]);
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
          max_value = std::max(max_value, data[j]);
        }
        max_value = 1.0 / max_value;
        if (std::isinf(max_value) || std::isnan(max_value)) {
          max_value = 0.0;
        }
        scales.push_back(max_value);
      }
    }
    return scales;
  };

  auto compute_var_scales = [&](
      ir::Graph* graph, Scope* scope std::unordered_set<std::string> ops,
      std::string weight_name, int axis) {
    for (auto* op_node :
         ir::TopologyVarientSort(*graph, static_cast<ir::SortKind>(0))) {
      if (!op_node->IsOp()) continue;

      auto* op_desc = op_node->Op();
      if (ops.count(op_desc->Type())) {
        auto var_name = op_desc->Input(weight_name)[0];
        auto* var = scope->FindVar(var_name);
        PADDLE_ENFORCE_NOT_NULL(
            var, "The input persistable var of %s op is not found.",
            op_desc->Type());
        auto* weight_tensor = var->GetMutable<LoDTensor>();
        const auto dims = weight_tensor->dims();
        int volume = 1;
        for (int i = 1; i < dims.size(); i++) {
          volume *= dims[i];
        }

        Tensor tmp_tensor;
        std::vector<int64_t> reshape_dims = {dims[0], volume};
        tmp_tensor->Resize(framework::make_ddim(reshape_dims));
        auto weight_data =
            weight_tensor->mutable_data<float>(platform::CPUPlace());
        auto tmp_data = tmp_tensor->mutable_data<float>(platform::CPUPlace());
        for (int i = 0; i < weight_tensor->numel(); i++) {
          tmp_data[i] = std::fabs(weight_data[i]);
        }

        auto scales_v = get_scales(&tmp_tensor, axis);
        auto* scales_t = GetTensorFromVector(scales_v);
        var_quant_scales[var_name] = std::make_pair(false, scales_t);
      }
    }
  };

  auto compute_single_gru_weight_scales = [&](
      Scope* scope, std::string wx_var_name,
      std::string wh_var_name) -> std::pair<bool, Tensor*> {
    auto* wx_var = scope->FindVar(wx_var_name);
    PADDLE_ENFORCE_NOT_NULL(
        wx_var, "The input persistable var %s is not found.", wx_var_name);
    auto* wh_var = scope->FindVar(wh_var_name);
    PADDLE_ENFORCE_NOT_NULL(
        wh_var, "The input persistable var %s is not found.", wh_var_name);
    const auto* wx_tensor = wx_var->GetMutable<LoDTensor>();
    const auto* wh_tensor = wh_var->GetMutable<LoDTensor>();
    const int OC = wh_tensor.dims()[0];
    std::vector<float> scale_ur(2 * OC);
    std::vector<float> scale_o(OC);
    for (int row_id = 0; row_id < wx_tensor.dims()[0]; row_id++) {
      for (int col_id = 0; col_id < 2 * OC; col_id++) {
        int idx = (row_id * wx_tensor.dims()[1]) + col_id;
        auto abs_value = std::fabs(wx_tensor.data<float>()[idx]);
        if (row_id == 0) {
          scale_ur[col_id] = abs_value;
        } else {
          if (abs_value > scale_ur[col_id]) scale_ur[col_id] = abs_value;
        }
      }
    }

    for (int i = 0; i < 2 * OC * OC; i++) {
      int col_id = i % (2 * OC);
      auto abs_value = std::fabs(wh_tensor.data<float>()[i]);
      if (abs_value > scale_ur[col_id]) scale_ur[col_id] = abs_value;
    }

    for (int row_id = 0; row_id < wx_tensor.dims()[0]; row_id++) {
      for (int col_id = 2 * OC; col_id < wx_tensor.dims()[1]; col_id++) {
        int idx = (row_id * wx_tensor.dims()[1]) + col_id;
        auto abs_value = std::fabs(wx_tensor.data<float>()[idx]);
        if (row_id == 0) {
          scale_o[col_id % OC] = abs_value;
        } else {
          if (abs_value > scale_o[col_id]) scale_o[col_id % OC] = abs_value;
        }
      }
    }

    for (int i = 2 * OC * OC; i < OC * wh_tensor.dims()[1]; i++) {
      int col_id = i % OC;
      auto abs_value = std::fabs(wh_tensor.data<float>()[i]);
      if (abs_value > scale_o[col_id]) scale_o[col_id] = abs_value;
    }
    scale_ur.insert(scale_ur.end(), scale_o.begin(), scale_o.end());
    transform(scale_ur.begin(), scale_ur.end(), scale_ur.begin(),
              [](float& c) { return 1 / c; });
    return std::make_pair(false, GetTensorFromVector(scale_ur));
  };
}

void RequantMkldnnFusePass::ApplyImpl(ir::Graph* graph) const {
  VLOG(3) << "Convert paddle model to mkldnn quantized model.";
  const std::string pattern_name = "requant_mkldnn_fuse_pass";
  FusePassBase::Init(pattern_name, graph);

  std::unordered_map<std::string, Tensor*> weight_thresholds;
  std::unordered_map<std::string, std::pair<int, Tensor*>> var_quant_scales;

  auto* scope = param_scope();
  GetQuantInfo(graph, scope, weight_thresholds, var_quant_scales);

  // ComputeWeightScales(graph, scope);
}

}  // namespace ir
}  // namespace framework
}  // namespace paddle

REGISTER_PASS(requant_mkldnn_fuse_pass,
              paddle::framework::ir::RequantMkldnnFusePass);

REGISTER_PASS_CAPABILITY(requant_mkldnn_fuse_pass)
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
