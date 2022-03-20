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

#pragma once

#include <string>
#include "paddle/fluid/framework/ir/graph_helper.h"

namespace paddle {
namespace framework {
namespace ir {

static void SaveInfoInTheFirstOp(
    ir::Graph* graph, const std::string& flag, const std::string& key_suffix,
    const std::unordered_map<std::string, std::vector<float>>& info_map) {
  VLOG(3) << "save variables in the first op's attr";

  const std::string suffix = "_" + key_suffix + "_" + flag;
  for (auto* op_node :
       ir::TopologyVarientSort(*graph, static_cast<ir::SortKind>(0))) {
    if (!op_node->IsOp() || op_node->Op()->Type() == "feed" ||
        op_node->Op()->Type() == "fetch")
      continue;

    op_node->Op()->SetAttr(flag, true);
    for (auto iter = info_map.begin(); iter != info_map.end(); ++iter) {
      op_node->Op()->SetAttr(iter->first + suffix, iter->second);
    }
    break;
  }
}

static void GetInfoFromTheFirstOp(
    ir::Graph* graph, const std::string& flag, const std::string& key_suffix,
    std::unordered_map<std::string, std::vector<float>>* info_map) {
  VLOG(3) << "get variables from the first op's attr";

  const std::string suffix = "_" + key_suffix + "_" + flag;
  for (auto* op_node :
       ir::TopologyVarientSort(*graph, static_cast<ir::SortKind>(0))) {
    if (!op_node->IsOp() || op_node->Op()->Type() == "feed" ||
        op_node->Op()->Type() == "fetch")
      continue;

    auto* op_desc = op_node->Op();
    if (op_desc->GetAttrIfExists<bool>(flag)) {
      op_desc->RemoveAttr(flag);
      std::vector<std::string> attr_names = op_desc->AttrNames();
      for (auto fake_name : attr_names) {
        size_t pos = fake_name.find(suffix);
        if (pos != std::string::npos) {
          std::string name = fake_name.substr(0, pos);
          auto scales_vector =
              BOOST_GET_CONST(std::vector<float>, op_desc->GetAttr(fake_name));
          info_map->insert(std::make_pair(name, scales_vector));
          op_desc->RemoveAttr(fake_name);
        }
      }
      break;
    }
  }
}

static void TransposeWeight(Tensor* input) {
  const auto in_dims = input->dims();
  std::vector<int> out_dim_v;
  std::vector<int> axis;
  for (int i = in_dims.size() - 1; i >= 0; i--) {
    axis.push_back(i);
    out_dim_v.push_back(in_dims[i]);
  }

  const auto out_dims = phi::make_ddim(out_dim_v);
  const int rank = axis.size();
  auto in_stride = phi::stride(in_dims);
  auto out_stride = phi::stride(out_dims);
  const int count = input->numel();

  Tensor trans_tensor;
  trans_tensor.Resize(out_dims);
  float* trans_data = trans_tensor.mutable_data<float>(platform::CPUPlace());
  float* in_data = input->mutable_data<float>(platform::CPUPlace());

  for (int64_t out_idx = 0; out_idx < count; ++out_idx) {
    int64_t in_idx = 0;
    int64_t tmp_idx = out_idx;
    for (int i = 0; i < rank; ++i) {
      const int64_t coordinate = tmp_idx / out_stride[i];
      tmp_idx -= coordinate * out_stride[i];
      in_idx += coordinate * in_stride[axis[i]];
    }
    trans_data[out_idx] = in_data[in_idx];
  }

  input->Resize(out_dims);
  for (int i = 0; i < input->numel(); i++) {
    in_data[i] = trans_data[i];
  }
}

static bool IsInt8Weight(
    Node* op_node, Scope* scope, const std::string& weight_name) {
  auto* op_desc = op_node->Op();
  auto var_name = op_desc->Input(weight_name)[0];
  auto* var = scope->FindVar(var_name);
  PADDLE_ENFORCE_NOT_NULL(
      var, platform::errors::NotFound(
               "The input persistable [%s] var of [%s] op is not found.",
               var_name, op_desc->Type()));
  auto* weight_tensor = var->GetMutable<LoDTensor>();
  auto* weight_data = weight_tensor->data<float>();
  bool is_int8 = true;
  for (int i = 0; i < weight_tensor->numel(); i++) {
    if (weight_data[i] - static_cast<int>(weight_data[i]) != 0) {
      is_int8 = false;
      break;
    }
  }
  return is_int8;
}

static void DequantizeOpWeights(
    Node* op_node, Scope* scope, const std::string& weight_name,
    const std::string& output_name,
    const std::unordered_map<std::string, std::vector<float>>&
        weight_thresholds) {
  auto* op_desc = op_node->Op();
  std::string weight_var_name = op_desc->Input(weight_name)[0];
  std::string output_var_name = op_desc->Output(output_name)[0];

  std::vector<float> scales;
  auto iter = weight_thresholds.find(output_var_name);
  if (iter != weight_thresholds.end()) {
    scales = iter->second;
  } else {
    PADDLE_THROW(paddle::platform::errors::Fatal(
        "Could not find threshold information for [%s] var, please check if "
        "the model is correct.",
        output_var_name));
  }

  auto* var = scope->FindVar(weight_var_name);
  PADDLE_ENFORCE_NOT_NULL(
      var, platform::errors::NotFound(
               "The input persistable [%s] var of [%s] op is not found.",
               weight_var_name, op_desc->Type()));
  auto* weight_tensor = var->GetMutable<LoDTensor>();
  const auto weight_dims = weight_tensor->dims();

  const int size = scales.size();
  if (size == 1 || size == weight_dims[0]) {
    auto* weight_data =
        weight_tensor->mutable_data<float>(platform::CPUPlace());
    for (int i = 0; i < weight_tensor->numel(); i++) {
      weight_data[i] /= 127;
    }

    TransposeWeight(weight_tensor);

    if (size == 1) {
      for (int i = 0; i < weight_tensor->numel(); i++) {
        weight_data[i] *= scales[0];
      }
    } else {
      for (int i = 0; i < weight_tensor->numel(); i++) {
        weight_data[i] *= scales[i % size];
      }
    }

    TransposeWeight(weight_tensor);
  } else if (weight_dims.size() > 1 && size == weight_dims[1]) {
    auto* weight_data =
        weight_tensor->mutable_data<float>(platform::CPUPlace());
    for (int i = 0; i < weight_tensor->numel(); i++) {
      weight_data[i] /= 127;
    }

    int step_n = 1;
    for (int i = 1; i < weight_dims.size(); i++) {
      step_n *= weight_dims[i];
    }
    int step_c = step_n / size;
    for (int i = 0; i < weight_dims[0]; i++) {
      int begin_n = i * step_n;
      for (int j = begin_n; j < begin_n + step_n; j++) {
        for (int k = 0; k < size; k++) {
          int begin_c = k * step_c;
          for (int m = begin_c; m < begin_c + step_c; m++) {
            weight_data[m] *= scales[k];
          }
        }
      }
    }
  } else {
    PADDLE_THROW(platform::errors::InvalidArgument(
        "The size of weight scales vector (%d) does not "
        "match the dimensions (%d) of the weights tensor %s.",
        size, weight_tensor->dims().size(), weight_var_name));
  }

  weight_tensor->Resize(weight_dims);
}

static void DequantizeWeights(
    ir::Graph* graph, Scope* scope,
    const std::unordered_map<std::string, std::vector<float>>&
        weight_thresholds) {
  VLOG(3) << "dequantize weight for ops which has weight";

  if (weight_thresholds.empty()) {
    VLOG(3)
        << "No need to dequantize weights because weight_thresholds is empty.";
    return;
  }

  for (auto* op_node :
       ir::TopologyVarientSort(*graph, static_cast<ir::SortKind>(0))) {
    if (!op_node->IsOp()) continue;
    if (op_node->Name() == "conv2d" || op_node->Name() == "depthwise_conv2d") {
      if (IsInt8Weight(op_node, scope, "Filter")) {
        DequantizeOpWeights(op_node, scope, "Filter", "Output",
                            weight_thresholds);
      }
    } else if (op_node->Name() == "mul" || op_node->Name() == "matmul" ||
               op_node->Name() == "matmul_v2") {
      if (IsInt8Weight(op_node, scope, "Y")) {
        DequantizeOpWeights(op_node, scope, "Y", "Out", weight_thresholds);
      }
    }
  }
}

}  // namespace ir
}  // namespace framework
}  // namespace paddle
