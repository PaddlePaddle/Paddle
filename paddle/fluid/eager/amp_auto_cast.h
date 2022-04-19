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

#include "paddle/fluid/eager/api/generated/fluid_generated/dygraph_forward_api.h"
#include "paddle/fluid/framework/convert_utils.h"

namespace egr {

static inline bool NeedCast(const paddle::experimental::Tensor& tensor,
                            const paddle::experimental::DataType& dst_dtype) {
  auto place = tensor.place();
  auto data_type = tensor.dtype();
  if (paddle::platform::is_gpu_place(place) ||
      paddle::platform::is_cuda_pinned_place(place) ||
      paddle::platform::is_xpu_place(place) ||
      paddle::platform::is_mlu_place(place) ||
      paddle::platform::is_npu_place(place) ||
      paddle::platform::is_npu_pinned_place(place)) {
    // CudaPinndePlace is added for varbase created by dataloader
    if ((data_type == paddle::experimental::DataType::FLOAT32 ||
         data_type == paddle::experimental::DataType::FLOAT16 ||
         data_type == paddle::experimental::DataType::BFLOAT16) &&
        (data_type != dst_dtype)) {
      return true;
    }
  }
  return false;
}

inline std::vector<paddle::experimental::Tensor> AmpAutoCasts(
    const std::string& inputs_name,
    const std::vector<paddle::experimental::Tensor>& inputs,
    const paddle::experimental::DataType& dst_dtype, std::string op_name) {
  VLOG(6) << "AMP AmpAutoCasts:"
          << " inputs(" << inputs_name << ") dst_dtype("
          << paddle::framework::DataType2String(dst_dtype) << ").";
  std::vector<paddle::experimental::Tensor> inputs_casted;
  for (auto& input : inputs) {
    if (NeedCast(input, dst_dtype)) {
      paddle::framework::AttributeMap cast_attrs = {
          {"in_dtype", paddle::framework::TransToProtoVarType(input.dtype())},
          {"out_dtype", paddle::framework::TransToProtoVarType(dst_dtype)}};
      inputs_casted.emplace_back(
          std::move(cast_dygraph_function(input, cast_attrs)));
    } else {
      inputs_casted.emplace_back(input);
    }
  }
  return inputs_casted;
}

inline paddle::experimental::Tensor AmpAutoCast(
    const std::string& input_name, const paddle::experimental::Tensor& input,
    const paddle::experimental::DataType& dst_dtype, std::string op_name) {
  VLOG(6) << "AMP AmpAutoCasts:"
          << " input(" << input_name << ") dst_dtype("
          << paddle::framework::DataType2String(dst_dtype) << ").";
  if (dst_dtype == paddle::experimental::DataType::FLOAT16) {
    if (op_name == "run_program") {
      return input;
    }
    if ((op_name == "batch_norm" || op_name == "layer_norm" ||
         op_name == "sync_batch_norm") &&
        input_name != "X") {
      return input;
    }
    if ((op_name == "fused_attention" || op_name == "fused_feedforward")) {
      if (input_name == "LnScale" || input_name == "LnBias" ||
          input_name == "Ln2Scale" || input_name == "Ln2Bias" ||
          input_name == "Ln1Scale" || input_name == "Ln1Bias") {
        return input;
      }
    }
  }
  if (NeedCast(input, dst_dtype)) {
    paddle::framework::AttributeMap cast_attrs = {
        {"in_dtype", paddle::framework::TransToProtoVarType(input.dtype())},
        {"out_dtype", paddle::framework::TransToProtoVarType(dst_dtype)}};
    return cast_dygraph_function(input, cast_attrs);
  }
  return input;
}

}  // namespace egr
