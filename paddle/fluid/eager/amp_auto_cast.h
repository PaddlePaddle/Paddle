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
#include "paddle/fluid/eager/api/manual/fluid_manual/dygraph_forward_api.h"
#include "paddle/fluid/framework/convert_utils.h"

namespace egr {

static inline bool NeedCast(const paddle::Tensor& tensor,
                            const phi::DataType& dst_dtype) {
  auto place = tensor.place();
  auto data_type = tensor.dtype();
  if (phi::is_gpu_place(place) || phi::is_cuda_pinned_place(place) ||
      phi::is_xpu_place(place) || phi::is_custom_place(place)) {
    // CudaPinnedPlace is added for varbase created by dataloader
    if ((data_type == phi::DataType::FLOAT32 ||
         data_type == phi::DataType::FLOAT16 ||
         data_type == phi::DataType::BFLOAT16) &&
        (data_type != dst_dtype)) {
      return true;
    }
  }
  return false;
}

inline std::vector<paddle::Tensor> AmpAutoCasts(
    const std::string& inputs_name,
    const std::vector<paddle::Tensor>& inputs,
    const phi::DataType& dst_dtype,
    std::string op_name UNUSED) {
  VLOG(6) << "AMP AmpAutoCasts:"
          << " inputs(" << inputs_name << ") dst_dtype("
          << phi::DataTypeToString(dst_dtype) << ").";
  std::vector<paddle::Tensor> inputs_casted;
  for (auto& input : inputs) {
    if (NeedCast(input, dst_dtype)) {
      paddle::framework::AttributeMap cast_attrs = {
          {"in_dtype", paddle::framework::TransToProtoVarType(input.dtype())},
          {"out_dtype", paddle::framework::TransToProtoVarType(dst_dtype)}};
      inputs_casted.emplace_back(cast_dygraph_function(input, cast_attrs));
    } else {
      inputs_casted.emplace_back(input);
    }
  }
  return inputs_casted;
}

inline paddle::Tensor AmpAutoCast(const std::string& input_name,
                                  const paddle::Tensor& input,
                                  const phi::DataType& dst_dtype,
                                  std::string op_name) {
  VLOG(6) << "AMP AmpAutoCasts: op_name(" << op_name << ") input(" << input_name
          << ") dst_dtype(" << phi::DataTypeToString(dst_dtype) << ").";

  if (op_name == "fused_softmax_mask" && input_name == "Mask" &&
      input.dtype() == phi::DataType::FLOAT32) {
    return input;
  }
  if (dst_dtype == phi::DataType::FLOAT16) {
    if (op_name == "run_program") {
      return input;
    }
    if ((op_name == "fused_attention" || op_name == "fused_feedforward")) {
      if (input_name == "LnScale" || input_name == "LnBias" ||
          input_name == "Ln2Scale" || input_name == "Ln2Bias" ||
          input_name == "Ln1Scale" || input_name == "Ln1Bias") {
        return input;
      }
    }
    if ((op_name == "batch_norm" || op_name == "layer_norm" ||
         op_name == "sync_batch_norm" || op_name == "weight_only_linear") &&
        input_name != "x") {
      return input;
    }
  } else if (dst_dtype == phi::DataType::BFLOAT16) {
    if ((op_name == "batch_norm" || op_name == "layer_norm" ||
         op_name == "sync_batch_norm" || op_name == "weight_only_linear") &&
        input_name != "x") {
      return input;
    }
  }

  if (NeedCast(input, dst_dtype)) {
    if (dst_dtype == phi::DataType::FLOAT32) {
      VLOG(5) << "got different data type, run type promotion automatically.";
      LOG_FIRST_N(WARNING, 1)
          << "got different data type, run type promotion automatically, this "
             "may cause data type been changed.";
    }
    paddle::framework::AttributeMap cast_attrs = {
        {"in_dtype", paddle::framework::TransToProtoVarType(input.dtype())},
        {"out_dtype", paddle::framework::TransToProtoVarType(dst_dtype)}};
    return cast_dygraph_function(input, cast_attrs);
  }
  return input;
}

}  // namespace egr
