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

#include "paddle/fluid/eager/api/generated/eager_generated/forwards/dygraph_functions.h"

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

inline paddle::experimental::Tensor Cast(
    const paddle::experimental::Tensor& input,
    const paddle::experimental::DataType& dst_dtype,
    const bool trace_backward = true) {
  if (input.is_sparse_coo_tensor() || input.is_sparse_csr_tensor()) {
    if (trace_backward) {
      return sparse::cast_ad_func(
          input, paddle::experimental::DataType::UNDEFINED, dst_dtype);
    } else {
      return paddle::experimental::sparse::cast(
          input, paddle::experimental::DataType::UNDEFINED, dst_dtype);
    }
  } else {
    if (trace_backward) {
      return cast_ad_func(input, dst_dtype);
    } else {
      return paddle::experimental::cast(input, dst_dtype);
    }
  }
}

inline std::vector<paddle::experimental::Tensor> EagerAmpAutoCasts(
    const std::string& inputs_name,
    const std::vector<paddle::experimental::Tensor>& inputs,
    const paddle::experimental::DataType& dst_dtype,
    std::string op_name,
    bool trace_backward = true) {
  VLOG(6) << "AMP AmpAutoCasts:"
          << " inputs(" << inputs_name << ") dst_dtype("
          << paddle::framework::DataType2String(dst_dtype) << ").";
  std::vector<paddle::experimental::Tensor> inputs_casted;
  for (auto& input : inputs) {
    if (NeedCast(input, dst_dtype)) {
      inputs_casted.emplace_back(std::move(Cast(input, dst_dtype)));
    } else {
      inputs_casted.emplace_back(input);
    }
  }
  return inputs_casted;
}

inline paddle::experimental::Tensor EagerAmpAutoCast(
    const std::string& input_name,
    const paddle::experimental::Tensor& input,
    const paddle::experimental::DataType& dst_dtype,
    const std::string& op_name,
    bool trace_backward = true) {
  VLOG(6) << "AMP AmpAutoCasts:"
          << " input(" << input_name << ") dst_dtype("
          << paddle::framework::DataType2String(dst_dtype) << ").";
  if (dst_dtype == paddle::experimental::DataType::FLOAT16) {
    if (op_name == "run_program") {
      return input;
    }
    if ((op_name == "batch_norm" || op_name == "layer_norm" ||
         op_name == "sync_batch_norm") &&
        input_name != "x") {
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
    return Cast(input, dst_dtype, trace_backward);
  }
  return input;
}

inline paddle::optional<paddle::experimental::Tensor> EagerAmpAutoCast(
    const std::string& input_name,
    const paddle::optional<paddle::experimental::Tensor>& input,
    const paddle::experimental::DataType& dst_dtype,
    const std::string& op_name,
    bool trace_backward = true) {
  if (input) {
    return EagerAmpAutoCast(
        input_name, *input, dst_dtype, op_name, trace_backward);
  }
  return paddle::none;
}

inline paddle::optional<std::vector<paddle::experimental::Tensor>>
EagerAmpAutoCasts(
    const std::string& inputs_name,
    const paddle::optional<std::vector<paddle::experimental::Tensor>>& inputs,
    const paddle::experimental::DataType& dst_dtype,
    std::string op_name,
    bool trace_backward = true) {
  if (inputs) {
    return EagerAmpAutoCasts(
        inputs_name, *inputs, dst_dtype, op_name, trace_backward);
  }
  return paddle::optional<std::vector<paddle::experimental::Tensor>>();
}

}  // namespace egr
