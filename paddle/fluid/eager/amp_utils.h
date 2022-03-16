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
#include <map>
#include <string>
#include "paddle/fluid/eager/api/generated/fluid_generated/dygraph_forward_api.h"
#include "paddle/fluid/framework/convert_utils.h"
#include "paddle/fluid/imperative/amp_auto_cast.h"

namespace egr {

static inline paddle::experimental::DataType GetPromoteType(
    const std::string& api_name,
    const std::map<std::string, std::vector<paddle::experimental::Tensor>>&
        name_tensors_map,
    const paddle::experimental::DataType amp_dtype) {
  VLOG(6) << "AMP: GetPromoteType for op " << api_name;
  auto dst_type = amp_dtype;
  for (const auto& pair : name_tensors_map) {
    for (const auto& tensor : pair.second) {
      if (tensor.dtype() == paddle::experimental::DataType::FLOAT32) {
        dst_type = tensor.dtype();
        break;
      }
    }
  }
  // NOTE(juncai): moving_average_abs_max_scale only consider the
  // dtype of input(X)
  if (api_name == "moving_average_abs_max_scale") {
    for (const auto& pair : name_tensors_map) {
      if (pair.first == "X" &&
          pair.second.front().dtype() ==
              paddle::experimental::DataType::FLOAT16) {
        dst_type = paddle::experimental::DataType::FLOAT16;
      }
    }
  }
  VLOG(6) << "PromoteType is: " << paddle::framework::DataType2String(dst_type);
  return dst_type;
}

// 获取amp数据类型
paddle::experimental::DataType GetAmpDestDtype(
    const std::string& amp_dtype, const paddle::imperative::AmpLevel& amp_level,
    const std::string& api_name,
    const std::map<std::string, std::vector<paddle::experimental::Tensor>>&
        name_tensors_map) {
  VLOG(6) << "AMP GetAmpDestDtype:"
          << " op(" << api_name << ") amp_dtype(" << amp_dtype << ") amp_level("
          << static_cast<int>(amp_level) << ").";
  if (amp_dtype == "float16") {
    if (amp_level == paddle::imperative::AmpLevel::O1) {
      if (paddle::imperative::AmpOperators::Instance()
              .GetMutableAllowOps()
              ->count(api_name)) {
        return paddle::experimental::DataType::FLOAT16;
      } else if (paddle::imperative::AmpOperators::Instance()
                     .GetMutableBlockOps()
                     ->count(api_name)) {
        return paddle::experimental::DataType::FLOAT32;
      } else {
        auto dst_type = GetPromoteType(api_name, name_tensors_map,
                                       paddle::experimental::DataType::FLOAT16);
        if (dst_type == paddle::experimental::DataType::FLOAT16 &&
            paddle::imperative::AmpOperators::Instance()
                .GetMutableUnsupportedFp16Ops()
                ->count(api_name)) {
          dst_type = paddle::experimental::DataType::FLOAT32;
        }
        return dst_type;
      }
    } else if (amp_level == paddle::imperative::AmpLevel::O2) {
      auto dst_type = paddle::experimental::DataType::FLOAT16;
      if (paddle::imperative::AmpOperators::Instance()
              .GetMutableUnsupportedFp16Ops()
              ->count(api_name) ||
          paddle::imperative::AmpOperators::Instance()
              .GetMutableBlockOps()
              ->count(api_name)) {
        dst_type = paddle::experimental::DataType::FLOAT32;
      }
      return dst_type;
    }
  } else if (amp_dtype == "bfloat16") {
    if (amp_level == paddle::imperative::AmpLevel::O1) {
      if (paddle::imperative::AmpOperators::Instance()
              .GetMutableAllowOps()
              ->count(api_name)) {
        return paddle::experimental::DataType::BFLOAT16;
      } else if (paddle::imperative::AmpOperators::Instance()
                     .GetMutableBlockOps()
                     ->count(api_name)) {
        return paddle::experimental::DataType::FLOAT32;
      } else {
        auto dst_type =
            GetPromoteType(api_name, name_tensors_map,
                           paddle::experimental::DataType::BFLOAT16);
        if (dst_type == paddle::experimental::DataType::BFLOAT16 &&
            paddle::imperative::AmpOperators::Instance()
                .GetMutableUnsupportedBf16Ops()
                ->count(api_name)) {
          dst_type = paddle::experimental::DataType::FLOAT32;
        }
        return dst_type;
      }
    } else if (amp_level == paddle::imperative::AmpLevel::O2) {
      auto dst_type = paddle::experimental::DataType::BFLOAT16;
      if (paddle::imperative::AmpOperators::Instance()
              .GetMutableUnsupportedBf16Ops()
              ->count(api_name) ||
          paddle::imperative::AmpOperators::Instance()
              .GetMutableBlockOps()
              ->count(api_name)) {
        dst_type = paddle::experimental::DataType::FLOAT32;
      }
      return dst_type;
    }
  }
  return paddle::experimental::DataType::FLOAT32;
}

static inline bool NeedCast(const paddle::experimental::Tensor& tensor,
                            const paddle::experimental::DataType& dst_dtype) {
  auto place = tensor.inner_place();
  auto data_type = tensor.dtype();
  if (paddle::platform::is_gpu_place(place) ||
      paddle::platform::is_cuda_pinned_place(place) ||
      paddle::platform::is_xpu_place(place)) {
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

std::vector<paddle::experimental::Tensor> AmpAutoCasts(
    const std::string& inputs_name,
    const std::vector<paddle::experimental::Tensor>& inputs,
    const paddle::experimental::DataType& dst_dtype, std::string api_name) {
  VLOG(6) << "AMP AmpAutoCasts:"
          << " inputs(" << inputs_name << ") dst_dtype("
          << paddle::framework::DataType2String(dst_dtype) << ").";
  std::vector<paddle::experimental::Tensor> inputs_casted = inputs;
  for (auto& input : inputs_casted) {
    if (NeedCast(input, dst_dtype)) {
      paddle::framework::AttributeMap cast_attrs = {
          {"in_dtype", paddle::framework::TransToProtoVarType(input.dtype())},
          {"out_dtype", paddle::framework::TransToProtoVarType(dst_dtype)}};
      input = cast_dygraph_function(input, cast_attrs);
    }
  }
  return inputs_casted;
}

paddle::experimental::Tensor AmpAutoCast(
    const std::string& input_name, const paddle::experimental::Tensor& input,
    const paddle::experimental::DataType& dst_dtype, std::string api_name) {
  VLOG(6) << "AMP AmpAutoCasts:"
          << " input(" << input_name << ") dst_dtype("
          << paddle::framework::DataType2String(dst_dtype) << ").";
  paddle::experimental::Tensor input_casted = input;

  if (dst_dtype == paddle::experimental::DataType::FLOAT16) {
    if (api_name == "run_program") {
      return input_casted;
    }
    if ((api_name == "batch_norm" || api_name == "layer_norm" ||
         api_name == "sync_batch_norm") &&
        input_name != "X") {
      return input_casted;
    }
    if ((api_name == "fused_attention" || api_name == "fused_feedforward")) {
      if (input_name == "LnScale" || input_name == "LnBias" ||
          input_name == "Ln2Scale" || input_name == "Ln2Bias" ||
          input_name == "Ln1Scale" || input_name == "Ln1Bias") {
        return input_casted;
      }
    }
  }

  if (NeedCast(input, dst_dtype)) {
    paddle::framework::AttributeMap cast_attrs = {
        {"in_dtype", paddle::framework::TransToProtoVarType(input.dtype())},
        {"out_dtype", paddle::framework::TransToProtoVarType(dst_dtype)}};
    input_casted = cast_dygraph_function(input, cast_attrs);
  }
  return input_casted;
}

}  // namespace egr
