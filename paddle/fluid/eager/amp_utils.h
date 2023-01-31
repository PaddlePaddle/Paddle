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

#include "paddle/fluid/eager/api/utils/global_utils.h"
#include "paddle/fluid/imperative/amp_auto_cast.h"

namespace egr {

static inline paddle::experimental::DataType GetPromoteType(
    const std::string& op_name,
    const paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                               kSlotSmallVectorSize>& amp_tensors_vector,
    const paddle::experimental::DataType& amp_dtype) {
  auto dst_type = amp_dtype;
  if (egr::Controller::Instance().GetCurrentTracer()->GetAmpDtype() ==
      "float16") {
    if (op_name == "batch_norm" || op_name == "layer_norm" ||
        op_name == "sync_batch_norm") {
      if (amp_tensors_vector[0][0].dtype() ==
          paddle::experimental::DataType::FLOAT32) {
        dst_type = paddle::experimental::DataType::FLOAT32;
      }
    } else if (op_name == "fused_attention") {
      for (size_t i = 0; i < amp_tensors_vector.size(); i++) {
        if (i != 3 || i != 4 || i != 9 || i != 10) {
          if (amp_tensors_vector[i][0].dtype() ==
              paddle::experimental::DataType::FLOAT32) {
            dst_type = paddle::experimental::DataType::FLOAT32;
            break;
          }
        }
      }
    } else if (op_name == "fused_feedforward") {
      for (size_t i = 0; i < amp_tensors_vector.size(); i++) {
        if (i != 7 || i != 8 || i != 9 || i != 10) {
          if (amp_tensors_vector[i][0].dtype() ==
              paddle::experimental::DataType::FLOAT32) {
            dst_type = paddle::experimental::DataType::FLOAT32;
            break;
          }
        }
      }
    } else {
      for (const auto& tensors : amp_tensors_vector) {
        for (const auto& tensor : tensors) {
          if (tensor.dtype() == paddle::experimental::DataType::FLOAT32) {
            dst_type = tensor.dtype();
            break;
          }
        }
      }
    }
  } else {
    for (const auto& tensors : amp_tensors_vector) {
      for (const auto& tensor : tensors) {
        if (tensor.dtype() == paddle::experimental::DataType::FLOAT32) {
          dst_type = tensor.dtype();
          break;
        }
      }
    }
  }
  // NOTE(juncai): moving_average_abs_max_scale only consider the dtype of
  // input(X)
  if (op_name == "moving_average_abs_max_scale") {
    if (amp_tensors_vector[0][0].dtype() ==
        paddle::experimental::DataType::FLOAT16) {
      dst_type = paddle::experimental::DataType::FLOAT16;
    }
  }
  return dst_type;
}

inline paddle::experimental::DataType GetAmpDestDtype(
    const std::string& op_name,
    const paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                               kSlotSmallVectorSize>& amp_tensors_vector) {
  auto amp_dtype =
      egr::Controller::Instance().GetCurrentTracer()->GetAmpDtype();
  auto amp_level = egr::Controller::Instance().GetAMPLevel();
  VLOG(6) << "AMP GetAmpDestDtype:"
          << " op(" << op_name << ") amp_dtype(" << amp_dtype << ") amp_level("
          << static_cast<int>(amp_level) << ").";
  if (amp_dtype == "float16") {
    if (amp_level == paddle::imperative::AmpLevel::O1) {
      if (paddle::imperative::AmpOperators::Instance()
              .GetMutableAllowOps()
              ->count(op_name)) {
        return paddle::experimental::DataType::FLOAT16;
      } else if (paddle::imperative::AmpOperators::Instance()
                     .GetMutableBlockOps()
                     ->count(op_name) ||
                 paddle::imperative::AmpOperators::Instance()
                     .GetMutableUnsupportedFp16Ops()
                     ->count(op_name)) {
        return paddle::experimental::DataType::FLOAT32;
      } else {
        auto dst_type = GetPromoteType(op_name,
                                       amp_tensors_vector,
                                       paddle::experimental::DataType::FLOAT16);
        if (dst_type == paddle::experimental::DataType::FLOAT16 &&
            paddle::imperative::AmpOperators::Instance()
                .GetMutableUnsupportedFp16Ops()
                ->count(op_name)) {
          dst_type = paddle::experimental::DataType::FLOAT32;
        }
        return dst_type;
      }
    } else if (amp_level == paddle::imperative::AmpLevel::O2) {
      auto dst_type = paddle::experimental::DataType::FLOAT16;
      if (paddle::imperative::AmpOperators::Instance()
              .GetMutableUnsupportedFp16Ops()
              ->count(op_name) ||
          paddle::imperative::AmpOperators::Instance()
              .GetMutableBlockOps()
              ->count(op_name)) {
        dst_type = paddle::experimental::DataType::FLOAT32;
      }
      return dst_type;
    }
  } else if (amp_dtype == "bfloat16") {
    if (amp_level == paddle::imperative::AmpLevel::O1) {
      if (paddle::imperative::AmpOperators::Instance()
              .GetMutableAllowOps()
              ->count(op_name)) {
        return paddle::experimental::DataType::BFLOAT16;
      } else if (paddle::imperative::AmpOperators::Instance()
                     .GetMutableBlockOps()
                     ->count(op_name)) {
        return paddle::experimental::DataType::FLOAT32;
      } else {
        auto dst_type =
            GetPromoteType(op_name,
                           amp_tensors_vector,
                           paddle::experimental::DataType::BFLOAT16);
        if (dst_type == paddle::experimental::DataType::BFLOAT16 &&
            paddle::imperative::AmpOperators::Instance()
                .GetMutableUnsupportedBf16Ops()
                ->count(op_name)) {
          dst_type = paddle::experimental::DataType::FLOAT32;
        }
        return dst_type;
      }
    } else if (amp_level == paddle::imperative::AmpLevel::O2) {
      auto dst_type = paddle::experimental::DataType::BFLOAT16;
      if (paddle::imperative::AmpOperators::Instance()
              .GetMutableUnsupportedBf16Ops()
              ->count(op_name) ||
          paddle::imperative::AmpOperators::Instance()
              .GetMutableBlockOps()
              ->count(op_name)) {
        dst_type = paddle::experimental::DataType::FLOAT32;
      }
      return dst_type;
    }
  }
  return paddle::experimental::DataType::FLOAT32;
}

}  // namespace egr
