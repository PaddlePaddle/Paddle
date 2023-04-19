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

static inline phi::DataType GetPromoteType(
    const std::string& op_name,
    const paddle::small_vector<std::vector<paddle::Tensor>,
                               kSlotSmallVectorSize>& amp_tensors_vector,
    const phi::DataType& amp_dtype) {
  auto dst_type = amp_dtype;
  if (egr::Controller::Instance().GetCurrentTracer()->GetAmpDtype() ==
      "float16") {
    if (op_name == "batch_norm" || op_name == "layer_norm" ||
        op_name == "sync_batch_norm") {
      if (amp_tensors_vector[0][0].dtype() == phi::DataType::FLOAT32) {
        dst_type = phi::DataType::FLOAT32;
      }
    } else if (op_name == "fused_attention") {
      for (size_t i = 0; i < amp_tensors_vector.size(); i++) {
        if (i != 3 || i != 4 || i != 9 || i != 10) {
          if (amp_tensors_vector[i][0].dtype() == phi::DataType::FLOAT32) {
            dst_type = phi::DataType::FLOAT32;
            break;
          }
        }
      }
    } else if (op_name == "fused_feedforward") {
      for (size_t i = 0; i < amp_tensors_vector.size(); i++) {
        if (i != 7 || i != 8 || i != 9 || i != 10) {
          if (amp_tensors_vector[i][0].dtype() == phi::DataType::FLOAT32) {
            dst_type = phi::DataType::FLOAT32;
            break;
          }
        }
      }
    } else {
      for (const auto& tensors : amp_tensors_vector) {
        for (const auto& tensor : tensors) {
          if (tensor.dtype() == phi::DataType::FLOAT32) {
            dst_type = tensor.dtype();
            break;
          }
        }
      }
    }
  } else {
    for (const auto& tensors : amp_tensors_vector) {
      for (const auto& tensor : tensors) {
        if (tensor.dtype() == phi::DataType::FLOAT32) {
          dst_type = tensor.dtype();
          break;
        }
      }
    }
  }
  // NOTE(juncai): moving_average_abs_max_scale only consider the dtype of
  // input(X)
  if (op_name == "moving_average_abs_max_scale") {
    if (amp_tensors_vector[0][0].dtype() == phi::DataType::FLOAT16) {
      dst_type = phi::DataType::FLOAT16;
    }
  }
  return dst_type;
}

inline phi::DataType GetDtypeWithPlace(
    const std::string& op_name,
    const paddle::small_vector<std::vector<paddle::Tensor>,
                               kSlotSmallVectorSize>& amp_tensors_vector,
    const phi::DataType amp_dtype) {
  if (amp_dtype == phi::DataType::FLOAT32) {
    return amp_dtype;
  }
  bool is_right_place = false;
  for (const auto& tensors : amp_tensors_vector) {
    for (const auto& tensor : tensors) {
      auto place = tensor.place();
      is_right_place = (paddle::platform::is_gpu_place(place) ||
                        paddle::platform::is_cuda_pinned_place(place) ||
                        paddle::platform::is_xpu_place(place) ||
                        paddle::platform::is_npu_place(place) ||
                        paddle::platform::is_npu_pinned_place(place) ||
                        paddle::platform::is_custom_place(place));
      if (is_right_place) {
        break;
      }
    }
  }

  if (!is_right_place) {
    VLOG(6) << "Change " << op_name << "'s AMP type from " << amp_dtype
            << " to FP32";
    return phi::DataType::FLOAT32;
  }
  return amp_dtype;
}

inline phi::DataType GetAmpDestDtype(
    const std::string& op_name,
    const paddle::small_vector<std::vector<paddle::Tensor>,
                               kSlotSmallVectorSize>& amp_tensors_vector) {
  auto amp_level = egr::Controller::Instance().GetAMPLevel();
  auto amp_setting_dtype =
      egr::Controller::Instance().GetCurrentTracer()->GetAmpPhiDtype();
  auto dst_type = amp_setting_dtype;

  if (paddle::imperative::AmpOperators::Instance().GetMutableAllowOps()->count(
          op_name)) {
    dst_type = amp_setting_dtype;
  } else if (paddle::imperative::AmpOperators::Instance()
                 .GetMutableBlockOps()
                 ->count(op_name)) {
    dst_type = phi::DataType::FLOAT32;
  } else {
    dst_type = GetPromoteType(op_name, amp_tensors_vector, amp_setting_dtype);
  }

  if (dst_type == amp_setting_dtype &&
      (paddle::imperative::AmpOperators::Instance()
           .GetMutableUnsupportedOps(amp_setting_dtype)
           ->count(op_name))) {
    dst_type = phi::DataType::FLOAT32;
  }

  dst_type = GetDtypeWithPlace(op_name, amp_tensors_vector, dst_type);
  VLOG(6) << "AMP GetAmpDestDtype:"
          << " op(" << op_name << ") amp_dtype(" << dst_type << ") amp_level("
          << static_cast<int>(amp_level) << ").";
  return dst_type;
}

}  // namespace egr
