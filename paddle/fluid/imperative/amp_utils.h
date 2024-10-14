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

#if !(defined(PADDLE_NO_PYTHON) && defined(PADDLE_ON_INFERENCE))
#include "paddle/fluid/eager/api/generated/eager_generated/forwards/dygraph_functions.h"
#endif
#include "paddle/fluid/eager/api/utils/global_utils.h"
#include "paddle/fluid/eager/type_defs.h"
#include "paddle/fluid/imperative/amp_auto_cast.h"
#include "paddle/fluid/pir/dialect/operator/ir/pd_api.h"
#include "paddle/fluid/pir/dialect/operator/utils/utils.h"
#include "paddle/phi/api/include/api.h"
#include "paddle/phi/api/include/sparse_api.h"
#include "paddle/utils/small_vector.h"

namespace paddle {
namespace imperative {
static inline phi::DataType GetDataType(const pir::Value& value) {
  return paddle::dialect::GetValueDataType(value);
}

static inline phi::DataType GetDataType(const paddle::Tensor& tensor) {
  return tensor.dtype();
}

template <class T>
static inline phi::DataType GetPromoteType(
    const std::string& op_name,
    const paddle::small_vector<std::vector<T>, egr::kSlotSmallVectorSize>&
        amp_tensors_vector,
    const phi::DataType& amp_dtype) {
  auto dst_type = amp_dtype;
  // only consider the dtype of input(X).
  if (op_name == "batch_norm" || op_name == "layer_norm" ||
      op_name == "sync_batch_norm" ||
      op_name == "moving_average_abs_max_scale") {
    if (GetDataType(amp_tensors_vector[0][0]) == phi::DataType::FLOAT32) {
      dst_type = phi::DataType::FLOAT32;
    }
    return dst_type;
  }

  if (egr::Controller::Instance().GetCurrentTracer()->GetAmpDtype() ==
      "float16") {
    if (op_name == "fused_attention") {
      for (size_t i = 0; i < amp_tensors_vector.size(); i++) {
        if (i < 3 || (i > 4 && i < 9) || i > 10) {
          if (GetDataType(amp_tensors_vector[i][0]) == phi::DataType::FLOAT32) {
            dst_type = phi::DataType::FLOAT32;
            return dst_type;
          }
        }
      }
    } else if (op_name == "fused_feedforward") {
      for (size_t i = 0; i < amp_tensors_vector.size(); i++) {
        if (i < 7 || i > 10) {
          if (GetDataType(amp_tensors_vector[i][0]) == phi::DataType::FLOAT32) {
            dst_type = phi::DataType::FLOAT32;
            return dst_type;
          }
        }
      }
    }
  }

  for (const auto& tensors : amp_tensors_vector) {
    for (const auto& tensor : tensors) {
      if (GetDataType(tensor) == phi::DataType::FLOAT32) {
        dst_type = GetDataType(tensor);
        break;
      }
    }
  }

  return dst_type;
}

static inline phi::DataType GetDtypeWithPlace(
    const std::string& op_name,
    const paddle::small_vector<std::vector<paddle::Tensor>,
                               egr::kSlotSmallVectorSize>& amp_tensors_vector,
    const phi::DataType amp_dtype) {
  if (amp_dtype == phi::DataType::FLOAT32) {
    return amp_dtype;
  }
  bool is_right_place = false;
  for (const auto& tensors : amp_tensors_vector) {
    for (const auto& tensor : tensors) {
      auto place = tensor.place();
      // TODO(lizhiyu): If the tensor is a dist-tensor, it's place may be
      // `unknown` in the no-calculation rank right now.
      //       We use `is_dist_tensor()` to avoid the bug temporarily. The
      //       dist-tensor in the no-calculation rank should have the right
      //       place.
      is_right_place =
          (tensor.is_dist_tensor() || phi::is_gpu_place(place) ||
           phi::is_cuda_pinned_place(place) || phi::is_xpu_place(place) ||
           phi::is_custom_place(place));
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

static inline phi::DataType GetDtypeWithPlace(
    const std::string& op_name UNUSED,
    const paddle::small_vector<std::vector<pir::Value>,
                               egr::kSlotSmallVectorSize>& amp_tensors_vector
        UNUSED,
    const phi::DataType amp_dtype) {
  return amp_dtype;
}

template <class T>
inline phi::DataType GetAmpDestDtype(
    const std::string& op_name,
    const paddle::small_vector<std::vector<T>, egr::kSlotSmallVectorSize>&
        amp_tensors_vector) {
  auto amp_level = egr::Controller::Instance().GetAMPLevel();
  auto amp_setting_dtype =
      egr::Controller::Instance().GetCurrentTracer()->GetAmpPhiDtype();
  auto dst_type = amp_setting_dtype;

  bool use_promote = true;
  if (amp_level == paddle::imperative::AmpLevel::O2) {
    use_promote =
        egr::Controller::Instance().GetCurrentTracer()->GetUsePromote();
  }

  if (use_promote) {
    if (paddle::imperative::AmpOperators::Instance()
            .GetMutableAllowOps()
            ->count(op_name)) {
      dst_type = amp_setting_dtype;
    } else if (paddle::imperative::AmpOperators::Instance()
                   .GetMutableBlockOps()
                   ->count(op_name)) {
      dst_type = phi::DataType::FLOAT32;
    } else {
      if (amp_level == paddle::imperative::AmpLevel::OD) {
        dst_type = phi::DataType::FLOAT32;
      } else {
        dst_type =
            GetPromoteType(op_name, amp_tensors_vector, amp_setting_dtype);
      }
    }
  } else {
    // use_promote can be set to false only for O2 training.
    if (paddle::imperative::AmpOperators::Instance()
            .GetMutableBlockOps()
            ->count(op_name)) {
      dst_type = phi::DataType::FLOAT32;
    }
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

static inline bool NeedCast(const paddle::Tensor& tensor,
                            const phi::DataType& dst_dtype) {
  auto place = tensor.place();
  auto data_type = tensor.dtype();
  // Except CPU judgment, other conditions should be consistent with
  // amp_utils.h's judgment
  if (phi::is_gpu_place(place) || phi::is_cuda_pinned_place(place) ||
      phi::is_xpu_place(place) || phi::is_custom_place(place) ||
      phi::is_cpu_place(place)) {
    // CudaPinnedPlace is added for varbase created by dataloader
    // Cpu place is for different place tensor, when input1 is cpu and input2
    // is gpu
    if ((data_type == phi::DataType::FLOAT32 ||
         data_type == phi::DataType::FLOAT16 ||
         data_type == phi::DataType::BFLOAT16) &&
        (data_type != dst_dtype)) {
      return true;
    }
  }
  return false;
}

static inline bool NeedCast(const pir::Value& value,
                            const phi::DataType& dst_dtype) {
  auto data_type = paddle::dialect::GetValueDataType(value);
  if ((data_type == phi::DataType::FLOAT32 ||
       data_type == phi::DataType::FLOAT16 ||
       data_type == phi::DataType::BFLOAT16) &&
      (data_type != dst_dtype)) {
    return true;
  }
  return false;
}

#if !(defined(PADDLE_NO_PYTHON) && defined(PADDLE_ON_INFERENCE))
static inline paddle::Tensor Cast(const paddle::Tensor& input,
                                  const phi::DataType& dst_dtype,
                                  const bool trace_backward = true) {
  if (input.is_sparse_coo_tensor() || input.is_sparse_csr_tensor()) {
    if (trace_backward) {
      return sparse::cast_ad_func(input, phi::DataType::UNDEFINED, dst_dtype);
    } else {
      return paddle::experimental::sparse::cast(
          input, phi::DataType::UNDEFINED, dst_dtype);
    }
  } else {
    if (trace_backward) {
      return cast_ad_func(input, dst_dtype);
    } else {
      return paddle::experimental::cast(input, dst_dtype);
    }
  }
}
#endif

static inline pir::Value Cast(const pir::Value& input,
                              const phi::DataType& dst_dtype,
                              const bool trace_backward UNUSED = true) {
  paddle::imperative::AutoCastGuard guard(
      egr::Controller::Instance().GetCurrentAmpAttrs(),
      paddle::imperative::AmpLevel::O0);
  return paddle::dialect::cast(input, dst_dtype);
}

template <class T>
inline std::vector<T> AmpAutoCasts(const std::string& inputs_name,
                                   const std::vector<T>& inputs,
                                   const phi::DataType& dst_dtype,
                                   std::string op_name UNUSED,
                                   bool trace_backward UNUSED = true) {
  VLOG(6) << "AMP AmpAutoCasts:"
          << " inputs(" << inputs_name << ") dst_dtype("
          << phi::DataTypeToString(dst_dtype) << ").";
  std::vector<T> inputs_casted;
  for (auto& input : inputs) {
    if (NeedCast(input, dst_dtype)) {
      inputs_casted.emplace_back(std::move(Cast(input, dst_dtype)));
    } else {
      inputs_casted.emplace_back(input);
    }
  }
  return inputs_casted;
}

template <class T>
inline T AmpAutoCast(const std::string& input_name,
                     const T& input,
                     const phi::DataType& dst_dtype,
                     const std::string& op_name,
                     bool trace_backward = true) {
  VLOG(6) << "AMP AmpAutoCasts: op_name(" << op_name << ")input(" << input_name
          << ") dst_dtype(" << phi::DataTypeToString(dst_dtype) << ").";

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
      if (input_name == "ln_scale" || input_name == "ln_bias" ||
          input_name == "ln_scale_2" || input_name == "ln_bias_2" ||
          input_name == "ln1_scale" || input_name == "ln1_bias" ||
          input_name == "ln2_scale" || input_name == "ln2_bias") {
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
    VLOG(6) << "Input : " << input.impl() << "NeedCast";
    return Cast(input, dst_dtype, trace_backward);
  }
  return input;
}

template <class T>
inline paddle::optional<T> AmpAutoCast(const std::string& input_name,
                                       const paddle::optional<T>& input,
                                       const phi::DataType& dst_dtype,
                                       const std::string& op_name,
                                       bool trace_backward = true) {
  if (input) {
    return AmpAutoCast(input_name, *input, dst_dtype, op_name, trace_backward);
  }
  return paddle::none;
}

template <class T>
inline paddle::optional<std::vector<T>> AmpAutoCasts(
    const std::string& inputs_name,
    const paddle::optional<std::vector<T>>& inputs,
    const phi::DataType& dst_dtype,
    std::string op_name,
    bool trace_backward = true) {
  if (inputs) {
    return AmpAutoCasts(
        inputs_name, *inputs, dst_dtype, op_name, trace_backward);
  }
  return paddle::optional<std::vector<T>>();
}

}  // namespace imperative
}  // namespace paddle
