// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/eager/api/utils/global_utils.h"
#include "paddle/phi/common/type_promotion_table.h"

namespace egr {

inline paddle::Tensor PromoteCast(const std::string& input_name,
                                  const paddle::Tensor& input,
                                  const phi::DataType& dst_dtype,
                                  bool trace_backward = true) {
  if (input.dtype() != dst_dtype) {
    return Cast(input, dst_dtype, trace_backward);
  } else {
    return input;
  }
}

static inline bool is_support_float(phi::DataType dtype) {
  if (dtype == phi::DataType::FLOAT16 || dtype == phi::DataType::FLOAT32 ||
      dtype == phi::DataType::FLOAT64 || dtype == phi::DataType::BFLOAT16) {
    return true;
  } else {
    return false;
  }
}

static inline bool is_support_int(phi::DataType dtype) {
  if (dtype == phi::DataType::INT32 || dtype == phi::DataType::INT64) {
    return true;
  } else {
    return false;
  }
}

inline phi::DataType GetPromoteDtype(
    const std::string& op_name,
    const paddle::small_vector<std::vector<paddle::Tensor>,
                               kSlotSmallVectorSize>& promote_tensors_vector) {
  return phi::promoteTypes(promote_tensors_vector[0][0].dtype(),
                           promote_tensors_vector[1][0].dtype());
}

inline bool NeedTypePromotion(
    const paddle::small_vector<std::vector<paddle::Tensor>,
                               kSlotSmallVectorSize>& promote_tensors_vector) {
  // Tensor + Tensor only support type promotion between float, int32, int64
  phi::DataType a = promote_tensors_vector[0][0].dtype();
  phi::DataType b = promote_tensors_vector[1][0].dtype();
  if ((a != b) && (is_support_float(a) || is_support_int(a)) &&
      (is_support_float(b) || is_support_int(b))) {
    return true;
  } else {
    return false;
  }
}

}  // namespace egr
