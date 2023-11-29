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
#include "paddle/fluid/imperative/type_promotion.h"

namespace egr {

inline int DataTypeToNum(const phi::DataType& dtype) {
  switch (dtype) {
    case phi::DataType::UINT8:
      return 0;
    case phi::DataType::INT8:
      return 1;
    case phi::DataType::INT16:
      return 2;
    case phi::DataType::INT32:
      return 3;
    case phi::DataType::INT64:
      return 4;
    case phi::DataType::FLOAT16:
      return 5;
    case phi::DataType::FLOAT32:
      return 6;
    case phi::DataType::FLOAT64:
      return 7;
    case phi::DataType::COMPLEX64:
      return 8;
    case phi::DataType::COMPLEX128:
      return 9;
    case phi::DataType::BOOL:
      return 10;
    case phi::DataType::BFLOAT16:
      return 11;
    default:
      PD_THROW("Invalid enum data type for type promote `", dtype, "`.");
  }
}

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

inline static phi::DataType promoteTypes(phi::DataType a, phi::DataType b) {
  constexpr auto u1 = phi::DataType::UINT8;
  constexpr auto i1 = phi::DataType::INT8;
  constexpr auto i2 = phi::DataType::INT16;
  constexpr auto i4 = phi::DataType::INT32;
  constexpr auto i8 = phi::DataType::INT64;
  constexpr auto f2 = phi::DataType::FLOAT16;
  constexpr auto f4 = phi::DataType::FLOAT32;
  constexpr auto f8 = phi::DataType::FLOAT64;
  constexpr auto c4 = phi::DataType::COMPLEX64;
  constexpr auto c8 = phi::DataType::COMPLEX128;
  constexpr auto b1 = phi::DataType::BOOL;
  constexpr auto bf = phi::DataType::BFLOAT16;

  static constexpr phi::DataType _promoteTypesLookup[12][12] = {
      /*        u1  i1  i2  i4  i8  f2  f4  f8  c4  c8  b1  bf*/
      /* u1 */ {u1, i2, i2, i4, i8, f2, f4, f8, c4, c8, u1, bf},
      /* i1 */ {i2, i1, i2, i4, i8, f2, f4, f8, c4, c8, i1, bf},
      /* i2 */ {i2, i2, i2, i4, i8, f2, f4, f8, c4, c8, i2, bf},
      /* i4 */ {i4, i4, i4, i4, i8, f2, f4, f8, c4, c8, i4, bf},
      /* i8 */ {i8, i8, i8, i8, i8, f2, f4, f8, c4, c8, i8, bf},
      /* f2 */ {f2, f2, f2, f2, f2, f2, f4, f8, c4, c8, f2, f4},
      /* f4 */ {f4, f4, f4, f4, f4, f4, f4, f8, c4, c8, f4, f4},
      /* f8 */ {f8, f8, f8, f8, f8, f8, f8, f8, c8, c8, f8, f8},
      /* c4 */ {c4, c4, c4, c4, c4, c4, c4, c8, c4, c8, c4, c4},
      /* c8 */ {c8, c8, c8, c8, c8, c8, c8, c8, c8, c8, c8, c8},
      /* b1 */ {u1, i1, i2, i4, i8, f2, f4, f8, c4, c8, b1, bf},
      /* bf */ {bf, bf, bf, bf, bf, f4, f4, f8, c4, c8, bf, bf},
  };
  return _promoteTypesLookup[DataTypeToNum(a)][DataTypeToNum(b)];
}

inline phi::DataType GetPromoteDtype(
    const std::string& op_name,
    const paddle::small_vector<std::vector<paddle::Tensor>,
                               kSlotSmallVectorSize>& promote_tensors_vector) {
  return promoteTypes(promote_tensors_vector[0][0].dtype(),
                      promote_tensors_vector[1][0].dtype());
}

inline bool NeedTypePromotion(
    const paddle::small_vector<std::vector<paddle::Tensor>,
                               kSlotSmallVectorSize>& promote_tensors_vector) {
  // T+T only support type promotion between float, int32, int64
  if ((promote_tensors_vector[0][0].dtype() !=
       promote_tensors_vector[1][0].dtype()) &&
      (is_support_float(a) || is_support_int(a)) &&
      (is_support_float(b) || is_support_int(b))) {
    return true;
  } else {
    return false;
  }
}

}  // namespace egr
