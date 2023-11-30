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

#include "paddle/phi/common/data_type.h"
namespace phi {

inline int DataTypeToNum(const DataType& dtype) {
  switch (dtype) {
    case DataType::UINT8:
      return 0;
    case DataType::INT8:
      return 1;
    case DataType::INT16:
      return 2;
    case DataType::INT32:
      return 3;
    case DataType::INT64:
      return 4;
    case DataType::FLOAT16:
      return 5;
    case DataType::FLOAT32:
      return 6;
    case DataType::FLOAT64:
      return 7;
    case DataType::COMPLEX64:
      return 8;
    case DataType::COMPLEX128:
      return 9;
    case DataType::BOOL:
      return 10;
    case DataType::BFLOAT16:
      return 11;
    default:
      PD_THROW("Invalid enum data type for type promote `", dtype, "`.");
  }
}

inline static DataType promoteTypes(DataType x, DataType y) {
  constexpr auto u1 = DataType::UINT8;
  constexpr auto i1 = DataType::INT8;
  constexpr auto i2 = DataType::INT16;
  constexpr auto i4 = DataType::INT32;
  constexpr auto i8 = DataType::INT64;
  constexpr auto f2 = DataType::FLOAT16;
  constexpr auto f4 = DataType::FLOAT32;
  constexpr auto f8 = DataType::FLOAT64;
  constexpr auto c4 = DataType::COMPLEX64;
  constexpr auto c8 = DataType::COMPLEX128;
  constexpr auto b1 = DataType::BOOL;
  constexpr auto bf = DataType::BFLOAT16;

  const int total_type_num = 12;

  static constexpr DataType
      _promoteTypesLookup[total_type_num][total_type_num] = {
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
  return _promoteTypesLookup[DataTypeToNum(x)][DataTypeToNum(y)];
}

static inline bool is_support_float(DataType dtype) {
  if (dtype == DataType::FLOAT16 || dtype == DataType::FLOAT32 ||
      dtype == DataType::FLOAT64 || dtype == DataType::BFLOAT16) {
    return true;
  } else {
    return false;
  }
}

inline phi::DataType GetPromoteDtype(const std::string& op_name,
                                     const DataType x,
                                     const DataType y) {
  // future will deal this by different rule
  if (op_name == "greater_than") {
    // bool logic
    return DataType::BOOL;
  } else {
    return phi::promoteTypes(x, y);
  }
}

inline bool NeedTypePromotion(const DataType x, const DataType y) {
  // Tensor + Tensor only support type promotion for float type
  if ((x != y) && is_support_float(x) && is_support_float(y)) {
    return true;
  } else {
    return false;
  }
}

}  // namespace phi
