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

#if !defined(_WIN32)

#include "paddle/phi/capi/include/c_data_type.h"

namespace phi {

namespace capi {

#define CPP_TYPE_TO_PD_DTYPE_REGISTER(_)                \
  _(bool, PD_DataType::BOOL)                            \
  _(phi::dtype::bfloat16, PD_DataType::BFLOAT16)        \
  _(phi::dtype::float16, PD_DataType::FLOAT16)          \
  _(float, PD_DataType::FLOAT32)                        \
  _(double, PD_DataType::FLOAT64)                       \
  _(uint8_t, PD_DataType::UINT8)                        \
  _(uint16_t, PD_DataType::UINT16)                      \
  _(uint32_t, PD_DataType::UINT32)                      \
  _(uint64_t, PD_DataType::UINT64)                      \
  _(int8_t, PD_DataType::INT8)                          \
  _(int16_t, PD_DataType::INT16)                        \
  _(int32_t, PD_DataType::INT32)                        \
  _(int64_t, PD_DataType::INT64)                        \
  _(phi::dtype::complex<float>, PD_DataType::COMPLEX64) \
  _(phi::dtype::complex<double>, PD_DataType::COMPLEX128)

template <typename T>
struct CppTypeToPDType;

#define CPP_TYPE_TO_PD_DTYPE(x, y)                    \
  template <>                                         \
  struct CppTypeToPDType<x> {                         \
    constexpr static PD_DataType Type() { return y; } \
  };

template <PD_DataType T>
struct PDTypeToCppType;

#define PD_DTYPE_TO_CPP_TYPE(x, y) \
  template <>                      \
  struct PDTypeToCppType<y> {      \
    using type = x;                \
  };

CPP_TYPE_TO_PD_DTYPE_REGISTER(CPP_TYPE_TO_PD_DTYPE)
CPP_TYPE_TO_PD_DTYPE_REGISTER(PD_DTYPE_TO_CPP_TYPE)

}  // namespace capi
}  // namespace phi

#endif
