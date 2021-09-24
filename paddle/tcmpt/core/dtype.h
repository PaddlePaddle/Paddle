/* Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#pragma once

#include <ostream>

// See Note [ Why still include the fluid headers? ]
#include "paddle/fluid/platform/bfloat16.h"
#include "paddle/fluid/platform/complex.h"
#include "paddle/fluid/platform/float16.h"

namespace pt {

using complex64 = paddle::platform::complex<float>;
using complex128 = paddle::platform::complex<double>;
using float16 = paddle::platform::float16;
using bfloat16 = paddle::platform::bfloat16;

/**
 * [ Why need new data type? ]
 *
 * The Var data type design in framework.proto is confusing, maybe we need
 * polish the VarType in framework.proto.
 *
 * We need to ensure that the operator library is relatively independent
 * and does not depend on the framework. Therefore, before calling the kernel
 * in the Tensor Compute library inside the framework, the internal
 * data type needs to be converted to the data type in the Tensor Compute
 * library.
 *
 */
enum class DataType {
  kUndef = 0,
  kBOOL,
  kINT8,   // Char
  kUINT8,  // BYte
  kINT16,
  kINT32,
  kINT64,
  kBFLOAT16,
  kFLOAT16,
  kFLOAT32,
  kFLOAT64,
  kCOMPLEX64,
  kCOMPLEX128,
  kNumDataTypes,
};

std::ostream& operator<<(std::ostream& os, DataType dtype);

#define PT_FOR_EACH_DATA_TYPE(_)     \
  _(bool, DataType::kBOOL)           \
  _(int8_t, DataType::kINT8)         \
  _(uint8_t, DataType::kUINT8)       \
  _(int16_t, DataType::kINT16)       \
  _(int, DataType::kINT32)           \
  _(int64_t, DataType::kINT64)       \
  _(bfloat16, DataType::kBFLOAT16)   \
  _(float16, DataType::kFLOAT16)     \
  _(float, DataType::kFLOAT32)       \
  _(double, DataType::kFLOAT64)      \
  _(complex64, DataType::kCOMPLEX64) \
  _(complex128, DataType::kCOMPLEX128)

template <pt::DataType T>
struct DataTypeToCppType;

template <typename T>
struct CppTypeToDataType;

#define PT_SPECIALIZE_DataTypeToCppType(cpp_type, data_type) \
  template <>                                                \
  struct DataTypeToCppType<data_type> {                      \
    using type = cpp_type;                                   \
  };

PT_FOR_EACH_DATA_TYPE(PT_SPECIALIZE_DataTypeToCppType)

#undef PT_SPECIALIZE_DataTypeToCppType

#define PT_SPECIALIZE_CppTypeToDataType(cpp_type, data_type) \
  template <>                                                \
  struct CppTypeToDataType<cpp_type> {                       \
    constexpr static DataType Type() { return data_type; }   \
  };

PT_FOR_EACH_DATA_TYPE(PT_SPECIALIZE_CppTypeToDataType)

#undef PT_SPECIALIZE_CppTypeToDataType

}  // namespace pt
