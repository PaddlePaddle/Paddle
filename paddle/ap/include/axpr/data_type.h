// Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

#include <functional>
#include "paddle/ap/include/axpr/type.h"
#include "paddle/phi/common/data_type.h"
#include "paddle/phi/common/float8_e4m3fn.h"
#include "paddle/phi/common/float8_e5m2.h"
#include "paddle/phi/common/pstring.h"

namespace ap::axpr {

using complex64 = ::phi::dtype::complex<float>;
using complex128 = ::phi::dtype::complex<double>;
using float16 = ::phi::dtype::float16;
using bfloat16 = ::phi::dtype::bfloat16;
using float8_e4m3fn = ::phi::dtype::float8_e4m3fn;
using float8_e5m2 = ::phi::dtype::float8_e5m2;
using pstring = ::phi::dtype::pstring;

#define PEXPR_FOR_EACH_ARITHMETIC_OP_SUPPORTED_TYPE(_) \
  _(bool)                                              \
  _(float)                                             \
  _(double)                                            \
  _(int8_t)                                            \
  _(uint8_t)                                           \
  _(int16_t)                                           \
  _(uint16_t)                                          \
  _(int32_t)                                           \
  _(uint32_t)                                          \
  _(int64_t)                                           \
  _(uint64_t)

namespace detail {

template <typename T>
struct IsArithmeticOpSupportedHelper {
  static constexpr bool value = false;
};

#define SPECIALIZE_IS_ARITHMETIC_OP_SUPPORTED(cpp_type) \
  template <>                                           \
  struct IsArithmeticOpSupportedHelper<cpp_type> {      \
    static constexpr bool value = true;                 \
  };

PEXPR_FOR_EACH_ARITHMETIC_OP_SUPPORTED_TYPE(
    SPECIALIZE_IS_ARITHMETIC_OP_SUPPORTED)

#undef SPECIALIZE_IS_ARITHMETIC_OP_SUPPORTED

}  // namespace detail

template <typename T>
constexpr bool IsArithmeticOpSupported() {
  return detail::IsArithmeticOpSupportedHelper<T>::value;
}

template <typename T>
struct GetDataTypeNameHelper;

#define SPECIALIZE_GET_CPP_TYPE_NAME(cpp_type, enum_type)    \
  template <>                                                \
  struct GetDataTypeNameHelper<cpp_type> {                   \
    static const char* Name() { return #cpp_type; }          \
  };                                                         \
  template <>                                                \
  struct GetDataTypeNameHelper<const cpp_type> {             \
    static const char* Name() { return "const_" #cpp_type; } \
  };
PD_FOR_EACH_DATA_TYPE(SPECIALIZE_GET_CPP_TYPE_NAME);
#undef SPECIALIZE_GET_CPP_TYPE_NAME
template <>
struct GetDataTypeNameHelper<adt::Undefined> {
  static const char* Name() { return "void"; }
};

template <>
struct GetDataTypeNameHelper<const adt::Undefined> {
  static const char* Name() { return "const_void"; }
};

template <typename T>
struct CppDataType : public std::monostate {
  using std::monostate::monostate;
  using type = T;
  const char* Name() const { return GetDataTypeNameHelper<T>::Name(); }
};

// clang-format off
using DataTypeImpl = std::variant<
#define MAKE_ARITHMETIC_TYPE_ALTERNATIVE(cpp_type, enum_type)    \
    CppDataType<cpp_type>,
    PD_FOR_EACH_DATA_TYPE(MAKE_ARITHMETIC_TYPE_ALTERNATIVE)
#undef MAKE_ARITHMETIC_TYPE_ALTERNATIVE
    CppDataType<adt::Undefined>>;
// clang-format on

struct DataType : public DataTypeImpl {
  using DataTypeImpl::DataTypeImpl;
  ADT_DEFINE_VARIANT_METHODS(DataTypeImpl);

  const char* Name() const {
    return Match([](const auto& impl) { return impl.Name(); });
  }

  std::size_t GetHashValue() const { return index(); }
};

template <>
struct TypeImpl<DataType> : public std::monostate {
  using value_type = DataType;

  const char* Name() const { return "DataType"; }
};

}  // namespace ap::axpr

namespace std {

template <>
struct hash<ap::axpr::DataType> {
  std::size_t operator()(ap::axpr::DataType dtype) const {
    return dtype.GetHashValue();
  }
};

}  // namespace std
