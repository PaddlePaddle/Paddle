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

#include "paddle/ap/include/axpr/data_value.h"

namespace ap::axpr {

// Same root cause as the explicit-instantiation workaround in
// interpreter.cc: xtrans/clang's `if constexpr` branch inside a
// std::visit lambda (see DataValue::StaticCastTo in data_value.h, which
// dispatches to `this->DataValueStaticCast<DstT>(cpp_value_impl)` for
// every (DstT, SrcT) pair from PD_FOR_EACH_DATA_TYPE cross product) does
// not implicitly instantiate the private member function template the
// way GCC/system-clang do; every (DstT, SrcT) combination ends up as an
// unresolved external reference at link time. Force explicit
// instantiation for all pairs so the definitions are actually emitted
// into this translation unit.
#define AP_EXPLICIT_INSTANTIATE_DATA_VALUE_STATIC_CAST_ROW(DstT)             \
  template Result<DataValue> DataValue::DataValueStaticCast<DstT, bool>(     \
      bool) const;                                                          \
  template Result<DataValue> DataValue::DataValueStaticCast<DstT, int8_t>(   \
      int8_t) const;                                                        \
  template Result<DataValue> DataValue::DataValueStaticCast<DstT, uint8_t>(  \
      uint8_t) const;                                                       \
  template Result<DataValue> DataValue::DataValueStaticCast<DstT, int16_t>(  \
      int16_t) const;                                                       \
  template Result<DataValue> DataValue::DataValueStaticCast<DstT, uint16_t>( \
      uint16_t) const;                                                      \
  template Result<DataValue> DataValue::DataValueStaticCast<DstT, int32_t>(  \
      int32_t) const;                                                       \
  template Result<DataValue> DataValue::DataValueStaticCast<DstT, uint32_t>( \
      uint32_t) const;                                                      \
  template Result<DataValue> DataValue::DataValueStaticCast<DstT, int64_t>(  \
      int64_t) const;                                                       \
  template Result<DataValue> DataValue::DataValueStaticCast<DstT, uint64_t>( \
      uint64_t) const;                                                      \
  template Result<DataValue> DataValue::DataValueStaticCast<DstT, bfloat16>( \
      bfloat16) const;                                                      \
  template Result<DataValue>                                                \
  DataValue::DataValueStaticCast<DstT, float8_e4m3fn>(                      \
      float8_e4m3fn) const;                                                 \
  template Result<DataValue>                                                \
  DataValue::DataValueStaticCast<DstT, float8_e5m2>(float8_e5m2) const;     \
  template Result<DataValue> DataValue::DataValueStaticCast<DstT, float16>(  \
      float16) const;                                                       \
  template Result<DataValue> DataValue::DataValueStaticCast<DstT, float>(    \
      float) const;                                                         \
  template Result<DataValue> DataValue::DataValueStaticCast<DstT, double>(   \
      double) const;                                                        \
  template Result<DataValue> DataValue::DataValueStaticCast<DstT, complex64>( \
      complex64) const;                                                     \
  template Result<DataValue>                                                \
  DataValue::DataValueStaticCast<DstT, complex128>(complex128) const;       \
  template Result<DataValue> DataValue::DataValueStaticCast<DstT, pstring>(  \
      pstring) const;                                                       \
  template Result<DataValue>                                                \
  DataValue::DataValueStaticCast<DstT, adt::Undefined>(adt::Undefined) const;

AP_EXPLICIT_INSTANTIATE_DATA_VALUE_STATIC_CAST_ROW(bool)
AP_EXPLICIT_INSTANTIATE_DATA_VALUE_STATIC_CAST_ROW(int8_t)
AP_EXPLICIT_INSTANTIATE_DATA_VALUE_STATIC_CAST_ROW(uint8_t)
AP_EXPLICIT_INSTANTIATE_DATA_VALUE_STATIC_CAST_ROW(int16_t)
AP_EXPLICIT_INSTANTIATE_DATA_VALUE_STATIC_CAST_ROW(uint16_t)
AP_EXPLICIT_INSTANTIATE_DATA_VALUE_STATIC_CAST_ROW(int32_t)
AP_EXPLICIT_INSTANTIATE_DATA_VALUE_STATIC_CAST_ROW(uint32_t)
AP_EXPLICIT_INSTANTIATE_DATA_VALUE_STATIC_CAST_ROW(int64_t)
AP_EXPLICIT_INSTANTIATE_DATA_VALUE_STATIC_CAST_ROW(uint64_t)
AP_EXPLICIT_INSTANTIATE_DATA_VALUE_STATIC_CAST_ROW(bfloat16)
AP_EXPLICIT_INSTANTIATE_DATA_VALUE_STATIC_CAST_ROW(float8_e4m3fn)
AP_EXPLICIT_INSTANTIATE_DATA_VALUE_STATIC_CAST_ROW(float8_e5m2)
AP_EXPLICIT_INSTANTIATE_DATA_VALUE_STATIC_CAST_ROW(float16)
AP_EXPLICIT_INSTANTIATE_DATA_VALUE_STATIC_CAST_ROW(float)
AP_EXPLICIT_INSTANTIATE_DATA_VALUE_STATIC_CAST_ROW(double)
AP_EXPLICIT_INSTANTIATE_DATA_VALUE_STATIC_CAST_ROW(complex64)
AP_EXPLICIT_INSTANTIATE_DATA_VALUE_STATIC_CAST_ROW(complex128)
AP_EXPLICIT_INSTANTIATE_DATA_VALUE_STATIC_CAST_ROW(pstring)
// DstT == pstring/adt::Undefined rows are included too: even though
// DataValueStaticCast() early-returns a TypeError for those two DstT
// cases (never reaching `static_cast<DstT>(v)`), the compiler still
// needs to instantiate every branch of the template body, so omitting
// these rows would leave the corresponding symbols undefined.
AP_EXPLICIT_INSTANTIATE_DATA_VALUE_STATIC_CAST_ROW(adt::Undefined)

#undef AP_EXPLICIT_INSTANTIATE_DATA_VALUE_STATIC_CAST_ROW

}  // namespace ap::axpr
