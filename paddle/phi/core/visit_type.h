/* Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.

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

#include "paddle/phi/api/ext/exception.h"
#include "paddle/phi/common/data_type.h"

namespace phi {

///////// Basic Marco ///////////

#define PD_PRIVATE_CASE_TYPE_USING_HINT(NAME, enum_type, type, HINT, ...) \
  case enum_type: {                                                       \
    using HINT = type;                                                    \
    __VA_ARGS__();                                                        \
    break;                                                                \
  }

#define PD_PRIVATE_CASE_TYPE(NAME, enum_type, type, ...) \
  PD_PRIVATE_CASE_TYPE_USING_HINT(NAME, enum_type, type, data_t, __VA_ARGS__)

///////// Floating Dispatch Marco ///////////

#define PD_VISIT_FLOATING_TYPES(TYPE, NAME, ...)                          \
  [&] {                                                                   \
    const auto& __dtype__ = TYPE;                                         \
    switch (__dtype__) {                                                  \
      PD_PRIVATE_CASE_TYPE(                                               \
          NAME, ::paddle::DataType::FLOAT32, float, __VA_ARGS__)          \
      PD_PRIVATE_CASE_TYPE(                                               \
          NAME, ::paddle::DataType::FLOAT64, double, __VA_ARGS__)         \
      default:                                                            \
        PD_THROW("function " #NAME " is not implemented for data type `", \
                 __dtype__,                                               \
                 "`");                                                    \
    }                                                                     \
  }()

#define PD_VISIT_FLOATING_AND_HALF_TYPES(TYPE, NAME, ...)                  \
  [&] {                                                                    \
    const auto& __dtype__ = TYPE;                                          \
    switch (__dtype__) {                                                   \
      PD_PRIVATE_CASE_TYPE(                                                \
          NAME, ::paddle::DataType::FLOAT32, float, __VA_ARGS__)           \
      PD_PRIVATE_CASE_TYPE(                                                \
          NAME, ::paddle::DataType::FLOAT64, double, __VA_ARGS__)          \
      PD_PRIVATE_CASE_TYPE(                                                \
          NAME, ::paddle::DataType::FLOAT16, paddle::float16, __VA_ARGS__) \
      default:                                                             \
        PD_THROW("function " #NAME " is not implemented for data type `",  \
                 __dtype__,                                                \
                 "`");                                                     \
    }                                                                      \
  }()

///////// Integral Dispatch Marco ///////////

#define PD_VISIT_INTEGRAL_TYPES(TYPE, NAME, ...)                              \
  [&] {                                                                       \
    const auto& __dtype__ = TYPE;                                             \
    switch (__dtype__) {                                                      \
      PD_PRIVATE_CASE_TYPE(NAME, ::paddle::DataType::INT32, int, __VA_ARGS__) \
      PD_PRIVATE_CASE_TYPE(                                                   \
          NAME, ::paddle::DataType::INT64, int64_t, __VA_ARGS__)              \
      PD_PRIVATE_CASE_TYPE(                                                   \
          NAME, ::paddle::DataType::INT8, int8_t, __VA_ARGS__)                \
      PD_PRIVATE_CASE_TYPE(                                                   \
          NAME, ::paddle::DataType::UINT8, uint8_t, __VA_ARGS__)              \
      PD_PRIVATE_CASE_TYPE(                                                   \
          NAME, ::paddle::DataType::INT16, int16_t, __VA_ARGS__)              \
      default:                                                                \
        PD_THROW("function " #NAME " is not implemented for data type `",     \
                 __dtype__,                                                   \
                 "`");                                                        \
    }                                                                         \
  }()

#define PD_VISIT_BASE_INTEGRAL_TYPES(TYPE, NAME, ...)                         \
  [&] {                                                                       \
    const auto& __dtype__ = TYPE;                                             \
    switch (__dtype__) {                                                      \
      PD_PRIVATE_CASE_TYPE(NAME, ::paddle::DataType::INT32, int, __VA_ARGS__) \
      PD_PRIVATE_CASE_TYPE(                                                   \
          NAME, ::paddle::DataType::INT64, int64_t, __VA_ARGS__)              \
      default:                                                                \
        PD_THROW("function " #NAME " is not implemented for data type `",     \
                 __dtype__,                                                   \
                 "`");                                                        \
    }                                                                         \
  }()

///////// Complex Dispatch Marco ///////////

#define PD_VISIT_COMPLEX_TYPES(TYPE, NAME, ...)                           \
  [&] {                                                                   \
    const auto& __dtype__ = TYPE;                                         \
    switch (__dtype__) {                                                  \
      PD_PRIVATE_CASE_TYPE(NAME,                                          \
                           ::paddle::DataType::COMPLEX64,                 \
                           ::paddle::complex64,                           \
                           __VA_ARGS__)                                   \
      PD_PRIVATE_CASE_TYPE(NAME,                                          \
                           ::paddle::DataType::COMPLEX128,                \
                           ::paddle::complex128,                          \
                           __VA_ARGS__)                                   \
      default:                                                            \
        PD_THROW("function " #NAME " is not implemented for data type `", \
                 __dtype__,                                               \
                 "`");                                                    \
    }                                                                     \
  }()

///////// Floating and Integral Dispatch Marco ///////////

#define PD_VISIT_FLOATING_AND_INTEGRAL_TYPES(TYPE, NAME, ...)                 \
  [&] {                                                                       \
    const auto& __dtype__ = TYPE;                                             \
    switch (__dtype__) {                                                      \
      PD_PRIVATE_CASE_TYPE(                                                   \
          NAME, ::paddle::DataType::FLOAT32, float, __VA_ARGS__)              \
      PD_PRIVATE_CASE_TYPE(                                                   \
          NAME, ::paddle::DataType::FLOAT64, double, __VA_ARGS__)             \
      PD_PRIVATE_CASE_TYPE(NAME, ::paddle::DataType::INT32, int, __VA_ARGS__) \
      PD_PRIVATE_CASE_TYPE(                                                   \
          NAME, ::paddle::DataType::INT64, int64_t, __VA_ARGS__)              \
      PD_PRIVATE_CASE_TYPE(                                                   \
          NAME, ::paddle::DataType::INT8, int8_t, __VA_ARGS__)                \
      PD_PRIVATE_CASE_TYPE(                                                   \
          NAME, ::paddle::DataType::UINT8, uint8_t, __VA_ARGS__)              \
      PD_PRIVATE_CASE_TYPE(                                                   \
          NAME, ::paddle::DataType::INT16, int16_t, __VA_ARGS__)              \
      default:                                                                \
        PD_THROW("function " #NAME " is not implemented for data type `",     \
                 __dtype__,                                                   \
                 "`");                                                        \
    }                                                                         \
  }()

///////// Floating and Complex Dispatch Marco ///////////

#define PD_VISIT_FLOATING_AND_COMPLEX_TYPES(TYPE, NAME, ...)              \
  [&] {                                                                   \
    const auto& __dtype__ = TYPE;                                         \
    switch (__dtype__) {                                                  \
      PD_PRIVATE_CASE_TYPE(                                               \
          NAME, ::paddle::DataType::FLOAT32, float, __VA_ARGS__)          \
      PD_PRIVATE_CASE_TYPE(                                               \
          NAME, ::paddle::DataType::FLOAT64, double, __VA_ARGS__)         \
      PD_PRIVATE_CASE_TYPE(NAME,                                          \
                           ::paddle::DataType::COMPLEX64,                 \
                           ::paddle::complex64,                           \
                           __VA_ARGS__)                                   \
      PD_PRIVATE_CASE_TYPE(NAME,                                          \
                           ::paddle::DataType::COMPLEX128,                \
                           ::paddle::complex128,                          \
                           __VA_ARGS__)                                   \
      default:                                                            \
        PD_THROW("function " #NAME " is not implemented for data type `", \
                 __dtype__,                                               \
                 "`");                                                    \
    }                                                                     \
  }()

///////// Floating and Complex and other type Dispatch Marco ///////////

#define PD_VISIT_FLOATING_AND_COMPLEX_AND_1_TYPE(                          \
    SPECIFIED_TYPE, TYPE, NAME, ...)                                       \
  [&] {                                                                    \
    const auto& __dtype__ = TYPE;                                          \
    switch (__dtype__) {                                                   \
      PD_PRIVATE_CASE_TYPE(NAME,                                           \
                           SPECIFIED_TYPE,                                 \
                           ::phi::DataTypeToCppType<SPECIFIED_TYPE>::type, \
                           __VA_ARGS__)                                    \
      PD_PRIVATE_CASE_TYPE(                                                \
          NAME, ::paddle::DataType::FLOAT32, float, __VA_ARGS__)           \
      PD_PRIVATE_CASE_TYPE(                                                \
          NAME, ::paddle::DataType::FLOAT64, double, __VA_ARGS__)          \
      PD_PRIVATE_CASE_TYPE(NAME,                                           \
                           ::paddle::DataType::COMPLEX64,                  \
                           ::paddle::complex64,                            \
                           __VA_ARGS__)                                    \
      PD_PRIVATE_CASE_TYPE(NAME,                                           \
                           ::paddle::DataType::COMPLEX128,                 \
                           ::paddle::complex128,                           \
                           __VA_ARGS__)                                    \
      default:                                                             \
        PD_THROW("function " #NAME " is not implemented for data type `",  \
                 __dtype__,                                                \
                 "`");                                                     \
    }                                                                      \
  }()

///////// Floating and Complex and 2 other type Dispatch Marco ///////////

#define PD_VISIT_FLOATING_AND_COMPLEX_AND_2_TYPES(                          \
    SPECIFIED_TYPE1, SPECIFIED_TYPE2, TYPE, NAME, ...)                      \
  [&] {                                                                     \
    const auto& __dtype__ = TYPE;                                           \
    switch (__dtype__) {                                                    \
      PD_PRIVATE_CASE_TYPE(NAME,                                            \
                           SPECIFIED_TYPE1,                                 \
                           ::phi::DataTypeToCppType<SPECIFIED_TYPE1>::type, \
                           __VA_ARGS__)                                     \
      PD_PRIVATE_CASE_TYPE(NAME,                                            \
                           SPECIFIED_TYPE2,                                 \
                           ::phi::DataTypeToCppType<SPECIFIED_TYPE2>::type, \
                           __VA_ARGS__)                                     \
      PD_PRIVATE_CASE_TYPE(                                                 \
          NAME, ::paddle::DataType::FLOAT32, float, __VA_ARGS__)            \
      PD_PRIVATE_CASE_TYPE(                                                 \
          NAME, ::paddle::DataType::FLOAT64, double, __VA_ARGS__)           \
      PD_PRIVATE_CASE_TYPE(NAME,                                            \
                           ::paddle::DataType::COMPLEX64,                   \
                           ::paddle::complex64,                             \
                           __VA_ARGS__)                                     \
      PD_PRIVATE_CASE_TYPE(NAME,                                            \
                           ::paddle::DataType::COMPLEX128,                  \
                           ::paddle::complex128,                            \
                           __VA_ARGS__)                                     \
      default:                                                              \
        PD_THROW("function " #NAME " is not implemented for data type `",   \
                 __dtype__,                                                 \
                 "`");                                                      \
    }                                                                       \
  }()

///////// Floating, Integral and Complex Dispatch Marco ///////////

#define PD_VISIT_FLOATING_AND_INTEGRAL_AND_COMPLEX_TYPES(TYPE, NAME, ...)     \
  [&] {                                                                       \
    const auto& __dtype__ = TYPE;                                             \
    switch (__dtype__) {                                                      \
      PD_PRIVATE_CASE_TYPE(                                                   \
          NAME, ::paddle::DataType::FLOAT32, float, __VA_ARGS__)              \
      PD_PRIVATE_CASE_TYPE(                                                   \
          NAME, ::paddle::DataType::FLOAT64, double, __VA_ARGS__)             \
      PD_PRIVATE_CASE_TYPE(NAME, ::paddle::DataType::INT32, int, __VA_ARGS__) \
      PD_PRIVATE_CASE_TYPE(                                                   \
          NAME, ::paddle::DataType::INT64, int64_t, __VA_ARGS__)              \
      PD_PRIVATE_CASE_TYPE(                                                   \
          NAME, ::paddle::DataType::INT8, int8_t, __VA_ARGS__)                \
      PD_PRIVATE_CASE_TYPE(                                                   \
          NAME, ::paddle::DataType::UINT8, uint8_t, __VA_ARGS__)              \
      PD_PRIVATE_CASE_TYPE(                                                   \
          NAME, ::paddle::DataType::INT16, int16_t, __VA_ARGS__)              \
      PD_PRIVATE_CASE_TYPE(NAME,                                              \
                           ::paddle::DataType::COMPLEX64,                     \
                           ::paddle::complex64,                               \
                           __VA_ARGS__)                                       \
      PD_PRIVATE_CASE_TYPE(NAME,                                              \
                           ::paddle::DataType::COMPLEX128,                    \
                           ::paddle::complex128,                              \
                           __VA_ARGS__)                                       \
      default:                                                                \
        PD_THROW("function " #NAME " is not implemented for data type `",     \
                 __dtype__,                                                   \
                 "`");                                                        \
    }                                                                         \
  }()

#define PD_VISIT_ALL_TYPES(TYPE, NAME, ...)                                    \
  [&] {                                                                        \
    const auto& __dtype__ = TYPE;                                              \
    switch (__dtype__) {                                                       \
      PD_PRIVATE_CASE_TYPE(NAME, ::phi::DataType::BOOL, bool, __VA_ARGS__)     \
      PD_PRIVATE_CASE_TYPE(NAME, ::phi::DataType::INT8, int8_t, __VA_ARGS__)   \
      PD_PRIVATE_CASE_TYPE(NAME, ::phi::DataType::UINT8, uint8_t, __VA_ARGS__) \
      PD_PRIVATE_CASE_TYPE(NAME, ::phi::DataType::INT16, int16_t, __VA_ARGS__) \
      PD_PRIVATE_CASE_TYPE(NAME, ::phi::DataType::INT32, int32_t, __VA_ARGS__) \
      PD_PRIVATE_CASE_TYPE(NAME, ::phi::DataType::INT64, int64_t, __VA_ARGS__) \
      PD_PRIVATE_CASE_TYPE(                                                    \
          NAME, ::phi::DataType::BFLOAT16, phi::bfloat16, __VA_ARGS__)         \
      PD_PRIVATE_CASE_TYPE(                                                    \
          NAME, ::phi::DataType::FLOAT16, phi::float16, __VA_ARGS__)           \
      PD_PRIVATE_CASE_TYPE(NAME, ::phi::DataType::FLOAT32, float, __VA_ARGS__) \
      PD_PRIVATE_CASE_TYPE(                                                    \
          NAME, ::phi::DataType::FLOAT64, double, __VA_ARGS__)                 \
      PD_PRIVATE_CASE_TYPE(                                                    \
          NAME, ::phi::DataType::COMPLEX64, phi::complex64, __VA_ARGS__)       \
      PD_PRIVATE_CASE_TYPE(                                                    \
          NAME, ::phi::DataType::COMPLEX128, phi::complex128, __VA_ARGS__)     \
      default:                                                                 \
        PADDLE_THROW(phi::errors::InvalidArgument(                             \
            "Invalid enum data type `%d`.", static_cast<int>(__dtype__)));     \
    }                                                                          \
  }()

#define PD_VISIT_BOOL_AND_FLOATING_AND_COMPLEX_AND_3_TYPES(                   \
    SPECIFIED_TYPE1, SPECIFIED_TYPE2, SPECIFIED_TYPE3, TYPE, NAME, ...)       \
  [&] {                                                                       \
    const auto& __dtype__ = TYPE;                                             \
    switch (__dtype__) {                                                      \
      PD_PRIVATE_CASE_TYPE(NAME, ::paddle::DataType::BOOL, bool, __VA_ARGS__) \
      PD_PRIVATE_CASE_TYPE(                                                   \
          NAME, ::paddle::DataType::FLOAT32, float, __VA_ARGS__)              \
      PD_PRIVATE_CASE_TYPE(                                                   \
          NAME, ::paddle::DataType::FLOAT64, double, __VA_ARGS__)             \
      PD_PRIVATE_CASE_TYPE(NAME,                                              \
                           ::paddle::DataType::COMPLEX64,                     \
                           ::paddle::complex64,                               \
                           __VA_ARGS__)                                       \
      PD_PRIVATE_CASE_TYPE(NAME,                                              \
                           ::paddle::DataType::COMPLEX128,                    \
                           ::paddle::complex128,                              \
                           __VA_ARGS__)                                       \
      PD_PRIVATE_CASE_TYPE(NAME,                                              \
                           SPECIFIED_TYPE1,                                   \
                           ::phi::DataTypeToCppType<SPECIFIED_TYPE1>::type,   \
                           __VA_ARGS__)                                       \
      PD_PRIVATE_CASE_TYPE(NAME,                                              \
                           SPECIFIED_TYPE2,                                   \
                           ::phi::DataTypeToCppType<SPECIFIED_TYPE2>::type,   \
                           __VA_ARGS__)                                       \
      PD_PRIVATE_CASE_TYPE(NAME,                                              \
                           SPECIFIED_TYPE3,                                   \
                           ::phi::DataTypeToCppType<SPECIFIED_TYPE3>::type,   \
                           __VA_ARGS__)                                       \
      default:                                                                \
        PD_THROW("function " #NAME " is not implemented for data type `",     \
                 __dtype__,                                                   \
                 "`");                                                        \
    }                                                                         \
  }()

#define PD_VISIT_XPU_REDUCE_TYPES(TYPE, NAME, ...)                             \
  [&] {                                                                        \
    const auto& __dtype__ = TYPE;                                              \
    switch (__dtype__) {                                                       \
      PD_PRIVATE_CASE_TYPE(NAME, ::phi::DataType::INT8, int8_t, __VA_ARGS__)   \
      PD_PRIVATE_CASE_TYPE(NAME, ::phi::DataType::INT32, int32_t, __VA_ARGS__) \
      PD_PRIVATE_CASE_TYPE(NAME, ::phi::DataType::INT64, int64_t, __VA_ARGS__) \
      PD_PRIVATE_CASE_TYPE(                                                    \
          NAME, ::phi::DataType::FLOAT16, phi::float16, __VA_ARGS__)           \
      PD_PRIVATE_CASE_TYPE(NAME, ::phi::DataType::FLOAT32, float, __VA_ARGS__) \
      default:                                                                 \
        PADDLE_THROW(phi::errors::InvalidArgument(                             \
            "Invalid enum data type `%d`.", static_cast<int>(__dtype__)));     \
    }                                                                          \
  }()

}  // namespace phi
