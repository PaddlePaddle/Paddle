// Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
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

#include <c10/core/ScalarType.h>
#include <c10/util/BFloat16.h>
#include <c10/util/Exception.h>
#include <c10/util/Float8_e4m3fn.h>
#include <c10/util/Float8_e5m2.h>
#include <c10/util/Half.h>

namespace at {

// For FP16 or BFloat16 inputs, ops should perform internal math in FP32.
template <typename scalar_t>
struct OpMathType {
  using type = scalar_t;
};
template <>
struct OpMathType<at::Half> {
  using type = float;
};
template <>
struct OpMathType<at::BFloat16> {
  using type = float;
};
template <>
struct OpMathType<at::Float8_e5m2> {
  using type = float;
};
template <>
struct OpMathType<at::Float8_e4m3fn> {
  using type = float;
};
template <>
struct OpMathType<c10::complex<Half>> {
  using type = c10::complex<float>;
};

template <typename T>
using opmath_type = typename OpMathType<T>::type;

inline c10::ScalarType toOpMathType(const c10::ScalarType type) {
  switch (type) {
#define DEFINE_CASE(scalar_t, TypeNum) \
  case ScalarType::TypeNum:            \
    return CppTypeToScalarType<at::opmath_type<scalar_t>>::value;

    AT_FORALL_SCALAR_TYPES_WITH_COMPLEX(DEFINE_CASE)
#undef DEFINE_CASE

    default:
      TORCH_INTERNAL_ASSERT(false, "Unrecognized ScalarType: ", type);
  }
}

}  // namespace at
