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

#include "paddle/phi/common/data_type.h"

using complex64 = phi::dtype::complex<float>;
using complex128 = phi::dtype::complex<double>;

namespace phi {
__device__ __forceinline__ dtype::float16 inline_abs(dtype::float16 x) {
  return static_cast<dtype::float16>(abs(static_cast<float>(x)));
}

__device__ __forceinline__ dtype::bfloat16 inline_abs(dtype::bfloat16 x) {
  return static_cast<dtype::bfloat16>(abs(static_cast<float>(x)));
}

__device__ __forceinline__ float inline_abs(float x) { return abs(x); }

__device__ __forceinline__ double inline_abs(double x) { return abs(x); }

__device__ __forceinline__ complex64 inline_abs(complex64 x) { return abs(x); }

__device__ __forceinline__ complex128 inline_abs(complex128 x) { return abs(x); }

__device__ __forceinline__ dtype::float16 inline_pow(dtype::float16 base,
                                                     dtype::float16 exponent) {
  return static_cast<dtype::float16>(
      pow(static_cast<float>(base), static_cast<float>(exponent)));
}
__device__ __forceinline__ dtype::bfloat16 inline_pow(
    dtype::bfloat16 base, dtype::bfloat16 exponent) {
  return static_cast<dtype::bfloat16>(
      pow(static_cast<float>(base), static_cast<float>(exponent)));
}
__device__ __forceinline__ float inline_pow(float base, float exponent) {
  return pow(base, exponent);
}
__device__ __forceinline__ double inline_pow(double base, double exponent) {
  return pow(base, exponent);
}

__device__ __forceinline__ complex64 inline_pow(complex64 base, complex64 exponent) {
  return pow(base, exponent);
}
__device__ __forceinline__ complex128 inline_pow(complex128 base, complex128 exponent) {
  return pow(base, exponent);
}

}  // namespace phi
