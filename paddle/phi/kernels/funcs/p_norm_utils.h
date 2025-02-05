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

namespace phi {
__device__ __forceinline__ dtype::float16 inline_abs(dtype::float16 x) {
  return static_cast<dtype::float16>(abs(static_cast<float>(x)));
}

__device__ __forceinline__ dtype::bfloat16 inline_abs(dtype::bfloat16 x) {
  return static_cast<dtype::bfloat16>(abs(static_cast<float>(x)));
}

__device__ __forceinline__ float inline_abs(float x) { return abs(x); }

__device__ __forceinline__ double inline_abs(double x) { return abs(x); }

template <typename T>
__device__ __forceinline__ int sgn(T val) {
  return (T(0) < val) - (val < T(0));
}

__device__ __forceinline__ int inline_sign(dtype::float16 x) {
  return sgn<dtype::float16>(x);
}

__device__ __forceinline__ int inline_sign(dtype::bfloat16 x) {
  return sgn<dtype::bfloat16>(x);
}

__device__ __forceinline__ int inline_sign(float x) { return sgn<float>(x); }

__device__ __forceinline__ int inline_sign(double x) { return sgn<double>(x); }

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

#ifndef _WIN32
// To avoid large .so size in Windows cuda11.8
__device__ __forceinline__ dtype::float16 inline_fabs(dtype::float16 x) {
  return static_cast<dtype::float16>(fabs(static_cast<float>(x)));
}
__device__ __forceinline__ dtype::bfloat16 inline_fabs(dtype::bfloat16 x) {
  return static_cast<dtype::bfloat16>(fabs(static_cast<float>(x)));
}
__device__ __forceinline__ float inline_fabs(float x) { return fabs(x); }
__device__ __forceinline__ double inline_fabs(double x) { return fabs(x); }

__device__ __forceinline__ dtype::float16 inline_square(dtype::float16 x) {
  return static_cast<dtype::float16>(static_cast<float>(x) *
                                     static_cast<float>(x));
}
__device__ __forceinline__ dtype::bfloat16 inline_square(dtype::bfloat16 x) {
  return static_cast<dtype::bfloat16>(static_cast<float>(x) *
                                      static_cast<float>(x));
}
__device__ __forceinline__ float inline_square(float x) { return x * x; }
__device__ __forceinline__ double inline_square(double x) { return x * x; }

__device__ __forceinline__ dtype::float16 inline_fabs_cubic(dtype::float16 x) {
  return static_cast<dtype::float16>(fabs(
      static_cast<float>(x) * static_cast<float>(x) * static_cast<float>(x)));
}
__device__ __forceinline__ dtype::bfloat16 inline_fabs_cubic(
    dtype::bfloat16 x) {
  return static_cast<dtype::bfloat16>(fabs(
      static_cast<float>(x) * static_cast<float>(x) * static_cast<float>(x)));
}
__device__ __forceinline__ float inline_fabs_cubic(float x) {
  return fabs(x * x * x);
}
__device__ __forceinline__ double inline_fabs_cubic(double x) {
  return fabs(x * x * x);
}
#endif
}  // namespace phi
