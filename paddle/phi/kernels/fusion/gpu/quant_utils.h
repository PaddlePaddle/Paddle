// Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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
#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <iostream>
#include <limits>

#include "paddle/phi/api/all.h"
#include "paddle/phi/common/float8_e4m3fn.h"
#include "paddle/phi/common/float8_e5m2.h"
#include "paddle/phi/kernels/funcs/math_cuda_utils.h"

#define DISPATCH_BOOL(condition, ConstName, ...) \
  {                                              \
    if (condition) {                             \
      constexpr bool ConstName = true;           \
      { __VA_ARGS__ }                            \
    } else {                                     \
      constexpr bool ConstName = false;          \
      { __VA_ARGS__ }                            \
    }                                            \
  }

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
#define BF16_MAX(a, b) __hmax(a, b)
#define BF16_ABS(x) __habs(x)
#else
#define BF16_MAX(a, b) \
  __float2bfloat16(fmaxf(__bfloat162float(a), __bfloat162float(b)))
#define BF16_ABS(x) __float2bfloat16(fabsf(__bfloat162float(x)))
#endif

// Perform swizzle transformation on 2D coordinates with relative offset to
// avoid bank conflicts
__device__ __forceinline__ int swizzled_2d_idx(const int outer_dim,
                                               const int inner_rank,
                                               const int inner_dim) {
  return outer_dim * inner_rank + outer_dim ^ inner_dim;
}
// ------------------------------ Numerical Part (from
// kitchen)--------------------------- Type trait for extreme values of fp8
// types. Used in the calculation of scale factors as a constexpr lookup from
// e4m3 or e5m2 to the max finite value.
template <typename T>
struct F8LimitsTrait;

template <>
struct F8LimitsTrait<__nv_fp8_e4m3> {
  static constexpr float max = 448.0f;
};
template <>
struct F8LimitsTrait<phi::float8_e4m3fn> {
  static constexpr float max = 448.0f;
};

template <>
struct F8LimitsTrait<__nv_fp8_e5m2> {
  static constexpr float max = 57344.0f;
};
template <>
struct F8LimitsTrait<phi::float8_e5m2> {
  static constexpr float max = 57344.0f;
};

// Type trait to resolve the max finite value
// represented by a input type to quantization.
// Or to represent max representable power of 2
// finite value.
template <typename T, bool ForcePow2>
struct HighPrecisionFloatScaleLimitsTrait;

template <>
struct HighPrecisionFloatScaleLimitsTrait<float, false> {
  static constexpr float max = std::numeric_limits<float>::max();
};

template <>
struct HighPrecisionFloatScaleLimitsTrait<float, true> {
  // Hex float format of 1.0 * 2 ^ 127
  static constexpr float max = 0x1.0p127;
};

template <>
struct HighPrecisionFloatScaleLimitsTrait<nv_bfloat16, false> {
  // Hex float format of 1.(7 bits of 1) * 2 ^ 127
  static constexpr float max = 0x1.FEp127;
};

template <>
struct HighPrecisionFloatScaleLimitsTrait<nv_bfloat16, true> {
  // Hex float format of 1.0 * 2 ^ 127
  static constexpr float max = 0x1.0p127;
};

template <>
struct HighPrecisionFloatScaleLimitsTrait<half, false> {
  // Hex float format of 1.(10 bits of 1) * 2 ^ 15
  static constexpr float max = 0x1.FFCp15;
};

template <>
struct HighPrecisionFloatScaleLimitsTrait<half, true> {
  // Hex float format of 1.0 * 2 ^ 15
  static constexpr float max = 0x1.0p15;
};
// ----------------------------- Scale Part ---------------------------
// Calculate the quantization scale for an individual data element
// given the amax(abs(tile)) value for a given quantization tile.
//
//
// Arguments:
// IType: data type of the tensor being quantized (float or bf16)
// OType: quantized data type (e4m3 or e5m2)
// pow_2_scaling: Whether to force the scale to be a power of 2.
// amax: The evaluation of amax(abs(tile)) for the quantization tile.
// eps: An epsilon used as a floor for amax.
template <typename IType, typename OType, bool Power2Scaling = false>
__device__ __forceinline__ float ComputeScale(const float amax,
                                              const float eps) {
  constexpr float fp8_max = F8LimitsTrait<OType>::max;

  // Clamping amax to avoid division by small numbers
  float amax_mod = fmaxf(amax, eps);

  // Handle overflow cases for non-clamped amax (eps is 0 or very small)
  if (amax_mod == 0.f) {
    // If amax is 0, return 1
    return 1.f;
  }
  // Compute scale factor
  float scale = fp8_max / amax_mod;

  if (isinf(scale)) {
    // If scale is infinity, return max value of IType
    return HighPrecisionFloatScaleLimitsTrait<IType, Power2Scaling>::max;
  }
  if (scale == 0.0) {
    return scale;
  }
  if constexpr (Power2Scaling) {
    uint32_t scale_bits = *reinterpret_cast<uint32_t *>(&scale);
    // Scale must be positive, shift it
    uint8_t exp = scale_bits >> 23;

    // inf scales already early returned, as did nan scales.
    // The cases to consider here are normals, zero, and subnormals.
    // zero is not possible with current math as
    // 448.0 / float_max == 1.31655e-36, which is the smallest
    // possible scale given current dtypes. It is still in the normal
    // fp32 range with an exponent of -120, so subnormals are also
    // not possible.
    int32_t normal_biased_exp = static_cast<int32_t>(exp) - 127;
    __builtin_assume(exp != 0);
    // Normal numbers case.

    scale = ldexpf(1.0f, normal_biased_exp);
  }
  return scale;
}
// -------------------------------------- From Kitchen
// ----------------------------------

inline int64_t size_to_dim(size_t k, std::vector<int64_t> dims) {
  PD_CHECK(k >= 0 && k <= dims.size());
  int64_t r = 1;
  for (size_t i = 0; i < k; ++i) {
    r *= dims[i];
  }
  return r;
}

__device__ __forceinline__ float warpReduceMax(float val) {
  for (int offset = 16; offset > 0; offset /= 2)
    val = fmaxf(val, __shfl_down_sync(0xFFFFFFFF, val, offset));
  return val;
}
