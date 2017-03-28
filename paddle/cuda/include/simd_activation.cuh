/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserve.
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

#include "hl_base.h"
#include <math.h>
#include <immintrin.h>

#ifdef __AVX__
#include "simd_math.cuh"
#endif

namespace hppl {
#ifndef __AVX__
/// forward activation
inline real relu(const real a) { return a > 0.0f ? a : 0.0f; }

inline real sigmoid(const real a) {
  const real min = SIGMOID_THRESHOLD_MIN;
  const real max = SIGMOID_THRESHOLD_MAX;
  real tmp = (a < min) ? min : ((a > max) ? max : a);
  return 1.0 / (1.0 + exp(-tmp));
}

inline real tanh(const real a) {
  real tmp = -2.0 * a;
  tmp = (tmp > EXP_MAX_INPUT) ? EXP_MAX_INPUT : tmp;
  return (2.0 / (1.0 + exp(tmp))) - 1.0;
}

inline real linear(const real a) { return a; }

/// backward activation
inline real relu(const real a, const real b) { return a * (b > 0.0f ? 1.0f : 0.0f); }
inline real sigmoid(const real a, const real b) { return a * b * (1 - b); }
inline real tanh(const real a, const real b) { return a * (1.0f - b * b); }
inline real linear(const real a, const real b) { return a; }    

namespace cpu {
static Active<real>::forward  forward [] = { sigmoid, relu, tanh, linear };
static Active<real>::backward backward[] = { sigmoid, relu, tanh, linear };
}

#else

inline __m256 exp(__m256 a) { return exp256_ps(a); }
inline __m256 log(__m256 a) { return log256_ps(a); }
inline __m256 sin(__m256 a) { return sin256_ps(a); }
inline __m256 cos(__m256 a) { return cos256_ps(a); }

inline __m256 relu(const __m256 a) {
  __m256 tmp = _mm256_set1_ps(0.0f);
  return _mm256_max_ps(a, tmp);
}

inline __m256 sigmoid(const __m256 a) {
  __m256 max = _mm256_set1_ps(SIGMOID_THRESHOLD_MAX);
  __m256 min = _mm256_set1_ps(SIGMOID_THRESHOLD_MIN);
  __m256 tmp = _mm256_max_ps(a, min);
  tmp = _mm256_min_ps(tmp, max);
  tmp = _mm256_sub_ps(_mm256_set1_ps(0.0f), tmp);
  tmp = exp(tmp);
  tmp = _mm256_add_ps(_mm256_set1_ps(1.0f), tmp);
  tmp = _mm256_div_ps(_mm256_set1_ps(1.0f), tmp);
  return tmp;
}

inline __m256 tanh(const __m256 a) {
  __m256 max = _mm256_set1_ps(EXP_MAX_INPUT);
  __m256 tmp = _mm256_mul_ps(_mm256_set1_ps(-2.0f), a);
  tmp = _mm256_min_ps(tmp, max);
  tmp = exp(tmp);
  return _mm256_sub_ps(_mm256_div_ps(_mm256_set1_ps(2.0f),
                                     _mm256_add_ps(_mm256_set1_ps(1.0f), tmp)),
                       _mm256_set1_ps(1.0f));
}

inline __m256 linear(const __m256 a) { return a; }

inline __m256 relu(const __m256 a, const __m256 b) {
  return _mm256_mul_ps(
      a,
      _mm256_and_ps(_mm256_cmp_ps(b, _mm256_set1_ps(0.0f), _CMP_GT_OS),
                    _mm256_set1_ps(1.0f)));
}

inline __m256 sigmoid(const __m256 a, const __m256 b) {
  return _mm256_mul_ps(_mm256_mul_ps(a, b),
                       _mm256_sub_ps(_mm256_set1_ps(1.0f), b));
}

inline __m256 tanh(const __m256 a, const __m256 b) {
  return _mm256_mul_ps(
      a, _mm256_sub_ps(_mm256_set1_ps(1.0f), _mm256_mul_ps(b, b)));
}

inline __m256 linear(const __m256 a, const __m256 b) { return a; }

namespace avx {
static Active<__m256>::forward  forward [] = { sigmoid, relu, tanh, linear };
static Active<__m256>::backward backward[] = { sigmoid, relu, tanh, linear };
}
#endif  // __AVX__

#ifdef __NVCC__
__device__ static real relu(const real a) {
  return a > 0.0f ? a : 0.0f;
}

__device__ static real sigmoid(const real a) {
  const real min = SIGMOID_THRESHOLD_MIN;
  const real max = SIGMOID_THRESHOLD_MAX;
  real tmp = (a < min) ? min : ((a > max) ? max : a);
#ifndef PADDLE_TYPE_DOUBLE
  return __fdividef(1.0f, 1.0f + __expf(-tmp));
#else
  return 1.0 / (1.0 + exp(-tmp));
#endif
}

__device__ static real tanh(const real a) {
#ifndef PADDLE_TYPE_DOUBLE
  return __fdividef(2.0f, (1.0f + __expf(-2.0f*a))) - 1.0f;
#else
  return (2.0 / (1.0 + exp(-2.0*a))) - 1.0;
#endif
}

__device__ static real linear(const real a) {
  return a;
}

__device__ static real relu(const real a, const real b) {
  return a * (b > 0.0f ? 1.0f : 0.0f);
}

__device__ static real sigmoid(const real a, const real b) {
  return a * b * (1 - b);
}

__device__ static real tanh(const real a, const real b) {
  return a * (1.0f - b * b);
}

__device__ static real linear(const real a, const real b) {
  return a;
}

namespace gpu {		
  static __device__ Active<real>::forward forward[] = { sigmoid, relu, tanh, linear };	
  static __device__ Active<real>::backward backward[] = { sigmoid, relu, tanh, linear };
}
#endif  // __NVCC__
}   // namespace hppl
