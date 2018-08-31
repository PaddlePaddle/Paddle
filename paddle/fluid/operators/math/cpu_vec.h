/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.

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
#include <cmath>
#include <functional>
#include <string>
#include "paddle/fluid/platform/cpu_info.h"
#ifdef __AVX__
#include <immintrin.h>
#endif

#ifdef PADDLE_WITH_MKLML
#include "paddle/fluid/platform/dynload/mklml.h"
#endif

namespace paddle {
namespace operators {
namespace math {

#define SIGMOID_THRESHOLD_MIN -40.0
#define SIGMOID_THRESHOLD_MAX 13.0

#define AVX_FLOAT_BLOCK 8
#define AVX_DOUBLE_BLOCK 4
#define AVX2_FLOAT_BLOCK 8
#define AVX2_DOUBLE_BLOCK 4
#define AVX512_FLOAT_BLOCK 16
#define AVX512_DOUBLE_BLOCK 8

template <typename T>
inline void vec_exp(const int n, const T* x, T* y) {
  for (int i = 0; i < n; ++i) {
    y[i] = std::exp(x[i]);
  }
}

template <typename T>
inline void vec_scal(const int n, const T a, T* x) {
  for (int i = 0; i < n; ++i) {
    x[i] = a * x[i];
  }
}

#ifdef PADDLE_WITH_MKLML
template <>
inline void vec_exp<float>(const int n, const float* x, float* y) {
  platform::dynload::vsExp(n, x, y);
}

template <>
inline void vec_exp<double>(const int n, const double* x, double* y) {
  platform::dynload::vdExp(n, x, y);
}

template <>
inline void vec_scal<float>(const int n, const float a, float* x) {
  platform::dynload::cblas_sscal(n, a, x, 1);
}

template <>
inline void vec_scal<double>(const int n, const double a, double* x) {
  platform::dynload::cblas_dscal(n, a, x, 1);
}
#endif

// MKL scal only support inplace, choose this if src and dst are not equal
template <typename T, platform::jit::cpu_isa_t isa = platform::jit::isa_any>
inline void vec_scal(const int n, const T a, const T* x, T* y) {
  for (int i = 0; i < n; ++i) {
    y[i] = a * x[i];
  }
}

template <>
inline void vec_scal<float, platform::jit::avx>(const int n, const float a,
                                                const float* x, float* y) {
#ifdef __AVX__
  constexpr int block = AVX_FLOAT_BLOCK;
  if (n < block) {
    vec_scal<float, platform::jit::isa_any>(n, a, x, y);
    return;
  }
  const int rest = n % block;
  const int end = n - rest;
  int i = 0;
  __m256 scalar = _mm256_set1_ps(a);
  __m256 tmp;
#define MOVE_ONE_STEP               \
  tmp = _mm256_loadu_ps(x + i);     \
  tmp = _mm256_mul_ps(tmp, scalar); \
  _mm256_storeu_ps(y + i, tmp)
  for (i = 0; i < end; i += block) {
    MOVE_ONE_STEP;
  }
#undef MOVE_ONE_STEP
  if (rest == 0) {
    return;
  }
  // can not continue move step if src and dst are inplace
  for (i = n - rest; i < n; ++i) {
    y[i] = a * x[i];
  }
#else
  vec_scal<float, platform::jit::isa_any>(n, a, x, y);
#endif
}

template <>
inline void vec_scal<float, platform::jit::avx2>(const int n, const float a,
                                                 const float* x, float* y) {
  vec_scal<float, platform::jit::avx>(n, a, x, y);
}

template <>
inline void vec_scal<float, platform::jit::avx512_common>(const int n,
                                                          const float a,
                                                          const float* x,
                                                          float* y) {
  // TODO(TJ): enable me
  vec_scal<float, platform::jit::avx2>(n, a, x, y);
}

template <typename T, platform::jit::cpu_isa_t isa = platform::jit::isa_any>
inline void vec_bias_sub(const int n, const T a, const T* x, T* y) {
  for (int i = 0; i < n; ++i) {
    y[i] = a - x[i];
  }
}

template <>
inline void vec_bias_sub<float, platform::jit::avx>(const int n, const float a,
                                                    const float* x, float* y) {
#ifdef __AVX__
  constexpr int block = AVX_FLOAT_BLOCK;
  if (n < block) {
    vec_bias_sub<float, platform::jit::isa_any>(n, a, x, y);
    return;
  }
  const int rest = n % block;
  const int end = n - rest;
  int i = 0;
  __m256 bias = _mm256_set1_ps(a);
  __m256 tmp;
#define MOVE_ONE_STEP             \
  tmp = _mm256_loadu_ps(x + i);   \
  tmp = _mm256_sub_ps(bias, tmp); \
  _mm256_storeu_ps(y + i, tmp)
  for (i = 0; i < end; i += block) {
    MOVE_ONE_STEP;
  }
#undef MOVE_ONE_STEP
  if (rest == 0) {
    return;
  }
  // can not continue move step if src and dst are inplace
  for (i = n - rest; i < n; ++i) {
    y[i] = a - x[i];
  }
#else
  vec_bias_sub<float, platform::jit::isa_any>(n, a, x, y);
#endif
}

template <>
inline void vec_bias_sub<float, platform::jit::avx2>(const int n, const float a,
                                                     const float* x, float* y) {
  vec_bias_sub<float, platform::jit::avx>(n, a, x, y);
}

template <>
inline void vec_bias_sub<float, platform::jit::avx512_common>(const int n,
                                                              const float a,
                                                              const float* x,
                                                              float* y) {
  // TODO(TJ): enable me
  vec_bias_sub<float, platform::jit::avx2>(n, a, x, y);
}

// out = x*y + (1-x)*z
template <typename T, platform::jit::cpu_isa_t isa = platform::jit::isa_any>
inline void vec_cross(const int n, const T* x, const T* y, const T* z, T* out) {
  for (int i = 0; i < n; ++i) {
    out[i] = x[i] * y[i] + (static_cast<T>(1) - x[i]) * z[i];
  }
}

template <>
inline void vec_cross<float, platform::jit::avx>(const int n, const float* x,
                                                 const float* y, const float* z,
                                                 float* out) {
#ifdef __AVX__
  constexpr int block = AVX_FLOAT_BLOCK;
  if (n < block) {
    vec_cross<float, platform::jit::isa_any>(n, x, y, z, out);
    return;
  }
  const int rest = n % block;
  const int end = n - rest;
  int i = 0;
  __m256 bias = _mm256_set1_ps(1.f);
  __m256 tmpx, tmpy, tmpz;
  for (i = 0; i < end; i += block) {
    tmpx = _mm256_loadu_ps(x + i);
    tmpy = _mm256_loadu_ps(y + i);
    tmpz = _mm256_loadu_ps(z + i);
    tmpy = _mm256_mul_ps(tmpx, tmpy);
    tmpx = _mm256_sub_ps(bias, tmpx);
    tmpz = _mm256_mul_ps(tmpx, tmpz);
    tmpz = _mm256_add_ps(tmpy, tmpz);
    _mm256_storeu_ps(out + i, tmpz);
  }
  if (rest == 0) {
    return;
  }
  // can not continue move step if src and dst are inplace
  for (i = n - rest; i < n; ++i) {
    out[i] = x[i] * y[i] + (1.f - x[i]) * z[i];
  }
#else
  vec_cross<float, platform::jit::isa_any>(n, x, y, z, out);
#endif
}

template <>
inline void vec_cross<float, platform::jit::avx2>(const int n, const float* x,
                                                  const float* y,
                                                  const float* z, float* out) {
  vec_cross<float, platform::jit::avx>(n, x, y, z, out);
}

template <>
inline void vec_cross<float, platform::jit::avx512_common>(
    const int n, const float* x, const float* y, const float* z, float* out) {
  // TODO(TJ): enable me
  vec_cross<float, platform::jit::avx>(n, x, y, z, out);
}

template <typename T, platform::jit::cpu_isa_t isa = platform::jit::isa_any>
inline void vec_add_bias(const int n, const T a, const T* x, T* y) {
  for (int i = 0; i < n; ++i) {
    y[i] = x[i] + a;
  }
}

template <>
inline void vec_add_bias<float, platform::jit::avx>(const int n, const float a,
                                                    const float* x, float* y) {
#ifdef __AVX__
  constexpr int block = AVX_FLOAT_BLOCK;
  if (n < block) {
    vec_add_bias<float, platform::jit::isa_any>(n, a, x, y);
    return;
  }
  const int rest = n % block;
  const int end = n - rest;
  int i = 0;
  __m256 bias = _mm256_set1_ps(a);
  __m256 tmp;
#define MOVE_ONE_STEP             \
  tmp = _mm256_loadu_ps(x + i);   \
  tmp = _mm256_add_ps(tmp, bias); \
  _mm256_storeu_ps(y + i, tmp)
  for (i = 0; i < end; i += block) {
    MOVE_ONE_STEP;
  }
#undef MOVE_ONE_STEP
  if (rest == 0) {
    return;
  }
  // can not continue move step if src and dst are inplace
  for (i = n - rest; i < n; ++i) {
    y[i] = x[i] + a;
  }
#else
  vec_add_bias<float, platform::jit::isa_any>(n, a, x, y);
#endif
}

template <>
inline void vec_add_bias<float, platform::jit::avx2>(const int n, const float a,
                                                     const float* x, float* y) {
  vec_add_bias<float, platform::jit::avx>(n, a, x, y);
}

template <>
inline void vec_add_bias<float, platform::jit::avx512_common>(const int n,
                                                              const float a,
                                                              const float* x,
                                                              float* y) {
  // TODO(TJ): enable me
  vec_add_bias<float, platform::jit::avx2>(n, a, x, y);
}

template <typename T, platform::jit::cpu_isa_t isa = platform::jit::isa_any>
inline void vec_identity(const int n, const T* x, T* y) {
  // do nothing
  return;
}

template <typename T, platform::jit::cpu_isa_t isa = platform::jit::isa_any>
inline void vec_sigmoid(const int n, const T* x, T* y) {
  const T min = SIGMOID_THRESHOLD_MIN;
  const T max = SIGMOID_THRESHOLD_MAX;
  for (int i = 0; i < n; ++i) {
    y[i] = (x[i] < min) ? min : ((x[i] > max) ? max : x[i]);
    y[i] = static_cast<T>(0) - y[i];
  }
  vec_exp<T>(n, y, y);
  for (int i = 0; i < n; ++i) {
    y[i] = static_cast<T>(1) / (static_cast<T>(1) + y[i]);
  }
}

template <>
inline void vec_sigmoid<float, platform::jit::avx>(const int n, const float* x,
                                                   float* y) {
#ifdef __AVX__
  constexpr int block = AVX_FLOAT_BLOCK;
  if (n < block) {
    vec_sigmoid<float, platform::jit::isa_any>(n, x, y);
    return;
  }
  const int rest = n % block;
  const int end = n - rest;
  int i = 0;
  __m256 max = _mm256_set1_ps(SIGMOID_THRESHOLD_MAX);
  __m256 min = _mm256_set1_ps(SIGMOID_THRESHOLD_MIN);
  __m256 zeros = _mm256_setzero_ps();
  __m256 tmp;
#define MOVE_ONE_STEP              \
  tmp = _mm256_loadu_ps(x + i);    \
  tmp = _mm256_max_ps(tmp, min);   \
  tmp = _mm256_min_ps(tmp, max);   \
  tmp = _mm256_sub_ps(zeros, tmp); \
  _mm256_storeu_ps(y + i, tmp)
  for (i = 0; i < end; i += block) {
    MOVE_ONE_STEP;
  }
#undef MOVE_ONE_STEP
  if (rest != 0) {
    // can not continue move step since the src and dst address could be equal
    const float xmin = SIGMOID_THRESHOLD_MIN;
    const float xmax = SIGMOID_THRESHOLD_MAX;
    for (i = n - rest; i < n; ++i) {
      y[i] = 0.f - ((x[i] < xmin) ? xmin : ((x[i] > xmax) ? xmax : x[i]));
    }
  }

  vec_exp<float>(n, y, y);

  __m256 ones = _mm256_set1_ps(1.0f);
#define MOVE_ONE_STEP             \
  tmp = _mm256_loadu_ps(y + i);   \
  tmp = _mm256_add_ps(ones, tmp); \
  tmp = _mm256_div_ps(ones, tmp); \
  _mm256_storeu_ps(y + i, tmp)
  for (i = 0; i < end; i += block) {
    MOVE_ONE_STEP;
  }
#undef MOVE_ONE_STEP
  if (rest == 0) {
    return;
  }
  // can not continue move step
  for (i = n - rest; i < n; ++i) {
    y[i] = 1.f / (1.f + y[i]);
  }
#else
  vec_sigmoid<float, platform::jit::isa_any>(n, x, y);
#endif
}

template <>
inline void vec_sigmoid<float, platform::jit::avx2>(const int n, const float* x,
                                                    float* y) {
  vec_sigmoid<float, platform::jit::avx>(n, x, y);
}

template <>
inline void vec_sigmoid<float, platform::jit::avx512_common>(const int n,
                                                             const float* x,
                                                             float* y) {
  // TODO(TJ): enable me
  vec_sigmoid<float, platform::jit::avx2>(n, x, y);
}

template <typename T, platform::jit::cpu_isa_t isa = platform::jit::isa_any>
inline void vec_tanh(const int n, const T* x, T* y) {
  vec_scal<T, isa>(n, static_cast<T>(2), x, y);
  vec_sigmoid<T, isa>(n, y, y);
  vec_scal<T>(n, static_cast<T>(2), y);
  vec_add_bias<T, isa>(n, static_cast<T>(-1), y, y);
}

// TODO(TJ): make relu clip
template <typename T, platform::jit::cpu_isa_t isa = platform::jit::isa_any>
inline void vec_relu(const int n, const T* x, T* y) {
  for (int i = 0; i < n; ++i) {
    y[i] = x[i] > 0 ? x[i] : 0;
  }
}

template <>
inline void vec_relu<float, platform::jit::avx>(const int n, const float* x,
                                                float* y) {
#ifdef __AVX__
  constexpr int block = AVX_FLOAT_BLOCK;
  if (n < block * 4) {
    vec_relu<float, platform::jit::isa_any>(n, x, y);
    return;
  }

  const int rest = n % block;
  const int end = n - rest;
  int i = 0;
  __m256 zeros = _mm256_setzero_ps();
  __m256 tmp;
#define MOVE_ONE_STEP              \
  tmp = _mm256_loadu_ps(x + i);    \
  tmp = _mm256_max_ps(tmp, zeros); \
  _mm256_storeu_ps(y + i, tmp)
  for (i = 0; i < end; i += block) {
    MOVE_ONE_STEP;
  }
  if (rest == 0) {
    return;
  }
  i = n - block;
  MOVE_ONE_STEP;
#undef MOVE_ONE_STEP

#else
  vec_relu<float, platform::jit::isa_any>(n, x, y);
#endif
}

template <>
inline void vec_relu<float, platform::jit::avx2>(const int n, const float* x,
                                                 float* y) {
  vec_relu<float, platform::jit::avx>(n, x, y);
}

template <>
inline void vec_relu<float, platform::jit::avx512_common>(const int n,
                                                          const float* x,
                                                          float* y) {
  // TODO(TJ): enable me
  vec_relu<float, platform::jit::avx2>(n, x, y);
}

// TODO(TJ): optimize double of sigmoid, tanh and relu if necessary

template <typename T, platform::jit::cpu_isa_t isa = platform::jit::isa_any>
class VecActivations {
 public:
  std::function<void(const int, const T*, T*)> operator()(
      const std::string& type) {
    if (type == "sigmoid") {
      return vec_sigmoid<T, isa>;
    } else if (type == "relu") {
      return vec_relu<T, isa>;
    } else if (type == "tanh") {
      return vec_tanh<T, isa>;
    } else if (type == "identity" || type == "") {
      return vec_identity<T, isa>;
    }
    LOG(FATAL) << "Not support type: " << type;
  }
};

}  // namespace math
}  // namespace operators
}  // namespace paddle
