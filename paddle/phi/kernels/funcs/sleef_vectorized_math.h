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

#ifdef PADDLE_WITH_SLEEF
#include <sleef.h>

#if defined(__AVX512F__) || defined(__AVX2__) || defined(__AVX__)
#include <immintrin.h>
#endif

#if defined(__AVX2__) || defined(__AVX__)
#define PADDLE_SLEEF_HAS_AVX2 1
#endif

#if defined(__AVX512F__)
#define PADDLE_SLEEF_HAS_AVX512 1
#endif

#endif  // PADDLE_WITH_SLEEF

#include <cstdint>
#include <cstring>
#include <type_traits>

namespace phi {
namespace funcs {
namespace sleef_vec {

// =============================================================================
// Scalar Sleef functions for pow
// =============================================================================

#ifdef PADDLE_WITH_SLEEF

template <typename T>
inline typename std::enable_if<std::is_same<T, float>::value, T>::type
pow_sleef_scalar(const T a, const T b) {
  return Sleef_powf1_u10(a, b);
}

template <typename T>
inline typename std::enable_if<std::is_same<T, double>::value, T>::type
pow_sleef_scalar(const T a, const T b) {
  return Sleef_powd1_u10(a, b);
}

#endif  // PADDLE_WITH_SLEEF

// =============================================================================
// Vectorized Sin/Cos functions - matches PyTorch's precision
// =============================================================================

#ifdef PADDLE_WITH_SLEEF

// -----------------------------------------------------------------------------
// AVX2 implementation (8 floats / 4 doubles at a time)
// -----------------------------------------------------------------------------
#ifdef PADDLE_SLEEF_HAS_AVX2

// Vectorized sin for float using AVX2 (matches PyTorch's Sleef_sinf8_u35)
inline void vsin_avx2_f32(float* out, const float* in, int64_t n) {
  constexpr int64_t VEC_SIZE = 8;  // AVX2: 256-bit = 8 floats
  int64_t i = 0;

  // Process 8 floats at a time
  for (; i + VEC_SIZE <= n; i += VEC_SIZE) {
    __m256 vec_in = _mm256_loadu_ps(in + i);
    __m256 vec_out = Sleef_sinf8_u35(vec_in);
    _mm256_storeu_ps(out + i, vec_out);
  }

  // Handle remaining elements with scalar version
  for (; i < n; ++i) {
    out[i] = Sleef_sinf1_u35(in[i]);
  }
}

// Vectorized cos for float using AVX2 (matches PyTorch's Sleef_cosf8_u35)
inline void vcos_avx2_f32(float* out, const float* in, int64_t n) {
  constexpr int64_t VEC_SIZE = 8;
  int64_t i = 0;

  for (; i + VEC_SIZE <= n; i += VEC_SIZE) {
    __m256 vec_in = _mm256_loadu_ps(in + i);
    __m256 vec_out = Sleef_cosf8_u35(vec_in);
    _mm256_storeu_ps(out + i, vec_out);
  }

  for (; i < n; ++i) {
    out[i] = Sleef_cosf1_u35(in[i]);
  }
}

// Vectorized sin for double using AVX2 (matches PyTorch's Sleef_sind4_u10)
inline void vsin_avx2_f64(double* out, const double* in, int64_t n) {
  constexpr int64_t VEC_SIZE = 4;  // AVX2: 256-bit = 4 doubles
  int64_t i = 0;

  for (; i + VEC_SIZE <= n; i += VEC_SIZE) {
    __m256d vec_in = _mm256_loadu_pd(in + i);
    __m256d vec_out = Sleef_sind4_u10(vec_in);
    _mm256_storeu_pd(out + i, vec_out);
  }

  for (; i < n; ++i) {
    out[i] = Sleef_sind1_u10(in[i]);
  }
}

// Vectorized cos for double using AVX2 (matches PyTorch's Sleef_cosd4_u10)
inline void vcos_avx2_f64(double* out, const double* in, int64_t n) {
  constexpr int64_t VEC_SIZE = 4;
  int64_t i = 0;

  for (; i + VEC_SIZE <= n; i += VEC_SIZE) {
    __m256d vec_in = _mm256_loadu_pd(in + i);
    __m256d vec_out = Sleef_cosd4_u10(vec_in);
    _mm256_storeu_pd(out + i, vec_out);
  }

  for (; i < n; ++i) {
    out[i] = Sleef_cosd1_u10(in[i]);
  }
}

// Vectorized pow for float using AVX2 (no native AVX2 pow, use scalar Sleef)
inline void vpow_avx2_f32(float* out,
                          const float* x,
                          const float* y,
                          int64_t n) {
  for (int64_t i = 0; i < n; ++i) {
    out[i] = Sleef_powf1_u10(x[i], y[i]);
  }
}

// Vectorized pow for double using AVX2
inline void vpow_avx2_f64(double* out,
                          const double* x,
                          const double* y,
                          int64_t n) {
  for (int64_t i = 0; i < n; ++i) {
    out[i] = Sleef_powd1_u10(x[i], y[i]);
  }
}

#endif  // PADDLE_SLEEF_HAS_AVX2

// -----------------------------------------------------------------------------
// AVX512 implementation (16 floats / 8 doubles at a time)
// -----------------------------------------------------------------------------
#ifdef PADDLE_SLEEF_HAS_AVX512

inline void vsin_avx512_f32(float* out, const float* in, int64_t n) {
  constexpr int64_t VEC_SIZE = 16;  // AVX512: 512-bit = 16 floats
  int64_t i = 0;

  for (; i + VEC_SIZE <= n; i += VEC_SIZE) {
    __m512 vec_in = _mm512_loadu_ps(in + i);
    __m512 vec_out = Sleef_sinf16_u35(vec_in);
    _mm512_storeu_ps(out + i, vec_out);
  }

  // Fallback to AVX2 for remaining >= 8 elements
#ifdef PADDLE_SLEEF_HAS_AVX2
  for (; i + 8 <= n; i += 8) {
    __m256 vec_in = _mm256_loadu_ps(in + i);
    __m256 vec_out = Sleef_sinf8_u35(vec_in);
    _mm256_storeu_ps(out + i, vec_out);
  }
#endif

  for (; i < n; ++i) {
    out[i] = Sleef_sinf1_u35(in[i]);
  }
}

inline void vcos_avx512_f32(float* out, const float* in, int64_t n) {
  constexpr int64_t VEC_SIZE = 16;
  int64_t i = 0;

  for (; i + VEC_SIZE <= n; i += VEC_SIZE) {
    __m512 vec_in = _mm512_loadu_ps(in + i);
    __m512 vec_out = Sleef_cosf16_u35(vec_in);
    _mm512_storeu_ps(out + i, vec_out);
  }

#ifdef PADDLE_SLEEF_HAS_AVX2
  for (; i + 8 <= n; i += 8) {
    __m256 vec_in = _mm256_loadu_ps(in + i);
    __m256 vec_out = Sleef_cosf8_u35(vec_in);
    _mm256_storeu_ps(out + i, vec_out);
  }
#endif

  for (; i < n; ++i) {
    out[i] = Sleef_cosf1_u35(in[i]);
  }
}

inline void vsin_avx512_f64(double* out, const double* in, int64_t n) {
  constexpr int64_t VEC_SIZE = 8;  // AVX512: 512-bit = 8 doubles
  int64_t i = 0;

  for (; i + VEC_SIZE <= n; i += VEC_SIZE) {
    __m512d vec_in = _mm512_loadu_pd(in + i);
    __m512d vec_out = Sleef_sind8_u10(vec_in);
    _mm512_storeu_pd(out + i, vec_out);
  }

#ifdef PADDLE_SLEEF_HAS_AVX2
  for (; i + 4 <= n; i += 4) {
    __m256d vec_in = _mm256_loadu_pd(in + i);
    __m256d vec_out = Sleef_sind4_u10(vec_in);
    _mm256_storeu_pd(out + i, vec_out);
  }
#endif

  for (; i < n; ++i) {
    out[i] = Sleef_sind1_u10(in[i]);
  }
}

inline void vcos_avx512_f64(double* out, const double* in, int64_t n) {
  constexpr int64_t VEC_SIZE = 8;
  int64_t i = 0;

  for (; i + VEC_SIZE <= n; i += VEC_SIZE) {
    __m512d vec_in = _mm512_loadu_pd(in + i);
    __m512d vec_out = Sleef_cosd8_u10(vec_in);
    _mm512_storeu_pd(out + i, vec_out);
  }

#ifdef PADDLE_SLEEF_HAS_AVX2
  for (; i + 4 <= n; i += 4) {
    __m256d vec_in = _mm256_loadu_pd(in + i);
    __m256d vec_out = Sleef_cosd4_u10(vec_in);
    _mm256_storeu_pd(out + i, vec_out);
  }
#endif

  for (; i < n; ++i) {
    out[i] = Sleef_cosd1_u10(in[i]);
  }
}

// Vectorized pow for float using AVX512 (16 floats at a time)
inline void vpow_avx512_f32(float* out,
                            const float* x,
                            const float* y,
                            int64_t n) {
  constexpr int64_t VEC_SIZE = 16;
  int64_t i = 0;

  for (; i + VEC_SIZE <= n; i += VEC_SIZE) {
    __m512 vec_x = _mm512_loadu_ps(x + i);
    __m512 vec_y = _mm512_loadu_ps(y + i);
    __m512 vec_out = Sleef_powf16_u10(vec_x, vec_y);
    _mm512_storeu_ps(out + i, vec_out);
  }

  for (; i < n; ++i) {
    out[i] = Sleef_powf1_u10(x[i], y[i]);
  }
}

// Vectorized pow for double using AVX512 (8 doubles at a time)
inline void vpow_avx512_f64(double* out,
                            const double* x,
                            const double* y,
                            int64_t n) {
  constexpr int64_t VEC_SIZE = 8;
  int64_t i = 0;

  for (; i + VEC_SIZE <= n; i += VEC_SIZE) {
    __m512d vec_x = _mm512_loadu_pd(x + i);
    __m512d vec_y = _mm512_loadu_pd(y + i);
    __m512d vec_out = Sleef_powd8_u10(vec_x, vec_y);
    _mm512_storeu_pd(out + i, vec_out);
  }

  for (; i < n; ++i) {
    out[i] = Sleef_powd1_u10(x[i], y[i]);
  }
}

#endif  // PADDLE_SLEEF_HAS_AVX512

// -----------------------------------------------------------------------------
// Scalar fallback (when SIMD is not available)
// -----------------------------------------------------------------------------
inline void vsin_scalar_f32(float* out, const float* in, int64_t n) {
  for (int64_t i = 0; i < n; ++i) {
    out[i] = Sleef_sinf1_u35(in[i]);
  }
}

inline void vcos_scalar_f32(float* out, const float* in, int64_t n) {
  for (int64_t i = 0; i < n; ++i) {
    out[i] = Sleef_cosf1_u35(in[i]);
  }
}

inline void vsin_scalar_f64(double* out, const double* in, int64_t n) {
  for (int64_t i = 0; i < n; ++i) {
    out[i] = Sleef_sind1_u10(in[i]);
  }
}

inline void vcos_scalar_f64(double* out, const double* in, int64_t n) {
  for (int64_t i = 0; i < n; ++i) {
    out[i] = Sleef_cosd1_u10(in[i]);
  }
}

// Scalar pow fallback
inline void vpow_scalar_f32(float* out,
                            const float* x,
                            const float* y,
                            int64_t n) {
  for (int64_t i = 0; i < n; ++i) {
    out[i] = Sleef_powf1_u10(x[i], y[i]);
  }
}

inline void vpow_scalar_f64(double* out,
                            const double* x,
                            const double* y,
                            int64_t n) {
  for (int64_t i = 0; i < n; ++i) {
    out[i] = Sleef_powd1_u10(x[i], y[i]);
  }
}

// -----------------------------------------------------------------------------
// Unified dispatch functions
// -----------------------------------------------------------------------------

// Vectorized sin for float - dispatches to best available SIMD
inline void vsin(float* out, const float* in, int64_t n) {
#ifdef PADDLE_SLEEF_HAS_AVX512
  vsin_avx512_f32(out, in, n);
#elif defined(PADDLE_SLEEF_HAS_AVX2)
  vsin_avx2_f32(out, in, n);
#else
  vsin_scalar_f32(out, in, n);
#endif
}

// Vectorized cos for float
inline void vcos(float* out, const float* in, int64_t n) {
#ifdef PADDLE_SLEEF_HAS_AVX512
  vcos_avx512_f32(out, in, n);
#elif defined(PADDLE_SLEEF_HAS_AVX2)
  vcos_avx2_f32(out, in, n);
#else
  vcos_scalar_f32(out, in, n);
#endif
}

// Vectorized sin for double
inline void vsin(double* out, const double* in, int64_t n) {
#ifdef PADDLE_SLEEF_HAS_AVX512
  vsin_avx512_f64(out, in, n);
#elif defined(PADDLE_SLEEF_HAS_AVX2)
  vsin_avx2_f64(out, in, n);
#else
  vsin_scalar_f64(out, in, n);
#endif
}

// Vectorized cos for double
inline void vcos(double* out, const double* in, int64_t n) {
#ifdef PADDLE_SLEEF_HAS_AVX512
  vcos_avx512_f64(out, in, n);
#elif defined(PADDLE_SLEEF_HAS_AVX2)
  vcos_avx2_f64(out, in, n);
#else
  vcos_scalar_f64(out, in, n);
#endif
}

// Vectorized pow for float - dispatches to best available SIMD
inline void vpow(float* out, const float* x, const float* y, int64_t n) {
#ifdef PADDLE_SLEEF_HAS_AVX512
  vpow_avx512_f32(out, x, y, n);
#elif defined(PADDLE_SLEEF_HAS_AVX2)
  vpow_avx2_f32(out, x, y, n);
#else
  vpow_scalar_f32(out, x, y, n);
#endif
}

// Vectorized pow for double
inline void vpow(double* out, const double* x, const double* y, int64_t n) {
#ifdef PADDLE_SLEEF_HAS_AVX512
  vpow_avx512_f64(out, x, y, n);
#elif defined(PADDLE_SLEEF_HAS_AVX2)
  vpow_avx2_f64(out, x, y, n);
#else
  vpow_scalar_f64(out, x, y, n);
#endif
}

#else  // !PADDLE_WITH_SLEEF

// Fallback to standard library when Sleef is not available
#include <cmath>

inline void vsin(float* out, const float* in, int64_t n) {
  for (int64_t i = 0; i < n; ++i) {
    out[i] = std::sin(in[i]);
  }
}

inline void vcos(float* out, const float* in, int64_t n) {
  for (int64_t i = 0; i < n; ++i) {
    out[i] = std::cos(in[i]);
  }
}

inline void vsin(double* out, const double* in, int64_t n) {
  for (int64_t i = 0; i < n; ++i) {
    out[i] = std::sin(in[i]);
  }
}

inline void vcos(double* out, const double* in, int64_t n) {
  for (int64_t i = 0; i < n; ++i) {
    out[i] = std::cos(in[i]);
  }
}

inline void vpow(float* out, const float* x, const float* y, int64_t n) {
  for (int64_t i = 0; i < n; ++i) {
    out[i] = std::pow(x[i], y[i]);
  }
}

inline void vpow(double* out, const double* x, const double* y, int64_t n) {
  for (int64_t i = 0; i < n; ++i) {
    out[i] = std::pow(x[i], y[i]);
  }
}

#endif  // PADDLE_WITH_SLEEF

// -----------------------------------------------------------------------------
// Check if vectorized path should be used
// -----------------------------------------------------------------------------
inline bool should_use_vectorized_path(const void* in_ptr,
                                       const void* out_ptr,
                                       int64_t numel) {
  // Use vectorized path when:
  // 1. SLEEF is available
  // 2. Element count is large enough to benefit from SIMD
  // 3. Memory is reasonably aligned (not strictly required for unaligned loads)
#ifdef PADDLE_WITH_SLEEF
  constexpr int64_t MIN_ELEMENTS_FOR_SIMD = 8;
  return numel >= MIN_ELEMENTS_FOR_SIMD;
#else
  return false;
#endif
}

}  // namespace sleef_vec
}  // namespace funcs
}  // namespace phi
