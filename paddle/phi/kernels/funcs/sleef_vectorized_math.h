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

// MKL VML runtime detection is only supported on Linux/Unix systems
// because it requires dlfcn.h (dlopen/dlsym) which is not available on Windows.
#if defined(PADDLE_WITH_MKLML) && !defined(_WIN32)
#define PADDLE_MKL_VML_RUNTIME_DETECTION 1
#include <dlfcn.h>
#include <mkl.h>

// Type definitions for MKL VML sin/cos functions
// These functions may not be available in lightweight mklml (libmklml_intel.so)
// but are present in full Intel MKL.
using vmsSin_t = void (*)(MKL_INT, const float*, float*, MKL_INT64);
using vmdSin_t = void (*)(MKL_INT, const double*, double*, MKL_INT64);
using vmsCos_t = void (*)(MKL_INT, const float*, float*, MKL_INT64);
using vmdCos_t = void (*)(MKL_INT, const double*, double*, MKL_INT64);
using vmsExp_t = void (*)(MKL_INT, const float*, float*, MKL_INT64);
using vmdExp_t = void (*)(MKL_INT, const double*, double*, MKL_INT64);

// VML mode constant: VML_HA | VML_FTZDAZ_OFF | VML_ERRMODE_IGNORE
// = 0x2 | 0x00140000 | 0x100 = 0x00140102
static constexpr MKL_INT64 VML_MODE_HA_FTZDAZ_OFF_ERRIGNORE = 0x00140102LL;

// RTLD_NOLOAD is a GNU extension (available on Linux via _GNU_SOURCE).
// Provide a fallback definition in case it is not defined.
#ifndef RTLD_NOLOAD
#define RTLD_NOLOAD 4
#endif

namespace detail {

// One-time initialization: try to make MKL VML functions available in global
// symbol scope. This handles the case where MKL VML symbols (vmsSin, etc.)
// are not directly exported globally, e.g. when other libraries are loaded
// with RTLD_LOCAL.
//
// Strategies tried in order:
// 1. Check if symbols are already visible via RTLD_DEFAULT (normal case).
// 2. Try to load the full Intel MKL runtime (libmkl_rt.so) and promote it
//    to global scope so its symbols become visible.
//
// Thread safety: C++11 guarantees function-local statics are initialized
// exactly once, even under concurrent access.
inline void ensure_mkl_vml_probed() {
  static bool probed = false;
  if (probed) return;
  probed = true;

  // Strategy 1: Already in global scope - nothing to do
  if (dlsym(RTLD_DEFAULT, "vmsSin") != nullptr) return;

  // Strategy 2: Full Intel MKL runtime library
  // First check if already loaded (RTLD_NOLOAD), then try a fresh load.
  void* handle = dlopen("libmkl_rt.so", RTLD_LAZY | RTLD_NOLOAD);
  if (!handle) {
    handle = dlopen("libmkl_rt.so", RTLD_LAZY);
  }
  if (handle && dlsym(handle, "vmsSin") != nullptr) {
    // Re-open with RTLD_GLOBAL to make symbols visible via RTLD_DEFAULT
    dlopen("libmkl_rt.so", RTLD_LAZY | RTLD_GLOBAL);
    return;
  }
}

}  // namespace detail

// Runtime detection of MKL VML functions via dlsym.
// Calls ensure_mkl_vml_probed() once to try multiple strategies for making
// MKL VML symbols visible, then resolves each function via RTLD_DEFAULT.
// Returns nullptr if the function is not available.
inline vmsSin_t get_vmsSin() {
  static vmsSin_t func = nullptr;
  static bool checked = false;
  if (!checked) {
    checked = true;
    detail::ensure_mkl_vml_probed();
    func = reinterpret_cast<vmsSin_t>(dlsym(RTLD_DEFAULT, "vmsSin"));
  }
  return func;
}

inline vmdSin_t get_vmdSin() {
  static vmdSin_t func = nullptr;
  static bool checked = false;
  if (!checked) {
    checked = true;
    detail::ensure_mkl_vml_probed();
    func = reinterpret_cast<vmdSin_t>(dlsym(RTLD_DEFAULT, "vmdSin"));
  }
  return func;
}

inline vmsCos_t get_vmsCos() {
  static vmsCos_t func = nullptr;
  static bool checked = false;
  if (!checked) {
    checked = true;
    detail::ensure_mkl_vml_probed();
    func = reinterpret_cast<vmsCos_t>(dlsym(RTLD_DEFAULT, "vmsCos"));
  }
  return func;
}

inline vmdCos_t get_vmdCos() {
  static vmdCos_t func = nullptr;
  static bool checked = false;
  if (!checked) {
    checked = true;
    detail::ensure_mkl_vml_probed();
    func = reinterpret_cast<vmdCos_t>(dlsym(RTLD_DEFAULT, "vmdCos"));
  }
  return func;
}

inline vmsExp_t get_vmsExp() {
  static vmsExp_t func = nullptr;
  static bool checked = false;
  if (!checked) {
    checked = true;
    detail::ensure_mkl_vml_probed();
    func = reinterpret_cast<vmsExp_t>(dlsym(RTLD_DEFAULT, "vmsExp"));
  }
  return func;
}

inline vmdExp_t get_vmdExp() {
  static vmdExp_t func = nullptr;
  static bool checked = false;
  if (!checked) {
    checked = true;
    detail::ensure_mkl_vml_probed();
    func = reinterpret_cast<vmdExp_t>(dlsym(RTLD_DEFAULT, "vmdExp"));
  }
  return func;
}

// Check if MKL VML sin/cos functions are available at runtime
inline bool mkl_vml_sincos_available() {
  static bool available = false;
  static bool checked = false;
  if (!checked) {
    checked = true;
    available = (get_vmsSin() != nullptr && get_vmdSin() != nullptr &&
                 get_vmsCos() != nullptr && get_vmdCos() != nullptr);
  }
  return available;
}

// Check if MKL VML exp functions are available at runtime
inline bool mkl_vml_exp_available() {
  static bool available = false;
  static bool checked = false;
  if (!checked) {
    checked = true;
    available = (get_vmsExp() != nullptr && get_vmdExp() != nullptr);
  }
  return available;
}

#endif  // PADDLE_WITH_MKLML && !_WIN32

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
// Vectorized Sin/Cos functions - high precision implementation
// =============================================================================

#ifdef PADDLE_WITH_SLEEF

// -----------------------------------------------------------------------------
// AVX2 implementation (8 floats / 4 doubles at a time)
// -----------------------------------------------------------------------------
#ifdef PADDLE_SLEEF_HAS_AVX2

// Vectorized sin for float using AVX2
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

// Vectorized cos for float using AVX2
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

// Vectorized sin for double using AVX2
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

// Vectorized cos for double using AVX2
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

// Vectorized exp for float using AVX2
inline void vexp_avx2_f32(float* out, const float* in, int64_t n) {
  constexpr int64_t VEC_SIZE = 8;  // AVX2: 256-bit = 8 floats
  int64_t i = 0;

  for (; i + VEC_SIZE <= n; i += VEC_SIZE) {
    __m256 vec_in = _mm256_loadu_ps(in + i);
    __m256 vec_out = Sleef_expf8_u10(vec_in);
    _mm256_storeu_ps(out + i, vec_out);
  }

  for (; i < n; ++i) {
    out[i] = Sleef_expf1_u10(in[i]);
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

// Vectorized exp for float using AVX512 (16 floats at a time)
inline void vexp_avx512_f32(float* out, const float* in, int64_t n) {
  constexpr int64_t VEC_SIZE = 16;  // AVX512: 512-bit = 16 floats
  int64_t i = 0;

  for (; i + VEC_SIZE <= n; i += VEC_SIZE) {
    __m512 vec_in = _mm512_loadu_ps(in + i);
    __m512 vec_out = Sleef_expf16_u10(vec_in);
    _mm512_storeu_ps(out + i, vec_out);
  }

#ifdef PADDLE_SLEEF_HAS_AVX2
  for (; i + 8 <= n; i += 8) {
    __m256 vec_in = _mm256_loadu_ps(in + i);
    __m256 vec_out = Sleef_expf8_u10(vec_in);
    _mm256_storeu_ps(out + i, vec_out);
  }
#endif

  for (; i < n; ++i) {
    out[i] = Sleef_expf1_u10(in[i]);
  }
}

// Vectorized exp for double using AVX512 (8 doubles at a time)
inline void vexp_avx512_f64(double* out, const double* in, int64_t n) {
  constexpr int64_t VEC_SIZE = 8;  // AVX512: 512-bit = 8 doubles
  int64_t i = 0;

  for (; i + VEC_SIZE <= n; i += VEC_SIZE) {
    __m512d vec_in = _mm512_loadu_pd(in + i);
    __m512d vec_out = Sleef_expd8_u10(vec_in);
    _mm512_storeu_pd(out + i, vec_out);
  }

#ifdef PADDLE_SLEEF_HAS_AVX2
  for (; i + 4 <= n; i += 4) {
    __m256d vec_in = _mm256_loadu_pd(in + i);
    __m256d vec_out = Sleef_expd4_u10(vec_in);
    _mm256_storeu_pd(out + i, vec_out);
  }
#endif

  for (; i < n; ++i) {
    out[i] = Sleef_expd1_u10(in[i]);
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

// Scalar exp fallback
inline void vexp_scalar_f32(float* out, const float* in, int64_t n) {
  for (int64_t i = 0; i < n; ++i) {
    out[i] = Sleef_expf1_u10(in[i]);
  }
}

inline void vexp_scalar_f64(double* out, const double* in, int64_t n) {
  for (int64_t i = 0; i < n; ++i) {
    out[i] = Sleef_expd1_u10(in[i]);
  }
}

// No separate MKL VML wrapper functions needed.
// Runtime detection is done inline in the dispatch functions below.

// -----------------------------------------------------------------------------
// Unified dispatch functions
// -----------------------------------------------------------------------------

// Vectorized sin for float - dispatches to best available implementation
inline void vsin(float* out, const float* in, int64_t n) {
#ifdef PADDLE_MKL_VML_RUNTIME_DETECTION
  auto mkl_sin = get_vmsSin();
  if (mkl_sin) {
    mkl_sin(static_cast<MKL_INT>(n), in, out, VML_MODE_HA_FTZDAZ_OFF_ERRIGNORE);
    return;
  }
#endif
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
#ifdef PADDLE_MKL_VML_RUNTIME_DETECTION
  auto mkl_cos = get_vmsCos();
  if (mkl_cos) {
    mkl_cos(static_cast<MKL_INT>(n), in, out, VML_MODE_HA_FTZDAZ_OFF_ERRIGNORE);
    return;
  }
#endif
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
#ifdef PADDLE_MKL_VML_RUNTIME_DETECTION
  auto mkl_sin = get_vmdSin();
  if (mkl_sin) {
    mkl_sin(static_cast<MKL_INT>(n), in, out, VML_MODE_HA_FTZDAZ_OFF_ERRIGNORE);
    return;
  }
#endif
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
#ifdef PADDLE_MKL_VML_RUNTIME_DETECTION
  auto mkl_cos = get_vmdCos();
  if (mkl_cos) {
    mkl_cos(static_cast<MKL_INT>(n), in, out, VML_MODE_HA_FTZDAZ_OFF_ERRIGNORE);
    return;
  }
#endif
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

// Vectorized exp for float - dispatches to best available implementation
inline void vexp(float* out, const float* in, int64_t n) {
#ifdef PADDLE_MKL_VML_RUNTIME_DETECTION
  auto mkl_exp = get_vmsExp();
  if (mkl_exp) {
    mkl_exp(static_cast<MKL_INT>(n), in, out, VML_MODE_HA_FTZDAZ_OFF_ERRIGNORE);
    return;
  }
#endif
#ifdef PADDLE_SLEEF_HAS_AVX512
  vexp_avx512_f32(out, in, n);
#elif defined(PADDLE_SLEEF_HAS_AVX2)
  vexp_avx2_f32(out, in, n);
#else
  vexp_scalar_f32(out, in, n);
#endif
}

// Vectorized exp for double
inline void vexp(double* out, const double* in, int64_t n) {
#ifdef PADDLE_MKL_VML_RUNTIME_DETECTION
  auto mkl_exp = get_vmdExp();
  if (mkl_exp) {
    mkl_exp(static_cast<MKL_INT>(n), in, out, VML_MODE_HA_FTZDAZ_OFF_ERRIGNORE);
    return;
  }
#endif
#ifdef PADDLE_SLEEF_HAS_AVX512
  vexp_avx512_f64(out, in, n);
#else
  vexp_scalar_f64(out, in, n);
#endif
}

#else  // !PADDLE_WITH_SLEEF

// Fallback to standard library when Sleef is not available
#include <cmath>

inline void vsin(float* out, const float* in, int64_t n) {
#ifdef PADDLE_MKL_VML_RUNTIME_DETECTION
  auto mkl_sin = get_vmsSin();
  if (mkl_sin) {
    mkl_sin(static_cast<MKL_INT>(n), in, out, VML_MODE_HA_FTZDAZ_OFF_ERRIGNORE);
    return;
  }
#endif
  for (int64_t i = 0; i < n; ++i) {
    out[i] = std::sin(in[i]);
  }
}

inline void vcos(float* out, const float* in, int64_t n) {
#ifdef PADDLE_MKL_VML_RUNTIME_DETECTION
  auto mkl_cos = get_vmsCos();
  if (mkl_cos) {
    mkl_cos(static_cast<MKL_INT>(n), in, out, VML_MODE_HA_FTZDAZ_OFF_ERRIGNORE);
    return;
  }
#endif
  for (int64_t i = 0; i < n; ++i) {
    out[i] = std::cos(in[i]);
  }
}

inline void vsin(double* out, const double* in, int64_t n) {
#ifdef PADDLE_MKL_VML_RUNTIME_DETECTION
  auto mkl_sin = get_vmdSin();
  if (mkl_sin) {
    mkl_sin(static_cast<MKL_INT>(n), in, out, VML_MODE_HA_FTZDAZ_OFF_ERRIGNORE);
    return;
  }
#endif
  for (int64_t i = 0; i < n; ++i) {
    out[i] = std::sin(in[i]);
  }
}

inline void vcos(double* out, const double* in, int64_t n) {
#ifdef PADDLE_MKL_VML_RUNTIME_DETECTION
  auto mkl_cos = get_vmdCos();
  if (mkl_cos) {
    mkl_cos(static_cast<MKL_INT>(n), in, out, VML_MODE_HA_FTZDAZ_OFF_ERRIGNORE);
    return;
  }
#endif
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

inline void vexp(float* out, const float* in, int64_t n) {
#ifdef PADDLE_MKL_VML_RUNTIME_DETECTION
  auto mkl_exp = get_vmsExp();
  if (mkl_exp) {
    mkl_exp(static_cast<MKL_INT>(n), in, out, VML_MODE_HA_FTZDAZ_OFF_ERRIGNORE);
    return;
  }
#endif
  for (int64_t i = 0; i < n; ++i) {
    out[i] = std::exp(in[i]);
  }
}

inline void vexp(double* out, const double* in, int64_t n) {
#ifdef PADDLE_MKL_VML_RUNTIME_DETECTION
  auto mkl_exp = get_vmdExp();
  if (mkl_exp) {
    mkl_exp(static_cast<MKL_INT>(n), in, out, VML_MODE_HA_FTZDAZ_OFF_ERRIGNORE);
    return;
  }
#endif
  for (int64_t i = 0; i < n; ++i) {
    out[i] = std::exp(in[i]);
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
  // 1. MKL VML sin/cos functions are available at runtime (works for any size)
  // 2. SLEEF is available and element count is large enough for SIMD
#ifdef PADDLE_MKL_VML_RUNTIME_DETECTION
  if (mkl_vml_sincos_available()) {
    return true;  // MKL VML works for any size
  }
#endif
#ifdef PADDLE_WITH_SLEEF
  constexpr int64_t MIN_ELEMENTS_FOR_SIMD = 8;
  return numel >= MIN_ELEMENTS_FOR_SIMD;
#else
  return false;
#endif
}

// Check if vectorized path should be used for exp operations
inline bool should_use_vectorized_path_for_exp(const void* in_ptr,
                                               const void* out_ptr,
                                               int64_t numel) {
  // Use vectorized path when:
  // 1. MKL VML exp functions are available at runtime (works for any size)
  // 2. SLEEF is available and element count is large enough for SIMD
#ifdef PADDLE_MKL_VML_RUNTIME_DETECTION
  if (mkl_vml_exp_available()) {
    return true;  // MKL VML works for any size
  }
#endif
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
