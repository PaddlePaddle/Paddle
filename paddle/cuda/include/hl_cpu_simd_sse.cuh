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

#ifndef HL_CPU_SIMD_SSE_CUH_
#define HL_CPU_SIMD_SSE_CUH_

#include <mmintrin.h>
#include <xmmintrin.h>
#include <emmintrin.h>

#define VECTOR_SIMD true
#define VECTOR_SIZE 16
#define VECTOR_SET  hl_vec_set

#ifndef PADDLE_TYPE_DOUBLE

typedef __m128  vecType;

/* number of float in vector */
#define VECTOR_LEN  4

template <class Agg>
inline real hl_agg_op(Agg agg, vecType mm) {
  __m128 lo = _mm_unpacklo_ps(mm, mm);
  __m128 hi = _mm_unpackhi_ps(mm, mm);
  __m128 tmp1 = agg.vecOp(lo, hi);
  __m128 tmp2 = _mm_movehl_ps(tmp1, tmp1);
  __m128 ret = agg.vecOp(tmp1, tmp2);

  return _mm_cvtss_f32(ret);
}

inline __m128 hl_vec_set(const real f) {
  return _mm_set_ps1(f);
}

inline __m128 hl_vec_classification_error(const __m128 a,
                                          const __m128 b,
                                          const __m128 p,
                                          const __m128 r) {
  __m128 tmp1 = _mm_cmpgt_ps(a, p);
  __m128 tmp2 = _mm_cmpgt_ps(b, p);
  __m128 tmp3 = _mm_xor_ps(tmp1, tmp2);
  return _mm_and_ps(tmp3, r);
}

#else

typedef __m128d vecType;

/* number of double in vector */
#define VECTOR_LEN  2

template <class Agg>
inline real hl_agg_op(Agg agg, vecType mm) {
  __m128d lo = _mm_unpacklo_pd(mm, mm);
  __m128d hi = _mm_unpackhi_pd(mm, mm);
  __m128d ret = agg.vecOp(lo, hi);

  return _mm_cvtsd_f64(ret);
}

inline __m128d hl_vec_set(const real d) {
#if defined(__APPLE__) || defined(__OSX__)
  return _mm_set1_pd(d);
#else
  return _mm_set_pd1(d);
#endif
}

inline __m128d hl_vec_classification_error(const __m128d a,
                                           const __m128d b,
                                           const __m128d p,
                                           const __m128d r) {
  __m128d tmp1 = _mm_cmpgt_pd(a, p);
  __m128d tmp2 = _mm_cmpgt_pd(b, p);
  __m128d tmp3 = _mm_xor_pd(tmp1, tmp2);
  return _mm_and_pd(tmp3, r);
}

#endif

#endif  // HL_CPU_SIMD_SSE_CUH_
