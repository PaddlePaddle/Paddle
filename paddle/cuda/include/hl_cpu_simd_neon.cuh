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

#ifndef HL_CPU_SIMD_NEON_CUH_
#define HL_CPU_SIMD_NEON_CUH_

#include <arm_neon.h>

#define VECTOR_SIMD true
#define VECTOR_SIZE 16
#define VECTOR_SET  hl_vec_set

#ifndef PADDLE_TYPE_DOUBLE

typedef float32x4_t vecType;

/* number of float in vector */
#define VECTOR_LEN  4

template <class Agg>
inline real hl_agg_op(Agg agg, vecType mm) {
  float32x4_t rev = vrev64q_f32(mm);
  float32x4_t tmp1 = agg.vecOp(rev, rev);
  float32x2_t lo = vget_high_f32(rev);
  float32x2_t hi = vget_low_f32(rev);
  float32x4_t tmp2 = vcombine_f32(hi, lo);
  float32x4_t ret = agg.vecOp(tmp1, tmp2);

  return vgetq_lane_f32(ret, 0);
}

inline float32x4_t hl_vec_set(const real f) {
  return vdupq_n_f32(f);
}

inline float32x4_t hl_vec_classification_error(const float32x4_t a,
                                               const float32x4_t b,
                                               const float32x4_t p,
                                               const float32x4_t r) {
  uint32x4_t tmp1 = vcgtq_f32(a, p);
  uint32x4_t tmp2 = vcgtq_f32(b, p);
  uint32x4_t tmp3 = veorq_u32(tmp1, tmp2);
  return vcvtq_f32_u32(vandq_u32(tmp3, vcvtq_u32_f32(r)));
}

#else

#ifdef __aarch64__
typedef float64x2_t vecType;

/* number of float in vector */
#define VECTOR_LEN  2
#define VECTOR_SET  vdupq_n_f64

#error To be implemented
#else
#error NEON instructions does not support double precision
#endif  // __aarch64__

#endif

#endif  // HL_CPU_SIMD_NEON_CUH_
