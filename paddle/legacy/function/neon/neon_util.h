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

#if defined(__ARM_NEON__) || defined(__ARM_NEON)

#include <arm_neon.h>

namespace paddle {

namespace neon {

inline float32x4_t vld1q_f32_aligned(const float* p) {
  return vld1q_f32(
      (const float*)__builtin_assume_aligned(p, sizeof(float32x4_t)));
}

#ifndef __aarch64__
inline float32_t vaddvq_f32(float32x4_t a) {
  float32x2_t v = vadd_f32(vget_high_f32(a), vget_low_f32(a));
  return vget_lane_f32(vpadd_f32(v, v), 0);
}

#define vmlaq_laneq_f32(a, b, v, lane) \
  vmlaq_n_f32(a, b, vgetq_lane_f32(v, lane))
#endif

}  // namespace neon
}  // namespace paddle

#endif
