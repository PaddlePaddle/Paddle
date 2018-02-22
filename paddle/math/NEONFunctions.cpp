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

#if defined(__ARM_NEON__) || defined(__ARM_NEON)

#include "NEONFunctions.h"
#include <arm_neon.h>

namespace paddle {
namespace neon {

// b[i] = a[i] > 0.0f ? a[i] : 0.0f
void relu(const float* a, float* b, int len) {
  int offset = len % 16;
  float32x4_t ma0, ma1, ma2, ma3;
  float32x4_t mb0, mb1, mb2, mb3;

  float32x4_t zero = vdupq_n_f32(0.f);
  for (int k = 0; k < len / 16; k++, a += 16, b += 16) {
    ma0 = vld1q_f32(a);
    ma1 = vld1q_f32(a + 4);
    ma2 = vld1q_f32(a + 8);
    ma3 = vld1q_f32(a + 12);

    mb0 = vmaxq_f32(ma0, zero);
    mb1 = vmaxq_f32(ma1, zero);
    mb2 = vmaxq_f32(ma2, zero);
    mb3 = vmaxq_f32(ma3, zero);

    vst1q_f32(b, mb0);
    vst1q_f32(b + 4, mb1);
    vst1q_f32(b + 8, mb2);
    vst1q_f32(b + 12, mb3);
  }

  for (int i = 0; i < offset; i++) {
    b[i] = a[i] > 0.0f ? a[i] : 0.0f;
  }
}

// b[i] = a[i] > 0.0f ? a[i] : a[i] * w
void prelu(const float* a, float w, float* b, int len) {
  int offset = len % 16;
  float32x4_t ma0, ma1, ma2, ma3;

  float32x4_t zero = vdupq_n_f32(0.f);
  float32x4_t vw = vdupq_n_f32(w);

  for (int k = 0; k < len / 16; k++, a += 16, b += 16) {
    ma0 = vld1q_f32(a);
    ma1 = vld1q_f32(a + 4);
    ma2 = vld1q_f32(a + 8);
    ma3 = vld1q_f32(a + 12);

    uint32x4_t flag0 = vcgtq_f32(ma0, zero);
    uint32x4_t flag1 = vcgtq_f32(ma1, zero);
    uint32x4_t flag2 = vcgtq_f32(ma2, zero);
    uint32x4_t flag3 = vcgtq_f32(ma3, zero);

    float32x4_t mul0 = vmulq_f32(ma0, vw);
    float32x4_t mul1 = vmulq_f32(ma1, vw);
    float32x4_t mul2 = vmulq_f32(ma2, vw);
    float32x4_t mul3 = vmulq_f32(ma3, vw);

    ma0 = vbslq_f32(flag0, ma0, mul0);
    ma1 = vbslq_f32(flag1, ma1, mul1);
    ma2 = vbslq_f32(flag2, ma2, mul2);
    ma3 = vbslq_f32(flag3, ma3, mul3);

    vst1q_f32(b, ma0);
    vst1q_f32(b + 4, ma1);
    vst1q_f32(b + 8, ma2);
    vst1q_f32(b + 12, ma3);
  }

  for (int i = 0; i < offset; i++) {
    b[i] = a[i] > 0.0f ? a[i] : a[i] * w;
  }
}

}  // namespace neon
}  // namespace paddle

#endif
