// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/lite/arm/math/dropout.h"
#include "paddle/fluid/lite/arm/math/funcs.h"

namespace paddle {
namespace lite {
namespace arm {
namespace math {

template <>
void dropout_down<float>(const float* din, float* dout, int num, float prob) {
  const float scale = 1.0f - prob;
  int cnt = num >> 4;
  int remain = num % 16;
  float32x4_t vscale = vdupq_n_f32(scale);
#pragma omp parallel for
  for (int i = 0; i < cnt; i++) {
    const float* din_ptr = din + (i << 4);
    float* dout_ptr = dout + (i << 4);

    float32x4_t din0 = vld1q_f32(din_ptr);
    float32x4_t din1 = vld1q_f32(din_ptr + 4);
    float32x4_t din2 = vld1q_f32(din_ptr + 8);
    float32x4_t din3 = vld1q_f32(din_ptr + 12);

    float32x4_t vmul0 = vmulq_f32(din0, vscale);
    float32x4_t vmul1 = vmulq_f32(din1, vscale);
    float32x4_t vmul2 = vmulq_f32(din2, vscale);
    float32x4_t vmul3 = vmulq_f32(din3, vscale);

    vst1q_f32(dout_ptr, vmul0);
    vst1q_f32(dout_ptr + 4, vmul1);
    vst1q_f32(dout_ptr + 8, vmul2);
    vst1q_f32(dout_ptr + 12, vmul3);
  }
  if (remain > 0) {
    const float* din_ptr = din + (cnt << 4);
    float* dout_ptr = dout + (cnt << 4);
    for (int i = 0; i < remain; i++) {
      *dout_ptr = *din_ptr * scale;
      dout_ptr++;
      din_ptr++;
    }
  }
}

template <>
void dropout_up<float>(const float* din, float* dout, int num) {
  int cnt = num >> 4;
  int remain = num % 16;
#pragma omp parallel for
  for (int i = 0; i < cnt; i++) {
    const float* din_ptr = din + (i << 4);
    float* dout_ptr = dout + (i << 4);

    float32x4_t din0 = vld1q_f32(din_ptr);
    float32x4_t din1 = vld1q_f32(din_ptr + 4);
    float32x4_t din2 = vld1q_f32(din_ptr + 8);
    float32x4_t din3 = vld1q_f32(din_ptr + 12);

    vst1q_f32(dout_ptr, din0);
    vst1q_f32(dout_ptr + 4, din1);
    vst1q_f32(dout_ptr + 8, din2);
    vst1q_f32(dout_ptr + 12, din3);
  }
  if (remain > 0) {
    const float* din_ptr = din + (cnt << 4);
    float* dout_ptr = dout + (cnt << 4);
    for (int i = 0; i < remain; i++) {
      *dout_ptr = *din_ptr;
      dout_ptr++;
      din_ptr++;
    }
  }
}

}  // namespace math
}  // namespace arm
}  // namespace lite
}  // namespace paddle
