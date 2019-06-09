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

#include "paddle/fluid/lite/arm/math/scale.h"
#include "paddle/fluid/lite/arm/math/funcs.h"

namespace paddle {
namespace lite {
namespace arm {
namespace math {

template <>
void scale<float>(const float* din, float* dout, int num, float scale,
                  float bias) {
  int cnt = num >> 4;
  int remain = num % 16;
  float32x4_t vscale = vdupq_n_f32(scale);
  float32x4_t vbias = vdupq_n_f32(bias);
#pragma omp parallel for
  for (int i = 0; i < cnt; i++) {
    const float* din_ptr = din + (i << 4);
    float* dout_ptr = dout + (i << 4);

    float32x4_t din0 = vld1q_f32(din_ptr);
    float32x4_t din1 = vld1q_f32(din_ptr + 4);
    float32x4_t din2 = vld1q_f32(din_ptr + 8);
    float32x4_t din3 = vld1q_f32(din_ptr + 12);

    float32x4_t vsum1 = vmlaq_f32(vbias, din0, vscale);
    float32x4_t vsum2 = vmlaq_f32(vbias, din1, vscale);
    float32x4_t vsum3 = vmlaq_f32(vbias, din2, vscale);
    float32x4_t vsum4 = vmlaq_f32(vbias, din3, vscale);

    vst1q_f32(dout_ptr, vsum1);
    vst1q_f32(dout_ptr + 4, vsum2);
    vst1q_f32(dout_ptr + 8, vsum3);
    vst1q_f32(dout_ptr + 12, vsum4);
  }
  if (remain > 0) {
    const float* din_ptr = din + (cnt << 4);
    float* dout_ptr = dout + (cnt << 4);
    for (int i = 0; i < remain; i++) {
      *dout_ptr = *din_ptr * scale + bias;
      dout_ptr++;
      din_ptr++;
    }
  }
}

}  // namespace math
}  // namespace arm
}  // namespace lite
}  // namespace paddle
