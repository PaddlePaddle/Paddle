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

template <>
void scale<float>(const float* din, float* dout, int outer_dim, int scale_dim,
                  int inner_dim, const float* scale_data,
                  const float* bias_data) {
  int cnt = inner_dim >> 4;
  int remain = inner_dim % 16;
  int size = inner_dim * scale_dim;
  for (int n = 0; n < outer_dim; n++) {
    const float* din_ptr_n = din + n * size;
    float* dout_ptr_n = dout + n * size;
#pragma omp parallel for
    for (int i = 0; i < scale_dim; i++) {
      const float* din_ptr = din_ptr_n + i * inner_dim;
      float* dout_ptr = dout_ptr_n + i * inner_dim;
      float scale = scale_data[i];
      float32x4_t vscale = vdupq_n_f32(scale);
      float bias = bias_data[i];
      float32x4_t vbias = vdupq_n_f32(bias);
      for (int j = 0; j < cnt; j++) {
        float32x4_t din0 = vld1q_f32(din_ptr);
        float32x4_t din1 = vld1q_f32(din_ptr + 4);
        float32x4_t din2 = vld1q_f32(din_ptr + 8);
        float32x4_t din3 = vld1q_f32(din_ptr + 12);

        float32x4_t vsum1 = vmlaq_f32(vbias, din0, vscale);
        float32x4_t vsum2 = vmlaq_f32(vbias, din1, vscale);
        float32x4_t vsum3 = vmlaq_f32(vbias, din2, vscale);
        float32x4_t vsum4 = vmlaq_f32(vbias, din3, vscale);

        din_ptr += 16;
        vst1q_f32(dout_ptr, vsum1);
        vst1q_f32(dout_ptr + 4, vsum2);
        vst1q_f32(dout_ptr + 8, vsum3);
        vst1q_f32(dout_ptr + 12, vsum4);

        dout_ptr += 16;
      }
      for (int j = 0; j < remain; j++) {
        *dout_ptr = *din_ptr * scale + bias;
        dout_ptr++;
        din_ptr++;
      }
    }
  }
}

template <>
void scale<float>(const float* din, float* dout, int outer_dim, int scale_dim,
                  const float* scale_data, const float* bias_data) {
  int cnt = scale_dim >> 4;
  int remain = scale_dim % 16;
  for (int n = 0; n < outer_dim; n++) {
    const float* din_ptr_n = din + n * scale_dim;
    float* dout_ptr_n = dout + n * scale_dim;
#pragma omp parallel for
    for (int i = 0; i < cnt; i++) {
      int idx = i << 4;
      const float* din_ptr = din_ptr_n + idx;
      const float* scale_ptr = scale_data + idx;
      const float* bias_ptr = bias_data + idx;
      float* dout_ptr = dout_ptr_n + idx;

      float32x4_t din0 = vld1q_f32(din_ptr);
      float32x4_t vscale0 = vld1q_f32(scale_ptr);
      float32x4_t vbias0 = vld1q_f32(bias_ptr);

      float32x4_t din1 = vld1q_f32(din_ptr + 4);
      float32x4_t vscale1 = vld1q_f32(scale_ptr + 4);
      float32x4_t vbias1 = vld1q_f32(bias_ptr + 4);

      float32x4_t din2 = vld1q_f32(din_ptr + 8);
      float32x4_t vscale2 = vld1q_f32(scale_ptr + 8);
      float32x4_t vbias2 = vld1q_f32(bias_ptr + 8);

      float32x4_t vsum1 = vmlaq_f32(vbias0, din0, vscale0);
      float32x4_t vsum2 = vmlaq_f32(vbias1, din1, vscale1);

      float32x4_t din3 = vld1q_f32(din_ptr + 12);
      float32x4_t vscale3 = vld1q_f32(scale_ptr + 12);
      float32x4_t vbias3 = vld1q_f32(bias_ptr + 12);

      vst1q_f32(dout_ptr, vsum1);
      vst1q_f32(dout_ptr + 4, vsum2);

      float32x4_t vsum3 = vmlaq_f32(vbias2, din2, vscale2);
      float32x4_t vsum4 = vmlaq_f32(vbias3, din3, vscale3);

      vst1q_f32(dout_ptr + 8, vsum3);
      vst1q_f32(dout_ptr + 12, vsum4);
    }
    int idx = cnt << 4;
    const float* din_ptr = din_ptr_n + idx;
    float* dout_ptr = dout_ptr_n + idx;
    const float* scale_ptr = scale_data + idx;
    const float* bias_ptr = bias_data + idx;
    for (int j = 0; j < remain; j++) {
      *dout_ptr = *din_ptr * (*scale_ptr) + (*bias_ptr);
      dout_ptr++;
      din_ptr++;
      scale_ptr++;
      bias_ptr++;
    }
  }
}

}  // namespace math
}  // namespace arm
}  // namespace lite
}  // namespace paddle
