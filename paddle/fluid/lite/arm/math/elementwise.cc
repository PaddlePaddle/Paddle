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

#include "paddle/fluid/lite/arm/math/elementwise.h"
#include "paddle/fluid/lite/arm/math/funcs.h"

namespace paddle {
namespace lite {
namespace arm {
namespace math {

template <>
void elementwise_add<float>(const float* dinx, const float* diny, float* dout,
                            int num) {
  int cnt = num >> 4;
  int remain = num % 16;
#pragma omp parallel for
  for (int i = 0; i < cnt; i++) {
    const float* dinx_ptr = dinx + (i << 4);
    const float* diny_ptr = diny + (i << 4);
    float* dout_ptr = dout + (i << 4);

    float32x4_t dinx0 = vld1q_f32(dinx_ptr);
    float32x4_t dinx1 = vld1q_f32(dinx_ptr + 4);
    float32x4_t dinx2 = vld1q_f32(dinx_ptr + 8);
    float32x4_t dinx3 = vld1q_f32(dinx_ptr + 12);

    float32x4_t diny0 = vld1q_f32(diny_ptr);
    float32x4_t diny1 = vld1q_f32(diny_ptr + 4);
    float32x4_t diny2 = vld1q_f32(diny_ptr + 8);
    float32x4_t diny3 = vld1q_f32(diny_ptr + 12);

    dinx0 = vaddq_f32(dinx0, diny0);
    dinx1 = vaddq_f32(dinx1, diny1);
    dinx2 = vaddq_f32(dinx2, diny2);
    dinx3 = vaddq_f32(dinx3, diny3);

    vst1q_f32(dout_ptr, dinx0);
    vst1q_f32(dout_ptr + 4, dinx1);
    vst1q_f32(dout_ptr + 8, dinx2);
    vst1q_f32(dout_ptr + 12, dinx3);
  }
  if (remain > 0) {
    const float* dinx_ptr = dinx + (cnt << 4);
    const float* diny_ptr = diny + (cnt << 4);
    float* dout_ptr = dout + (cnt << 4);
    for (int i = 0; i < remain; i++) {
      *dout_ptr = *dinx_ptr + *diny_ptr;
      dout_ptr++;
      dinx_ptr++;
      diny_ptr++;
    }
  }
}

template <>
void elementwise_add_relu<float>(const float* dinx, const float* diny,
                                 float* dout, int num) {
  int cnt = num >> 4;
  int remain = num % 16;
  float32x4_t vzero = vdupq_n_f32(0.f);
#pragma omp parallel for
  for (int i = 0; i < cnt; i++) {
    const float* dinx_ptr = dinx + (i << 4);
    const float* diny_ptr = diny + (i << 4);
    float* dout_ptr = dout + (i << 4);

    float32x4_t dinx0 = vld1q_f32(dinx_ptr);
    float32x4_t dinx1 = vld1q_f32(dinx_ptr + 4);
    float32x4_t dinx2 = vld1q_f32(dinx_ptr + 8);
    float32x4_t dinx3 = vld1q_f32(dinx_ptr + 12);

    float32x4_t diny0 = vld1q_f32(diny_ptr);
    float32x4_t diny1 = vld1q_f32(diny_ptr + 4);
    float32x4_t diny2 = vld1q_f32(diny_ptr + 8);
    float32x4_t diny3 = vld1q_f32(diny_ptr + 12);

    dinx0 = vaddq_f32(dinx0, diny0);
    dinx1 = vaddq_f32(dinx1, diny1);
    dinx2 = vaddq_f32(dinx2, diny2);
    dinx3 = vaddq_f32(dinx3, diny3);

    // relu
    dinx0 = vmaxq_f32(dinx0, vzero);
    dinx1 = vmaxq_f32(dinx1, vzero);
    dinx2 = vmaxq_f32(dinx2, vzero);
    dinx3 = vmaxq_f32(dinx3, vzero);

    vst1q_f32(dout_ptr, dinx0);
    vst1q_f32(dout_ptr + 4, dinx1);
    vst1q_f32(dout_ptr + 8, dinx2);
    vst1q_f32(dout_ptr + 12, dinx3);
  }
  if (remain > 0) {
    const float* dinx_ptr = dinx + (cnt << 4);
    const float* diny_ptr = diny + (cnt << 4);
    float* dout_ptr = dout + (cnt << 4);
    for (int i = 0; i < remain; i++) {
      float tmp = *dinx_ptr + *diny_ptr;
      *dout_ptr = tmp > 0.f ? tmp : 0.f;
      dout_ptr++;
      dinx_ptr++;
      diny_ptr++;
    }
  }
}

template <>
void elementwise_add_broadcast<float>(const float* dinx, const float* diny,
                                      float* dout, int batch, int channels,
                                      int num) {
#pragma omp parallel for collapse(2)
  for (int i = 0; i < batch; ++i) {
    for (int j = 0; j < channels; ++j) {
      int offset = (i * channels + j) * num;
      const float* din_ptr = dinx + offset;
      const float diny_data = diny[j];
      float* dout_ptr = dout + offset;

      int cnt = num >> 4;
      int remain = num % 16;
      float32x4_t rb = vdupq_n_f32(diny_data);
      for (int k = 0; k < cnt; ++k) {
        float32x4_t din0 = vld1q_f32(din_ptr);
        float32x4_t din1 = vld1q_f32(din_ptr + 4);
        float32x4_t din2 = vld1q_f32(din_ptr + 8);
        float32x4_t din3 = vld1q_f32(din_ptr + 12);

        din0 = vaddq_f32(din0, rb);
        din1 = vaddq_f32(din1, rb);
        din2 = vaddq_f32(din2, rb);
        din3 = vaddq_f32(din3, rb);

        vst1q_f32(dout_ptr, din0);
        vst1q_f32(dout_ptr + 4, din1);
        vst1q_f32(dout_ptr + 8, din2);
        vst1q_f32(dout_ptr + 12, din3);
        din_ptr += 16;
        dout_ptr += 16;
      }
      if (remain >= 8) {
        float32x4_t din0 = vld1q_f32(din_ptr);
        float32x4_t din1 = vld1q_f32(din_ptr + 4);
        din0 = vaddq_f32(din0, rb);
        din1 = vaddq_f32(din1, rb);
        vst1q_f32(dout_ptr, din0);
        vst1q_f32(dout_ptr + 4, din1);
        din_ptr += 8;
        dout_ptr += 8;
        remain -= 8;
      }
      if (remain >= 4) {
        float32x4_t din0 = vld1q_f32(din_ptr);
        din0 = vaddq_f32(din0, rb);
        vst1q_f32(dout_ptr, din0);
        din_ptr += 4;
        dout_ptr += 4;
        remain -= 4;
      }
      if (remain > 0) {
        for (int p = 0; p < remain; p++) {
          *dout_ptr = *din_ptr + diny_data;
          dout_ptr++;
          din_ptr++;
        }
      }
    }
  }
}

template <>
void elementwise_add_relu_broadcast<float>(const float* dinx, const float* diny,
                                           float* dout, int batch, int channels,
                                           int num) {
  float32x4_t vzero = vdupq_n_f32(0.f);
#pragma omp parallel for collapse(2)
  for (int i = 0; i < batch; ++i) {
    for (int j = 0; j < channels; ++j) {
      int offset = (i * channels + j) * num;
      const float* din_ptr = dinx + offset;
      const float diny_data = diny[j];
      float* dout_ptr = dout + offset;

      int cnt = num >> 4;
      int remain = num % 16;
      float32x4_t rb = vdupq_n_f32(diny_data);
      for (int k = 0; k < cnt; ++k) {
        float32x4_t din0 = vld1q_f32(din_ptr);
        float32x4_t din1 = vld1q_f32(din_ptr + 4);
        float32x4_t din2 = vld1q_f32(din_ptr + 8);
        float32x4_t din3 = vld1q_f32(din_ptr + 12);

        din0 = vaddq_f32(din0, rb);
        din1 = vaddq_f32(din1, rb);
        din2 = vaddq_f32(din2, rb);
        din3 = vaddq_f32(din3, rb);

        // relu
        din0 = vmaxq_f32(din0, vzero);
        din1 = vmaxq_f32(din1, vzero);
        din2 = vmaxq_f32(din2, vzero);
        din3 = vmaxq_f32(din3, vzero);

        vst1q_f32(dout_ptr, din0);
        vst1q_f32(dout_ptr + 4, din1);
        vst1q_f32(dout_ptr + 8, din2);
        vst1q_f32(dout_ptr + 12, din3);
        din_ptr += 16;
        dout_ptr += 16;
      }
      if (remain >= 8) {
        float32x4_t din0 = vld1q_f32(din_ptr);
        float32x4_t din1 = vld1q_f32(din_ptr + 4);
        din0 = vaddq_f32(din0, rb);
        din1 = vaddq_f32(din1, rb);
        // relu
        din0 = vmaxq_f32(din0, vzero);
        din1 = vmaxq_f32(din1, vzero);
        vst1q_f32(dout_ptr, din0);
        vst1q_f32(dout_ptr + 4, din1);
        din_ptr += 8;
        dout_ptr += 8;
        remain -= 8;
      }
      if (remain >= 4) {
        float32x4_t din0 = vld1q_f32(din_ptr);
        din0 = vaddq_f32(din0, rb);
        // relu
        din0 = vmaxq_f32(din0, vzero);
        vst1q_f32(dout_ptr, din0);
        din_ptr += 4;
        dout_ptr += 4;
        remain -= 4;
      }
      if (remain > 0) {
        for (int p = 0; p < remain; p++) {
          float tmp = *din_ptr + diny_data;
          *dout_ptr = tmp > 0.f ? tmp : 0.f;
          dout_ptr++;
          din_ptr++;
        }
      }
    }
  }
}

}  // namespace math
}  // namespace arm
}  // namespace lite
}  // namespace paddle
