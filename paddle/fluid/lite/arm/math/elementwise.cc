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

    float32x4_t vsum0 = vaddq_f32(dinx0, diny0);
    float32x4_t vsum1 = vaddq_f32(dinx1, diny1);
    float32x4_t vsum2 = vaddq_f32(dinx2, diny2);
    float32x4_t vsum3 = vaddq_f32(dinx3, diny3);

    vst1q_f32(dout_ptr, vsum0);
    vst1q_f32(dout_ptr + 4, vsum1);
    vst1q_f32(dout_ptr + 8, vsum2);
    vst1q_f32(dout_ptr + 12, vsum3);
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

}  // namespace math
}  // namespace arm
}  // namespace lite
}  // namespace paddle
