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

#include "paddle/fluid/lite/arm/math/funcs.h"
#include <arm_neon.h>

namespace paddle {
namespace lite {
namespace arm {
namespace math {

template <>
void fill_bias_fc<float>(float *out, const float *bias, const int num,
                         const int channel) {
  int cnt = channel >> 4;
  int remain = channel & 15;

  for (int j = 0; j < num; ++j) {
    const float *ptr_bias = bias;
    float *ptr_out = out + j * channel;

    float32x4_t vout1;
    float32x4_t vout2;
    float32x4_t vout3;
    float32x4_t vout4;

    for (int i = 0; i < cnt; ++i) {
      float32x4_t vin1 = vld1q_f32(ptr_out);
      float32x4_t vb1 = vld1q_f32(ptr_bias);

      float32x4_t vin2 = vld1q_f32(ptr_out + 4);
      float32x4_t vb2 = vld1q_f32(ptr_bias + 4);

      float32x4_t vin3 = vld1q_f32(ptr_out + 8);
      float32x4_t vb3 = vld1q_f32(ptr_bias + 8);

      float32x4_t vin4 = vld1q_f32(ptr_out + 12);
      float32x4_t vb4 = vld1q_f32(ptr_bias + 12);

      vout1 = vaddq_f32(vin1, vb1);
      vout2 = vaddq_f32(vin2, vb2);
      vout3 = vaddq_f32(vin3, vb3);
      vout4 = vaddq_f32(vin4, vb4);

      vst1q_f32(ptr_out, vout1);
      vst1q_f32(ptr_out + 4, vout2);
      vst1q_f32(ptr_out + 8, vout3);
      vst1q_f32(ptr_out + 12, vout4);

      ptr_out += 16;
      ptr_bias += 16;
    }
#if 0
        if (cnt > 0) {
            asm(
            "1: \n"
            "vld1.32 {d0-d1}, [%[ptr_out]]    @ load data\n"
            "vld1.32 {d2-d3}, [%[ptr_bias]]!  @ load data\n"
            "vadd.f32 q2, q0, q1              @ add bias\n"
            "vst1.32  {d4-d5}, [%[ptr_out]]!  @ store result\n"
            "subs   %[cnt], #1                @ loop count -1\n"
            "bne    1b                        @ jump to main loop\n"
            :[ptr_out] "+r"(ptr_out), [ptr_bias] "+r"(ptr_bias), \
                    [cnt] "+r"(cnt)
            :
            :"q0", "q1", "q2"
            );
        }
#endif
    for (int i = 0; i < remain; ++i) {
      *(ptr_out++) += *(ptr_bias++);
    }
  }
}

template <>
void fill_bias_fc<int>(int *out, const int *bias, const int num,
                       const int channel) {
  int cnt = channel >> 4;
  int remain = channel & 15;

  for (int j = 0; j < num; ++j) {
    const int *ptr_bias = bias;
    int *ptr_out = out + j * channel;

    int32x4_t vout1;
    int32x4_t vout2;
    int32x4_t vout3;
    int32x4_t vout4;

    for (int i = 0; i < cnt; ++i) {
      int32x4_t vin1 = vld1q_s32(ptr_out);
      int32x4_t vb1 = vld1q_s32(ptr_bias);

      int32x4_t vin2 = vld1q_s32(ptr_out + 4);
      int32x4_t vb2 = vld1q_s32(ptr_bias + 4);

      int32x4_t vin3 = vld1q_s32(ptr_out + 8);
      int32x4_t vb3 = vld1q_s32(ptr_bias + 8);

      int32x4_t vin4 = vld1q_s32(ptr_out + 12);
      int32x4_t vb4 = vld1q_s32(ptr_bias + 12);

      vout1 = vaddq_s32(vin1, vb1);
      vout2 = vaddq_s32(vin2, vb2);
      vout3 = vaddq_s32(vin3, vb3);
      vout4 = vaddq_s32(vin4, vb4);

      vst1q_s32(ptr_out, vout1);
      vst1q_s32(ptr_out + 4, vout2);
      vst1q_s32(ptr_out + 8, vout3);
      vst1q_s32(ptr_out + 12, vout4);

      ptr_out += 16;
      ptr_bias += 16;
    }

#if 0
        if (cnt > 0) {
        asm(
        "1: \n"
        "vld1.32 {d0-d1}, [%[ptr_out]]    @ load data\n"
        "vld1.32 {d2-d3}, [%[ptr_bias]]!  @ load data\n"
        "vadd.s32 q2, q0, q1              @ add bias\n"
        "vst1.32  {d4-d5}, [%[ptr_out]]!  @ store result\n"
        "subs   %[cnt], #1                @ loop count -1\n"
        "bne    1b                        @ jump to main loop\n"
        :[ptr_out] "+r"(ptr_out), [ptr_bias] "+r"(ptr_bias), \
                [cnt] "+r"(cnt)
        :
        :"q0", "q1", "q2"
        );
    }
#endif
    for (int i = 0; i < remain; ++i) {
      *(ptr_out++) += *(ptr_bias++);
    }
  }
}

}  // namespace math
}  // namespace arm
}  // namespace lite
}  // namespace paddle
