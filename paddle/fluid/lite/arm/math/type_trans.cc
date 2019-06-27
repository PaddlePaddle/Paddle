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

#include "paddle/fluid/lite/arm/math/type_trans.h"
#include <arm_neon.h>
#include <string.h>
#include <vector>
#include "paddle/fluid/lite/arm/math/saturate.h"

namespace paddle {
namespace lite {
namespace arm {
namespace math {

template <typename dtype>
void int32_to_dtype(const int* din, dtype* dout, const float* scale,
                    int axis_size, int64_t outer_size, int64_t inner_size);

void fp32_to_int8(const float* din, int8_t* dout, const float* scale,
                  int axis_size, int64_t outer_size, int64_t inner_size) {
  int cnt = inner_size / 16;
  int remain = inner_size & 15;
  int64_t loop_size = outer_size * axis_size;

#pragma omp parallel for
  for (int j = 0; j < loop_size; ++j) {
    float inv_scale = 1.f / scale[j % axis_size];
    float32x4_t vzero = vdupq_n_f32(0.f);
    float32x4_t vscale = vdupq_n_f32(inv_scale);
    float32x4_t vpoff = vdupq_n_f32(0.5f);
    float32x4_t vnoff = vdupq_n_f32(-0.5f);
    const float* din_c = din + j * inner_size;
    signed char* dout_c = dout + j * inner_size;
    if (cnt > 0) {
      int cnt_loop = cnt;
      const float* din_ptr = din_c;
      signed char* dout_ptr = dout_c;
#ifdef __aarch64__
      asm volatile(
          "ldp q0, q1, [%[in]], #32                           \n"
          "ldp q2, q3, [%[in]], #32                   \n"
          "0:                                         \n" /* main loop */
          "fmul v4.4s, v0.4s, %[scale].4s             \n"
          "fmul v5.4s, v1.4s, %[scale].4s             \n"
          "fmul v6.4s, v2.4s, %[scale].4s             \n"
          "fmul v7.4s, v3.4s, %[scale].4s             \n"
          "ldp q0, q1, [%[in]], #32                   \n"
          "subs %[cnt], %[cnt], #1                    \n"
          "FCVTAS v8.4s, v4.4s                        \n"
          "FCVTAS v9.4s, v5.4s                        \n"
          "FCVTAS v10.4s, v6.4s                       \n"
          "FCVTAS v11.4s, v7.4s                       \n"
          "ldp q2, q3, [%[in]], #32                   \n"
          "sqxtn    v4.4h, v8.4s                      \n"
          "sqxtn2   v4.8h, v9.4s                      \n"
          "sqxtn    v5.4h, v10.4s                     \n"
          "sqxtn2   v5.8h, v11.4s                     \n"
          "sqxtn    v8.8b, v4.8h                      \n"
          "sqxtn2   v8.16b, v5.8h                     \n"
          "str q8, [%[out]], #16                      \n"
          "bne    0b                                  \n"
          : [in] "+r"(din_ptr), [out] "+r"(dout_ptr), [cnt] "+r"(cnt_loop)
          : [scale] "w"(vscale)
          : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10",
            "v11");
#else
      asm volatile(
          "vld1.32 {d0-d3},    [%[din]]!                  @ load in0~in7\n"
          "vld1.32    {d4-d7},    [%[din]]!       @ load in8~in16\n"
          "0:                                     @ main loop\n"
          "vand.i32   q4, %q[vpoff], %q[vpoff]    @ set offset, 0.5\n"
          "vand.i32   q5, q4, q4                  @ set offset, 0.5\n"
          "vand.i32   q6, q4, q4                  @ set offset, 0.5\n"
          "vand.i32   q7, q4, q4                  @ set offset, 0.5\n"
          "vcgt.f32   q8, q0, %q[vzero]           @ get mask > 0, in0\n"
          "vcgt.f32   q9, q1, %q[vzero]           @ get mask > 0, in1\n"
          "vcgt.f32   q10, q2, %q[vzero]          @ get mask > 0, in2\n"
          "vcgt.f32   q11, q3, %q[vzero]          @ get mask > 0, in3\n"
          "vbif.f32   q4, %q[vnoff], q8           @ get right offset\n"
          "vbif.f32   q5, %q[vnoff], q9           @ get right offset\n"
          "vbif.f32   q6, %q[vnoff], q10          @ get right offset\n"
          "vbif.f32   q7, %q[vnoff], q11          @ get right offset\n"
          "vmla.f32   q4, q0, %q[vscale]          @ mul scale\n"
          "vmla.f32   q5, q1, %q[vscale]          @ mul scale\n"
          "vmla.f32   q6, q2, %q[vscale]          @ mul scale\n"
          "vmla.f32   q7, q3, %q[vscale]          @ mul scale\n"
          "vcvt.s32.f32  q0, q4                   @ cvt to int32\n"
          "vcvt.s32.f32  q1, q5                   @ cvt to int32\n"
          "vcvt.s32.f32  q2, q6                   @ cvt to int32\n"
          "vcvt.s32.f32  q3, q7                   @ cvt to int32\n"
          "vqmovn.s32 d8, q0                      @ cnt to int16\n"
          "vqmovn.s32 d9, q1                      @ cnt to int16\n"
          "vqmovn.s32 d10, q2                     @ cnt to int16\n"
          "vqmovn.s32 d11, q3                     @ cnt to int16\n"
          "vld1.32 {d0-d3},    [%[din]]!          @ load in0~in7\n"
          "vqmovn.s16 d12, q4                     @ cnt to int8\n"
          "vqmovn.s16 d13, q5                     @ cnt to int8\n"
          "vld1.32 {d4-d7},    [%[din]]!          @ load in8~in16\n"
          "vst1.32    {d12-d13},  [%[dout]]!      @ write to output\n"
          "subs   %[cnt], #1                      @ loop count -1\n"
          "bne    0b                              @ to main loop\n"

          : [dout] "+r"(dout_ptr), [din] "+r"(din_ptr), [cnt] "+r"(cnt_loop)
          : [vscale] "w"(vscale), [vpoff] "w"(vpoff), [vnoff] "w"(vnoff),
            [vzero] "w"(vzero)
          : "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7", "q8", "q9", "q10",
            "q11");
#endif
    }
    const float* din_r = din_c + 16 * cnt;
    signed char* dout_r = dout_c + 16 * cnt;
    for (int i = 0; i < remain; ++i) {
      dout_r[i] = saturate_cast<int8_t>(roundf(inv_scale * din_r[i]));
    }
  }
}

void fp32_to_int16(const float* din, int16_t* dout, const float* scale,
                   int axis_size, int64_t outer_size, int64_t inner_size) {
  int cnt = inner_size / 8;
  int remain = inner_size & 7;
  int64_t loop_size = outer_size * axis_size;

#pragma omp parallel for
  for (int j = 0; j < loop_size; ++j) {
    float inv_scale = 1.f / scale[j % axis_size];
    float32x4_t vzero = vdupq_n_f32(0.f);
    float32x4_t vscale = vdupq_n_f32(inv_scale);
    float32x4_t vpoff = vdupq_n_f32(0.5f);
    float32x4_t vnoff = vdupq_n_f32(-0.5f);
    const float* din_c = din + j * inner_size;
    int16_t* dout_c = dout + j * inner_size;
    if (cnt > 0) {
      int cnt_loop = cnt;
      const float* din_ptr = din_c;
      int16_t* dout_ptr = dout_c;
#ifdef __aarch64__
      asm volatile(
          "ldp q0, q1, [%[in]], #32                   \n"
          "0:                                         \n" /* main loop */
          "fmul v4.4s, v0.4s, %[scale].4s             \n"
          "fmul v5.4s, v1.4s, %[scale].4s             \n"
          "ldp q0, q1, [%[in]], #32                   \n"
          "subs %[cnt], %[cnt], #1                    \n"
          "FCVTAS v8.4s, v4.4s                        \n"
          "FCVTAS v9.4s, v5.4s                        \n"
          "sqxtn    v4.4h, v8.4s                      \n"
          "sqxtn2   v4.8h, v9.4s                      \n"
          "str q4, [%[out]], #16                      \n"
          "bne    0b                                  \n"
          : [in] "+r"(din_ptr), [out] "+r"(dout_ptr), [cnt] "+r"(cnt_loop)
          : [scale] "w"(vscale)
          : "v0", "v1", "v4", "v5", "v8", "v9");
#else
      asm volatile(
          "vld1.32 {d0-d3}, [%[din]]!             @ load in0~in7\n"
          "0:                                     @ main loop\n"
          "vand.i32   q4, %q[vpoff], %q[vpoff]    @ set offset, 0.5\n"
          "vand.i32   q5, q4, q4                  @ set offset, 0.5\n"
          "vand.i32   q6, q4, q4                  @ set offset, 0.5\n"
          "vand.i32   q7, q4, q4                  @ set offset, 0.5\n"
          "vcgt.f32   q8, q0, %q[vzero]           @ get mask > 0, in0\n"
          "vcgt.f32   q9, q1, %q[vzero]           @ get mask > 0, in1\n"
          "vbif.f32   q4, %q[vnoff], q8           @ get right offset\n"
          "vbif.f32   q5, %q[vnoff], q9           @ get right offset\n"
          "vmla.f32   q4, q0, %q[vscale]          @ mul scale\n"
          "vmla.f32   q5, q1, %q[vscale]          @ mul scale\n"
          "vcvt.s32.f32  q0, q4                   @ cvt to int32\n"
          "vcvt.s32.f32  q1, q5                   @ cvt to int32\n"
          "vqmovn.s32 d8, q0                      @ cnt to int16\n"
          "vqmovn.s32 d9, q1                      @ cnt to int16\n"
          "vld1.32 {d0-d3},  [%[din]]!            @ load in0~in7\n"
          "vst1.32 {d8-d9},  [%[dout]]!           @ write to output\n"
          "subs   %[cnt], #1                      @ loop count -1\n"
          "bne    0b                              @ to main loop\n"

          : [dout] "+r"(dout_ptr), [din] "+r"(din_ptr), [cnt] "+r"(cnt_loop)
          : [vscale] "w"(vscale), [vpoff] "w"(vpoff), [vnoff] "w"(vnoff),
            [vzero] "w"(vzero)
          : "q0", "q1", "q4", "q5", "q6", "q7", "q8", "q9");
#endif
    }
    const float* din_r = din_c + 8 * cnt;
    int16_t* dout_r = dout_c + 8 * cnt;
    for (int i = 0; i < remain; ++i) {
      dout_r[i] = saturate_cast<int16_t>(roundf(inv_scale * din_r[i]));
    }
  }
}

void int8_to_fp32(const int8_t* in, float* out, const float* scale,
                  int axis_size, int64_t outer_size, int64_t inner_size) {
  int cnt = inner_size / 16;
  int remain = inner_size & 15;
  int64_t loop_size = axis_size * outer_size;
#pragma omp parallel for
  for (int64_t n = 0; n < loop_size; ++n) {
    float in_scale = scale[n % axis_size];
    const signed char* din_c = in + n * inner_size;
    float* dout_c = out + n * inner_size;
    float32x4_t vscale = vdupq_n_f32(in_scale);
    if (cnt > 0) {
      int loop = cnt;
      const signed char* din_ptr = din_c;
      float* dout_ptr = dout_c;
#ifdef __aarch64__
      asm volatile(
          "ldp     d0, d1, [%[in]], #16               \n" /* load 16 int8*/
          "0:                                 \n"         /* main loop */
          "sshll   v2.8h, v0.8b, #0           \n"         /* trans to int16*/
          "sshll   v3.8h, v1.8b, #0           \n"         /* trans to int16*/

          "sshll   v4.4s, v2.4h, #0           \n" /* trans to int32*/
          "sshll2  v5.4s, v2.8h, #0           \n" /* trans to int32*/
          "sshll   v6.4s, v3.4h, #0           \n" /* trans to int32*/
          "sshll2  v7.4s, v3.8h, #0           \n" /* trans to int32*/

          "ldp     d0, d1, [%[in]], #16       \n" /* load 16 int8*/

          "scvtf   v8.4s, v4.4s               \n" /* trans to fp32*/
          "scvtf   v9.4s, v5.4s               \n" /* trans to fp32*/
          "scvtf   v10.4s, v6.4s              \n" /* trans to fp32*/
          "scvtf   v11.4s, v7.4s              \n" /* trans to fp32*/

          "subs    %[loop], %[loop], #1       \n"

          "fmul    v4.4s, v8.4s, %[scale].4s  \n" /* mul with scale*/
          "fmul    v5.4s, v9.4s, %[scale].4s  \n" /* mul with scale*/
          "fmul    v6.4s, v10.4s, %[scale].4s \n" /* mul with scale*/
          "fmul    v7.4s, v11.4s, %[scale].4s \n" /* mul with scale*/

          "stp     q4, q5, [%[out]], #32      \n" /* write to memory*/
          "stp     q6, q7, [%[out]], #32      \n" /* write to memory*/

          "bne     0b                         \n"
          : [loop] "+r"(loop), [in] "+r"(din_ptr), [out] "+r"(dout_ptr)
          : [scale] "w"(vscale)
          : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10",
            "v11");
#else
      asm volatile(
          "vld1.32    {d0-d1},    [%[in]]!            @ load 16 int8\n"
          "0:                                 @ main loop\n"
          "vmovl.s8      q2, d0               @ trans to int16\n"
          "vmovl.s8      q3, d1               @ trans to int16\n"
          "vmovl.s16     q4, d4               @ trans to int32\n"
          "vmovl.s16     q5, d5               @ trans to int32\n"
          "vmovl.s16     q6, d6               @ trans to int32\n"
          "vmovl.s16     q7, d7               @ trans to int32\n"
          "vcvt.f32.s32  q0, q4               @ trans to fp32\n"
          "vcvt.f32.s32  q1, q5               @ trans to fp32\n"
          "vcvt.f32.s32  q2, q6               @ trans to fp32\n"
          "vcvt.f32.s32  q3, q7               @ trans to fp32\n"
          "vmul.f32      q4, q0, %q[scale]    @ mul with scale\n"
          "vmul.f32      q5, q1, %q[scale]    @ mul with scale\n"
          "vmul.f32      q6, q2, %q[scale]    @ mul with scale\n"
          "vmul.f32      q7, q3, %q[scale]    @ mul with scale\n"

          "vld1.32    {d0-d1},    [%[in]]!    @ load 16 int8\n"

          "subs          %[loop], #1            \n"

          "vst1.f32      {d8-d11}, [%[out]]!  @ write to memory\n"
          "vst1.f32      {d12-d15}, [%[out]]! @ write to memory\n"

          "bne           0b                     \n"
          : [loop] "+r"(loop), [in] "+r"(din_ptr), [out] "+r"(dout_ptr)
          : [scale] "w"(vscale)
          : "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7");
#endif  // __aarch64__
    }
    const signed char* din_r = din_c + 16 * cnt;
    float* dout_r = dout_c + 16 * cnt;
    for (int i = 0; i < remain; ++i) {
      dout_r[i] = in_scale * din_r[i];
    }
  }
}

void int16_to_fp32(const int16_t* in, float* out, const float* scale,
                   int axis_size, int64_t outer_size, int64_t inner_size) {
  int cnt = inner_size / 16;
  int remain = inner_size & 15;
  int64_t loop_size = axis_size * outer_size;
#pragma omp parallel for
  for (int64_t n = 0; n < loop_size; ++n) {
    float in_scale = scale[n % axis_size];
    const int16_t* din_c = in + n * inner_size;
    float* dout_c = out + n * inner_size;
    float32x4_t vscale = vdupq_n_f32(in_scale);
    if (cnt > 0) {
      int loop = cnt;
      const int16_t* din_ptr = din_c;
      float* dout_ptr = dout_c;
#ifdef __aarch64__
      asm volatile(
          "ldp     q0, q1, [%[in]], #32               \n" /* load 16 int16*/
          "0:                                 \n"         /* main loop */
          "sshll   v4.4s, v0.4h, #0           \n"         /* trans to int32*/
          "sshll2  v5.4s, v0.8h, #0           \n"         /* trans to int32*/
          "sshll   v6.4s, v1.4h, #0           \n"         /* trans to int32*/
          "sshll2  v7.4s, v1.8h, #0           \n"         /* trans to int32*/

          "ldp     q0, q1, [%[in]], #32       \n" /* load 16 int16*/

          "scvtf   v8.4s, v4.4s               \n" /* trans to fp32*/
          "scvtf   v9.4s, v5.4s               \n" /* trans to fp32*/
          "scvtf   v10.4s, v6.4s              \n" /* trans to fp32*/
          "scvtf   v11.4s, v7.4s              \n" /* trans to fp32*/

          "subs    %[loop], %[loop], #1       \n"

          "fmul    v4.4s, v8.4s, %[scale].4s  \n" /* mul with scale*/
          "fmul    v5.4s, v9.4s, %[scale].4s  \n" /* mul with scale*/
          "fmul    v6.4s, v10.4s, %[scale].4s \n" /* mul with scale*/
          "fmul    v7.4s, v11.4s, %[scale].4s \n" /* mul with scale*/

          "stp     q4, q5, [%[out]], #32      \n" /* write to memory*/
          "stp     q6, q7, [%[out]], #32      \n" /* write to memory*/

          "bne     0b                         \n"
          : [loop] "+r"(loop), [in] "+r"(din_ptr), [out] "+r"(dout_ptr)
          : [scale] "w"(vscale)
          : "v0", "v1", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11");
#else
      asm volatile(
          "vld1.32    {d0-d3},    [%[in]]!            @ load 16 int16\n"
          "0:                                 @ main loop\n"
          "vmovl.s16     q4, d0               @ trans to int32\n"
          "vmovl.s16     q5, d1               @ trans to int32\n"
          "vmovl.s16     q6, d2               @ trans to int32\n"
          "vmovl.s16     q7, d3               @ trans to int32\n"
          "vcvt.f32.s32  q0, q4               @ trans to fp32\n"
          "vcvt.f32.s32  q1, q5               @ trans to fp32\n"
          "vcvt.f32.s32  q2, q6               @ trans to fp32\n"
          "vcvt.f32.s32  q3, q7               @ trans to fp32\n"
          "vmul.f32      q4, q0, %q[scale]    @ mul with scale\n"
          "vmul.f32      q5, q1, %q[scale]    @ mul with scale\n"
          "vmul.f32      q6, q2, %q[scale]    @ mul with scale\n"
          "vmul.f32      q7, q3, %q[scale]    @ mul with scale\n"

          "vld1.32    {d0-d3},    [%[in]]!    @ load 16 int8\n"

          "subs          %[loop], #1            \n"

          "vst1.f32      {d8-d11}, [%[out]]!  @ write to memory\n"
          "vst1.f32      {d12-d15}, [%[out]]! @ write to memory\n"

          "bne           0b                     \n"
          : [loop] "+r"(loop), [in] "+r"(din_ptr), [out] "+r"(dout_ptr)
          : [scale] "w"(vscale)
          : "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7");
#endif  // __aarch64__
    }
    const int16_t* din_r = din_c + 16 * cnt;
    float* dout_r = dout_c + 16 * cnt;
    for (int i = 0; i < remain; ++i) {
      dout_r[i] = in_scale * din_r[i];
    }
  }
}

void int32_to_fp32(const int* din, float* dout, const float* scale,
                   int axis_size, int64_t outer_size, int64_t inner_size) {
  int cnt = inner_size / 16;
  int remain = inner_size & 15;
  int64_t loop_size = axis_size * outer_size;
#pragma omp parallel for
  for (int64_t n = 0; n < loop_size; ++n) {
    float in_scale = scale[n % axis_size];
    const int* din_c = din + n * inner_size;
    float* dout_c = dout + n * inner_size;
    float32x4_t vscale = vdupq_n_f32(in_scale);
    if (cnt > 0) {
      int loop = cnt;
      const int* din_ptr = din_c;
      float* dout_ptr = dout_c;
#ifdef __aarch64__
      asm volatile(
          "ldp     q0, q1, [%[in]], #32               \n"
          "ldp  q2, q3, [%[in]], #32          \n"
          "0:                                 \n"
          "scvtf   v4.4s, v0.4s               \n"
          "scvtf   v5.4s, v1.4s               \n"
          "scvtf   v6.4s, v2.4s               \n"
          "scvtf   v7.4s, v3.4s               \n"
          "ldp  q0, q1, [%[in]], #32          \n"
          "fmul    v8.4s, v4.4s, %[scale].4s  \n"
          "fmul    v9.4s, v5.4s, %[scale].4s  \n"
          "fmul    v10.4s, v6.4s, %[scale].4s \n"
          "fmul    v11.4s, v7.4s, %[scale].4s \n"
          "ldp  q2, q3, [%[in]], #32          \n"
          "stp     q8, q9, [%[out]], #32      \n"
          "stp     q10, q11, [%[out]], #32    \n"
          "subs    %[loop], %[loop], #1       \n"
          "bne     0b                         \n"
          : [loop] "+r"(loop), [in] "+r"(din_ptr), [out] "+r"(dout_ptr)
          : [scale] "w"(vscale)
          : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10",
            "v11");
#else
      asm volatile(
          "vld1.s32       {d0-d3}, [%[in]]!               \n"
          "vld1.s32       {d4-d7}, [%[in]]!       \n"
          "0:                                     \n"
          "vcvt.f32.s32   q4, q0                  \n"
          "vcvt.f32.s32   q5, q1                  \n"
          "vcvt.f32.s32   q6, q2                  \n"
          "vcvt.f32.s32   q7, q3                  \n"
          "vld1.s32       {d0-d3}, [%[in]]!       \n"
          "vmul.f32       q8, q4, %q[scale]       \n"
          "vmul.f32       q9, q5, %q[scale]       \n"
          "vmul.f32       q10, q6, %q[scale]      \n"
          "vmul.f32       q11, q7, %q[scale]      \n"
          "vld1.s32       {d4-d7}, [%[in]]!       \n"
          "subs           %[loop], #1             \n"
          "vst1.f32       {d16-d19}, [%[out]]!    \n"
          "vst1.f32       {d20-d23}, [%[out]]!    \n"
          "bne            0b                      \n"
          : [loop] "+r"(loop), [in] "+r"(din_ptr), [out] "+r"(dout_ptr)
          : [scale] "w"(vscale)
          : "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7", "q8", "q9", "q10",
            "q11");
#endif  // __aarch64__
    }
    const int* din_r = din_c + 16 * cnt;
    float* dout_r = dout_c + 16 * cnt;
    for (int i = 0; i < remain; ++i) {
      dout_r[i] = in_scale * din_r[i];
    }
  }
}

void int32_to_int8(const int* din, signed char* dout, const float* scale,
                   int axis_size, int64_t outer_size, int64_t inner_size) {
  int cnt = inner_size / 16;
  int remain = inner_size & 15;
  int64_t loop_size = outer_size * axis_size;
#pragma omp parallel for
  for (int64_t n = 0; n < loop_size; ++n) {
    float in_scale = scale[n % axis_size];
    const int* din_c = din + n * inner_size;
    signed char* dout_c = dout + n * inner_size;
    float32x4_t vscale = vdupq_n_f32(in_scale);
    float32x4_t vzero = vdupq_n_f32(0.f);
    float32x4_t vpoff = vdupq_n_f32(0.5f);
    float32x4_t vnoff = vdupq_n_f32(-0.5f);
    if (cnt > 0) {
      int loop = cnt;
      const int* din_ptr = din_c;
      signed char* dout_ptr = dout_c;
#ifdef __aarch64__
      asm volatile(
          "0:                                        \n"
          "ld1     {v0.4s, v1.4s}, [%[in]], #32      \n"
          "ld1     {v2.4s, v3.4s}, [%[in]], #32      \n"

          "scvtf   v4.4s, v0.4s                      \n"
          "scvtf   v5.4s, v1.4s                      \n"
          "scvtf   v6.4s, v2.4s                      \n"
          "scvtf   v7.4s, v3.4s                      \n"

          "fmul    v0.4s, v4.4s, %[scale].4s         \n"
          "fmul    v1.4s, v5.4s, %[scale].4s         \n"
          "fmul    v2.4s, v6.4s, %[scale].4s         \n"
          "fmul    v3.4s, v7.4s, %[scale].4s         \n"

          "fcvtas  v4.4s, v0.4s                      \n"
          "fcvtas  v5.4s, v1.4s                      \n"
          "fcvtas  v6.4s, v2.4s                      \n"
          "fcvtas  v7.4s, v3.4s                      \n"

          "sqxtn   v0.4h, v4.4s                      \n"
          "sqxtn2  v0.8h, v5.4s                      \n"
          "sqxtn   v1.4h, v6.4s                      \n"
          "sqxtn2  v1.8h, v7.4s                      \n"

          "sqxtn   v2.8b, v0.8h                      \n"
          "sqxtn2  v2.16b, v1.8h                     \n"

          "st1     {v2.16b}, [%[out]], #16           \n"
          "subs    %[loop], %[loop], #1              \n"
          "bne     0b                                \n"
          : [loop] "+r"(loop), [in] "+r"(din_ptr), [out] "+r"(dout_ptr)
          : [scale] "w"(vscale)
          : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7");
#else
      asm volatile(
          "vld1.32 {d0-d3},    [%[din]]!                  @ load in0~in7\n"
          "vld1.32    {d4-d7},    [%[din]]!       @ load in8~in16\n"
          "0:                                     @ main loop\n"
          "vcvt.f32.s32   q4, q0                  @ cvt to float\n"
          "vcvt.f32.s32   q5, q1                  @ cvt to float\n"
          "vcvt.f32.s32   q6, q2                  @ cvt to float\n"
          "vcvt.f32.s32   q7, q3                  @ cvt to float\n"
          "vand.i32   q0, %q[vpoff], %q[vpoff]    @ set offset, 0.5\n"
          "vand.i32   q1, q0, q0                  @ set offset, 0.5\n"
          "vand.i32   q2, q0, q0                  @ set offset, 0.5\n"
          "vand.i32   q3, q0, q0                  @ set offset, 0.5\n"
          "vcgt.f32   q8, q4, %q[vzero]           @ get mask > 0, in0\n"
          "vcgt.f32   q9, q5, %q[vzero]           @ get mask > 0, in1\n"
          "vcgt.f32   q10, q6, %q[vzero]          @ get mask > 0, in2\n"
          "vcgt.f32   q11, q7, %q[vzero]          @ get mask > 0, in3\n"
          "vbif.f32   q0, %q[vnoff], q8           @ get right offset\n"
          "vbif.f32   q1, %q[vnoff], q9           @ get right offset\n"
          "vbif.f32   q2, %q[vnoff], q10          @ get right offset\n"
          "vbif.f32   q3, %q[vnoff], q11          @ get right offset\n"
          "vmla.f32   q0, q4, %q[vscale]          @ mul scale\n"
          "vmla.f32   q1, q5, %q[vscale]          @ mul scale\n"
          "vmla.f32   q2, q6, %q[vscale]          @ mul scale\n"
          "vmla.f32   q3, q7, %q[vscale]          @ mul scale\n"
          "vcvt.s32.f32  q4, q0                   @ cvt to int32\n"
          "vcvt.s32.f32  q5, q1                   @ cvt to int32\n"
          "vcvt.s32.f32  q6, q2                   @ cvt to int32\n"
          "vcvt.s32.f32  q7, q3                   @ cvt to int32\n"
          "vqmovn.s32 d16, q4                     @ cnt to int16\n"
          "vqmovn.s32 d17, q5                     @ cnt to int16\n"
          "vqmovn.s32 d18, q6                     @ cnt to int16\n"
          "vqmovn.s32 d19, q7                     @ cnt to int16\n"
          "vld1.32 {d0-d3},    [%[din]]!          @ load in0~in7\n"
          "vqmovn.s16 d8, q8                      @ cnt to int8\n"
          "vqmovn.s16 d9, q9                      @ cnt to int8\n"
          "vld1.32 {d4-d7},    [%[din]]!          @ load in8~in16\n"
          "vst1.32 {d8-d9},    [%[dout]]!         @ write to output\n"
          "subs   %[loop], #1                     @ loop count -1\n"
          "bne    0b                              @ to main loop\n"
          : [loop] "+r"(loop), [din] "+r"(din_ptr), [dout] "+r"(dout_ptr)
          : [vscale] "w"(vscale), [vzero] "w"(vzero), [vnoff] "w"(vnoff),
            [vpoff] "w"(vpoff)
          : "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7", "q8", "q9", "q10",
            "q11");
#endif  // __aarch64__
    }
    const int* din_r = din_c + 16 * cnt;
    int8_t* dout_r = dout_c + 16 * cnt;
    for (int i = 0; i < remain; ++i) {
      dout_r[i] = saturate_cast<int8_t>(roundf(in_scale * din_r[i]));
    }
  }
}

/******************************************/
/********    kernel implement     *********/
/******************************************/
float compute_max_kernel(const float* din, int64_t size) {
  float max_value = 0.f;
  int cnt = size / 16;
  int remain = size & 15;
  float32x4_t vmax_val = vdupq_n_f32(0.f);
  const float* ptr_in = din;
  if (cnt > 0) {
    int loop_cnt = cnt;
#ifdef __aarch64__
    asm volatile(
        "ld1 {v0.4s, v1.4s}, [%[in]], #32               \n"
        "ld1 {v2.4s, v3.4s}, [%[in]], #32               \n"
        "0:                                             \n"
        "fabs v4.4s, v0.4s                              \n"
        "fabs v5.4s, v1.4s                              \n"
        "fabs v6.4s, v2.4s                              \n"
        "fabs v7.4s, v3.4s                              \n"
        "ld1 {v0.4s, v1.4s}, [%[in]], #32               \n"
        "fmax v2.4s, v4.4s, v5.4s                       \n"
        "fmax v3.4s, v6.4s, v7.4s                       \n"
        "fmax v4.4s, v2.4s, v3.4s                       \n"
        "ld1 {v2.4s, v3.4s}, [%[in]], #32               \n"
        "fmax %[max_val].4s, v4.4s, %[max_val].4s       \n"
        "subs %[cnt], %[cnt], #1                        \n"
        "bne    0b                                 \n"
        : [in] "+r"(ptr_in), [cnt] "+r"(loop_cnt), [max_val] "+w"(vmax_val)
        :
        : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7");
#else
    asm volatile(
        "vld1.32   {d0-d3}, [%[in]]!                        @ load 8 float\n"
        "vld1.32   {d4-d7}, [%[in]]!                @ load 8 float\n"
        "0:                                         @ main loop\n"
        "vabs.f32 q4, q0                            @ abs \n"
        "vabs.f32 q5, q1                            @ abs \n"
        "vabs.f32 q6, q2                            @ abs \n"
        "vabs.f32 q7, q3                            @ abs \n"
        "vld1.32   {d0-d3}, [%[in]]!                @ load 8 float\n"
        "vmax.f32 q2, q4, q5                        @ max \n"
        "vmax.f32 q3, q6, q7                        @ max \n"
        "vmax.f32 q4, q2, q3                        @ max \n"
        "vld1.32   {d4-d7}, [%[in]]!                @ load 8 float\n"
        "vmax.f32 %q[max_val], q4, %q[max_val]      @ max \n"
        "subs %[cnt], #1                            @ loop count -1\n"
        "bne    0b                                  @ jump to main loop\n"

        : [in] "+r"(ptr_in), [cnt] "+r"(loop_cnt), [max_val] "+w"(vmax_val)
        :
        : "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7");
#endif
    float32x2_t vmax_p =
        vpmax_f32(vget_high_f32(vmax_val), vget_low_f32(vmax_val));
    float max0 = vget_lane_f32(vmax_p, 0);
    float max1 = vget_lane_f32(vmax_p, 1);
    float max2 = max0 > max1 ? max0 : max1;
    max_value = max_value > max2 ? max_value : max2;
  }
  ptr_in = din + 16 * cnt;
  for (int i = 0; i < remain; ++i) {
    float data = fabsf(*(ptr_in++));
    max_value = fmaxf(max_value, data);
  }
  return max_value;
}

std::vector<float> get_tensor_scale_n(const float* in_data, int axis_size,
                                      int64_t inner_size, float scale_factor) {
  std::vector<float> scale_out(axis_size);
#pragma omp parallel for
  for (int c = 0; c < axis_size; ++c) {              // num
    const float* ptr_in = in_data + c * inner_size;  // channel*width*height
    scale_out[c] = compute_max_kernel(ptr_in, inner_size) / scale_factor;
  }
  return scale_out;
}

std::vector<float> get_tensor_scale_chw(const float* in_data, int axis_size,
                                        int64_t outer_size, int64_t inner_size,
                                        float scale_factor) {
  std::vector<float> scale_out(axis_size);
  int64_t inner_size_with_axis = axis_size * inner_size;
#pragma omp parallel for
  for (int c = 0; c < axis_size; ++c) {
    const float* din = in_data + c * inner_size;
    float max_val = 0.f;
    for (int j = 0; j < outer_size; ++j) {
      const float* ptr_in = din + j * inner_size_with_axis;
      max_val = fmaxf(compute_max_kernel(ptr_in, inner_size), max_val);
    }
    scale_out[c] = max_val / scale_factor;
  }
  return scale_out;
}

void int32_to_int32(const int* din, int* dout, const float* scale,
                    int axis_size, int64_t outer_size, int64_t inner_size) {
  int size_all = outer_size * axis_size * inner_size;
  memmove(dout, din, size_all * sizeof(int));
}

template <>
void int32_to_dtype(const int* din, float* dout, const float* scale,
                    int axis_size, int64_t outer_size, int64_t inner_size) {
  return int32_to_fp32(din, dout, scale, axis_size, outer_size, inner_size);
}

template <>
void int32_to_dtype(const int* din, signed char* dout, const float* scale,
                    int axis_size, int64_t outer_size, int64_t inner_size) {
  return int32_to_int8(din, dout, scale, axis_size, outer_size, inner_size);
}

template <>
void int32_to_dtype(const int* din, int* dout, const float* scale,
                    int axis_size, int64_t outer_size, int64_t inner_size) {
  return int32_to_int32(din, dout, scale, axis_size, outer_size, inner_size);
}

bool trans_tensor_int32_to_int8(Tensor* tin, Tensor* tout, float input_scale,
                                float output_scale,
                                std::vector<float> weights_scale, int axis) {
  tout->Resize(tin->dims());

  // compute scale
  std::vector<float> scale(weights_scale.size());
  for (int i = 0; i < weights_scale.size(); ++i) {
    scale[i] = input_scale * weights_scale[i] / output_scale;
  }

  auto i_dims = tin->dims();
  int outer_size = i_dims.count(0, axis);
  int axis_size = i_dims[axis];
  int inner_size = i_dims.count(axis + 1, i_dims.size());

  const int* i_data = tin->data<int32_t>();
  signed char* o_data = tout->mutable_data<signed char>();
  int32_to_int8(i_data, o_data, scale.data(), axis_size, outer_size,
                inner_size);
  return true;
}

template <>
bool get_tensor_scale<PRECISION(kFloat)>(const Tensor& tin,
                                         std::vector<float>* scale_out,
                                         int axis, float scale_factor) {
  int axis_size = 1;
  if (axis >= 0 && axis < tin.dims().size()) {
    axis_size = tin.dims()[axis];
  }
  int outer_size = 1;
  if (axis >= 0) {
    outer_size = tin.dims().count(0, axis);
  }
  int64_t inner_size = tin.dims().count(axis + 1, tin.dims().size());

  const float* in_data = static_cast<const float*>(tin.data<float>());
  if (axis <= 0) {
    *scale_out =
        get_tensor_scale_n(in_data, axis_size, inner_size, scale_factor);
  } else {
    *scale_out = get_tensor_scale_chw(in_data, axis_size, outer_size,
                                      inner_size, scale_factor);
  }
  return true;
}

bool trans_tensor_int32_to_fp32(Tensor* tin, Tensor* tout, float input_scale,
                                std::vector<float> weights_scale, int axis) {
  tout->Resize(tin->dims());

  // compute scale
  std::vector<float> scale(weights_scale.size());
  for (int i = 0; i < weights_scale.size(); ++i) {
    scale[i] = input_scale * weights_scale[i];
  }

  auto i_dims = tin->dims();
  int outer_size = i_dims.count(0, axis);
  int axis_size = i_dims[axis];
  int inner_size = i_dims.count(axis + 1, i_dims.size());

  const auto* i_data = tin->data<int32_t>();
  float* o_data = tout->mutable_data<float>();
  //! convert to fp32
  int32_to_fp32(i_data, o_data, scale.data(), axis_size, outer_size,
                inner_size);
  return true;
}

bool trans_tensor_fp32_to_int8(Tensor* tin, Tensor* tout, float input_scale) {
  tout->Resize(tin->dims());

  // compute scale
  std::vector<float> scale({input_scale});
  int inner_size = tin->dims().product();

  const auto* i_data = tin->data<float>();
  int8_t* o_data = tout->mutable_data<int8_t>();
  fp32_to_int8(i_data, o_data, scale.data(), 1, 1, inner_size);
  return true;
}

bool trans_fp32_bias_to_int32_basic(Tensor* tin, Tensor* tout, float in_scale,
                                    std::vector<float> vector_weight_scale) {
  tout->Resize(tin->dims());

  const float* i_data = tin->data<float>();
  int* o_data = tout->mutable_data<int>();
  for (int i = 0; i < tin->dims().product(); ++i) {
    o_data[i] =
        static_cast<int>(roundf(i_data[i] / in_scale / vector_weight_scale[i]));
  }
  return true;
}

template <>
bool trans_tensor_dtype<PRECISION(kInt32), PRECISION(kInt8)>(
    Tensor* tin, Tensor* tout, float input_scale, float output_scale,
    std::vector<float> weights_scale) {
  return trans_tensor_int32_to_int8(tin, tout, input_scale, output_scale,
                                    weights_scale, 1);
}

template <>
bool trans_tensor_dtype<PRECISION(kInt32), PRECISION(kFloat)>(
    Tensor* tin, Tensor* tout, float input_scale, float output_scale,
    std::vector<float> weights_scale) {
  return trans_tensor_int32_to_fp32(tin, tout, input_scale, weights_scale, 1);
}

}  // namespace math
}  // namespace arm
}  // namespace lite
}  // namespace paddle
