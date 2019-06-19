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

#include <arm_neon.h>
#include "paddle/fluid/lite/arm/math/conv_block_utils.h"
#include "paddle/fluid/lite/arm/math/conv_impl.h"
#include "paddle/fluid/lite/core/context.h"
#include "paddle/fluid/lite/operators/op_params.h"
#ifdef ARM_WITH_OMP
#include <omp.h>
#endif

namespace paddle {
namespace lite {
namespace arm {
namespace math {

void conv_3x3s1_direct_fp32(const float* i_data, float* o_data, int bs, int oc,
                            int oh, int ow, int ic, int ih, int win,
                            const float* weights, const float* bias,
                            const operators::ConvParam& param,
                            ARMContext* ctx) {
  const int threads = ctx->threads();
  int l2_size = ctx->l2_cache_size() / sizeof(float);

  const int pad_h = param.paddings[0];
  const int pad_w = param.paddings[1];
  const int hout_c_block = 4;
  const int hout_r_kernel = 2;
  const int wout_block = 4;
  const int wout_round = ((ow + wout_block - 1) / wout_block) * wout_block;
  const int win_round = wout_round + 2;
  bool flag_relu = param.fuse_relu;
  bool flag_bias = param.bias != nullptr;
  // if (param.activation_param.has_active) {
  //   if (param.activation_param.active == Active_relu &&
  //       fabs(param.activation_param.negative_slope) < 1e-6f) {
  //     flag_relu = true;
  //   }
  // }
  int hout_r_block = (l2_size - 2 * win_round * ic) /
                     (win_round * ic + hout_c_block * wout_round * threads);
  hout_r_block = hout_r_block > oh ? oh : hout_r_block;
  hout_r_block = (hout_r_block / hout_r_kernel) * hout_r_kernel;
  hout_r_block = hout_r_block < hout_r_kernel ? hout_r_kernel : hout_r_block;

  const int hin_r_block = hout_r_block + 2;

  float* tmp_work_space = ctx->workspace_data<float>();
  float ptr_zero[win_round];  // NOLINT
  memset(ptr_zero, 0, sizeof(float) * win_round);
  float ptr_write[wout_round];  // NOLINT

  int in_len = win_round * ic;
  int pre_in_size = hin_r_block * in_len;
  int pre_out_size = hout_c_block * hout_r_block * wout_round;

  float* pre_din = tmp_work_space;

  int size_in_channel = win * ih;
  int size_out_channel = ow * oh;
  int w_stride = ic * 9;                 // kernel_w * kernel_h;
  int w_stride_chin = hout_c_block * 9;  // kernel_w * kernel_h *

  int ws = -pad_w;
  int we = ws + win_round;
  int w_loop = wout_round / 4;

  int c_remain = oc - (oc / hout_c_block) * hout_c_block;
  int c_round_down = (oc / hout_c_block) * hout_c_block;

  int out_row_stride = hout_c_block * wout_round;
  for (int n = 0; n < bs; ++n) {
    const float* din_batch = i_data + n * ic * size_in_channel;
    float* dout_batch = o_data + n * oc * size_out_channel;
    for (int h = 0; h < oh; h += hout_r_block) {
      int h_kernel = hout_r_block;
      if (h + hout_r_block > oh) {
        h_kernel = oh - h;
      }
      int hs = h - pad_h;
      int he = hs + h_kernel + 2;
      prepack_input_nxw(din_batch, pre_din, 0, ic, hs, he, ws, we, ic, win, ih,
                        ptr_zero);
#pragma omp parallel for num_threads(threads)
      for (int c = 0; c < oc - (hout_c_block - 1); c += hout_c_block) {
#ifdef ARM_WITH_OMP
        float* pre_out =
            pre_din + pre_in_size + omp_get_thread_num() * pre_out_size;
#else
        float* pre_out = pre_din + pre_in_size;
#endif
        const float* block_inr0 = pre_din;
        const float* block_inr1 = block_inr0 + in_len;
        const float* block_inr2 = block_inr1 + in_len;
        const float* block_inr3 = block_inr2 + in_len;

        const float* weight_c = weights + c * w_stride;
        const float* bias_ptr = ptr_zero;
        if (flag_bias) {
          bias_ptr = bias + c;
        }
        fill_packed_biasc4(pre_out, bias_ptr,
                           wout_round * hout_c_block * h_kernel);

        for (int hk = 0; hk < h_kernel; hk += hout_r_kernel) {
          const float* wc0 = weight_c;

          const float* inr0 = block_inr0;
          const float* inr1 = block_inr1;
          const float* inr2 = block_inr2;
          const float* inr3 = block_inr3;

          float* pre_out0 = pre_out + hk * out_row_stride;
          float* pre_out1 = pre_out0 + out_row_stride;
#ifdef __aarch64__
          for (int i = 0; i < ic; ++i) {
            float* ptr_out0 = pre_out0;
            float* ptr_out1 = pre_out1;

            float32x4_t w0 = vld1q_f32(wc0);       // w0, v23
            float32x4_t w1 = vld1q_f32(wc0 + 4);   // w1, v24
            float32x4_t w2 = vld1q_f32(wc0 + 8);   // w2, v25
            float32x4_t w3 = vld1q_f32(wc0 + 12);  // w3, v26
            float32x4_t w4 = vld1q_f32(wc0 + 16);  // w4, v27
            float32x4_t w5 = vld1q_f32(wc0 + 20);  // w5, v28
            float32x4_t w6 = vld1q_f32(wc0 + 24);  // w6, v29
            float32x4_t w7 = vld1q_f32(wc0 + 28);  // w7, v30
            float32x4_t w8 = vld1q_f32(wc0 + 32);  // w8, v31

            const float* r0 = inr0;
            const float* r1 = inr1;
            const float* r2 = inr2;
            const float* r3 = inr3;

            int cnt = w_loop;
            asm volatile(
                "ldp    q15, q16, [%[ptr_out0]]             \n" /* load outr00,
                                                                   outr01*/
                "ldp    q17, q18, [%[ptr_out0], #32]\n" /* load outr02, outr03*/
                "ldp    q19, q20, [%[ptr_out1]]     \n" /* load outr10, outr11*/
                "ldp    q21, q22, [%[ptr_out1], #32]\n" /* load outr10, outr11*/
                "ldp    q0, q1,   [%[r0]], #16      \n" /* load input r0*/
                "ldp    q2, q3,   [%[r1]], #16      \n" /* load input r1*/
                "2:                                 \n" /* main loop*/
                /*  r0, r1, mul w0, get out r0, r1 */
                "fmla   v15.4s ,  %[w0].4s,  v0.s[0]\n" /* outr00 = w0 * r0[0]*/
                "fmla   v16.4s ,  %[w0].4s,  v0.s[1]\n" /* outr01 = w0 * r0[1]*/
                "fmla   v17.4s ,  %[w0].4s,  v0.s[2]\n" /* outr02 = w0 * r0[2]*/
                "fmla   v18.4s ,  %[w0].4s,  v0.s[3]\n" /* outr03 = w0 * r0[3]*/
                "fmla   v19.4s ,  %[w0].4s,  v2.s[0]\n" /* outr10 = w0 * r1[0]*/
                "fmla   v20.4s ,  %[w0].4s,  v2.s[1]\n" /* outr11 = w0 * r1[1]*/
                "fmla   v21.4s ,  %[w0].4s,  v2.s[2]\n" /* outr12 = w0 * r1[2]*/
                "fmla   v22.4s ,  %[w0].4s,  v2.s[3]\n" /* outr13 = w0 * r1[3]*/

                /*  r0, r1, mul w1, get out r0, r1 */
                "fmla   v15.4s ,  %[w1].4s,  v0.s[1]\n" /* outr00 = w1 * r0[1]*/
                "fmla   v16.4s ,  %[w1].4s,  v0.s[2]\n" /* outr01 = w1 * r0[2]*/
                "fmla   v17.4s ,  %[w1].4s,  v0.s[3]\n" /* outr02 = w1 * r0[3]*/
                "fmla   v18.4s ,  %[w1].4s,  v1.s[0]\n" /* outr03 = w1 * r0[4]*/
                "fmla   v19.4s ,  %[w1].4s,  v2.s[1]\n" /* outr10 = w1 * r1[1]*/
                "fmla   v20.4s ,  %[w1].4s,  v2.s[2]\n" /* outr11 = w1 * r1[2]*/
                "fmla   v21.4s ,  %[w1].4s,  v2.s[3]\n" /* outr12 = w1 * r1[3]*/
                "fmla   v22.4s ,  %[w1].4s,  v3.s[0]\n" /* outr13 = w1 * r1[4]*/

                "ldp    q4, q5,   [%[r2]], #16      \n" /* load input r2*/

                /*  r0, r1, mul w2, get out r0, r1 */
                "fmla   v15.4s ,  %[w2].4s,  v0.s[2]\n" /* outr00 = w2 * r0[2]*/
                "fmla   v16.4s ,  %[w2].4s,  v0.s[3]\n" /* outr01 = w2 * r0[3]*/
                "fmla   v17.4s ,  %[w2].4s,  v1.s[0]\n" /* outr02 = w2 * r0[0]*/
                "fmla   v18.4s ,  %[w2].4s,  v1.s[1]\n" /* outr03 = w2 * r0[1]*/
                "fmla   v19.4s ,  %[w2].4s,  v2.s[2]\n" /* outr10 = w2 * r1[2]*/
                "fmla   v20.4s ,  %[w2].4s,  v2.s[3]\n" /* outr11 = w2 * r1[3]*/
                "fmla   v21.4s ,  %[w2].4s,  v3.s[0]\n" /* outr12 = w2 * r1[0]*/
                "fmla   v22.4s ,  %[w2].4s,  v3.s[1]\n" /* outr13 = w2 * r1[1]*/

                /*  r1, r2, mul w3, get out r0, r1 */
                "fmla   v15.4s ,  %[w3].4s,  v2.s[0]\n" /* outr00 = w3 * r1[0]*/
                "fmla   v16.4s ,  %[w3].4s,  v2.s[1]\n" /* outr01 = w3 * r1[1]*/
                "fmla   v17.4s ,  %[w3].4s,  v2.s[2]\n" /* outr02 = w3 * r1[2]*/
                "fmla   v18.4s ,  %[w3].4s,  v2.s[3]\n" /* outr03 = w3 * r1[3]*/
                "fmla   v19.4s ,  %[w3].4s,  v4.s[0]\n" /* outr10 = w3 * r2[0]*/
                "fmla   v20.4s ,  %[w3].4s,  v4.s[1]\n" /* outr11 = w3 * r2[1]*/
                "fmla   v21.4s ,  %[w3].4s,  v4.s[2]\n" /* outr12 = w3 * r2[2]*/
                "fmla   v22.4s ,  %[w3].4s,  v4.s[3]\n" /* outr13 = w3 * r2[3]*/

                "ldp    q0, q1,   [%[r0]], #16      \n" /* load next input r0*/

                /*  r1, r2, mul w4, get out r0, r1 */
                "fmla   v15.4s ,  %[w4].4s,  v2.s[1]\n" /* outr00 = w4 * r1[1]*/
                "fmla   v16.4s ,  %[w4].4s,  v2.s[2]\n" /* outr01 = w4 * r1[2]*/
                "fmla   v17.4s ,  %[w4].4s,  v2.s[3]\n" /* outr02 = w4 * r1[3]*/
                "fmla   v18.4s ,  %[w4].4s,  v3.s[0]\n" /* outr03 = w4 * r1[4]*/
                "fmla   v19.4s ,  %[w4].4s,  v4.s[1]\n" /* outr10 = w4 * r2[1]*/
                "fmla   v20.4s ,  %[w4].4s,  v4.s[2]\n" /* outr11 = w4 * r2[2]*/
                "fmla   v21.4s ,  %[w4].4s,  v4.s[3]\n" /* outr12 = w4 * r2[3]*/
                "fmla   v22.4s ,  %[w4].4s,  v5.s[0]\n" /* outr13 = w4 * r2[4]*/

                "ldp    q6, q7,   [%[r3]], #16      \n" /* load input r3*/

                /*  r1, r2, mul w5, get out r0, r1 */
                "fmla   v15.4s ,  %[w5].4s,  v2.s[2]\n" /* outr00 = w5 * r1[2]*/
                "fmla   v16.4s ,  %[w5].4s,  v2.s[3]\n" /* outr01 = w5 * r1[3]*/
                "fmla   v17.4s ,  %[w5].4s,  v3.s[0]\n" /* outr02 = w5 * r1[0]*/
                "fmla   v18.4s ,  %[w5].4s,  v3.s[1]\n" /* outr03 = w5 * r1[1]*/
                "fmla   v19.4s ,  %[w5].4s,  v4.s[2]\n" /* outr10 = w5 * r2[2]*/
                "fmla   v20.4s ,  %[w5].4s,  v4.s[3]\n" /* outr11 = w5 * r2[3]*/
                "fmla   v21.4s ,  %[w5].4s,  v5.s[0]\n" /* outr12 = w5 * r2[0]*/
                "fmla   v22.4s ,  %[w5].4s,  v5.s[1]\n" /* outr13 = w5 * r2[1]*/

                /*  r2, r3, mul w6, get out r0, r1 */
                "fmla   v15.4s ,  %[w6].4s,  v4.s[0]\n" /* outr00 = w6 * r2[0]*/
                "fmla   v16.4s ,  %[w6].4s,  v4.s[1]\n" /* outr01 = w6 * r2[1]*/
                "fmla   v17.4s ,  %[w6].4s,  v4.s[2]\n" /* outr02 = w6 * r2[2]*/
                "fmla   v18.4s ,  %[w6].4s,  v4.s[3]\n" /* outr03 = w6 * r2[3]*/
                "fmla   v19.4s ,  %[w6].4s,  v6.s[0]\n" /* outr10 = w6 * r3[0]*/
                "fmla   v20.4s ,  %[w6].4s,  v6.s[1]\n" /* outr11 = w6 * r3[1]*/
                "fmla   v21.4s ,  %[w6].4s,  v6.s[2]\n" /* outr12 = w6 * r3[2]*/
                "fmla   v22.4s ,  %[w6].4s,  v6.s[3]\n" /* outr13 = w6 * r3[3]*/

                "ldp    q2, q3,   [%[r1]], #16      \n" /* load next input r1*/

                /*  r2, r3, mul w7, get out r0, r1 */
                "fmla   v15.4s ,  %[w7].4s,  v4.s[1]\n" /* outr00 = w7 * r2[1]*/
                "fmla   v16.4s ,  %[w7].4s,  v4.s[2]\n" /* outr01 = w7 * r2[2]*/
                "fmla   v17.4s ,  %[w7].4s,  v4.s[3]\n" /* outr02 = w7 * r2[3]*/
                "fmla   v18.4s ,  %[w7].4s,  v5.s[0]\n" /* outr03 = w7 * r2[4]*/
                "fmla   v19.4s ,  %[w7].4s,  v6.s[1]\n" /* outr10 = w7 * r3[1]*/
                "fmla   v20.4s ,  %[w7].4s,  v6.s[2]\n" /* outr11 = w7 * r3[2]*/
                "fmla   v21.4s ,  %[w7].4s,  v6.s[3]\n" /* outr12 = w7 * r3[3]*/
                "fmla   v22.4s ,  %[w7].4s,  v7.s[0]\n" /* outr13 = w7 * r3[4]*/

                "subs   %w[cnt], %w[cnt], #1        \n" /*loop count -1*/

                /*  r2, r3, mul w8, get out r0, r1 */
                "fmla   v15.4s ,  %[w8].4s,  v4.s[2]\n" /* outr00 = w8 * r2[2]*/
                "fmla   v16.4s ,  %[w8].4s,  v4.s[3]\n" /* outr01 = w8 * r2[3]*/
                "fmla   v17.4s ,  %[w8].4s,  v5.s[0]\n" /* outr02 = w8 * r2[0]*/
                "fmla   v18.4s ,  %[w8].4s,  v5.s[1]\n" /* outr03 = w8 * r2[1]*/

                "stp    q15, q16, [%[ptr_out0]], #32\n" /* save outr00, outr01*/
                "fmla   v19.4s ,  %[w8].4s,  v6.s[2]\n" /* outr10 = w8 * r3[2]*/
                "stp    q17, q18, [%[ptr_out0]], #32\n" /* save outr02, outr03*/
                "fmla   v20.4s ,  %[w8].4s,  v6.s[3]\n" /* outr11 = w8 * r3[3]*/
                "ldp    q15, q16, [%[ptr_out0]]     \n" /* load outr00, outr01*/
                "fmla   v21.4s ,  %[w8].4s,  v7.s[0]\n" /* outr12 = w8 * r3[0]*/
                "ldp    q17, q18, [%[ptr_out0], #32]\n" /* load outr02, outr03*/
                "fmla   v22.4s ,  %[w8].4s,  v7.s[1]\n" /* outr13 = w8 * r3[1]*/
                "stp    q19, q20, [%[ptr_out1]], #32\n" /* save outr10, outr11*/
                "stp    q21, q22, [%[ptr_out1]], #32\n" /* save outr12, outr13*/
                "ldp    q19, q20, [%[ptr_out1]]     \n" /* load outr10, outr11*/
                "ldp    q21, q22, [%[ptr_out1], #32]\n" /* load outr12, outr13*/
                "bne    2b                          \n" /* jump to main loop*/

                : [cnt] "+r"(cnt), [r0] "+r"(r0), [r1] "+r"(r1), [r2] "+r"(r2),
                  [r3] "+r"(r3), [ptr_out0] "+r"(ptr_out0),
                  [ptr_out1] "+r"(ptr_out1)
                : [w0] "w"(w0), [w1] "w"(w1), [w2] "w"(w2), [w3] "w"(w3),
                  [w4] "w"(w4), [w5] "w"(w5), [w6] "w"(w6), [w7] "w"(w7),
                  [w8] "w"(w8)
                : "cc", "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6",
                  "v7", "v15", "v16", "v17", "v18", "v19", "v20", "v21", "v22");

            wc0 += 9 * hout_c_block;
            inr0 += win_round;
            inr1 += win_round;
            inr2 += win_round;
            inr3 += win_round;
          }
#else   // not __aarch64__
          for (int i = 0; i < ic; ++i) {
            const float* wc0 = weight_c + i * w_stride_chin;

            float* ptr_out0 = pre_out0;
            float* ptr_out1 = pre_out1;

            const float* r0 = inr0;
            const float* r1 = inr1;
            const float* r2 = inr2;
            const float* r3 = inr3;

            int cnt = w_loop;
            asm volatile(
                "vld1.32    {d16-d19}, [%[ptr_out0]]!                       @ "
                "load outr0, w0, w1, c0~c3\n"
                "vld1.32    {d20-d23}, [%[ptr_out0]]                @ load "
                "outr0, w2, w3, c0~c3\n"

                /* load weights */
                "vld1.32    {d10-d13}, [%[wc0]]!                    @ load w0, "
                "w1, to q5, q6\n"
                "vld1.32    {d14-d15}, [%[wc0]]!                    @ load w2, "
                "to q7\n"

                /* load r0, r1 */
                "vld1.32    {d0-d1}, [%[r0]]!                       @ load r0, "
                "4 float\n"
                "vld1.32    {d2}, [%[r0]]                           @ load r0, "
                "2 float\n"

                "sub    %[ptr_out0], %[ptr_out0], #32               @ ptr_out0 "
                "- 32, to start address\n"

                /* main loop */
                "0:                                                 @ main "
                "loop\n"
                /* mul r0 with w0, w1, w2, get out r0 */
                "vld1.32    {d24-d27}, [%[ptr_out1]]!               @ load "
                "outr1, w0, w1, c0~c3\n"
                "vmla.f32   q8, q5, d0[0]                           @ w0 * "
                "inr00\n"
                "vld1.32    {d28-d31}, [%[ptr_out1]]                @ load "
                "outr1, w2, w3, c0~c3\n"
                "vmla.f32   q9, q5, d0[1]                           @ w0 * "
                "inr01\n"
                "vmla.f32   q10, q5, d1[0]                          @ w0 * "
                "inr02\n"
                "vmla.f32   q11, q5, d1[1]                          @ w0 * "
                "inr03\n"
                "vld1.32    {d3-d4}, [%[r1]]!                       @ load r1, "
                "4 float\n"
                "vmla.f32   q8, q6, d0[1]                           @ w1 * "
                "inr01\n"
                "vmla.f32   q9, q6, d1[0]                           @ w1 * "
                "inr02\n"
                "vmla.f32   q10, q6, d1[1]                          @ w1 * "
                "inr03\n"
                "vmla.f32   q11, q6, d2[0]                          @ w1 * "
                "inr04\n"
                "vld1.32    {d5}, [%[r1]]                           @ load r0, "
                "2 float\n"
                "vmla.f32   q8, q7, d1[0]                           @ w2 * "
                "inr02\n"
                "vmla.f32   q9, q7, d1[1]                           @ w2 * "
                "inr03\n"
                "vmla.f32   q10, q7, d2[0]                          @ w2 * "
                "inr04\n"
                "vmla.f32   q11, q7, d2[1]                          @ w2 * "
                "inr05\n"

                "sub    %[ptr_out1], %[ptr_out1], #32               @ ptr_out1 "
                "- 32, to start address\n"

                /* mul r1 with w0, w1, w2, get out r1 */
                "vmla.f32   q12, q5, d3[0]                          @ w0 * "
                "inr10\n"
                "vmla.f32   q13, q5, d3[1]                          @ w0 * "
                "inr11\n"
                "vmla.f32   q14, q5, d4[0]                          @ w0 * "
                "inr12\n"
                "vmla.f32   q15, q5, d4[1]                          @ w0 * "
                "inr13\n"
                "vmla.f32   q12, q6, d3[1]                          @ w1 * "
                "inr11\n"
                "vmla.f32   q13, q6, d4[0]                          @ w1 * "
                "inr12\n"
                "vmla.f32   q14, q6, d4[1]                          @ w1 * "
                "inr13\n"
                "vmla.f32   q15, q6, d5[0]                          @ w1 * "
                "inr14\n"
                "vld1.32    {d10-d13}, [%[wc0]]!                    @ load w3, "
                "w4, to q5, q6\n"
                "vmla.f32   q12, q7, d4[0]                          @ w2 * "
                "inr12\n"
                "vmla.f32   q13, q7, d4[1]                          @ w2 * "
                "inr13\n"
                "vmla.f32   q14, q7, d5[0]                          @ w2 * "
                "inr14\n"
                "vmla.f32   q15, q7, d5[1]                          @ w2 * "
                "inr15\n"
                "vld1.32    {d14-d15}, [%[wc0]]!                    @ load w5, "
                "to q7\n"

                /* mul r1 with w3, w4, w5, get out r0 */
                "vmla.f32   q8, q5, d3[0]                           @ w3 * "
                "inr10\n"
                "vmla.f32   q9, q5, d3[1]                           @ w3 * "
                "inr11\n"
                "vmla.f32   q10, q5, d4[0]                          @ w3 * "
                "inr12\n"
                "vmla.f32   q11, q5, d4[1]                          @ w3 * "
                "inr13\n"
                "vld1.32    {d0-d1}, [%[r2]]!                       @ load r2, "
                "4 float\n"
                "vmla.f32   q8, q6, d3[1]                           @ w4 * "
                "inr11\n"
                "vmla.f32   q9, q6, d4[0]                           @ w4 * "
                "inr12\n"
                "vmla.f32   q10, q6, d4[1]                          @ w4 * "
                "inr13\n"
                "vmla.f32   q11, q6, d5[0]                          @ w4 * "
                "inr14\n"
                "vld1.32    {d2}, [%[r2]]                           @ load r2, "
                "2 float\n"
                "vmla.f32   q8, q7, d4[0]                           @ w5 * "
                "inr12\n"
                "vmla.f32   q9, q7, d4[1]                           @ w5 * "
                "inr13\n"
                "vmla.f32   q10, q7, d5[0]                          @ w5 * "
                "inr14\n"
                "vmla.f32   q11, q7, d5[1]                          @ w5 * "
                "inr15\n"

                /* mul r2 with w3, w4, w5, get out r1 */
                "vmla.f32   q12, q5, d0[0]                          @ w3 * "
                "inr20\n"
                "vmla.f32   q13, q5, d0[1]                          @ w3 * "
                "inr21\n"
                "vmla.f32   q14, q5, d1[0]                          @ w3 * "
                "inr22\n"
                "vmla.f32   q15, q5, d1[1]                          @ w3 * "
                "inr23\n"
                "vmla.f32   q12, q6, d0[1]                          @ w4 * "
                "inr21\n"
                "vmla.f32   q13, q6, d1[0]                          @ w4 * "
                "inr22\n"
                "vmla.f32   q14, q6, d1[1]                          @ w4 * "
                "inr23\n"
                "vmla.f32   q15, q6, d2[0]                          @ w4 * "
                "inr24\n"
                "vld1.32    {d10-d13}, [%[wc0]]!                    @ load w6, "
                "w7, to q5, q6\n"
                "vmla.f32   q12, q7, d1[0]                          @ w5 * "
                "inr22\n"
                "vmla.f32   q13, q7, d1[1]                          @ w5 * "
                "inr23\n"
                "vmla.f32   q14, q7, d2[0]                          @ w5 * "
                "inr24\n"
                "vmla.f32   q15, q7, d2[1]                          @ w5 * "
                "inr25\n"
                "vld1.32    {d14-d15}, [%[wc0]]!                    @ load w8, "
                "to q7\n"

                "sub    %[wc0], %[wc0], #144                        @ wc0 - "
                "144 to start address\n"

                /* mul r2 with w6, w7, w8, get out r0 */
                "vmla.f32   q8, q5, d0[0]                           @ w6 * "
                "inr20\n"
                "vmla.f32   q9, q5, d0[1]                           @ w6 * "
                "inr21\n"
                "vld1.32    {d3-d4}, [%[r3]]!                       @ load r3, "
                "4 float\n"
                "vmla.f32   q10, q5, d1[0]                          @ w6 * "
                "inr22\n"
                "vmla.f32   q11, q5, d1[1]                          @ w6 * "
                "inr23\n"
                "vmla.f32   q8, q6, d0[1]                           @ w7 * "
                "inr21\n"
                "vmla.f32   q9, q6, d1[0]                           @ w7 * "
                "inr22\n"
                "vld1.32    {d5}, [%[r3]]                           @ load r3, "
                "2 float\n"
                "vmla.f32   q10, q6, d1[1]                          @ w7 * "
                "inr23\n"
                "vmla.f32   q11, q6, d2[0]                          @ w7 * "
                "inr24\n"
                "vmla.f32   q8, q7, d1[0]                           @ w8 * "
                "inr22\n"
                "vmla.f32   q9, q7, d1[1]                           @ w8 * "
                "inr23\n"
                "vld1.32    {d0-d1}, [%[r0]]!                       @ load r0, "
                "4 float\n"
                "vmla.f32   q10, q7, d2[0]                          @ w8 * "
                "inr24\n"
                "vmla.f32   q11, q7, d2[1]                          @ w8 * "
                "inr25\n"
                "vld1.32    {d2}, [%[r0]]                           @ load r0, "
                "2 float\n"

                /* mul r3 with w6, w7, w8, get out r1 */
                "vmla.f32   q12, q5, d3[0]                          @ w6 * "
                "inr20\n"
                "vmla.f32   q13, q5, d3[1]                          @ w6 * "
                "inr21\n"
                "vst1.32    {d16-d19}, [%[ptr_out0]]!               @ save "
                "r00, r01, c0~c3\n"
                "vmla.f32   q14, q5, d4[0]                          @ w6 * "
                "inr22\n"
                "vmla.f32   q15, q5, d4[1]                          @ w6 * "
                "inr23\n"
                "vst1.32    {d20-d23}, [%[ptr_out0]]!               @ save "
                "r02, r03, c0~c3\n"
                "vmla.f32   q12, q6, d3[1]                          @ w7 * "
                "inr21\n"
                "vmla.f32   q13, q6, d4[0]                          @ w7 * "
                "inr22\n"
                "vld1.32    {d16-d19}, [%[ptr_out0]]!               @ load "
                "outr0, w0, w1, c0~c3\n"
                "vmla.f32   q14, q6, d4[1]                          @ w7 * "
                "inr23\n"
                "vmla.f32   q15, q6, d5[0]                          @ w7 * "
                "inr24\n"
                "vld1.32    {d10-d13}, [%[wc0]]!                    @ load w0, "
                "w1, to q5, q6\n"
                "vmla.f32   q12, q7, d4[0]                          @ w8 * "
                "inr22\n"
                "vmla.f32   q13, q7, d4[1]                          @ w8 * "
                "inr23\n"
                "vld1.32    {d20-d23}, [%[ptr_out0]]                @ load "
                "outr0, w2, w3, c0~c3\n"
                "vmla.f32   q14, q7, d5[0]                          @ w8 * "
                "inr24\n"
                "vmla.f32   q15, q7, d5[1]                          @ w8 * "
                "inr25\n"

                "vst1.32    {d24-d27}, [%[ptr_out1]]!               @ save "
                "r10, r11, c0~c3\n"
                "vst1.32    {d28-d31}, [%[ptr_out1]]!               @ save "
                "r12, r13, c0~c3\n"
                "vld1.32    {d14-d15}, [%[wc0]]!                    @ load w2, "
                "to q7\n"

                "sub    %[ptr_out0], %[ptr_out0], #32               @ ptr_out0 "
                "- 32, to start address\n"

                "subs   %[cnt], #1                                  @ loop "
                "count--\n"
                "bne    0b                                          @ jump to "
                "main loop\n"

                : [cnt] "+r"(cnt), [r0] "+r"(r0), [r1] "+r"(r1), [r2] "+r"(r2),
                  [r3] "+r"(r3), [ptr_out0] "+r"(ptr_out0),
                  [ptr_out1] "+r"(ptr_out1), [wc0] "+r"(wc0)
                :
                : "cc", "memory", "q0", "q1", "q2", "q3", "q4", "q5", "q6",
                  "q7", "q8", "q9", "q10", "q11", "q12", "q13", "q14", "q15");

            inr0 += win_round;
            inr1 += win_round;
            inr2 += win_round;
            inr3 += win_round;
          }
#endif  // __aarch64__
          block_inr0 = block_inr2;
          block_inr1 = block_inr3;
          block_inr2 = block_inr1 + in_len;
          block_inr3 = block_inr2 + in_len;
        }
        write_to_output_c4_fp32(pre_out, dout_batch, c, c + hout_c_block, h,
                                h + h_kernel, 0, wout_round, oc, oh, ow,
                                flag_relu, ptr_write);
      }
      const float* weight_remain_ptr = weights + c_round_down * w_stride;
#pragma omp parallel for num_threads(threads)
      for (int c = 0; c < c_remain; ++c) {
#ifdef ARM_WITH_OMP
        float* pre_out =
            pre_din + pre_in_size + omp_get_thread_num() * pre_out_size;
#else
        float* pre_out = pre_din + pre_in_size;
#endif

        int c_idx = c_round_down + c;

        int h_kernel = hout_r_block;
        if (h + hout_r_block > oh) {
          h_kernel = oh - h;
        }

        const float* block_inr0 = pre_din;
        const float* block_inr1 = block_inr0 + in_len;
        const float* block_inr2 = block_inr1 + in_len;
        const float* block_inr3 = block_inr2 + in_len;

        const float* bias_ptr = ptr_zero;
        if (flag_bias) {
          bias_ptr = bias + c_idx;
        }
        fill_bias(pre_out, bias_ptr, 1, wout_round * h_kernel);

        for (int hk = 0; hk < h_kernel; hk += hout_r_kernel) {
          const float* wc0 = weight_remain_ptr;

          const float* inr0 = block_inr0;
          const float* inr1 = block_inr1;
          const float* inr2 = block_inr2;
          const float* inr3 = block_inr3;

          float* pre_out0 = pre_out + hk * wout_round;
          float* pre_out1 = pre_out0 + wout_round;
#ifdef __aarch64__
          for (int i = 0; i < ic; ++i) {
            float* ptr_out0 = pre_out0;
            float* ptr_out1 = pre_out1;

            float32x4_t w0 = vdupq_n_f32(wc0[c]);       // w0, v23
            float32x4_t w1 = vdupq_n_f32(wc0[4 + c]);   // w1, v24
            float32x4_t w2 = vdupq_n_f32(wc0[8 + c]);   // w2, v25
            float32x4_t w3 = vdupq_n_f32(wc0[12 + c]);  // w3, v26
            float32x4_t w4 = vdupq_n_f32(wc0[16 + c]);  // w4, v27
            float32x4_t w5 = vdupq_n_f32(wc0[20 + c]);  // w5, v28
            float32x4_t w6 = vdupq_n_f32(wc0[24 + c]);  // w6, v29
            float32x4_t w7 = vdupq_n_f32(wc0[28 + c]);  // w7, v30
            float32x4_t w8 = vdupq_n_f32(wc0[32 + c]);  // w8, v31

            const float* r0 = inr0;
            const float* r1 = inr1;
            const float* r2 = inr2;
            const float* r3 = inr3;

            int cnt = w_loop;
            asm volatile(
                "ldr    q21, [%[ptr_out0]]                  \n" /* load outr0,
                                                                   w0~w3*/
                "ldr    q22, [%[ptr_out1]]          \n" /* load outr1, w0~w3*/
                "ldp    q0, q1,   [%[r0]], #16      \n" /* load input r0*/
                "ldp    q2, q3,   [%[r1]], #16      \n" /* load input r1*/
                "ldp    q4, q5,   [%[r2]], #16      \n" /* load input r2*/
                "ldp    q6, q7,   [%[r3]], #16      \n" /* load input r3*/
                "2:                                 \n" /* main loop*/

                "fmla   v21.4s ,  %[w0].4s,  v0.4s  \n" /* outr0 = w0 * r0*/
                "fmla   v22.4s ,  %[w0].4s,  v2.4s  \n" /* outr1 = w0 * r1*/

                "ext    v8.16b,  v0.16b,  v1.16b, #4   \n" /* shift r0 left 1*/
                "ext    v10.16b,  v2.16b,  v3.16b, #4  \n" /* shift r1 left 1*/
                "ext    v9.16b,  v0.16b,  v1.16b, #8   \n" /* shift r0 left 2*/
                "ext    v11.16b,  v2.16b,  v3.16b, #8  \n" /* shift r1 left 2*/

                "ldp    q0, q1,   [%[r0]], #16      \n" /* load input r0*/

                "fmla   v21.4s ,  %[w1].4s,  v8.4s  \n" /* outr0 = w1 * r1*/
                "fmla   v22.4s ,  %[w1].4s,  v10.4s \n" /* outr1 = w1 * r2*/

                "fmla   v21.4s ,  %[w2].4s,  v9.4s  \n" /* outr0 = w2 * r1*/
                "fmla   v22.4s ,  %[w2].4s,  v11.4s \n" /* outr1 = w2 * r2*/

                "fmla   v21.4s ,  %[w3].4s,  v2.4s  \n" /* outr0 = w3 * r1*/
                "fmla   v22.4s ,  %[w3].4s,  v4.4s  \n" /* outr1 = w3 * r2*/

                "ext    v12.16b,  v4.16b,  v5.16b, #4\n" /* shift r2 left 1*/
                "ext    v14.16b,  v6.16b,  v7.16b, #4\n" /* shift r3 left 1*/
                "ext    v13.16b,  v4.16b,  v5.16b, #8\n" /* shift r2 left 2*/
                "ext    v15.16b,  v6.16b,  v7.16b, #8\n" /* shift r3 left 2*/

                "fmla   v21.4s ,  %[w4].4s,  v10.4s \n" /* outr0 = w4 * r1*/
                "fmla   v22.4s ,  %[w4].4s,  v12.4s \n" /* outr1 = w4 * r2*/

                "fmla   v21.4s ,  %[w5].4s,  v11.4s \n" /* outr0 = w5 * r1*/
                "fmla   v22.4s ,  %[w5].4s,  v13.4s \n" /* outr1 = w5 * r2*/

                "ldp    q2, q3,   [%[r1]], #16      \n" /* load input r0*/

                "fmla   v21.4s ,  %[w6].4s,  v4.4s  \n" /* outr0 = w6 * r2*/
                "fmla   v22.4s ,  %[w6].4s,  v6.4s  \n" /* outr1 = w6 * r3*/

                "ldp    q4, q5,   [%[r2]], #16      \n" /* load input r2*/

                "fmla   v21.4s ,  %[w7].4s,  v12.4s \n" /* outr0 = w7 * r1*/
                "fmla   v22.4s ,  %[w7].4s,  v14.4s \n" /* outr1 = w7 * r2*/

                "ldp    q6, q7,   [%[r3]], #16      \n" /* load input r3*/

                "fmla   v21.4s ,  %[w8].4s,  v13.4s \n" /* outr0 = w8 * r1*/
                "fmla   v22.4s ,  %[w8].4s,  v15.4s \n" /* outr1 = w8 * r2*/

                "str    q21,    [%[ptr_out0]], #16  \n" /*write output r0*/
                "str    q22,    [%[ptr_out1]], #16  \n" /*write output r1*/

                "subs   %w[cnt], %w[cnt], #1        \n" /*loop count -1*/

                "ldr    q21, [%[ptr_out0]]          \n" /* load outr0, w0~w3*/
                "ldr    q22, [%[ptr_out1]]          \n" /* load outr1, w0~w3*/

                "bne    2b                          \n" /* jump to main loop*/

                : [cnt] "+r"(cnt), [r0] "+r"(r0), [r1] "+r"(r1), [r2] "+r"(r2),
                  [r3] "+r"(r3), [ptr_out0] "+r"(ptr_out0),
                  [ptr_out1] "+r"(ptr_out1)
                : [w0] "w"(w0), [w1] "w"(w1), [w2] "w"(w2), [w3] "w"(w3),
                  [w4] "w"(w4), [w5] "w"(w5), [w6] "w"(w6), [w7] "w"(w7),
                  [w8] "w"(w8)
                : "cc", "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6",
                  "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15",
                  "v21", "v22");

            wc0 += 9 * hout_c_block;
            inr0 += win_round;
            inr1 += win_round;
            inr2 += win_round;
            inr3 += win_round;
          }
#else   // not __aarch64__
          for (int i = 0; i < ic; ++i) {
            float* ptr_out0 = pre_out0;
            float* ptr_out1 = pre_out1;

            //! get valid weights of current output channel
            float w_tmp[10] = {
                wc0[c],      wc0[c + 4],  wc0[c + 8],  wc0[c + 12], wc0[c + 16],
                wc0[c + 20], wc0[c + 24], wc0[c + 28], wc0[c + 32], 0.f};
            float32x4_t w0 = vld1q_f32(w_tmp);      // w0, w1, w2, q0
            float32x4_t w1 = vld1q_f32(w_tmp + 3);  // w3, w4, w5, q1
            float32x4_t w2 = vld1q_f32(w_tmp + 6);  // w6, w7, w8, q2

            const float* r0 = inr0;
            const float* r1 = inr1;
            const float* r2 = inr2;
            const float* r3 = inr3;
            int cnt = w_loop / 2;
            if (cnt > 0) {
              asm volatile(
                  "vld1.32    {d24-d27},    [%[ptr_out0]]                 @ "
                  "load or00, or01\n"
                  "vld1.32    {d6-d9},    [%[r0]]!                @ load r0, 8 "
                  "float\n"
                  "vld1.32    {d10},       [%[r0]]                @ load r0, 2 "
                  "float\n"
                  /* main loop */
                  "0:                                             @ main loop\n"
                  /* r0 * w0, w1, w2, get out r0*/
                  "vld1.32    {d28-d31},    [%[ptr_out1]]         @ load or10, "
                  "or11\n"
                  "vext.32    q8, q3, q4, #1                      @ r0, shift "
                  "left 1, get 1, 2, 3, 4\n"
                  "vext.32    q9, q4, q5, #1                      @ r0, shift "
                  "left 1, get 5, 6, 7, 8\n"
                  "vmla.f32   q12,    q3, %e[w0][0]               @ w00 * r0, "
                  "0, 1, 2, 3\n"
                  "vmla.f32   q13,    q4, %e[w0][0]               @ w00 * r0, "
                  "4, 5, 6, 7\n"
                  "vext.32    q10, q3, q4, #2                     @ r0, shift "
                  "left 2, get 2, 3, 4, 5\n"
                  "vext.32    q11, q4, q5, #2                     @ r0, shift "
                  "left 2, get 6, 7, 8, 9\n"
                  "vmla.f32   q12,    q8, %e[w0][1]               @ w01 * r0, "
                  "1, 2, 3, 4\n"
                  "vmla.f32   q13,    q9, %e[w0][1]               @ w01 * r0, "
                  "5, 6, 7, 8\n"
                  "vld1.32    {d6-d9},    [%[r1]]!                @ load r1, 8 "
                  "float\n"
                  "vmla.f32   q12,    q10, %f[w0][0]              @ w02 * r0, "
                  "2, 3, 4, 5\n"
                  "vmla.f32   q13,    q11, %f[w0][0]              @ w02 * r0, "
                  "6, 7, 8, 9\n"
                  "vld1.32    {d10},       [%[r1]]                @ load r1, 2 "
                  "float\n"

                  /* r1 * w3, w4, w5, get out r0*/
                  /* r1 * w0, w1, w2, get out r1*/
                  "vmla.f32   q12,    q3, %e[w1][0]               @ w10 * r1, "
                  "0, 1, 2, 3\n"
                  "vmla.f32   q13,    q4, %e[w1][0]               @ w10 * r1, "
                  "4, 5, 6, 7\n"
                  "vext.32    q8, q3, q4, #1                      @ r1, shift "
                  "left 1, get 1, 2, 3, 4\n"
                  "vext.32    q9, q4, q5, #1                      @ r1, shift "
                  "left 1, get 5, 6, 7, 8\n"
                  "vmla.f32   q14,    q3, %e[w0][0]               @ w00 * r1, "
                  "0, 1, 2, 3\n"
                  "vmla.f32   q15,    q4, %e[w0][0]               @ w00 * r1, "
                  "4, 5, 6, 7\n"
                  "vext.32    q10, q3, q4, #2                     @ r1, shift "
                  "left 2, get 2, 3, 4, 5\n"
                  "vext.32    q11, q4, q5, #2                     @ r1, shift "
                  "left 2, get 6, 7, 8, 9\n"
                  "vmla.f32   q12,    q8, %e[w1][1]               @ w11 * r1, "
                  "1, 2, 3, 4\n"
                  "vmla.f32   q13,    q9, %e[w1][1]               @ w11 * r1, "
                  "5, 6, 7, 8\n"
                  "vmla.f32   q14,    q8, %e[w0][1]               @ w01 * r1, "
                  "1, 2, 3, 4\n"
                  "vmla.f32   q15,    q9, %e[w0][1]               @ w01 * r1, "
                  "5, 6, 7, 8\n"
                  "vld1.32    {d6-d9},    [%[r2]]!                @ load r2, 8 "
                  "float\n"
                  "vmla.f32   q12,    q10, %f[w1][0]              @ w12 * r1, "
                  "2, 3, 4, 5\n"
                  "vmla.f32   q13,    q11, %f[w1][0]              @ w12 * r1, "
                  "6, 7, 8, 9\n"
                  "vmla.f32   q14,    q10, %f[w0][0]              @ w02 * r1, "
                  "2, 3, 4, 5\n"
                  "vmla.f32   q15,    q11, %f[w0][0]              @ w02 * r1, "
                  "6, 7, 8, 9\n"
                  "vld1.32    {d10},    [%[r2]]                   @ load r2, 2 "
                  "float\n"

                  /* r2 * w6, w7, w8, get out r0*/
                  /* r2 * w3, w4, w5, get out r1*/
                  "vmla.f32   q12,    q3, %e[w2][0]               @ w20 * r2, "
                  "0, 1, 2, 3\n"
                  "vmla.f32   q13,    q4, %e[w2][0]               @ w20 * r2, "
                  "4, 5, 6, 7\n"
                  "vext.32    q8, q3, q4, #1                      @ r2, shift "
                  "left 1, get 1, 2, 3, 4\n"
                  "vext.32    q9, q4, q5, #1                      @ r2, shift "
                  "left 1, get 5, 6, 7, 8\n"
                  "vmla.f32   q14,    q3, %e[w1][0]               @ w10 * r2, "
                  "0, 1, 2, 3\n"
                  "vmla.f32   q15,    q4, %e[w1][0]               @ w10 * r2, "
                  "4, 5, 6, 7\n"
                  "vext.32    q10, q3, q4, #2                     @ r2, shift "
                  "left 2, get 2, 3, 4, 5\n"
                  "vext.32    q11, q4, q5, #2                     @ r2, shift "
                  "left 2, get 6, 7, 8, 9\n"
                  "vmla.f32   q12,    q8, %e[w2][1]               @ w21 * r2, "
                  "1, 2, 3, 4\n"
                  "vmla.f32   q13,    q9, %e[w2][1]               @ w21 * r2, "
                  "5, 6, 7, 8\n"
                  "vmla.f32   q14,    q8, %e[w1][1]               @ w11 * r2, "
                  "1, 2, 3, 4\n"
                  "vmla.f32   q15,    q9, %e[w1][1]                @ w11 * r2, "
                  "5, 6, 7, 8\n"
                  "vld1.32    {d6-d9},    [%[r3]]!                @ load r3, 8 "
                  "float\n"
                  "vmla.f32   q12,    q10, %f[w2][0]              @ w22 * r2, "
                  "2, 3, 4, 5\n"
                  "vmla.f32   q13,    q11, %f[w2][0]              @ w22 * r2, "
                  "6, 7, 8, 9\n"
                  "vmla.f32   q14,    q10, %f[w1][0]              @ w12 * r2, "
                  "2, 3, 4, 5\n"
                  "vmla.f32   q15,    q11, %f[w1][0]              @ w12 * r2, "
                  "6, 7, 8, 9\n"
                  "vld1.32    {d10},    [%[r3]]                   @ load r3, 2 "
                  "float\n"

                  /* r3 * w6, w7, w8, get out r1*/
                  "vext.32    q8, q3, q4, #1                      @ r3, shift "
                  "left 1, get 1, 2, 3, 4\n"
                  "vext.32    q9, q4, q5, #1                      @ r3, shift "
                  "left 1, get 5, 6, 7, 8\n"
                  "vmla.f32   q14,    q3, %e[w2][0]               @ w20 * r3, "
                  "0, 1, 2, 3\n"
                  "vmla.f32   q15,    q4, %e[w2][0]               @ w20 * r3, "
                  "4, 5, 6, 7\n"
                  "vst1.32    {d24-d27},  [%[ptr_out0]]!          @ save or00, "
                  "or01\n"
                  "vext.32    q10, q3, q4, #2                     @ r3, shift "
                  "left 2, get 2, 3, 4, 5\n"
                  "vext.32    q11, q4, q5, #2                     @ r3, shift "
                  "left 2, get 6, 7, 8, 9\n"
                  "vmla.f32   q14,    q8, %e[w2][1]               @ w21 * r3, "
                  "0, 1, 2, 3\n"
                  "vmla.f32   q15,    q9, %e[w2][1]               @ w21 * r3, "
                  "4, 5, 6, 7\n"
                  "vld1.32    {d24-d27},  [%[ptr_out0]]           @ load or00, "
                  "or01\n"
                  "vld1.32    {d6-d9},    [%[r0]]!                @ load r3, 8 "
                  "float\n"
                  "vmla.f32   q14,    q10, %f[w2][0]              @ w22 * r3, "
                  "2, 3, 4, 5\n"
                  "vmla.f32   q15,    q11, %f[w2][0]              @ w22 * r3, "
                  "6, 7, 8, 9\n"
                  "vld1.32    {d10},    [%[r0]]                   @ load r0, 2 "
                  "float\n"
                  "vst1.32    {d28-d31},  [%[ptr_out1]]!          @ save or10, "
                  "or11\n"

                  "subs   %[cnt], #1                              @loop count "
                  "-1\n"
                  "bne    0b                                      @ jump to "
                  "main loop\n"

                  : [cnt] "+r"(cnt), [r0] "+r"(r0), [r1] "+r"(r1),
                    [r2] "+r"(r2), [r3] "+r"(r3), [ptr_out0] "+r"(ptr_out0),
                    [ptr_out1] "+r"(ptr_out1)
                  : [w0] "w"(w0), [w1] "w"(w1), [w2] "w"(w2)
                  : "cc", "memory", "q3", "q4", "q5", "q6", "q7", "q8", "q9",
                    "q10", "q11", "q12", "q13", "q14", "q15");
              r0 -= 8;
            }
            //! deal with remain ow
            if (w_loop & 1) {
              ptr_out0[0] +=
                  r0[0] * w_tmp[0] + r0[1] * w_tmp[1] + r0[2] * w_tmp[2] +
                  r1[0] * w_tmp[3] + r1[1] * w_tmp[4] + r1[2] * w_tmp[5] +
                  r2[0] * w_tmp[6] + r2[1] * w_tmp[7] + r2[2] * w_tmp[8];

              ptr_out0[1] +=
                  r0[1] * w_tmp[0] + r0[2] * w_tmp[1] + r0[3] * w_tmp[2] +
                  r1[1] * w_tmp[3] + r1[2] * w_tmp[4] + r1[3] * w_tmp[5] +
                  r2[1] * w_tmp[6] + r2[2] * w_tmp[7] + r2[3] * w_tmp[8];

              ptr_out0[2] +=
                  r0[2] * w_tmp[0] + r0[3] * w_tmp[1] + r0[4] * w_tmp[2] +
                  r1[2] * w_tmp[3] + r1[3] * w_tmp[4] + r1[4] * w_tmp[5] +
                  r2[2] * w_tmp[6] + r2[3] * w_tmp[7] + r2[4] * w_tmp[8];

              ptr_out0[3] +=
                  r0[3] * w_tmp[0] + r0[4] * w_tmp[1] + r0[5] * w_tmp[2] +
                  r1[3] * w_tmp[3] + r1[4] * w_tmp[4] + r1[5] * w_tmp[5] +
                  r2[3] * w_tmp[6] + r2[4] * w_tmp[7] + r2[5] * w_tmp[8];

              ptr_out1[0] +=
                  r1[0] * w_tmp[0] + r1[1] * w_tmp[1] + r1[2] * w_tmp[2] +
                  r2[0] * w_tmp[3] + r2[1] * w_tmp[4] + r2[2] * w_tmp[5] +
                  r3[0] * w_tmp[6] + r3[1] * w_tmp[7] + r3[2] * w_tmp[8];

              ptr_out1[1] +=
                  r1[1] * w_tmp[0] + r1[2] * w_tmp[1] + r1[3] * w_tmp[2] +
                  r2[1] * w_tmp[3] + r2[2] * w_tmp[4] + r2[3] * w_tmp[5] +
                  r3[1] * w_tmp[6] + r3[2] * w_tmp[7] + r3[3] * w_tmp[8];

              ptr_out1[2] +=
                  r1[2] * w_tmp[0] + r1[3] * w_tmp[1] + r1[4] * w_tmp[2] +
                  r2[2] * w_tmp[3] + r2[3] * w_tmp[4] + r2[4] * w_tmp[5] +
                  r3[2] * w_tmp[6] + r3[3] * w_tmp[7] + r3[4] * w_tmp[8];

              ptr_out1[3] +=
                  r1[3] * w_tmp[0] + r1[4] * w_tmp[1] + r1[5] * w_tmp[2] +
                  r2[3] * w_tmp[3] + r2[4] * w_tmp[4] + r2[5] * w_tmp[5] +
                  r3[3] * w_tmp[6] + r3[4] * w_tmp[7] + r3[5] * w_tmp[8];
            }

            wc0 += 36;
            inr0 += win_round;
            inr1 += win_round;
            inr2 += win_round;
            inr3 += win_round;
          }
#endif  // __aarch64__
          block_inr0 = block_inr2;
          block_inr1 = block_inr3;
          block_inr2 = block_inr1 + in_len;
          block_inr3 = block_inr2 + in_len;
        }
        write_to_output_c1_fp32(pre_out, dout_batch, c_idx, c_idx + 1, h,
                                h + h_kernel, 0, wout_round, oc, oh, ow,
                                flag_relu, ptr_write);
      }
    }
  }
}

}  // namespace math
}  // namespace arm
}  // namespace lite
}  // namespace paddle
