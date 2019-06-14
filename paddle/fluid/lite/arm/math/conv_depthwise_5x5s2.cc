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

#include "paddle/fluid/lite/arm/math/conv_depthwise.h"
#include <arm_neon.h>

namespace paddle {
namespace lite {
namespace arm {
namespace math {

void conv_depthwise_5x5s2p2(const float* din, float* dout, int num, int ch_out,
                            int h_out, int w_out, int ch_in, int h_in, int w_in,
                            const float* weights, const float* bias,
                            bool flag_bias, bool flag_relu, ARMContext* ctx);

void conv_depthwise_5x5s2p2_relu(const float* din, float* dout, int num,
                                 int ch_out, int h_out, int w_out, int ch_in,
                                 int h_in, int w_in, const float* weights,
                                 const float* bias, bool flag_bias,
                                 bool flag_relu, ARMContext* ctx);

void conv_depthwise_5x5s2p2_s(const float* din, float* dout, int num,
                              int ch_out, int h_out, int w_out, int ch_in,
                              int h_in, int w_in, const float* weights,
                              const float* bias, bool flag_bias, bool flag_relu,
                              ARMContext* ctx);

void conv_depthwise_5x5s2p2_relu_s(const float* din, float* dout, int num,
                                   int ch_out, int h_out, int w_out, int ch_in,
                                   int h_in, int w_in, const float* weights,
                                   const float* bias, bool flag_bias,
                                   bool flag_relu, ARMContext* ctx);

void conv_depthwise_5x5s2(const float* din, float* dout, int num, int chout,
                          int hout, int wout, int chin, int hin, int win,
                          const float* weights, const float* bias, int pad,
                          bool flag_bias, bool flag_relu, ARMContext* ctx) {
  if (pad == 2) {
    if (win >= 9) {
      if (flag_relu) {
        conv_depthwise_5x5s2p2_relu(din, dout, num, chout, hout, wout, chin,
                                    hin, win, weights, bias, flag_bias,
                                    flag_relu, ctx);
      } else {
        conv_depthwise_5x5s2p2(din, dout, num, chout, hout, wout, chin, hin,
                               win, weights, bias, flag_bias, flag_relu, ctx);
      }
    } else {
      if (flag_relu) {
        conv_depthwise_5x5s2p2_relu_s(din, dout, num, chout, hout, wout, chin,
                                      hin, win, weights, bias, flag_bias,
                                      flag_relu, ctx);
      } else {
        conv_depthwise_5x5s2p2_s(din, dout, num, chout, hout, wout, chin, hin,
                                 win, weights, bias, flag_bias, flag_relu, ctx);
      }
    }
  }
}

#ifdef __aarch64__

//! larger depthwise, win >= 9;
void conv_depthwise_5x5s2p2(const float* din, float* dout, int num, int ch_out,
                            int h_out, int w_out, int ch_in, int h_in, int w_in,
                            const float* weights, const float* bias,
                            bool flag_bias, bool flag_relu, ARMContext* ctx) {
  CHECK_GE(w_in, 9) << "only support win >= 9";
  int w_out_round = (w_out + 3) / 4 * 4;
  int cnt = (w_out_round - 4) / 4;
  int mid_cnt = cnt - 1;
  int right_start = cnt * 2 * 4 - 2;
  int mask_cnt = 12 - (w_in - right_start);
  int mask[12];
  memset(mask, 0xff, 12 * sizeof(int));
  for (int i = 0; i < mask_cnt; ++i) {
    mask[11 - i] = 0;
  }
  float* zero_ptr = ctx->workspace_data<float>();
  memset(zero_ptr, 0, w_in * sizeof(float));
  float* write_ptr = zero_ptr + w_in;
  int in_spatial_size = w_in * h_in;
  int out_spatial_size = w_out * h_out;
  int weights_saptial_size = 25;

  for (int n = 0; n < num; ++n) {
    const float* din_batch = din + n * in_spatial_size * ch_in;
    float* dout_batch = dout + n * out_spatial_size * ch_out;
#pragma omp parallel for
    for (int c = 0; c < ch_in; ++c) {
      const float* din_ch = din_batch + c * in_spatial_size;
      float* dout_ch = dout_batch + c * out_spatial_size;
      const float* din0 = zero_ptr;
      const float* din1 = zero_ptr;
      const float* din2 = din_ch;
      const float* din3 = din2 + w_in;
      const float* din4 = din3 + w_in;
      const float* din5 = din4 + w_in;
      const float* din6 = din5 + w_in;

      float out_buf0[4];
      float out_buf1[4];
      float* dout0 = dout_ch;
      float* dout1 = dout0 + w_out;

      const float* weights_c = weights + c * weights_saptial_size;
      for (int h = 0; h < h_out; h += 2) {
        //! (h * 2 - 2) + 6 > h_in - 1
        if (h * 2 + 5 > h_in) {
          switch (h * 2 + 5 - h_in) {
            case 6:
              din1 = zero_ptr;
            case 5:
              din2 = zero_ptr;
            case 4:
              din3 = zero_ptr;
            case 3:
              din4 = zero_ptr;
            case 2:
              din5 = zero_ptr;
            case 1:
              din6 = zero_ptr;
            default:
              break;
          }
        }
        if (h + 2 > h_out) {
          switch (h + 2 - h_out) {
            case 1:
              dout1 = write_ptr;
            default:
              break;
          }
        }
        const float* din_ptr0 = din0;
        const float* din_ptr1 = din1;
        const float* din_ptr2 = din2;
        const float* din_ptr3 = din3;
        const float* din_ptr4 = din4;
        const float* din_ptr5 = din5;
        const float* din_ptr6 = din6;

        const float* weights_ptr = weights_c;
        float* dout_ptr0 = dout0;
        float* dout_ptr1 = dout1;

        float bias_c = 0.f;
        if (flag_bias) {
          bias_c = bias[c];
        }
        float vbias[4] = {bias_c, bias_c, bias_c, bias_c};
        int* mask_ptr = mask;
        int loop = mid_cnt;
        const int s_8 = 8;
        const int s_16 = 16;

        //! in r0, r1/r4, r2/r5, r3/r6: x 0 2 4 -- v8   v13  v18  v23
        //! in r0, r1/r4, r2/r5, r3/r6: x 1 3 5 -- v9   v14  v19  v24
        //! in r0, r1/r4, r2/r5, r3/r6: 0 2 4 6 -- v6   v11  v16  v21
        //! in r0, r1/r4, r2/r5, r3/r6: 1 3 5 7 -- v7   v12  v17  v22
        //! in r0, r1/r4, r2/r5, r3/r6: 2 4 6 8 -- v10  v15  v20  v25
        //! out r0, r1 -- v26, v27
        asm volatile(
            "movi   v31.4s, #0x0\n"
            "prfm pldl1keep, [%[din_ptr0]]  \n"
            "prfm pldl1keep, [%[din_ptr1]]  \n"
            "prfm pldl1keep, [%[din_ptr2]]  \n"
            "prfm pldl1keep, [%[din_ptr3]]  \n"
            "prfm pldl1keep, [%[din_ptr4]]  \n"
            "prfm pldl1keep, [%[din_ptr5]]  \n"
            "prfm pldl1keep, [%[din_ptr6]]  \n"
            "prfm pldl1keep, [%[weights]]   \n"
            "prfm pldl1keep, [%[mask]]      \n"
            // left
            "ld2 {v6.4s, v7.4s}, [%[din_ptr0]], #32             \n"  // r0 v6: 0
                                                                     // 2 4 6,
                                                                     // v7: 1 3
                                                                     // 5 7
            "ext v8.16b, v31.16b, v6.16b, #12                   \n"  // r0 v8: x
                                                                     // 0 2 4
            "ld2 {v11.4s, v12.4s}, [%[din_ptr1]], #32           \n"  // r1 v11:
                                                                     // 0 2 4 6,
                                                                     // v12: 1 3
                                                                     // 5 7
            "ext v9.16b, v31.16b, v7.16b, #12                   \n"  // r0 v9: x
                                                                     // 1 3 5
            "ld1 {v0.4s, v1.4s}, [%[weights]], #32              \n"  // load
                                                                     // weights
                                                                     // 0-7
            "ext v10.16b, v6.16b, v31.16b, #4                   \n"
            "ld1 {v10.s}[3], [%[din_ptr0]]                      \n"  // r0 v10:
                                                                     // 2 4 6 8
            "sub %[din_ptr0], %[din_ptr0], #8                   \n"
            "ext v13.16b, v31.16b, v11.16b, #12                 \n"  // r1 v13:
                                                                     // x 0 2 4
            "ld2 {v16.4s, v17.4s}, [%[din_ptr2]], #32           \n"  // r2 v16:
                                                                     // 0 2 4 6,
                                                                     // v17: 1 3
                                                                     // 5 7
            "ext v14.16b, v31.16b, v12.16b, #12                 \n"  // r1 v14:
                                                                     // x 1 3 5
            "ld1 {v2.4s, v3.4s}, [%[weights]], #32              \n"  // load
                                                                     // weights
                                                                     // 8-15
            "ext v15.16b, v11.16b, v31.16b, #4                  \n"
            "ld1 {v15.s}[3], [%[din_ptr1]]                      \n"  // r1 v15:
                                                                     // 2 4 6
            "sub %[din_ptr1], %[din_ptr1], #8                   \n"
            "ext v18.16b, v31.16b, v16.16b, #12                 \n"  // r2 v18:
                                                                     // x 0 2 4
            "ld1 {v4.4s, v5.4s}, [%[weights]], #32              \n"  // load
                                                                     // weights
                                                                     // 16-23
            "ext v19.16b, v31.16b, v17.16b, #12                 \n"  // r2 v19:
                                                                     // x 1 3 5
            "ld2 {v21.4s, v22.4s}, [%[din_ptr3]], #32           \n"  // r3 v21:
                                                                     // 0 2 4 6,
                                                                     // v22: 1 3
                                                                     // 5 7
            "ext v20.16b, v16.16b, v31.16b, #4                  \n"
            "ld1 {v20.s}[3], [%[din_ptr2]]                      \n"  // r2 v20:
                                                                     // 2 4 6 8
            "sub %[din_ptr2], %[din_ptr2], #8                   \n"
            "ext v23.16b, v31.16b, v21.16b, #12                 \n"  // r3 v23:
                                                                     // x 0 2 4
            "ld1 {v30.4s}, [%[weights]]                         \n"  // load
                                                                     // weights
                                                                     // 24
            "ext v24.16b, v31.16b, v22.16b, #12                 \n"  // r3 v24:
                                                                     // x 1 3 5
            "ld1 {v26.4s}, [%[vbias]]                           \n"  // load
                                                                     // bias to
                                                                     // out_r0
            "ext v25.16b, v21.16b, v31.16b, #4                  \n"
            "ld1 {v25.s}[3], [%[din_ptr3]]                      \n"  // r2 v25:
                                                                     // 2 4 6 8
            "sub %[din_ptr3], %[din_ptr3], #8                   \n"
            "mov v27.16b, v26.16b                               \n"  // load
                                                                     // bias to
                                                                     // out_r1
            "mov v28.16b, v31.16b                               \n"  // load
                                                                     // zero to
                                                                     // out_r0
            "mov v29.16b, v31.16b                               \n"  // load
                                                                     // zero to
                                                                     // out_r1

            "fmla v26.4s, v8.4s, v0.s[0]                        \n"  // out r0:
                                                                     // w0
            "fmla v28.4s, v9.4s, v0.s[1]                        \n"  // out r0:
                                                                     // w1
            "fmla v26.4s, v6.4s, v0.s[2]                        \n"  // out r0:
                                                                     // w2
            "fmla v28.4s, v7.4s, v0.s[3]                        \n"  // out r0:
                                                                     // w3

            "ld2 {v8.4s, v9.4s}, [%[din_ptr0]], %[s_8]          \n"  // next r0
                                                                     // v8: 0 2
                                                                     // 4 6, v9:
                                                                     // 1 3 5 7

            "fmla v26.4s, v10.4s, v1.s[0]                       \n"  // out r0:
                                                                     // w4
            "fmla v28.4s, v13.4s, v1.s[1]                       \n"  // out r0:
                                                                     // w5
            "fmla v26.4s, v14.4s, v1.s[2]                       \n"  // out r0:
                                                                     // w6
            "fmla v28.4s, v11.4s, v1.s[3]                       \n"  // out r0:
                                                                     // w7

            "ld2 {v6.4s, v7.4s}, [%[din_ptr0]], %[s_8]          \n"  // next r0
                                                                     // v6: 2 4
                                                                     // 6 8, v7:
                                                                     // 3 5 7 9

            "fmla v26.4s, v12.4s, v2.s[0]                       \n"  // out r0:
                                                                     // w8
            "fmla v28.4s, v15.4s, v2.s[1]                       \n"  // out r0:
                                                                     // w9
            "fmla v26.4s, v18.4s, v2.s[2]                       \n"  // out r0:
                                                                     // w10
            "fmla v28.4s, v19.4s, v2.s[3]                       \n"  // out r0:
                                                                     // w11

            "ld2 {v10.4s, v11.4s}, [%[din_ptr0]], %[s_16]       \n"  // next r0
                                                                     // v10: 4 6
                                                                     // 8 10,
                                                                     // v11:
                                                                     // trash
                                                                     // register

            "fmla v26.4s, v16.4s, v3.s[0]                       \n"  // out r0:
                                                                     // w12
            "fmla v28.4s, v17.4s, v3.s[1]                       \n"  // out r0:
                                                                     // w13
            "fmla v26.4s, v20.4s, v3.s[2]                       \n"  // out r0:
                                                                     // w14
            "fmla v28.4s, v23.4s, v3.s[3]                       \n"  // out r0:
                                                                     // w15
            "prfm pldl1keep, [%[din_ptr0]]                      \n"

            "ld2 {v11.4s, v12.4s}, [%[din_ptr4]], #32           \n"  // r4 v11:
                                                                     // 0 2 4 6,
                                                                     // v12: 1 3
                                                                     // 5 7

            "fmla v26.4s, v24.4s, v4.s[0]                       \n"  // out r0:
                                                                     // w16
            "fmla v28.4s, v21.4s, v4.s[1]                       \n"  // out r0:
                                                                     // w17

            "ext v13.16b, v31.16b, v11.16b, #12                 \n"  // r4 v13:
                                                                     // x 0 2 4
            "ext v14.16b, v31.16b, v12.16b, #12                 \n"  // r4 v14:
                                                                     // x 1 3 5
            "ext v15.16b, v11.16b, v31.16b, #4                  \n"

            "fmla v26.4s, v22.4s, v4.s[2]                       \n"  // out r0:
                                                                     // w18
            "fmla v28.4s, v25.4s, v4.s[3]                       \n"  // out r0:
                                                                     // w19

            "ld1 {v15.s}[3], [%[din_ptr4]]                      \n"  // r4 v15:
                                                                     // 2 4 6

            "fmla v27.4s, v18.4s, v0.s[0]                       \n"  // out r1:
                                                                     // w0
            "fmla v29.4s, v19.4s, v0.s[1]                       \n"  // out r1:
                                                                     // w1

            "sub %[din_ptr4], %[din_ptr4], #8                   \n"

            "fmla v27.4s, v16.4s, v0.s[2]                       \n"  // out r1:
                                                                     // w2
            "fmla v29.4s, v17.4s, v0.s[3]                       \n"  // out r1:
                                                                     // w3
            "fmla v27.4s, v20.4s, v1.s[0]                       \n"  // out r1:
                                                                     // w4
            "fmla v29.4s, v23.4s, v1.s[1]                       \n"  // out r1:
                                                                     // w5

            "ld2 {v16.4s, v17.4s}, [%[din_ptr5]], #32           \n"  // r5 v16:
                                                                     // 0 2 4 6,
                                                                     // v17: 1 3
                                                                     // 5 7

            "fmla v27.4s, v24.4s, v1.s[2]                       \n"  // out r1:
                                                                     // w6
            "fmla v29.4s, v21.4s, v1.s[3]                       \n"  // out r1:
                                                                     // w7

            "ext v18.16b, v31.16b, v16.16b, #12                 \n"  // r5 v18:
                                                                     // x 0 2 4
            "ext v19.16b, v31.16b, v17.16b, #12                 \n"  // r5 v19:
                                                                     // x 1 3 5
            "ext v20.16b, v16.16b, v31.16b, #4                  \n"

            "fmla v27.4s, v22.4s, v2.s[0]                       \n"  // out r1:
                                                                     // w8
            "fmla v29.4s, v25.4s, v2.s[1]                       \n"  // out r1:
                                                                     // w9

            "ld1 {v20.s}[3], [%[din_ptr5]]                      \n"  // r5 v20:
                                                                     // 2 4 6
            "ld2 {v21.4s, v22.4s}, [%[din_ptr6]], #32           \n"  // r6 v21:
                                                                     // 0 2 4 6,
                                                                     // v22: 1 3
                                                                     // 5 7

            "ext v23.16b, v31.16b, v21.16b, #12                 \n"  // r6 v23:
                                                                     // x 0 2 4
            "ext v24.16b, v31.16b, v22.16b, #12                 \n"  // r6 v24:
                                                                     // x 1 3 5
            "ext v25.16b, v21.16b, v31.16b, #4                  \n"
            "sub %[din_ptr5], %[din_ptr5], #8                   \n"

            "fmla v26.4s, v11.4s, v5.s[2]                       \n"  // out r0:
                                                                     // w22
            "fmla v28.4s, v12.4s, v5.s[3]                       \n"  // out r0:
                                                                     // w23

            "ld1 {v25.s}[3], [%[din_ptr6]]                      \n"  // r6 v25:
                                                                     // 2 4 6

            "fmla v26.4s, v13.4s, v5.s[0]                       \n"  // out r0:
                                                                     // w20
            "fmla v28.4s, v14.4s, v5.s[1]                       \n"  // out r0:
                                                                     // w21

            "sub %[din_ptr6], %[din_ptr6], #8                   \n"

            "fmla v26.4s, v15.4s, v30.s[0]                      \n"  // out r0:
                                                                     // w24
            "fmla v27.4s, v13.4s, v2.s[2]                       \n"  // out r1:
                                                                     // w10

            "fadd v26.4s, v26.4s, v28.4s                        \n"
            "fmla v29.4s, v14.4s, v2.s[3]                       \n"  // out r1:
                                                                     // w11

            "ld2 {v13.4s, v14.4s}, [%[din_ptr1]], %[s_8]        \n"  // next r1
                                                                     // v13: 0 2
                                                                     // 4 6,
                                                                     // v14: 1 3
                                                                     // 5 7
            "fmla v27.4s, v11.4s, v3.s[0]                       \n"  // out r1:
                                                                     // w12
            "fmla v29.4s, v12.4s, v3.s[1]                       \n"  // out r1:
                                                                     // w13

            "st1 {v26.4s}, [%[dout_ptr0]], %[s_16]              \n"  // store
                                                                     // output
                                                                     // r0
            "ld2 {v11.4s, v12.4s}, [%[din_ptr1]], %[s_8]        \n"  // next r1
                                                                     // v11: 2 4
                                                                     // 6 8,
                                                                     // v12: 3 5
                                                                     // 7 9

            "fmla v27.4s, v15.4s, v3.s[2]                       \n"  // out r1:
                                                                     // w14
            "fmla v29.4s, v16.4s, v4.s[1]                       \n"  // out r1:
                                                                     // w17
            "fmla v27.4s, v18.4s, v3.s[3]                       \n"  // out r1:
                                                                     // w15
            "fmla v29.4s, v19.4s, v4.s[0]                       \n"  // out r1:
                                                                     // w16

            "ld2 {v15.4s, v16.4s}, [%[din_ptr1]], %[s_16]       \n"  // next r1
                                                                     // v15: 4 6
                                                                     // 8 10,
                                                                     // v16:
                                                                     // trash
                                                                     // register

            "fmla v27.4s, v17.4s, v4.s[2]                       \n"  // out r1:
                                                                     // w18
            "fmla v29.4s, v20.4s, v4.s[3]                       \n"  // out r1:
                                                                     // w19

            "ld2 {v18.4s, v19.4s}, [%[din_ptr2]], %[s_8]        \n"  // next r2
                                                                     // v18: 0 2
                                                                     // 4 6,
                                                                     // v19: 1 3
                                                                     // 5 7
            "ld2 {v16.4s, v17.4s}, [%[din_ptr2]], %[s_8]        \n"  // next r2
                                                                     // v16: 2 4
                                                                     // 6 8,
                                                                     // v11: 3 5
                                                                     // 7 9

            "fmla v27.4s, v23.4s, v5.s[0]                       \n"  // out r1:
                                                                     // w20
            "fmla v29.4s, v21.4s, v5.s[2]                       \n"  // out r1:
                                                                     // w22
            "fmla v27.4s, v24.4s, v5.s[1]                       \n"  // out r1:
                                                                     // w21
            "fmla v29.4s, v22.4s, v5.s[3]                       \n"  // out r1:
                                                                     // w23

            "ld2 {v20.4s, v21.4s}, [%[din_ptr2]], %[s_16]       \n"  // next r2
                                                                     // v20: 4 6
                                                                     // 8 10,
                                                                     // v21:
                                                                     // trash
                                                                     // register
            "ld2 {v23.4s, v24.4s}, [%[din_ptr3]], %[s_8]        \n"  // next r3
                                                                     // v23: 0 2
                                                                     // 4 6,
                                                                     // v24: 1 3
                                                                     // 5 7

            "fmla v27.4s, v25.4s, v30.s[0]                      \n"  // out r1:
                                                                     // w24

            "ld2 {v21.4s, v22.4s}, [%[din_ptr3]], %[s_8]        \n"  // next r3
                                                                     // v21: 2 4
                                                                     // 6 8,
                                                                     // v22: 3 5
                                                                     // 7 9
            "ld2 {v25.4s, v26.4s}, [%[din_ptr3]], %[s_16]       \n"  // next r3
                                                                     // v25: 4 6
                                                                     // 8 10,
                                                                     // v26:
                                                                     // trash
                                                                     // register

            "fadd v27.4s, v27.4s, v29.4s                        \n"
            "cmp %w[mid_cnt], #1                                \n"

            "prfm pldl1keep, [%[din_ptr1]]                      \n"
            "prfm pldl1keep, [%[din_ptr2]]                      \n"
            "prfm pldl1keep, [%[din_ptr3]]                      \n"

            "st1 {v27.4s}, [%[dout_ptr1]], #16                  \n"
            "blt 2f                                             \n"

            // mid loop
            "1:                                                 \n"
            "ld1 {v26.4s}, [%[vbias]]                           \n"
            "mov v27.16b, v26.16b                               \n"
            "mov v28.16b, v31.16b                               \n"
            "mov v29.16b, v31.16b                               \n"

            // out_r0 r0-r3
            "fmla v26.4s, v8.4s, v0.s[0]                        \n"
            "fmla v28.4s, v9.4s, v0.s[1]                        \n"
            "fmla v26.4s, v6.4s, v0.s[2]                        \n"
            "fmla v28.4s, v7.4s, v0.s[3]                        \n"

            "ld2 {v8.4s, v9.4s}, [%[din_ptr0]], %[s_8]          \n"

            "fmla v26.4s, v10.4s, v1.s[0]                       \n"
            "fmla v28.4s, v11.4s, v1.s[3]                       \n"

            "ld2 {v6.4s, v7.4s}, [%[din_ptr0]], %[s_8]          \n"

            "fmla v26.4s, v14.4s, v1.s[2]                       \n"
            "fmla v28.4s, v13.4s, v1.s[1]                       \n"

            "ld2 {v10.4s, v11.4s}, [%[din_ptr0]], %[s_16]       \n"
            "prfm pldl1keep, [%[din_ptr0]]                      \n"

            "fmla v26.4s, v12.4s, v2.s[0]                       \n"
            "fmla v28.4s, v15.4s, v2.s[1]                       \n"

            "ld2 {v13.4s, v14.4s}, [%[din_ptr4]], %[s_8]        \n"

            "fmla v26.4s, v16.4s, v3.s[0]                       \n"
            "fmla v27.4s, v16.4s, v0.s[2]                       \n"

            "ld2 {v11.4s, v12.4s}, [%[din_ptr4]], %[s_8]        \n"

            "fmla v28.4s, v19.4s, v2.s[3]                       \n"
            "fmla v29.4s, v19.4s, v0.s[1]                       \n"

            "ld2 {v15.4s, v16.4s}, [%[din_ptr4]], %[s_16]       \n"
            "prfm pldl1keep, [%[din_ptr4]]                      \n"

            "fmla v26.4s, v18.4s, v2.s[2]                       \n"
            "fmla v27.4s, v18.4s, v0.s[0]                       \n"

            "fmla v28.4s, v17.4s, v3.s[1]                       \n"
            "fmla v29.4s, v17.4s, v0.s[3]                       \n"

            "ld2 {v18.4s, v19.4s}, [%[din_ptr5]], %[s_8]        \n"

            "fmla v26.4s, v20.4s, v3.s[2]                       \n"
            "fmla v27.4s, v20.4s, v1.s[0]                       \n"

            "ld2 {v16.4s, v17.4s}, [%[din_ptr5]], %[s_8]        \n"

            "fmla v29.4s, v21.4s, v1.s[3]                       \n"
            "fmla v28.4s, v21.4s, v4.s[1]                       \n"
            "fmla v28.4s, v23.4s, v3.s[3]                       \n"
            "fmla v29.4s, v23.4s, v1.s[1]                       \n"

            "ld2 {v20.4s, v21.4s}, [%[din_ptr5]], %[s_16]       \n"
            "prfm pldl1keep, [%[din_ptr5]]                      \n"

            "fmla v26.4s, v24.4s, v4.s[0]                       \n"
            "fmla v27.4s, v24.4s, v1.s[2]                       \n"

            "ld2 {v23.4s, v24.4s}, [%[din_ptr6]], %[s_8]        \n"

            "fmla v27.4s, v22.4s, v2.s[0]                       \n"
            "fmla v26.4s, v22.4s, v4.s[2]                       \n"

            "fmla v28.4s, v25.4s, v4.s[3]                       \n"
            "fmla v29.4s, v25.4s, v2.s[1]                       \n"

            "ld2 {v21.4s, v22.4s}, [%[din_ptr6]], %[s_8]        \n"
            "fadd v28.4s, v26.4s, v28.4s                        \n"

            "ld2 {v25.4s, v26.4s}, [%[din_ptr6]], %[s_16]       \n"
            "mov v26.16b, v31.16b                               \n"
            "prfm pldl1keep, [%[din_ptr6]]                      \n"

            "fmla v26.4s, v13.4s, v5.s[0]                       \n"
            "fmla v28.4s, v14.4s, v5.s[1]                       \n"
            "fmla v27.4s, v13.4s, v2.s[2]                       \n"
            "fmla v29.4s, v14.4s, v2.s[3]                       \n"

            "ld2 {v13.4s, v14.4s}, [%[din_ptr1]], %[s_8]        \n"

            "fmla v26.4s, v11.4s, v5.s[2]                       \n"
            "fmla v28.4s, v12.4s, v5.s[3]                       \n"
            "fmla v27.4s, v11.4s, v3.s[0]                       \n"
            "fmla v29.4s, v12.4s, v3.s[1]                       \n"

            "ld2 {v11.4s, v12.4s}, [%[din_ptr1]], %[s_8]        \n"

            "fmla v26.4s, v15.4s, v30.s[0]                      \n"
            "fmla v27.4s, v15.4s, v3.s[2]                       \n"
            "fmla v29.4s, v16.4s, v4.s[1]                       \n"
            "fmla v27.4s, v17.4s, v4.s[2]                       \n"

            "ld2 {v15.4s, v16.4s}, [%[din_ptr1]], %[s_16]       \n"
            "prfm pldl1keep, [%[din_ptr1]]                      \n"

            "fmla v29.4s, v18.4s, v3.s[3]                       \n"
            "fmla v27.4s, v19.4s, v4.s[0]                       \n"

            "ld2 {v18.4s, v19.4s}, [%[din_ptr2]], %[s_8]        \n"

            "fmla v29.4s, v20.4s, v4.s[3]                       \n"

            "ld2 {v16.4s, v17.4s}, [%[din_ptr2]], %[s_8]        \n"

            "fmla v27.4s, v23.4s, v5.s[0]                       \n"
            "fmla v27.4s, v21.4s, v5.s[2]                       \n"

            "ld2 {v20.4s, v21.4s}, [%[din_ptr2]], %[s_16]       \n"

            "fmla v29.4s, v24.4s, v5.s[1]                       \n"

            "ld2 {v23.4s, v24.4s}, [%[din_ptr3]], %[s_8]        \n"
            "prfm pldl1keep, [%[din_ptr2]]                      \n"

            "fmla v29.4s, v22.4s, v5.s[3]                       \n"

            "ld2 {v21.4s, v22.4s}, [%[din_ptr3]], %[s_8]        \n"

            "fmla v27.4s, v25.4s, v30.s[0]                      \n"

            "fadd v26.4s, v26.4s, v28.4s                        \n"

            "prfm pldl1keep, [%[din_ptr3]]                      \n"

            "fadd v27.4s, v27.4s, v29.4s                        \n"

            "st1 {v26.4s}, [%[dout_ptr0]], #16                  \n"
            "st1 {v27.4s}, [%[dout_ptr1]], #16                  \n"

            "ld2 {v25.4s, v26.4s}, [%[din_ptr3]], %[s_16]       \n"
            "subs %w[mid_cnt], %w[mid_cnt], #1                  \n"
            "bne 1b                                             \n"

            "2:                                                 \n"
            "ld2 {v26.4s, v27.4s}, [%[mask]], %[s_8]            \n"
            "ld2 {v28.4s, v29.4s}, [%[mask]], %[s_8]            \n"
            "bif v8.16b, v31.16b, v26.16b                       \n"
            "bif v9.16b, v31.16b, v27.16b                       \n"
            "bif v6.16b, v31.16b, v28.16b                       \n"
            "bif v7.16b, v31.16b, v29.16b                       \n"

            "bif v13.16b, v31.16b, v26.16b                      \n"
            "bif v14.16b, v31.16b, v27.16b                      \n"
            "bif v11.16b, v31.16b, v28.16b                      \n"
            "bif v12.16b, v31.16b, v29.16b                      \n"

            "bif v18.16b, v31.16b, v26.16b                      \n"
            "bif v19.16b, v31.16b, v27.16b                      \n"
            "bif v16.16b, v31.16b, v28.16b                      \n"
            "bif v17.16b, v31.16b, v29.16b                      \n"

            "bif v23.16b, v31.16b, v26.16b                      \n"
            "bif v24.16b, v31.16b, v27.16b                      \n"
            "bif v21.16b, v31.16b, v28.16b                      \n"
            "bif v22.16b, v31.16b, v29.16b                      \n"

            "ld2 {v28.4s, v29.4s}, [%[mask]]                    \n"
            "ld1 {v26.4s}, [%[vbias]]                           \n"
            "mov v29.16b, v31.16b                               \n"

            "bif v10.16b, v31.16b, v28.16b                      \n"
            "bif v15.16b, v31.16b, v28.16b                      \n"

            "mov v27.16b, v26.16b                               \n"

            "bif v20.16b, v31.16b, v28.16b                      \n"
            "bif v25.16b, v31.16b, v28.16b                      \n"
            "mov v28.16b, v31.16b                               \n"

            "fmla v26.4s, v8.4s, v0.s[0]                        \n"
            "fmla v28.4s, v9.4s, v0.s[1]                        \n"
            "fmla v26.4s, v6.4s, v0.s[2]                        \n"
            "fmla v28.4s, v7.4s, v0.s[3]                        \n"

            "fmla v26.4s, v10.4s, v1.s[0]                       \n"
            "fmla v28.4s, v13.4s, v1.s[1]                       \n"
            "fmla v26.4s, v14.4s, v1.s[2]                       \n"
            "fmla v28.4s, v11.4s, v1.s[3]                       \n"

            "sub %[mask], %[mask], #16                          \n"
            "ld2 {v6.4s, v7.4s}, [%[mask]], %[s_8]              \n"
            "ld2 {v8.4s, v9.4s}, [%[mask]], %[s_8]              \n"
            "ld2 {v10.4s, v11.4s}, [%[mask]]                    \n"

            "fmla v26.4s, v12.4s, v2.s[0]                       \n"
            "fmla v28.4s, v15.4s, v2.s[1]                       \n"

            "ld2 {v13.4s, v14.4s}, [%[din_ptr4]], %[s_8]        \n"

            "fmla v26.4s, v16.4s, v3.s[0]                       \n"
            "fmla v28.4s, v17.4s, v3.s[1]                       \n"

            "ld2 {v11.4s, v12.4s}, [%[din_ptr4]], %[s_8]        \n"

            "fmla v27.4s, v16.4s, v0.s[2]                       \n"
            "fmla v29.4s, v17.4s, v0.s[3]                       \n"

            "ld2 {v15.4s, v16.4s}, [%[din_ptr4]]                \n"

            "fmla v26.4s, v18.4s, v2.s[2]                       \n"
            "fmla v28.4s, v19.4s, v2.s[3]                       \n"
            "fmla v27.4s, v18.4s, v0.s[0]                       \n"
            "fmla v29.4s, v19.4s, v0.s[1]                       \n"

            "bif  v13.16b, v31.16b, v6.16b                      \n"
            "bif  v14.16b, v31.16b, v7.16b                      \n"
            "bif  v11.16b, v31.16b, v8.16b                      \n"
            "bif  v12.16b, v31.16b, v9.16b                      \n"
            "bif  v15.16b, v31.16b, v10.16b                     \n"

            "ld2 {v18.4s, v19.4s}, [%[din_ptr5]], %[s_8]        \n"

            "fmla v26.4s, v20.4s, v3.s[2]                       \n"
            "fmla v27.4s, v20.4s, v1.s[0]                       \n"

            "ld2 {v16.4s, v17.4s}, [%[din_ptr5]], %[s_8]        \n"

            "fmla v29.4s, v21.4s, v1.s[3]                       \n"
            "fmla v28.4s, v21.4s, v4.s[1]                       \n"

            "ld2 {v20.4s, v21.4s}, [%[din_ptr5]]                \n"

            "fmla v28.4s, v23.4s, v3.s[3]                       \n"
            "fmla v29.4s, v23.4s, v1.s[1]                       \n"
            "fmla v27.4s, v24.4s, v1.s[2]                       \n"
            "fmla v26.4s, v24.4s, v4.s[0]                       \n"

            "bif  v18.16b, v31.16b, v6.16b                      \n"
            "bif  v19.16b, v31.16b, v7.16b                      \n"
            "bif  v16.16b, v31.16b, v8.16b                      \n"
            "bif  v17.16b, v31.16b, v9.16b                      \n"
            "bif  v20.16b, v31.16b, v10.16b                     \n"

            "ld2 {v23.4s, v24.4s}, [%[din_ptr6]], %[s_8]        \n"

            "fmla v27.4s, v22.4s, v2.s[0]                       \n"
            "fmla v26.4s, v22.4s, v4.s[2]                       \n"

            "ld2 {v21.4s, v22.4s}, [%[din_ptr6]], %[s_8]        \n"

            "fmla v28.4s, v25.4s, v4.s[3]                       \n"
            "fmla v29.4s, v25.4s, v2.s[1]                       \n"
            "fadd v28.4s, v28.4s, v26.4s                        \n"

            "ld2 {v25.4s, v26.4s}, [%[din_ptr6]]                \n"
            "mov v26.16b, v31.16b                               \n"

            "bif  v23.16b, v31.16b, v6.16b                      \n"
            "bif  v24.16b, v31.16b, v7.16b                      \n"
            "bif  v21.16b, v31.16b, v8.16b                      \n"
            "bif  v22.16b, v31.16b, v9.16b                      \n"
            "bif  v25.16b, v31.16b, v10.16b                     \n"

            "fmla v26.4s, v13.4s, v5.s[0]                       \n"
            "fmla v28.4s, v14.4s, v5.s[1]                       \n"
            "fmla v26.4s, v11.4s, v5.s[2]                       \n"
            "fmla v28.4s, v12.4s, v5.s[3]                       \n"
            "fmla v26.4s, v15.4s, v30.s[0]                      \n"

            "fmla v27.4s, v13.4s, v2.s[2]                       \n"
            "fmla v29.4s, v14.4s, v2.s[3]                       \n"
            "fmla v27.4s, v11.4s, v3.s[0]                       \n"
            "fmla v29.4s, v12.4s, v3.s[1]                       \n"

            "fadd v26.4s, v26.4s, v28.4s                        \n"
            "fmla v27.4s, v15.4s, v3.s[2]                       \n"
            "fmla v29.4s, v18.4s, v3.s[3]                       \n"
            "fmla v27.4s, v19.4s, v4.s[0]                       \n"
            "fmla v29.4s, v16.4s, v4.s[1]                       \n"

            "st1 {v26.4s}, [%[out_buf0]]                        \n"
            "fmla v27.4s, v17.4s, v4.s[2]                       \n"
            "fmla v29.4s, v20.4s, v4.s[3]                       \n"
            "fmla v27.4s, v23.4s, v5.s[0]                       \n"
            "fmla v29.4s, v24.4s, v5.s[1]                       \n"

            "fmla v27.4s, v21.4s, v5.s[2]                       \n"
            "fmla v29.4s, v22.4s, v5.s[3]                       \n"
            "fmla v27.4s, v25.4s, v30.s[0]                      \n"
            "fadd v27.4s, v27.4s, v29.4s                        \n"

            "st1 {v27.4s}, [%[out_buf1]]                        \n"

            : [dout_ptr0] "+r"(dout_ptr0), [dout_ptr1] "+r"(dout_ptr1),
              [mid_cnt] "+r"(loop), [din_ptr0] "+r"(din_ptr0),
              [din_ptr1] "+r"(din_ptr1), [din_ptr2] "+r"(din_ptr2),
              [din_ptr3] "+r"(din_ptr3), [din_ptr4] "+r"(din_ptr4),
              [din_ptr5] "+r"(din_ptr5), [din_ptr6] "+r"(din_ptr6),
              [mask] "+r"(mask_ptr), [weights] "+r"(weights_ptr)
            : [vbias] "r"(vbias), [out_buf0] "r"(out_buf0),
              [out_buf1] "r"(out_buf1), [s_8] "r"(s_8), [s_16] "r"(s_16)
            : "memory", "cc", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7",
              "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16",
              "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25",
              "v26", "v27", "v28", "v29", "v30", "v31");

        int remain_cnt = w_out - (mid_cnt + 1) * 4;
        for (int i = 0; i < remain_cnt; ++i) {
          dout_ptr0[i] = out_buf0[i];
          dout_ptr1[i] = out_buf1[i];
        }
        din0 = din4;
        din1 = din5;
        din2 = din6;
        din3 = din6 + w_in;
        din4 = din3 + w_in;
        din5 = din4 + w_in;
        din6 = din5 + w_in;
        dout0 = dout1 + w_out;
        dout1 = dout0 + w_out;
      }
    }
  }
}

//! larger depthwise, win >= 9;
void conv_depthwise_5x5s2p2_relu(const float* din, float* dout, int num,
                                 int ch_out, int h_out, int w_out, int ch_in,
                                 int h_in, int w_in, const float* weights,
                                 const float* bias, bool flag_bias,
                                 bool flag_relu, ARMContext* ctx) {
  CHECK_GE(w_in, 9) << "only support win >= 9";
  int w_out_round = (w_out + 3) / 4 * 4;
  int cnt = (w_out_round - 4) / 4;
  int mid_cnt = cnt - 1;
  int right_start = cnt * 2 * 4 - 2;
  int mask_cnt = 12 - (w_in - right_start);
  int mask[12];
  memset(mask, 0xff, 12 * sizeof(int));
  for (int i = 0; i < mask_cnt; ++i) {
    mask[11 - i] = 0;
  }
  float* zero_ptr = ctx->workspace_data<float>();
  memset(zero_ptr, 0, w_in * sizeof(float));
  float* write_ptr = zero_ptr + w_in;
  int in_spatial_size = w_in * h_in;
  int out_spatial_size = w_out * h_out;
  int weights_saptial_size = 25;

  for (int n = 0; n < num; ++n) {
    const float* din_batch = din + n * in_spatial_size * ch_in;
    float* dout_batch = dout + n * out_spatial_size * ch_out;

#pragma omp parallel for
    for (int c = 0; c < ch_in; ++c) {
      const float* din_ch = din_batch + c * in_spatial_size;
      float* dout_ch = dout_batch + c * out_spatial_size;
      const float* din0 = zero_ptr;
      const float* din1 = zero_ptr;
      const float* din2 = din_ch;
      const float* din3 = din2 + w_in;
      const float* din4 = din3 + w_in;
      const float* din5 = din4 + w_in;
      const float* din6 = din5 + w_in;

      float out_buf0[4];
      float out_buf1[4];
      float* dout0 = dout_ch;
      float* dout1 = dout0 + w_out;

      const float* weights_c = weights + c * weights_saptial_size;
      for (int h = 0; h < h_out; h += 2) {
        //! (h * 2 - 2) + 6 > h_in - 1
        if (h * 2 + 5 > h_in) {
          switch (h * 2 + 5 - h_in) {
            case 6:
              din1 = zero_ptr;
            case 5:
              din2 = zero_ptr;
            case 4:
              din3 = zero_ptr;
            case 3:
              din4 = zero_ptr;
            case 2:
              din5 = zero_ptr;
            case 1:
              din6 = zero_ptr;
            default:
              break;
          }
        }
        if (h + 2 > h_out) {
          switch (h + 2 - h_out) {
            case 1:
              dout1 = write_ptr;
            default:
              break;
          }
        }
        const float* din_ptr0 = din0;
        const float* din_ptr1 = din1;
        const float* din_ptr2 = din2;
        const float* din_ptr3 = din3;
        const float* din_ptr4 = din4;
        const float* din_ptr5 = din5;
        const float* din_ptr6 = din6;

        const float* weights_ptr = weights_c;
        float* dout_ptr0 = dout0;
        float* dout_ptr1 = dout1;

        float bias_c = 0.f;
        if (flag_bias) {
          bias_c = bias[c];
        }
        float vbias[4] = {bias_c, bias_c, bias_c, bias_c};
        int* mask_ptr = mask;
        int loop = mid_cnt;
        const int s_8 = 8;
        const int s_16 = 16;

        //! in r0, r1/r4, r2/r5, r3/r6: x 0 2 4 -- v8   v13  v18  v23
        //! in r0, r1/r4, r2/r5, r3/r6: x 1 3 5 -- v9   v14  v19  v24
        //! in r0, r1/r4, r2/r5, r3/r6: 0 2 4 6 -- v6   v11  v16  v21
        //! in r0, r1/r4, r2/r5, r3/r6: 1 3 5 7 -- v7   v12  v17  v22
        //! in r0, r1/r4, r2/r5, r3/r6: 2 4 6 8 -- v10  v15  v20  v25
        //! out r0, r1 -- v26, v27
        asm volatile(
            "movi   v31.4s, #0x0\n"
            "prfm pldl1keep, [%[din_ptr0]]  \n"
            "prfm pldl1keep, [%[din_ptr1]]  \n"
            "prfm pldl1keep, [%[din_ptr2]]  \n"
            "prfm pldl1keep, [%[din_ptr3]]  \n"
            "prfm pldl1keep, [%[din_ptr4]]  \n"
            "prfm pldl1keep, [%[din_ptr5]]  \n"
            "prfm pldl1keep, [%[din_ptr6]]  \n"
            "prfm pldl1keep, [%[weights]]   \n"
            "prfm pldl1keep, [%[mask]]      \n"
            // left
            "ld2 {v6.4s, v7.4s}, [%[din_ptr0]], #32             \n"  // r0 v6: 0
                                                                     // 2 4 6,
                                                                     // v7: 1 3
                                                                     // 5 7
            "ext v8.16b, v31.16b, v6.16b, #12                   \n"  // r0 v8: x
                                                                     // 0 2 4
            "ld2 {v11.4s, v12.4s}, [%[din_ptr1]], #32           \n"  // r1 v11:
                                                                     // 0 2 4 6,
                                                                     // v12: 1 3
                                                                     // 5 7
            "ext v9.16b, v31.16b, v7.16b, #12                   \n"  // r0 v9: x
                                                                     // 1 3 5
            "ld1 {v0.4s, v1.4s}, [%[weights]], #32              \n"  // load
                                                                     // weights
                                                                     // 0-7
            "ext v10.16b, v6.16b, v31.16b, #4                   \n"
            "ld1 {v10.s}[3], [%[din_ptr0]]                      \n"  // r0 v10:
                                                                     // 2 4 6 8
            "sub %[din_ptr0], %[din_ptr0], #8                   \n"
            "ext v13.16b, v31.16b, v11.16b, #12                 \n"  // r1 v13:
                                                                     // x 0 2 4
            "ld2 {v16.4s, v17.4s}, [%[din_ptr2]], #32           \n"  // r2 v16:
                                                                     // 0 2 4 6,
                                                                     // v17: 1 3
                                                                     // 5 7
            "ext v14.16b, v31.16b, v12.16b, #12                 \n"  // r1 v14:
                                                                     // x 1 3 5
            "ld1 {v2.4s, v3.4s}, [%[weights]], #32              \n"  // load
                                                                     // weights
                                                                     // 8-15
            "ext v15.16b, v11.16b, v31.16b, #4                  \n"
            "ld1 {v15.s}[3], [%[din_ptr1]]                      \n"  // r1 v15:
                                                                     // 2 4 6
            "sub %[din_ptr1], %[din_ptr1], #8                   \n"
            "ext v18.16b, v31.16b, v16.16b, #12                 \n"  // r2 v18:
                                                                     // x 0 2 4
            "ld1 {v4.4s, v5.4s}, [%[weights]], #32              \n"  // load
                                                                     // weights
                                                                     // 16-23
            "ext v19.16b, v31.16b, v17.16b, #12                 \n"  // r2 v19:
                                                                     // x 1 3 5
            "ld2 {v21.4s, v22.4s}, [%[din_ptr3]], #32           \n"  // r3 v21:
                                                                     // 0 2 4 6,
                                                                     // v22: 1 3
                                                                     // 5 7
            "ext v20.16b, v16.16b, v31.16b, #4                  \n"
            "ld1 {v20.s}[3], [%[din_ptr2]]                      \n"  // r2 v20:
                                                                     // 2 4 6 8
            "sub %[din_ptr2], %[din_ptr2], #8                   \n"
            "ext v23.16b, v31.16b, v21.16b, #12                 \n"  // r3 v23:
                                                                     // x 0 2 4
            "ld1 {v30.4s}, [%[weights]]                         \n"  // load
                                                                     // weights
                                                                     // 24
            "ext v24.16b, v31.16b, v22.16b, #12                 \n"  // r3 v24:
                                                                     // x 1 3 5
            "ld1 {v26.4s}, [%[vbias]]                           \n"  // load
                                                                     // bias to
                                                                     // out_r0
            "ext v25.16b, v21.16b, v31.16b, #4                  \n"
            "ld1 {v25.s}[3], [%[din_ptr3]]                      \n"  // r2 v25:
                                                                     // 2 4 6 8
            "sub %[din_ptr3], %[din_ptr3], #8                   \n"
            "mov v27.16b, v26.16b                               \n"  // load
                                                                     // bias to
                                                                     // out_r1
            "mov v28.16b, v31.16b                               \n"  // load
                                                                     // zero to
                                                                     // out_r0
            "mov v29.16b, v31.16b                               \n"  // load
                                                                     // zero to
                                                                     // out_r1

            "fmla v26.4s, v8.4s, v0.s[0]                        \n"  // out r0:
                                                                     // w0
            "fmla v28.4s, v9.4s, v0.s[1]                        \n"  // out r0:
                                                                     // w1
            "fmla v26.4s, v6.4s, v0.s[2]                        \n"  // out r0:
                                                                     // w2
            "fmla v28.4s, v7.4s, v0.s[3]                        \n"  // out r0:
                                                                     // w3

            "ld2 {v8.4s, v9.4s}, [%[din_ptr0]], %[s_8]          \n"  // next r0
                                                                     // v8: 0 2
                                                                     // 4 6, v9:
                                                                     // 1 3 5 7

            "fmla v26.4s, v10.4s, v1.s[0]                       \n"  // out r0:
                                                                     // w4
            "fmla v28.4s, v13.4s, v1.s[1]                       \n"  // out r0:
                                                                     // w5
            "fmla v26.4s, v14.4s, v1.s[2]                       \n"  // out r0:
                                                                     // w6
            "fmla v28.4s, v11.4s, v1.s[3]                       \n"  // out r0:
                                                                     // w7

            "ld2 {v6.4s, v7.4s}, [%[din_ptr0]], %[s_8]          \n"  // next r0
                                                                     // v6: 2 4
                                                                     // 6 8, v7:
                                                                     // 3 5 7 9

            "fmla v26.4s, v12.4s, v2.s[0]                       \n"  // out r0:
                                                                     // w8
            "fmla v28.4s, v15.4s, v2.s[1]                       \n"  // out r0:
                                                                     // w9
            "fmla v26.4s, v18.4s, v2.s[2]                       \n"  // out r0:
                                                                     // w10
            "fmla v28.4s, v19.4s, v2.s[3]                       \n"  // out r0:
                                                                     // w11

            "ld2 {v10.4s, v11.4s}, [%[din_ptr0]], %[s_16]       \n"  // next r0
                                                                     // v10: 4 6
                                                                     // 8 10,
                                                                     // v11:
                                                                     // trash
                                                                     // register

            "fmla v26.4s, v16.4s, v3.s[0]                       \n"  // out r0:
                                                                     // w12
            "fmla v28.4s, v17.4s, v3.s[1]                       \n"  // out r0:
                                                                     // w13
            "fmla v26.4s, v20.4s, v3.s[2]                       \n"  // out r0:
                                                                     // w14
            "fmla v28.4s, v23.4s, v3.s[3]                       \n"  // out r0:
                                                                     // w15
            "prfm pldl1keep, [%[din_ptr0]]                      \n"

            "ld2 {v11.4s, v12.4s}, [%[din_ptr4]], #32           \n"  // r4 v11:
                                                                     // 0 2 4 6,
                                                                     // v12: 1 3
                                                                     // 5 7

            "fmla v26.4s, v24.4s, v4.s[0]                       \n"  // out r0:
                                                                     // w16
            "fmla v28.4s, v21.4s, v4.s[1]                       \n"  // out r0:
                                                                     // w17

            "ext v13.16b, v31.16b, v11.16b, #12                 \n"  // r4 v13:
                                                                     // x 0 2 4
            "ext v14.16b, v31.16b, v12.16b, #12                 \n"  // r4 v14:
                                                                     // x 1 3 5
            "ext v15.16b, v11.16b, v31.16b, #4                  \n"

            "fmla v26.4s, v22.4s, v4.s[2]                       \n"  // out r0:
                                                                     // w18
            "fmla v28.4s, v25.4s, v4.s[3]                       \n"  // out r0:
                                                                     // w19

            "ld1 {v15.s}[3], [%[din_ptr4]]                      \n"  // r4 v15:
                                                                     // 2 4 6

            "fmla v27.4s, v18.4s, v0.s[0]                       \n"  // out r1:
                                                                     // w0
            "fmla v29.4s, v19.4s, v0.s[1]                       \n"  // out r1:
                                                                     // w1

            "sub %[din_ptr4], %[din_ptr4], #8                   \n"

            "fmla v27.4s, v16.4s, v0.s[2]                       \n"  // out r1:
                                                                     // w2
            "fmla v29.4s, v17.4s, v0.s[3]                       \n"  // out r1:
                                                                     // w3
            "fmla v27.4s, v20.4s, v1.s[0]                       \n"  // out r1:
                                                                     // w4
            "fmla v29.4s, v23.4s, v1.s[1]                       \n"  // out r1:
                                                                     // w5

            "ld2 {v16.4s, v17.4s}, [%[din_ptr5]], #32           \n"  // r5 v16:
                                                                     // 0 2 4 6,
                                                                     // v17: 1 3
                                                                     // 5 7

            "fmla v27.4s, v24.4s, v1.s[2]                       \n"  // out r1:
                                                                     // w6
            "fmla v29.4s, v21.4s, v1.s[3]                       \n"  // out r1:
                                                                     // w7

            "ext v18.16b, v31.16b, v16.16b, #12                 \n"  // r5 v18:
                                                                     // x 0 2 4
            "ext v19.16b, v31.16b, v17.16b, #12                 \n"  // r5 v19:
                                                                     // x 1 3 5
            "ext v20.16b, v16.16b, v31.16b, #4                  \n"

            "fmla v27.4s, v22.4s, v2.s[0]                       \n"  // out r1:
                                                                     // w8
            "fmla v29.4s, v25.4s, v2.s[1]                       \n"  // out r1:
                                                                     // w9

            "ld1 {v20.s}[3], [%[din_ptr5]]                      \n"  // r5 v20:
                                                                     // 2 4 6
            "ld2 {v21.4s, v22.4s}, [%[din_ptr6]], #32           \n"  // r6 v21:
                                                                     // 0 2 4 6,
                                                                     // v22: 1 3
                                                                     // 5 7

            "ext v23.16b, v31.16b, v21.16b, #12                 \n"  // r6 v23:
                                                                     // x 0 2 4
            "ext v24.16b, v31.16b, v22.16b, #12                 \n"  // r6 v24:
                                                                     // x 1 3 5
            "ext v25.16b, v21.16b, v31.16b, #4                  \n"
            "sub %[din_ptr5], %[din_ptr5], #8                   \n"

            "fmla v26.4s, v11.4s, v5.s[2]                       \n"  // out r0:
                                                                     // w22
            "fmla v28.4s, v12.4s, v5.s[3]                       \n"  // out r0:
                                                                     // w23

            "ld1 {v25.s}[3], [%[din_ptr6]]                      \n"  // r6 v25:
                                                                     // 2 4 6

            "fmla v26.4s, v13.4s, v5.s[0]                       \n"  // out r0:
                                                                     // w20
            "fmla v28.4s, v14.4s, v5.s[1]                       \n"  // out r0:
                                                                     // w21

            "sub %[din_ptr6], %[din_ptr6], #8                   \n"

            "fmla v26.4s, v15.4s, v30.s[0]                      \n"  // out r0:
                                                                     // w24
            "fmla v27.4s, v13.4s, v2.s[2]                       \n"  // out r1:
                                                                     // w10

            "fadd v26.4s, v26.4s, v28.4s                        \n"
            "fmla v29.4s, v14.4s, v2.s[3]                       \n"  // out r1:
                                                                     // w11
            "fmax v26.4s, v26.4s, v31.4s                        \n"

            "ld2 {v13.4s, v14.4s}, [%[din_ptr1]], %[s_8]        \n"  // next r1
                                                                     // v13: 0 2
                                                                     // 4 6,
                                                                     // v14: 1 3
                                                                     // 5 7
            "fmla v27.4s, v11.4s, v3.s[0]                       \n"  // out r1:
                                                                     // w12
            "fmla v29.4s, v12.4s, v3.s[1]                       \n"  // out r1:
                                                                     // w13

            "st1 {v26.4s}, [%[dout_ptr0]], %[s_16]              \n"  // store
                                                                     // output
                                                                     // r0
            "ld2 {v11.4s, v12.4s}, [%[din_ptr1]], %[s_8]        \n"  // next r1
                                                                     // v11: 2 4
                                                                     // 6 8,
                                                                     // v12: 3 5
                                                                     // 7 9

            "fmla v27.4s, v15.4s, v3.s[2]                       \n"  // out r1:
                                                                     // w14
            "fmla v29.4s, v16.4s, v4.s[1]                       \n"  // out r1:
                                                                     // w17
            "fmla v27.4s, v18.4s, v3.s[3]                       \n"  // out r1:
                                                                     // w15
            "fmla v29.4s, v19.4s, v4.s[0]                       \n"  // out r1:
                                                                     // w16

            "ld2 {v15.4s, v16.4s}, [%[din_ptr1]], %[s_16]       \n"  // next r1
                                                                     // v15: 4 6
                                                                     // 8 10,
                                                                     // v16:
                                                                     // trash
                                                                     // register

            "fmla v27.4s, v17.4s, v4.s[2]                       \n"  // out r1:
                                                                     // w18
            "fmla v29.4s, v20.4s, v4.s[3]                       \n"  // out r1:
                                                                     // w19

            "ld2 {v18.4s, v19.4s}, [%[din_ptr2]], %[s_8]        \n"  // next r2
                                                                     // v18: 0 2
                                                                     // 4 6,
                                                                     // v19: 1 3
                                                                     // 5 7
            "ld2 {v16.4s, v17.4s}, [%[din_ptr2]], %[s_8]        \n"  // next r2
                                                                     // v16: 2 4
                                                                     // 6 8,
                                                                     // v11: 3 5
                                                                     // 7 9

            "fmla v27.4s, v23.4s, v5.s[0]                       \n"  // out r1:
                                                                     // w20
            "fmla v29.4s, v21.4s, v5.s[2]                       \n"  // out r1:
                                                                     // w22
            "fmla v27.4s, v24.4s, v5.s[1]                       \n"  // out r1:
                                                                     // w21
            "fmla v29.4s, v22.4s, v5.s[3]                       \n"  // out r1:
                                                                     // w23

            "ld2 {v20.4s, v21.4s}, [%[din_ptr2]], %[s_16]       \n"  // next r2
                                                                     // v20: 4 6
                                                                     // 8 10,
                                                                     // v21:
                                                                     // trash
                                                                     // register
            "ld2 {v23.4s, v24.4s}, [%[din_ptr3]], %[s_8]        \n"  // next r3
                                                                     // v23: 0 2
                                                                     // 4 6,
                                                                     // v24: 1 3
                                                                     // 5 7

            "fmla v27.4s, v25.4s, v30.s[0]                      \n"  // out r1:
                                                                     // w24

            "ld2 {v21.4s, v22.4s}, [%[din_ptr3]], %[s_8]        \n"  // next r3
                                                                     // v21: 2 4
                                                                     // 6 8,
                                                                     // v22: 3 5
                                                                     // 7 9
            "ld2 {v25.4s, v26.4s}, [%[din_ptr3]], %[s_16]       \n"  // next r3
                                                                     // v25: 4 6
                                                                     // 8 10,
                                                                     // v26:
                                                                     // trash
                                                                     // register

            "fadd v27.4s, v27.4s, v29.4s                        \n"
            "fmax v27.4s, v27.4s, v31.4s                        \n"
            "cmp %w[mid_cnt], #1                                \n"
            "prfm pldl1keep, [%[din_ptr1]]                      \n"
            "prfm pldl1keep, [%[din_ptr2]]                      \n"
            "prfm pldl1keep, [%[din_ptr3]]                      \n"
            "st1 {v27.4s}, [%[dout_ptr1]], #16                  \n"
            "blt 2f                                             \n"

            // mid loop
            "1:                                                 \n"
            "ld1 {v26.4s}, [%[vbias]]                           \n"
            "mov v27.16b, v26.16b                               \n"
            "mov v28.16b, v31.16b                               \n"
            "mov v29.16b, v31.16b                               \n"

            // out_r0 r0-r3
            "fmla v26.4s, v8.4s, v0.s[0]                        \n"
            "fmla v28.4s, v9.4s, v0.s[1]                        \n"
            "fmla v26.4s, v6.4s, v0.s[2]                        \n"
            "fmla v28.4s, v7.4s, v0.s[3]                        \n"

            "ld2 {v8.4s, v9.4s}, [%[din_ptr0]], %[s_8]          \n"

            "fmla v26.4s, v10.4s, v1.s[0]                       \n"
            "fmla v28.4s, v11.4s, v1.s[3]                       \n"

            "ld2 {v6.4s, v7.4s}, [%[din_ptr0]], %[s_8]          \n"

            "fmla v26.4s, v14.4s, v1.s[2]                       \n"
            "fmla v28.4s, v13.4s, v1.s[1]                       \n"

            "ld2 {v10.4s, v11.4s}, [%[din_ptr0]], %[s_16]       \n"
            "prfm pldl1keep, [%[din_ptr0]]                      \n"

            "fmla v26.4s, v12.4s, v2.s[0]                       \n"
            "fmla v28.4s, v15.4s, v2.s[1]                       \n"

            "ld2 {v13.4s, v14.4s}, [%[din_ptr4]], %[s_8]        \n"

            "fmla v26.4s, v16.4s, v3.s[0]                       \n"
            "fmla v27.4s, v16.4s, v0.s[2]                       \n"

            "ld2 {v11.4s, v12.4s}, [%[din_ptr4]], %[s_8]        \n"

            "fmla v28.4s, v19.4s, v2.s[3]                       \n"
            "fmla v29.4s, v19.4s, v0.s[1]                       \n"

            "ld2 {v15.4s, v16.4s}, [%[din_ptr4]], %[s_16]       \n"
            "prfm pldl1keep, [%[din_ptr4]]                      \n"

            "fmla v26.4s, v18.4s, v2.s[2]                       \n"
            "fmla v27.4s, v18.4s, v0.s[0]                       \n"

            "fmla v28.4s, v17.4s, v3.s[1]                       \n"
            "fmla v29.4s, v17.4s, v0.s[3]                       \n"

            "ld2 {v18.4s, v19.4s}, [%[din_ptr5]], %[s_8]        \n"

            "fmla v26.4s, v20.4s, v3.s[2]                       \n"
            "fmla v27.4s, v20.4s, v1.s[0]                       \n"

            "ld2 {v16.4s, v17.4s}, [%[din_ptr5]], %[s_8]        \n"

            "fmla v29.4s, v21.4s, v1.s[3]                       \n"
            "fmla v28.4s, v21.4s, v4.s[1]                       \n"
            "fmla v28.4s, v23.4s, v3.s[3]                       \n"
            "fmla v29.4s, v23.4s, v1.s[1]                       \n"

            "ld2 {v20.4s, v21.4s}, [%[din_ptr5]], %[s_16]       \n"
            "prfm pldl1keep, [%[din_ptr5]]                      \n"

            "fmla v26.4s, v24.4s, v4.s[0]                       \n"
            "fmla v27.4s, v24.4s, v1.s[2]                       \n"

            "ld2 {v23.4s, v24.4s}, [%[din_ptr6]], %[s_8]        \n"

            "fmla v27.4s, v22.4s, v2.s[0]                       \n"
            "fmla v26.4s, v22.4s, v4.s[2]                       \n"

            "fmla v28.4s, v25.4s, v4.s[3]                       \n"
            "fmla v29.4s, v25.4s, v2.s[1]                       \n"

            "ld2 {v21.4s, v22.4s}, [%[din_ptr6]], %[s_8]        \n"
            "fadd v28.4s, v26.4s, v28.4s                        \n"

            "ld2 {v25.4s, v26.4s}, [%[din_ptr6]], %[s_16]       \n"
            "mov v26.16b, v31.16b                               \n"
            "prfm pldl1keep, [%[din_ptr6]]                      \n"

            "fmla v26.4s, v13.4s, v5.s[0]                       \n"
            "fmla v28.4s, v14.4s, v5.s[1]                       \n"
            "fmla v27.4s, v13.4s, v2.s[2]                       \n"
            "fmla v29.4s, v14.4s, v2.s[3]                       \n"

            "ld2 {v13.4s, v14.4s}, [%[din_ptr1]], %[s_8]        \n"

            "fmla v26.4s, v11.4s, v5.s[2]                       \n"
            "fmla v28.4s, v12.4s, v5.s[3]                       \n"
            "fmla v27.4s, v11.4s, v3.s[0]                       \n"
            "fmla v29.4s, v12.4s, v3.s[1]                       \n"

            "ld2 {v11.4s, v12.4s}, [%[din_ptr1]], %[s_8]        \n"

            "fmla v26.4s, v15.4s, v30.s[0]                      \n"
            "fmla v27.4s, v15.4s, v3.s[2]                       \n"
            "fmla v29.4s, v16.4s, v4.s[1]                       \n"
            "fmla v27.4s, v17.4s, v4.s[2]                       \n"

            "ld2 {v15.4s, v16.4s}, [%[din_ptr1]], %[s_16]       \n"
            "prfm pldl1keep, [%[din_ptr1]]                      \n"

            "fmla v29.4s, v18.4s, v3.s[3]                       \n"
            "fmla v27.4s, v19.4s, v4.s[0]                       \n"

            "ld2 {v18.4s, v19.4s}, [%[din_ptr2]], %[s_8]        \n"

            "fmla v29.4s, v20.4s, v4.s[3]                       \n"

            "ld2 {v16.4s, v17.4s}, [%[din_ptr2]], %[s_8]        \n"

            "fmla v27.4s, v23.4s, v5.s[0]                       \n"
            "fmla v27.4s, v21.4s, v5.s[2]                       \n"

            "ld2 {v20.4s, v21.4s}, [%[din_ptr2]], %[s_16]       \n"

            "fmla v29.4s, v24.4s, v5.s[1]                       \n"

            "ld2 {v23.4s, v24.4s}, [%[din_ptr3]], %[s_8]        \n"
            "prfm pldl1keep, [%[din_ptr2]]                      \n"

            "fmla v29.4s, v22.4s, v5.s[3]                       \n"

            "ld2 {v21.4s, v22.4s}, [%[din_ptr3]], %[s_8]        \n"

            "fmla v27.4s, v25.4s, v30.s[0]                      \n"

            "fadd v26.4s, v26.4s, v28.4s                        \n"
            "fadd v27.4s, v27.4s, v29.4s                        \n"
            "fmax v26.4s, v26.4s, v31.4s                        \n"
            "fmax v27.4s, v27.4s, v31.4s                        \n"

            "prfm pldl1keep, [%[din_ptr3]]                      \n"
            "st1 {v26.4s}, [%[dout_ptr0]], #16                  \n"
            "st1 {v27.4s}, [%[dout_ptr1]], #16                  \n"

            "ld2 {v25.4s, v26.4s}, [%[din_ptr3]], %[s_16]       \n"
            "subs %w[mid_cnt], %w[mid_cnt], #1                  \n"
            "bne 1b                                             \n"

            "2:                                                 \n"
            "ld2 {v26.4s, v27.4s}, [%[mask]], %[s_8]            \n"
            "ld2 {v28.4s, v29.4s}, [%[mask]], %[s_8]            \n"
            "bif v8.16b, v31.16b, v26.16b                       \n"
            "bif v9.16b, v31.16b, v27.16b                       \n"
            "bif v6.16b, v31.16b, v28.16b                       \n"
            "bif v7.16b, v31.16b, v29.16b                       \n"

            "bif v13.16b, v31.16b, v26.16b                      \n"
            "bif v14.16b, v31.16b, v27.16b                      \n"
            "bif v11.16b, v31.16b, v28.16b                      \n"
            "bif v12.16b, v31.16b, v29.16b                      \n"

            "bif v18.16b, v31.16b, v26.16b                      \n"
            "bif v19.16b, v31.16b, v27.16b                      \n"
            "bif v16.16b, v31.16b, v28.16b                      \n"
            "bif v17.16b, v31.16b, v29.16b                      \n"

            "bif v23.16b, v31.16b, v26.16b                      \n"
            "bif v24.16b, v31.16b, v27.16b                      \n"
            "bif v21.16b, v31.16b, v28.16b                      \n"
            "bif v22.16b, v31.16b, v29.16b                      \n"

            "ld2 {v28.4s, v29.4s}, [%[mask]]                    \n"
            "ld1 {v26.4s}, [%[vbias]]                           \n"
            "mov v29.16b, v31.16b                               \n"

            "bif v10.16b, v31.16b, v28.16b                      \n"
            "bif v15.16b, v31.16b, v28.16b                      \n"

            "mov v27.16b, v26.16b                               \n"

            "bif v20.16b, v31.16b, v28.16b                      \n"
            "bif v25.16b, v31.16b, v28.16b                      \n"
            "mov v28.16b, v31.16b                               \n"

            "fmla v26.4s, v8.4s, v0.s[0]                        \n"
            "fmla v28.4s, v9.4s, v0.s[1]                        \n"
            "fmla v26.4s, v6.4s, v0.s[2]                        \n"
            "fmla v28.4s, v7.4s, v0.s[3]                        \n"

            "fmla v26.4s, v10.4s, v1.s[0]                       \n"
            "fmla v28.4s, v13.4s, v1.s[1]                       \n"
            "fmla v26.4s, v14.4s, v1.s[2]                       \n"
            "fmla v28.4s, v11.4s, v1.s[3]                       \n"

            "sub %[mask], %[mask], #16                          \n"
            "ld2 {v6.4s, v7.4s}, [%[mask]], %[s_8]              \n"
            "ld2 {v8.4s, v9.4s}, [%[mask]], %[s_8]              \n"
            "ld2 {v10.4s, v11.4s}, [%[mask]]                    \n"

            "fmla v26.4s, v12.4s, v2.s[0]                       \n"
            "fmla v28.4s, v15.4s, v2.s[1]                       \n"

            "ld2 {v13.4s, v14.4s}, [%[din_ptr4]], %[s_8]        \n"

            "fmla v26.4s, v16.4s, v3.s[0]                       \n"
            "fmla v28.4s, v17.4s, v3.s[1]                       \n"

            "ld2 {v11.4s, v12.4s}, [%[din_ptr4]], %[s_8]        \n"

            "fmla v27.4s, v16.4s, v0.s[2]                       \n"
            "fmla v29.4s, v17.4s, v0.s[3]                       \n"

            "ld2 {v15.4s, v16.4s}, [%[din_ptr4]]                \n"

            "fmla v26.4s, v18.4s, v2.s[2]                       \n"
            "fmla v28.4s, v19.4s, v2.s[3]                       \n"
            "fmla v27.4s, v18.4s, v0.s[0]                       \n"
            "fmla v29.4s, v19.4s, v0.s[1]                       \n"

            "bif  v13.16b, v31.16b, v6.16b                      \n"
            "bif  v14.16b, v31.16b, v7.16b                      \n"
            "bif  v11.16b, v31.16b, v8.16b                      \n"
            "bif  v12.16b, v31.16b, v9.16b                      \n"
            "bif  v15.16b, v31.16b, v10.16b                     \n"

            "ld2 {v18.4s, v19.4s}, [%[din_ptr5]], %[s_8]        \n"

            "fmla v26.4s, v20.4s, v3.s[2]                       \n"
            "fmla v27.4s, v20.4s, v1.s[0]                       \n"

            "ld2 {v16.4s, v17.4s}, [%[din_ptr5]], %[s_8]        \n"

            "fmla v29.4s, v21.4s, v1.s[3]                       \n"
            "fmla v28.4s, v21.4s, v4.s[1]                       \n"

            "ld2 {v20.4s, v21.4s}, [%[din_ptr5]]                \n"

            "fmla v28.4s, v23.4s, v3.s[3]                       \n"
            "fmla v29.4s, v23.4s, v1.s[1]                       \n"
            "fmla v27.4s, v24.4s, v1.s[2]                       \n"
            "fmla v26.4s, v24.4s, v4.s[0]                       \n"

            "bif  v18.16b, v31.16b, v6.16b                      \n"
            "bif  v19.16b, v31.16b, v7.16b                      \n"
            "bif  v16.16b, v31.16b, v8.16b                      \n"
            "bif  v17.16b, v31.16b, v9.16b                      \n"
            "bif  v20.16b, v31.16b, v10.16b                     \n"

            "ld2 {v23.4s, v24.4s}, [%[din_ptr6]], %[s_8]        \n"

            "fmla v27.4s, v22.4s, v2.s[0]                       \n"
            "fmla v26.4s, v22.4s, v4.s[2]                       \n"

            "ld2 {v21.4s, v22.4s}, [%[din_ptr6]], %[s_8]        \n"

            "fmla v28.4s, v25.4s, v4.s[3]                       \n"
            "fmla v29.4s, v25.4s, v2.s[1]                       \n"
            "fadd v28.4s, v28.4s, v26.4s                        \n"

            "ld2 {v25.4s, v26.4s}, [%[din_ptr6]]                \n"
            "mov v26.16b, v31.16b                               \n"

            "bif  v23.16b, v31.16b, v6.16b                      \n"
            "bif  v24.16b, v31.16b, v7.16b                      \n"
            "bif  v21.16b, v31.16b, v8.16b                      \n"
            "bif  v22.16b, v31.16b, v9.16b                      \n"
            "bif  v25.16b, v31.16b, v10.16b                     \n"

            "fmla v26.4s, v13.4s, v5.s[0]                       \n"
            "fmla v28.4s, v14.4s, v5.s[1]                       \n"
            "fmla v26.4s, v11.4s, v5.s[2]                       \n"
            "fmla v28.4s, v12.4s, v5.s[3]                       \n"
            "fmla v26.4s, v15.4s, v30.s[0]                      \n"

            "fmla v27.4s, v13.4s, v2.s[2]                       \n"
            "fmla v29.4s, v14.4s, v2.s[3]                       \n"
            "fmla v27.4s, v11.4s, v3.s[0]                       \n"
            "fmla v29.4s, v12.4s, v3.s[1]                       \n"

            "fadd v26.4s, v26.4s, v28.4s                        \n"
            "fmla v27.4s, v15.4s, v3.s[2]                       \n"
            "fmla v29.4s, v18.4s, v3.s[3]                       \n"
            "fmla v27.4s, v19.4s, v4.s[0]                       \n"
            "fmla v29.4s, v16.4s, v4.s[1]                       \n"

            "fmax v26.4s, v26.4s, v31.4s                        \n"
            "fmla v27.4s, v17.4s, v4.s[2]                       \n"
            "fmla v29.4s, v20.4s, v4.s[3]                       \n"
            "fmla v27.4s, v23.4s, v5.s[0]                       \n"
            "fmla v29.4s, v24.4s, v5.s[1]                       \n"

            "st1 {v26.4s}, [%[out_buf0]]                        \n"
            "fmla v27.4s, v21.4s, v5.s[2]                       \n"
            "fmla v29.4s, v22.4s, v5.s[3]                       \n"
            "fmla v27.4s, v25.4s, v30.s[0]                      \n"
            "fadd v27.4s, v27.4s, v29.4s                        \n"

            "fmax v27.4s, v27.4s, v31.4s                        \n"
            "st1 {v27.4s}, [%[out_buf1]]                        \n"

            : [dout_ptr0] "+r"(dout_ptr0), [dout_ptr1] "+r"(dout_ptr1),
              [mid_cnt] "+r"(loop), [din_ptr0] "+r"(din_ptr0),
              [din_ptr1] "+r"(din_ptr1), [din_ptr2] "+r"(din_ptr2),
              [din_ptr3] "+r"(din_ptr3), [din_ptr4] "+r"(din_ptr4),
              [din_ptr5] "+r"(din_ptr5), [din_ptr6] "+r"(din_ptr6),
              [mask] "+r"(mask_ptr), [weights] "+r"(weights_ptr)
            : [vbias] "r"(vbias), [out_buf0] "r"(out_buf0),
              [out_buf1] "r"(out_buf1), [s_8] "r"(s_8), [s_16] "r"(s_16)
            : "memory", "cc", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7",
              "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16",
              "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25",
              "v26", "v27", "v28", "v29", "v30", "v31");

        int remain_cnt = w_out - (mid_cnt + 1) * 4;
        for (int i = 0; i < remain_cnt; ++i) {
          dout_ptr0[i] = out_buf0[i];
          dout_ptr1[i] = out_buf1[i];
        }
        din0 = din4;
        din1 = din5;
        din2 = din6;
        din3 = din6 + w_in;
        din4 = din3 + w_in;
        din5 = din4 + w_in;
        din6 = din5 + w_in;
        dout0 = dout1 + w_out;
        dout1 = dout0 + w_out;
      }
    }
  }
}

//! small depthwise, win < 9;
void conv_depthwise_5x5s2p2_s(const float* din, float* dout, int num,
                              int ch_out, int h_out, int w_out, int ch_in,
                              int h_in, int w_in, const float* weights,
                              const float* bias, bool flag_bias, bool flag_relu,
                              ARMContext* ctx) {
  CHECK_LT(w_in, 9) << "only support win < 9";
  int w_out_round = (w_out + 3) / 4 * 4;
  int mask_cnt = 12 - w_in - 2;
  int mask[12];
  memset(mask, 0xff, 12 * sizeof(int));
  for (int i = 0; i < mask_cnt; ++i) {
    mask[11 - i] = 0;
  }
  float* zero_ptr = ctx->workspace_data<float>();
  memset(zero_ptr, 0, w_in * sizeof(float));
  int in_spatial_size = w_in * h_in;
  int out_spatial_size = w_out * h_out;
  int weights_saptial_size = 25;

  for (int n = 0; n < num; ++n) {
    const float* din_batch = din + n * in_spatial_size * ch_in;
    float* dout_batch = dout + n * out_spatial_size * ch_out;
#pragma omp parallel for
    for (int c = 0; c < ch_in; ++c) {
      const float* din_ch = din_batch + c * in_spatial_size;
      float* dout_ch = dout_batch + c * out_spatial_size;
      const float* din0 = zero_ptr;
      const float* din1 = zero_ptr;
      const float* din2 = din_ch;
      const float* din3 = din2 + w_in;
      const float* din4 = din3 + w_in;

      float out_buf0[4];
      float out_buf1[4];
      float* dout0 = dout_ch;
      float* dout1 = dout0 + w_out;

      const float* weights_c = weights + c * weights_saptial_size;
      for (int h = 0; h < h_out; h += 1) {
        //! (h * 2 - 2) + 4 > h_in - 1
        if (h * 2 + 3 > h_in) {
          switch (h * 2 + 3 - h_in) {
            case 4:
              din1 = zero_ptr;
            case 3:
              din2 = zero_ptr;
            case 2:
              din3 = zero_ptr;
            case 1:
              din4 = zero_ptr;
            default:
              break;
          }
        }

        const float* din_ptr0 = din0;
        const float* din_ptr1 = din1;
        const float* din_ptr2 = din2;
        const float* din_ptr3 = din3;
        const float* din_ptr4 = din4;

        const float* weights_ptr = weights_c;
        float* dout_ptr0 = dout0;

        float bias_c = 0.f;
        if (flag_bias) {
          bias_c = bias[c];
        }
        float vbias[4] = {bias_c, bias_c, bias_c, bias_c};
        int* mask_ptr = mask;
        const int s_8 = 8;
        //! in r0/r4, r1, r2, r3: x 0 2 4 -- v8   v13  v18  v23  v28
        //! in r0/r4, r1, r2, r3: x 1 3 5 -- v9   v14  v19  v24  v29
        //! in r0/r4, r1, r2, r3: 0 2 4 6 -- v6   v11  v16  v21  v26
        //! in r0/r4, r1, r2, r3: 1 3 5 7 -- v7   v12  v17  v22  v27
        //! in r0/r4, r1, r2, r3: 2 4 6 8 -- v10  v15  v20  v25  v30
        //! out r0 -- v4
        asm volatile(
            "movi   v31.4s, #0x0\n"
            "prfm pldl1keep, [%[din_ptr0]]  \n"
            "prfm pldl1keep, [%[din_ptr1]]  \n"
            "prfm pldl1keep, [%[din_ptr2]]  \n"
            "prfm pldl1keep, [%[din_ptr3]]  \n"
            "prfm pldl1keep, [%[din_ptr4]]  \n"
            "prfm pldl1keep, [%[weights]]   \n"
            "prfm pldl1keep, [%[mask]]      \n"

            //! load mask
            "ld2 {v0.4s, v1.4s}, [%[mask]], %[s_8]  \n"
            "ld2 {v2.4s, v3.4s}, [%[mask]], %[s_8]  \n"
            "ld2 {v4.4s, v5.4s}, [%[mask]]  \n"

            //! load and extract input
            "ld2 {v6.4s, v7.4s},   [%[din_ptr0]], #32  \n"
            "ld2 {v11.4s, v12.4s}, [%[din_ptr1]], #32 \n"
            "ld2 {v16.4s, v17.4s}, [%[din_ptr2]], #32 \n"
            "ld2 {v21.4s, v22.4s}, [%[din_ptr3]], #32 \n"
            "ld2 {v26.4s, v27.4s}, [%[din_ptr4]], #32 \n"

            "ext v8.16b, v31.16b, v6.16b, #12  \n"
            "ext v9.16b, v31.16b, v7.16b, #12  \n"
            "ext v13.16b, v31.16b, v11.16b, #12  \n"
            "ext v14.16b, v31.16b, v12.16b, #12  \n"

            "ext v18.16b, v31.16b, v16.16b, #12  \n"
            "ext v19.16b, v31.16b, v17.16b, #12  \n"
            "ext v23.16b, v31.16b, v21.16b, #12  \n"
            "ext v24.16b, v31.16b, v22.16b, #12  \n"
            "ext v28.16b, v31.16b, v26.16b, #12  \n"
            "ext v29.16b, v31.16b, v27.16b, #12  \n"

            "ext v10.16b, v6.16b,  v31.16b, #4  \n"
            "ext v15.16b, v11.16b, v31.16b, #4  \n"
            "ext v20.16b, v16.16b, v31.16b, #4  \n"
            "ext v25.16b, v21.16b, v31.16b, #4  \n"
            "ext v30.16b, v26.16b, v31.16b, #4  \n"

            "bif v8.16b, v31.16b, v0.16b  \n"
            "bif v9.16b, v31.16b, v1.16b  \n"
            "bif v6.16b, v31.16b, v2.16b  \n"
            "bif v7.16b, v31.16b, v3.16b  \n"

            "bif v13.16b, v31.16b, v0.16b  \n"
            "bif v14.16b, v31.16b, v1.16b  \n"
            "bif v11.16b, v31.16b, v2.16b  \n"
            "bif v12.16b, v31.16b, v3.16b  \n"

            "bif v18.16b, v31.16b, v0.16b  \n"
            "bif v19.16b, v31.16b, v1.16b  \n"
            "bif v16.16b, v31.16b, v2.16b  \n"
            "bif v17.16b, v31.16b, v3.16b  \n"

            "ld1 {v10.s}[3], [%[din_ptr0]]  \n"
            "ld1 {v15.s}[3], [%[din_ptr1]]  \n"
            "ld1 {v20.s}[3], [%[din_ptr2]]  \n"
            "ld1 {v25.s}[3], [%[din_ptr3]]  \n"
            "ld1 {v30.s}[3], [%[din_ptr4]]  \n"

            "bif v23.16b, v31.16b, v0.16b  \n"
            "bif v24.16b, v31.16b, v1.16b  \n"
            "bif v21.16b, v31.16b, v2.16b  \n"
            "bif v22.16b, v31.16b, v3.16b  \n"

            "bif v28.16b, v31.16b, v0.16b  \n"
            "bif v29.16b, v31.16b, v1.16b  \n"
            "bif v26.16b, v31.16b, v2.16b  \n"
            "bif v27.16b, v31.16b, v3.16b  \n"

            "bif v10.16b, v31.16b, v4.16b  \n"
            "bif v15.16b, v31.16b, v4.16b  \n"
            "bif v20.16b, v31.16b, v4.16b  \n"
            "bif v25.16b, v31.16b, v4.16b  \n"
            "bif v30.16b, v31.16b, v4.16b  \n"

            "ld1 {v4.4s}, [%[vbias]]  \n"
            "mov v5.16b, v31.16b  \n"

            "ld1 {v0.4s, v1.4s}, [%[weights]], #32  \n"  // load weights 0-7
            "ld1 {v2.4s, v3.4s}, [%[weights]], #32  \n"  // load weights 8-15

            //! compute
            "fmla v4.4s, v8.4s, v0.s[0]  \n"  // out r0: w0
            "fmla v5.4s, v9.4s, v0.s[1]  \n"  // out r0: w1
            "fmla v4.4s, v6.4s, v0.s[2]  \n"  // out r0: w2
            "fmla v5.4s, v7.4s, v0.s[3]  \n"  // out r0: w3

            "fmla v4.4s, v10.4s, v1.s[0]  \n"  // out r0: w4
            "fmla v5.4s, v13.4s, v1.s[1]  \n"  // out r0: w5
            "fmla v4.4s, v14.4s, v1.s[2]  \n"  // out r0: w6
            "fmla v5.4s, v11.4s, v1.s[3]  \n"  // out r0: w7

            "ld1 {v6.4s, v7.4s}, [%[weights]], #32  \n"  // load weights 16-23
            "ld1 {v8.s}[0], [%[weights]]  \n"            // load weights 24

            "fmla v4.4s, v12.4s, v2.s[0]  \n"  // out r0: w8
            "fmla v5.4s, v15.4s, v2.s[1]  \n"  // out r0: w9
            "fmla v4.4s, v18.4s, v2.s[2]  \n"  // out r0: w10
            "fmla v5.4s, v19.4s, v2.s[3]  \n"  // out r0: w11

            "fmla v4.4s, v16.4s, v3.s[0]  \n"  // out r0: w12
            "fmla v5.4s, v17.4s, v3.s[1]  \n"  // out r0: w13
            "fmla v4.4s, v20.4s, v3.s[2]  \n"  // out r0: w14
            "fmla v5.4s, v23.4s, v3.s[3]  \n"  // out r0: w15

            "fmla v4.4s, v24.4s, v6.s[0]  \n"  // out r0: w16
            "fmla v5.4s, v21.4s, v6.s[1]  \n"  // out r0: w17
            "fmla v4.4s, v22.4s, v6.s[2]  \n"  // out r0: w18
            "fmla v5.4s, v25.4s, v6.s[3]  \n"  // out r0: w19

            "fmla v4.4s, v28.4s, v7.s[0]  \n"  // out r0: w20
            "fmla v5.4s, v29.4s, v7.s[1]  \n"  // out r0: w21
            "fmla v4.4s, v26.4s, v7.s[2]  \n"  // out r0: w22
            "fmla v5.4s, v27.4s, v7.s[3]  \n"  // out r0: w23
            "fmla v4.4s, v30.4s, v8.s[0] \n"   // out r0: w24

            "fadd v4.4s, v4.4s, v5.4s  \n"  // add out to v4
            "st1 {v4.4s}, [%[out_buf0]]  \n"

            : [dout_ptr0] "+r"(dout_ptr0), [din_ptr0] "+r"(din_ptr0),
              [din_ptr1] "+r"(din_ptr1), [din_ptr2] "+r"(din_ptr2),
              [din_ptr3] "+r"(din_ptr3), [din_ptr4] "+r"(din_ptr4),
              [mask] "+r"(mask_ptr), [weights] "+r"(weights_ptr)
            : [vbias] "r"(vbias), [out_buf0] "r"(out_buf0),
              [out_buf1] "r"(out_buf1), [s_8] "r"(s_8)
            : "memory", "cc", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7",
              "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16",
              "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25",
              "v26", "v27", "v28", "v29", "v30", "v31");
        for (int i = 0; i < w_out; ++i) {
          dout_ptr0[i] = out_buf0[i];
        }
        din0 = din2;
        din1 = din3;
        din2 = din4;
        din3 = din2 + w_in;
        din4 = din3 + w_in;
        dout0 += w_out;
      }
    }
  }
}

//! small depthwise, win < 9;
void conv_depthwise_5x5s2p2_relu_s(const float* din, float* dout, int num,
                                   int ch_out, int h_out, int w_out, int ch_in,
                                   int h_in, int w_in, const float* weights,
                                   const float* bias, bool flag_bias,
                                   bool flag_relu, ARMContext* ctx) {
  CHECK_LT(w_in, 9) << "only support win < 9";
  int w_out_round = (w_out + 3) / 4 * 4;
  int mask_cnt = 12 - w_in - 2;
  int mask[12];
  memset(mask, 0xff, 12 * sizeof(int));
  for (int i = 0; i < mask_cnt; ++i) {
    mask[11 - i] = 0;
  }
  float* zero_ptr = ctx->workspace_data<float>();
  memset(zero_ptr, 0, w_in * sizeof(float));
  int in_spatial_size = w_in * h_in;
  int out_spatial_size = w_out * h_out;
  int weights_saptial_size = 25;

  for (int n = 0; n < num; ++n) {
    const float* din_batch = din + n * in_spatial_size * ch_in;
    float* dout_batch = dout + n * out_spatial_size * ch_out;
#pragma omp parallel for
    for (int c = 0; c < ch_in; ++c) {
      const float* din_ch = din_batch + c * in_spatial_size;
      float* dout_ch = dout_batch + c * out_spatial_size;
      const float* din0 = zero_ptr;
      const float* din1 = zero_ptr;
      const float* din2 = din_ch;
      const float* din3 = din2 + w_in;
      const float* din4 = din3 + w_in;

      float out_buf0[4];
      float out_buf1[4];
      float* dout0 = dout_ch;
      float* dout1 = dout0 + w_out;

      const float* weights_c = weights + c * weights_saptial_size;
      for (int h = 0; h < h_out; h += 1) {
        //! (h * 2 - 2) + 4 > h_in - 1
        if (h * 2 + 3 > h_in) {
          switch (h * 2 + 3 - h_in) {
            case 4:
              din1 = zero_ptr;
            case 3:
              din2 = zero_ptr;
            case 2:
              din3 = zero_ptr;
            case 1:
              din4 = zero_ptr;
            default:
              break;
          }
        }
        const float* din_ptr0 = din0;
        const float* din_ptr1 = din1;
        const float* din_ptr2 = din2;
        const float* din_ptr3 = din3;
        const float* din_ptr4 = din4;

        const float* weights_ptr = weights_c;
        float* dout_ptr0 = dout0;

        float bias_c = 0.f;
        if (flag_bias) {
          bias_c = bias[c];
        }
        float vbias[4] = {bias_c, bias_c, bias_c, bias_c};
        int* mask_ptr = mask;
        const int s_8 = 8;
        //! in r0/r4, r1, r2, r3: x 0 2 4 -- v8   v13  v18  v23  v28
        //! in r0/r4, r1, r2, r3: x 1 3 5 -- v9   v14  v19  v24  v29
        //! in r0/r4, r1, r2, r3: 0 2 4 6 -- v6   v11  v16  v21  v26
        //! in r0/r4, r1, r2, r3: 1 3 5 7 -- v7   v12  v17  v22  v27
        //! in r0/r4, r1, r2, r3: 2 4 6 8 -- v10  v15  v20  v25  v30
        //! out r0 -- v4
        asm volatile(
            "movi   v31.4s, #0x0\n"
            "prfm pldl1keep, [%[din_ptr0]]  \n"
            "prfm pldl1keep, [%[din_ptr1]]  \n"
            "prfm pldl1keep, [%[din_ptr2]]  \n"
            "prfm pldl1keep, [%[din_ptr3]]  \n"
            "prfm pldl1keep, [%[din_ptr4]]  \n"
            "prfm pldl1keep, [%[weights]]   \n"
            "prfm pldl1keep, [%[mask]]      \n"

            //! load mask
            "ld2 {v0.4s, v1.4s}, [%[mask]], %[s_8]  \n"
            "ld2 {v2.4s, v3.4s}, [%[mask]], %[s_8]  \n"
            "ld2 {v4.4s, v5.4s}, [%[mask]]  \n"

            //! load and extract input
            "ld2 {v6.4s, v7.4s},   [%[din_ptr0]], #32  \n"
            "ld2 {v11.4s, v12.4s}, [%[din_ptr1]], #32 \n"
            "ld2 {v16.4s, v17.4s}, [%[din_ptr2]], #32 \n"
            "ld2 {v21.4s, v22.4s}, [%[din_ptr3]], #32 \n"
            "ld2 {v26.4s, v27.4s}, [%[din_ptr4]], #32 \n"

            "ext v8.16b, v31.16b, v6.16b, #12  \n"
            "ext v9.16b, v31.16b, v7.16b, #12  \n"
            "ext v13.16b, v31.16b, v11.16b, #12  \n"
            "ext v14.16b, v31.16b, v12.16b, #12  \n"

            "ext v18.16b, v31.16b, v16.16b, #12  \n"
            "ext v19.16b, v31.16b, v17.16b, #12  \n"
            "ext v23.16b, v31.16b, v21.16b, #12  \n"
            "ext v24.16b, v31.16b, v22.16b, #12  \n"
            "ext v28.16b, v31.16b, v26.16b, #12  \n"
            "ext v29.16b, v31.16b, v27.16b, #12  \n"

            "ext v10.16b, v6.16b,  v31.16b, #4  \n"
            "ext v15.16b, v11.16b, v31.16b, #4  \n"
            "ext v20.16b, v16.16b, v31.16b, #4  \n"
            "ext v25.16b, v21.16b, v31.16b, #4  \n"
            "ext v30.16b, v26.16b, v31.16b, #4  \n"

            "bif v8.16b, v31.16b, v0.16b  \n"
            "bif v9.16b, v31.16b, v1.16b  \n"
            "bif v6.16b, v31.16b, v2.16b  \n"
            "bif v7.16b, v31.16b, v3.16b  \n"

            "bif v13.16b, v31.16b, v0.16b  \n"
            "bif v14.16b, v31.16b, v1.16b  \n"
            "bif v11.16b, v31.16b, v2.16b  \n"
            "bif v12.16b, v31.16b, v3.16b  \n"

            "bif v18.16b, v31.16b, v0.16b  \n"
            "bif v19.16b, v31.16b, v1.16b  \n"
            "bif v16.16b, v31.16b, v2.16b  \n"
            "bif v17.16b, v31.16b, v3.16b  \n"

            "ld1 {v10.s}[3], [%[din_ptr0]]  \n"
            "ld1 {v15.s}[3], [%[din_ptr1]]  \n"
            "ld1 {v20.s}[3], [%[din_ptr2]]  \n"
            "ld1 {v25.s}[3], [%[din_ptr3]]  \n"
            "ld1 {v30.s}[3], [%[din_ptr4]]  \n"

            "bif v23.16b, v31.16b, v0.16b  \n"
            "bif v24.16b, v31.16b, v1.16b  \n"
            "bif v21.16b, v31.16b, v2.16b  \n"
            "bif v22.16b, v31.16b, v3.16b  \n"

            "bif v28.16b, v31.16b, v0.16b  \n"
            "bif v29.16b, v31.16b, v1.16b  \n"
            "bif v26.16b, v31.16b, v2.16b  \n"
            "bif v27.16b, v31.16b, v3.16b  \n"

            "bif v10.16b, v31.16b, v4.16b  \n"
            "bif v15.16b, v31.16b, v4.16b  \n"
            "bif v20.16b, v31.16b, v4.16b  \n"
            "bif v25.16b, v31.16b, v4.16b  \n"
            "bif v30.16b, v31.16b, v4.16b  \n"

            "ld1 {v4.4s}, [%[vbias]]  \n"
            "mov v5.16b, v31.16b  \n"

            "ld1 {v0.4s, v1.4s}, [%[weights]], #32  \n"  // load weights 0-7
            "ld1 {v2.4s, v3.4s}, [%[weights]], #32  \n"  // load weights 8-15

            //! compute
            "fmla v4.4s, v8.4s, v0.s[0]  \n"  // out r0: w0
            "fmla v5.4s, v9.4s, v0.s[1]  \n"  // out r0: w1
            "fmla v4.4s, v6.4s, v0.s[2]  \n"  // out r0: w2
            "fmla v5.4s, v7.4s, v0.s[3]  \n"  // out r0: w3

            "fmla v4.4s, v10.4s, v1.s[0]  \n"  // out r0: w4
            "fmla v5.4s, v13.4s, v1.s[1]  \n"  // out r0: w5
            "fmla v4.4s, v14.4s, v1.s[2]  \n"  // out r0: w6
            "fmla v5.4s, v11.4s, v1.s[3]  \n"  // out r0: w7

            "ld1 {v6.4s, v7.4s}, [%[weights]], #32  \n"  // load weights 16-23
            "ld1 {v8.s}[0], [%[weights]]  \n"            // load weights 24

            "fmla v4.4s, v12.4s, v2.s[0]  \n"  // out r0: w8
            "fmla v5.4s, v15.4s, v2.s[1]  \n"  // out r0: w9
            "fmla v4.4s, v18.4s, v2.s[2]  \n"  // out r0: w10
            "fmla v5.4s, v19.4s, v2.s[3]  \n"  // out r0: w11

            "fmla v4.4s, v16.4s, v3.s[0]  \n"  // out r0: w12
            "fmla v5.4s, v17.4s, v3.s[1]  \n"  // out r0: w13
            "fmla v4.4s, v20.4s, v3.s[2]  \n"  // out r0: w14
            "fmla v5.4s, v23.4s, v3.s[3]  \n"  // out r0: w15

            "fmla v4.4s, v24.4s, v6.s[0]  \n"  // out r0: w16
            "fmla v5.4s, v21.4s, v6.s[1]  \n"  // out r0: w17
            "fmla v4.4s, v22.4s, v6.s[2]  \n"  // out r0: w18
            "fmla v5.4s, v25.4s, v6.s[3]  \n"  // out r0: w19

            "fmla v4.4s, v28.4s, v7.s[0]  \n"  // out r0: w20
            "fmla v5.4s, v29.4s, v7.s[1]  \n"  // out r0: w21
            "fmla v4.4s, v26.4s, v7.s[2]  \n"  // out r0: w22
            "fmla v5.4s, v27.4s, v7.s[3]  \n"  // out r0: w23
            "fmla v4.4s, v30.4s, v8.s[0]  \n"  // out r0: w24

            "fadd v4.4s, v4.4s, v5.4s     \n"  // add out to v4
            "fmax v4.4s, v4.4s, v31.4s    \n"
            "st1 {v4.4s}, [%[out_buf0]]   \n"

            : [dout_ptr0] "+r"(dout_ptr0), [din_ptr0] "+r"(din_ptr0),
              [din_ptr1] "+r"(din_ptr1), [din_ptr2] "+r"(din_ptr2),
              [din_ptr3] "+r"(din_ptr3), [din_ptr4] "+r"(din_ptr4),
              [mask] "+r"(mask_ptr), [weights] "+r"(weights_ptr)
            : [vbias] "r"(vbias), [out_buf0] "r"(out_buf0),
              [out_buf1] "r"(out_buf1), [s_8] "r"(s_8)
            : "memory", "cc", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7",
              "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16",
              "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25",
              "v26", "v27", "v28", "v29", "v30", "v31");
        for (int i = 0; i < w_out; ++i) {
          dout_ptr0[i] = out_buf0[i];
        }
        din0 = din2;
        din1 = din3;
        din2 = din4;
        din3 = din2 + w_in;
        din4 = din3 + w_in;
        dout0 += w_out;
      }
    }
  }
}

#else

//! larger depthwise, win >= 9;
void conv_depthwise_5x5s2p2(const float* din, float* dout, int num, int ch_out,
                            int h_out, int w_out, int ch_in, int h_in, int w_in,
                            const float* weights, const float* bias,
                            bool flag_bias, bool flag_relu, ARMContext* ctx) {
  // printf("invoke 5x5s2p2 armv7\n");
  CHECK_GE(w_in, 9) << "only support win >= 9";
  int w_out_round = (w_out + 3) / 4 * 4;
  int cnt = (w_out_round - 4) / 4;
  int mid_cnt = cnt - 1;
  int right_start = cnt * 2 * 4 - 2;
  int mask_cnt = 12 - (w_in - right_start);
  int mask[12];
  memset(mask, 0xff, 12 * sizeof(int));
  for (int i = 0; i < mask_cnt; ++i) {
    mask[11 - i] = 0;
  }
  float* zero_ptr = ctx->workspace_data<float>();
  memset(zero_ptr, 0, w_in * sizeof(float));
  int in_spatial_size = w_in * h_in;
  int out_spatial_size = w_out * h_out;
  int weights_saptial_size = 25;

  for (int n = 0; n < num; ++n) {
    const float* din_batch = din + n * in_spatial_size * ch_in;
    float* dout_batch = dout + n * out_spatial_size * ch_out;
#pragma omp parallel for
    for (int c = 0; c < ch_in; ++c) {
      const float* din_ch = din_batch + c * in_spatial_size;
      float* dout_ch = dout_batch + c * out_spatial_size;
      const float* din0 = zero_ptr;
      const float* din1 = zero_ptr;
      const float* din2 = din_ch;
      const float* din3 = din2 + w_in;
      const float* din4 = din3 + w_in;

      float out_buf0[4];
      float* dout0 = dout_ch;

      const float* weights_c = weights + c * weights_saptial_size;
      float32x4_t w0 = vld1q_f32(weights_c);
      float32x4_t w1 = vld1q_f32(weights_c + 4);
      float32x4_t w2 = vld1q_f32(weights_c + 8);
      float32x4_t w3 = vld1q_f32(weights_c + 12);
      float32x4_t w4 = vld1q_f32(weights_c + 16);
      float32x4_t w5 = vld1q_f32(weights_c + 20);
      for (int h = 0; h < h_out; h += 1) {
        //! (h * 2 - 2) + 4 > h_in - 1
        if (h * 2 + 3 > h_in) {
          switch (h * 2 + 3 - h_in) {
            case 4:
              din1 = zero_ptr;
            case 3:
              din2 = zero_ptr;
            case 2:
              din3 = zero_ptr;
            case 1:
              din4 = zero_ptr;
            default:
              break;
          }
        }
        const float* din_ptr0 = din0;
        const float* din_ptr1 = din1;
        const float* din_ptr2 = din2;
        const float* din_ptr3 = din3;
        const float* din_ptr4 = din4;

        const float* weights_ptr = weights_c + 24;
        float* dout_ptr0 = dout0;

        float bias_c = 0.f;
        if (flag_bias) {
          bias_c = bias[c];
        }
        float vbias[4] = {bias_c, bias_c, bias_c, bias_c};
        int* mask_ptr = mask;
        int loop = mid_cnt;
        const int s_8 = 8;
        const int s_16 = 16;

        asm volatile(
            "vmov.i32   q15, #0x0           \n"
            "pld [%[din_ptr0]]              \n"
            "pld [%[din_ptr1]]              \n"
            "pld [%[din_ptr2]]              \n"
            "pld [%[din_ptr3]]              \n"
            "pld [%[din_ptr4]]              \n"
            "pld [%[mask]]                  \n"

            // left
            "vld2.32 {d16-d19}, [%[din_ptr0]]!          \n"
            "vld1.32 {d26-d29}, [%[vbias]]              \n"
            "vext.32 q6, q15, q8, #3                    \n"
            "vext.32 q7, q15, q9, #3                    \n"
            "vext.32 q10, q8, q15, #1                   \n"
            "vmov.32 q14, q15                           \n"

            // r0
            "vmla.f32 q13, q8, %f[w0][0]                \n"
            "vmla.f32 q14, q9, %f[w0][1]                \n"

            "vld1.32 {d21[1]}, [%[din_ptr0]]            \n"
            "vld2.32 {d16-d19}, [%[din_ptr1]]!          \n"
            "sub %[din_ptr0], #8  \n"

            "vmla.f32 q13, q6, %e[w0][0]                \n"
            "vmla.f32 q14, q7, %e[w0][1]                \n"
            "vmla.f32 q13, q10, %e[w1][0]               \n"

            "vext.32 q6, q15, q8, #3                    \n"
            "vext.32 q7, q15, q9, #3                    \n"
            "vext.32 q10, q8, q15, #1                   \n"

            // r1
            "vmla.f32 q13, q8, %f[w1][1]                \n"
            "vmla.f32 q14, q9, %e[w2][0]                \n"

            "vld1.32 {d21[1]}, [%[din_ptr1]]            \n"
            "vld2.32 {d16-d19}, [%[din_ptr2]]!          \n"
            "sub %[din_ptr1], #8                        \n"

            "vmla.f32 q13, q6, %e[w1][1]                \n"
            "vmla.f32 q14, q7, %f[w1][0]                \n"
            "vmla.f32 q13, q10, %e[w2][1]               \n"

            "vext.32 q6, q15, q8, #3                    \n"
            "vext.32 q7, q15, q9, #3                    \n"
            "vext.32 q10, q8, q15, #1                   \n"

            // r2
            "vmla.f32 q13, q8, %e[w3][0]                \n"
            "vmla.f32 q14, q9, %e[w3][1]                \n"

            "vld1.32 {d21[1]}, [%[din_ptr2]]            \n"
            "vld2.32 {d16-d19}, [%[din_ptr3]]!          \n"
            "sub %[din_ptr2], #8                        \n"

            "vmla.f32 q13, q6, %f[w2][0]                \n"
            "vmla.f32 q14, q7, %f[w2][1]                \n"
            "vmla.f32 q13, q10, %f[w3][0]               \n"

            "vext.32 q6, q15, q8, #3                    \n"
            "vext.32 q7, q15, q9, #3                    \n"
            "vext.32 q10, q8, q15, #1                   \n"

            // r3
            "vmla.f32 q13, q8, %e[w4][1]                \n"
            "vmla.f32 q14, q9, %f[w4][0]                \n"

            "vld1.32 {d21[1]}, [%[din_ptr3]]            \n"
            "vld2.32 {d16-d19}, [%[din_ptr4]]!          \n"
            "sub %[din_ptr3], #8                        \n"

            "vmla.f32 q13, q6, %f[w3][1]                \n"
            "vmla.f32 q14, q7, %e[w4][0]                \n"
            "vmla.f32 q13, q10, %f[w4][1]               \n"

            "vext.32 q6, q15, q8, #3                    \n"
            "vext.32 q7, q15, q9, #3                    \n"
            "vext.32 q10, q8, q15, #1                   \n"

            // r4
            "vmla.f32 q13, q6, %e[w5][0]                \n"
            "vmla.f32 q14, q7, %e[w5][1]                \n"

            "vld1.32 {d21[1]}, [%[din_ptr4]]            \n"
            "vld2.32 {d12-d15}, [%[din_ptr0]], %[s_8]   \n"
            "sub %[din_ptr4], #8                        \n"

            "vmla.f32 q13, q8, %f[w5][0]                \n"
            "vmla.f32 q14, q9, %f[w5][1]                \n"

            "vld2.32 {d16-d19}, [%[din_ptr0]], %[s_8]   \n"

            "vmov.32 q12, %q[w0]                        \n"
            "vld1.32 {%e[w0][0]}, [%[weights]]          \n"
            "vmla.f32 q13, q10, %e[w0][0]               \n"
            "vadd.f32 q13, q13, q14                     \n"
            "vmov.32 %q[w0], q12                        \n"
            "cmp %[mid_cnt], #1                         \n"
            "vld2.32 {d20-d23}, [%[din_ptr0]], %[s_16]  \n"
            "vst1.32 {d26-d27}, [%[dout_ptr0]]!         \n"
            "pld [%[din_ptr0]]                          \n"
            "blt 2f                                     \n"

            // mid
            "1:                                         \n"
            "vld1.32 {d26-d27}, [%[vbias]]              \n"
            "vmov.32 q14, q15                           \n"

            // r0
            "vmla.f32 q13, q6,  %e[w0][0]               \n"
            "vmla.f32 q14, q7,  %e[w0][1]               \n"

            "vld2.32 {d12-d15}, [%[din_ptr1]], %[s_8]   \n"

            "vmla.f32 q13, q8,  %f[w0][0]               \n"
            "vmla.f32 q14, q9,  %f[w0][1]               \n"

            "vld2.32 {d16-d19}, [%[din_ptr1]], %[s_8]   \n"

            "vmla.f32 q13, q10, %e[w1][0]               \n"

            "vld2.32 {d20-d23}, [%[din_ptr1]], %[s_16]  \n"

            // r1
            "vmla.f32 q13, q6,  %e[w1][1]               \n"
            "vmla.f32 q14, q7,  %f[w1][0]               \n"
            "pld [%[din_ptr1]]                          \n"

            "vld2.32 {d12-d15}, [%[din_ptr2]], %[s_8]   \n"

            "vmla.f32 q13, q8,  %f[w1][1]               \n"
            "vmla.f32 q14, q9,  %e[w2][0]               \n"

            "vld2.32 {d16-d19}, [%[din_ptr2]], %[s_8]   \n"

            "vmla.f32 q13, q10, %e[w2][1]               \n"

            "vld2.32 {d20-d23}, [%[din_ptr2]], %[s_16]  \n"

            // r2
            "vmla.f32 q13, q6,  %f[w2][0]               \n"
            "vmla.f32 q14, q7,  %f[w2][1]               \n"
            "pld [%[din_ptr2]]                          \n"

            "vld2.32 {d12-d15}, [%[din_ptr3]], %[s_8]   \n"

            "vmla.f32 q13, q8,  %e[w3][0]               \n"
            "vmla.f32 q14, q9,  %e[w3][1]               \n"

            "vld2.32 {d16-d19}, [%[din_ptr3]], %[s_8]   \n"

            "vmla.f32 q13, q10, %f[w3][0]               \n"

            "vld2.32 {d20-d23}, [%[din_ptr3]], %[s_16]  \n"

            // r3
            "vmla.f32 q13, q6,  %f[w3][1]               \n"
            "vmla.f32 q14, q7,  %e[w4][0]               \n"
            "pld [%[din_ptr3]]                          \n"

            "vld2.32 {d12-d15}, [%[din_ptr4]], %[s_8]   \n"

            "vmla.f32 q13, q8,  %e[w4][1]               \n"
            "vmla.f32 q14, q9,  %f[w4][0]               \n"

            "vld2.32 {d16-d19}, [%[din_ptr4]], %[s_8]   \n"

            "vmla.f32 q13, q10, %f[w4][1]               \n"

            "vld2.32 {d20-d23}, [%[din_ptr4]], %[s_16]  \n"

            // r4
            "vmla.f32 q13, q6,  %e[w5][0]               \n"
            "vmla.f32 q14, q7,  %e[w5][1]               \n"
            "pld [%[din_ptr4]]                          \n"

            "vld2.32 {d12-d15}, [%[din_ptr0]], %[s_8]   \n"
            "vld1.32 {%e[w0][0]}, [%[weights]]          \n"

            "vmla.f32 q13, q8,  %f[w5][0]               \n"
            "vmla.f32 q14, q9,  %f[w5][1]               \n"

            "vld2.32 {d16-d19}, [%[din_ptr0]], %[s_8]   \n"

            "vmla.f32 q13, q10, %e[w0][0]               \n"

            "vld2.32 {d20-d23}, [%[din_ptr0]], %[s_16]  \n"

            "vmov.32 %q[w0], q12                        \n"
            "vadd.f32 q13, q13, q14                     \n"
            "subs %[mid_cnt], #1                        \n"
            "vst1.32 {d26-d27}, [%[dout_ptr0]]!         \n"
            "bne 1b                                     \n"

            "2:                                         \n"
            "vld2.32 {d22-d25}, [%[mask]], %[s_8]       \n"
            "vld1.32 {d26-d27}, [%[vbias]]              \n"
            "vmov.32 q14, q15                           \n"

            // r0
            "vbif.32 q6, q15, q11                       \n"
            "vbif.32 q7, q15, q12                       \n"
            "vmla.f32 q13, q6, %e[w0][0]                \n"
            "vmla.f32 q14, q7, %e[w0][1]                \n"

            "vld2.32 {d22-d25}, [%[mask]], %[s_8]       \n"
            "vld2.32 {d12-d15}, [%[din_ptr1]], %[s_8]   \n"
            "vbif.32 q8, q15, q11                       \n"
            "vbif.32 q9, q15, q12                       \n"
            "vmla.f32 q13, q8, %f[w0][0]                \n"
            "vmla.f32 q14, q9, %f[w0][1]                \n"

            "vld2.32 {d22-d25}, [%[mask]]               \n"
            "sub %[mask], #16                           \n"
            "vld2.32 {d16-d19}, [%[din_ptr1]], %[s_8]   \n"
            "vbif.32 q10, q15, q11                      \n"
            "vmla.f32 q13, q10, %e[w1][0]               \n"

            // r1
            "vld2.32 {d20-d23}, [%[din_ptr1]]           \n"
            "vld2.32 {d22-d25}, [%[mask]], %[s_8]       \n"
            "vbif.32 q6, q15, q11                       \n"
            "vbif.32 q7, q15, q12                       \n"
            "vmla.f32 q13, q6, %e[w1][1]                \n"
            "vmla.f32 q14, q7, %f[w1][0]                \n"

            "vld2.32 {d22-d25}, [%[mask]], %[s_8]       \n"
            "vld2.32 {d12-d15}, [%[din_ptr2]], %[s_8]   \n"
            "vbif.32 q8, q15, q11                       \n"
            "vbif.32 q9, q15, q12                       \n"
            "vmla.f32 q13, q8, %f[w1][1]                \n"
            "vmla.f32 q14, q9, %e[w2][0]                \n"

            "vld2.32 {d22-d25}, [%[mask]]               \n"
            "sub %[mask], #16                           \n"
            "vld2.32 {d16-d19}, [%[din_ptr2]], %[s_8]   \n"
            "vbif.32 q10, q15, q11                      \n"
            "vmla.f32 q13, q10, %e[w2][1]               \n"

            // r2
            "vld2.32 {d20-d23}, [%[din_ptr2]]           \n"
            "vld2.32 {d22-d25}, [%[mask]], %[s_8]       \n"
            "vbif.32 q6, q15, q11                       \n"
            "vbif.32 q7, q15, q12                       \n"
            "vmla.f32 q13, q6, %f[w2][0]                \n"
            "vmla.f32 q14, q7, %f[w2][1]                \n"

            "vld2.32 {d12-d15}, [%[din_ptr3]], %[s_8]   \n"
            "vld2.32 {d22-d25}, [%[mask]], %[s_8]       \n"
            "vbif.32 q8, q15, q11                       \n"
            "vbif.32 q9, q15, q12                       \n"
            "vmla.f32 q13, q8, %e[w3][0]                \n"
            "vmla.f32 q14, q9, %e[w3][1]                \n"

            "vld2.32 {d22-d25}, [%[mask]]               \n"
            "sub %[mask], #16                           \n"
            "vld2.32 {d16-d19}, [%[din_ptr3]], %[s_8]   \n"
            "vbif.32 q10, q15, q11                      \n"
            "vmla.f32 q13, q10, %f[w3][0]               \n"

            // r3
            "vld2.32 {d20-d23}, [%[din_ptr3]]           \n"
            "vld2.32 {d22-d25}, [%[mask]], %[s_8]       \n"
            "vbif.32 q6, q15, q11                       \n"
            "vbif.32 q7, q15, q12                       \n"
            "vmla.f32 q13, q6, %f[w3][1]                \n"
            "vmla.f32 q14, q7, %e[w4][0]                \n"

            "vld2.32 {d22-d25}, [%[mask]], %[s_8]       \n"
            "vld2.32 {d12-d15}, [%[din_ptr4]], %[s_8]   \n"
            "vbif.32 q8, q15, q11                       \n"
            "vbif.32 q9, q15, q12                       \n"
            "vmla.f32 q13, q8, %e[w4][1]                \n"
            "vmla.f32 q14, q9, %f[w4][0]                \n"

            "vld2.32 {d22-d25}, [%[mask]]               \n"
            "sub %[mask], #16                           \n"
            "vld2.32 {d16-d19}, [%[din_ptr4]], %[s_8]   \n"
            "vbif.32 q10, q15, q11                      \n"
            "vmla.f32 q13, q10, %f[w4][1]               \n"

            // r4
            "vld2.32 {d20-d23}, [%[din_ptr4]]           \n"
            "vld2.32 {d22-d25}, [%[mask]], %[s_8]       \n"
            "vbif.32 q6, q15, q11                       \n"
            "vbif.32 q7, q15, q12                       \n"
            "vmla.f32 q13, q6, %e[w5][0]                \n"
            "vmla.f32 q14, q7, %e[w5][1]                \n"

            "vld2.32 {d22-d25}, [%[mask]], %[s_8]       \n"
            "vld1.32 {d12[0]}, [%[weights]]             \n"
            "vbif.32 q8, q15, q11                       \n"
            "vbif.32 q9, q15, q12                       \n"
            "vmla.f32 q13, q8, %f[w5][0]                \n"
            "vmla.f32 q14, q9, %f[w5][1]                \n"

            "vld2.32 {d22-d25}, [%[mask]]               \n"
            "vbif.32 q10, q15, q11                      \n"
            "vmla.f32 q13, q10, d12[0]                  \n"

            "vadd.f32 q13, q13, q14                     \n"
            "vst1.32 {d26-d27}, [%[out_buf0]]           \n"

            : [dout_ptr0] "+r"(dout_ptr0), [mid_cnt] "+r"(loop),
              [din_ptr0] "+r"(din_ptr0), [din_ptr1] "+r"(din_ptr1),
              [din_ptr2] "+r"(din_ptr2), [din_ptr3] "+r"(din_ptr3),
              [din_ptr4] "+r"(din_ptr4), [mask] "+r"(mask_ptr),
              [weights] "+r"(weights_ptr)
            : [w0] "w"(w0), [w1] "w"(w1), [w2] "w"(w2), [w3] "w"(w3),
              [w4] "w"(w4), [w5] "w"(w5), [vbias] "r"(vbias),
              [out_buf0] "r"(out_buf0), [s_8] "r"(s_8), [s_16] "r"(s_16)
            : "memory", "cc", "q6", "q7", "q8", "q9", "q10", "q11", "q12",
              "q13", "q14", "q15");

        int remain_cnt = w_out - (mid_cnt + 1) * 4;
        for (int i = 0; i < remain_cnt; ++i) {
          dout_ptr0[i] = out_buf0[i];
        }

        din0 = din2;
        din1 = din3;
        din2 = din4;
        din3 = din2 + w_in;
        din4 = din3 + w_in;
        dout0 += w_out;
      }
    }
  }
}

//! larger depthwise, win >= 9;
void conv_depthwise_5x5s2p2_relu(const float* din, float* dout, int num,
                                 int ch_out, int h_out, int w_out, int ch_in,
                                 int h_in, int w_in, const float* weights,
                                 const float* bias, bool flag_bias,
                                 bool flag_relu, ARMContext* ctx) {
  // printf("invoke 5x5s2p2 armv7\n");
  CHECK_GE(w_in, 9) << "only support win >= 9";
  int w_out_round = (w_out + 3) / 4 * 4;
  int cnt = (w_out_round - 4) / 4;
  int mid_cnt = cnt - 1;
  int right_start = cnt * 2 * 4 - 2;
  int mask_cnt = 12 - (w_in - right_start);
  int mask[12];
  memset(mask, 0xff, 12 * sizeof(int));
  for (int i = 0; i < mask_cnt; ++i) {
    mask[11 - i] = 0;
  }
  float* zero_ptr = ctx->workspace_data<float>();
  memset(zero_ptr, 0, w_in * sizeof(float));
  int in_spatial_size = w_in * h_in;
  int out_spatial_size = w_out * h_out;
  int weights_saptial_size = 25;

  for (int n = 0; n < num; ++n) {
    const float* din_batch = din + n * in_spatial_size * ch_in;
    float* dout_batch = dout + n * out_spatial_size * ch_out;
#pragma omp parallel for
    for (int c = 0; c < ch_in; ++c) {
      const float* din_ch = din_batch + c * in_spatial_size;
      float* dout_ch = dout_batch + c * out_spatial_size;
      const float* din0 = zero_ptr;
      const float* din1 = zero_ptr;
      const float* din2 = din_ch;
      const float* din3 = din2 + w_in;
      const float* din4 = din3 + w_in;

      float out_buf0[4];
      float* dout0 = dout_ch;

      const float* weights_c = weights + c * weights_saptial_size;
      float32x4_t w0 = vld1q_f32(weights_c);
      float32x4_t w1 = vld1q_f32(weights_c + 4);
      float32x4_t w2 = vld1q_f32(weights_c + 8);
      float32x4_t w3 = vld1q_f32(weights_c + 12);
      float32x4_t w4 = vld1q_f32(weights_c + 16);
      float32x4_t w5 = vld1q_f32(weights_c + 20);
      for (int h = 0; h < h_out; h += 1) {
        //! (h * 2 - 2) + 4 > h_in - 1
        if (h * 2 + 3 > h_in) {
          switch (h * 2 + 3 - h_in) {
            case 4:
              din1 = zero_ptr;
            case 3:
              din2 = zero_ptr;
            case 2:
              din3 = zero_ptr;
            case 1:
              din4 = zero_ptr;
            default:
              break;
          }
        }
        const float* din_ptr0 = din0;
        const float* din_ptr1 = din1;
        const float* din_ptr2 = din2;
        const float* din_ptr3 = din3;
        const float* din_ptr4 = din4;

        const float* weights_ptr = weights_c + 24;
        float* dout_ptr0 = dout0;

        float bias_c = 0.f;
        if (flag_bias) {
          bias_c = bias[c];
        }
        float vbias[4] = {bias_c, bias_c, bias_c, bias_c};
        int* mask_ptr = mask;
        int loop = mid_cnt;
        const int s_8 = 8;
        const int s_16 = 16;

        asm volatile(
            "vmov.i32   q15, #0x0           \n"
            "pld [%[din_ptr0]]              \n"
            "pld [%[din_ptr1]]              \n"
            "pld [%[din_ptr2]]              \n"
            "pld [%[din_ptr3]]              \n"
            "pld [%[din_ptr4]]              \n"
            "pld [%[mask]]                  \n"

            // left
            "vld2.32 {d16-d19}, [%[din_ptr0]]!          \n"
            "vld1.32 {d26-d29}, [%[vbias]]              \n"
            "vext.32 q6, q15, q8, #3                    \n"
            "vext.32 q7, q15, q9, #3                    \n"
            "vext.32 q10, q8, q15, #1                   \n"
            "vmov.32 q14, q15                           \n"

            // r0
            "vmla.f32 q13, q8, %f[w0][0]                \n"
            "vmla.f32 q14, q9, %f[w0][1]                \n"

            "vld1.32 {d21[1]}, [%[din_ptr0]]            \n"
            "vld2.32 {d16-d19}, [%[din_ptr1]]!          \n"
            "sub %[din_ptr0], #8  \n"

            "vmla.f32 q13, q6, %e[w0][0]                \n"
            "vmla.f32 q14, q7, %e[w0][1]                \n"
            "vmla.f32 q13, q10, %e[w1][0]               \n"

            "vext.32 q6, q15, q8, #3                    \n"
            "vext.32 q7, q15, q9, #3                    \n"
            "vext.32 q10, q8, q15, #1                   \n"

            // r1
            "vmla.f32 q13, q8, %f[w1][1]                \n"
            "vmla.f32 q14, q9, %e[w2][0]                \n"

            "vld1.32 {d21[1]}, [%[din_ptr1]]            \n"
            "vld2.32 {d16-d19}, [%[din_ptr2]]!          \n"
            "sub %[din_ptr1], #8                        \n"

            "vmla.f32 q13, q6, %e[w1][1]                \n"
            "vmla.f32 q14, q7, %f[w1][0]                \n"
            "vmla.f32 q13, q10, %e[w2][1]               \n"

            "vext.32 q6, q15, q8, #3                    \n"
            "vext.32 q7, q15, q9, #3                    \n"
            "vext.32 q10, q8, q15, #1                   \n"

            // r2
            "vmla.f32 q13, q8, %e[w3][0]                \n"
            "vmla.f32 q14, q9, %e[w3][1]                \n"

            "vld1.32 {d21[1]}, [%[din_ptr2]]            \n"
            "vld2.32 {d16-d19}, [%[din_ptr3]]!          \n"
            "sub %[din_ptr2], #8                        \n"

            "vmla.f32 q13, q6, %f[w2][0]                \n"
            "vmla.f32 q14, q7, %f[w2][1]                \n"
            "vmla.f32 q13, q10, %f[w3][0]               \n"

            "vext.32 q6, q15, q8, #3                    \n"
            "vext.32 q7, q15, q9, #3                    \n"
            "vext.32 q10, q8, q15, #1                   \n"

            // r3
            "vmla.f32 q13, q8, %e[w4][1]                \n"
            "vmla.f32 q14, q9, %f[w4][0]                \n"

            "vld1.32 {d21[1]}, [%[din_ptr3]]            \n"
            "vld2.32 {d16-d19}, [%[din_ptr4]]!          \n"
            "sub %[din_ptr3], #8                        \n"

            "vmla.f32 q13, q6, %f[w3][1]                \n"
            "vmla.f32 q14, q7, %e[w4][0]                \n"
            "vmla.f32 q13, q10, %f[w4][1]               \n"

            "vext.32 q6, q15, q8, #3                    \n"
            "vext.32 q7, q15, q9, #3                    \n"
            "vext.32 q10, q8, q15, #1                   \n"

            // r4
            "vmla.f32 q13, q6, %e[w5][0]                \n"
            "vmla.f32 q14, q7, %e[w5][1]                \n"

            "vld1.32 {d21[1]}, [%[din_ptr4]]            \n"
            "vld2.32 {d12-d15}, [%[din_ptr0]], %[s_8]   \n"
            "sub %[din_ptr4], #8                        \n"

            "vmla.f32 q13, q8, %f[w5][0]                \n"
            "vmla.f32 q14, q9, %f[w5][1]                \n"

            "vld2.32 {d16-d19}, [%[din_ptr0]], %[s_8]   \n"

            "vmov.32 q12, %q[w0]                        \n"
            "vld1.32 {%e[w0][0]}, [%[weights]]          \n"
            "vmla.f32 q13, q10, %e[w0][0]               \n"
            "vadd.f32 q13, q13, q14                     \n"
            "vmov.f32 %q[w0], q12                        \n"
            "vmax.f32 q13, q13, q15                     \n"
            "cmp %[mid_cnt], #1                         \n"
            "vld2.32 {d20-d23}, [%[din_ptr0]], %[s_16]  \n"
            "vst1.32 {d26-d27}, [%[dout_ptr0]]!         \n"
            "pld [%[din_ptr0]]                          \n"
            "blt 2f                                     \n"

            // mid
            "1:                                         \n"
            "vld1.32 {d26-d27}, [%[vbias]]              \n"
            "vmov.32 q14, q15                           \n"

            // r0
            "vmla.f32 q13, q6,  %e[w0][0]               \n"
            "vmla.f32 q14, q7,  %e[w0][1]               \n"

            "vld2.32 {d12-d15}, [%[din_ptr1]], %[s_8]   \n"

            "vmla.f32 q13, q8,  %f[w0][0]               \n"
            "vmla.f32 q14, q9,  %f[w0][1]               \n"

            "vld2.32 {d16-d19}, [%[din_ptr1]], %[s_8]   \n"

            "vmla.f32 q13, q10, %e[w1][0]               \n"

            "vld2.32 {d20-d23}, [%[din_ptr1]], %[s_16]  \n"

            // r1
            "vmla.f32 q13, q6,  %e[w1][1]               \n"
            "vmla.f32 q14, q7,  %f[w1][0]               \n"
            "pld [%[din_ptr1]]                          \n"

            "vld2.32 {d12-d15}, [%[din_ptr2]], %[s_8]   \n"

            "vmla.f32 q13, q8,  %f[w1][1]               \n"
            "vmla.f32 q14, q9,  %e[w2][0]               \n"

            "vld2.32 {d16-d19}, [%[din_ptr2]], %[s_8]   \n"

            "vmla.f32 q13, q10, %e[w2][1]               \n"

            "vld2.32 {d20-d23}, [%[din_ptr2]], %[s_16]  \n"

            // r2
            "vmla.f32 q13, q6,  %f[w2][0]               \n"
            "vmla.f32 q14, q7,  %f[w2][1]               \n"
            "pld [%[din_ptr2]]                          \n"

            "vld2.32 {d12-d15}, [%[din_ptr3]], %[s_8]   \n"

            "vmla.f32 q13, q8,  %e[w3][0]               \n"
            "vmla.f32 q14, q9,  %e[w3][1]               \n"

            "vld2.32 {d16-d19}, [%[din_ptr3]], %[s_8]   \n"

            "vmla.f32 q13, q10, %f[w3][0]               \n"

            "vld2.32 {d20-d23}, [%[din_ptr3]], %[s_16]  \n"

            // r3
            "vmla.f32 q13, q6,  %f[w3][1]               \n"
            "vmla.f32 q14, q7,  %e[w4][0]               \n"
            "pld [%[din_ptr3]]                          \n"

            "vld2.32 {d12-d15}, [%[din_ptr4]], %[s_8]   \n"

            "vmla.f32 q13, q8,  %e[w4][1]               \n"
            "vmla.f32 q14, q9,  %f[w4][0]               \n"

            "vld2.32 {d16-d19}, [%[din_ptr4]], %[s_8]   \n"

            "vmla.f32 q13, q10, %f[w4][1]               \n"

            "vld2.32 {d20-d23}, [%[din_ptr4]], %[s_16]  \n"

            // r4
            "vmla.f32 q13, q6,  %e[w5][0]               \n"
            "vmla.f32 q14, q7,  %e[w5][1]               \n"
            "pld [%[din_ptr4]]                          \n"

            "vld2.32 {d12-d15}, [%[din_ptr0]], %[s_8]   \n"
            "vld1.32 {%e[w0][0]}, [%[weights]]          \n"

            "vmla.f32 q13, q8,  %f[w5][0]               \n"
            "vmla.f32 q14, q9,  %f[w5][1]               \n"

            "vld2.32 {d16-d19}, [%[din_ptr0]], %[s_8]   \n"

            "vmla.f32 q13, q10, %e[w0][0]               \n"

            "vld2.32 {d20-d23}, [%[din_ptr0]], %[s_16]  \n"

            "vmov.32 %q[w0], q12                        \n"
            "vadd.f32 q13, q13, q14                     \n"
            "vmax.f32 q13, q13, q15                     \n"
            "subs %[mid_cnt], #1                        \n"
            "vst1.32 {d26-d27}, [%[dout_ptr0]]!         \n"
            "bne 1b                                     \n"

            "2:                                         \n"
            "vld2.32 {d22-d25}, [%[mask]], %[s_8]       \n"
            "vld1.32 {d26-d27}, [%[vbias]]              \n"
            "vmov.32 q14, q15                           \n"

            // r0
            "vbif.32 q6, q15, q11                       \n"
            "vbif.32 q7, q15, q12                       \n"
            "vmla.f32 q13, q6, %e[w0][0]                \n"
            "vmla.f32 q14, q7, %e[w0][1]                \n"

            "vld2.32 {d22-d25}, [%[mask]], %[s_8]       \n"
            "vld2.32 {d12-d15}, [%[din_ptr1]], %[s_8]   \n"
            "vbif.32 q8, q15, q11                       \n"
            "vbif.32 q9, q15, q12                       \n"
            "vmla.f32 q13, q8, %f[w0][0]                \n"
            "vmla.f32 q14, q9, %f[w0][1]                \n"

            "vld2.32 {d22-d25}, [%[mask]]               \n"
            "sub %[mask], #16                           \n"
            "vld2.32 {d16-d19}, [%[din_ptr1]], %[s_8]   \n"
            "vbif.32 q10, q15, q11                      \n"
            "vmla.f32 q13, q10, %e[w1][0]               \n"

            // r1
            "vld2.32 {d20-d23}, [%[din_ptr1]]           \n"
            "vld2.32 {d22-d25}, [%[mask]], %[s_8]       \n"
            "vbif.32 q6, q15, q11                       \n"
            "vbif.32 q7, q15, q12                       \n"
            "vmla.f32 q13, q6, %e[w1][1]                \n"
            "vmla.f32 q14, q7, %f[w1][0]                \n"

            "vld2.32 {d22-d25}, [%[mask]], %[s_8]       \n"
            "vld2.32 {d12-d15}, [%[din_ptr2]], %[s_8]   \n"
            "vbif.32 q8, q15, q11                       \n"
            "vbif.32 q9, q15, q12                       \n"
            "vmla.f32 q13, q8, %f[w1][1]                \n"
            "vmla.f32 q14, q9, %e[w2][0]                \n"

            "vld2.32 {d22-d25}, [%[mask]]               \n"
            "sub %[mask], #16                           \n"
            "vld2.32 {d16-d19}, [%[din_ptr2]], %[s_8]   \n"
            "vbif.32 q10, q15, q11                      \n"
            "vmla.f32 q13, q10, %e[w2][1]               \n"

            // r2
            "vld2.32 {d20-d23}, [%[din_ptr2]]           \n"
            "vld2.32 {d22-d25}, [%[mask]], %[s_8]       \n"
            "vbif.32 q6, q15, q11                       \n"
            "vbif.32 q7, q15, q12                       \n"
            "vmla.f32 q13, q6, %f[w2][0]                \n"
            "vmla.f32 q14, q7, %f[w2][1]                \n"

            "vld2.32 {d12-d15}, [%[din_ptr3]], %[s_8]   \n"
            "vld2.32 {d22-d25}, [%[mask]], %[s_8]       \n"
            "vbif.32 q8, q15, q11                       \n"
            "vbif.32 q9, q15, q12                       \n"
            "vmla.f32 q13, q8, %e[w3][0]                \n"
            "vmla.f32 q14, q9, %e[w3][1]                \n"

            "vld2.32 {d22-d25}, [%[mask]]               \n"
            "sub %[mask], #16                           \n"
            "vld2.32 {d16-d19}, [%[din_ptr3]], %[s_8]   \n"
            "vbif.32 q10, q15, q11                      \n"
            "vmla.f32 q13, q10, %f[w3][0]               \n"

            // r3
            "vld2.32 {d20-d23}, [%[din_ptr3]]           \n"
            "vld2.32 {d22-d25}, [%[mask]], %[s_8]       \n"
            "vbif.32 q6, q15, q11                       \n"
            "vbif.32 q7, q15, q12                       \n"
            "vmla.f32 q13, q6, %f[w3][1]                \n"
            "vmla.f32 q14, q7, %e[w4][0]                \n"

            "vld2.32 {d22-d25}, [%[mask]], %[s_8]       \n"
            "vld2.32 {d12-d15}, [%[din_ptr4]], %[s_8]   \n"
            "vbif.32 q8, q15, q11                       \n"
            "vbif.32 q9, q15, q12                       \n"
            "vmla.f32 q13, q8, %e[w4][1]                \n"
            "vmla.f32 q14, q9, %f[w4][0]                \n"

            "vld2.32 {d22-d25}, [%[mask]]               \n"
            "sub %[mask], #16                           \n"
            "vld2.32 {d16-d19}, [%[din_ptr4]], %[s_8]   \n"
            "vbif.32 q10, q15, q11                      \n"
            "vmla.f32 q13, q10, %f[w4][1]               \n"

            // r4
            "vld2.32 {d20-d23}, [%[din_ptr4]]           \n"
            "vld2.32 {d22-d25}, [%[mask]], %[s_8]       \n"
            "vbif.32 q6, q15, q11                       \n"
            "vbif.32 q7, q15, q12                       \n"
            "vmla.f32 q13, q6, %e[w5][0]                \n"
            "vmla.f32 q14, q7, %e[w5][1]                \n"

            "vld2.32 {d22-d25}, [%[mask]], %[s_8]       \n"
            "vld1.32 {d12[0]}, [%[weights]]             \n"
            "vbif.32 q8, q15, q11                       \n"
            "vbif.32 q9, q15, q12                       \n"
            "vmla.f32 q13, q8, %f[w5][0]                \n"
            "vmla.f32 q14, q9, %f[w5][1]                \n"

            "vld2.32 {d22-d25}, [%[mask]]               \n"
            "vbif.32 q10, q15, q11                      \n"
            "vmla.f32 q13, q10, d12[0]                  \n"

            "vadd.f32 q13, q13, q14                     \n"
            "vmax.f32 q13, q13, q15                     \n"
            "vst1.32 {d26-d27}, [%[out_buf0]]           \n"

            : [dout_ptr0] "+r"(dout_ptr0), [mid_cnt] "+r"(loop),
              [din_ptr0] "+r"(din_ptr0), [din_ptr1] "+r"(din_ptr1),
              [din_ptr2] "+r"(din_ptr2), [din_ptr3] "+r"(din_ptr3),
              [din_ptr4] "+r"(din_ptr4), [mask] "+r"(mask_ptr),
              [weights] "+r"(weights_ptr)
            : [w0] "w"(w0), [w1] "w"(w1), [w2] "w"(w2), [w3] "w"(w3),
              [w4] "w"(w4), [w5] "w"(w5), [vbias] "r"(vbias),
              [out_buf0] "r"(out_buf0), [s_8] "r"(s_8), [s_16] "r"(s_16)
            : "memory", "cc", "q6", "q7", "q8", "q9", "q10", "q11", "q12",
              "q13", "q14", "q15");

        int remain_cnt = w_out - (mid_cnt + 1) * 4;
        for (int i = 0; i < remain_cnt; ++i) {
          dout_ptr0[i] = out_buf0[i];
        }

        din0 = din2;
        din1 = din3;
        din2 = din4;
        din3 = din2 + w_in;
        din4 = din3 + w_in;
        dout0 += w_out;
      }
    }
  }
}

//! small depthwise, win < 9;
void conv_depthwise_5x5s2p2_s(const float* din, float* dout, int num,
                              int ch_out, int h_out, int w_out, int ch_in,
                              int h_in, int w_in, const float* weights,
                              const float* bias, bool flag_bias, bool flag_relu,
                              ARMContext* ctx) {
  CHECK_LT(w_in, 9) << "only support win < 9";
  int w_out_round = (w_out + 3) / 4 * 4;
  int mask_cnt = 12 - w_in - 2;
  int mask[12];
  memset(mask, 0xff, 12 * sizeof(int));
  for (int i = 0; i < mask_cnt; ++i) {
    mask[11 - i] = 0;
  }
  float* zero_ptr = ctx->workspace_data<float>();
  memset(zero_ptr, 0, w_in * sizeof(float));
  int in_spatial_size = w_in * h_in;
  int out_spatial_size = w_out * h_out;
  int weights_saptial_size = 25;

  for (int n = 0; n < num; ++n) {
    const float* din_batch = din + n * in_spatial_size * ch_in;
    float* dout_batch = dout + n * out_spatial_size * ch_out;
#pragma omp parallel for
    for (int c = 0; c < ch_in; ++c) {
      const float* din_ch = din_batch + c * in_spatial_size;
      float* dout_ch = dout_batch + c * out_spatial_size;
      const float* din0 = zero_ptr;
      const float* din1 = zero_ptr;
      const float* din2 = din_ch;
      const float* din3 = din2 + w_in;
      const float* din4 = din3 + w_in;

      float out_buf0[4];
      float out_buf1[4];
      float* dout0 = dout_ch;
      float* dout1 = dout0 + w_out;

      const float* weights_c = weights + c * weights_saptial_size;
      float32x4_t w0 = vld1q_f32(weights_c);
      float32x4_t w1 = vld1q_f32(weights_c + 4);
      float32x4_t w2 = vld1q_f32(weights_c + 8);
      float32x4_t w3 = vld1q_f32(weights_c + 12);
      float32x4_t w4 = vld1q_f32(weights_c + 16);
      float32x4_t w5 = vld1q_f32(weights_c + 20);
      for (int h = 0; h < h_out; h += 1) {
        //! (h * 2 - 2) + 4 > h_in - 1
        if (h * 2 + 3 > h_in) {
          switch (h * 2 + 3 - h_in) {
            case 4:
              din1 = zero_ptr;
            case 3:
              din2 = zero_ptr;
            case 2:
              din3 = zero_ptr;
            case 1:
              din4 = zero_ptr;
            default:
              break;
          }
        }
        const float* din_ptr0 = din0;
        const float* din_ptr1 = din1;
        const float* din_ptr2 = din2;
        const float* din_ptr3 = din3;
        const float* din_ptr4 = din4;

        const float* weights_ptr = weights_c + 24;
        float* dout_ptr0 = dout0;

        float bias_c = 0.f;
        if (flag_bias) {
          bias_c = bias[c];
        }
        float vbias[4] = {bias_c, bias_c, bias_c, bias_c};
        int* mask_ptr = mask;
        const int s_8 = 8;

        asm volatile(
            "vmov.i32  q15, #0x0                 \n"
            "pld [%[din_ptr0]]                   \n"
            "pld [%[din_ptr1]]                   \n"
            "pld [%[din_ptr2]]                   \n"
            "pld [%[din_ptr3]]                   \n"
            "pld [%[din_ptr4]]                   \n"
            "vld1.32 {d26-d27}, [%[vbias]]       \n"
            "vmov.32 q14, q15                    \n"
            "vld2.32 {d16-d19}, [%[din_ptr0]]!   \n"

            // r0
            "vld2.32 {d22-d25}, [%[mask]], %[s_8]       \n"
            "vext.32 q6, q15, q8, #3                    \n"
            "vext.32 q7, q15, q9, #3                    \n"
            "vext.32 q10, q8, q15, #1                   \n"
            "vld1.32 {d21[1]}, [%[din_ptr0]]            \n"

            "vbif.32 q6, q15, q11                       \n"
            "vbif.32 q7, q15, q12                       \n"
            "vmla.f32 q13, q6, %e[w0][0]                \n"
            "vmla.f32 q14, q7, %e[w0][1]                \n"

            "vld2.32 {d22-d25}, [%[mask]], %[s_8]       \n"
            "vbif.32 q8, q15, q11                       \n"
            "vbif.32 q9, q15, q12                       \n"
            "vmla.f32 q13, q8, %f[w0][0]                \n"
            "vmla.f32 q14, q9, %f[w0][1]                \n"

            "vld2.32 {d22-d25}, [%[mask]]               \n"
            "vld2.32 {d16-d19}, [%[din_ptr1]]!          \n"
            "sub %[mask], #16                           \n"
            "vbif.32 q10, q15, q11                      \n"
            "vmla.f32 q13, q10, %e[w1][0]               \n"

            // r1
            "vld2.32 {d22-d25}, [%[mask]], %[s_8]       \n"
            "vext.32 q6, q15, q8, #3                    \n"
            "vext.32 q7, q15, q9, #3                    \n"
            "vext.32 q10, q8, q15, #1                   \n"
            "vld1.32 {d21[1]}, [%[din_ptr1]]            \n"

            "vbif.32 q6, q15, q11                       \n"
            "vbif.32 q7, q15, q12                       \n"
            "vmla.f32 q14, q6, %e[w1][1]                \n"
            "vmla.f32 q13, q7, %f[w1][0]                \n"

            "vld2.32 {d22-d25}, [%[mask]], %[s_8]       \n"
            "vbif.32 q8, q15, q11                       \n"
            "vbif.32 q9, q15, q12                       \n"
            "vmla.f32 q14, q8, %f[w1][1]                \n"
            "vmla.f32 q13, q9, %e[w2][0]                \n"

            "vld2.32 {d22-d25}, [%[mask]]               \n"
            "vld2.32 {d16-d19}, [%[din_ptr2]]!          \n"
            "sub %[mask], #16                           \n"
            "vbif.32 q10, q15, q11                      \n"
            "vmla.f32 q14, q10, %e[w2][1]               \n"

            // r2
            "vld2.32 {d22-d25}, [%[mask]], %[s_8]       \n"
            "vext.32 q6, q15, q8, #3                    \n"
            "vext.32 q7, q15, q9, #3                    \n"
            "vext.32 q10, q8, q15, #1                   \n"
            "vld1.32 {d21[1]}, [%[din_ptr2]]            \n"

            "vbif.32 q6, q15, q11                       \n"
            "vbif.32 q7, q15, q12                       \n"
            "vmla.f32 q13, q6, %f[w2][0]                \n"
            "vmla.f32 q14, q7, %f[w2][1]                \n"

            "vld2.32 {d22-d25}, [%[mask]], %[s_8]       \n"
            "vbif.32 q8, q15, q11                       \n"
            "vbif.32 q9, q15, q12                       \n"
            "vmla.f32 q13, q8, %e[w3][0]                \n"
            "vmla.f32 q14, q9, %e[w3][1]                \n"

            "vld2.32 {d22-d25}, [%[mask]]               \n"
            "vld2.32 {d16-d19}, [%[din_ptr3]]!          \n"
            "sub %[mask], #16                           \n"
            "vbif.32 q10, q15, q11                      \n"
            "vmla.f32 q13, q10, %f[w3][0]               \n"

            // r3
            "vld2.32 {d22-d25}, [%[mask]], %[s_8]       \n"
            "vext.32 q6, q15, q8, #3                    \n"
            "vext.32 q7, q15, q9, #3                    \n"
            "vext.32 q10, q8, q15, #1                   \n"
            "vld1.32 {d21[1]}, [%[din_ptr3]]            \n"

            "vbif.32 q6, q15, q11                       \n"
            "vbif.32 q7, q15, q12                       \n"
            "vmla.f32 q14, q6, %f[w3][1]                \n"
            "vmla.f32 q13, q7, %e[w4][0]                \n"

            "vld2.32 {d22-d25}, [%[mask]], %[s_8]       \n"
            "vbif.32 q8, q15, q11                       \n"
            "vbif.32 q9, q15, q12                       \n"
            "vmla.f32 q14, q8, %e[w4][1]                \n"
            "vmla.f32 q13, q9, %f[w4][0]                \n"

            "vld2.32 {d22-d25}, [%[mask]]               \n"
            "vld2.32 {d16-d19}, [%[din_ptr4]]!          \n"
            "sub %[mask], #16                           \n"
            "vbif.32 q10, q15, q11                      \n"
            "vmla.f32 q14, q10, %f[w4][1]               \n"

            // r4
            "vld2.32 {d22-d25}, [%[mask]], %[s_8]       \n"
            "vext.32 q6, q15, q8, #3                    \n"
            "vext.32 q7, q15, q9, #3                    \n"
            "vext.32 q10, q8, q15, #1                   \n"
            "vld1.32 {d21[1]}, [%[din_ptr4]]            \n"

            "vbif.32 q6, q15, q11                       \n"
            "vbif.32 q7, q15, q12                       \n"
            "vmla.f32 q13, q6, %e[w5][0]                \n"
            "vmla.f32 q14, q7, %e[w5][1]                \n"

            "vld2.32 {d22-d25}, [%[mask]], %[s_8]       \n"
            "vld1.32 {d12[0]}, [%[weights]]             \n"
            "vbif.32 q8, q15, q11                       \n"
            "vbif.32 q9, q15, q12                       \n"
            "vmla.f32 q13, q8, %f[w5][0]                \n"
            "vmla.f32 q14, q9, %f[w5][1]                \n"

            "vld2.32 {d22-d25}, [%[mask]]               \n"
            "vbif.32 q10, q15, q11                      \n"
            "vmla.f32 q13, q10, d12[0]                  \n"

            "vadd.f32 q13, q13, q14                     \n"
            "vst1.32 {d26-d27}, [%[out_buf0]]           \n"

            : [dout_ptr0] "+r"(dout_ptr0), [din_ptr0] "+r"(din_ptr0),
              [din_ptr1] "+r"(din_ptr1), [din_ptr2] "+r"(din_ptr2),
              [din_ptr3] "+r"(din_ptr3), [din_ptr4] "+r"(din_ptr4),
              [mask] "+r"(mask_ptr), [weights] "+r"(weights_ptr)
            : [vbias] "r"(vbias), [out_buf0] "r"(out_buf0), [s_8] "r"(s_8),
              [w0] "w"(w0), [w1] "w"(w1), [w2] "w"(w2), [w3] "w"(w3),
              [w4] "w"(w4), [w5] "w"(w5)
            : "memory", "cc", "q6", "q7", "q8", "q9", "q10", "q11", "q12",
              "q13", "q14", "q15");
        for (int i = 0; i < w_out; ++i) {
          dout_ptr0[i] = out_buf0[i];
        }
        din0 = din2;
        din1 = din3;
        din2 = din4;
        din3 = din2 + w_in;
        din4 = din3 + w_in;
        dout0 += w_out;
      }
    }
  }
}

//! small depthwise, win < 9;
void conv_depthwise_5x5s2p2_relu_s(const float* din, float* dout, int num,
                                   int ch_out, int h_out, int w_out, int ch_in,
                                   int h_in, int w_in, const float* weights,
                                   const float* bias, bool flag_bias,
                                   bool flag_relu, ARMContext* ctx) {
  CHECK_LT(w_in, 9) << "only support win < 9\n";
  int w_out_round = (w_out + 3) / 4 * 4;
  int mask_cnt = 12 - w_in - 2;
  int mask[12];
  memset(mask, 0xff, 12 * sizeof(int));
  for (int i = 0; i < mask_cnt; ++i) {
    mask[11 - i] = 0;
  }
  float* zero_ptr = ctx->workspace_data<float>();
  memset(zero_ptr, 0, w_in * sizeof(float));
  int in_spatial_size = w_in * h_in;
  int out_spatial_size = w_out * h_out;
  int weights_saptial_size = 25;

  for (int n = 0; n < num; ++n) {
    const float* din_batch = din + n * in_spatial_size * ch_in;
    float* dout_batch = dout + n * out_spatial_size * ch_out;
#pragma omp parallel for
    for (int c = 0; c < ch_in; ++c) {
      const float* din_ch = din_batch + c * in_spatial_size;
      float* dout_ch = dout_batch + c * out_spatial_size;
      const float* din0 = zero_ptr;
      const float* din1 = zero_ptr;
      const float* din2 = din_ch;
      const float* din3 = din2 + w_in;
      const float* din4 = din3 + w_in;

      float out_buf0[4];
      float out_buf1[4];
      float* dout0 = dout_ch;
      float* dout1 = dout0 + w_out;

      const float* weights_c = weights + c * weights_saptial_size;
      float32x4_t w0 = vld1q_f32(weights_c);
      float32x4_t w1 = vld1q_f32(weights_c + 4);
      float32x4_t w2 = vld1q_f32(weights_c + 8);
      float32x4_t w3 = vld1q_f32(weights_c + 12);
      float32x4_t w4 = vld1q_f32(weights_c + 16);
      float32x4_t w5 = vld1q_f32(weights_c + 20);
      for (int h = 0; h < h_out; h += 1) {
        //! (h * 2 - 2) + 4 > h_in - 1
        if (h * 2 + 3 > h_in) {
          switch (h * 2 + 3 - h_in) {
            case 4:
              din1 = zero_ptr;
            case 3:
              din2 = zero_ptr;
            case 2:
              din3 = zero_ptr;
            case 1:
              din4 = zero_ptr;
            default:
              break;
          }
        }
        const float* din_ptr0 = din0;
        const float* din_ptr1 = din1;
        const float* din_ptr2 = din2;
        const float* din_ptr3 = din3;
        const float* din_ptr4 = din4;

        const float* weights_ptr = weights_c + 24;
        float* dout_ptr0 = dout0;

        float bias_c = 0.f;
        if (flag_bias) {
          bias_c = bias[c];
        }
        float vbias[4] = {bias_c, bias_c, bias_c, bias_c};
        int* mask_ptr = mask;
        const int s_8 = 8;

        asm volatile(
            "vmov.i32  q15, #0x0                \n"
            "pld [%[din_ptr0]]                  \n"
            "pld [%[din_ptr1]]                  \n"
            "pld [%[din_ptr2]]                  \n"
            "pld [%[din_ptr3]]                  \n"
            "pld [%[din_ptr4]]                  \n"
            "vld1.32 {d26-d27}, [%[vbias]]      \n"
            "vmov.32 q14, q15                   \n"
            "vld2.32 {d16-d19}, [%[din_ptr0]]!  \n"

            // r0
            "vld2.32 {d22-d25}, [%[mask]], %[s_8]       \n"
            "vext.32 q6, q15, q8, #3                    \n"
            "vext.32 q7, q15, q9, #3                    \n"
            "vext.32 q10, q8, q15, #1                   \n"
            "vld1.32 {d21[1]}, [%[din_ptr0]]            \n"

            "vbif.32 q6, q15, q11                       \n"
            "vbif.32 q7, q15, q12                       \n"
            "vmla.f32 q13, q6, %e[w0][0]                \n"
            "vmla.f32 q14, q7, %e[w0][1]                \n"

            "vld2.32 {d22-d25}, [%[mask]], %[s_8]       \n"
            "vbif.32 q8, q15, q11                       \n"
            "vbif.32 q9, q15, q12                       \n"
            "vmla.f32 q13, q8, %f[w0][0]                \n"
            "vmla.f32 q14, q9, %f[w0][1]                \n"

            "vld2.32 {d22-d25}, [%[mask]]               \n"
            "vld2.32 {d16-d19}, [%[din_ptr1]]!          \n"
            "sub %[mask], #16                           \n"
            "vbif.32 q10, q15, q11                      \n"
            "vmla.f32 q13, q10, %e[w1][0]               \n"

            // r1
            "vld2.32 {d22-d25}, [%[mask]], %[s_8]       \n"
            "vext.32 q6, q15, q8, #3                    \n"
            "vext.32 q7, q15, q9, #3                    \n"
            "vext.32 q10, q8, q15, #1                   \n"
            "vld1.32 {d21[1]}, [%[din_ptr1]]            \n"

            "vbif.32 q6, q15, q11                       \n"
            "vbif.32 q7, q15, q12                       \n"
            "vmla.f32 q14, q6, %e[w1][1]                \n"
            "vmla.f32 q13, q7, %f[w1][0]                \n"

            "vld2.32 {d22-d25}, [%[mask]], %[s_8]       \n"
            "vbif.32 q8, q15, q11                       \n"
            "vbif.32 q9, q15, q12                       \n"
            "vmla.f32 q14, q8, %f[w1][1]                \n"
            "vmla.f32 q13, q9, %e[w2][0]                \n"

            "vld2.32 {d22-d25}, [%[mask]]               \n"
            "vld2.32 {d16-d19}, [%[din_ptr2]]!          \n"
            "sub %[mask], #16                           \n"
            "vbif.32 q10, q15, q11                      \n"
            "vmla.f32 q14, q10, %e[w2][1]               \n"

            // r2
            "vld2.32 {d22-d25}, [%[mask]], %[s_8]       \n"
            "vext.32 q6, q15, q8, #3                    \n"
            "vext.32 q7, q15, q9, #3                    \n"
            "vext.32 q10, q8, q15, #1                   \n"
            "vld1.32 {d21[1]}, [%[din_ptr2]]            \n"

            "vbif.32 q6, q15, q11                       \n"
            "vbif.32 q7, q15, q12                       \n"
            "vmla.f32 q13, q6, %f[w2][0]                \n"
            "vmla.f32 q14, q7, %f[w2][1]                \n"

            "vld2.32 {d22-d25}, [%[mask]], %[s_8]       \n"
            "vbif.32 q8, q15, q11                       \n"
            "vbif.32 q9, q15, q12                       \n"
            "vmla.f32 q13, q8, %e[w3][0]                \n"
            "vmla.f32 q14, q9, %e[w3][1]                \n"

            "vld2.32 {d22-d25}, [%[mask]]               \n"
            "vld2.32 {d16-d19}, [%[din_ptr3]]!          \n"
            "sub %[mask], #16                           \n"
            "vbif.32 q10, q15, q11                      \n"
            "vmla.f32 q13, q10, %f[w3][0]               \n"

            // r3
            "vld2.32 {d22-d25}, [%[mask]], %[s_8]       \n"
            "vext.32 q6, q15, q8, #3                    \n"
            "vext.32 q7, q15, q9, #3                    \n"
            "vext.32 q10, q8, q15, #1                   \n"
            "vld1.32 {d21[1]}, [%[din_ptr3]]            \n"

            "vbif.32 q6, q15, q11                       \n"
            "vbif.32 q7, q15, q12                       \n"
            "vmla.f32 q14, q6, %f[w3][1]                \n"
            "vmla.f32 q13, q7, %e[w4][0]                \n"

            "vld2.32 {d22-d25}, [%[mask]], %[s_8]       \n"
            "vbif.32 q8, q15, q11                       \n"
            "vbif.32 q9, q15, q12                       \n"
            "vmla.f32 q14, q8, %e[w4][1]                \n"
            "vmla.f32 q13, q9, %f[w4][0]                \n"

            "vld2.32 {d22-d25}, [%[mask]]               \n"
            "vld2.32 {d16-d19}, [%[din_ptr4]]!          \n"
            "sub %[mask], #16                           \n"
            "vbif.32 q10, q15, q11                      \n"
            "vmla.f32 q14, q10, %f[w4][1]               \n"

            // r4
            "vld2.32 {d22-d25}, [%[mask]], %[s_8]       \n"
            "vext.32 q6, q15, q8, #3                    \n"
            "vext.32 q7, q15, q9, #3                    \n"
            "vext.32 q10, q8, q15, #1                   \n"
            "vld1.32 {d21[1]}, [%[din_ptr4]]            \n"

            "vbif.32 q6, q15, q11                       \n"
            "vbif.32 q7, q15, q12                       \n"
            "vmla.f32 q13, q6, %e[w5][0]                \n"
            "vmla.f32 q14, q7, %e[w5][1]                \n"

            "vld2.32 {d22-d25}, [%[mask]], %[s_8]       \n"
            "vld1.32 {d12[0]}, [%[weights]]             \n"
            "vbif.32 q8, q15, q11                       \n"
            "vbif.32 q9, q15, q12                       \n"
            "vmla.f32 q13, q8, %f[w5][0]                \n"
            "vmla.f32 q14, q9, %f[w5][1]                \n"

            "vld2.32 {d22-d25}, [%[mask]]               \n"
            "vbif.32 q10, q15, q11                      \n"
            "vmla.f32 q13, q10, d12[0]                  \n"

            "vadd.f32 q13, q13, q14                     \n"
            "vmax.f32 q13, q13, q15                     \n"
            "vst1.32 {d26-d27}, [%[out_buf0]]           \n"

            : [dout_ptr0] "+r"(dout_ptr0), [din_ptr0] "+r"(din_ptr0),
              [din_ptr1] "+r"(din_ptr1), [din_ptr2] "+r"(din_ptr2),
              [din_ptr3] "+r"(din_ptr3), [din_ptr4] "+r"(din_ptr4),
              [mask] "+r"(mask_ptr), [weights] "+r"(weights_ptr)
            : [vbias] "r"(vbias), [out_buf0] "r"(out_buf0), [s_8] "r"(s_8),
              [w0] "w"(w0), [w1] "w"(w1), [w2] "w"(w2), [w3] "w"(w3),
              [w4] "w"(w4), [w5] "w"(w5)
            : "memory", "cc", "q6", "q7", "q8", "q9", "q10", "q11", "q12",
              "q13", "q14", "q15");
        for (int i = 0; i < w_out; ++i) {
          dout_ptr0[i] = out_buf0[i];
        }
        din0 = din2;
        din1 = din3;
        din2 = din4;
        din3 = din2 + w_in;
        din4 = din3 + w_in;
        dout0 += w_out;
      }
    }
  }
}
#endif  // __aarch64__

}  // namespace math
}  // namespace arm
}  // namespace lite
}  // namespace paddle
