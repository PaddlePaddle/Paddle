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

#ifdef __aarch64__
int conv_3x3s2_direct_int8_c_num() { return 8; }
void conv_3x3s2_direct_int8(const int8_t* din, int32_t* dout, int num,
                            int chout, int hout, int wout, int chin, int hin,
                            int win, const int8_t* weights, const int32_t* bias,
                            const operators::ConvParam& param,
                            Context<TARGET(kARM)>* ctx, PrecisionType out_type,
                            const float* scale) {
  //! 3x3s2 int8 convolution, implemented by direct algorithm
  //! prepack input to tmp buffer
  //! write output to tmp buffer
  int threads = ctx->threads();
  int stride_w = param.strides[1];
  int pad_w = param.paddings[1];
  int pad_h = param.paddings[0];
  bool flag_relu = param.fuse_relu;
  bool flag_bias = (param.bias != nullptr);

  //! set 2/3 l2 cache
  int l2_size = ctx->l2_cache_size() / 3 * 2;
  const int hout_c_block = 8;
  const int hout_r_kernel = 2;
  const int wout_round = ((wout + 3) / 4) * 4;
  const int win_round = wout_round * stride_w + 1;

  //! get h block
  //! win_round * chin * hin_r_block * sizeof(int8_t) + wout_round *
  //! hout_c_block * hout_r_block * threads * sizeof(int32_t)= l2_size
  //! win_round = 2 * wout_round + 1
  //! hin_r_block = 2 * hout_r_block + 1
  int hout_r_block =
      (l2_size - 2 * wout_round * chin - chin) /
      ((4 * wout_round + 2) * chin + wout_round * hout_c_block * threads * 4);
  hout_r_block = hout_r_block > hout ? hout : hout_r_block;
  hout_r_block = (hout_r_block / hout_r_kernel) * hout_r_kernel;
  hout_r_block = hout_r_block < hout_r_kernel ? hout_r_kernel : hout_r_block;

  const int hin_r_block = hout_r_block * 2 + 1;

  int8_t* tmp_work_space = ctx->workspace_data<int8_t>();
  int zero_size = chout > (win_round + 3) / 4 ? chout : (win_round + 3) / 4;
  const int kZeroSize = zero_size;
  int32_t ptr_zero[kZeroSize];
  memset(ptr_zero, 0, sizeof(int32_t) * zero_size);
  const int kWoutRound = wout_round;
  int32_t ptr_write[kWoutRound];

  int in_len = win_round * chin;
  int pre_in_size = hin_r_block * in_len;
  int pre_out_size = hout_c_block * hout_r_block * wout_round;

  //! l2_cache start
  int8_t* pre_din = tmp_work_space;

  int size_in_channel = win * hin;
  int size_out_channel = wout * hout;
  int w_stride = chin * 9;

  int ws = -pad_w;
  int we = ws + win_round;
  int w_loop = wout_round / 4;

  int out_row_stride = hout_c_block * wout_round;

  for (int n = 0; n < num; ++n) {
    const int8_t* din_batch = din + n * chin * size_in_channel;
    int8_t* dout_batch =
        reinterpret_cast<int8_t*>(dout) +
        n * chout * size_out_channel * PrecisionTypeLength(out_type);
    for (int h = 0; h < hout; h += hout_r_block) {
      int h_kernel = hout_r_block;
      if (h + hout_r_block > hout) {
        h_kernel = hout - h;
      }
      int hs = h * 2 - pad_h;
      int he = hs + h_kernel * 2 + 1;
      prepack_input_nxw(din_batch, pre_din, 0, chin, hs, he, ws, we, chin, win,
                        hin, reinterpret_cast<int8_t*>(ptr_zero));

      const int8_t* cblock_inr0 = pre_din;
      const int8_t* cblock_inr1 = cblock_inr0 + in_len;
      const int8_t* cblock_inr2 = cblock_inr1 + in_len;
      const int8_t* cblock_inr3 = cblock_inr2 + in_len;
      const int8_t* cblock_inr4 = cblock_inr3 + in_len;

#pragma omp parallel for num_threads(threads)
      for (int c = 0; c < chout; c += hout_c_block) {
#ifdef ARM_WITH_OMP
        int32_t* pre_out =
            reinterpret_cast<int*>(pre_din + (pre_in_size + 3) / 4 * 4) +
            omp_get_thread_num() * pre_out_size;
#else
        int32_t* pre_out =
            reinterpret_cast<int32_t*>(pre_din + (pre_in_size + 3) / 4 * 4);
#endif
        const int8_t* block_inr0 = cblock_inr0;
        const int8_t* block_inr1 = cblock_inr1;
        const int8_t* block_inr2 = cblock_inr2;
        const int8_t* block_inr3 = cblock_inr3;
        const int8_t* block_inr4 = cblock_inr4;

        const int8_t* weight_c = weights + c * w_stride;
        const int32_t* bias_ptr = ptr_zero;
        if (flag_bias) {
          bias_ptr = bias + c;
        }

        fill_packed_bias_nxmw_int8(bias_ptr, pre_out, 8, h_kernel, wout_round);
        for (int hk = 0; hk < h_kernel; hk += hout_r_kernel) {
          const int8_t* wc0 = weight_c;

          const int8_t* inr0 = block_inr0;
          const int8_t* inr1 = block_inr1;
          const int8_t* inr2 = block_inr2;
          const int8_t* inr3 = block_inr3;
          const int8_t* inr4 = block_inr4;

          int32_t* pre_out0 = pre_out + hk * out_row_stride;
          int32_t* pre_out1 = pre_out0 + out_row_stride;
          for (int i = 0; i < chin; ++i) {
            int16x8_t v0 = vmovl_s8(vld1_s8(wc0));       // w0
            int16x8_t v1 = vmovl_s8(vld1_s8(wc0 + 8));   // w1
            int16x8_t v2 = vmovl_s8(vld1_s8(wc0 + 16));  // w2,

            int16x8_t v3 = vmovl_s8(vld1_s8(wc0 + 24));  // w3
            int16x8_t v4 = vmovl_s8(vld1_s8(wc0 + 32));  // w4
            int16x8_t v5 = vmovl_s8(vld1_s8(wc0 + 40));  // w5

            int16x8_t v6 = vmovl_s8(vld1_s8(wc0 + 48));  // w6
            int16x8_t v7 = vmovl_s8(vld1_s8(wc0 + 56));  // w7
            int16x8_t v8 = vmovl_s8(vld1_s8(wc0 + 64));  // w8

            const int8_t* r0 = inr0;
            const int8_t* r1 = inr1;
            const int8_t* r2 = inr2;
            const int8_t* r3 = inr3;
            const int8_t* r4 = inr4;

            int32_t* ptr_out0 = pre_out0;
            int32_t* ptr_out1 = pre_out1;
            int cnt = w_loop;

            asm volatile(
                "ldr    q0,    [%[r0]], #8  \n" /* load input r0 */
                "ldr    q1,    [%[r2]], #8  \n" /* load input r2 */
                "sshll  v0.8h, v0.8b, #0    \n" /*  r0: int8 -> int16 */
                "sshll  v1.8h, v1.8b, #0    \n" /*  r1: int8 -> int16*/
                "1:                         \n" /* main loop */

                /* r0, r2 mul w00 */
                "smull   v4.4s,   %[v0].4h,  v0.h[0]\n" /* outr00 = v0 * r0[0]
                                                           */
                "smull2  v5.4s,   %[v0].8h,  v0.h[0]\n" /* outr00 = v0 * r0[0]
                                                           */
                "smull   v6.4s,   %[v0].4h,  v0.h[2]\n" /* outr01 = v0 * r0[2]
                                                           */
                "smull2  v7.4s,   %[v0].8h,  v0.h[2]\n" /* outr00 = v0 * r0[0]
                                                           */
                "smull   v8.4s,   %[v0].4h,  v0.h[4]\n" /* outr02 = v0 * r0[4]
                                                           */
                "smull2  v9.4s,   %[v0].8h,  v0.h[4]\n" /* outr00 = v0 * r0[0]
                                                           */
                "smull   v10.4s,  %[v0].4h,  v0.h[6]\n" /* outr03 = v0 * r0[6]
                                                           */
                "smull2  v11.4s,  %[v0].8h,  v0.h[6]\n" /* outr00 = v0 * r0[0]
                                                           */

                "smull   v12.4s,  %[v0].4h,  v1.h[0]\n" /* outr10 = v0 * r2[0]
                                                           */
                "smull2  v13.4s,  %[v0].8h,  v1.h[0]\n" /* outr11 = v0 * r2[2]
                                                           */
                "smull   v14.4s,  %[v0].4h,  v1.h[2]\n" /* outr12 = v0 * r2[4]
                                                           */
                "smull2  v15.4s,  %[v0].8h,  v1.h[2]\n" /* outr13 = v0 * r2[6]
                                                           */
                "smull   v16.4s,  %[v0].4h,  v1.h[4]\n" /* outr10 = v0 * r2[0]
                                                           */
                "smull2  v17.4s,  %[v0].8h,  v1.h[4]\n" /* outr11 = v0 * r2[2]
                                                           */
                "smull   v18.4s,  %[v0].4h,  v1.h[6]\n" /* outr12 = v0 * r2[4]
                                                           */
                "smull2  v19.4s,  %[v0].8h,  v1.h[6]\n" /* outr13 = v0 * r2[6]
                                                           */

                /* r2, mul w06 */
                "smlal   v4.4s,   %[v6].4h,  v1.h[0]\n" /* outr00 = v6 * r2[1]
                                                           */
                "smlal2  v5.4s,   %[v6].8h,  v1.h[0]\n" /* outr01 = v6 * r2[3]
                                                           */
                "smlal   v6.4s,   %[v6].4h,  v1.h[2]\n" /* outr02 = v6 * r2[5]
                                                           */
                "smlal2  v7.4s,   %[v6].8h,  v1.h[2]\n" /* outr03 = v6 * r2[7]
                                                           */
                "smlal   v8.4s,   %[v6].4h,  v1.h[4]\n" /* outr00 = v6 * r2[1]
                                                           */
                "smlal2  v9.4s,   %[v6].8h,  v1.h[4]\n" /* outr01 = v6 * r2[3]
                                                           */
                "smlal   v10.4s,  %[v6].4h,  v1.h[6]\n" /* outr02 = v6 * r2[5]
                                                           */
                "smlal2  v11.4s,  %[v6].8h,  v1.h[6]\n" /* outr03 = v6 * r2[7]
                                                           */

                "ldr    q2,      [%[r0]]        \n" /* load r0, 9th
                                                       data,v10.s[0] */

                /*  r0, r2, mul w01 */
                "smlal   v4.4s,   %[v1].4h,  v0.h[1]\n" /* outr00 = v0 * r0[0]
                                                           */
                "smlal2  v5.4s,   %[v1].8h,  v0.h[1]\n" /* outr00 = v0 * r0[0]
                                                           */
                "smlal   v6.4s,   %[v1].4h,  v0.h[3]\n" /* outr01 = v0 * r0[2]
                                                           */
                "smlal2  v7.4s,   %[v1].8h,  v0.h[3]\n" /* outr00 = v0 * r0[0]
                                                           */
                "sshll   v2.8h,   v2.8b,     #0     \n" /*  r0: int8 -> int16 */
                "smlal   v8.4s,   %[v1].4h,  v0.h[5]\n" /* outr02 = v0 * r0[4]
                                                           */
                "smlal2  v9.4s,   %[v1].8h,  v0.h[5]\n" /* outr00 = v0 * r0[0]
                                                           */
                "smlal   v10.4s,  %[v1].4h,  v0.h[7]\n" /* outr03 = v0 * r0[6]
                                                           */
                "smlal2  v11.4s,  %[v1].8h,  v0.h[7]\n" /* outr00 = v0 * r0[0]
                                                           */

                "smlal   v12.4s,  %[v1].4h,  v1.h[1]\n" /* outr10 = v0 * r2[0]
                                                           */
                "smlal2  v13.4s,  %[v1].8h,  v1.h[1]\n" /* outr11 = v0 * r2[2]
                                                           */
                "smlal   v14.4s,  %[v1].4h,  v1.h[3]\n" /* outr12 = v0 * r2[4]
                                                           */
                "smlal2  v15.4s,  %[v1].8h,  v1.h[3]\n" /* outr13 = v0 * r2[6]
                                                           */
                "smlal   v16.4s,  %[v1].4h,  v1.h[5]\n" /* outr10 = v0 * r2[0]
                                                           */
                "smlal2  v17.4s,  %[v1].8h,  v1.h[5]\n" /* outr11 = v0 * r2[2]
                                                           */
                "smlal   v18.4s,  %[v1].4h,  v1.h[7]\n" /* outr12 = v0 * r2[4]
                                                           */
                "smlal2  v19.4s,  %[v1].8h,  v1.h[7]\n" /* outr13 = v0 * r2[6]
                                                           */

                /* r2, mul w07 */
                "smlal   v4.4s,   %[v7].4h,  v1.h[1]\n" /* outr00 = v6 * r2[1]
                                                           */
                "smlal2  v5.4s,   %[v7].8h,  v1.h[1]\n" /* outr01 = v6 * r2[3]
                                                           */
                "smlal   v6.4s,   %[v7].4h,  v1.h[3]\n" /* outr02 = v6 * r2[5]
                                                           */
                "smlal2  v7.4s,   %[v7].8h,  v1.h[3]\n" /* outr03 = v6 * r2[7]
                                                           */
                "smlal   v8.4s,   %[v7].4h,  v1.h[5]\n" /* outr00 = v6 * r2[1]
                                                           */
                "smlal2  v9.4s,   %[v7].8h,  v1.h[5]\n" /* outr01 = v6 * r2[3]
                                                           */
                "smlal   v10.4s,  %[v7].4h,  v1.h[7]\n" /* outr02 = v6 * r2[5]
                                                           */
                "smlal2  v11.4s,  %[v7].8h,  v1.h[7]\n" /* outr03 = v6 * r2[7]
                                                           */

                "ldr     q3,      [%[r2]]        \n" /* load r2, 9th
                                                        data,v11.s[0] */

                /*  r0, r2, mul w02 */
                "smlal   v4.4s,   %[v2].4h,  v0.h[2]\n" /* outr00 = v0 * r0[0]
                                                           */
                "smlal2  v5.4s,   %[v2].8h,  v0.h[2]\n" /* outr00 = v0 * r0[0]
                                                           */
                "smlal   v6.4s,   %[v2].4h,  v0.h[4]\n" /* outr01 = v0 * r0[2]
                                                           */
                "smlal2  v7.4s,   %[v2].8h,  v0.h[4]\n" /* outr00 = v0 * r0[0]
                                                           */
                "sshll   v3.8h,   v3.8b,     #0     \n" /* r2: int8 -> int16*/
                "smlal   v8.4s,   %[v2].4h,  v0.h[6]\n" /* outr02 = v0 * r0[4]
                                                           */
                "smlal2  v9.4s,   %[v2].8h,  v0.h[6]\n" /* outr00 = v0 * r0[0]
                                                           */
                "smlal   v10.4s,  %[v2].4h,  v2.h[0]\n" /* outr03 = v0 * r0[6]
                                                           */
                "smlal2  v11.4s,  %[v2].8h,  v2.h[0]\n" /* outr00 = v0 * r0[0]
                                                           */

                "ldr     q0, [%[r1]], #8 \n" /* load input r1 */

                "smlal   v12.4s,  %[v2].4h,  v1.h[2]\n" /* outr10 = v0 * r2[0]
                                                           */
                "smlal2  v13.4s,  %[v2].8h,  v1.h[2]\n" /* outr11 = v0 * r2[2]
                                                           */
                "smlal   v14.4s,  %[v2].4h,  v1.h[4]\n" /* outr12 = v0 * r2[4]
                                                           */
                "smlal2  v15.4s,  %[v2].8h,  v1.h[4]\n" /* outr13 = v0 * r2[6]
                                                           */
                "sshll   v0.8h,   v0.8b,     #0     \n" /* r1 : int8 -> int16 */
                "smlal   v16.4s,  %[v2].4h,  v1.h[6]\n" /* outr10 = v0 * r2[0]
                                                           */
                "smlal2  v17.4s,  %[v2].8h,  v1.h[6]\n" /* outr11 = v0 * r2[2]
                                                           */
                "smlal   v18.4s,  %[v2].4h,  v3.h[0]\n" /* outr12 = v0 * r2[4]
                                                           */
                "smlal2  v19.4s,  %[v2].8h,  v3.h[0]\n" /* outr13 = v0 * r2[6]
                                                           */

                /* r2, mul w08 */
                "smlal   v4.4s,   %[v8].4h,  v1.h[2]\n" /* outr00 = v6 * r2[1]
                                                           */
                "smlal2  v5.4s,   %[v8].8h,  v1.h[2]\n" /* outr01 = v6 * r2[3]
                                                           */
                "smlal   v6.4s,   %[v8].4h,  v1.h[4]\n" /* outr02 = v6 * r2[5]
                                                           */
                "smlal2  v7.4s,   %[v8].8h,  v1.h[4]\n" /* outr03 = v6 * r2[7]
                                                           */
                "smlal   v8.4s,   %[v8].4h,  v1.h[6]\n" /* outr00 = v6 * r2[1]
                                                           */
                "smlal2  v9.4s,   %[v8].8h,  v1.h[6]\n" /* outr01 = v6 * r2[3]
                                                           */
                "smlal   v10.4s,  %[v8].4h,  v3.h[0]\n" /* outr02 = v6 * r2[5]
                                                           */
                "smlal2  v11.4s,  %[v8].8h,  v3.h[0]\n" /* outr03 = v6 * r2[7]
                                                           */

                "ldr     q1, [%[r3]], #8 \n" /* load input r3 */

                /*  r1, r3, mul w03 */
                "smlal   v4.4s,   %[v3].4h,  v0.h[0]\n" /* outr00 = v0 * r0[0]
                                                           */
                "smlal2  v5.4s,   %[v3].8h,  v0.h[0]\n" /* outr00 = v0 * r0[0]
                                                           */
                "smlal   v6.4s,   %[v3].4h,  v0.h[2]\n" /* outr01 = v0 * r0[2]
                                                           */
                "smlal2  v7.4s,   %[v3].8h,  v0.h[2]\n" /* outr00 = v0 * r0[0]
                                                           */
                "sshll   v1.8h,   v1.8b,     #0     \n" /* r3: int8 -> int16 */
                "smlal   v8.4s,   %[v3].4h,  v0.h[4]\n" /* outr02 = v0 * r0[4]
                                                           */
                "smlal2  v9.4s,   %[v3].8h,  v0.h[4]\n" /* outr00 = v0 * r0[0]
                                                           */
                "smlal   v10.4s,  %[v3].4h,  v0.h[6]\n" /* outr03 = v0 * r0[6]
                                                           */
                "smlal2  v11.4s,  %[v3].8h,  v0.h[6]\n" /* outr00 = v0 * r0[0]
                                                           */
                "ldr     q2,       [%[r1]]          \n" /* load r1, 9th
                                                           data,v10.s[0] */

                "smlal   v12.4s,  %[v3].4h,  v1.h[0]\n" /* outr10 = v0 * r2[0]
                                                           */
                "smlal2  v13.4s,  %[v3].8h,  v1.h[0]\n" /* outr11 = v0 * r2[2]
                                                           */
                "smlal   v14.4s,  %[v3].4h,  v1.h[2]\n" /* outr12 = v0 * r2[4]
                                                           */
                "smlal2  v15.4s,  %[v3].8h,  v1.h[2]\n" /* outr13 = v0 * r2[6]
                                                           */
                "ldr     q3,      [%[r3]]          \n"  /* load r3, 9th
                                                           data,v11.s[0] */
                "smlal   v16.4s,  %[v3].4h,  v1.h[4]\n" /* outr10 = v0 * r2[0]
                                                           */
                "smlal2  v17.4s,  %[v3].8h,  v1.h[4]\n" /* outr11 = v0 * r2[2]
                                                           */
                "smlal   v18.4s,  %[v3].4h,  v1.h[6]\n" /* outr12 = v0 * r2[4]
                                                           */
                "smlal2  v19.4s,  %[v3].8h,  v1.h[6]\n" /* outr13 = v0 * r2[6]
                                                           */
                "sshll v2.8h, v2.8b, #0 \n"             /* r1 : int8 -> int16 */

                /*  r1, r3, mul w05 */
                "smlal   v4.4s,   %[v5].4h,  v0.h[2]\n" /* outr00 = v0 * r0[0]
                                                           */
                "smlal2  v5.4s,   %[v5].8h,  v0.h[2]\n" /* outr00 = v0 * r0[0]
                                                           */
                "smlal   v6.4s,   %[v5].4h,  v0.h[4]\n" /* outr01 = v0 * r0[2]
                                                           */
                "smlal2  v7.4s,   %[v5].8h,  v0.h[4]\n" /* outr00 = v0 * r0[0]
                                                           */
                "sshll   v3.8h,   v3.8b,     #0     \n" /* r3 : int8 -> int16 */
                "smlal   v8.4s,   %[v5].4h,  v0.h[6]\n" /* outr02 = v0 * r0[4]
                                                           */
                "smlal2  v9.4s,   %[v5].8h,  v0.h[6]\n" /* outr00 = v0 * r0[0]
                                                           */
                "smlal   v10.4s,  %[v5].4h,  v2.h[0]\n" /* outr03 = v0 * r0[6]
                                                           */
                "smlal2  v11.4s,  %[v5].8h,  v2.h[0]\n" /* outr00 = v0 * r0[0]
                                                           */

                "smlal   v12.4s,  %[v5].4h,  v1.h[2]\n" /* outr10 = v0 * r2[0]
                                                           */
                "smlal2  v13.4s,  %[v5].8h,  v1.h[2]\n" /* outr11 = v0 * r2[2]
                                                           */
                "smlal   v14.4s,  %[v5].4h,  v1.h[4]\n" /* outr12 = v0 * r2[4]
                                                           */
                "smlal2  v15.4s,  %[v5].8h,  v1.h[4]\n" /* outr13 = v0 * r2[6]
                                                           */
                "smlal   v16.4s,  %[v5].4h,  v1.h[6]\n" /* outr10 = v0 * r2[0]
                                                           */
                "smlal2  v17.4s,  %[v5].8h,  v1.h[6]\n" /* outr11 = v0 * r2[2]
                                                           */
                "smlal   v18.4s,  %[v5].4h,  v3.h[0]\n" /* outr12 = v0 * r2[4]
                                                           */
                "smlal2  v19.4s,  %[v5].8h,  v3.h[0]\n" /* outr13 = v0 * r2[6]
                                                           */

                "subs    %w[cnt], %w[cnt], #1       \n" /* loop count -1 */

                /*  r1, r3, mul w04 */
                "smlal   v4.4s,   %[v4].4h,  v0.h[1]\n" /* outr00 = v0 * r0[0]
                                                           */
                "smlal2  v5.4s,   %[v4].8h,  v0.h[1]\n" /* outr00 = v0 * r0[0]
                                                           */
                "smlal   v6.4s,   %[v4].4h,  v0.h[3]\n" /* outr01 = v0 * r0[2]
                                                           */
                "smlal2  v7.4s,   %[v4].8h,  v0.h[3]\n" /* outr00 = v0 * r0[0]
                                                           */
                "smlal   v8.4s,   %[v4].4h,  v0.h[5]\n" /* outr02 = v0 * r0[4]
                                                           */
                "smlal2  v9.4s,   %[v4].8h,  v0.h[5]\n" /* outr00 = v0 * r0[0]
                                                           */
                "smlal   v10.4s,  %[v4].4h,  v0.h[7]\n" /* outr03 = v0 * r0[6]
                                                           */
                "smlal2  v11.4s,  %[v4].8h,  v0.h[7]\n" /* outr00 = v0 * r0[0]
                                                           */

                "ldr     q0, [%[r4]], #8            \n" /* load input r4 */

                "smlal   v12.4s,  %[v4].4h,  v1.h[1]\n" /* outr10 = v0 * r2[0]
                                                           */
                "smlal2  v13.4s,  %[v4].8h,  v1.h[1]\n" /* outr11 = v0 * r2[2]
                                                           */
                "smlal   v14.4s,  %[v4].4h,  v1.h[3]\n" /* outr12 = v0 * r2[4]
                                                           */
                "smlal2  v15.4s,  %[v4].8h,  v1.h[3]\n" /* outr13 = v0 * r2[6]
                                                           */
                "sshll   v0.8h,   v0.8b,     #0     \n" /* r4 : int8 -> int16 */
                "smlal   v16.4s,  %[v4].4h,  v1.h[5]\n" /* outr10 = v0 * r2[0]
                                                           */
                "smlal2  v17.4s,  %[v4].8h,  v1.h[5]\n" /* outr11 = v0 * r2[2]
                                                           */
                "smlal   v18.4s,  %[v4].4h,  v1.h[7]\n" /* outr12 = v0 * r2[4]
                                                           */
                "smlal2  v19.4s,  %[v4].8h,  v1.h[7]\n" /* outr13 = v0 * r2[6]
                                                           */

                "ldr     q2,      [%[r4]]           \n" /* load r4, 9th
                                                           data,v10.s[0] */
                "sshll   v2.8h,   v2.8b,     #0     \n" /* r4 : int8 -> int16 */

                "ldp     q1, q3, [%[ptr_out0]]      \n"  /* load ptr_out + 0  ->
                                                            q2, q3 */
                "ldp     q20, q21, [%[ptr_out0], #32]\n" /* load ptr_out + 32 ->
                                                            q4, q5 */

                "add     v4.4s,  v1.4s ,  v4.4s     \n" /* v10 = outr00[0].low
                                                           + q2 */
                "add     v5.4s,  v3.4s ,  v5.4s     \n" /* v11 = outr00[0].high
                                                           + q3 */
                "add     v6.4s,  v20.4s,  v6.4s     \n" /* v12 = outr01[0].low
                                                           + q4 */
                "add     v7.4s,  v21.4s,  v7.4s     \n" /* v13 = outr01[0].high
                                                           + q5 */

                "ldp     q1 , q3 , [%[ptr_out0], #64]\n" /* load ptr_out + 64 ->
                                                            q6, q7 */
                "ldp     q20, q21, [%[ptr_out0], #96]\n" /* load ptr_out + 96 ->
                                                            q8, q9 */

                "stp     q4,  q5 , [%[ptr_out0]], #32\n" /* store q10, q11 ->
                                                            ptr_out   */
                "stp     q6,  q7 , [%[ptr_out0]], #32\n" /* store q10, q11 ->
                                                            ptr_out   */

                "add     v8.4s ,  v1.4s ,  v8.4s     \n" /* v10 = outr00[0].low
                                                            + q2 */
                "add     v9.4s ,  v3.4s ,  v9.4s     \n" /* v11 = outr00[0].high
                                                            + q3 */
                "add     v10.4s,  v20.4s,  v10.4s    \n" /* v12 = outr01[0].low
                                                            + q4 */
                "add     v11.4s,  v21.4s,  v11.4s    \n" /* v13 = outr01[0].high
                                                            + q5 */
                "stp     q8,  q9,  [%[ptr_out0]], #32\n" /* store q14, q15 ->
                                                            ptr_out += 64 */
                "stp     q10, q11, [%[ptr_out0]], #32\n" /* store q16, q17 ->
                                                            ptr_out += 96 */

                /* r4, mul w08 */
                "smlal   v12.4s,   %[v8].4h,  v0.h[2]\n" /* outr00 = v0 * r0[0]
                                                            */
                "smlal2  v13.4s,   %[v8].8h,  v0.h[2]\n" /* outr00 = v0 * r0[0]
                                                            */
                "smlal   v14.4s,   %[v8].4h,  v0.h[4]\n" /* outr01 = v0 * r0[2]
                                                            */
                "smlal2  v15.4s,   %[v8].8h,  v0.h[4]\n" /* outr00 = v0 * r0[0]
                                                            */

                "smlal   v16.4s,   %[v8].4h,  v0.h[6]\n" /* outr02 = v0 * r0[4]
                                                            */
                "smlal2  v17.4s,   %[v8].8h,  v0.h[6]\n" /* outr00 = v0 * r0[0]
                                                            */
                "smlal   v18.4s,   %[v8].4h,  v2.h[0]\n" /* outr03 = v0 * r0[6]
                                                            */
                "smlal2  v19.4s,   %[v8].8h,  v2.h[0]\n" /* outr00 = v0 * r0[0]
                                                            */

                /* r4, mul w07 */
                "smlal   v12.4s,   %[v7].4h,  v0.h[1]\n" /* outr00 = v0 * r0[0]
                                                            */
                "smlal2  v13.4s,   %[v7].8h,  v0.h[1]\n" /* outr00 = v0 * r0[0]
                                                            */
                "smlal   v14.4s,   %[v7].4h,  v0.h[3]\n" /* outr01 = v0 * r0[2]
                                                            */
                "smlal2  v15.4s,   %[v7].8h,  v0.h[3]\n" /* outr00 = v0 * r0[0]
                                                            */

                "ldr     q1,   [%[r2]], #8            \n" /* load input r2 */

                "smlal   v16.4s,   %[v7].4h,  v0.h[5]\n" /* outr02 = v0 * r0[4]
                                                            */
                "smlal2  v17.4s,   %[v7].8h,  v0.h[5]\n" /* outr00 = v0 * r0[0]
                                                            */
                "smlal   v18.4s,   %[v7].4h,  v0.h[7]\n" /* outr03 = v0 * r0[6]
                                                            */
                "smlal2  v19.4s,   %[v7].8h,  v0.h[7]\n" /* outr00 = v0 * r0[0]
                                                            */

                "sshll   v1.8h,    v1.8b,     #0     \n" /*  r2: int8 -> int16
                                                            */

                /* r4, mul w06 */
                "ldp     q4,  q5,  [%[ptr_out1]]     \n" /* load ptr_out + 0  ->
                                                            q2, q3 */

                "smlal   v12.4s,   %[v6].4h,  v0.h[0]\n" /* outr00 = v0 * r0[0]
                                                            */
                "smlal2  v13.4s,   %[v6].8h,  v0.h[0]\n" /* outr00 = v0 * r0[0]
                                                            */
                "smlal   v14.4s,   %[v6].4h,  v0.h[2]\n" /* outr01 = v0 * r0[2]
                                                            */

                "ldp     q8,  q9,  [%[ptr_out1], #64]\n" /* load ptr_out + 64 ->
                                                            q6, q7 */

                "smlal2  v15.4s,   %[v6].8h,  v0.h[2]\n" /* outr00 = v0 * r0[0]
                                                            */
                "smlal   v16.4s,   %[v6].4h,  v0.h[4]\n" /* outr02 = v0 * r0[4]
                                                            */
                "smlal2  v17.4s,   %[v6].8h,  v0.h[4]\n" /* outr00 = v0 * r0[0]
                                                            */

                "ldp     q10, q11, [%[ptr_out1], #96]\n" /* load ptr_out + 96 ->
                                                            q8, q9 */

                "smlal   v18.4s,   %[v6].4h,  v0.h[6]\n" /* outr03 = v0 * r0[6]
                                                            */
                "smlal2  v19.4s,   %[v6].8h,  v0.h[6]\n" /* outr00 = v0 * r0[0]
                                                            */

                "ldr     q0,   [%[r0]], #8           \n" /* load input r2 */
                "ldp     q6,   q7, [%[ptr_out1], #32]\n" /* load ptr_out + 32 ->
                                                            q4, q5 */

                "sshll   v0.8h, v0.8b, #0            \n" /* r0: int8 -> int16 */

                /* store outr1 */
                "add   v12.4s, v4.4s , v12.4s\n" /* v10 = outr10[0].low  + q2 */
                "add   v13.4s, v5.4s , v13.4s\n" /* v11 = outr10[0].high + q3 */
                "add   v14.4s, v6.4s , v14.4s\n" /* v12 = outr11[0].low  + q4 */
                "add   v15.4s, v7.4s , v15.4s\n" /* v13 = outr11[0].high + q5 */

                "stp   q12, q13, [%[ptr_out1]], #32\n" /* store q10, q11 ->
                                                          ptr_out       */

                "add   v16.4s, v8.4s , v16.4s\n" /* v14 = outr12[0].low  + q6 */
                "add   v17.4s, v9.4s , v17.4s\n" /* v15 = outr12[0].high + q7 */

                "stp   q14, q15, [%[ptr_out1]], #32\n" /* store q12, q13 ->
                                                          ptr_out += 32 */

                "add   v18.4s, v10.4s, v18.4s\n" /* v16 = outr13[0].low  + q8 */
                "add   v19.4s, v11.4s, v19.4s\n" /* v17 = outr13[0].high + q9 */

                "stp   q16, q17, [%[ptr_out1]], #32\n" /* store q14, q15 ->
                                                          ptr_out += 64 */
                "stp   q18, q19, [%[ptr_out1]], #32\n" /* store q16, q17 ->
                                                          ptr_out += 96 */

                "bne     1b                        \n" /* jump to main loop */

                : [cnt] "+r"(cnt), [r0] "+r"(r0), [r1] "+r"(r1), [r2] "+r"(r2),
                  [r3] "+r"(r3), [r4] "+r"(r4), [ptr_out0] "+r"(ptr_out0),
                  [ptr_out1] "+r"(ptr_out1)
                : [v0] "w"(v0), [v1] "w"(v1), [v2] "w"(v2), [v3] "w"(v3),
                  [v4] "w"(v4), [v5] "w"(v5), [v6] "w"(v6), [v7] "w"(v7),
                  [v8] "w"(v8)
                : "cc", "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6",
                  "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15",
                  "v16", "v17", "v18", "v19", "v20", "v21", "v22");

            wc0 += 9 * hout_c_block;
            inr0 += win_round;
            inr1 += win_round;
            inr2 += win_round;
            inr3 += win_round;
            inr4 += win_round;
          }
          block_inr0 = block_inr4;
          block_inr1 = block_inr0 + in_len;
          block_inr2 = block_inr1 + in_len;
          block_inr3 = block_inr2 + in_len;
          block_inr4 = block_inr3 + in_len;
        }
        if (out_type == PRECISION(kFloat)) {
          write_to_output_c8_int32_1(
              pre_out, reinterpret_cast<float*>(dout_batch), hout_c_block, 2, c,
              c + hout_c_block, h, h + h_kernel, 0, wout_round, chout, hout,
              wout, flag_relu, reinterpret_cast<float*>(ptr_write), &scale[c],
              out_type);
        } else if (out_type == PRECISION(kInt8)) {
          write_to_output_c8_int32_1(
              pre_out, dout_batch, hout_c_block, 2, c, c + hout_c_block, h,
              h + h_kernel, 0, wout_round, chout, hout, wout, flag_relu,
              reinterpret_cast<signed char*>(ptr_write), &scale[c], out_type);
        } else {
          write_to_output_c8_int32(pre_out, reinterpret_cast<int*>(dout_batch),
                                   hout_c_block, 2, c, c + hout_c_block, h,
                                   h + h_kernel, 0, wout_round, chout, hout,
                                   wout, flag_relu, ptr_write);
        }
      }
    }
  }
}

#else  // __aarch64__
int conv_3x3s2_direct_int8_c_num() { return 4; }
void conv_3x3s2_direct_int8(const int8_t* din, int32_t* dout, int num,
                            int chout, int hout, int wout, int chin, int hin,
                            int win, const int8_t* weights, const int32_t* bias,
                            const operators::ConvParam& param,
                            Context<TARGET(kARM)>* ctx, PrecisionType out_type,
                            const float* scale) {
  //! 3x3s2 int8 convolution, implemented by direct algorithm
  //! prepack input to tmp buffer
  //! write output to tmp buffer
  int threads = ctx->threads();
  int stride_w = param.strides[1];
  int pad_w = param.paddings[1];
  int pad_h = param.paddings[0];
  bool flag_relu = param.fuse_relu;
  bool flag_bias = (param.bias != nullptr);

  //! set 2/3 l2 cache
  int l2_size = ctx->l2_cache_size() / 3 * 2;
  const int hout_c_block = 4;
  const int hout_r_kernel = 1;
  const int wout_round = ((wout + 3) / 4) * 4;
  const int win_round = wout_round * stride_w + 1;

  //! get h block
  //! win_round * chin * hin_r_block * sizeof(int8_t) + wout_round *
  //! hout_c_block * hout_r_block * threads * sizeof(int32_t)= l2_size
  //! win_round = 2 * wout_round + 1
  //! hin_r_block = 2 * hout_r_block + 1
  int hout_r_block =
      (l2_size - 2 * wout_round * chin - chin) /
      ((4 * wout_round + 2) * chin + wout_round * hout_c_block * threads * 4);
  hout_r_block = hout_r_block > hout ? hout : hout_r_block;
  hout_r_block = (hout_r_block / hout_r_kernel) * hout_r_kernel;
  hout_r_block = hout_r_block < hout_r_kernel ? hout_r_kernel : hout_r_block;

  const int hin_r_block = hout_r_block * 2 + 1;

  int8_t* tmp_work_space = ctx->workspace_data<int8_t>();
  int zero_size = chout > (win_round + 3) / 4 ? chout : (win_round + 3) / 4;
  const int kZeroSize = zero_size;
  int32_t ptr_zero[kZeroSize];
  memset(ptr_zero, 0, sizeof(int32_t) * zero_size);
  const int kWoutRound = wout_round;
  int32_t ptr_write[kWoutRound];

  int in_len = win_round * chin;
  int pre_in_size = hin_r_block * in_len;
  int pre_out_size = hout_c_block * hout_r_block * wout_round;

  //! l2_cache start
  int8_t* pre_din = tmp_work_space;

  int size_in_channel = win * hin;
  int size_out_channel = wout * hout;
  int w_stride = chin * 9;

  int ws = -pad_w;
  int we = ws + win_round;
  int w_loop = wout_round / 4;

  int out_row_stride = hout_c_block * wout_round;

  for (int n = 0; n < num; ++n) {
    const int8_t* din_batch = din + n * chin * size_in_channel;
    int8_t* dout_batch =
        reinterpret_cast<int8_t*>(dout) +
        n * chout * size_out_channel * PrecisionTypeLength(out_type);
    for (int h = 0; h < hout; h += hout_r_block) {
      int h_kernel = hout_r_block;
      if (h + hout_r_block > hout) {
        h_kernel = hout - h;
      }
      int hs = h * 2 - pad_h;
      int he = hs + h_kernel * 2 + 1;
      prepack_input_nxw(din_batch, pre_din, 0, chin, hs, he, ws, we, chin, win,
                        hin, reinterpret_cast<int8_t*>(ptr_zero));

      const int8_t* cblock_inr0 = pre_din;
      const int8_t* cblock_inr1 = cblock_inr0 + in_len;
      const int8_t* cblock_inr2 = cblock_inr1 + in_len;
#pragma omp parallel for num_threads(threads)
      for (int c = 0; c < chout; c += hout_c_block) {
#ifdef ARM_WITH_OMP
        int32_t* pre_out =
            reinterpret_cast<int*>(pre_din + (pre_in_size + 3) / 4 * 4) +
            omp_get_thread_num() * pre_out_size;
#else
        int32_t* pre_out =
            reinterpret_cast<int32_t*>(pre_din + (pre_in_size + 3) / 4 * 4);
#endif
        const int8_t* block_inr0 = cblock_inr0;
        const int8_t* block_inr1 = cblock_inr1;
        const int8_t* block_inr2 = cblock_inr2;

        const int8_t* weight_c = weights + c * w_stride;
        const int32_t* bias_ptr = ptr_zero;
        if (flag_bias) {
          bias_ptr = bias + c;
        }

        fill_packed_bias_nxmw_int8(bias_ptr, pre_out, 4, h_kernel, wout_round);
        for (int hk = 0; hk < h_kernel; hk += hout_r_kernel) {
          const int8_t* wc0 = weight_c;

          const int8_t* inr0 = block_inr0;
          const int8_t* inr1 = block_inr1;
          const int8_t* inr2 = block_inr2;

          int32_t* pre_out0 = pre_out + hk * out_row_stride;
          for (int i = 0; i < chin; ++i) {
            const int8_t* r0 = inr0;
            const int8_t* r1 = inr1;
            const int8_t* r2 = inr2;

            int32_t* ptr_out0 = pre_out0;
            const signed char* ptr_wc0 = wc0;
            int cnt = w_loop;
            asm volatile(
                "vld1.s32   {d0-d3}, [%[wc0]]!  \n" /* w0-w7 */
                "vld1.s32   {d4},   [%[wc0]]!   \n" /* w8 */
                "vmovl.s8   q3,   d0            \n" /* q3 = w0, w1 */
                "vmovl.s8   q4,   d1            \n" /* q4 = w2 ,w3 */
                "vmovl.s8   q5,   d2            \n" /* q5 = w4, w5 */
                "vmovl.s8   q6,   d3            \n" /* q6 = w6, w7 */
                "vmovl.s8   q7,   d4            \n" /* q7 = w8 */
                "vld1.s32   {d0}, [%[r0]]!      \n" /* load input r0 -> d0 */
                "vmovl.s8   q0,   d0            \n" /* movl d0 -> q0 */
                "1:                             \n" /* main loop */

                /* r0 mul w0 */
                "vmull.s16 q8, d6, d0[0]   \n" /* q8 = w0 * r0[0] */
                "vmull.s16 q9, d6, d0[2]   \n" /* q9 = w0 * r0[2] */
                "vmull.s16 q10, d6, d1[0]  \n" /* q10 = w0 * r0[4] */
                "vmull.s16 q11, d6, d1[2]  \n" /* q11 = w0 * r0[6] */

                "vld1.s32 {d2}, [%[r1]]!   \n" /* load input r1 -> d2 */
                "vmovl.s8 q1,   d2         \n" /* movl d2 -> q1 */

                /* r0 mul w1 */
                "vmlal.s16 q8, d7, d0[1]   \n" /* q8 = w1 * r0[1] */
                "vmlal.s16 q9, d7, d0[3]   \n" /* q9 = w1 * r0[3] */
                "vmlal.s16 q10, d7, d1[1]  \n" /* q10 = w1 * r0[5] */
                "vmlal.s16 q11, d7, d1[3]  \n" /* q11 = w1 * r0[7] */

                "vld1.s32 {d4}, [%[r0]]    \n" /* load r0[8] -> d4 */
                "vmovl.s8 q2  ,  d4        \n" /* movl d4 -> q2 */

                /* r0 mul w2 */
                "vmlal.s16 q8, d8, d0[2]   \n" /* q8 = w2 * r0[2] */
                "vmlal.s16 q9, d8, d1[0]   \n" /* q9 = w2 * r0[4] */
                "vmlal.s16 q10, d8, d1[2]  \n" /* q10 = w2 * r0[6] */
                "vmlal.s16 q11, d8, d4[0]  \n" /* q11 = w2 * r0[8] */

                "subs       %[cnt], #1     \n" /* loop count -1 */

                /* r1 mul w3 */
                "vmlal.s16 q8, d9, d2[0]   \n" /* q8 = w3 * r1[0] */
                "vmlal.s16 q9, d9, d2[2]   \n" /* q9 = w3 * r1[2] */
                "vmlal.s16 q10, d9, d3[0]  \n" /* q10 = w3 * r1[4] */
                "vmlal.s16 q11, d9, d3[2]  \n" /* q11 = w3 * r1[6] */

                "vld1.s32 {d4}, [%[r2]]!   \n" /* load input r2 -> d4*/
                "vmovl.s8   q2,   d4       \n" /* movl d4 -> q2 */

                /* r1 mul w4 */
                "vmlal.s16 q8, d10, d2[1]   \n" /* q8 = w4 * r1[1] */
                "vmlal.s16 q9, d10, d2[3]   \n" /* q9 = w4 * r1[3] */
                "vmlal.s16 q10, d10, d3[1]  \n" /* q10 = w4 * r1[5] */
                "vmlal.s16 q11, d10, d3[3]  \n" /* q11 = w4 * r1[7] */

                "vld1.s32 {d0}, [%[r1]]     \n" /* load r1[8] -> d0 */
                "vmovl.s8   q0,   d0        \n" /* movl d0 -> q0 */

                /* r1 mul w5 */
                "vmlal.s16 q8, d11, d2[2]   \n" /* q8 = w5 * r1[2] */
                "vmlal.s16 q9, d11, d3[0]   \n" /* q9 = w5 * r1[4] */
                "vmlal.s16 q10, d11, d3[2]  \n" /* q10 = w5 * r1[6] */
                "vmlal.s16 q11, d11, d0[0]  \n" /* q11 = w5 * r1[8] */

                /* r2 mul w6 */
                "vmlal.s16 q8, d12, d4[0]   \n" /* q8 = w6 * r2[0] */
                "vmlal.s16 q9, d12, d4[2]   \n" /* q9 = w6 * r2[2] */
                "vmlal.s16 q10, d12, d5[0]  \n" /* q10 = w6 * r2[4] */
                "vmlal.s16 q11, d12, d5[2]  \n" /* q11 = w6 * r2[6] */

                "vld1.s32 {d24-d27}, [%[ptr_out0]] \n" /* load output -> q12,
                                                          q13 */

                /* r2 mul w7 */
                "vmlal.s16 q8, d13, d4[1]   \n" /* q8 = w7 * r2[1] */
                "vmlal.s16 q9, d13, d4[3]   \n" /* q9 = w7 * r2[3] */
                "vmlal.s16 q10, d13, d5[1]  \n" /* q10 = w7 * r2[5] */
                "vmlal.s16 q11, d13, d5[3]  \n" /* q11 = w7 * r2[7] */

                "vld1.s32 {d0}, [%[r2]]     \n" /* load r2[8] -> d0 */
                "vmovl.s8   q0,   d0        \n" /* movl d0 -> q0 */

                /* r2 mul w8 */
                "vmlal.s16 q8, d14, d4[2]   \n" /* q8 = w8 * r2[2] */
                "vmlal.s16 q9, d14, d5[0]   \n" /* q9 = w8 * r2[4] */
                "vmlal.s16 q10, d14, d5[2]  \n" /* q10 = w8 * r2[6] */
                "vmlal.s16 q11, d14, d0[0]  \n" /* q11 = w8 * r2[8] */

                "vadd.s32  q12, q8, q12     \n"         /* out[0] += q8 */
                "vadd.s32  q13, q9, q13     \n"         /* out[1] += q9 */
                "vst1.s32 {d24-d27}, [%[ptr_out0]]! \n" /* store q12, q13 ->
                                                           output[0,1] */

                "vld1.s32  {d0}, [%[r0]]!   \n" /* load next input r0 -> d0*/
                "vmovl.s8   q0,   d0        \n" /* movl d0 -> q0 */

                "vld1.s32 {d28-d31}, [%[ptr_out0]] \n"  /* load output[0,1] ->
                                                           q14, q15 */
                "vadd.s32  q14, q10, q14    \n"         /* out[2] += q10 */
                "vadd.s32  q15, q11, q15    \n"         /* out[3] += q11 */
                "vst1.s32 {d28-d31}, [%[ptr_out0]]! \n" /* store q14, q15 ->
                                                           output[2,3] */

                "bne        1b             \n" /* jump to main loop */

                : [cnt] "+r"(cnt), [r0] "+r"(r0), [r1] "+r"(r1), [r2] "+r"(r2),
                  [ptr_out0] "+r"(ptr_out0), [wc0] "+r"(ptr_wc0)
                :
                : "cc", "memory", "q0", "q1", "q2", "q3", "q4", "q5", "q6",
                  "q7", "q8", "q9", "q10", "q11", "q12", "q13", "q14", "q15");
            wc0 += 9 * hout_c_block;
            inr0 += win_round;
            inr1 += win_round;
            inr2 += win_round;
          }
          block_inr0 = block_inr2;
          block_inr1 = block_inr0 + in_len;
          block_inr2 = block_inr1 + in_len;
        }
        if (out_type == PRECISION(kFloat)) {
          write_to_output_c4_int32_1(
              pre_out, reinterpret_cast<float*>(dout_batch), hout_c_block, 1, c,
              c + hout_c_block, h, h + h_kernel, 0, wout_round, chout, hout,
              wout, flag_relu, reinterpret_cast<float*>(ptr_write), &scale[c],
              out_type);
        } else if (out_type == PRECISION(kInt8)) {
          write_to_output_c4_int32_1(
              pre_out, dout_batch, hout_c_block, 1, c, c + hout_c_block, h,
              h + h_kernel, 0, wout_round, chout, hout, wout, flag_relu,
              reinterpret_cast<signed char*>(ptr_write), &scale[c], out_type);
        } else {
          write_to_output_c4_int32(pre_out, reinterpret_cast<int*>(dout_batch),
                                   hout_c_block, 1, c, c + hout_c_block, h,
                                   h + h_kernel, 0, wout_round, chout, hout,
                                   wout, flag_relu, ptr_write);
        }
      }
    }
  }
}
#endif  // __aarch64__

}  // namespace math
}  // namespace arm
}  // namespace lite
}  // namespace paddle
