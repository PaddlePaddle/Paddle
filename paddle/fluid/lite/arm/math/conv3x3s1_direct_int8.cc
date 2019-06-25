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
void conv_3x3s1_direct_int8(const int8_t* din, int32_t* dout, int num,
                            int chout, int hout, int wout, int chin, int hin,
                            int win, const int8_t* weights, const int32_t* bias,
                            const operators::ConvParam& param,
                            Context<TARGET(kARM)>* ctx, PrecisionType out_type,
                            const float* scale) {
  const int hin_r_block = 4;
  const int hout_c_block = 4;  // 8;
  const int hout_r_block = 2;

  int stride_w = param.strides[1];
  int pad_w = param.paddings[1];
  int pad_h = param.paddings[0];
  bool flag_relu = param.fuse_relu;
  bool flag_bias = (param.bias != nullptr);

  int wout_round = ((wout + 3) / 4) * 4;
  int win_round = wout_round * stride_w + 4;

  int threads = ctx->threads();

  int* tmp_work_space = ctx->workspace_data<int>();
  int* ptr_zero = tmp_work_space;
  memset(ptr_zero, 0, sizeof(int) * win_round);
  int* ptr_write = ptr_zero + win_round;

  int in_len = win_round * chin;
  int pre_in_size = hin_r_block * in_len;
  int pre_out_size = hout_c_block * hout_r_block * wout_round;

  signed char* pre_din = reinterpret_cast<signed char*>(ptr_write + wout_round);

  int size_in_channel = win * hin;
  int size_out_channel = wout * hout;
  int w_stride = chin * 9;

  int ws = -pad_w;
  int we = ws + win_round;
  int w_loop = wout_round / 4;

  int size_out = wout_round * hout_c_block;

  // printf("win_round: %d, wout_round: %d, ws: %d, we: %d\n", win_round,
  // wout_round, ws, we);
  // here
  for (int n = 0; n < num; ++n) {
    const signed char* din_batch =
        static_cast<const signed char*>(din) + n * chin * size_in_channel;
    signed char* dout_batch =
        reinterpret_cast<signed char*>(dout) +
        n * chout * size_out_channel * PrecisionTypeLength(out_type);

    for (int h = 0; h < hout; h += 2) {
      int hs = h - pad_h;
      int he = hs + 4;
      // printf("hs: %d, he: %d, chin: %d, hin: %d, win: %d \n", hs, he, chin,
      // hin, win);
      prepack_input_nxw(din_batch, pre_din, 0, chin, hs, he, ws, we, chin, win,
                        hin, (signed char*)ptr_zero);

#pragma omp parallel for num_threads(threads)
      for (int c = 0; c < chout; c += hout_c_block) {
#ifdef ARM_WITH_OMP
        int* pre_out =
            reinterpret_cast<int*>(pre_din + (pre_in_size + 3) / 4 * 4) +
            omp_get_thread_num() * pre_out_size;
#else
        int* pre_out =
            reinterpret_cast<int*>(pre_din + (pre_in_size + 3) / 4 * 4);
#endif
        // printf("ptr_zero_int: %x, ptr_zero: %x, ptr_write: %x, pre_din: %x,
        // pre_out: %x \n", ptr_zero_int, ptr_zero, ptr_write, pre_din,
        // pre_out);
        const signed char* inr0 = pre_din;
        const signed char* inr1 = inr0 + in_len;
        const signed char* inr2 = inr1 + in_len;
        const signed char* inr3 = inr2 + in_len;

        const signed char* wc0 =
            static_cast<const signed char*>(weights) + c * w_stride;

        const int* bias_ptr = ptr_zero;
        if (flag_bias) {
          bias_ptr = static_cast<const int*>(bias) + c;
        }
        // hout_r_block * wout_round * hout_c_block
        fill_packed_bias_nxmw_int8(bias_ptr, pre_out, hout_c_block,
                                   hout_r_block, wout_round);

        for (int i = 0; i < chin; ++i) {
          const signed char* r0 = inr0;
          const signed char* r1 = inr1;
          const signed char* r2 = inr2;
          const signed char* r3 = inr3;

          int* ptr_out0 = pre_out;
          int* ptr_out1 = pre_out + size_out;

          int cnt = w_loop;
          const signed char* ptr_wc0 = wc0;

          asm volatile(
              "ldp   q4, q5, [%[wc0]] \n"  /* w4 w5 w6 w7 */
              "ldr   q6, [%[wc0], #32] \n" /* w8 */
              "SXTL  v11.8h, v4.8b \n"     /* w to int16 */
              "SXTL2 v12.8h, v4.16b \n"    /* w to int16 */
              "SXTL  v13.8h, v5.8b \n"     /*  to int16 */
              "SXTL2 v14.8h, v5.16b \n"    /* to int16 */
              "SXTL  v15.8h, v6.8b \n"     /* to int16 */
              "1:                     \n"  /* main loop*/
              "ldr  d0, [%[r0]]    \n"     /* load data din0-dinn7*/
              "SXTL  v1.8h,  v0.8b \n"     /* to int16 */

              /*output 1st row*/
              "smull  v16.4s, v11.4h, v1.h[0]   \n" /*  */
              "smull v17.4s, v11.4h, v1.h[1]   \n"  /*  */
              "smull  v18.4s, v11.4h, v1.h[2]   \n" /*  */
              "smull v19.4s, v11.4h, v1.h[3]   \n"  /*  */

              "add   %[r0], %[r0], #4\n"

              /*output 1st row*/
              "smlal2  v16.4s, v11.8h, v1.h[1]   \n" /*  */
              "smlal2 v17.4s, v11.8h, v1.h[2]   \n"  /*  */
              "smlal2  v18.4s, v11.8h, v1.h[3]   \n" /*  */
              "smlal2 v19.4s, v11.8h, v1.h[4]   \n"  /*  */

              "ldr  d0, [%[r1]]    \n" /* load data */

              /*output 1st row*/
              "smlal  v16.4s, v12.4h, v1.h[2]   \n" /*  */
              "smlal v17.4s, v12.4h, v1.h[3]   \n"  /*  */
              "SXTL  v2.8h,  v0.8b \n"              /* to int16 */
              "smlal  v18.4s, v12.4h, v1.h[4]   \n" /*  */
              "smlal v19.4s, v12.4h, v1.h[5]   \n"  /*  */

              "add   %[r1], %[r1], #4  \n"

              /*output 1st row*/
              "smlal2  v16.4s, v12.8h, v2.h[0]   \n" /*  */
              "smlal2 v17.4s, v12.8h, v2.h[1]   \n"  /*  */
              "smlal2  v18.4s, v12.8h, v2.h[2]   \n" /*  */
              "smlal2 v19.4s, v12.8h, v2.h[3]   \n"  /*  */

              /*output 1st row*/
              "smlal  v16.4s, v13.4h, v2.h[1]   \n" /*  */
              "smlal v17.4s, v13.4h, v2.h[2]   \n"  /*  */
              "smlal  v18.4s, v13.4h, v2.h[3]   \n" /*  */
              "smlal v19.4s, v13.4h, v2.h[4]   \n"  /*  */

              /*output 1st row*/
              "smlal2  v16.4s, v13.8h, v2.h[2]   \n" /*  */
              "smlal2 v17.4s, v13.8h, v2.h[3]   \n"  /*  */
              "smlal2  v18.4s, v13.8h, v2.h[4]   \n" /*  */
              "smlal2 v19.4s, v13.8h, v2.h[5]   \n"  /*  */

              /*output 2rd row*/
              "smull  v24.4s, v11.4h, v2.h[0]   \n" /*  */
              "smull v25.4s, v11.4h, v2.h[1]   \n"  /*  */
              "smull  v26.4s, v11.4h, v2.h[2]   \n" /*  */
              "smull v27.4s, v11.4h, v2.h[3]   \n"  /*  */

              /*output 2rd row*/
              "smlal2  v24.4s, v11.8h, v2.h[1]   \n" /*  */
              "smlal2 v25.4s, v11.8h, v2.h[2]   \n"  /*  */
              "smlal2  v26.4s, v11.8h, v2.h[3]   \n" /*  */
              "smlal2 v27.4s, v11.8h, v2.h[4]   \n"  /*  */

              "ldr  d0, [%[r2]]    \n" /* load data */

              /*output 2rd row*/
              "smlal  v24.4s, v12.4h, v2.h[2]   \n" /*  */
              "smlal v25.4s, v12.4h, v2.h[3]   \n"  /*  */
              "SXTL  v1.8h,  v0.8b \n"              /* to int16 */
              "smlal  v26.4s, v12.4h, v2.h[4]   \n" /*  */
              "smlal v27.4s, v12.4h, v2.h[5]   \n"  /*  */

              /*output 1st row*/
              "smlal  v16.4s, v14.4h, v1.h[0]   \n" /*  */
              "smlal v17.4s, v14.4h, v1.h[1]   \n"  /*  */
              "smlal  v18.4s, v14.4h, v1.h[2]   \n" /*  */
              "smlal v19.4s, v14.4h, v1.h[3]   \n"  /*  */

              "add   %[r2], %[r2], #4  \n"

              /*output 1st row*/
              "smlal2  v16.4s, v14.8h, v1.h[1]   \n" /*  */
              "smlal2 v17.4s, v14.8h, v1.h[2]   \n"  /*  */
              "smlal2  v18.4s, v14.8h, v1.h[3]   \n" /*  */
              "smlal2 v19.4s, v14.8h, v1.h[4]   \n"  /*  */

              "ldp    q3, q4, [%[ptr_out0]]             \n"
              "ldp    q5, q6, [%[ptr_out0], #32]             \n"

              /*output 1st row*/
              "smlal  v16.4s, v15.4h, v1.h[2]   \n" /*  */
              "smlal v17.4s, v15.4h, v1.h[3]   \n"  /*  */
              "smlal  v18.4s, v15.4h, v1.h[4]   \n" /*  */
              "smlal v19.4s, v15.4h, v1.h[5]   \n"  /*  */

              "ADD    v3.4s, v16.4s, v3.4s              \n"
              "ADD    v4.4s, v17.4s, v4.4s              \n"
              "ADD    v5.4s, v18.4s, v5.4s              \n"
              "ADD    v6.4s, v19.4s, v6.4s              \n"

              "stp    q3, q4, [%[ptr_out0]], #32          \n"   /* save to
                                                                   output*/
              "stp    q5, q6, [%[ptr_out0]], #32            \n" /* save to
                                                                   output*/

              /*output 2rd row*/
              "smlal2  v24.4s, v12.8h, v1.h[0]   \n" /*  */
              "smlal2 v25.4s, v12.8h, v1.h[1]   \n"  /*  */
              "smlal2  v26.4s, v12.8h, v1.h[2]   \n" /*  */
              "smlal2 v27.4s, v12.8h, v1.h[3]   \n"  /*  */

              /*output 2rd row*/
              "smlal  v24.4s, v13.4h, v1.h[1]   \n" /*  */
              "smlal v25.4s, v13.4h, v1.h[2]   \n"  /*  */
              "smlal  v26.4s, v13.4h, v1.h[3]   \n" /*  */
              "smlal v27.4s, v13.4h, v1.h[4]   \n"  /*  */

              "ldr  d0, [%[r3]]    \n" /* load data */

              /*output 2rd row*/
              "smlal2  v24.4s, v13.8h, v1.h[2]   \n" /*  */
              "smlal2 v25.4s, v13.8h, v1.h[3]   \n"  /*  */
              "SXTL  v2.8h,  v0.8b \n"               /* to int16 */
              "smlal2  v26.4s, v13.8h, v1.h[4]   \n" /*  */
              "smlal2 v27.4s, v13.8h, v1.h[5]   \n"  /*  */

              /*output 2rd row*/
              "smlal  v24.4s, v14.4h, v2.h[0]   \n" /*  */
              "smlal v25.4s, v14.4h, v2.h[1]   \n"  /*  */
              "smlal  v26.4s, v14.4h, v2.h[2]   \n" /*  */
              "smlal v27.4s, v14.4h, v2.h[3]   \n"  /*  */

              "add   %[r3], %[r3], #4  \n"

              /*output 2rd row*/
              "smlal2  v24.4s, v14.8h, v2.h[1]   \n" /*  */
              "smlal2 v25.4s, v14.8h, v2.h[2]   \n"  /*  */
              "smlal2  v26.4s, v14.8h, v2.h[3]   \n" /*  */
              "smlal2 v27.4s, v14.8h, v2.h[4]   \n"  /*  */

              "ldp    q3, q4, [%[ptr_out1]]             \n"
              "ldp    q5, q6, [%[ptr_out1], #32]             \n"

              "subs    %w[cnt], %w[cnt], #1     \n" /* loop count -1 */

              /*output 2rd row*/
              "smlal  v24.4s, v15.4h, v2.h[2]   \n" /*  */
              "smlal v25.4s, v15.4h, v2.h[3]   \n"  /*  */
              "smlal  v26.4s, v15.4h, v2.h[4]   \n" /*  */
              "smlal v27.4s, v15.4h, v2.h[5]   \n"  /*  */

              "ADD    v3.4s, v24.4s, v3.4s              \n"
              "ADD    v4.4s, v25.4s, v4.4s              \n"
              "ADD    v5.4s, v26.4s, v5.4s              \n"
              "ADD    v6.4s, v27.4s, v6.4s              \n"

              "stp    q3, q4, [%[ptr_out1]], #32        \n" /* save to output*/
              "stp    q5, q6, [%[ptr_out1]], #32        \n" /* save to output*/

              "bne    1b                          \n" /* jump to main loop*/

              : [cnt] "+r"(cnt), [wc0] "+r"(ptr_wc0), [r0] "+r"(r0),
                [r1] "+r"(r1), [r2] "+r"(r2), [r3] "+r"(r3),
                [ptr_out0] "+r"(ptr_out0), [ptr_out1] "+r"(ptr_out1)
              :
              : "cc", "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v11",
                "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19", "v24",
                "v25", "v26", "v27"

              );

          wc0 += 9 * hout_c_block;
          inr0 += win_round;
          inr1 += win_round;
          inr2 += win_round;
          inr3 += win_round;
        }
        if (out_type == PRECISION(kFloat)) {
          write_to_output_c4_int32_1(
              pre_out, reinterpret_cast<float*>(dout_batch), hout_c_block,
              hout_r_block, c, c + 4, h, h + 2, 0, wout_round, chout, hout,
              wout, flag_relu, reinterpret_cast<float*>(ptr_write), &scale[c],
              out_type);
        } else if (out_type == PRECISION(kInt8)) {
          write_to_output_c4_int32_1(
              pre_out, dout_batch, hout_c_block, hout_r_block, c, c + 4, h,
              h + 2, 0, wout_round, chout, hout, wout, flag_relu,
              reinterpret_cast<signed char*>(ptr_write), &scale[c], out_type);
        } else {  // int32
          write_to_output_c4_int32(pre_out, reinterpret_cast<int*>(dout_batch),
                                   hout_c_block, hout_r_block, c, c + 4, h,
                                   h + 2, 0, wout_round, chout, hout, wout,
                                   flag_relu, ptr_write);
        }
      }
    }
  }
}

#else

void conv_3x3s1_direct_int8(const int8_t* din, int32_t* dout, int num,
                            int chout, int hout, int wout, int chin, int hin,
                            int win, const int8_t* weights, const int32_t* bias,
                            const operators::ConvParam& param,
                            Context<TARGET(kARM)>* ctx, PrecisionType out_type,
                            const float* scale) {
  // printf("conv2_3x3s1_direct_int8 \n");

  const int hin_r_block = 4;
  const int hout_c_block = 4;  // 8
  const int hout_r_block = 2;

  int stride_w = param.strides[1];
  int pad_w = param.paddings[1];
  int pad_h = param.paddings[0];
  bool flag_relu = param.fuse_relu;
  bool flag_bias = (param.bias != nullptr);

  int wout_round = ((wout + 3) / 4) * 4;
  int win_round = wout_round * stride_w + 4;

  int threads = ctx->threads();

  int* tmp_work_space = ctx->workspace_data<int>();
  int* ptr_zero = tmp_work_space;
  memset(ptr_zero, 0, sizeof(int) * win_round);
  int* ptr_write = ptr_zero + win_round;

  int in_len = win_round * chin;
  int pre_in_size = hin_r_block * in_len;
  int pre_out_size = hout_c_block * hout_r_block * wout_round;

  signed char* pre_din = reinterpret_cast<signed char*>(ptr_write + wout_round);

  int size_in_channel = win * hin;
  int size_out_channel = wout * hout;
  int w_stride = chin * 9;

  int ws = -pad_w;
  int we = ws + win_round;
  int w_loop = wout_round / 4;

  int size_out = wout_round * hout_c_block;

  // printf("win_round: %d, wout_round: %d, ws: %d, we: %d\n", win_round,
  // wout_round, ws, we);

  for (int n = 0; n < num; ++n) {
    const signed char* din_batch =
        static_cast<const signed char*>(din) + n * chin * size_in_channel;
    signed char* dout_batch =
        reinterpret_cast<signed char*>(dout) +
        n * chout * size_out_channel * PrecisionTypeLength(out_type);

    for (int h = 0; h < hout; h += 2) {
      int hs = h - pad_h;
      int he = hs + 4;
      // printf("hs: %d, he: %d, chin: %d, hin: %d, win: %d \n", hs, he, chin,
      // hin, win);
      prepack_input_nxw(din_batch, pre_din, 0, chin, hs, he, ws, we, chin, win,
                        hin, (signed char*)ptr_zero);

#pragma omp parallel for num_threads(threads)
      for (int c = 0; c < chout; c += hout_c_block) {  // 4
#ifdef ARM_WITH_OMP
        int* pre_out =
            reinterpret_cast<int*>(pre_din + (pre_in_size + 3) / 4 * 4) +
            omp_get_thread_num() * pre_out_size;
#else
        int* pre_out =
            reinterpret_cast<int*>(pre_din + (pre_in_size + 3) / 4 * 4);
#endif
        // printf("ptr_zero_int: %x, ptr_zero: %x, ptr_write: %x, pre_din: %x,
        // pre_out: %x \n", ptr_zero_int, ptr_zero, ptr_write, pre_din,
        // pre_out);
        const signed char* inr0 = pre_din;
        const signed char* inr1 = inr0 + in_len;
        const signed char* inr2 = inr1 + in_len;
        const signed char* inr3 = inr2 + in_len;

        const signed char* wc0 =
            static_cast<const signed char*>(weights) + c * w_stride;

        const int* bias_ptr = ptr_zero;
        if (flag_bias) {
          bias_ptr = static_cast<const int*>(bias) + c;
        }
        // hout_r_block * wout_round * hout_c_block
        fill_packed_bias_nxmw_int8(bias_ptr, pre_out, hout_c_block,
                                   hout_r_block, wout_round);

        for (int i = 0; i < chin; ++i) {
          const signed char* r0 = inr0;
          const signed char* r1 = inr1;
          const signed char* r2 = inr2;
          const signed char* r3 = inr3;

          int* ptr_out0 = pre_out;
          int* ptr_out1 = pre_out + size_out;

          int cnt = w_loop;
          const signed char* ptr_wc = wc0;

          asm volatile(
              "vld1.s8 {d0-d3}, [%[wc0]]!     \n" /* wc0, wc1, wc2, wc3, wc4,
                                                     wc5, wc6, wc7*/
              "vld1.s8 {d4},    [%[wc0]]!     \n" /*  wc8 */
              "vmovl.s8   q3,   d0            \n" /* q3 = w0, w1 */
              "vmovl.s8   q4,   d1            \n" /* q4 = w2 ,w3 */
              "vmovl.s8   q5,   d2            \n" /* q5 = w4, w5 */
              "vmovl.s8   q6,   d3            \n" /* q6 = w6, w7 */
              "vmovl.s8   q7,   d4            \n" /* q7 = w8 */

              "1:                           \n" /* main loop*/
              "vld1.s32  {d0}, [%[r0]]    \n"   /* load data din0-dinn7*/
              "vmovl.s8   q0,   d0        \n"   /* movl d0 -> q0 */
              /*output 1st row*/
              "vmull.s16 q8, d6, d0[0]   \n" /* q8 = w0 * r0[0] */
              "vmull.s16 q9, d6, d0[1]   \n" /* q9 = w0 * r0[2] */
              "vmull.s16 q10, d6, d0[2]  \n" /* q10 = w0 * r0[4] */
              "vmull.s16 q11, d6, d0[3]  \n" /* q11 = w0 * r0[6] */

              "add   %[r0], #4           \n"

              /*output 1st row*/
              "vmlal.s16 q8, d7, d0[1]   \n" /* q8 = w1 * r0[1] */
              "vmlal.s16 q9, d7, d0[2]   \n" /* q9 = w1 * r0[2] */
              "vmlal.s16 q10, d7, d0[3]  \n" /* q10 = w1 * r0[3] */
              "vmlal.s16 q11, d7, d1[0]  \n" /* q11 = w1 * r0[4] */

              "vld1.s32 {d2}, [%[r1]]    \n" /* load input r1 -> d2 */
              "vmovl.s8  q1,   d2        \n" /* movl d2 -> q1 */

              /*output 1st row*/
              "vmlal.s16 q8, d8, d0[2]   \n" /* q8 = w2 * r0[2] */
              "vmlal.s16 q9, d8, d0[3]   \n" /* q9 = w2 * r0[3] */
              "vmlal.s16 q10, d8, d1[0]  \n" /* q10 = w2 * r0[4] */
              "vmlal.s16 q11, d8, d1[1]  \n" /* q11 = w2 * r0[5] */

              /*output 1st row*/
              "vmlal.s16 q8, d9, d2[0]   \n" /*  */
              "vmlal.s16 q9, d9, d2[1]   \n" /*  */
              "vmlal.s16 q10, d9, d2[2]  \n" /*  */
              "vmlal.s16 q11, d9, d2[3]  \n" /*  */

              "add   %[r1],   #4         \n"

              /*output 1st row*/
              "vmlal.s16 q8, d10, d2[1]  \n" /*  */
              "vmlal.s16 q9, d10, d2[2]  \n" /*  */
              "vmlal.s16 q10, d10, d2[3] \n" /*  */
              "vmlal.s16 q11, d10, d3[0] \n" /*  */

              /*output 1st row*/
              "vmlal.s16 q8, d11, d2[2]  \n" /*  */
              "vmlal.s16 q9, d11, d2[3]  \n" /*  */
              "vmlal.s16 q10, d11, d3[0] \n" /*  */
              "vmlal.s16 q11, d11, d3[1] \n" /*  */

              /*output 2rd row*/
              "vmull.s16 q12, d6, d2[0]  \n" /*  */
              "vmull.s16 q13, d6, d2[1]  \n" /*  */
              "vmull.s16 q14, d6, d2[2]  \n" /*  */
              "vmull.s16 q15, d6, d2[3]  \n" /*  */

              "vld1.s32 {d0}, [%[r2]]    \n" /* load input r2 -> d2 */
              "vmovl.s8 q0,   d0         \n" /* movl d2 -> q1 */

              /*output 2rd row*/
              "vmlal.s16 q12, d7, d2[1]  \n" /*  */
              "vmlal.s16 q13, d7, d2[2]  \n" /*  */
              "vmlal.s16 q14, d7, d2[3]  \n" /*  */
              "vmlal.s16 q15, d7, d3[0]  \n" /*  */

              /*output 2rd row*/
              "vmlal.s16 q12, d8, d2[2]  \n" /*  */
              "vmlal.s16 q13, d8, d2[3]  \n" /*  */
              "vmlal.s16 q14, d8, d3[0]  \n" /*  */
              "vmlal.s16 q15, d8, d3[1]  \n" /*  */

              "add   %[r2], #4           \n"

              /*output 1st row*/
              "vmlal.s16 q8, d12, d0[0]   \n" /*  */
              "vmlal.s16 q9, d12, d0[1]   \n" /*  */
              "vmlal.s16 q10, d12, d0[2]  \n" /*  */
              "vmlal.s16 q11, d12, d0[3]  \n" /*  */

              /*output 1st row*/
              "vmlal.s16 q8, d13, d0[1]   \n" /*  */
              "vmlal.s16 q9, d13, d0[2]   \n" /*  */
              "vmlal.s16 q10, d13, d0[3]  \n" /*  */
              "vmlal.s16 q11, d13, d1[0]  \n" /*  */

              "vld1.32    {d2-d5}, [%[ptr_out0]]   \n" /* load ptr_out -> q, q
                                                          */

              /*output 1st row*/
              "vmlal.s16 q8, d14, d0[2]   \n" /*  */
              "vmlal.s16 q9, d14, d0[3]   \n" /*  */
              "vmlal.s16 q10, d14, d1[0]  \n" /*  */
              "vmlal.s16 q11, d14, d1[1]  \n" /*  */

              /*load & store output 1st row*/
              "vadd.s32  q1, q8, q1                \n" /* out[0] += q8 */
              "vadd.s32  q2, q9, q2                \n" /* out[0] += q8 */
              "vst1.s32   {d2-d5}, [%[ptr_out0]]!  \n"

              /*output 2rd row*/
              "vmlal.s16 q12, d9, d0[0]   \n" /*  */
              "vmlal.s16 q13, d9, d0[1]   \n" /*  */
              "vmlal.s16 q14, d9, d0[2]   \n" /*  */
              "vmlal.s16 q15, d9, d0[3]   \n" /*  */

              "vld1.32    {d2-d5}, [%[ptr_out0]]   \n" /* load ptr_out -> q2, q3
                                                          */

              /*output 2rd row */
              "vmlal.s16 q12, d10, d0[1]   \n" /*  */
              "vmlal.s16 q13, d10, d0[2]   \n" /*  */
              "vadd.s32  q1, q10, q1       \n" /* out[0] += q */
              "vadd.s32  q2, q11, q2       \n" /* out[1] += q */

              "vmlal.s16 q14, d10, d0[3]            \n" /*  */
              "vst1.s32   {d2-d5}, [%[ptr_out0]]!   \n"
              "vmlal.s16 q15, d10, d1[0]            \n" /*  */

              /*output 2rd row */
              "vmlal.s16 q12, d11, d0[2]            \n" /*  */
              "vmlal.s16 q13, d11, d0[3]            \n" /*  */

              "vld1.s32 {d4}, [%[r3]]               \n" /* load input r2 -> d2
                                                           */
              "vmovl.s8 q2,   d4                    \n" /* movl d2 -> q2 */

              "vmlal.s16 q14, d11, d1[0]            \n" /*  */
              "vmlal.s16 q15, d11, d1[1]            \n" /*  */

              "add   %[r3], #4                      \n"

              /*output 2rd row */
              "vmlal.s16 q12, d12, d4[0]             \n" /*  */
              "vmlal.s16 q13, d12, d4[1]             \n" /*  */
              "vmlal.s16 q14, d12, d4[2]             \n" /*  */
              "vmlal.s16 q15, d12, d4[3]             \n" /*  */

              "vld1.32    {d0-d3}, [%[ptr_out1]]     \n" /*  */

              /*output 2rd row */
              "vmlal.s16 q12, d13, d4[1]             \n" /*  */
              "vmlal.s16 q13, d13, d4[2]             \n" /*  */
              "vmlal.s16 q14, d13, d4[3]             \n" /*  */
              "vmlal.s16 q15, d13, d5[0]             \n" /*  */

              "subs  %[cnt], #1                      \n"

              /*output 2rd row */
              "vmlal.s16 q12, d14, d4[2]             \n" /*  */
              "vmlal.s16 q13, d14, d4[3]             \n" /*  */
              "vmlal.s16 q14, d14, d5[0]             \n" /*  */
              "vmlal.s16 q15, d14, d5[1]             \n" /*  */

              /*output 2rd row*/
              "vadd.s32  q0, q12, q0                 \n" /*  */
              "vadd.s32  q1, q13, q1                 \n" /*  */
              "vst1.s32   {d0-d3}, [%[ptr_out1]]!    \n"

              "vld1.32    {d0-d3}, [%[ptr_out1]]     \n" /*  */
              "vadd.s32  q0, q14, q0                 \n" /*  */
              "vadd.s32  q1, q15, q1                 \n" /*  */
              "vst1.s32   {d0-d3}, [%[ptr_out1]]!    \n"

              "bne    1b                             \n" /* jump to main loop*/

              : [cnt] "+r"(cnt), [r0] "+r"(r0), [r1] "+r"(r1), [r2] "+r"(r2),
                [r3] "+r"(r3), [ptr_out0] "+r"(ptr_out0),
                [ptr_out1] "+r"(ptr_out1), [wc0] "+r"(ptr_wc)
              :
              : "cc", "memory", "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7",
                "q8", "q9", "q10", "q11", "q12", "q13", "q14", "q15");

          wc0 += 9 * hout_c_block;
          inr0 += win_round;
          inr1 += win_round;
          inr2 += win_round;
          inr3 += win_round;
        }

        if (out_type == PRECISION(kFloat)) {
          write_to_output_c4_int32_1(
              pre_out, reinterpret_cast<float*>(dout_batch), hout_c_block,
              hout_r_block, c, c + 4, h, h + 2, 0, wout_round, chout, hout,
              wout, flag_relu, reinterpret_cast<float*>(ptr_write), &scale[c],
              out_type);
        } else if (out_type == PRECISION(kInt8)) {
          write_to_output_c4_int32_1(
              pre_out, dout_batch, hout_c_block, hout_r_block, c, c + 4, h,
              h + 2, 0, wout_round, chout, hout, wout, flag_relu,
              reinterpret_cast<signed char*>(ptr_write), &scale[c], out_type);
        } else {  // int32
          write_to_output_c4_int32(pre_out, reinterpret_cast<int*>(dout_batch),
                                   hout_c_block, hout_r_block, c, c + 4, h,
                                   h + 2, 0, wout_round, chout, hout, wout,
                                   flag_relu, ptr_write);
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
