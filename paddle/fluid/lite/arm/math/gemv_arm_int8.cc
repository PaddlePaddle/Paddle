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

#include "paddle/fluid/lite/arm/math/gemv_arm_int8.h"
#include <arm_neon.h>
#include "paddle/fluid/lite/arm/math/saturate.h"

namespace paddle {
namespace lite {
namespace arm {
namespace math {

template <typename dtype>
inline void write_gemv_out(const int* in, dtype* out, const float* scale);

template <>
inline void write_gemv_out(const int* in, int* out, const float* scale) {
  out[0] = in[0];
}

template <>
inline void write_gemv_out(const int* in, float* out, const float* scale) {
  out[0] = in[0] * scale[0];
}

template <>
inline void write_gemv_out(const int* in, signed char* out,
                           const float* scale) {
  out[0] = saturate_cast<signed char>(roundf(in[0] * scale[0]));
}

template <typename dtype>
bool gemv_int8(const int8_t* A, const int8_t* x, dtype* y, bool transA, int M,
               int N, const float* scale, bool is_bias, const int* bias,
               bool is_relu) {
  if (transA) {
    LOG(ERROR) << "ERROR: sgemv, transA is not supported now";
    return false;
  }
  dtype* data_out = y;
  const int8_t* data_in = x;
  const int8_t* weights_ptr = A;
  int cnt = N >> 4;
  int tail = N & 15;
  int flag_bias = is_bias ? 1 : 0;

#ifdef __aarch64__
  int out_cnt = M >> 3;
#pragma omp parallel for
  for (int j = 0; j < out_cnt; j++) {
    int out_idx = j * 8;
    dtype* out_ptr = data_out + out_idx;
    const float* scale_ptr = scale + out_idx;
    int ptr_out[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    const int8_t* ptr_in = data_in;
    const int8_t* ptr_w0 = weights_ptr + (N * out_idx);
    const int8_t* ptr_w1 = ptr_w0 + N;
    const int8_t* ptr_w2 = ptr_w1 + N;
    const int8_t* ptr_w3 = ptr_w2 + N;
    const int8_t* ptr_w4 = ptr_w3 + N;
    const int8_t* ptr_w5 = ptr_w4 + N;
    const int8_t* ptr_w6 = ptr_w5 + N;
    const int8_t* ptr_w7 = ptr_w6 + N;
    const int* bias_ptr = is_bias ? (bias + out_idx) : nullptr;
    int cnt_loop = cnt;
    asm volatile(
        "prfm  pldl1keep, [%[in]]           \n" /* preload din */
        "prfm  pldl1keep, [%[w0]]   \n"         /* preload w0 */
        "prfm  pldl1keep, [%[w1]]   \n"         /* preload w1 */
        "prfm  pldl1keep, [%[w2]]   \n"         /* preload w2 */
        "prfm  pldl1keep, [%[w3]]   \n"         /* preload w3 */
        "prfm  pldl1keep, [%[w4]]   \n"         /* preload w4 */
        "prfm  pldl1keep, [%[w5]]   \n"         /* preload w5 */
        "prfm  pldl1keep, [%[w6]]   \n"         /* preload w6 */
        "prfm  pldl1keep, [%[w7]]   \n"         /* preload w7 */
        "movi   v0.4s,  #0          \n"         /* set out0 to 0 */
        "movi   v1.4s,  #0          \n"         /* set out1 to 0 */
        "movi   v2.4s,  #0          \n"         /* set out2 to 0 */
        "movi   v3.4s,  #0          \n"         /* set out3 to 0 */
        "movi   v4.4s,  #0          \n"         /* set out4 to 0 */
        "movi   v5.4s,  #0          \n"         /* set out5 to 0 */
        "movi   v6.4s,  #0          \n"         /* set out6 to 0 */
        "movi   v7.4s,  #0          \n"         /* set out7 to 0 */
        /* check main loop */
        "cmp %w[cnt], #1            \n" /* check whether has main loop */
        "blt  2f                    \n" /* jump to tail */
        /* main loop */
        "1:                         \n"  /* main loop */
        "ldr    q8,     [%[in]], #16 \n" /* load input, 16 int8 */
        "ldr    q9,     [%[w0]], #16 \n" /* load w0, 16 int8 */
        "ldr    q10,    [%[w1]], #16 \n" /* load w1, 16 int8 */
        "ldr    q11,    [%[w2]], #16 \n" /* load w2, 16 int8 */
        "ldr    q12,    [%[w3]], #16 \n" /* load w3, 16 int8 */
        "ldr    q13,    [%[w4]], #16 \n" /* load w4, 16 int8 */
        "ldr    q14,    [%[w5]], #16 \n" /* load w5, 16 int8 */
        "ldr    q15,    [%[w6]], #16 \n" /* load w6, 16 int8 */
        "ldr    q16,    [%[w7]], #16 \n" /* load w7, 16 int8 */
        /* mul, lower 8 int8 * int8 = int16 */
        "smull  v18.8h, v8.8b, v9.8b \n" /* mul in * w0, low, 8 int8 */
        "smull  v19.8h, v8.8b, v10.8b\n" /* mul in * w1, low, 8 int8 */
        "smull  v20.8h, v8.8b, v11.8b\n" /* mul in * w2, low, 8 int8 */
        "smull  v21.8h, v8.8b, v12.8b\n" /* mul in * w3, low, 8 int8 */
        "smull  v22.8h, v8.8b, v13.8b\n" /* mul in * w4, low, 8 int8 */
        "smull  v23.8h, v8.8b, v14.8b\n" /* mul in * w5, low, 8 int8 */
        "smull  v24.8h, v8.8b, v15.8b\n" /* mul in * w6, low, 8 int8 */
        "smull  v25.8h, v8.8b, v16.8b\n" /* mul in * w7, low, 8 int8 */
        /* mul, higher 8 int8 * int8 + int16 = int16 */
        "smlal2 v18.8h,v8.16b,v9.16b \n" /* mul in * w0, high, 8 int8 */
        "smlal2 v19.8h,v8.16b,v10.16b\n" /* mul in * w1, high, 8 int8 */
        "smlal2 v20.8h,v8.16b,v11.16b\n" /* mul in * w2, high, 8 int8 */
        "smlal2 v21.8h,v8.16b,v12.16b\n" /* mul in * w2, high, 8 int8 */
        "smlal2 v22.8h,v8.16b,v13.16b\n" /* mul in * w2, high, 8 int8 */
        "smlal2 v23.8h,v8.16b,v14.16b\n" /* mul in * w2, high, 8 int8 */
        "smlal2 v24.8h,v8.16b,v15.16b\n" /* mul in * w2, high, 8 int8 */
        "smlal2 v25.8h,v8.16b,v16.16b\n" /* mul in * w2, high, 8 int8 */
        "subs %w[cnt], %w[cnt], #1   \n" /* sub main loop count */
        /* add int16 to int32 */
        "sadalp v0.4s, v18.8h \n"        /* pair acc, 8 int16 -> 4 int32 */
        "sadalp v1.4s, v19.8h \n"        /* pair acc, 8 int16 -> 4 int32 */
        "sadalp v2.4s, v20.8h \n"        /* pair acc, 8 int16 -> 4 int32 */
        "sadalp v3.4s, v21.8h \n"        /* pair acc, 8 int16 -> 4 int32 */
        "sadalp v4.4s, v22.8h \n"        /* pair acc, 8 int16 -> 4 int32 */
        "sadalp v5.4s, v23.8h \n"        /* pair acc, 8 int16 -> 4 int32 */
        "sadalp v6.4s, v24.8h \n"        /* pair acc, 8 int16 -> 4 int32 */
        "sadalp v7.4s, v25.8h \n"        /* pair acc, 8 int16 -> 4 int32 */
        "bne 1b                      \n" /* jump to main loop */
        /* pair add to final result */
        "2:                          \n" /* reduce to scale */
        "addp v8.4s , v0.4s , v1.4s  \n" /* pair add to 4 int32*/
        "addp v9.4s , v2.4s , v3.4s  \n" /* pair add to 4 int32*/
        "addp v10.4s, v4.4s , v5.4s  \n" /* pair add to 4 int32*/
        "addp v11.4s, v6.4s , v7.4s  \n" /* pair add to 4 int32*/

        "addp v12.4s, v8.4s , v9.4s  \n" /* pair add to 4 int32*/
        "addp v13.4s, v10.4s, v11.4s \n" /* pair add to 4 int32*/

        "cmp %w[bias], #1           \n" /* check whether has bias */
        "blt  0f                    \n" /* jump to tail */
        "ldp   q8, q9, [%[bias_ptr]]\n" /* load bias to q8, q9*/
        "add v12.4s, v12.4s, v8.4s  \n" /* add bias */
        "add v13.4s, v13.4s, v9.4s  \n" /* add bias */
        "0:                         \n" /* end of add bias */

        /* write to output */
        "stp q12, q13, [%[out]]     \n" /* save result */
        : [in] "+r"(ptr_in), [w0] "+r"(ptr_w0), [w1] "+r"(ptr_w1),
          [w2] "+r"(ptr_w2), [w3] "+r"(ptr_w3), [w4] "+r"(ptr_w4),
          [w5] "+r"(ptr_w5), [w6] "+r"(ptr_w6), [w7] "+r"(ptr_w7),
          [cnt] "+r"(cnt_loop)
        : [out] "r"(ptr_out), [bias_ptr] "r"(bias_ptr), [bias] "r"(flag_bias)
        : "cc", "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8",
          "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18",
          "v19", "v20", "v21", "v22", "v23", "v24", "v25");
    for (int i = 0; i < tail; ++i) {
      ptr_out[0] += ptr_in[i] * ptr_w0[i];
      ptr_out[1] += ptr_in[i] * ptr_w1[i];
      ptr_out[2] += ptr_in[i] * ptr_w2[i];
      ptr_out[3] += ptr_in[i] * ptr_w3[i];
      ptr_out[4] += ptr_in[i] * ptr_w4[i];
      ptr_out[5] += ptr_in[i] * ptr_w5[i];
      ptr_out[6] += ptr_in[i] * ptr_w6[i];
      ptr_out[7] += ptr_in[i] * ptr_w7[i];
    }
    if (is_relu) {
      ptr_out[0] = ptr_out[0] > 0 ? ptr_out[0] : 0;
      ptr_out[1] = ptr_out[1] > 0 ? ptr_out[1] : 0;
      ptr_out[2] = ptr_out[2] > 0 ? ptr_out[2] : 0;
      ptr_out[3] = ptr_out[3] > 0 ? ptr_out[3] : 0;
      ptr_out[4] = ptr_out[4] > 0 ? ptr_out[4] : 0;
      ptr_out[5] = ptr_out[5] > 0 ? ptr_out[5] : 0;
      ptr_out[6] = ptr_out[6] > 0 ? ptr_out[6] : 0;
      ptr_out[7] = ptr_out[7] > 0 ? ptr_out[7] : 0;
    }

    write_gemv_out(ptr_out, out_ptr, scale_ptr);
    write_gemv_out(ptr_out + 1, out_ptr + 1, scale_ptr + 1);
    write_gemv_out(ptr_out + 2, out_ptr + 2, scale_ptr + 2);
    write_gemv_out(ptr_out + 3, out_ptr + 3, scale_ptr + 3);
    write_gemv_out(ptr_out + 4, out_ptr + 4, scale_ptr + 4);
    write_gemv_out(ptr_out + 5, out_ptr + 5, scale_ptr + 5);
    write_gemv_out(ptr_out + 6, out_ptr + 6, scale_ptr + 6);
    write_gemv_out(ptr_out + 7, out_ptr + 7, scale_ptr + 7);
  }

//! deal with remains
#pragma omp parallel for
  for (int j = out_cnt * 8; j < M; j++) {
    // int *ptr_out = data_out + j;
    dtype* out_ptr = data_out + j;
    const float* scale_ptr = scale + j;
    int ptr_out[1] = {0};
    const int8_t* ptr_in = data_in;
    const int8_t* ptr_w0 = weights_ptr + (N * j);
    int cnt_loop = cnt;
    int bias0 = is_bias ? bias[j] : 0;
    asm volatile(
        "prfm  pldl1keep, [%[in]]               \n" /* preload din */
        "prfm  pldl1keep, [%[w0]]       \n"         /* preload w0 */
        "movi   v0.4s,  #0              \n"         /* set out0 to 0 */
        "fmov   s0, %w[bias0]           \n"         /* set bias */
        /* check main loop */
        "cmp %w[cnt], #1                \n" /* check whether has main loop */
        "blt  2f                        \n" /* jump to tail */
        /* main loop */
        "1:                             \n" /* main loop */
        "ldr    q8,     [%[in]], #16    \n" /* load input, 16 int8 */
        "ldr    q9,     [%[w0]], #16    \n" /* load w0, 16 int8 */
        /* mul, lower 8 int8 * int8 = int16 */
        "smull  v18.8h, v8.8b, v9.8b    \n" /* mul in * w0, low, 8 int8 */
        "subs %w[cnt], %w[cnt], #1      \n" /* sub main loop count */
        /* mul, higher 8 int8 * int8 + int16 = int16 */
        "smlal2 v18.8h,v8.16b,v9.16b    \n" /* mul in * w0, high, 8 int8 */
        /* add int16 to int32 */
        "sadalp v0.4s, v18.8h           \n" /* pair acc, 8 int16 -> 4 int32 */
        "bne 1b                         \n" /* jump to main loop */
        /* pair add to final result */
        "2:                             \n" /* reduce to scale */
        "addv   s8, v0.4s               \n" /* reduction to out0 */
        /* write to output */
        "str s8, [%[out]]               \n" /* save result */
        : [in] "+r"(ptr_in), [w0] "+r"(ptr_w0), [cnt] "+r"(cnt_loop)
        : [out] "r"(ptr_out), [bias0] "r"(bias0)
        : "cc", "memory", "v0", "v8", "v9", "v18");
    for (int i = 0; i < tail; ++i) {
      ptr_out[0] += ptr_in[i] * ptr_w0[i];
    }
    if (is_relu) {
      ptr_out[0] = ptr_out[0] > 0 ? ptr_out[0] : 0;
    }
    write_gemv_out(ptr_out, out_ptr, scale_ptr);
  }
#else  //__aarch64__ // NOLINT
  int out_cnt = M >> 2;
#pragma omp parallel for
  for (int j = 0; j < out_cnt; j++) {
    int out_idx = j * 4;
    dtype* out_ptr = data_out + out_idx;
    const float* scale_ptr = scale + out_idx;
    int ptr_out[4] = {0, 0, 0, 0};
    const int8_t* ptr_in = data_in;
    const int8_t* ptr_w0 = weights_ptr + (N * out_idx);
    const int8_t* ptr_w1 = ptr_w0 + N;
    const int8_t* ptr_w2 = ptr_w1 + N;
    const int8_t* ptr_w3 = ptr_w2 + N;
    int cnt_loop = cnt;
    int bias0 = is_bias ? bias[out_idx] : 0;
    int bias1 = is_bias ? bias[out_idx + 1] : 0;
    int bias2 = is_bias ? bias[out_idx + 2] : 0;
    int bias3 = is_bias ? bias[out_idx + 3] : 0;
    asm volatile(
        "pld [%[in]]                    @ preload cache line, input\n"
        "pld [%[w0]]                    @ preload cache line, weights r0\n"
        "pld [%[w1]]                    @ preload cache line, weights r1\n"
        "pld [%[w2]]                    @ preload cache line, weights r2\n"
        "pld [%[w3]]                    @ preload cache line, weights r3\n"
        "vmov.u32 q0, #0                @ set q0 to 0\n"
        "vmov.u32 q1, #0                @ set q1 to 0\n"
        "vmov.u32 q2, #0                @ set q2 to 0\n"
        "vmov.u32 q3, #0                @ set q3 to 0\n"
        "vmov s0, %[bias0]              @ set q0 to bias0\n"
        "vmov s4, %[bias1]              @ set q1 to bias1\n"
        "vmov s8, %[bias2]              @ set q2 to bias2\n"
        "vmov s12,%[bias3]              @ set q3 to bias3\n"
        // "vld1.32 {d20-d21}, %[bias]     @ load bias data"
        "cmp %[cnt], #1                 @ check whether has main loop\n"
        "blt  2f                        @ jump to pair add\n"
        /* main loop */
        "1:                             @ main loop\n"
        "vld1.8 {d8-d9}, [%[in]]!       @ load input, q4\n"
        "vld1.8 {d12-d13}, [%[w0]]!     @ load weights r0, q6\n"
        "vld1.8 {d14-d15}, [%[w1]]!     @ load weights r1, q7\n"
        "vld1.8 {d16-d17}, [%[w2]]!     @ load weights r2, q8\n"
        "vld1.8 {d18-d19}, [%[w3]]!     @ load weights r3, q9\n"
        /* mul, int8 * int8 = int16 */
        "vmull.s8 q12, d8, d12          @ mul add\n"
        "vmull.s8 q13, d8, d14          @ mul add\n"
        "vmull.s8 q14, d8, d16          @ mul add\n"
        "vmull.s8 q15, d8, d18          @ mul add\n"
        /* mla, int8 * int8 + int16 = int16 */
        "vmlal.s8 q12, d9, d13          @ mul add\n"
        "vmlal.s8 q13, d9, d15          @ mul add\n"
        "vmlal.s8 q14, d9, d17          @ mul add\n"
        "vmlal.s8 q15, d9, d19          @ mul add\n"
        /* pacc, int16 + int32 = int32 */
        "vpadal.s16 q0, q12             @ pair acc\n"
        "vpadal.s16 q1, q13             @ pair acc\n"
        "vpadal.s16 q2, q14             @ pair acc\n"
        "vpadal.s16 q3, q15             @ pair acc\n"
        "subs %[cnt], #1                @ sub loop count \n"
        /* check loop end */
        "bne 1b                         @ jump to main loop\n"
        /* pair add to final result */
        "2:                             @ pair add \n"
        "vpadd.s32 d8, d0, d1           @ pair add, first step\n"
        "vpadd.s32 d9, d2, d3           @ pair add, first step\n"
        "vpadd.s32 d10, d4, d5          @ pair add, first step\n"
        "vpadd.s32 d11, d6, d7          @ pair add, first step\n"
        "vpadd.s32 d0, d8, d9           @ pair add, second step\n"
        "vpadd.s32 d1, d10, d11         @ pair add, second step\n"
        /* write output */
        "vst1.32 {d0-d1}, [%[out]]      @ save result\n"
        : [in] "+r"(ptr_in), [w0] "+r"(ptr_w0), [w1] "+r"(ptr_w1),
          [w2] "+r"(ptr_w2), [w3] "+r"(ptr_w3), [cnt] "+r"(cnt_loop)
        : [bias0] "r"(bias0), [bias1] "r"(bias1), [bias2] "r"(bias2),
          [bias3] "r"(bias3), [out] "r"(ptr_out)
        : "cc", "memory", "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7", "q8",
          "q9", "q12", "q13", "q14", "q15");
    for (int i = 0; i < tail; ++i) {
      ptr_out[0] += ptr_in[i] * ptr_w0[i];
      ptr_out[1] += ptr_in[i] * ptr_w1[i];
      ptr_out[2] += ptr_in[i] * ptr_w2[i];
      ptr_out[3] += ptr_in[i] * ptr_w3[i];
    }
    if (is_relu) {
      ptr_out[0] = ptr_out[0] > 0 ? ptr_out[0] : 0;
      ptr_out[1] = ptr_out[1] > 0 ? ptr_out[1] : 0;
      ptr_out[2] = ptr_out[2] > 0 ? ptr_out[2] : 0;
      ptr_out[3] = ptr_out[3] > 0 ? ptr_out[3] : 0;
    }
    write_gemv_out(ptr_out, out_ptr, scale_ptr);
    write_gemv_out(ptr_out + 1, out_ptr + 1, scale_ptr + 1);
    write_gemv_out(ptr_out + 2, out_ptr + 2, scale_ptr + 2);
    write_gemv_out(ptr_out + 3, out_ptr + 3, scale_ptr + 3);
  }
//! deal with remains
#pragma omp parallel for
  for (int j = out_cnt * 4; j < M; j++) {
    dtype* out_ptr = data_out + j;
    const float* scale_ptr = scale + j;
    int ptr_out[1] = {0};
    const int8_t* ptr_in = data_in;
    const int8_t* ptr_w0 = weights_ptr + (N * j);
    int cnt_loop = cnt;
    int bias0 = is_bias ? bias[j] : 0;
    asm volatile(
        "pld [%[in]]                                @ preload cache line, "
        "input\n"
        "pld [%[w0]]                        @ preload cache line, weights r0\n"
        "vmov.u32 q0, #0                    @ set q0 to 0\n"
        "vmov s0, %[bias0]                  @ set q0 to bias0\n"
        "cmp %[cnt], #1                     @ check whether has main loop\n"
        "blt  2f                            @ jump to tail\n"
        /* main loop */
        "1:                                 @ main loop\n"
        "vld1.8 {d24-d25}, [%[in]]!         @ load input, q12\n"
        "vld1.8 {d28-d29}, [%[w0]]!         @ load weights q14\n"
        /* mull int8 * int8 = int16*/
        "vmull.s8 q1, d24, d28              @ mul add\n"
        "vmlal.s8 q1, d25, d29              @ mul add\n"
        "subs %[cnt] , #1                   @ sub loop count \n"
        /* pacc int16 + int32 = int32*/
        "vpadal.s16 q0, q1                  @ pair acc\n"
        "bne 1b                             @ jump to main loop\n"
        /* pair add to final result */
        "2:                                 @ end processing\n"
        "vpadd.s32 d2, d0, d1               @ pair add, first step\n"
        "vpadd.s32 d0, d2, d2               @ pair add, final step\n"
        /* write output */
        "vst1.32 {d0[0]}, [%[out]]          @ save result\n"
        : [in] "+r"(ptr_in), [w0] "+r"(ptr_w0), [cnt] "+r"(cnt_loop)
        : [bias0] "r"(bias0), [out] "r"(ptr_out)
        : "cc", "memory", "q0", "q1", "q12", "q13");
    for (int i = 0; i < tail; ++i) {
      ptr_out[0] += ptr_in[i] * ptr_w0[i];
    }
    if (is_relu) {
      ptr_out[0] = ptr_out[0] > 0 ? ptr_out[0] : 0;
    }
    write_gemv_out(ptr_out, out_ptr, scale_ptr);
  }
#endif  //__aarch64__ // NOLINT
  return true;
}

template bool gemv_int8<float>(const int8_t* A, const int8_t* x, float* y,
                               bool transA, int M, int N, const float* scale,
                               bool is_bias, const int* bias, bool is_relu);
template bool gemv_int8<int>(const int8_t* A, const int8_t* x, int* y,
                             bool transA, int M, int N, const float* scale,
                             bool is_bias, const int* bias, bool is_relu);
template bool gemv_int8<signed char>(const int8_t* A, const int8_t* x,
                                     signed char* y, bool transA, int M, int N,
                                     const float* scale, bool is_bias,
                                     const int* bias, bool is_relu);

}  // namespace math
}  // namespace arm
}  // namespace lite
}  // namespace paddle
