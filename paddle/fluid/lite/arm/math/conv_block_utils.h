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

#pragma once
#include <arm_neon.h>
#include <cmath>
#include "paddle/fluid/lite/arm/math/saturate.h"
#include "paddle/fluid/lite/core/target_wrapper.h"
#include "paddle/fluid/lite/utils/cp_logging.h"

namespace paddle {
namespace lite {
namespace arm {
namespace math {

#define AKMAX(a, b) ((a) > (b) ? (a) : (b))

inline void fill_packed_biasc4(float* dout, const float* bias, int size) {
  float32x4_t vb = vld1q_f32(bias);
  int cnt = size / 4;
  for (int i = 0; i < cnt; ++i) {
    vst1q_f32(dout, vb);
    dout += 4;
  }
}

/*preprocessing weights
* input weights: [chout, chin/ group, kh, kw] --> outputs weights: [chout / n,
* chin/ group, kh, kw, n]
*/
template <typename dtype>
static bool conv_trans_weights_numc(const dtype* din, dtype* dout, int chout,
                                    int chin, int n, int kernel_size) {
  if (n <= 0) {
    LOG(ERROR) << "ch_n and hei_n are more than zero";
    return false;
  }
  int c_loop = chout / n;
  int chout_round = (chout + n - 1) / n;
  int win_stride = chin * kernel_size;
  int wout_stride = n * win_stride;
  int co = 0;
  for (; co < c_loop; ++co) {
    dtype* dout_c = dout + co * wout_stride;
    const dtype* din_array[n];
    din_array[0] = din + co * wout_stride;
    for (int i = 1; i < n; i++) {
      din_array[i] = din_array[i - 1] + win_stride;
    }
    for (int ci = 0; ci < chin; ++ci) {
      for (int k = 0; k < kernel_size; ++k) {
        for (int i = 0; i < n; i++) {
          *(dout_c++) = *(din_array[i]++);
        }
      }
    }
  }
  // pad final chout
  if (chout_round > c_loop) {
    dtype* dout_c = dout + c_loop * wout_stride;
    const dtype* din_array[n];
    din_array[0] = din + c_loop * wout_stride;
    for (int i = 1; i < n; i++) {
      din_array[i] = din_array[i - 1] + win_stride;
    }
    // deal remain
    int cremain = chout_round * n - chout;
    for (int i = 1; i <= cremain; i++) {
      din_array[n - i] = din_array[0];
    }
    for (int ci = 0; ci < chin; ++ci) {
      for (int k = 0; k < kernel_size; ++k) {
        for (int i = 0; i < n; i++) {
          *(dout_c++) = *(din_array[i]++);
        }
      }
    }
  }
  return true;
}
/*preprocessing inputs
* input din: [1, chin, he-hs, we - ws] --> outputs dout: [n, chin, 1, we - ws]
* n = he - hs
*/
template <typename dtype>
static bool prepack_input_nxw(const dtype* din, dtype* dout, int cs, int ce,
                              int hs, int he, int ws, int we, int channel,
                              int width, int height, dtype* zero_ptr) {
  int n = he - hs;
  if (n <= 0) {
    LOG(ERROR) << "hei_n is more than zero";
    return false;
  }
  int w0 = ws < 0 ? 0 : ws;
  int w1 = we > width ? width : we;

  int size_w = we - ws;
  int size_wc_len = size_w * channel;
  int size_c = width * height;

  int valid_w = w1 - w0;
  size_t valid_w_byte = valid_w * sizeof(dtype);

  dtype* out_array[n];
  out_array[0] = dout;
  for (int i = 1; i < n; i++) {
    out_array[i] = out_array[i - 1] + size_wc_len;
  }

  for (int c = 0; c < channel; ++c) {
    int j = 0;
    // valid height
    for (int i = hs; i < he; i++) {
      // get address
      const dtype* in_array;
      if (i < 0 || i >= height) {
        in_array = zero_ptr;
      } else {
        in_array = din + i * width;
      }

      for (int w = ws; w < w0; ++w) {
        *(out_array[j]++) = 0.f;
      }
      memcpy(out_array[j], in_array, valid_w_byte);
      out_array[j] += valid_w;
      for (int w = w1; w < we; ++w) {
        *(out_array[j]++) = 0.f;
      }
      j++;
    }
    din += size_c;
  }
  return true;
}

/*wirte result in outputs
* input din: [n, c, h, w], output dout: [n, c, h, w]
*/
inline bool write_to_output_c1_fp32(const float* din, float* dout, int cs,
                                    int ce, int hs, int he, int ws, int we,
                                    int channel, int height, int width,
                                    bool flag_relu, float* trash_ptr) {
  if (cs > channel) {
    return true;
  }

  const int c1 = 1;
  const int w4 = 4;

  int size_c_out = width * height;

  float* doutc0r0 = dout + cs * size_c_out + hs * width + ws;

  const float* ptr_din = din;

  int size_h = (he > height ? height : he) - hs;  // size_h == hei_n

  int w_round = we - ws;
  int cnt = (width - ws) / w4;

  for (int i = 0; i < size_h; i++) {
    int size_w = i * width;
    float* doutc0_ptr = doutc0r0 + size_w;  // doutc0r0 + width;
    const float* din_hei_ptr = ptr_din + i * w_round * c1;
    if (cnt > 0) {
      int cnt_loop = cnt;
      if (flag_relu) {
#ifdef __aarch64__
        asm volatile(
            "ldr q0, [%[ptr_din]], #16      \n" /* load data, c0r0, c0r1, c0r2,
                                                   c0r3 */
            "movi v20.4s, #0                \n" /* for relu */
            "1:                             \n" /* main loop*/
            "fmax   v1.4s, v0.4s, v20.4s    \n" /*relu*/
            "ldr q0, [%[ptr_din]], #16      \n" /* load data, c0r0, c0r1, c0r2,
                                                   c0r3 */
            "subs   %w[cnt], %w[cnt], #1    \n" /* loop count -1*/
            "str    q1, [%[doutc0r0]], #16  \n" /* store c0r0*/
            "bne    1b                      \n" /* jump to main loop*/
            : [doutc0r0] "+r"(doutc0_ptr), [cnt] "+r"(cnt_loop),
              [ptr_din] "+r"(din_hei_ptr)
            :
            : "v0", "v1", "v20");
#else
        asm volatile(
            "vld1.32 {d0-d1}, [%[ptr_din]]!                 @ load data, c0r0, "
            "c1r0, c0r1, c1r1, , c0r2, c1r2, c0r3, c1r3\n"
            "vmov.u32 q15, #0                       @ dump zero\n"
            "1:                                     @ main loop\n"

            "vmax.f32   q1, q0, q15                 @ relu\n"
            "vld1.32 {d0-d1}, [%[ptr_din]]!         @ load data \n"

            "vst1.32  {d2-d3}, [%[doutc0r0]]!       @ store result, add "
            "pointer\n"

            "subs   %[cnt], %[cnt], #1              @ loop count - 1\n"

            "bne    1b                              @ jump to main loop\n"

            : [doutc0r0] "+r"(doutc0_ptr), [ptr_din] "+r"(din_hei_ptr),
              [cnt] "+r"(cnt_loop)
            :
            : "q0", "q1", "q15");
#endif
      } else {
#ifdef __aarch64__
        asm volatile(
            "ldr q0, [%[ptr_din]], #16      \n" /* load data, c0r0, c0r1, c0r2,
                                                   c0r3 */
            "1:                             \n" /* main loop*/
            "str    q0, [%[doutc0r0]], #16  \n" /* store c2r0*/
            "subs   %w[cnt], %w[cnt], #1    \n" /* loop count -1*/
            "ldr q0, [%[ptr_din]], #16      \n" /* load data, c0r0, c0r1, c0r2,
                                                   c0r3 */
            "bne    1b                      \n" /* jump to main loop*/

            : [doutc0r0] "+r"(doutc0_ptr), [cnt] "+r"(cnt_loop),
              [ptr_din] "+r"(din_hei_ptr)
            :
            : "v0");
#else
        asm volatile(
            "vld1.32 {d0-d1}, [%[ptr_din]]!                 @ load data, c0r0, "
            "c0r1, c0r2, c0r3\n"
            "1:                                     @ main loop\n"
            "vst1.32  {d0-d1}, [%[doutc0r0]]!       @ store result, add "
            "pointer\n"
            "subs   %[cnt], %[cnt], #1              @ loop count - 1\n"
            "vld1.32 {d0-d1}, [%[ptr_din]]!         @ load data \n"
            "bne    1b                              @ jump to main loop\n"

            : [doutc0r0] "+r"(doutc0_ptr), [ptr_din] "+r"(din_hei_ptr),
              [cnt] "+r"(cnt_loop)
            :
            : "q0");
#endif
      }
    }
    if (we > width) {
      int offset = i * w_round * c1 + c1 * w4 * cnt;
      din_hei_ptr = ptr_din + offset;
      int j = we - w4;
      if (flag_relu) {
        for (; j < width; ++j) {
          *(doutc0_ptr++) = AKMAX(din_hei_ptr[0], 0.f);
          din_hei_ptr++;
        }
      } else {
        for (; j < width; ++j) {
          *(doutc0_ptr++) = *(din_hei_ptr++);
        }
      }
    }
  }
  return true;
}

/*wirte result in outputs
* input din: [n, c / 4, h, w * 4], output dout: [n, c, h, w]
*/
inline bool write_to_output_c2_fp32(const float* din, float* dout, int cs,
                                    int ce, int hs, int he, int ws, int we,
                                    int channel, int height, int width,
                                    bool flag_relu, float* trash_ptr) {
  if (cs > channel) {
    return true;
  }

  const int c2 = 2;
  const int w4 = 4;

  //    float trash_ptr[width];

  int size_c_out = width * height;

  float* doutc0r0 = dout + cs * size_c_out + hs * width + ws;
  float* doutc1r0 = doutc0r0 + size_c_out;

  const float* ptr_din = din;

  int size_h = (he > height ? height : he) - hs;  // size_h == hei_n

  int w_round = we - ws;
  int cnt = (width - ws) / w4;

  for (int i = 0; i < size_h; i++) {
    int size_w = i * width;
    float* doutc0_ptr = doutc0r0 + size_w;  // doutc0r0 + width;
    float* doutc1_ptr = doutc1r0 + size_w;
    if (ce > channel) {
      switch (ce - channel) {
        case 1:
          doutc1_ptr = trash_ptr;
        default:
          break;
      }
    }
    const float* din_hei_ptr = ptr_din + i * w_round * c2;
    if (cnt > 0) {
      int cnt_loop = cnt;
      if (flag_relu) {
#ifdef __aarch64__
        asm volatile(
            "ldp q0, q1, [%[ptr_din]], #32  \n" /* load data, c0r0, c1r0, c0r1,
                                                   c1r1, , c0r2, c1r2, c0r3,
                                                   c1r3 */
            "movi v20.4s, #0                \n" /* for relu */
            "1:                             \n" /* main loop*/
            "trn1   v2.4s, v0.4s, v1.4s     \n" /* trans q0, q1*/
            "trn2   v3.4s, v0.4s, v1.4s     \n" /* trans q0, q1*/
            "ldp q0, q1, [%[ptr_din]], #32  \n" /* load data, c0r0, c1r0, c0r1,
                                                   c1r1, , c0r2, c1r2, c0r3,
                                                   c1r3  */
            "trn1   v4.2d, v2.2d, v3.2d     \n" /* trans q8, q10*/
            "trn2   v5.2d, v2.2d, v3.2d     \n" /* trans q8, q10*/

            "fmax   v2.4s, v4.4s, v20.4s    \n" /*relu*/
            "fmax   v3.4s, v5.4s, v20.4s    \n" /*relu*/

            "subs   %w[cnt], %w[cnt], #1    \n" /* loop count -1*/

            "str    q2, [%[doutc0r0]], #16  \n" /* store c0r0*/
            "str    q3, [%[doutc1r0]], #16  \n" /* store c2r0*/

            "bne    1b                      \n" /* jump to main loop*/

            : [doutc0r0] "+r"(doutc0_ptr), [doutc1r0] "+r"(doutc1_ptr),
              [cnt] "+r"(cnt_loop), [ptr_din] "+r"(din_hei_ptr)
            :
            : "v0", "v1", "v2", "v3", "v4", "v5", "v20");
#else
        asm volatile(
            "vld1.32 {d0-d3}, [%[ptr_din]]!                 @ load data, c0r0, "
            "c1r0, c0r1, c1r1, , c0r2, c1r2, c0r3, c1r3\n"
            "vmov.u32 q15, #0                       @ dump zero\n"
            "1:                                     @ main loop\n"
            "vtrn.32 d0, d1                         @ trans data:c0r0, c0r1, "
            "c1r0, c1r1 \n"
            "vtrn.32 d2, d3                         @ trans data:c0r2, c0r3, "
            "c1r2, c1r3 \n"

            "vswp  d1, d2                           @ swap data\n"

            "vmax.f32   q0, q0, q15                 @ relu\n"
            "vmax.f32   q1, q1, q15                 @ relu\n"

            "vst1.32  {d0-d1}, [%[doutc0r0]]!       @ store result, add "
            "pointer\n"
            "vst1.32  {d2-d3}, [%[doutc1r0]]!       @ store result, add "
            "pointer\n"

            "subs   %[cnt], %[cnt], #1              @ loop count - 1\n"

            "vld1.32 {d0-d3}, [%[ptr_din]]!         @ load data \n"

            "bne    1b                              @ jump to main loop\n"

            : [doutc0r0] "+r"(doutc0_ptr), [doutc1r0] "+r"(doutc1_ptr),
              [ptr_din] "+r"(din_hei_ptr), [cnt] "+r"(cnt_loop)
            :
            : "q0", "q1", "q2", "q3", "q15");
#endif
      } else {
#ifdef __aarch64__
        asm volatile(
            "ldp q0, q1, [%[ptr_din]], #32  \n" /* load data, c0r0, c1r0, c0r1,
                                                   c1r1, , c0r2, c1r2, c0r3,
                                                   c1r3 */
            "1:                             \n" /* main loop*/
            "trn1   v2.4s, v0.4s, v1.4s     \n" /* trans q0, q1*/
            "trn2   v3.4s, v0.4s, v1.4s     \n" /* trans q0, q1*/
            "ldp q0, q1, [%[ptr_din]], #32  \n" /* load data, c0r0, c1r0, c0r1,
                                                   c1r1, , c0r2, c1r2, c0r3,
                                                   c1r3  */
            "trn1   v4.2d, v2.2d, v3.2d     \n" /* trans q8, q10*/
            "trn2   v5.2d, v2.2d, v3.2d     \n" /* trans q8, q10*/

            "subs   %w[cnt], %w[cnt], #1    \n" /* loop count -1*/

            "str    q4, [%[doutc0r0]], #16  \n" /* store c0r0*/
            "str    q5, [%[doutc1r0]], #16  \n" /* store c2r0*/

            "bne    1b                      \n" /* jump to main loop*/

            : [doutc0r0] "+r"(doutc0_ptr), [doutc1r0] "+r"(doutc1_ptr),
              [cnt] "+r"(cnt_loop), [ptr_din] "+r"(din_hei_ptr)
            :
            : "v0", "v1", "v2", "v3", "v4", "v5");
#else
        asm volatile(
            "vld1.32 {d0-d3}, [%[ptr_din]]!                 @ load data, c0r0, "
            "c1r0, c0r1, c1r1, , c0r2, c1r2, c0r3, c1r3\n"
            "1:                                     @ main loop\n"
            "vtrn.32 d0, d1                         @ trans data:c0r0, c0r1, "
            "c1r0, c1r1 \n"
            "vtrn.32 d2, d3                         @ trans data:c0r2, c0r3, "
            "c1r2, c1r3 \n"

            "vswp  d1, d2                           @ swap data\n"

            "vst1.32  {d0-d1}, [%[doutc0r0]]!       @ store result, add "
            "pointer\n"
            "vst1.32  {d2-d3}, [%[doutc1r0]]!       @ store result, add "
            "pointer\n"

            "subs   %[cnt], %[cnt], #1              @ loop count - 1\n"

            "vld1.32 {d0-d3}, [%[ptr_din]]!         @ load data \n"

            "bne    1b                              @ jump to main loop\n"

            : [doutc0r0] "+r"(doutc0_ptr), [doutc1r0] "+r"(doutc1_ptr),
              [ptr_din] "+r"(din_hei_ptr), [cnt] "+r"(cnt_loop)
            :
            : "q0", "q1", "q2", "q3", "q15");
#endif
      }
    }
    if (we > width) {
      int offset = i * w_round * c2 + c2 * w4 * cnt;
      din_hei_ptr = ptr_din + offset;
      int j = we - w4;
      if (flag_relu) {
        for (; j < width; ++j) {
          *(doutc0_ptr++) = AKMAX(din_hei_ptr[0], 0.f);
          *(doutc1_ptr++) = AKMAX(din_hei_ptr[1], 0.f);
          din_hei_ptr += 2;
        }
      } else {
        for (; j < width; ++j) {
          *(doutc0_ptr++) = *(din_hei_ptr++);
          *(doutc1_ptr++) = *(din_hei_ptr++);
        }
      }
    }
  }
  return true;
}

/*wirte result in outputs
* input din: [n, c / 4, h, w * 4], output dout: [n, c, h, w]
*/
inline bool write_to_output_c4_fp32(const float* din, float* dout, int cs,
                                    int ce, int hs, int he, int ws, int we,
                                    int channel, int height, int width,
                                    bool flag_relu, float* trash_ptr) {
  const int c4 = 4;
  const int w4 = 4;
  const int w_round = we - ws;
  const int ch_n = ce - cs;
  if (ch_n != 4) {
    LOG(ERROR) << "write_to_output_c4_fp32 ch_n must be equal 4 and hei_n is "
                  "more than zero";
    return false;
  }
  int size_c_out = width * height;

  float* doutc0r0 = dout + cs * size_c_out + hs * width + ws;
  float* doutc1r0 = doutc0r0 + size_c_out;
  float* doutc2r0 = doutc1r0 + size_c_out;
  float* doutc3r0 = doutc2r0 + size_c_out;

  const float* ptr_din = din;

  int size_h = (he > height ? height : he) - hs;  // size_h == hei_n

  int cnt = (width - ws) / w4;

  for (int i = 0; i < size_h; i++) {
    int size_w = i * width;
    float* doutc0_ptr = doutc0r0 + size_w;  // doutc0r0 + width;
    float* doutc1_ptr = doutc1r0 + size_w;
    float* doutc2_ptr = doutc2r0 + size_w;
    float* doutc3_ptr = doutc3r0 + size_w;
    if (ce > channel) {
      switch (ce - channel) {
        case 3:
          doutc1_ptr = trash_ptr;
        case 2:
          doutc2_ptr = trash_ptr;
        case 1:
          doutc3_ptr = trash_ptr;
        default:
          break;
      }
    }
    const float* din_hei_ptr = ptr_din + i * w_round * ch_n;
    if (cnt > 0) {
      int cnt_loop = cnt;
      if (flag_relu) {
#ifdef __aarch64__
        asm volatile(
            "ldp q0, q1, [%[ptr_din]], #32  \n" /* load r00, r01 to q0, q1 */
            "ldp q2, q3, [%[ptr_din]], #32  \n" /* load r02, r03 to q2, q3 */
            "movi v20.4s, #0                \n" /* for relu */
            "1:                             \n" /* main loop*/
            "trn1   v8.4s, v0.4s, v1.4s     \n" /* trans q0, q1*/
            "trn2   v9.4s, v0.4s, v1.4s     \n" /* trans q0, q1*/
            "ldp q0, q1, [%[ptr_din]], #32  \n" /* load r00, r01 to q0, q1 */
            "trn1   v10.4s, v2.4s, v3.4s    \n" /* trans q2, q3*/
            "trn2   v11.4s, v2.4s, v3.4s    \n" /* trans q2, q3*/
            "ldp q2, q3, [%[ptr_din]], #32  \n" /* load r02, r03 to q2, q3 */
            "trn1   v16.2d, v8.2d, v10.2d   \n" /* trans q8, q10*/
            "trn2   v17.2d, v8.2d, v10.2d   \n" /* trans q8, q10*/
            "trn1   v18.2d, v9.2d, v11.2d   \n" /* trans q9, q11*/
            "trn2   v19.2d, v9.2d, v11.2d   \n" /* trans q9, q11*/
            "fmax   v16.4s, v16.4s, v20.4s  \n" /*relu*/
            "fmax   v17.4s, v17.4s, v20.4s  \n" /*relu*/
            "fmax   v18.4s, v18.4s, v20.4s  \n" /*relu*/
            "fmax   v19.4s, v19.4s, v20.4s  \n" /*relu*/
            "str    q16, [%[doutc0r0]], #16 \n" /* store c0r0*/
            "str    q17, [%[doutc2r0]], #16 \n" /* store c2r0*/
            "str    q18, [%[doutc1r0]], #16 \n" /* store c1r0*/
            "str    q19, [%[doutc3r0]], #16 \n" /* store c3r0*/

            "subs   %w[cnt], %w[cnt], #1    \n" /* loop count -1*/
            "bne    1b                      \n" /* jump to main loop*/

            : [doutc0r0] "+r"(doutc0_ptr), [doutc1r0] "+r"(doutc1_ptr),
              [doutc2r0] "+r"(doutc2_ptr), [doutc3r0] "+r"(doutc3_ptr),
              [cnt] "+r"(cnt_loop), [ptr_din] "+r"(din_hei_ptr)
            :
            : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10",
              "v11", "v12", "v13", "v14", "v16", "v17", "v18", "v19", "v20");
#else
        asm volatile(
            "vld1.32 {d0-d3}, [%[ptr_din]]!                 @load data \n"
            "vld1.32 {d4-d7}, [%[ptr_din]]!         @load data \n"
            "vmov.u32 q15, #0                       @ dump zero\n"
            "1:                                     @ main loop\n"
            "vtrn.32 q0, q1                         @ trans data:c00c01c20c21 "
            "\n"
            "vtrn.32 q2, q3                         @ trans data:c02c03c22c23 "
            "\n"

            "vswp   d1, d4                          @ swap data\n"
            "vswp   d3, d6                          @ swap data\n"

            "vmax.f32   q0, q0, q15        @ relu\n"
            "vmax.f32   q1, q1, q15        @ relu\n"
            "vmax.f32   q2, q2, q15        @ relu\n"
            "vmax.f32   q3, q3, q15        @ relu\n"

            "vst1.32  {d0-d1}, [%[doutc0r0]]!     @ store result, add pointer\n"
            "vst1.32  {d2-d3}, [%[doutc1r0]]!     @ store result, add pointer\n"
            "vst1.32  {d4-d5}, [%[doutc2r0]]!     @ store result, add pointer\n"
            "vst1.32  {d6-d7}, [%[doutc3r0]]!     @ store result, add pointer\n"

            "subs   %[cnt], %[cnt], #1    @ loop count - 1\n"

            "vld1.32 {d0-d3}, [%[ptr_din]]!        @load data \n"
            "vld1.32 {d4-d7}, [%[ptr_din]]!        @load data \n"

            "bne    1b                            @ jump to main loop\n"

            : [doutc0r0] "+r"(doutc0_ptr), [doutc1r0] "+r"(doutc1_ptr),
              [doutc2r0] "+r"(doutc2_ptr), [doutc3r0] "+r"(doutc3_ptr),
              [ptr_din] "+r"(din_hei_ptr), [cnt] "+r"(cnt_loop)
            :
            : "q0", "q1", "q2", "q3", "q15");
#endif
      } else {
#ifdef __aarch64__
        asm volatile(
            "ldp q0, q1, [%[ptr_din]], #32  \n" /* load r00, r01 to q0, q1 */
            "ldp q2, q3, [%[ptr_din]], #32  \n" /* load r02, r03 to q2, q3 */
            "1:                             \n" /* main loop*/
            "trn1   v8.4s, v0.4s, v1.4s     \n" /* trans q0, q1*/
            "trn2   v9.4s, v0.4s, v1.4s     \n" /* trans q0, q1*/
            "ldp q0, q1, [%[ptr_din]], #32  \n" /* load r00, r01 to q0, q1 */
            "trn1   v10.4s, v2.4s, v3.4s    \n" /* trans q2, q3*/
            "trn2   v11.4s, v2.4s, v3.4s    \n" /* trans q2, q3*/
            "ldp q2, q3, [%[ptr_din]], #32  \n" /* load r02, r03 to q2, q3 */
            "trn1   v16.2d, v8.2d, v10.2d   \n" /* trans q8, q10*/
            "trn2   v17.2d, v8.2d, v10.2d   \n" /* trans q8, q10*/
            "trn1   v18.2d, v9.2d, v11.2d   \n" /* trans q9, q11*/
            "trn2   v19.2d, v9.2d, v11.2d   \n" /* trans q9, q11*/
            "str    q16, [%[doutc0r0]], #16 \n" /* store c0r0*/
            "str    q17, [%[doutc2r0]], #16 \n" /* store c2r0*/
            "str    q18, [%[doutc1r0]], #16 \n" /* store c1r0*/
            "str    q19, [%[doutc3r0]], #16 \n" /* store c3r0*/

            "subs   %w[cnt], %w[cnt], #1    \n" /* loop count -1*/
            "bne    1b                      \n" /* jump to main loop*/

            : [doutc0r0] "+r"(doutc0_ptr), [doutc1r0] "+r"(doutc1_ptr),
              [doutc2r0] "+r"(doutc2_ptr), [doutc3r0] "+r"(doutc3_ptr),
              [cnt] "+r"(cnt_loop), [ptr_din] "+r"(din_hei_ptr)
            :
            : "v0", "v1", "v2", "v3", "v8", "v9", "v10", "v11", "v16", "v17",
              "v18", "v19");
#else
        asm volatile(
            "vld1.32 {d0-d3}, [%[ptr_din]]!                 @load data \n"
            "vld1.32 {d4-d7}, [%[ptr_din]]!         @load data \n"
            "1:                                     @ main loop\n"
            "vtrn.32 q0, q1                         @ trans data:c00c01c20c21 "
            "\n"
            "vtrn.32 q2, q3                         @ trans data:c02c03c22c23 "
            "\n"

            "vswp   d1, d4                          @ swap data\n"
            "vswp   d3, d6                          @ swap data\n"

            "vst1.32  {d0-d1}, [%[doutc0r0]]!     @ store result, add pointer\n"
            "vst1.32  {d2-d3}, [%[doutc1r0]]!     @ store result, add pointer\n"
            "vst1.32  {d4-d5}, [%[doutc2r0]]!     @ store result, add pointer\n"
            "vst1.32  {d6-d7}, [%[doutc3r0]]!     @ store result, add pointer\n"

            "subs   %[cnt], %[cnt], #1    @ loop count - 1\n"

            "vld1.32 {d0-d3}, [%[ptr_din]]!        @load data \n"
            "vld1.32 {d4-d7}, [%[ptr_din]]!        @load data \n"

            "bne    1b                            @ jump to main loop\n"

            : [doutc0r0] "+r"(doutc0_ptr), [doutc1r0] "+r"(doutc1_ptr),
              [doutc2r0] "+r"(doutc2_ptr), [doutc3r0] "+r"(doutc3_ptr),
              [ptr_din] "+r"(din_hei_ptr), [cnt] "+r"(cnt_loop)
            :
            : "q0", "q1", "q2", "q3");
#endif
      }
    }
    if (we > width) {
      int offset = i * w_round * c4 + c4 * w4 * cnt;
      din_hei_ptr = ptr_din + offset;
      int j = we - w4;
      if (flag_relu) {
        for (; j < width; ++j) {
          *(doutc0_ptr++) = AKMAX(din_hei_ptr[0], 0.f);
          *(doutc1_ptr++) = AKMAX(din_hei_ptr[1], 0.f);
          *(doutc2_ptr++) = AKMAX(din_hei_ptr[2], 0.f);
          *(doutc3_ptr++) = AKMAX(din_hei_ptr[3], 0.f);
          din_hei_ptr += w4;
        }
      } else {
        for (; j < width; ++j) {
          *(doutc0_ptr++) = din_hei_ptr[0];
          *(doutc1_ptr++) = din_hei_ptr[1];
          *(doutc2_ptr++) = din_hei_ptr[2];
          *(doutc3_ptr++) = din_hei_ptr[3];
          din_hei_ptr += w4;
        }
      }
    }
  }
  return true;
}

/*wirte result in outputs
* input din: [n, c / 8, h, w * 8], output dout: [n, c, h, w]
*/
inline bool write_to_output_c8_fp32(const float* din, float* dout, int ch_n,
                                    int hei_n, int cs, int ce, int hs, int he,
                                    int ws, int we, int channel, int height,
                                    int width, bool flag_relu,
                                    float* trash_ptr) {
  if (ch_n != 8 || hei_n <= 0) {
    LOG(ERROR) << "ch_n must be equal 8 and hei_n is more than zero";
    return false;
  }
  int size_c_out = width * height;

  float* doutc0r0 = dout + cs * size_c_out + hs * width + ws;
  float* doutc1r0 = doutc0r0 + size_c_out;
  float* doutc2r0 = doutc1r0 + size_c_out;
  float* doutc3r0 = doutc2r0 + size_c_out;
  float* doutc4r0 = doutc3r0 + size_c_out;
  float* doutc5r0 = doutc4r0 + size_c_out;
  float* doutc6r0 = doutc5r0 + size_c_out;
  float* doutc7r0 = doutc6r0 + size_c_out;

  const float* ptr_din = din;

  int size_h = (he > height ? height : he) - hs;  // size_h == hei_n

  int valid_w = we - ws;
  int cnt = valid_w / 4;

  if (we > width) {
    cnt--;
  }
  if (flag_relu) {
    for (int i = 0; i < size_h; i++) {
      int size_w = i * width;
      float* doutc0_ptr = doutc0r0 + size_w;  // doutc0r0 + width;
      float* doutc1_ptr = doutc1r0 + size_w;
      float* doutc2_ptr = doutc2r0 + size_w;
      float* doutc3_ptr = doutc3r0 + size_w;
      float* doutc4_ptr = doutc4r0 + size_w;
      float* doutc5_ptr = doutc5r0 + size_w;
      float* doutc6_ptr = doutc6r0 + size_w;
      float* doutc7_ptr = doutc7r0 + size_w;
      if (ce > channel) {
        switch (ce - channel) {
          case 7:
            doutc1_ptr = trash_ptr;
          case 6:
            doutc2_ptr = trash_ptr;
          case 5:
            doutc3_ptr = trash_ptr;
          case 4:
            doutc4_ptr = trash_ptr;
          case 3:
            doutc5_ptr = trash_ptr;
          case 2:
            doutc6_ptr = trash_ptr;
          case 1:
            doutc7_ptr = trash_ptr;
          default:
            break;
        }
      }
      ptr_din = din + i * valid_w * ch_n;
      const float* din_hei_ptr = ptr_din;
      if (cnt > 0) {
        int cnt_loop = cnt;
#ifdef __aarch64__
        asm volatile(
            "ldp q0, q1, [%[ptr_din]], #32  \n" /* load r00, r01 to q0, q1 */
            "ldp q2, q3, [%[ptr_din]], #32  \n" /* load r02, r03 to q2, q3 */
            "ldp q4, q5, [%[ptr_din]], #32  \n" /* load r00, r01 to q0, q1 */
            "ldp q6, q7, [%[ptr_din]], #32  \n" /* load r02, r03 to q2, q3 */
            "movi v20.4s, #0                \n" /* for relu */
            "1:                             \n" /* main loop*/
            "trn1   v8.4s, v0.4s, v2.4s     \n" /* trans q0, q1*/
            "trn2   v9.4s, v0.4s, v2.4s     \n" /* trans q0, q1*/
            "trn1   v10.4s, v1.4s, v3.4s    \n" /* trans q2, q3*/
            "trn2   v11.4s, v1.4s, v3.4s    \n" /* trans q2, q3*/
            "ldp q0, q1, [%[ptr_din]], #32  \n" /* load r00, r01 to q0, q1 */

            "trn1   v12.4s, v4.4s, v6.4s    \n" /* trans q0, q1*/
            "trn2   v13.4s, v4.4s, v6.4s    \n" /* trans q0, q1*/
            "trn1   v14.4s, v5.4s, v7.4s    \n" /* trans q2, q3*/
            "trn2   v15.4s, v5.4s, v7.4s    \n" /* trans q2, q3*/
            "ldp q2, q3, [%[ptr_din]], #32  \n" /* load r02, r03 to q2, q3 */

            "trn1   v16.2d, v8.2d, v12.2d   \n" /* trans q8, q10 00 01 02 03*/
            "trn2   v17.2d, v8.2d, v12.2d   \n" /* trans q8, q10 20 21 22 23*/
            "trn1   v18.2d, v9.2d, v13.2d   \n" /* trans q9, q11 10 11 12 13*/
            "trn2   v19.2d, v9.2d, v13.2d   \n" /* trans q9, q11 30 31 32 33*/
            "ldp q4, q5, [%[ptr_din]], #32  \n" /* load r00, r01 to q0, q1 */

            "trn1   v8.2d, v10.2d, v14.2d   \n" /* trans q8, q10 40 41 42 43*/
            "trn2   v9.2d, v10.2d, v14.2d   \n" /* trans q8, q10 60 61 62 63*/
            "trn1   v12.2d, v11.2d, v15.2d  \n" /* trans q9, q11 50 51 52 53*/
            "trn2   v13.2d, v11.2d, v15.2d  \n" /* trans q9, q11 70 71 72 73*/
            "ldp q6, q7, [%[ptr_din]], #32  \n" /* load r02, r03 to q2, q3 */

            "fmax   v16.4s, v16.4s, v20.4s  \n" /*relu*/
            "fmax   v17.4s, v17.4s, v20.4s  \n" /*relu*/
            "fmax   v18.4s, v18.4s, v20.4s  \n" /*relu*/
            "fmax   v19.4s, v19.4s, v20.4s  \n" /*relu*/

            "fmax   v8.4s,  v8.4s,  v20.4s  \n" /*relu*/
            "fmax   v9.4s,  v9.4s,  v20.4s  \n" /*relu*/
            "fmax   v12.4s, v12.4s, v20.4s  \n" /*relu*/
            "fmax   v13.4s, v13.4s, v20.4s  \n" /*relu*/

            "str    q16, [%[doutc0r0]], #16 \n" /* store c0r0*/
            "str    q17, [%[doutc2r0]], #16 \n" /* store c2r0*/
            "str    q18, [%[doutc1r0]], #16 \n" /* store c1r0*/
            "str    q19, [%[doutc3r0]], #16 \n" /* store c3r0*/

            "subs   %w[cnt], %w[cnt], #1    \n" /* loop count -1*/
            "str    q8,  [%[doutc4r0]], #16 \n" /* store c0r0*/
            "str    q9,  [%[doutc6r0]], #16 \n" /* store c2r0*/
            "str    q12, [%[doutc5r0]], #16 \n" /* store c1r0*/
            "str    q13, [%[doutc7r0]], #16 \n" /* store c3r0*/

            "bne    1b                      \n" /* jump to main loop*/

            : [doutc0r0] "+r"(doutc0_ptr), [doutc1r0] "+r"(doutc1_ptr),
              [doutc2r0] "+r"(doutc2_ptr), [doutc3r0] "+r"(doutc3_ptr),
              [doutc4r0] "+r"(doutc4_ptr), [doutc5r0] "+r"(doutc5_ptr),
              [doutc6r0] "+r"(doutc6_ptr), [doutc7r0] "+r"(doutc7_ptr),
              [cnt] "+r"(cnt_loop), [ptr_din] "+r"(din_hei_ptr)
            :
            : "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10",
              "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19",
              "v20");
#else
        asm volatile(
            "vld1.32 {d0-d3}, [%[ptr_din]]!        @load data \n"
            "vld1.32 {d4-d7}, [%[ptr_din]]!        @load data \n"
            "vld1.32 {d8-d11}, [%[ptr_din]]!       @load data \n"
            "vld1.32 {d12-d15}, [%[ptr_din]]!      @load data \n"
            "vmov.u32 q15, #0                      @ dump zero\n"
            "1:                                    @ main loop\n"
            "vtrn.32   q0, q2                      @ trans q0, q2 \n"
            "vtrn.32   q4, q6                      @ trans q4, q6 \n"
            "vswp.32   d1, d8                      @ swap  d1, d8 \n"
            "vswp.32   d5, d12                     @ swap  d5, d12\n"

            "vtrn.32   q1, q3                      @ trans q1, q3 \n"
            "vtrn.32   q5, q7                      @ trans q5, q7 \n"
            "vswp.32   d3, d10                     @ swap  d3, d10\n"
            "vswp.32   d7, d14                     @ swap  d7, d14\n"

            "vmax.f32  q0, q0, q15                 @ relu\n"
            "vmax.f32  q1, q1, q15                 @ relu\n"
            "vmax.f32  q2, q2, q15                 @ relu\n"
            "vmax.f32  q3, q3, q15                 @ relu\n"

            "vmax.f32  q4, q4, q15                 @ relu\n"
            "vmax.f32  q5, q5, q15                 @ relu\n"
            "vmax.f32  q6, q6, q15                 @ relu\n"
            "vmax.f32  q7, q7, q15                 @ relu\n"

            "subs   %[cnt], %[cnt], #1             @ loop count - 1\n"
            "vst1.32   {d0-d1}, [%[doutc0r0]]!     @ store result, add "
            "pointer\n"
            "vst1.32   {d2-d3}, [%[doutc4r0]]!     @ store result, add "
            "pointer\n"
            "vst1.32   {d4-d5}, [%[doutc1r0]]!     @ store result, add "
            "pointer\n"
            "vst1.32   {d6-d7}, [%[doutc5r0]]!     @ store result, add "
            "pointer\n"

            "vld1.32   {d0-d3}, [%[ptr_din]]!      @load data \n"
            "vld1.32   {d4-d7}, [%[ptr_din]]!      @load data \n"

            "vst1.32   {d8-d9},   [%[doutc2r0]]!   @ store result, add "
            "pointer\n"
            "vst1.32   {d10-d11}, [%[doutc6r0]]!   @ store result, add "
            "pointer\n"
            "vst1.32   {d12-d13}, [%[doutc3r0]]!   @ store result, add "
            "pointer\n"
            "vst1.32   {d14-d15}, [%[doutc7r0]]!   @ store result, add "
            "pointer\n"

            "vld1.32 {d8-d11}, [%[ptr_din]]!       @load data \n"
            "vld1.32 {d12-d15}, [%[ptr_din]]!      @load data \n"

            "bne    1b                             @ jump to main loop\n"

            : [doutc0r0] "+r"(doutc0_ptr), [doutc1r0] "+r"(doutc1_ptr),
              [doutc2r0] "+r"(doutc2_ptr), [doutc3r0] "+r"(doutc3_ptr),
              [doutc4r0] "+r"(doutc4_ptr), [doutc5r0] "+r"(doutc5_ptr),
              [doutc6r0] "+r"(doutc6_ptr), [doutc7r0] "+r"(doutc7_ptr),
              [ptr_din] "+r"(din_hei_ptr), [cnt] "+r"(cnt_loop)
            :
            : "q0", "q1", "q2", "q3", "q4", "q15");
#endif
      }
      if (we > width) {
        int offset = 32 * (valid_w / 4 - 1);
        din_hei_ptr = ptr_din + offset;
        int i = we - 4;
        for (; i < width; ++i) {
          *(doutc0_ptr++) = AKMAX(din_hei_ptr[0], 0.f);
          *(doutc1_ptr++) = AKMAX(din_hei_ptr[1], 0.f);
          *(doutc2_ptr++) = AKMAX(din_hei_ptr[2], 0.f);
          *(doutc3_ptr++) = AKMAX(din_hei_ptr[3], 0.f);
          *(doutc4_ptr++) = AKMAX(din_hei_ptr[4], 0.f);
          *(doutc5_ptr++) = AKMAX(din_hei_ptr[5], 0.f);
          *(doutc6_ptr++) = AKMAX(din_hei_ptr[6], 0.f);
          *(doutc7_ptr++) = AKMAX(din_hei_ptr[7], 0.f);
          din_hei_ptr += 8;
        }
      }
    }
  } else {
    for (int i = 0; i < size_h; i++) {
      int size_w = i * width;
      float* doutc0_ptr = doutc0r0 + size_w;  // doutc0r0 + width;
      float* doutc1_ptr = doutc1r0 + size_w;
      float* doutc2_ptr = doutc2r0 + size_w;
      float* doutc3_ptr = doutc3r0 + size_w;
      float* doutc4_ptr = doutc4r0 + size_w;
      float* doutc5_ptr = doutc5r0 + size_w;
      float* doutc6_ptr = doutc6r0 + size_w;
      float* doutc7_ptr = doutc7r0 + size_w;
      if (ce > channel) {
        switch (ce - channel) {
          case 7:
            doutc1_ptr = trash_ptr;
          case 6:
            doutc2_ptr = trash_ptr;
          case 5:
            doutc3_ptr = trash_ptr;
          case 4:
            doutc4_ptr = trash_ptr;
          case 3:
            doutc5_ptr = trash_ptr;
          case 2:
            doutc6_ptr = trash_ptr;
          case 1:
            doutc7_ptr = trash_ptr;
          default:
            break;
        }
      }
      ptr_din = din + i * valid_w * ch_n;
      const float* din_hei_ptr = ptr_din;
      if (cnt > 0) {
        int cnt_loop = cnt;
#ifdef __aarch64__
        asm volatile(
            "ldp q0, q1, [%[ptr_din]], #32  \n" /* load r00, r01 to q0, q1 */
            "ldp q2, q3, [%[ptr_din]], #32  \n" /* load r02, r03 to q2, q3 */
            "ldp q4, q5, [%[ptr_din]], #32  \n" /* load r00, r01 to q0, q1 */
            "ldp q6, q7, [%[ptr_din]], #32  \n" /* load r02, r03 to q2, q3 */
            "1:                             \n" /* main loop*/
            "trn1   v8.4s, v0.4s, v2.4s     \n" /* trans q0, q1*/
            "trn2   v9.4s, v0.4s, v2.4s     \n" /* trans q0, q1*/
            "trn1   v10.4s, v1.4s, v3.4s    \n" /* trans q2, q3*/
            "trn2   v11.4s, v1.4s, v3.4s    \n" /* trans q2, q3*/
            "ldp q0, q1, [%[ptr_din]], #32  \n" /* load r00, r01 to q0, q1 */

            "trn1   v12.4s, v4.4s, v6.4s    \n" /* trans q0, q1*/
            "trn2   v13.4s, v4.4s, v6.4s    \n" /* trans q0, q1*/
            "trn1   v14.4s, v5.4s, v7.4s    \n" /* trans q2, q3*/
            "trn2   v15.4s, v5.4s, v7.4s    \n" /* trans q2, q3*/
            "ldp q2, q3, [%[ptr_din]], #32  \n" /* load r02, r03 to q2, q3 */

            "trn1   v16.2d, v8.2d, v12.2d   \n" /* trans q8, q10 00 01 02 03*/
            "trn2   v17.2d, v8.2d, v12.2d   \n" /* trans q8, q10 20 21 22 23*/
            "trn1   v18.2d, v9.2d, v13.2d   \n" /* trans q9, q11 10 11 12 13*/
            "trn2   v19.2d, v9.2d, v13.2d   \n" /* trans q9, q11 30 31 32 33*/
            "ldp q4, q5, [%[ptr_din]], #32  \n" /* load r00, r01 to q0, q1 */

            "trn1   v8.2d, v10.2d, v14.2d   \n" /* trans q8, q10 40 41 42 43*/
            "trn2   v9.2d, v10.2d, v14.2d   \n" /* trans q8, q10 60 61 62 63*/
            "trn1   v12.2d, v11.2d, v15.2d  \n" /* trans q9, q11 50 51 52 53*/
            "trn2   v13.2d, v11.2d, v15.2d  \n" /* trans q9, q11 70 71 72 73*/
            "ldp q6, q7, [%[ptr_din]], #32  \n" /* load r02, r03 to q2, q3 */

            "str    q16, [%[doutc0r0]], #16 \n" /* store c0r0*/
            "str    q17, [%[doutc2r0]], #16 \n" /* store c2r0*/
            "str    q18, [%[doutc1r0]], #16 \n" /* store c1r0*/
            "str    q19, [%[doutc3r0]], #16 \n" /* store c3r0*/

            "subs   %w[cnt], %w[cnt], #1    \n" /* loop count -1*/
            "str    q8,  [%[doutc4r0]], #16 \n" /* store c0r0*/
            "str    q9,  [%[doutc6r0]], #16 \n" /* store c2r0*/
            "str    q12, [%[doutc5r0]], #16 \n" /* store c1r0*/
            "str    q13, [%[doutc7r0]], #16 \n" /* store c3r0*/

            "bne    1b                      \n" /* jump to main loop*/

            : [doutc0r0] "+r"(doutc0_ptr), [doutc1r0] "+r"(doutc1_ptr),
              [doutc2r0] "+r"(doutc2_ptr), [doutc3r0] "+r"(doutc3_ptr),
              [doutc4r0] "+r"(doutc4_ptr), [doutc5r0] "+r"(doutc5_ptr),
              [doutc6r0] "+r"(doutc6_ptr), [doutc7r0] "+r"(doutc7_ptr),
              [cnt] "+r"(cnt_loop), [ptr_din] "+r"(din_hei_ptr)
            :
            : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10",
              "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19",
              "v20");
#else
        asm volatile(
            "vld1.32   {d0-d3}, [%[ptr_din]]!      @load data \n"
            "vld1.32   {d4-d7}, [%[ptr_din]]!      @load data \n"
            "vld1.32   {d8-d11}, [%[ptr_din]]!     @load data \n"
            "vld1.32   {d12-d15}, [%[ptr_din]]!    @load data \n"
            "1:                                    @ main loop\n"
            "vtrn.32   q0, q2                      @ trans q0, q2 \n"
            "vtrn.32   q4, q6                      @ trans q4, q6 \n"
            "vswp.32   d1, d8                      @ swap  d1, d8 \n"
            "vswp.32   d5, d12                     @ swap  d5, d12\n"

            "vtrn.32   q1, q3                      @ trans q1, q3 \n"
            "vtrn.32   q5, q7                      @ trans q5, q7 \n"
            "vswp.32   d3, d10                     @ swap  d3, d10\n"
            "vswp.32   d7, d14                     @ swap  d7, d14\n"

            "subs      %[cnt], %[cnt], #1          @ loop count - 1\n"

            "vst1.32   {d0-d1},   [%[doutc0r0]]!   @ store result, add "
            "pointer\n"
            "vst1.32   {d2-d3},   [%[doutc4r0]]!   @ store result, add "
            "pointer\n"
            "vst1.32   {d4-d5},   [%[doutc1r0]]!   @ store result, add "
            "pointer\n"
            "vst1.32   {d6-d7},   [%[doutc5r0]]!   @ store result, add "
            "pointer\n"

            "vld1.32   {d0-d3},   [%[ptr_din]]!    @load data \n"
            "vld1.32   {d4-d7},   [%[ptr_din]]!    @load data \n"

            "vst1.32   {d8-d9},   [%[doutc2r0]]!   @ store result, add "
            "pointer\n"
            "vst1.32   {d10-d11}, [%[doutc6r0]]!   @ store result, add "
            "pointer\n"
            "vst1.32   {d12-d13}, [%[doutc3r0]]!   @ store result, add "
            "pointer\n"
            "vst1.32   {d14-d15}, [%[doutc7r0]]!   @ store result, add "
            "pointer\n"

            "vld1.32 {d8-d11},  [%[ptr_din]]!      @load data \n"
            "vld1.32 {d12-d15}, [%[ptr_din]]!      @load data \n"

            "bne    1b                             @ jump to main loop\n"

            : [doutc0r0] "+r"(doutc0_ptr), [doutc1r0] "+r"(doutc1_ptr),
              [doutc2r0] "+r"(doutc2_ptr), [doutc3r0] "+r"(doutc3_ptr),
              [doutc4r0] "+r"(doutc4_ptr), [doutc5r0] "+r"(doutc5_ptr),
              [doutc6r0] "+r"(doutc6_ptr), [doutc7r0] "+r"(doutc7_ptr),
              [ptr_din] "+r"(din_hei_ptr), [cnt] "+r"(cnt_loop)
            :
            : "q0", "q1", "q2", "q3", "q4");
#endif
      }
      if (we > width) {
        int offset = 32 * (valid_w / 4 - 1);
        din_hei_ptr = ptr_din + offset;
        int i = we - 4;
        for (; i < width; ++i) {
          *(doutc0_ptr++) = din_hei_ptr[0];
          *(doutc1_ptr++) = din_hei_ptr[1];
          *(doutc2_ptr++) = din_hei_ptr[2];
          *(doutc3_ptr++) = din_hei_ptr[3];
          *(doutc4_ptr++) = din_hei_ptr[4];
          *(doutc5_ptr++) = din_hei_ptr[5];
          *(doutc6_ptr++) = din_hei_ptr[6];
          *(doutc7_ptr++) = din_hei_ptr[7];
          din_hei_ptr += 8;
        }
      }
    }
  }
  return true;
}

/*wirte result in outputs
* input din: [n, c / 4, h, w * 4], output dout: [n, c, h, w]
*/
inline bool write_to_output_c4_int32(const int* din, int* dout, int ch_n,
                                     int hei_n, int cs, int ce, int hs, int he,
                                     int ws, int we, int channel, int height,
                                     int width, bool flag_relu,
                                     int* trash_ptr) {
  if (ch_n != 4 || hei_n <= 0) {
    LOG(ERROR) << "ch_n must be equal 4 and hei_n is more than zero";
    return false;
  }
  int size_c_out = width * height;

  int* doutc0r0 = dout + cs * size_c_out + hs * width + ws;
  int* doutc1r0 = doutc0r0 + size_c_out;
  int* doutc2r0 = doutc1r0 + size_c_out;
  int* doutc3r0 = doutc2r0 + size_c_out;

  const int* ptr_din = din;

  int size_h = (he > height ? height : he) - hs;  // size_h == hei_n

  int valid_w = we - ws;
  int cnt = valid_w / 4;

  if (we > width) {
    cnt--;
  }
  if (flag_relu) {
    for (int i = 0; i < size_h; i++) {
      int size_w = i * width;
      int* doutc0_ptr = doutc0r0 + size_w;  // doutc0r0 + width;
      int* doutc1_ptr = doutc1r0 + size_w;
      int* doutc2_ptr = doutc2r0 + size_w;
      int* doutc3_ptr = doutc3r0 + size_w;
      if (ce > channel) {
        switch (ce - channel) {
          case 3:
            doutc1_ptr = trash_ptr;
          case 2:
            doutc2_ptr = trash_ptr;
          case 1:
            doutc3_ptr = trash_ptr;
          default:
            break;
        }
      }
      ptr_din = din + i * valid_w * ch_n;
      const int* din_hei_ptr = ptr_din;
      if (cnt > 0) {
        int cnt_loop = cnt;
#ifdef __aarch64__
        asm volatile(
            "ldp q0, q1, [%[ptr_din]], #32  \n" /* load r00, r01 to q0, q1 */
            "ldp q2, q3, [%[ptr_din]], #32  \n" /* load r02, r03 to q2, q3 */
            "movi v20.4s, #0                \n" /* for relu */
            "1:                             \n" /* main loop*/
            "trn1   v8.4s, v0.4s, v1.4s     \n" /* trans q0, q1*/
            "trn2   v9.4s, v0.4s, v1.4s     \n" /* trans q0, q1*/
            "ldp q0, q1, [%[ptr_din]], #32  \n" /* load r00, r01 to q0, q1 */
            "trn1   v10.4s, v2.4s, v3.4s    \n" /* trans q2, q3*/
            "trn2   v11.4s, v2.4s, v3.4s    \n" /* trans q2, q3*/
            "ldp q2, q3, [%[ptr_din]], #32  \n" /* load r02, r03 to q2, q3 */
            "trn1   v16.2d, v8.2d, v10.2d   \n" /* trans q8, q10*/
            "trn2   v17.2d, v8.2d, v10.2d   \n" /* trans q8, q10*/
            "trn1   v18.2d, v9.2d, v11.2d   \n" /* trans q9, q11*/
            "trn2   v19.2d, v9.2d, v11.2d   \n" /* trans q9, q11*/
            "smax   v16.4s, v16.4s, v20.4s  \n" /* relu */
            "smax   v17.4s, v17.4s, v20.4s  \n" /* relu */
            "smax   v18.4s, v18.4s, v20.4s  \n" /* relu */
            "smax   v19.4s, v19.4s, v20.4s  \n" /* relu */
            "str    q16, [%[doutc0r0]], #16 \n" /* store c0r0*/
            "str    q17, [%[doutc2r0]], #16 \n" /* store c2r0*/
            "str    q18, [%[doutc1r0]], #16 \n" /* store c1r0*/
            "str    q19, [%[doutc3r0]], #16 \n" /* store c3r0*/

            "subs   %w[cnt], %w[cnt], #1    \n" /* loop count -1*/
            "bne    1b                      \n" /* jump to main loop*/

            : [doutc0r0] "+r"(doutc0_ptr), [doutc1r0] "+r"(doutc1_ptr),
              [doutc2r0] "+r"(doutc2_ptr), [doutc3r0] "+r"(doutc3_ptr),
              [cnt] "+r"(cnt_loop), [ptr_din] "+r"(din_hei_ptr)
            :
            : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10",
              "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19",
              "v20");
#else
        asm volatile(
            "vld1.32    {d0-d3}, [%[ptr_din]]!    @load data \n"
            "vld1.32    {d4-d7}, [%[ptr_din]]!    @load data \n"
            "vmov.u32   q15, #0                   @ dump zero\n"
            "1:                                   @ main loop\n"
            "vtrn.32    q0, q1                    @ trans q0, q1 \n"
            "vtrn.32    q2, q3                    @ trans q2, q3 \n"
            "vswp.32    d1, d4                    @ swap d1, d4  \n"
            "vswp.32    d3, d6                    @ swap d3, d6  \n"

            "vmax.s32   q0, q0, q15               @ relu\n"
            "vmax.s32   q1, q1, q15               @ relu\n"
            "vmax.s32   q2, q2, q15               @ relu\n"
            "vmax.s32   q3, q3, q15               @ relu\n"

            "vst1.32  {d0-d1}, [%[doutc0r0]]!     @ store result, add pointer\n"
            "vst1.32  {d2-d3}, [%[doutc1r0]]!     @ store result, add pointer\n"
            "vst1.32  {d4-d5}, [%[doutc2r0]]!     @ store result, add pointer\n"
            "vst1.32  {d6-d7}, [%[doutc3r0]]!     @ store result, add pointer\n"

            "subs   %[cnt], %[cnt], #1            @ loop count - 1\n"

            "vld1.32 {d0-d3}, [%[ptr_din]]!       @load data \n"
            "vld1.32 {d4-d7}, [%[ptr_din]]!       @load data \n"

            "bne    1b                            @ jump to main loop\n"

            : [doutc0r0] "+r"(doutc0_ptr), [doutc1r0] "+r"(doutc1_ptr),
              [doutc2r0] "+r"(doutc2_ptr), [doutc3r0] "+r"(doutc3_ptr),
              [ptr_din] "+r"(din_hei_ptr), [cnt] "+r"(cnt_loop)
            :
            : "q0", "q1", "q2", "q3", "q4", "q15");
#endif
      }
      if (we > width) {
        int offset = 16 * (valid_w / 4 - 1);
        din_hei_ptr = ptr_din + offset;
        int i = we - 4;
        for (; i < width; ++i) {
          *(doutc0_ptr++) = AKMAX(din_hei_ptr[0], 0);
          *(doutc1_ptr++) = AKMAX(din_hei_ptr[1], 0);
          *(doutc2_ptr++) = AKMAX(din_hei_ptr[2], 0);
          *(doutc3_ptr++) = AKMAX(din_hei_ptr[3], 0);
          din_hei_ptr += 4;
        }
      }
    }
  } else {
    for (int i = 0; i < size_h; i++) {
      int size_w = i * width;
      int* doutc0_ptr = doutc0r0 + size_w;  // doutc0r0 + width;
      int* doutc1_ptr = doutc1r0 + size_w;
      int* doutc2_ptr = doutc2r0 + size_w;
      int* doutc3_ptr = doutc3r0 + size_w;
      if (ce > channel) {
        switch (ce - channel) {
          case 3:
            doutc1_ptr = trash_ptr;
          case 2:
            doutc2_ptr = trash_ptr;
          case 1:
            doutc3_ptr = trash_ptr;
          default:
            break;
        }
      }
      ptr_din = din + i * valid_w * ch_n;
      const int* din_hei_ptr = ptr_din;
      if (cnt > 0) {
        int cnt_loop = cnt;
#ifdef __aarch64__
        asm volatile(
            "ldp q0, q1, [%[ptr_din]], #32  \n" /* load r00, r01 to q0, q1 */
            "ldp q2, q3, [%[ptr_din]], #32  \n" /* load r02, r03 to q2, q3 */
            "1:                             \n" /* main loop*/
            "trn1   v8.4s, v0.4s, v1.4s     \n" /* trans q0, q1*/
            "trn2   v9.4s, v0.4s, v1.4s     \n" /* trans q0, q1*/
            "ldp q0, q1, [%[ptr_din]], #32  \n" /* load r00, r01 to q0, q1 */
            "trn1   v10.4s, v2.4s, v3.4s    \n" /* trans q2, q3*/
            "trn2   v11.4s, v2.4s, v3.4s    \n" /* trans q2, q3*/
            "ldp q2, q3, [%[ptr_din]], #32  \n" /* load r02, r03 to q2, q3 */
            "trn1   v16.2d, v8.2d, v10.2d   \n" /* trans q8, q10*/
            "trn2   v17.2d, v8.2d, v10.2d   \n" /* trans q8, q10*/
            "trn1   v18.2d, v9.2d, v11.2d   \n" /* trans q9, q11*/
            "trn2   v19.2d, v9.2d, v11.2d   \n" /* trans q9, q11*/

            "str    q16, [%[doutc0r0]], #16 \n" /* store c0r0*/
            "str    q17, [%[doutc2r0]], #16 \n" /* store c2r0*/
            "str    q18, [%[doutc1r0]], #16 \n" /* store c1r0*/
            "str    q19, [%[doutc3r0]], #16 \n" /* store c3r0*/

            "subs   %w[cnt], %w[cnt], #1    \n" /* loop count -1*/
            "bne    1b                      \n" /* jump to main loop*/

            : [doutc0r0] "+r"(doutc0_ptr), [doutc1r0] "+r"(doutc1_ptr),
              [doutc2r0] "+r"(doutc2_ptr), [doutc3r0] "+r"(doutc3_ptr),
              [cnt] "+r"(cnt_loop), [ptr_din] "+r"(din_hei_ptr)
            :
            : "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10",
              "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19",
              "v20");
#else
        asm volatile(
            "vld1.32    {d0-d3}, [%[ptr_din]]!     @load data \n"
            "vld1.32    {d4-d7}, [%[ptr_din]]!     @load data \n"
            "1:                                    @ main loop\n"
            "vtrn.32    q0, q1                     @ trans q0, q1\n"
            "vtrn.32    q2, q3                     @ trans q2, q3\n"
            "vswp.32    d1, d4                     @ swap d1, d4 \n"
            "vswp.32    d3, d6                     @ swap d3, d6 \n"

            "subs       %[cnt], %[cnt], #1         @ loop count - 1\n"
            "vst1.32    {d0-d1}, [%[doutc0r0]]!    @ store result, add "
            "pointer\n"
            "vst1.32    {d2-d3}, [%[doutc1r0]]!    @ store result, add "
            "pointer\n"
            "vst1.32    {d4-d5}, [%[doutc2r0]]!    @ store result, add "
            "pointer\n"
            "vst1.32    {d6-d7}, [%[doutc3r0]]!    @ store result, add "
            "pointer\n"

            "vld1.32    {d0-d3}, [%[ptr_din]]!     @load data \n"
            "vld1.32    {d4-d7}, [%[ptr_din]]!     @load data \n"

            "bne    1b                             @ jump to main loop\n"

            : [doutc0r0] "+r"(doutc0_ptr), [doutc1r0] "+r"(doutc1_ptr),
              [doutc2r0] "+r"(doutc2_ptr), [doutc3r0] "+r"(doutc3_ptr),
              [ptr_din] "+r"(din_hei_ptr), [cnt] "+r"(cnt_loop)
            :
            : "q0", "q1", "q2", "q3", "q4", "q15");
#endif
      }
      if (we > width) {
        int offset = 16 * (valid_w / 4 - 1);
        din_hei_ptr = ptr_din + offset;
        int i = we - 4;
        for (; i < width; ++i) {
          *(doutc0_ptr++) = din_hei_ptr[0];
          *(doutc1_ptr++) = din_hei_ptr[1];
          *(doutc2_ptr++) = din_hei_ptr[2];
          *(doutc3_ptr++) = din_hei_ptr[3];
          din_hei_ptr += 4;
        }
      }
    }
  }
  return true;
}

/*wirte result in outputs --int8, fp32
* input din: [n, c / 4, h, w * 4], output dout: [n, c, h, w]
*/
template <typename dtype>
inline bool write_to_output_c4_int32_1(const int* din, dtype* dout, int ch_n,
                                       int hei_n, int cs, int ce, int hs,
                                       int he, int ws, int we, int channel,
                                       int height, int width, bool flag_relu,
                                       dtype* trash_ptr, const float* scale,
                                       PrecisionType out_dtype) {
  if (ch_n != 4 || hei_n <= 0) {
    LOG(ERROR) << "ch_n must be equal 4 and hei_n is more than zero";
    return false;
  }
  int size_c_out = width * height;

  dtype* doutc0r0 = dout + cs * size_c_out + hs * width + ws;
  dtype* doutc1r0 = doutc0r0 + size_c_out;
  dtype* doutc2r0 = doutc1r0 + size_c_out;
  dtype* doutc3r0 = doutc2r0 + size_c_out;

  const int* ptr_din = din;

  int size_h = (he > height ? height : he) - hs;  // size_h == hei_n

  int valid_w = we - ws;
  int cnt = valid_w / 4;

  float32x4_t w_scale = vld1q_f32(scale);
  // float32x4_t vzero = vdupq_n_f32(0.f);

  if (we > width) {
    cnt--;
  }
  if (out_dtype == PRECISION(kFloat)) {
    // int32_to_fp32
    if (flag_relu) {
      for (int i = 0; i < size_h; i++) {
        int size_w = i * width;
        dtype* doutc0_ptr = doutc0r0 + size_w;  // doutc0r0 + width;
        dtype* doutc1_ptr = doutc1r0 + size_w;
        dtype* doutc2_ptr = doutc2r0 + size_w;
        dtype* doutc3_ptr = doutc3r0 + size_w;
        if (ce > channel) {
          switch (ce - channel) {
            case 3:
              doutc1_ptr = trash_ptr;
            case 2:
              doutc2_ptr = trash_ptr;
            case 1:
              doutc3_ptr = trash_ptr;
            default:
              break;
          }
        }
        ptr_din = din + i * valid_w * ch_n;
        const int* din_hei_ptr = ptr_din;
        if (cnt > 0) {
          int cnt_loop = cnt;
#ifdef __aarch64__
          asm volatile(
              "ldp q0, q1, [%[ptr_din]], #32  \n" /* load r00, r01 to q0, q1 */
              "ldp q2, q3, [%[ptr_din]], #32  \n" /* load r02, r03 to q2, q3 */
              "movi v20.4s, #0                \n" /* for relu */
              "1:                             \n" /* main loop*/
              "trn1   v8.4s, v0.4s, v1.4s     \n" /* trans q0, q1*/
              "trn2   v9.4s, v0.4s, v1.4s     \n" /* trans q0, q1*/
              "ldp q0, q1, [%[ptr_din]], #32  \n" /* load r00, r01 to q0, q1 */
              "trn1   v10.4s, v2.4s, v3.4s    \n" /* trans q2, q3*/
              "trn2   v11.4s, v2.4s, v3.4s    \n" /* trans q2, q3*/
              "ldp q2, q3, [%[ptr_din]], #32  \n" /* load r02, r03 to q2, q3 */
              "trn1   v16.2d, v8.2d, v10.2d   \n" /* trans q8, q10*/
              "trn2   v17.2d, v8.2d, v10.2d   \n" /* trans q8, q10*/
              "trn1   v18.2d, v9.2d, v11.2d   \n" /* trans q9, q11*/
              "trn2   v19.2d, v9.2d, v11.2d   \n" /* trans q9, q11*/
              "smax   v16.4s, v16.4s, v20.4s  \n" /* relu */
              "smax   v17.4s, v17.4s, v20.4s  \n" /* relu */
              "smax   v18.4s, v18.4s, v20.4s  \n" /* relu */
              "smax   v19.4s, v19.4s, v20.4s  \n" /* relu */
              // int32 --> fp32
              "scvtf   v4.4s, v16.4s               \n"
              "scvtf   v5.4s, v17.4s               \n"
              "scvtf   v6.4s, v18.4s               \n"
              "scvtf   v7.4s, v19.4s               \n"
              // mul
              "fmul    v16.4s, v4.4s, %[scale].s[0]  \n"
              "fmul    v17.4s, v5.4s, %[scale].s[2] \n"
              "fmul    v18.4s, v6.4s, %[scale].s[1] \n"
              "fmul    v19.4s, v7.4s, %[scale].s[3] \n"
              // res
              "str    q16, [%[doutc0r0]], #16 \n" /* store c0r0*/
              "str    q17, [%[doutc2r0]], #16 \n" /* store c2r0*/
              "str    q18, [%[doutc1r0]], #16 \n" /* store c1r0*/
              "str    q19, [%[doutc3r0]], #16 \n" /* store c3r0*/

              "subs   %w[cnt], %w[cnt], #1    \n" /* loop count -1*/
              "bne    1b                      \n" /* jump to main loop*/

              : [doutc0r0] "+r"(doutc0_ptr), [doutc1r0] "+r"(doutc1_ptr),
                [doutc2r0] "+r"(doutc2_ptr), [doutc3r0] "+r"(doutc3_ptr),
                [cnt] "+r"(cnt_loop), [ptr_din] "+r"(din_hei_ptr)
              : [scale] "w"(w_scale)
              : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9",
                "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18",
                "v19", "v20");
#else
          asm volatile(
              "vld1.32    {d4-d7}, [%[ptr_din]]!    @load data \n"
              "vld1.32    {d8-d11}, [%[ptr_din]]!    @load data \n"
              "vmov.u32   q15, #0                   @ dump zero\n"
              "1:                                   @ main loop\n"
              "vtrn.32    q2, q3                    @ trans q0, q1 \n"
              "vtrn.32    q4, q5                    @ trans q2, q3 \n"
              "vswp.32    d5, d8                    @ swap d1, d4  \n"
              "vswp.32    d7, d10                    @ swap d3, d6  \n"

              "vmax.s32   q2, q2, q15               @ relu\n"
              "vmax.s32   q3, q3, q15               @ relu\n"
              "vmax.s32   q4, q4, q15               @ relu\n"
              "vmax.s32   q5, q5, q15               @ relu\n"

              // int32-> fp32
              "vcvt.f32.s32   q6, q2                  \n"
              "vcvt.f32.s32   q7, q3                  \n"
              "vcvt.f32.s32   q8, q4                  \n"
              "vcvt.f32.s32   q9, q5                  \n"

              // mul
              "vmul.f32  q2, q6, %e[scale][0]       \n"
              "vmul.f32  q3, q7, %e[scale][1]       \n"
              "vmul.f32  q4, q8, %f[scale][0]      \n"
              "vmul.f32  q5, q9, %f[scale][1]     \n"

              "vst1.32  {d4-d5}, [%[doutc0r0]]!     @ store result, add "
              "pointer\n"
              "vst1.32  {d6-d7}, [%[doutc1r0]]!     @ store result, add "
              "pointer\n"
              "vst1.32  {d8-d9}, [%[doutc2r0]]!     @ store result, add "
              "pointer\n"
              "vst1.32  {d10-d11}, [%[doutc3r0]]!     @ store result, add "
              "pointer\n"

              "subs   %[cnt], %[cnt], #1            @ loop count - 1\n"

              "vld1.32 {d4-d7}, [%[ptr_din]]!       @load data \n"
              "vld1.32 {d8-d11}, [%[ptr_din]]!       @load data \n"

              "bne    1b                            @ jump to main loop\n"

              : [doutc0r0] "+r"(doutc0_ptr), [doutc1r0] "+r"(doutc1_ptr),
                [doutc2r0] "+r"(doutc2_ptr), [doutc3r0] "+r"(doutc3_ptr),
                [ptr_din] "+r"(din_hei_ptr), [cnt] "+r"(cnt_loop)
              : [scale] "w"(w_scale)
              : "q2", "q3", "q4", "q5", "q6", "q7", "q8", "q9", "q10", "q11",
                "q12", "q13", "q14", "q15");
#endif
        }
        if (we > width) {
          int offset = 16 * (valid_w / 4 - 1);
          din_hei_ptr = ptr_din + offset;
          int j = we - 4;
          for (; j < width; ++j) {
            *(doutc0_ptr++) = AKMAX(din_hei_ptr[0] * scale[0], 0);
            *(doutc1_ptr++) = AKMAX(din_hei_ptr[1] * scale[1], 0);
            *(doutc2_ptr++) = AKMAX(din_hei_ptr[2] * scale[2], 0);
            *(doutc3_ptr++) = AKMAX(din_hei_ptr[3] * scale[3], 0);
            din_hei_ptr += 4;
          }
        }
      }
    } else {
      for (int i = 0; i < size_h; i++) {
        int size_w = i * width;
        dtype* doutc0_ptr = doutc0r0 + size_w;  // doutc0r0 + width;
        dtype* doutc1_ptr = doutc1r0 + size_w;
        dtype* doutc2_ptr = doutc2r0 + size_w;
        dtype* doutc3_ptr = doutc3r0 + size_w;
        if (ce > channel) {
          switch (ce - channel) {
            case 3:
              doutc1_ptr = trash_ptr;
            case 2:
              doutc2_ptr = trash_ptr;
            case 1:
              doutc3_ptr = trash_ptr;
            default:
              break;
          }
        }
        ptr_din = din + i * valid_w * ch_n;
        const int* din_hei_ptr = ptr_din;
        if (cnt > 0) {
          int cnt_loop = cnt;
#ifdef __aarch64__
          asm volatile(
              "ldp q0, q1, [%[ptr_din]], #32  \n" /* load r00, r01 to q0, q1 */
              "ldp q2, q3, [%[ptr_din]], #32  \n" /* load r02, r03 to q2, q3 */
              "movi v20.4s, #0                \n" /* for relu */
              "1:                             \n" /* main loop*/
              "trn1   v8.4s, v0.4s, v1.4s     \n" /* trans q0, q1*/
              "trn2   v9.4s, v0.4s, v1.4s     \n" /* trans q0, q1*/
              "ldp q0, q1, [%[ptr_din]], #32  \n" /* load r00, r01 to q0, q1 */
              "trn1   v10.4s, v2.4s, v3.4s    \n" /* trans q2, q3*/
              "trn2   v11.4s, v2.4s, v3.4s    \n" /* trans q2, q3*/
              "ldp q2, q3, [%[ptr_din]], #32  \n" /* load r02, r03 to q2, q3 */
              "trn1   v16.2d, v8.2d, v10.2d   \n" /* trans q8, q10*/
              "trn2   v17.2d, v8.2d, v10.2d   \n" /* trans q8, q10*/
              "trn1   v18.2d, v9.2d, v11.2d   \n" /* trans q9, q11*/
              "trn2   v19.2d, v9.2d, v11.2d   \n" /* trans q9, q11*/
              // int32 --> fp32
              "scvtf   v4.4s, v16.4s               \n"
              "scvtf   v5.4s, v17.4s               \n"
              "scvtf   v6.4s, v18.4s               \n"
              "scvtf   v7.4s, v19.4s               \n"
              // mul
              "fmul    v16.4s, v4.4s, %[scale].s[0]  \n"
              "fmul    v17.4s, v5.4s, %[scale].s[2]  \n"
              "fmul    v18.4s, v6.4s, %[scale].s[1] \n"
              "fmul    v19.4s, v7.4s, %[scale].s[3] \n"
              // res
              "str    q16, [%[doutc0r0]], #16 \n" /* store c0r0*/
              "str    q17, [%[doutc2r0]], #16 \n" /* store c2r0*/
              "str    q18, [%[doutc1r0]], #16 \n" /* store c1r0*/
              "str    q19, [%[doutc3r0]], #16 \n" /* store c3r0*/

              "subs   %w[cnt], %w[cnt], #1    \n" /* loop count -1*/
              "bne    1b                      \n" /* jump to main loop*/

              : [doutc0r0] "+r"(doutc0_ptr), [doutc1r0] "+r"(doutc1_ptr),
                [doutc2r0] "+r"(doutc2_ptr), [doutc3r0] "+r"(doutc3_ptr),
                [cnt] "+r"(cnt_loop), [ptr_din] "+r"(din_hei_ptr)
              : [scale] "w"(w_scale)
              : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9",
                "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18",
                "v19", "v20");
#else
          asm volatile(
              "vld1.32    {d4-d7}, [%[ptr_din]]!    @load data \n"
              "vld1.32    {d8-d11}, [%[ptr_din]]!    @load data \n"
              "vmov.u32   q15, #0                   @ dump zero\n"
              "1:                                   @ main loop\n"
              "vtrn.32    q2, q3                    @ trans q0, q1 \n"
              "vtrn.32    q4, q5                    @ trans q2, q3 \n"
              "vswp.32    d5, d8                    @ swap d1, d4  \n"
              "vswp.32    d7, d10                    @ swap d3, d6  \n"

              // int32-> fp32
              "vcvt.f32.s32   q6, q2                  \n"
              "vcvt.f32.s32   q7, q3                  \n"
              "vcvt.f32.s32   q8, q4                  \n"
              "vcvt.f32.s32   q9, q5                  \n"

              // mul
              "vmul.f32  q2, q6, %e[scale][0]       \n"
              "vmul.f32  q3, q7, %e[scale][1]       \n"
              "vmul.f32  q4, q8, %f[scale][0]      \n"
              "vmul.f32  q5, q9, %f[scale][1]     \n"

              "vst1.32  {d4-d5}, [%[doutc0r0]]!     @ store result, add "
              "pointer\n"
              "vst1.32  {d6-d7}, [%[doutc1r0]]!     @ store result, add "
              "pointer\n"
              "vst1.32  {d8-d9}, [%[doutc2r0]]!     @ store result, add "
              "pointer\n"
              "vst1.32  {d10-d11}, [%[doutc3r0]]!     @ store result, add "
              "pointer\n"

              "subs   %[cnt], %[cnt], #1            @ loop count - 1\n"

              "vld1.32 {d4-d7}, [%[ptr_din]]!       @load data \n"
              "vld1.32 {d8-d11}, [%[ptr_din]]!       @load data \n"

              "bne    1b                            @ jump to main loop\n"

              : [doutc0r0] "+r"(doutc0_ptr), [doutc1r0] "+r"(doutc1_ptr),
                [doutc2r0] "+r"(doutc2_ptr), [doutc3r0] "+r"(doutc3_ptr),
                [ptr_din] "+r"(din_hei_ptr), [cnt] "+r"(cnt_loop)
              : [scale] "w"(w_scale)
              : "q2", "q3", "q4", "q5", "q6", "q7", "q8", "q9", "q10", "q11",
                "q12", "q13", "q14", "q15");
#endif
        }
        if (we > width) {
          int offset = 16 * (valid_w / 4 - 1);
          din_hei_ptr = ptr_din + offset;
          int j = we - 4;
          for (; j < width; ++j) {
            *(doutc0_ptr++) = din_hei_ptr[0] * scale[0];
            *(doutc1_ptr++) = din_hei_ptr[1] * scale[1];
            *(doutc2_ptr++) = din_hei_ptr[2] * scale[2];
            *(doutc3_ptr++) = din_hei_ptr[3] * scale[3];
            din_hei_ptr += 4;
          }
        }
      }
    }

  } else if (out_dtype == PRECISION(kInt8)) {
    // int32_to_int8
    if (flag_relu) {
      for (int i = 0; i < size_h; i++) {
        int size_w = i * width;
        dtype* doutc0_ptr = doutc0r0 + size_w;  // doutc0r0 + width;
        dtype* doutc1_ptr = doutc1r0 + size_w;
        dtype* doutc2_ptr = doutc2r0 + size_w;
        dtype* doutc3_ptr = doutc3r0 + size_w;
        if (ce > channel) {
          switch (ce - channel) {
            case 3:
              doutc1_ptr = trash_ptr;
            case 2:
              doutc2_ptr = trash_ptr;
            case 1:
              doutc3_ptr = trash_ptr;
            default:
              break;
          }
        }
        ptr_din = din + i * valid_w * ch_n;
        const int* din_hei_ptr = ptr_din;
        if (cnt > 0) {
          int cnt_loop = cnt;
#ifdef __aarch64__
          asm volatile(
              "ldp q0, q1, [%[ptr_din]], #32  \n" /* load r00, r01 to q0, q1 */
              "ldp q2, q3, [%[ptr_din]], #32  \n" /* load r02, r03 to q2, q3 */
              "movi v20.4s, #0                \n" /* for relu */
              "1:                             \n" /* main loop*/
              "trn1   v8.4s, v0.4s, v1.4s     \n" /* trans q0, q1*/
              "trn2   v9.4s, v0.4s, v1.4s     \n" /* trans q0, q1*/
              "ldp q0, q1, [%[ptr_din]], #32  \n" /* load r00, r01 to q0, q1 */
              "trn1   v10.4s, v2.4s, v3.4s    \n" /* trans q2, q3*/
              "trn2   v11.4s, v2.4s, v3.4s    \n" /* trans q2, q3*/
              "ldp q2, q3, [%[ptr_din]], #32  \n" /* load r02, r03 to q2, q3 */
              "trn1   v16.2d, v8.2d, v10.2d   \n" /* trans q8, q10*/
              "trn2   v17.2d, v8.2d, v10.2d   \n" /* trans q8, q10*/
              "trn1   v18.2d, v9.2d, v11.2d   \n" /* trans q9, q11*/
              "trn2   v19.2d, v9.2d, v11.2d   \n" /* trans q9, q11*/
              "smax   v16.4s, v16.4s, v20.4s  \n" /* relu */
              "smax   v17.4s, v17.4s, v20.4s  \n" /* relu */
              "smax   v18.4s, v18.4s, v20.4s  \n" /* relu */
              "smax   v19.4s, v19.4s, v20.4s  \n" /* relu */
              // int32 --> fp32
              "scvtf   v4.4s, v16.4s               \n"
              "scvtf   v5.4s, v17.4s               \n"
              "scvtf   v6.4s, v18.4s               \n"
              "scvtf   v7.4s, v19.4s               \n"

              // mul
              "fmul    v16.4s, v4.4s, %[scale].s[0]  \n"
              "fmul    v17.4s, v5.4s, %[scale].s[2]  \n"
              "fmul    v18.4s, v6.4s, %[scale].s[1] \n"
              "fmul    v19.4s, v7.4s, %[scale].s[3] \n"

              // fp32-int32
              "fcvtas  v4.4s, v16.4s                      \n"
              "fcvtas  v5.4s, v17.4s                      \n"
              "fcvtas  v6.4s, v18.4s                      \n"
              "fcvtas  v7.4s, v19.4s                      \n"

              // int32-int16
              "sqxtn   v8.4h, v4.4s                      \n"
              "sqxtn   v9.4h, v5.4s                      \n"
              "sqxtn   v10.4h, v6.4s                      \n"
              "sqxtn   v11.4h, v7.4s                      \n"

              "sqxtn  v16.8b, v8.8h                      \n"
              "sqxtn  v17.8b, v9.8h                     \n"
              "sqxtn  v18.8b, v10.8h                      \n"
              "sqxtn  v19.8b, v11.8h                     \n"
              // res
              "str     s16, [%[doutc0r0]], #4           \n"
              "str     s17, [%[doutc2r0]], #4           \n"
              "str     s18, [%[doutc1r0]], #4           \n"
              "str     s19, [%[doutc3r0]], #4           \n"

              "subs   %w[cnt], %w[cnt], #1    \n" /* loop count -1*/
              "bne    1b                      \n" /* jump to main loop*/

              : [doutc0r0] "+r"(doutc0_ptr), [doutc1r0] "+r"(doutc1_ptr),
                [doutc2r0] "+r"(doutc2_ptr), [doutc3r0] "+r"(doutc3_ptr),
                [cnt] "+r"(cnt_loop), [ptr_din] "+r"(din_hei_ptr)
              : [scale] "w"(w_scale)
              : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9",
                "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18",
                "v19", "v20");
#else
          asm volatile(
              "vld1.32    {d4-d7}, [%[ptr_din]]!    @load data \n"
              "vld1.32    {d8-d11}, [%[ptr_din]]!    @load data \n"
              "vmov.u32   q15, #0                   @ dump zero\n"
              "1:                                   @ main loop\n"
              "vtrn.32    q2, q3                    @ trans q0, q1 \n"
              "vtrn.32    q4, q5                    @ trans q2, q3 \n"
              "vswp.32    d5, d8                    @ swap d1, d4  \n"
              "vswp.32    d7, d10                    @ swap d3, d6  \n"

              "vmax.s32   q2, q2, q15               @ relu\n"
              "vmax.s32   q3, q3, q15               @ relu\n"
              "vmax.s32   q4, q4, q15               @ relu\n"
              "vmax.s32   q5, q5, q15               @ relu\n"

              // int32-> fp32
              "vcvt.f32.s32   q6, q2                  \n"
              "vcvt.f32.s32   q7, q3                  \n"
              "vcvt.f32.s32   q8, q4                  \n"
              "vcvt.f32.s32   q9, q5                  \n"

              "vmov.f32 q2, #0.5                    \n"

              // "vand.i32   q0, %q[vpoff], %q[vpoff]    @ set offset, 0.5\n"
              "vand.i32   q3, q2, q2                  @ set offset, 0.5\n"
              "vand.i32   q4, q2, q2                  @ set offset, 0.5\n"
              "vand.i32   q5, q2, q2                  @ set offset, 0.5\n"

              "vcgt.f32   q10, q6, q15           @ get mask > 0, in0\n"
              "vcgt.f32   q11, q7, q15           @ get mask > 0, in1\n"
              "vcgt.f32   q12, q8, q15          @ get mask > 0, in2\n"
              "vcgt.f32   q13, q9, q15          @ get mask > 0, in3\n"

              "vmov.f32 q15, #-0.5                    \n"

              "vbif.f32   q2, q15, q10           @ get right offset\n"
              "vbif.f32   q3, q15, q11           @ get right offset\n"
              "vbif.f32   q4, q15, q12          @ get right offset\n"
              "vbif.f32   q5, q15, q13          @ get right offset\n"

              "vmla.f32   q2, q6, %e[scale][0]          @ mul scale\n"
              "vmla.f32   q3, q7, %e[scale][1]          @ mul scale\n"
              "vmla.f32   q4, q8, %f[scale][0]          @ mul scale\n"
              "vmla.f32   q5, q9, %f[scale][1]          @ mul scale\n"

              "vcvt.s32.f32  q6, q2                   @ cvt to int32\n"
              "vcvt.s32.f32  q7, q3                   @ cvt to int32\n"
              "vcvt.s32.f32  q8, q4                   @ cvt to int32\n"
              "vcvt.s32.f32  q9, q5                   @ cvt to int32\n"

              "vqmovn.s32 d20, q6                     @ cnt to int16\n"
              "vqmovn.s32 d22, q7                     @ cnt to int16\n"
              "vqmovn.s32 d24, q8                     @ cnt to int16\n"
              "vqmovn.s32 d26, q9                     @ cnt to int16\n"

              "vqmovn.s16 d8, q10                      @ cnt to int8\n"
              "vqmovn.s16 d9, q11                      @ cnt to int8\n"
              "vqmovn.s16 d10, q12                      @ cnt to int8\n"
              "vqmovn.s16 d11, q13                      @ cnt to int8\n"

              "vst1.32 {d8[0]},    [%[doutc0r0]]         @ write to output\n"
              "vst1.32 {d9[0]},    [%[doutc1r0]]         @ write to output\n"
              "vst1.32 {d10[0]},    [%[doutc2r0]]         @ write to output\n"
              "vst1.32 {d11[0]},    [%[doutc3r0]]         @ write to output\n"

              "add %[doutc0r0], #4 \n"
              "add %[doutc1r0], #4 \n"
              "add %[doutc2r0], #4 \n"
              "add %[doutc3r0], #4 \n"

              "subs   %[cnt], %[cnt], #1            @ loop count - 1\n"
              "vmov.u32   q15, #0                   @ dump zero\n"

              "vld1.32 {d4-d7}, [%[ptr_din]]!       @load data \n"
              "vld1.32 {d8-d11}, [%[ptr_din]]!       @load data \n"

              "bne    1b                            @ jump to main loop\n"

              : [doutc0r0] "+r"(doutc0_ptr), [doutc1r0] "+r"(doutc1_ptr),
                [doutc2r0] "+r"(doutc2_ptr), [doutc3r0] "+r"(doutc3_ptr),
                [ptr_din] "+r"(din_hei_ptr), [cnt] "+r"(cnt_loop)
              : [scale] "w"(w_scale)
              : "q2", "q3", "q4", "q5", "q6", "q7", "q8", "q9", "q10", "q11",
                "q12", "q13", "q14", "q15");
#endif
        }
        if (we > width) {
          int offset = 16 * (valid_w / 4 - 1);
          din_hei_ptr = ptr_din + offset;
          int j = we - 4;
          for (; j < width; ++j) {
            *(doutc0_ptr++) = saturate_cast<signed char>(
                roundf(AKMAX(din_hei_ptr[0], 0) * scale[0]));
            *(doutc1_ptr++) = saturate_cast<signed char>(
                roundf(AKMAX(din_hei_ptr[1], 0) * scale[1]));
            *(doutc2_ptr++) = saturate_cast<signed char>(
                roundf(AKMAX(din_hei_ptr[2], 0) * scale[2]));
            *(doutc3_ptr++) = saturate_cast<signed char>(
                roundf(AKMAX(din_hei_ptr[3], 0) * scale[3]));
            din_hei_ptr += 4;
          }
        }
      }
    } else {
      for (int i = 0; i < size_h; i++) {  // size_h
        int size_w = i * width;
        dtype* doutc0_ptr = doutc0r0 + size_w;  // doutc0r0 + width;
        dtype* doutc1_ptr = doutc1r0 + size_w;
        dtype* doutc2_ptr = doutc2r0 + size_w;
        dtype* doutc3_ptr = doutc3r0 + size_w;
        if (ce > channel) {
          switch (ce - channel) {
            case 3:
              doutc1_ptr = trash_ptr;
            case 2:
              doutc2_ptr = trash_ptr;
            case 1:
              doutc3_ptr = trash_ptr;
            default:
              break;
          }
        }
        ptr_din = din + i * valid_w * ch_n;
        const int* din_hei_ptr = ptr_din;
        if (cnt > 0) {
          int cnt_loop = cnt;
#ifdef __aarch64__
          asm volatile(
              "ldp q0, q1, [%[ptr_din]], #32  \n" /* load r00, r01 to q0, q1 */
              "ldp q2, q3, [%[ptr_din]], #32  \n" /* load r02, r03 to q2, q3 */
              "movi v20.4s, #0                \n" /* for relu */
              "1:                             \n" /* main loop*/
              "trn1   v8.4s, v0.4s, v1.4s     \n" /* trans q0, q1*/
              "trn2   v9.4s, v0.4s, v1.4s     \n" /* trans q0, q1*/
              "ldp q0, q1, [%[ptr_din]], #32  \n" /* load r00, r01 to q0, q1 */
              "trn1   v10.4s, v2.4s, v3.4s    \n" /* trans q2, q3*/
              "trn2   v11.4s, v2.4s, v3.4s    \n" /* trans q2, q3*/
              "ldp q2, q3, [%[ptr_din]], #32  \n" /* load r02, r03 to q2, q3 */
              "trn1   v16.2d, v8.2d, v10.2d   \n" /* trans q8, q10*/
              "trn2   v17.2d, v8.2d, v10.2d   \n" /* trans q8, q10*/
              "trn1   v18.2d, v9.2d, v11.2d   \n" /* trans q9, q11*/
              "trn2   v19.2d, v9.2d, v11.2d   \n" /* trans q9, q11*/
              // int32 --> fp32
              "scvtf   v4.4s, v16.4s               \n"
              "scvtf   v5.4s, v17.4s               \n"
              "scvtf   v6.4s, v18.4s               \n"
              "scvtf   v7.4s, v19.4s               \n"

              // mul
              "fmul    v16.4s, v4.4s, %[scale].s[0]  \n"
              "fmul    v17.4s, v5.4s, %[scale].s[2]  \n"
              "fmul    v18.4s, v6.4s, %[scale].s[1] \n"
              "fmul    v19.4s, v7.4s, %[scale].s[3] \n"

              // fp32-int32
              "fcvtas  v4.4s, v16.4s                      \n"
              "fcvtas  v5.4s, v17.4s                      \n"
              "fcvtas  v6.4s, v18.4s                      \n"
              "fcvtas  v7.4s, v19.4s                      \n"

              // int32-int16
              "sqxtn   v8.4h, v4.4s                      \n"
              "sqxtn   v9.4h, v5.4s                      \n"
              "sqxtn   v10.4h, v6.4s                      \n"
              "sqxtn   v11.4h, v7.4s                      \n"

              "sqxtn  v16.8b, v8.8h                      \n"
              "sqxtn  v17.8b, v9.8h                     \n"
              "sqxtn  v18.8b, v10.8h                      \n"
              "sqxtn  v19.8b, v11.8h                     \n"
              // res
              "str     s16, [%[doutc0r0]], #4           \n"
              "str     s17, [%[doutc2r0]], #4           \n"
              "str     s18, [%[doutc1r0]], #4           \n"
              "str     s19, [%[doutc3r0]], #4           \n"

              "subs   %w[cnt], %w[cnt], #1    \n" /* loop count -1*/
              "bne    1b                      \n" /* jump to main loop*/

              : [doutc0r0] "+r"(doutc0_ptr), [doutc1r0] "+r"(doutc1_ptr),
                [doutc2r0] "+r"(doutc2_ptr), [doutc3r0] "+r"(doutc3_ptr),
                [cnt] "+r"(cnt_loop), [ptr_din] "+r"(din_hei_ptr)
              : [scale] "w"(w_scale)
              : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9",
                "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18",
                "v19", "v20");
#else
          asm volatile(
              "vld1.32    {d4-d7}, [%[ptr_din]]!    @load data \n"
              "vld1.32    {d8-d11}, [%[ptr_din]]!    @load data \n"
              "vmov.u32   q15, #0                   @ dump zero\n"
              "1:                                   @ main loop\n"
              "vtrn.32    q2, q3                    @ trans q0, q1 \n"
              "vtrn.32    q4, q5                    @ trans q2, q3 \n"
              "vswp.32    d5, d8                    @ swap d1, d4  \n"
              "vswp.32    d7, d10                    @ swap d3, d6  \n"

              // int32-> fp32
              "vcvt.f32.s32   q6, q2                  \n"
              "vcvt.f32.s32   q7, q3                  \n"
              "vcvt.f32.s32   q8, q4                  \n"
              "vcvt.f32.s32   q9, q5                  \n"

              "vmov.f32 q2, #0.5                    \n"

              // "vand.i32   q0, %q[vpoff], %q[vpoff]    @ set offset, 0.5\n"
              "vand.i32   q3, q2, q2                  @ set offset, 0.5\n"
              "vand.i32   q4, q2, q2                  @ set offset, 0.5\n"
              "vand.i32   q5, q2, q2                  @ set offset, 0.5\n"

              "vcgt.f32   q10, q6, q15           @ get mask > 0, in0\n"
              "vcgt.f32   q11, q7, q15           @ get mask > 0, in1\n"
              "vcgt.f32   q12, q8, q15          @ get mask > 0, in2\n"
              "vcgt.f32   q13, q9, q15          @ get mask > 0, in3\n"

              "vmov.f32 q15, #-0.5                    \n"

              "vbif.f32   q2, q15, q10           @ get right offset\n"
              "vbif.f32   q3, q15, q11           @ get right offset\n"
              "vbif.f32   q4, q15, q12          @ get right offset\n"
              "vbif.f32   q5, q15, q13          @ get right offset\n"

              "vmla.f32   q2, q6, %e[scale][0]          @ mul scale\n"
              "vmla.f32   q3, q7, %e[scale][1]          @ mul scale\n"
              "vmla.f32   q4, q8, %f[scale][0]          @ mul scale\n"
              "vmla.f32   q5, q9, %f[scale][1]          @ mul scale\n"

              "vcvt.s32.f32  q6, q2                   @ cvt to int32\n"
              "vcvt.s32.f32  q7, q3                   @ cvt to int32\n"
              "vcvt.s32.f32  q8, q4                   @ cvt to int32\n"
              "vcvt.s32.f32  q9, q5                   @ cvt to int32\n"

              "vqmovn.s32 d20, q6                     @ cnt to int16\n"
              "vqmovn.s32 d22, q7                     @ cnt to int16\n"
              "vqmovn.s32 d24, q8                     @ cnt to int16\n"
              "vqmovn.s32 d26, q9                     @ cnt to int16\n"

              "vqmovn.s16 d8, q10                      @ cnt to int8\n"
              "vqmovn.s16 d9, q11                      @ cnt to int8\n"
              "vqmovn.s16 d10, q12                      @ cnt to int8\n"
              "vqmovn.s16 d11, q13                      @ cnt to int8\n"

              "vst1.32 {d8[0]},    [%[doutc0r0]]         @ write to output\n"
              "vst1.32 {d9[0]},    [%[doutc1r0]]         @ write to output\n"
              "vst1.32 {d10[0]},    [%[doutc2r0]]         @ write to output\n"
              "vst1.32 {d11[0]},    [%[doutc3r0]]         @ write to output\n"

              "add %[doutc0r0], #4 \n"
              "add %[doutc1r0], #4 \n"
              "add %[doutc2r0], #4 \n"
              "add %[doutc3r0], #4 \n"

              "subs   %[cnt], %[cnt], #1            @ loop count - 1\n"

              "vld1.32 {d4-d7}, [%[ptr_din]]!       @load data \n"
              "vld1.32 {d8-d11}, [%[ptr_din]]!       @load data \n"
              "vmov.u32   q15, #0                   @ dump zero\n"

              "bne    1b                            @ jump to main loop\n"

              : [doutc0r0] "+r"(doutc0_ptr), [doutc1r0] "+r"(doutc1_ptr),
                [doutc2r0] "+r"(doutc2_ptr), [doutc3r0] "+r"(doutc3_ptr),
                [ptr_din] "+r"(din_hei_ptr), [cnt] "+r"(cnt_loop)
              : [scale] "w"(w_scale)
              : "q2", "q3", "q4", "q5", "q6", "q7", "q8", "q9", "q10", "q11",
                "q12", "q13", "q14", "q15");
#endif
        }
        if (we > width) {
          int offset = 16 * (valid_w / 4 - 1);
          din_hei_ptr = ptr_din + offset;
          int j = we - 4;
          for (; j < width; ++j) {
            *(doutc0_ptr++) =
                saturate_cast<int8_t>(roundf(din_hei_ptr[0] * scale[0]));
            *(doutc1_ptr++) =
                saturate_cast<int8_t>(roundf(din_hei_ptr[1] * scale[1]));
            *(doutc2_ptr++) =
                saturate_cast<int8_t>(roundf(din_hei_ptr[2] * scale[2]));
            *(doutc3_ptr++) =
                saturate_cast<int8_t>(roundf(din_hei_ptr[3] * scale[3]));
            din_hei_ptr += 4;
          }
        }
      }
    }
  } else {
    LOG(ERROR) << "ERROR: unsupported input data type!!";
    return false;
  }
  return true;
}

/*wirte result in outputs
* input din: [n, c / 8, h, w * 8], output dout: [n, c, h, w]
*/
inline bool write_to_output_c8_int32(const int* din, int* dout, int ch_n,
                                     int hei_n, int cs, int ce, int hs, int he,
                                     int ws, int we, int channel, int height,
                                     int width, bool flag_relu,
                                     int* trash_ptr) {
  if (ch_n != 8 || hei_n <= 0) {
    LOG(ERROR) << "ch_n must be equal 8 and hei_n is more than zero";
    return false;
  }
  int size_c_out = width * height;

  int* doutc0r0 = dout + cs * size_c_out + hs * width + ws;
  int* doutc1r0 = doutc0r0 + size_c_out;
  int* doutc2r0 = doutc1r0 + size_c_out;
  int* doutc3r0 = doutc2r0 + size_c_out;
  int* doutc4r0 = doutc3r0 + size_c_out;
  int* doutc5r0 = doutc4r0 + size_c_out;
  int* doutc6r0 = doutc5r0 + size_c_out;
  int* doutc7r0 = doutc6r0 + size_c_out;

  const int* ptr_din = din;

  int size_h = (he > height ? height : he) - hs;  // size_h == hei_n

  int valid_w = we - ws;
  int cnt = valid_w / 4;

  if (we > width) {
    cnt--;
  }
  if (flag_relu) {
    for (int i = 0; i < size_h; i++) {
      int size_w = i * width;
      int* doutc0_ptr = doutc0r0 + size_w;  // doutc0r0 + width;
      int* doutc1_ptr = doutc1r0 + size_w;
      int* doutc2_ptr = doutc2r0 + size_w;
      int* doutc3_ptr = doutc3r0 + size_w;
      int* doutc4_ptr = doutc4r0 + size_w;
      int* doutc5_ptr = doutc5r0 + size_w;
      int* doutc6_ptr = doutc6r0 + size_w;
      int* doutc7_ptr = doutc7r0 + size_w;
      if (ce > channel) {
        switch (ce - channel) {
          case 7:
            doutc1_ptr = trash_ptr;
          case 6:
            doutc2_ptr = trash_ptr;
          case 5:
            doutc3_ptr = trash_ptr;
          case 4:
            doutc4_ptr = trash_ptr;
          case 3:
            doutc5_ptr = trash_ptr;
          case 2:
            doutc6_ptr = trash_ptr;
          case 1:
            doutc7_ptr = trash_ptr;
          default:
            break;
        }
      }
      ptr_din = din + i * valid_w * ch_n;
      const int* din_hei_ptr = ptr_din;
      if (cnt > 0) {
        int cnt_loop = cnt;
#ifdef __aarch64__
        asm volatile(
            "ldp q0, q1, [%[ptr_din]], #32  \n" /* load r00, r01 to q0, q1 */
            "ldp q2, q3, [%[ptr_din]], #32  \n" /* load r02, r03 to q2, q3 */
            "ldp q4, q5, [%[ptr_din]], #32  \n" /* load r00, r01 to q0, q1 */
            "ldp q6, q7, [%[ptr_din]], #32  \n" /* load r02, r03 to q2, q3 */
            "movi v20.4s, #0                \n" /* for relu */
            "1:                             \n" /* main loop*/
            "trn1   v8.4s, v0.4s, v2.4s     \n" /* trans q0, q1*/
            "trn2   v9.4s, v0.4s, v2.4s     \n" /* trans q0, q1*/
            "trn1   v10.4s, v1.4s, v3.4s    \n" /* trans q2, q3*/
            "trn2   v11.4s, v1.4s, v3.4s    \n" /* trans q2, q3*/
            "ldp q0, q1, [%[ptr_din]], #32  \n" /* load r00, r01 to q0, q1 */

            "trn1   v12.4s, v4.4s, v6.4s    \n" /* trans q0, q1*/
            "trn2   v13.4s, v4.4s, v6.4s    \n" /* trans q0, q1*/
            "trn1   v14.4s, v5.4s, v7.4s    \n" /* trans q2, q3*/
            "trn2   v15.4s, v5.4s, v7.4s    \n" /* trans q2, q3*/
            "ldp q2, q3, [%[ptr_din]], #32  \n" /* load r02, r03 to q2, q3 */

            "trn1   v16.2d, v8.2d, v12.2d   \n" /* trans q8, q10 00 01 02 03*/
            "trn2   v17.2d, v8.2d, v12.2d   \n" /* trans q8, q10 20 21 22 23*/
            "trn1   v18.2d, v9.2d, v13.2d   \n" /* trans q9, q11 10 11 12 13*/
            "trn2   v19.2d, v9.2d, v13.2d   \n" /* trans q9, q11 30 31 32 33*/
            "ldp q4, q5, [%[ptr_din]], #32  \n" /* load r00, r01 to q0, q1 */

            "trn1   v8.2d, v10.2d, v14.2d   \n" /* trans q8, q10 40 41 42 43*/
            "trn2   v9.2d, v10.2d, v14.2d   \n" /* trans q8, q10 60 61 62 63*/
            "trn1   v12.2d, v11.2d, v15.2d  \n" /* trans q9, q11 50 51 52 53*/
            "trn2   v13.2d, v11.2d, v15.2d  \n" /* trans q9, q11 70 71 72 73*/
            "ldp q6, q7, [%[ptr_din]], #32  \n" /* load r02, r03 to q2, q3 */

            "smax   v16.4s, v16.4s, v20.4s  \n" /*relu*/
            "smax   v17.4s, v17.4s, v20.4s  \n" /*relu*/
            "smax   v18.4s, v18.4s, v20.4s  \n" /*relu*/
            "smax   v19.4s, v19.4s, v20.4s  \n" /*relu*/

            "smax   v8.4s, v8.4s, v20.4s    \n" /*relu*/
            "smax   v9.4s, v9.4s, v20.4s    \n" /*relu*/
            "smax   v12.4s, v12.4s, v20.4s  \n" /*relu*/
            "smax   v13.4s, v13.4s, v20.4s  \n" /*relu*/

            "str    q16, [%[doutc0r0]], #16 \n" /* store c0r0*/
            "str    q17, [%[doutc2r0]], #16 \n" /* store c2r0*/
            "str    q18, [%[doutc1r0]], #16 \n" /* store c1r0*/
            "str    q19, [%[doutc3r0]], #16 \n" /* store c3r0*/

            "subs   %w[cnt], %w[cnt],  #1   \n" /* loop count -1*/
            "str    q8, [%[doutc4r0]], #16  \n" /* store c0r0*/
            "str    q9, [%[doutc6r0]], #16  \n" /* store c2r0*/
            "str    q12, [%[doutc5r0]], #16 \n" /* store c1r0*/
            "str    q13, [%[doutc7r0]], #16 \n" /* store c3r0*/

            "bne    1b                      \n" /* jump to main loop*/

            : [doutc0r0] "+r"(doutc0_ptr), [doutc1r0] "+r"(doutc1_ptr),
              [doutc2r0] "+r"(doutc2_ptr), [doutc3r0] "+r"(doutc3_ptr),
              [doutc4r0] "+r"(doutc4_ptr), [doutc5r0] "+r"(doutc5_ptr),
              [doutc6r0] "+r"(doutc6_ptr), [doutc7r0] "+r"(doutc7_ptr),
              [cnt] "+r"(cnt_loop), [ptr_din] "+r"(din_hei_ptr)
            :
            : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10",
              "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19",
              "v20");
#else
        asm volatile(
            "vld1.32 {d0-d3},   [%[ptr_din]]!   @load data \n"
            "vld1.32 {d4-d7},   [%[ptr_din]]!   @load data \n"
            "vld1.32 {d8-d11},  [%[ptr_din]]!   @load data \n"
            "vld1.32 {d12-d15}, [%[ptr_din]]!   @load data \n"
            "vmov.s32   q15, #0                 @ dump zero\n"
            "1:                                 @ main loop\n"
            "vtrn.32    q0, q2                  @ trans q0, q2 \n"
            "vtrn.32    q4, q6                  @ trans q4, q6 \n"
            "vswp.32    d1, d8                  @ swap  d1, d8 \n"
            "vswp.32    d5, d12                 @ swap  d5, d12\n"

            "vtrn.32    q1, q3                  @ trans q1, q3 \n"
            "vtrn.32    q5, q7                  @ trans q5, q7 \n"
            "vswp.32    d3, d10                 @ swap  d3, d10\n"
            "vswp.32    d7, d14                 @ swap  d7, d14\n"

            "vmax.s32   q0, q0, q15             @ relu\n"
            "vmax.s32   q1, q1, q15             @ relu\n"
            "vmax.s32   q2, q2, q15             @ relu\n"
            "vmax.s32   q3, q3, q15             @ relu\n"

            "vmax.s32   q4, q4, q15             @ relu\n"
            "vmax.s32   q5, q5, q15             @ relu\n"
            "vmax.s32   q6, q6, q15             @ relu\n"
            "vmax.s32   q7, q7, q15             @ relu\n"

            "subs   %[cnt], %[cnt], #1          @ loop count - 1\n"

            "vst1.32  {d0-d1}, [%[doutc0r0]]!   @ store result, add pointer\n"
            "vst1.32  {d2-d3}, [%[doutc4r0]]!   @ store result, add pointer\n"
            "vst1.32  {d4-d5}, [%[doutc1r0]]!   @ store result, add pointer\n"
            "vst1.32  {d6-d7}, [%[doutc5r0]]!   @ store result, add pointer\n"

            "vld1.32 {d0-d3}, [%[ptr_din]]!      @load data \n"
            "vld1.32 {d4-d7}, [%[ptr_din]]!      @load data \n"

            "vst1.32  {d8-d9},   [%[doutc2r0]]!  @ store result, add pointer\n"
            "vst1.32  {d10-d11}, [%[doutc6r0]]!  @ store result, add pointer\n"
            "vst1.32  {d12-d13}, [%[doutc3r0]]!  @ store result, add pointer\n"
            "vst1.32  {d14-d15}, [%[doutc7r0]]!  @ store result, add pointer\n"

            "vld1.32 {d8-d11},  [%[ptr_din]]!    @load data \n"
            "vld1.32 {d12-d15}, [%[ptr_din]]!    @load data \n"

            "bne    1b                           @ jump to main loop\n"

            : [doutc0r0] "+r"(doutc0_ptr), [doutc1r0] "+r"(doutc1_ptr),
              [doutc2r0] "+r"(doutc2_ptr), [doutc3r0] "+r"(doutc3_ptr),
              [doutc4r0] "+r"(doutc4_ptr), [doutc5r0] "+r"(doutc5_ptr),
              [doutc6r0] "+r"(doutc6_ptr), [doutc7r0] "+r"(doutc7_ptr),
              [ptr_din] "+r"(din_hei_ptr), [cnt] "+r"(cnt_loop)
            :
            : "q0", "q1", "q2", "q3", "q4", "q15");
#endif
      }
      if (we > width) {
        int offset = 32 * (valid_w / 4 - 1);
        din_hei_ptr = ptr_din + offset;
        int i = we - 4;
        for (; i < width; ++i) {
          *(doutc0_ptr++) = AKMAX(din_hei_ptr[0], 0);
          *(doutc1_ptr++) = AKMAX(din_hei_ptr[1], 0);
          *(doutc2_ptr++) = AKMAX(din_hei_ptr[2], 0);
          *(doutc3_ptr++) = AKMAX(din_hei_ptr[3], 0);
          *(doutc4_ptr++) = AKMAX(din_hei_ptr[4], 0);
          *(doutc5_ptr++) = AKMAX(din_hei_ptr[5], 0);
          *(doutc6_ptr++) = AKMAX(din_hei_ptr[6], 0);
          *(doutc7_ptr++) = AKMAX(din_hei_ptr[7], 0);
          din_hei_ptr += 8;
        }
      }
    }
  } else {
    for (int i = 0; i < size_h; i++) {
      int size_w = i * width;
      int* doutc0_ptr = doutc0r0 + size_w;  // doutc0r0 + width;
      int* doutc1_ptr = doutc1r0 + size_w;
      int* doutc2_ptr = doutc2r0 + size_w;
      int* doutc3_ptr = doutc3r0 + size_w;
      int* doutc4_ptr = doutc4r0 + size_w;
      int* doutc5_ptr = doutc5r0 + size_w;
      int* doutc6_ptr = doutc6r0 + size_w;
      int* doutc7_ptr = doutc7r0 + size_w;
      if (ce > channel) {
        switch (ce - channel) {
          case 7:
            doutc1_ptr = trash_ptr;
          case 6:
            doutc2_ptr = trash_ptr;
          case 5:
            doutc3_ptr = trash_ptr;
          case 4:
            doutc4_ptr = trash_ptr;
          case 3:
            doutc5_ptr = trash_ptr;
          case 2:
            doutc6_ptr = trash_ptr;
          case 1:
            doutc7_ptr = trash_ptr;
          default:
            break;
        }
      }
      ptr_din = din + i * valid_w * ch_n;
      const int* din_hei_ptr = ptr_din;
      if (cnt > 0) {
        int cnt_loop = cnt;
#ifdef __aarch64__
        asm volatile(
            "ldp q0, q1, [%[ptr_din]], #32  \n" /* load r00, r01 to q0, q1 */
            "ldp q2, q3, [%[ptr_din]], #32  \n" /* load r02, r03 to q2, q3 */
            "ldp q4, q5, [%[ptr_din]], #32  \n" /* load r00, r01 to q0, q1 */
            "ldp q6, q7, [%[ptr_din]], #32  \n" /* load r02, r03 to q2, q3 */
            "1:                             \n" /* main loop*/
            "trn1   v8.4s, v0.4s, v2.4s     \n" /* trans q0, q1*/
            "trn2   v9.4s, v0.4s, v2.4s     \n" /* trans q0, q1*/
            "trn1   v10.4s, v1.4s, v3.4s    \n" /* trans q2, q3*/
            "trn2   v11.4s, v1.4s, v3.4s    \n" /* trans q2, q3*/
            "ldp q0, q1, [%[ptr_din]], #32  \n" /* load r00, r01 to q0, q1 */

            "trn1   v12.4s, v4.4s, v6.4s    \n" /* trans q0, q1*/
            "trn2   v13.4s, v4.4s, v6.4s    \n" /* trans q0, q1*/
            "trn1   v14.4s, v5.4s, v7.4s    \n" /* trans q2, q3*/
            "trn2   v15.4s, v5.4s, v7.4s    \n" /* trans q2, q3*/
            "ldp q2, q3, [%[ptr_din]], #32  \n" /* load r02, r03 to q2, q3 */

            "trn1   v16.2d, v8.2d, v12.2d   \n" /* trans q8, q10 00 01 02 03*/
            "trn2   v17.2d, v8.2d, v12.2d   \n" /* trans q8, q10 20 21 22 23*/
            "trn1   v18.2d, v9.2d, v13.2d   \n" /* trans q9, q11 10 11 12 13*/
            "trn2   v19.2d, v9.2d, v13.2d   \n" /* trans q9, q11 30 31 32 33*/
            "ldp q4, q5, [%[ptr_din]], #32  \n" /* load r00, r01 to q0, q1 */

            "trn1   v8.2d, v10.2d, v14.2d   \n" /* trans q8, q10 40 41 42 43*/
            "trn2   v9.2d, v10.2d, v14.2d   \n" /* trans q8, q10 60 61 62 63*/
            "trn1   v12.2d, v11.2d, v15.2d  \n" /* trans q9, q11 50 51 52 53*/
            "trn2   v13.2d, v11.2d, v15.2d  \n" /* trans q9, q11 70 71 72 73*/
            "ldp q6, q7, [%[ptr_din]], #32  \n" /* load r02, r03 to q2, q3 */

            "str    q16, [%[doutc0r0]], #16 \n" /* store c0r0*/
            "str    q17, [%[doutc2r0]], #16 \n" /* store c2r0*/
            "str    q18, [%[doutc1r0]], #16 \n" /* store c1r0*/
            "str    q19, [%[doutc3r0]], #16 \n" /* store c3r0*/

            "subs   %w[cnt], %w[cnt], #1    \n" /* loop count -1*/
            "str    q8,  [%[doutc4r0]], #16 \n" /* store c0r0*/
            "str    q9,  [%[doutc6r0]], #16 \n" /* store c2r0*/
            "str    q12, [%[doutc5r0]], #16 \n" /* store c1r0*/
            "str    q13, [%[doutc7r0]], #16 \n" /* store c3r0*/

            "bne    1b                      \n" /* jump to main loop*/

            : [doutc0r0] "+r"(doutc0_ptr), [doutc1r0] "+r"(doutc1_ptr),
              [doutc2r0] "+r"(doutc2_ptr), [doutc3r0] "+r"(doutc3_ptr),
              [doutc4r0] "+r"(doutc4_ptr), [doutc5r0] "+r"(doutc5_ptr),
              [doutc6r0] "+r"(doutc6_ptr), [doutc7r0] "+r"(doutc7_ptr),
              [cnt] "+r"(cnt_loop), [ptr_din] "+r"(din_hei_ptr)
            :
            : "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10",
              "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19",
              "v20");
#else
        asm volatile(
            "vld1.32 {d0-d3},   [%[ptr_din]]!    @load data \n"
            "vld1.32 {d4-d7},   [%[ptr_din]]!    @load data \n"
            "vld1.32 {d8-d11},  [%[ptr_din]]!    @load data \n"
            "vld1.32 {d12-d15}, [%[ptr_din]]!    @load data \n"
            "1:                                  @ main loop\n"
            "vtrn.32    q0, q2                   @ trans q0, q2 \n"
            "vtrn.32    q4, q6                   @ trans q4, q6 \n"
            "vswp.32    d1, d8                   @ swap  d1, d8 \n"
            "vswp.32    d5, d12                  @ swap  d5, d12\n"

            "vtrn.32    q1, q3                   @ trans q1, q3 \n"
            "vtrn.32    q5, q7                   @ trans q5, q7 \n"
            "vswp.32    d3, d10                  @ swap  d3, d10\n"
            "vswp.32    d7, d14                  @ swap  d7, d14\n"

            "subs   %[cnt], %[cnt], #1           @ loop count - 1\n"

            "vst1.32  {d0-d1}, [%[doutc0r0]]!    @ store result, add pointer\n"
            "vst1.32  {d2-d3}, [%[doutc4r0]]!    @ store result, add pointer\n"
            "vst1.32  {d4-d5}, [%[doutc1r0]]!    @ store result, add pointer\n"
            "vst1.32  {d6-d7}, [%[doutc5r0]]!    @ store result, add pointer\n"

            "vld1.32  {d0-d3}, [%[ptr_din]]!     @load data \n"
            "vld1.32  {d4-d7}, [%[ptr_din]]!     @load data \n"

            "vst1.32  {d8-d9},   [%[doutc2r0]]!  @ store result, add pointer\n"
            "vst1.32  {d10-d11}, [%[doutc6r0]]!  @ store result, add pointer\n"
            "vst1.32  {d12-d13}, [%[doutc3r0]]!  @ store result, add pointer\n"
            "vst1.32  {d14-d15}, [%[doutc7r0]]!  @ store result, add pointer\n"

            "vld1.32  {d8-d11},  [%[ptr_din]]!   @load data \n"
            "vld1.32  {d12-d15}, [%[ptr_din]]!   @load data \n"

            "bne    1b                           @ jump to main loop\n"

            : [doutc0r0] "+r"(doutc0_ptr), [doutc1r0] "+r"(doutc1_ptr),
              [doutc2r0] "+r"(doutc2_ptr), [doutc3r0] "+r"(doutc3_ptr),
              [doutc4r0] "+r"(doutc4_ptr), [doutc5r0] "+r"(doutc5_ptr),
              [doutc6r0] "+r"(doutc6_ptr), [doutc7r0] "+r"(doutc7_ptr),
              [ptr_din] "+r"(din_hei_ptr), [cnt] "+r"(cnt_loop)
            :
            : "q0", "q1", "q2", "q3", "q4", "q15");
#endif
      }
      if (we > width) {
        int offset = 32 * (valid_w / 4 - 1);
        din_hei_ptr = ptr_din + offset;
        int i = we - 4;
        for (; i < width; ++i) {
          *(doutc0_ptr++) = din_hei_ptr[0];
          *(doutc1_ptr++) = din_hei_ptr[1];
          *(doutc2_ptr++) = din_hei_ptr[2];
          *(doutc3_ptr++) = din_hei_ptr[3];
          *(doutc4_ptr++) = din_hei_ptr[4];
          *(doutc5_ptr++) = din_hei_ptr[5];
          *(doutc6_ptr++) = din_hei_ptr[6];
          *(doutc7_ptr++) = din_hei_ptr[7];
          din_hei_ptr += 8;
        }
      }
    }
  }
  return true;
}

/*wirte result in outputs--int8, fp32
* input din: [n, c / 8, h, w * 8], output dout: [n, c, h, w]
*/
template <typename dtype>
static bool write_to_output_c8_int32_1(const int* din, dtype* dout, int ch_n,
                                       int hei_n, int cs, int ce, int hs,
                                       int he, int ws, int we, int channel,
                                       int height, int width, bool flag_relu,
                                       dtype* trash_ptr, const float* scale,
                                       PrecisionType out_dtype) {
  if (ch_n != 8 || hei_n <= 0) {
    LOG(ERROR) << "ch_n must be equal 8 and hei_n is more than zero";
    return false;
  }
  int size_c_out = width * height;

  dtype* doutc0r0 = dout + cs * size_c_out + hs * width + ws;
  dtype* doutc1r0 = doutc0r0 + size_c_out;
  dtype* doutc2r0 = doutc1r0 + size_c_out;
  dtype* doutc3r0 = doutc2r0 + size_c_out;
  dtype* doutc4r0 = doutc3r0 + size_c_out;
  dtype* doutc5r0 = doutc4r0 + size_c_out;
  dtype* doutc6r0 = doutc5r0 + size_c_out;
  dtype* doutc7r0 = doutc6r0 + size_c_out;

  const int* ptr_din = din;

  int size_h = (he > height ? height : he) - hs;  // size_h == hei_n

  int valid_w = we - ws;
  int cnt = valid_w / 4;

  float32x4_t w_scale0 = vld1q_f32(scale);
  float32x4_t w_scale1 = vld1q_f32(scale + 4);

  float32x4_t vzero = vdupq_n_f32(0.f);

  if (we > width) {
    cnt--;
  }
  if (out_dtype == PRECISION(kFloat)) {
    if (flag_relu) {
      for (int i = 0; i < size_h; i++) {
        int size_w = i * width;
        dtype* doutc0_ptr = doutc0r0 + size_w;  // doutc0r0 + width;
        dtype* doutc1_ptr = doutc1r0 + size_w;
        dtype* doutc2_ptr = doutc2r0 + size_w;
        dtype* doutc3_ptr = doutc3r0 + size_w;
        dtype* doutc4_ptr = doutc4r0 + size_w;
        dtype* doutc5_ptr = doutc5r0 + size_w;
        dtype* doutc6_ptr = doutc6r0 + size_w;
        dtype* doutc7_ptr = doutc7r0 + size_w;
        if (ce > channel) {
          switch (ce - channel) {
            case 7:
              doutc1_ptr = trash_ptr;
            case 6:
              doutc2_ptr = trash_ptr;
            case 5:
              doutc3_ptr = trash_ptr;
            case 4:
              doutc4_ptr = trash_ptr;
            case 3:
              doutc5_ptr = trash_ptr;
            case 2:
              doutc6_ptr = trash_ptr;
            case 1:
              doutc7_ptr = trash_ptr;
            default:
              break;
          }
        }
        ptr_din = din + i * valid_w * ch_n;
        const int* din_hei_ptr = ptr_din;
        if (cnt > 0) {
          int cnt_loop = cnt;
#ifdef __aarch64__
          asm volatile(
              "ldp q0, q1, [%[ptr_din]], #32  \n" /* load r00, r01 to q0, q1 */
              "ldp q2, q3, [%[ptr_din]], #32  \n" /* load r02, r03 to q2, q3 */
              "ldp q4, q5, [%[ptr_din]], #32  \n" /* load r00, r01 to q0, q1 */
              "ldp q6, q7, [%[ptr_din]], #32  \n" /* load r02, r03 to q2, q3 */
              "movi v20.4s, #0                \n" /* for relu */
              "1:                             \n" /* main loop*/
              "trn1   v8.4s, v0.4s, v2.4s     \n" /* trans q0, q1*/
              "trn2   v9.4s, v0.4s, v2.4s     \n" /* trans q0, q1*/
              "trn1   v10.4s, v1.4s, v3.4s    \n" /* trans q2, q3*/
              "trn2   v11.4s, v1.4s, v3.4s    \n" /* trans q2, q3*/
              "ldp q0, q1, [%[ptr_din]], #32  \n" /* load r00, r01 to q0, q1 */

              "trn1   v12.4s, v4.4s, v6.4s    \n" /* trans q0, q1*/
              "trn2   v13.4s, v4.4s, v6.4s    \n" /* trans q0, q1*/
              "trn1   v14.4s, v5.4s, v7.4s    \n" /* trans q2, q3*/
              "trn2   v15.4s, v5.4s, v7.4s    \n" /* trans q2, q3*/
              "ldp q2, q3, [%[ptr_din]], #32  \n" /* load r02, r03 to q2, q3 */

              "trn1   v16.2d, v8.2d, v12.2d   \n" /* trans q8, q10 00 01 02 03*/
              "trn2   v17.2d, v8.2d, v12.2d   \n" /* trans q8, q10 20 21 22 23*/
              "trn1   v18.2d, v9.2d, v13.2d   \n" /* trans q9, q11 10 11 12 13*/
              "trn2   v19.2d, v9.2d, v13.2d   \n" /* trans q9, q11 30 31 32 33*/
              "ldp q4, q5, [%[ptr_din]], #32  \n" /* load r00, r01 to q0, q1 */

              "trn1   v8.2d, v10.2d, v14.2d   \n" /* trans q8, q10 40 41 42 43*/
              "trn2   v9.2d, v10.2d, v14.2d   \n" /* trans q8, q10 60 61 62 63*/
              "trn1   v12.2d, v11.2d, v15.2d  \n" /* trans q9, q11 50 51 52 53*/
              "trn2   v13.2d, v11.2d, v15.2d  \n" /* trans q9, q11 70 71 72 73*/
              "ldp q6, q7, [%[ptr_din]], #32  \n" /* load r02, r03 to q2, q3 */

              "smax   v16.4s, v16.4s, v20.4s  \n" /*relu*/
              "smax   v17.4s, v17.4s, v20.4s  \n" /*relu*/
              "smax   v18.4s, v18.4s, v20.4s  \n" /*relu*/
              "smax   v19.4s, v19.4s, v20.4s  \n" /*relu*/

              "smax   v8.4s, v8.4s, v20.4s    \n" /*relu*/
              "smax   v9.4s, v9.4s, v20.4s    \n" /*relu*/
              "smax   v12.4s, v12.4s, v20.4s  \n" /*relu*/
              "smax   v13.4s, v13.4s, v20.4s  \n" /*relu*/

              // int32->fp32
              "scvtf   v10.4s, v16.4s               \n"
              "scvtf   v11.4s, v17.4s               \n"
              "scvtf   v14.4s, v18.4s               \n"
              "scvtf   v15.4s, v19.4s               \n"
              // mul
              "fmul    v16.4s, v10.4s, %[scale0].s[0]  \n"
              "fmul    v17.4s, v11.4s, %[scale0].s[2] \n"
              "fmul    v18.4s, v14.4s, %[scale0].s[1] \n"
              "fmul    v19.4s, v15.4s, %[scale0].s[3] \n"

              "scvtf   v10.4s, v8.4s               \n"
              "scvtf   v11.4s, v9.4s               \n"
              "scvtf   v14.4s, v12.4s               \n"
              "scvtf   v15.4s, v13.4s               \n"

              "str    q16, [%[doutc0r0]], #16 \n" /* store c0r0*/
              "str    q17, [%[doutc2r0]], #16 \n" /* store c2r0*/
              "str    q18, [%[doutc1r0]], #16 \n" /* store c1r0*/
              "str    q19, [%[doutc3r0]], #16 \n" /* store c3r0*/

              // mul
              "fmul    v8.4s, v10.4s, %[scale1].s[0]  \n"
              "fmul    v9.4s, v11.4s, %[scale1].s[2] \n"
              "fmul    v12.4s, v14.4s, %[scale1].s[1] \n"
              "fmul    v13.4s, v15.4s, %[scale1].s[3] \n"

              "subs   %w[cnt], %w[cnt],  #1   \n" /* loop count -1*/
              "str    q8, [%[doutc4r0]], #16  \n" /* store c0r0*/
              "str    q9, [%[doutc6r0]], #16  \n" /* store c2r0*/
              "str    q12, [%[doutc5r0]], #16 \n" /* store c1r0*/
              "str    q13, [%[doutc7r0]], #16 \n" /* store c3r0*/

              "bne    1b                      \n" /* jump to main loop*/

              : [doutc0r0] "+r"(doutc0_ptr), [doutc1r0] "+r"(doutc1_ptr),
                [doutc2r0] "+r"(doutc2_ptr), [doutc3r0] "+r"(doutc3_ptr),
                [doutc4r0] "+r"(doutc4_ptr), [doutc5r0] "+r"(doutc5_ptr),
                [doutc6r0] "+r"(doutc6_ptr), [doutc7r0] "+r"(doutc7_ptr),
                [cnt] "+r"(cnt_loop), [ptr_din] "+r"(din_hei_ptr)
              : [scale0] "w"(w_scale0), [scale1] "w"(w_scale1)
              : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9",
                "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18",
                "v19", "v20");
#else
          asm volatile(
              "vld1.32 {d0-d3},   [%[ptr_din]]!   @load data \n"
              "vld1.32 {d4-d7},   [%[ptr_din]]!   @load data \n"
              "vld1.32 {d8-d11},  [%[ptr_din]]!   @load data \n"
              "vld1.32 {d12-d15}, [%[ptr_din]]!   @load data \n"
              "vmov.s32   q15, #0                 @ dump zero\n"
              "1:                                 @ main loop\n"
              "vmax.s32   q0, q0, q15             @ relu\n"
              "vmax.s32   q1, q1, q15             @ relu\n"
              "vmax.s32   q2, q2, q15             @ relu\n"
              "vmax.s32   q3, q3, q15             @ relu\n"

              "vmax.s32   q4, q4, q15             @ relu\n"
              "vmax.s32   q5, q5, q15             @ relu\n"
              "vmax.s32   q6, q6, q15             @ relu\n"
              "vmax.s32   q7, q7, q15             @ relu\n"

              // int32-> fp32
              "vcvt.f32.s32   q8, q0                  \n"
              "vcvt.f32.s32   q9, q1                  \n"
              "vcvt.f32.s32   q10, q2                  \n"
              "vcvt.f32.s32   q11, q3                  \n"

              // mul
              "vmul.f32  q0, q8, %q[scale0]       \n"
              "vmul.f32  q1, q9, %q[scale1]       \n"
              "vmul.f32  q2, q10, %q[scale0]      \n"
              "vmul.f32  q3, q11, %q[scale1]      \n"

              // int32-> fp32
              "vcvt.f32.s32   q8, q4                  \n"
              "vcvt.f32.s32   q9, q5                  \n"
              "vcvt.f32.s32   q10, q6                  \n"
              "vcvt.f32.s32   q11, q7                  \n"

              // mul
              "vmul.f32  q4, q8, %q[scale0]       \n"
              "vmul.f32  q5, q9, %q[scale1]        \n"
              "vmul.f32  q6, q10, %q[scale0]      \n"
              "vmul.f32  q7, q11, %q[scale1]      \n"

              "vtrn.32    q0, q2                  @ trans q0, q2 \n"
              "vtrn.32    q4, q6                  @ trans q4, q6 \n"
              "vswp.32    d1, d8                  @ swap  d1, d8 \n"
              "vswp.32    d5, d12                 @ swap  d5, d12\n"

              "vtrn.32    q1, q3                  @ trans q1, q3 \n"
              "vtrn.32    q5, q7                  @ trans q5, q7 \n"
              "vswp.32    d3, d10                 @ swap  d3, d10\n"
              "vswp.32    d7, d14                 @ swap  d7, d14\n"

              "vst1.32  {d0-d1}, [%[doutc0r0]]!   @ store result, add pointer\n"
              "vst1.32  {d4-d5}, [%[doutc1r0]]!   @ store result, add pointer\n"
              "vst1.32  {d8-d9},   [%[doutc2r0]]!  @ store result, add "
              "pointer\n"
              "vst1.32  {d12-d13}, [%[doutc3r0]]!  @ store result, add "
              "pointer\n"

              "vst1.32  {d2-d3}, [%[doutc4r0]]!   @ store result, add pointer\n"
              "vst1.32  {d6-d7}, [%[doutc5r0]]!   @ store result, add pointer\n"
              "vst1.32  {d10-d11}, [%[doutc6r0]]!  @ store result, add "
              "pointer\n"
              "vst1.32  {d14-d15}, [%[doutc7r0]]!  @ store result, add "
              "pointer\n"

              "subs   %[cnt], %[cnt], #1          @ loop count - 1\n"

              "vld1.32 {d0-d3}, [%[ptr_din]]!      @load data \n"
              "vld1.32 {d4-d7}, [%[ptr_din]]!      @load data \n"
              "vld1.32 {d8-d11},  [%[ptr_din]]!    @load data \n"
              "vld1.32 {d12-d15}, [%[ptr_din]]!    @load data \n"

              "bne    1b                           @ jump to main loop\n"

              : [doutc0r0] "+r"(doutc0_ptr), [doutc1r0] "+r"(doutc1_ptr),
                [doutc2r0] "+r"(doutc2_ptr), [doutc3r0] "+r"(doutc3_ptr),
                [doutc4r0] "+r"(doutc4_ptr), [doutc5r0] "+r"(doutc5_ptr),
                [doutc6r0] "+r"(doutc6_ptr), [doutc7r0] "+r"(doutc7_ptr),
                [ptr_din] "+r"(din_hei_ptr), [cnt] "+r"(cnt_loop)
              : [scale0] "w"(w_scale0), [scale1] "w"(w_scale1)
              : "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7", "q8", "q9",
                "q10", "q11", "q15");
#endif
        }
        if (we > width) {
          int offset = 32 * (valid_w / 4 - 1);
          din_hei_ptr = ptr_din + offset;
          int i = we - 4;
          for (; i < width; ++i) {
            *(doutc0_ptr++) = AKMAX(din_hei_ptr[0] * scale[0], 0);
            *(doutc1_ptr++) = AKMAX(din_hei_ptr[1] * scale[1], 0);
            *(doutc2_ptr++) = AKMAX(din_hei_ptr[2] * scale[2], 0);
            *(doutc3_ptr++) = AKMAX(din_hei_ptr[3] * scale[3], 0);
            *(doutc4_ptr++) = AKMAX(din_hei_ptr[4] * scale[4], 0);
            *(doutc5_ptr++) = AKMAX(din_hei_ptr[5] * scale[5], 0);
            *(doutc6_ptr++) = AKMAX(din_hei_ptr[6] * scale[6], 0);
            *(doutc7_ptr++) = AKMAX(din_hei_ptr[7] * scale[7], 0);
            din_hei_ptr += 8;
          }
        }
      }
    } else {
      for (int i = 0; i < size_h; i++) {
        int size_w = i * width;
        dtype* doutc0_ptr = doutc0r0 + size_w;  // doutc0r0 + width;
        dtype* doutc1_ptr = doutc1r0 + size_w;
        dtype* doutc2_ptr = doutc2r0 + size_w;
        dtype* doutc3_ptr = doutc3r0 + size_w;
        dtype* doutc4_ptr = doutc4r0 + size_w;
        dtype* doutc5_ptr = doutc5r0 + size_w;
        dtype* doutc6_ptr = doutc6r0 + size_w;
        dtype* doutc7_ptr = doutc7r0 + size_w;
        if (ce > channel) {
          switch (ce - channel) {
            case 7:
              doutc1_ptr = trash_ptr;
            case 6:
              doutc2_ptr = trash_ptr;
            case 5:
              doutc3_ptr = trash_ptr;
            case 4:
              doutc4_ptr = trash_ptr;
            case 3:
              doutc5_ptr = trash_ptr;
            case 2:
              doutc6_ptr = trash_ptr;
            case 1:
              doutc7_ptr = trash_ptr;
            default:
              break;
          }
        }
        ptr_din = din + i * valid_w * ch_n;
        const int* din_hei_ptr = ptr_din;
        if (cnt > 0) {
          int cnt_loop = cnt;
#ifdef __aarch64__
          asm volatile(
              "ldp q0, q1, [%[ptr_din]], #32  \n" /* load r00, r01 to q0, q1 */
              "ldp q2, q3, [%[ptr_din]], #32  \n" /* load r02, r03 to q2, q3 */
              "ldp q4, q5, [%[ptr_din]], #32  \n" /* load r00, r01 to q0, q1 */
              "ldp q6, q7, [%[ptr_din]], #32  \n" /* load r02, r03 to q2, q3 */
              "movi v20.4s, #0                \n" /* for relu */
              "1:                             \n" /* main loop*/
              "trn1   v8.4s, v0.4s, v2.4s     \n" /* trans q0, q1*/
              "trn2   v9.4s, v0.4s, v2.4s     \n" /* trans q0, q1*/
              "trn1   v10.4s, v1.4s, v3.4s    \n" /* trans q2, q3*/
              "trn2   v11.4s, v1.4s, v3.4s    \n" /* trans q2, q3*/
              "ldp q0, q1, [%[ptr_din]], #32  \n" /* load r00, r01 to q0, q1 */

              "trn1   v12.4s, v4.4s, v6.4s    \n" /* trans q0, q1*/
              "trn2   v13.4s, v4.4s, v6.4s    \n" /* trans q0, q1*/
              "trn1   v14.4s, v5.4s, v7.4s    \n" /* trans q2, q3*/
              "trn2   v15.4s, v5.4s, v7.4s    \n" /* trans q2, q3*/
              "ldp q2, q3, [%[ptr_din]], #32  \n" /* load r02, r03 to q2, q3 */

              "trn1   v16.2d, v8.2d, v12.2d   \n" /* trans q8, q10 00 01 02 03*/
              "trn2   v17.2d, v8.2d, v12.2d   \n" /* trans q8, q10 20 21 22 23*/
              "trn1   v18.2d, v9.2d, v13.2d   \n" /* trans q9, q11 10 11 12 13*/
              "trn2   v19.2d, v9.2d, v13.2d   \n" /* trans q9, q11 30 31 32 33*/
              "ldp q4, q5, [%[ptr_din]], #32  \n" /* load r00, r01 to q0, q1 */

              "trn1   v8.2d, v10.2d, v14.2d   \n" /* trans q8, q10 40 41 42 43*/
              "trn2   v9.2d, v10.2d, v14.2d   \n" /* trans q8, q10 60 61 62 63*/
              "trn1   v12.2d, v11.2d, v15.2d  \n" /* trans q9, q11 50 51 52 53*/
              "trn2   v13.2d, v11.2d, v15.2d  \n" /* trans q9, q11 70 71 72 73*/
              "ldp q6, q7, [%[ptr_din]], #32  \n" /* load r02, r03 to q2, q3 */

              // int32->fp32
              "scvtf   v10.4s, v16.4s               \n"
              "scvtf   v11.4s, v17.4s               \n"
              "scvtf   v14.4s, v18.4s               \n"
              "scvtf   v15.4s, v19.4s               \n"
              // mul
              "fmul    v16.4s, v10.4s, %[scale0].s[0]  \n"
              "fmul    v17.4s, v11.4s, %[scale0].s[2] \n"
              "fmul    v18.4s, v14.4s, %[scale0].s[1] \n"
              "fmul    v19.4s, v15.4s, %[scale0].s[3] \n"

              "scvtf   v10.4s, v8.4s               \n"
              "scvtf   v11.4s, v9.4s               \n"
              "scvtf   v14.4s, v12.4s               \n"
              "scvtf   v15.4s, v13.4s               \n"

              "str    q16, [%[doutc0r0]], #16 \n" /* store c0r0*/
              "str    q17, [%[doutc2r0]], #16 \n" /* store c2r0*/
              "str    q18, [%[doutc1r0]], #16 \n" /* store c1r0*/
              "str    q19, [%[doutc3r0]], #16 \n" /* store c3r0*/

              // mul
              "fmul    v8.4s, v10.4s, %[scale1].s[0]  \n"
              "fmul    v9.4s, v11.4s, %[scale1].s[2] \n"
              "fmul    v12.4s, v14.4s, %[scale1].s[1] \n"
              "fmul    v13.4s, v15.4s, %[scale1].s[3] \n"

              "subs   %w[cnt], %w[cnt],  #1   \n" /* loop count -1*/
              "str    q8, [%[doutc4r0]], #16  \n" /* store c0r0*/
              "str    q9, [%[doutc6r0]], #16  \n" /* store c2r0*/
              "str    q12, [%[doutc5r0]], #16 \n" /* store c1r0*/
              "str    q13, [%[doutc7r0]], #16 \n" /* store c3r0*/

              "bne    1b                      \n" /* jump to main loop*/

              : [doutc0r0] "+r"(doutc0_ptr), [doutc1r0] "+r"(doutc1_ptr),
                [doutc2r0] "+r"(doutc2_ptr), [doutc3r0] "+r"(doutc3_ptr),
                [doutc4r0] "+r"(doutc4_ptr), [doutc5r0] "+r"(doutc5_ptr),
                [doutc6r0] "+r"(doutc6_ptr), [doutc7r0] "+r"(doutc7_ptr),
                [cnt] "+r"(cnt_loop), [ptr_din] "+r"(din_hei_ptr)
              : [scale0] "w"(w_scale0), [scale1] "w"(w_scale1)
              : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9",
                "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18",
                "v19", "v20");
#else
          asm volatile(
              "vld1.32 {d0-d3},   [%[ptr_din]]!   @load data \n"
              "vld1.32 {d4-d7},   [%[ptr_din]]!   @load data \n"
              "vld1.32 {d8-d11},  [%[ptr_din]]!   @load data \n"
              "vld1.32 {d12-d15}, [%[ptr_din]]!   @load data \n"
              "vmov.s32   q15, #0                 @ dump zero\n"
              "1:                                 @ main loop\n"
              // int32-> fp32
              "vcvt.f32.s32   q8, q0                  \n"
              "vcvt.f32.s32   q9, q1                  \n"
              "vcvt.f32.s32   q10, q2                  \n"
              "vcvt.f32.s32   q11, q3                  \n"

              // mul
              "vmul.f32  q0, q8, %q[scale0]       \n"
              "vmul.f32  q1, q9, %q[scale1]       \n"
              "vmul.f32  q2, q10, %q[scale0]      \n"
              "vmul.f32  q3, q11, %q[scale1]      \n"

              // int32-> fp32
              "vcvt.f32.s32   q8, q4                  \n"
              "vcvt.f32.s32   q9, q5                  \n"
              "vcvt.f32.s32   q10, q6                  \n"
              "vcvt.f32.s32   q11, q7                  \n"

              // mul
              "vmul.f32  q4, q8, %q[scale0]       \n"
              "vmul.f32  q5, q9, %q[scale1]        \n"
              "vmul.f32  q6, q10, %q[scale0]      \n"
              "vmul.f32  q7, q11, %q[scale1]      \n"

              "vtrn.32    q0, q2                  @ trans q0, q2 \n"
              "vtrn.32    q4, q6                  @ trans q4, q6 \n"
              "vswp.32    d1, d8                  @ swap  d1, d8 \n"
              "vswp.32    d5, d12                 @ swap  d5, d12\n"

              "vtrn.32    q1, q3                  @ trans q1, q3 \n"
              "vtrn.32    q5, q7                  @ trans q5, q7 \n"
              "vswp.32    d3, d10                 @ swap  d3, d10\n"
              "vswp.32    d7, d14                 @ swap  d7, d14\n"

              "vst1.32  {d0-d1}, [%[doutc0r0]]!   @ store result, add pointer\n"
              "vst1.32  {d4-d5}, [%[doutc1r0]]!   @ store result, add pointer\n"
              "vst1.32  {d8-d9},   [%[doutc2r0]]!  @ store result, add "
              "pointer\n"
              "vst1.32  {d12-d13}, [%[doutc3r0]]!  @ store result, add "
              "pointer\n"

              "vst1.32  {d2-d3}, [%[doutc4r0]]!   @ store result, add pointer\n"
              "vst1.32  {d6-d7}, [%[doutc5r0]]!   @ store result, add pointer\n"
              "vst1.32  {d10-d11}, [%[doutc6r0]]!  @ store result, add "
              "pointer\n"
              "vst1.32  {d14-d15}, [%[doutc7r0]]!  @ store result, add "
              "pointer\n"

              "subs   %[cnt], %[cnt], #1          @ loop count - 1\n"

              "vld1.32 {d0-d3}, [%[ptr_din]]!      @load data \n"
              "vld1.32 {d4-d7}, [%[ptr_din]]!      @load data \n"
              "vld1.32 {d8-d11},  [%[ptr_din]]!    @load data \n"
              "vld1.32 {d12-d15}, [%[ptr_din]]!    @load data \n"

              "bne    1b                           @ jump to main loop\n"

              : [doutc0r0] "+r"(doutc0_ptr), [doutc1r0] "+r"(doutc1_ptr),
                [doutc2r0] "+r"(doutc2_ptr), [doutc3r0] "+r"(doutc3_ptr),
                [doutc4r0] "+r"(doutc4_ptr), [doutc5r0] "+r"(doutc5_ptr),
                [doutc6r0] "+r"(doutc6_ptr), [doutc7r0] "+r"(doutc7_ptr),
                [ptr_din] "+r"(din_hei_ptr), [cnt] "+r"(cnt_loop)
              : [scale0] "w"(w_scale0), [scale1] "w"(w_scale1)
              : "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7", "q8", "q9",
                "q10", "q11", "q15");
#endif
        }
        if (we > width) {
          int offset = 32 * (valid_w / 4 - 1);
          din_hei_ptr = ptr_din + offset;
          int i = we - 4;
          for (; i < width; ++i) {
            *(doutc0_ptr++) = din_hei_ptr[0] * scale[0];
            *(doutc1_ptr++) = din_hei_ptr[1] * scale[1];
            *(doutc2_ptr++) = din_hei_ptr[2] * scale[2];
            *(doutc3_ptr++) = din_hei_ptr[3] * scale[3];
            *(doutc4_ptr++) = din_hei_ptr[4] * scale[4];
            *(doutc5_ptr++) = din_hei_ptr[5] * scale[5];
            *(doutc6_ptr++) = din_hei_ptr[6] * scale[6];
            *(doutc7_ptr++) = din_hei_ptr[7] * scale[7];
            din_hei_ptr += 8;
          }
        }
      }
    }
  } else if (out_dtype == PRECISION(kInt8)) {
    // int32_to_int8
    float32x4_t vpoff = vdupq_n_f32(0.5f);
    float32x4_t vnoff = vdupq_n_f32(-0.5f);
    if (flag_relu) {
      for (int i = 0; i < size_h; i++) {
        int size_w = i * width;
        dtype* doutc0_ptr = doutc0r0 + size_w;  // doutc0r0 + width;
        dtype* doutc1_ptr = doutc1r0 + size_w;
        dtype* doutc2_ptr = doutc2r0 + size_w;
        dtype* doutc3_ptr = doutc3r0 + size_w;
        dtype* doutc4_ptr = doutc4r0 + size_w;
        dtype* doutc5_ptr = doutc5r0 + size_w;
        dtype* doutc6_ptr = doutc6r0 + size_w;
        dtype* doutc7_ptr = doutc7r0 + size_w;
        if (ce > channel) {
          switch (ce - channel) {
            case 7:
              doutc1_ptr = trash_ptr;
            case 6:
              doutc2_ptr = trash_ptr;
            case 5:
              doutc3_ptr = trash_ptr;
            case 4:
              doutc4_ptr = trash_ptr;
            case 3:
              doutc5_ptr = trash_ptr;
            case 2:
              doutc6_ptr = trash_ptr;
            case 1:
              doutc7_ptr = trash_ptr;
            default:
              break;
          }
        }
        ptr_din = din + i * valid_w * ch_n;
        const int* din_hei_ptr = ptr_din;
        if (cnt > 0) {
          int cnt_loop = cnt;
#ifdef __aarch64__
          asm volatile(
              "ldp q0, q1, [%[ptr_din]], #32  \n" /* load r00, r01 to q0, q1 */
              "ldp q2, q3, [%[ptr_din]], #32  \n" /* load r02, r03 to q2, q3 */
              "ldp q4, q5, [%[ptr_din]], #32  \n" /* load r00, r01 to q0, q1 */
              "ldp q6, q7, [%[ptr_din]], #32  \n" /* load r02, r03 to q2, q3 */
              // "movi v20.4s, #0                \n"         /* for relu */
              "1:                             \n" /* main loop*/
              "trn1   v8.4s, v0.4s, v2.4s     \n" /* trans q0, q1*/
              "trn2   v9.4s, v0.4s, v2.4s     \n" /* trans q0, q1*/
              "trn1   v10.4s, v1.4s, v3.4s    \n" /* trans q2, q3*/
              "trn2   v11.4s, v1.4s, v3.4s    \n" /* trans q2, q3*/
              "ldp q0, q1, [%[ptr_din]], #32  \n" /* load r00, r01 to q0, q1 */

              "trn1   v12.4s, v4.4s, v6.4s    \n" /* trans q0, q1*/
              "trn2   v13.4s, v4.4s, v6.4s    \n" /* trans q0, q1*/
              "trn1   v14.4s, v5.4s, v7.4s    \n" /* trans q2, q3*/
              "trn2   v15.4s, v5.4s, v7.4s    \n" /* trans q2, q3*/
              "ldp q2, q3, [%[ptr_din]], #32  \n" /* load r02, r03 to q2, q3 */

              "trn1   v16.2d, v8.2d, v12.2d   \n" /* trans q8, q10 00 01 02 03*/
              "trn2   v17.2d, v8.2d, v12.2d   \n" /* trans q8, q10 20 21 22 23*/
              "trn1   v18.2d, v9.2d, v13.2d   \n" /* trans q9, q11 10 11 12 13*/
              "trn2   v19.2d, v9.2d, v13.2d   \n" /* trans q9, q11 30 31 32 33*/
              "ldp q4, q5, [%[ptr_din]], #32  \n" /* load r00, r01 to q0, q1 */

              "trn1   v8.2d, v10.2d, v14.2d   \n" /* trans q8, q10 40 41 42 43*/
              "trn2   v9.2d, v10.2d, v14.2d   \n" /* trans q8, q10 60 61 62 63*/
              "trn1   v12.2d, v11.2d, v15.2d  \n" /* trans q9, q11 50 51 52 53*/
              "trn2   v13.2d, v11.2d, v15.2d  \n" /* trans q9, q11 70 71 72 73*/
              "ldp q6, q7, [%[ptr_din]], #32  \n" /* load r02, r03 to q2, q3 */

              "smax   v16.4s, v16.4s, %[vzero].4s  \n" /*relu*/
              "smax   v17.4s, v17.4s, %[vzero].4s  \n" /*relu*/
              "smax   v18.4s, v18.4s, %[vzero].4s  \n" /*relu*/
              "smax   v19.4s, v19.4s, %[vzero].4s  \n" /*relu*/

              "smax   v8.4s, v8.4s, %[vzero].4s    \n" /*relu*/
              "smax   v9.4s, v9.4s, %[vzero].4s    \n" /*relu*/
              "smax   v12.4s, v12.4s, %[vzero].4s  \n" /*relu*/
              "smax   v13.4s, v13.4s, %[vzero].4s  \n" /*relu*/

              // int32 --> fp32
              "scvtf   v10.4s, v16.4s               \n"
              "scvtf   v11.4s, v17.4s               \n"
              "scvtf   v14.4s, v18.4s               \n"
              "scvtf   v15.4s, v19.4s               \n"

              "scvtf   v20.4s, v8.4s               \n"
              "scvtf   v21.4s, v9.4s               \n"
              "scvtf   v22.4s, v12.4s               \n"
              "scvtf   v23.4s, v13.4s               \n"

              // mul
              "fmul    v16.4s, v10.4s, %[scale0].s[0]  \n"
              "fmul    v17.4s, v11.4s, %[scale0].s[2]  \n"
              "fmul    v18.4s, v14.4s, %[scale0].s[1] \n"
              "fmul    v19.4s, v15.4s, %[scale0].s[3] \n"

              "fmul    v8.4s, v20.4s, %[scale1].s[0]  \n"
              "fmul    v9.4s, v21.4s, %[scale1].s[2]  \n"
              "fmul    v12.4s, v22.4s, %[scale1].s[1] \n"
              "fmul    v13.4s, v23.4s, %[scale1].s[3] \n"

              // fp32-int32
              "fcvtas  v10.4s, v16.4s                      \n"
              "fcvtas  v11.4s, v17.4s                      \n"
              "fcvtas  v14.4s, v18.4s                      \n"
              "fcvtas  v15.4s, v19.4s                      \n"

              "fcvtas  v20.4s, v8.4s                      \n"
              "fcvtas  v21.4s, v9.4s                      \n"
              "fcvtas  v22.4s, v12.4s                      \n"
              "fcvtas  v23.4s, v13.4s                      \n"

              // int32-int16
              "sqxtn   v16.4h, v10.4s                      \n"
              "sqxtn   v17.4h, v11.4s                      \n"
              "sqxtn   v18.4h, v14.4s                      \n"
              "sqxtn   v19.4h, v15.4s                      \n"

              "sqxtn   v8.4h, v20.4s                      \n"
              "sqxtn   v9.4h, v21.4s                      \n"
              "sqxtn   v12.4h, v22.4s                      \n"
              "sqxtn   v13.4h, v23.4s                      \n"

              // int16-int8
              "sqxtn  v10.8b, v16.8h                      \n"
              "sqxtn  v11.8b, v17.8h                     \n"
              "sqxtn  v14.8b, v18.8h                      \n"
              "sqxtn  v15.8b, v19.8h                     \n"

              "sqxtn  v20.8b, v8.8h                      \n"
              "sqxtn  v21.8b, v9.8h                     \n"
              "sqxtn  v22.8b, v12.8h                      \n"
              "sqxtn  v23.8b, v13.8h                     \n"

              "str    s10, [%[doutc0r0]], #4 \n" /* store c0r0*/
              "str    s11, [%[doutc2r0]], #4 \n" /* store c2r0*/
              "str    s14, [%[doutc1r0]], #4 \n" /* store c1r0*/
              "str    s15, [%[doutc3r0]], #4 \n" /* store c3r0*/

              "subs   %w[cnt], %w[cnt],  #1   \n" /* loop count -1*/
              "str    s20, [%[doutc4r0]], #4  \n" /* store c0r0*/
              "str    s21, [%[doutc6r0]], #4  \n" /* store c2r0*/
              "str    s22, [%[doutc5r0]], #4 \n"  /* store c1r0*/
              "str    s23, [%[doutc7r0]], #4 \n"  /* store c3r0*/

              "bne    1b                      \n" /* jump to main loop*/
              : [doutc0r0] "+r"(doutc0_ptr), [doutc1r0] "+r"(doutc1_ptr),
                [doutc2r0] "+r"(doutc2_ptr), [doutc3r0] "+r"(doutc3_ptr),
                [doutc4r0] "+r"(doutc4_ptr), [doutc5r0] "+r"(doutc5_ptr),
                [doutc6r0] "+r"(doutc6_ptr), [doutc7r0] "+r"(doutc7_ptr),
                [cnt] "+r"(cnt_loop), [ptr_din] "+r"(din_hei_ptr)
              :
              [scale0] "w"(w_scale0), [scale1] "w"(w_scale1), [vzero] "w"(vzero)
              : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9",
                "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18",
                "v19", "v20", "v21", "v22", "v23");
#else
          asm volatile(
              "vld1.32 {d8-d11},   [%[ptr_din]]!   @load data \n"
              "vld1.32 {d12-d15},   [%[ptr_din]]!   @load data \n"

              "1:                                 @ main loop\n"
              "vmax.s32   q4, q4, %q[vzero]             @ relu\n"
              "vmax.s32   q5, q5, %q[vzero]             @ relu\n"
              "vmax.s32   q6, q6, %q[vzero]             @ relu\n"
              "vmax.s32   q7, q7, %q[vzero]             @ relu\n"

              // int32-> fp32
              "vmov.f32 q15, #0.5                    \n"
              "vcvt.f32.s32   q8, q4                  \n"
              "vcvt.f32.s32   q9, q5                  \n"
              "vcvt.f32.s32   q10, q6                  \n"
              "vcvt.f32.s32   q11, q7                  \n"

              "vand.i32   q4, q15, q15    @ set offset, 0.5\n"
              "vand.i32   q5, q15, q15                  @ set offset, 0.5\n"
              "vand.i32   q6, q15, q15                  @ set offset, 0.5\n"
              "vand.i32   q7, q15, q15                  @ set offset, 0.5\n"

              "vmov.f32 q15, #-0.5                    \n"

              "vcgt.f32   q12, q8, %q[vzero]           @ get mask > 0, in0\n"
              "vcgt.f32   q13, q9, %q[vzero]           @ get mask > 0, in0\n"
              "vcgt.f32   q14, q10, %q[vzero]           @ get mask > 0, in0\n"
              "vcgt.f32   q3, q11, %q[vzero]           @ get mask > 0, in0\n"

              "vbif.f32   q4, q15, q12           @ get right offset\n"
              "vbif.f32   q5, q15, q13           @ get right offset\n"
              "vbif.f32   q6, q15, q14           @ get right offset\n"
              "vbif.f32   q7, q15, q3           @ get right offset\n"

              "vld1.32 {d24-d27},  [%[ptr_din]]!   @load data \n"
              "vld1.32 {d28-d29}, [%[ptr_din]]!   @load data \n"
              "vld1.32 {d6-d7}, [%[ptr_din]]!   @load data \n"

              "vmla.f32   q4, q8, %q[scale0]          @ mul scale\n"
              "vmla.f32   q5, q9, %q[scale1]         @ mul scale\n"
              "vmla.f32   q6, q10, %q[scale0]          @ mul scale\n"
              "vmla.f32   q7, q11, %q[scale1]          @ mul scale\n"

              "vmax.s32   q12, q12, %q[vzero]             @ relu\n"
              "vmax.s32   q13, q13, %q[vzero]             @ relu\n"
              "vmax.s32   q14, q14, %q[vzero]             @ relu\n"
              "vmax.s32   q3, q3, %q[vzero]             @ relu\n"

              "vcvt.s32.f32  q8, q4                   @ cvt to int32\n"
              "vcvt.s32.f32  q9, q5                   @ cvt to int32\n"
              "vcvt.s32.f32  q10, q6                   @ cvt to int32\n"
              "vcvt.s32.f32  q11, q7                   @ cvt to int32\n"

              "vqmovn.s32 d8, q8                     @ cnt to int16\n"
              "vqmovn.s32 d10, q9                     @ cnt to int16\n"
              "vqmovn.s32 d12, q10                     @ cnt to int16\n"
              "vqmovn.s32 d14, q11                     @ cnt to int16\n"

              "vqmovn.s16 d16, q4                      @ cnt to int8\n"
              "vqmovn.s16 d17, q5                      @ cnt to int8\n"
              "vqmovn.s16 d18, q6                      @ cnt to int8\n"
              "vqmovn.s16 d19, q7                      @ cnt to int8\n"

              "vmov.f32 q15, #0.5                    \n"

              "vcvt.f32.s32   q4, q12                  \n"
              "vcvt.f32.s32   q5, q13                  \n"
              "vcvt.f32.s32   q6, q14                  \n"
              "vcvt.f32.s32   q7, q3                  \n"

              "vand.i32   q12, q15, q15    @ set offset, 0.5\n"
              "vand.i32   q13, q15, q15                  @ set offset, 0.5\n"
              "vand.i32   q14, q15, q15                  @ set offset, 0.5\n"
              "vand.i32   q3, q15, q15                  @ set offset, 0.5\n"

              "vmov.f32 q15, #-0.5                    \n"

              "vcgt.f32   q10, q4, %q[vzero]           @ get mask > 0, in0\n"
              "vcgt.f32   q11, q5, %q[vzero]           @ get mask > 0, in0\n"

              "vbif.f32   q12, q15, q10           @ get right offset\n"
              "vbif.f32   q13, q15, q11           @ get right offset\n"

              "vcgt.f32   q10, q6, %q[vzero]           @ get mask > 0, in0\n"
              "vcgt.f32   q11, q7, %q[vzero]           @ get mask > 0, in0\n"

              "vbif.f32   q14, q15, q10           @ get right offset\n"
              "vbif.f32   q3, q15, q11           @ get right offset\n"

              "vmla.f32   q12, q4, %q[scale0]          @ mul scale\n"
              "vmla.f32   q13, q5, %q[scale1]           @ mul scale\n"
              "vmla.f32   q14, q6, %q[scale0]           @ mul scale\n"
              "vmla.f32   q3, q7, %q[scale1]           @ mul scale\n"

              "vcvt.s32.f32  q4, q12                   @ cvt to int32\n"
              "vcvt.s32.f32  q5, q13                   @ cvt to int32\n"
              "vcvt.s32.f32  q6, q14                   @ cvt to int32\n"
              "vcvt.s32.f32  q7, q3                   @ cvt to int32\n"

              "vqmovn.s32 d24, q4                     @ cnt to int16\n"
              "vqmovn.s32 d26, q5                     @ cnt to int16\n"
              "vqmovn.s32 d28, q6                     @ cnt to int16\n"
              "vqmovn.s32 d6, q7                    @ cnt to int16\n"

              "vqmovn.s16 d20, q12                      @ cnt to int8\n"
              "vqmovn.s16 d21, q13                      @ cnt to int8\n"
              "vqmovn.s16 d22, q14                      @ cnt to int8\n"
              "vqmovn.s16 d23, q3                      @ cnt to int8\n"

              "vtrn.8    d16, d18                  @ trans q0, q2 \n"
              "vtrn.8    d20, d22                 @ trans q4, q6 \n"
              "vtrn.16    d16, d20                  @ trans q0, q2 \n"
              "vtrn.16    d18, d22                 @ trans q4, q6 \n"

              "vtrn.8    d17, d19                  @ trans q0, q2 \n"
              "vtrn.8    d21, d23                 @ trans q4, q6 \n"
              "vtrn.16    d17, d21                  @ trans q0, q2 \n"
              "vtrn.16    d19, d23                 @ trans q4, q6 \n"

              "vld1.32 {d8-d11},   [%[ptr_din]]!   @load data \n"
              "vld1.32 {d12-d15},   [%[ptr_din]]!   @load data \n"

              "vst1.32  {d16[0]},   [%[doutc0r0]]  @ store result, add "
              "pointer\n"
              "vst1.32  {d18[0]},   [%[doutc1r0]]  @ store result, add "
              "pointer\n"
              "vst1.32  {d20[0]},   [%[doutc2r0]]  @ store result, add "
              "pointer\n"
              "vst1.32  {d22[0]},   [%[doutc3r0]]  @ store result, add "
              "pointer\n"

              "vst1.32  {d17[0]},   [%[doutc4r0]]  @ store result, add "
              "pointer\n"
              "vst1.32  {d19[0]},   [%[doutc5r0]]  @ store result, add "
              "pointer\n"
              "vst1.32  {d21[0]},   [%[doutc6r0]]  @ store result, add "
              "pointer\n"
              "vst1.32  {d23[0]},   [%[doutc7r0]]  @ store result, add "
              "pointer\n"

              "add %[doutc0r0], #4                @ add \n"
              "add %[doutc1r0], #4                @ add \n"
              "add %[doutc2r0], #4                @ add \n"
              "add %[doutc3r0], #4                @ add \n"

              "subs   %[cnt], %[cnt], #1          @ loop count - 1\n"

              "add %[doutc4r0], #4                @ add \n"
              "add %[doutc5r0], #4                @ add \n"
              "add %[doutc6r0], #4                @ add \n"
              "add %[doutc7r0], #4                @ add \n"
              "bne    1b                           @ jump to main loop\n"

              : [doutc0r0] "+r"(doutc0_ptr), [doutc1r0] "+r"(doutc1_ptr),
                [doutc2r0] "+r"(doutc2_ptr), [doutc3r0] "+r"(doutc3_ptr),
                [doutc4r0] "+r"(doutc4_ptr), [doutc5r0] "+r"(doutc5_ptr),
                [doutc6r0] "+r"(doutc6_ptr), [doutc7r0] "+r"(doutc7_ptr),
                [ptr_din] "+r"(din_hei_ptr), [cnt] "+r"(cnt_loop)
              :
              [scale0] "w"(w_scale0), [scale1] "w"(w_scale1), [vzero] "w"(vzero)
              : "q3", "q4", "q5", "q6", "q7", "q8", "q9", "q10", "q11", "q12",
                "q13", "q14", "q15");
#endif
        }
        if (we > width) {
          int offset = 32 * (valid_w / 4 - 1);
          din_hei_ptr = ptr_din + offset;
          int i = we - 4;
          for (; i < width; ++i) {
            *(doutc0_ptr++) = saturate_cast<signed char>(
                roundf(AKMAX(din_hei_ptr[0] * scale[0], 0)));
            *(doutc1_ptr++) = saturate_cast<signed char>(
                roundf(AKMAX(din_hei_ptr[1] * scale[1], 0)));
            *(doutc2_ptr++) = saturate_cast<signed char>(
                roundf(AKMAX(din_hei_ptr[2] * scale[2], 0)));
            *(doutc3_ptr++) = saturate_cast<signed char>(
                roundf(AKMAX(din_hei_ptr[3] * scale[3], 0)));
            *(doutc4_ptr++) = saturate_cast<signed char>(
                roundf(AKMAX(din_hei_ptr[4] * scale[4], 0)));
            *(doutc5_ptr++) = saturate_cast<signed char>(
                roundf(AKMAX(din_hei_ptr[5] * scale[5], 0)));
            *(doutc6_ptr++) = saturate_cast<signed char>(
                roundf(AKMAX(din_hei_ptr[6] * scale[6], 0)));
            *(doutc7_ptr++) = saturate_cast<signed char>(
                roundf(AKMAX(din_hei_ptr[7] * scale[7], 0)));
            din_hei_ptr += 8;
          }
        }
      }
    } else {
      for (int i = 0; i < size_h; i++) {
        int size_w = i * width;
        dtype* doutc0_ptr = doutc0r0 + size_w;  // doutc0r0 + width;
        dtype* doutc1_ptr = doutc1r0 + size_w;
        dtype* doutc2_ptr = doutc2r0 + size_w;
        dtype* doutc3_ptr = doutc3r0 + size_w;
        dtype* doutc4_ptr = doutc4r0 + size_w;
        dtype* doutc5_ptr = doutc5r0 + size_w;
        dtype* doutc6_ptr = doutc6r0 + size_w;
        dtype* doutc7_ptr = doutc7r0 + size_w;
        if (ce > channel) {
          switch (ce - channel) {
            case 7:
              doutc1_ptr = trash_ptr;
            case 6:
              doutc2_ptr = trash_ptr;
            case 5:
              doutc3_ptr = trash_ptr;
            case 4:
              doutc4_ptr = trash_ptr;
            case 3:
              doutc5_ptr = trash_ptr;
            case 2:
              doutc6_ptr = trash_ptr;
            case 1:
              doutc7_ptr = trash_ptr;
            default:
              break;
          }
        }
        ptr_din = din + i * valid_w * ch_n;
        const int* din_hei_ptr = ptr_din;
        if (cnt > 0) {
          int cnt_loop = cnt;
#ifdef __aarch64__
          asm volatile(
              "ldp q0, q1, [%[ptr_din]], #32  \n" /* load r00, r01 to q0, q1 */
              "ldp q2, q3, [%[ptr_din]], #32  \n" /* load r02, r03 to q2, q3 */
              "ldp q4, q5, [%[ptr_din]], #32  \n" /* load r00, r01 to q0, q1 */
              "ldp q6, q7, [%[ptr_din]], #32  \n" /* load r02, r03 to q2, q3 */
              // "movi v20.4s, #0                \n"         /* for relu */
              "1:                             \n" /* main loop*/
              "trn1   v8.4s, v0.4s, v2.4s     \n" /* trans q0, q1*/
              "trn2   v9.4s, v0.4s, v2.4s     \n" /* trans q0, q1*/
              "trn1   v10.4s, v1.4s, v3.4s    \n" /* trans q2, q3*/
              "trn2   v11.4s, v1.4s, v3.4s    \n" /* trans q2, q3*/
              "ldp q0, q1, [%[ptr_din]], #32  \n" /* load r00, r01 to q0, q1 */

              "trn1   v12.4s, v4.4s, v6.4s    \n" /* trans q0, q1*/
              "trn2   v13.4s, v4.4s, v6.4s    \n" /* trans q0, q1*/
              "trn1   v14.4s, v5.4s, v7.4s    \n" /* trans q2, q3*/
              "trn2   v15.4s, v5.4s, v7.4s    \n" /* trans q2, q3*/
              "ldp q2, q3, [%[ptr_din]], #32  \n" /* load r02, r03 to q2, q3 */

              "trn1   v16.2d, v8.2d, v12.2d   \n" /* trans q8, q10 00 01 02 03*/
              "trn2   v17.2d, v8.2d, v12.2d   \n" /* trans q8, q10 20 21 22 23*/
              "trn1   v18.2d, v9.2d, v13.2d   \n" /* trans q9, q11 10 11 12 13*/
              "trn2   v19.2d, v9.2d, v13.2d   \n" /* trans q9, q11 30 31 32 33*/
              "ldp q4, q5, [%[ptr_din]], #32  \n" /* load r00, r01 to q0, q1 */

              "trn1   v8.2d, v10.2d, v14.2d   \n" /* trans q8, q10 40 41 42 43*/
              "trn2   v9.2d, v10.2d, v14.2d   \n" /* trans q8, q10 60 61 62 63*/
              "trn1   v12.2d, v11.2d, v15.2d  \n" /* trans q9, q11 50 51 52 53*/
              "trn2   v13.2d, v11.2d, v15.2d  \n" /* trans q9, q11 70 71 72 73*/
              "ldp q6, q7, [%[ptr_din]], #32  \n" /* load r02, r03 to q2, q3 */

              // int32 --> fp32
              "scvtf   v10.4s, v16.4s               \n"
              "scvtf   v11.4s, v17.4s               \n"
              "scvtf   v14.4s, v18.4s               \n"
              "scvtf   v15.4s, v19.4s               \n"

              "scvtf   v20.4s, v8.4s               \n"
              "scvtf   v21.4s, v9.4s               \n"
              "scvtf   v22.4s, v12.4s               \n"
              "scvtf   v23.4s, v13.4s               \n"

              // mul
              "fmul    v16.4s, v10.4s, %[scale0].s[0]  \n"
              "fmul    v17.4s, v11.4s, %[scale0].s[2]  \n"
              "fmul    v18.4s, v14.4s, %[scale0].s[1] \n"
              "fmul    v19.4s, v15.4s, %[scale0].s[3] \n"

              "fmul    v8.4s, v20.4s, %[scale1].s[0]  \n"
              "fmul    v9.4s, v21.4s, %[scale1].s[2]  \n"
              "fmul    v12.4s, v22.4s, %[scale1].s[1] \n"
              "fmul    v13.4s, v23.4s, %[scale1].s[3] \n"

              // fp32-int32
              "fcvtas  v10.4s, v16.4s                      \n"
              "fcvtas  v11.4s, v17.4s                      \n"
              "fcvtas  v14.4s, v18.4s                      \n"
              "fcvtas  v15.4s, v19.4s                      \n"

              "fcvtas  v20.4s, v8.4s                      \n"
              "fcvtas  v21.4s, v9.4s                      \n"
              "fcvtas  v22.4s, v12.4s                      \n"
              "fcvtas  v23.4s, v13.4s                      \n"

              // int32-int16
              "sqxtn   v16.4h, v10.4s                      \n"
              "sqxtn   v17.4h, v11.4s                      \n"
              "sqxtn   v18.4h, v14.4s                      \n"
              "sqxtn   v19.4h, v15.4s                      \n"

              "sqxtn   v8.4h, v20.4s                      \n"
              "sqxtn   v9.4h, v21.4s                      \n"
              "sqxtn   v12.4h, v22.4s                      \n"
              "sqxtn   v13.4h, v23.4s                      \n"

              // int16-int8
              "sqxtn  v10.8b, v16.8h                      \n"
              "sqxtn  v11.8b, v17.8h                     \n"
              "sqxtn  v14.8b, v18.8h                      \n"
              "sqxtn  v15.8b, v19.8h                     \n"

              "sqxtn  v20.8b, v8.8h                      \n"
              "sqxtn  v21.8b, v9.8h                     \n"
              "sqxtn  v22.8b, v12.8h                      \n"
              "sqxtn  v23.8b, v13.8h                     \n"

              "str    s10, [%[doutc0r0]], #4 \n" /* store c0r0*/
              "str    s11, [%[doutc2r0]], #4 \n" /* store c2r0*/
              "str    s14, [%[doutc1r0]], #4 \n" /* store c1r0*/
              "str    s15, [%[doutc3r0]], #4 \n" /* store c3r0*/

              "subs   %w[cnt], %w[cnt],  #1   \n" /* loop count -1*/
              "str    s20, [%[doutc4r0]], #4  \n" /* store c0r0*/
              "str    s21, [%[doutc6r0]], #4  \n" /* store c2r0*/
              "str    s22, [%[doutc5r0]], #4 \n"  /* store c1r0*/
              "str    s23, [%[doutc7r0]], #4 \n"  /* store c3r0*/

              "bne    1b                      \n" /* jump to main loop*/

              : [doutc0r0] "+r"(doutc0_ptr), [doutc1r0] "+r"(doutc1_ptr),
                [doutc2r0] "+r"(doutc2_ptr), [doutc3r0] "+r"(doutc3_ptr),
                [doutc4r0] "+r"(doutc4_ptr), [doutc5r0] "+r"(doutc5_ptr),
                [doutc6r0] "+r"(doutc6_ptr), [doutc7r0] "+r"(doutc7_ptr),
                [cnt] "+r"(cnt_loop), [ptr_din] "+r"(din_hei_ptr)
              : [scale0] "w"(w_scale0), [scale1] "w"(w_scale1)
              : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9",
                "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18",
                "v19", "v20", "v21", "v22", "v23");
#else
          asm volatile(
              "vld1.32 {d8-d11},   [%[ptr_din]]!   @load data \n"
              "vld1.32 {d12-d15},   [%[ptr_din]]!   @load data \n"

              "1:                                 @ main loop\n"
              // int32-> fp32
              "vmov.f32 q15, #0.5                    \n"
              "vcvt.f32.s32   q8, q4                  \n"
              "vcvt.f32.s32   q9, q5                  \n"
              "vcvt.f32.s32   q10, q6                  \n"
              "vcvt.f32.s32   q11, q7                  \n"

              "vand.i32   q4, q15, q15    @ set offset, 0.5\n"
              "vand.i32   q5, q4, q4                  @ set offset, 0.5\n"
              "vand.i32   q6, q4, q4                  @ set offset, 0.5\n"
              "vand.i32   q7, q4, q4                  @ set offset, 0.5\n"

              "vmov.f32 q15, #-0.5                    \n"

              "vcgt.f32   q12, q8, %q[vzero]           @ get mask > 0, in0\n"
              "vcgt.f32   q13, q9, %q[vzero]           @ get mask > 0, in0\n"
              "vcgt.f32   q14, q10, %q[vzero]           @ get mask > 0, in0\n"
              "vcgt.f32   q3, q11, %q[vzero]           @ get mask > 0, in0\n"

              "vbif.f32   q4, q15, q12           @ get right offset\n"
              "vbif.f32   q5, q15, q13           @ get right offset\n"
              "vbif.f32   q6, q15, q14           @ get right offset\n"
              "vbif.f32   q7, q15, q3           @ get right offset\n"

              "vld1.32 {d24-d27},  [%[ptr_din]]!   @load data \n"
              "vld1.32 {d28-d29}, [%[ptr_din]]!   @load data \n"
              "vld1.32 {d6-d7}, [%[ptr_din]]!   @load data \n"

              "vmla.f32   q4, q8, %q[scale0]          @ mul scale\n"
              "vmla.f32   q5, q9, %q[scale1]           @ mul scale\n"
              "vmla.f32   q6, q10, %q[scale0]           @ mul scale\n"
              "vmla.f32   q7, q11, %q[scale1]          @ mul scale\n"

              "vcvt.s32.f32  q8, q4                   @ cvt to int32\n"
              "vcvt.s32.f32  q9, q5                   @ cvt to int32\n"
              "vcvt.s32.f32  q10, q6                   @ cvt to int32\n"
              "vcvt.s32.f32  q11, q7                   @ cvt to int32\n"

              "vqmovn.s32 d8, q8                     @ cnt to int16\n"
              "vqmovn.s32 d10, q9                     @ cnt to int16\n"
              "vqmovn.s32 d12, q10                     @ cnt to int16\n"
              "vqmovn.s32 d14, q11                     @ cnt to int16\n"

              "vqmovn.s16 d16, q4                      @ cnt to int8\n"
              "vqmovn.s16 d17, q5                      @ cnt to int8\n"
              "vqmovn.s16 d18, q6                      @ cnt to int8\n"
              "vqmovn.s16 d19, q7                      @ cnt to int8\n"

              "vmov.f32 q15, #0.5                    \n"

              "vcvt.f32.s32   q4, q12                  \n"
              "vcvt.f32.s32   q5, q13                  \n"
              "vcvt.f32.s32   q6, q14                  \n"
              "vcvt.f32.s32   q7, q3                  \n"

              "vand.i32   q12, q15, q15    @ set offset, 0.5\n"
              "vand.i32   q13, q12, q12                  @ set offset, 0.5\n"
              "vand.i32   q14, q12, q12                  @ set offset, 0.5\n"
              "vand.i32   q3, q12, q12                  @ set offset, 0.5\n"

              "vmov.f32 q15, #-0.5                    \n"

              "vcgt.f32   q10, q4, %q[vzero]           @ get mask > 0, in0\n"
              "vcgt.f32   q11, q5, %q[vzero]           @ get mask > 0, in0\n"

              "vbif.f32   q12, q15, q10           @ get right offset\n"
              "vbif.f32   q13, q15, q11           @ get right offset\n"

              "vcgt.f32   q10, q6, %q[vzero]           @ get mask > 0, in0\n"
              "vcgt.f32   q11, q7, %q[vzero]           @ get mask > 0, in0\n"

              "vbif.f32   q14, q15, q10           @ get right offset\n"
              "vbif.f32   q3, q15, q11           @ get right offset\n"

              "vmla.f32   q12, q4, %q[scale0]           @ mul scale\n"
              "vmla.f32   q13, q5, %q[scale1]        @ mul scale\n"
              "vmla.f32   q14, q6, %q[scale0]          @ mul scale\n"
              "vmla.f32   q3, q7, %q[scale1]         @ mul scale\n"

              "vcvt.s32.f32  q4, q12                   @ cvt to int32\n"
              "vcvt.s32.f32  q5, q13                   @ cvt to int32\n"
              "vcvt.s32.f32  q6, q14                   @ cvt to int32\n"
              "vcvt.s32.f32  q7, q3                   @ cvt to int32\n"

              "vqmovn.s32 d24, q4                     @ cnt to int16\n"
              "vqmovn.s32 d26, q5                     @ cnt to int16\n"
              "vqmovn.s32 d28, q6                     @ cnt to int16\n"
              "vqmovn.s32 d6, q7                    @ cnt to int16\n"

              "vqmovn.s16 d20, q12                      @ cnt to int8\n"
              "vqmovn.s16 d21, q13                      @ cnt to int8\n"
              "vqmovn.s16 d22, q14                      @ cnt to int8\n"
              "vqmovn.s16 d23, q3                      @ cnt to int8\n"

              "vtrn.8    d16, d18                  @ trans q0, q2 \n"
              "vtrn.8    d20, d22                 @ trans q4, q6 \n"
              "vtrn.16    d16, d20                  @ trans q0, q2 \n"
              "vtrn.16    d18, d22                 @ trans q4, q6 \n"

              "vtrn.8    d17, d19                  @ trans q0, q2 \n"
              "vtrn.8    d21, d23                 @ trans q4, q6 \n"
              "vtrn.16    d17, d21                  @ trans q0, q2 \n"
              "vtrn.16    d19, d23                 @ trans q4, q6 \n"

              "vld1.32 {d8-d11},   [%[ptr_din]]!   @load data \n"
              "vld1.32 {d12-d15},   [%[ptr_din]]!   @load data \n"

              "vst1.32  {d16[0]},   [%[doutc0r0]]  @ store result, add "
              "pointer\n"
              "vst1.32  {d18[0]},   [%[doutc1r0]]  @ store result, add "
              "pointer\n"
              "vst1.32  {d20[0]},   [%[doutc2r0]]  @ store result, add "
              "pointer\n"
              "vst1.32  {d22[0]},   [%[doutc3r0]]  @ store result, add "
              "pointer\n"

              "vst1.32  {d17[0]},   [%[doutc4r0]]  @ store result, add "
              "pointer\n"
              "vst1.32  {d19[0]},   [%[doutc5r0]]  @ store result, add "
              "pointer\n"
              "vst1.32  {d21[0]},   [%[doutc6r0]]  @ store result, add "
              "pointer\n"
              "vst1.32  {d23[0]},   [%[doutc7r0]]  @ store result, add "
              "pointer\n"

              "add %[doutc0r0], #4                @ add \n"
              "add %[doutc1r0], #4                @ add \n"
              "add %[doutc2r0], #4                @ add \n"
              "add %[doutc3r0], #4                @ add \n"

              "subs   %[cnt], %[cnt], #1          @ loop count - 1\n"

              "add %[doutc4r0], #4                @ add \n"
              "add %[doutc5r0], #4                @ add \n"
              "add %[doutc6r0], #4                @ add \n"
              "add %[doutc7r0], #4                @ add \n"
              "bne    1b                           @ jump to main loop\n"

              : [doutc0r0] "+r"(doutc0_ptr), [doutc1r0] "+r"(doutc1_ptr),
                [doutc2r0] "+r"(doutc2_ptr), [doutc3r0] "+r"(doutc3_ptr),
                [doutc4r0] "+r"(doutc4_ptr), [doutc5r0] "+r"(doutc5_ptr),
                [doutc6r0] "+r"(doutc6_ptr), [doutc7r0] "+r"(doutc7_ptr),
                [ptr_din] "+r"(din_hei_ptr), [cnt] "+r"(cnt_loop)
              :
              [scale0] "w"(w_scale0), [scale1] "w"(w_scale1), [vzero] "w"(vzero)
              : "q3", "q4", "q5", "q6", "q7", "q8", "q9", "q10", "q11", "q12",
                "q13", "q14", "q15");
#endif
        }
        if (we > width) {
          int offset = 32 * (valid_w / 4 - 1);
          din_hei_ptr = ptr_din + offset;
          int i = we - 4;
          for (; i < width; ++i) {
            *(doutc0_ptr++) =
                saturate_cast<signed char>(roundf(din_hei_ptr[0] * scale[0]));
            *(doutc1_ptr++) =
                saturate_cast<signed char>(roundf(din_hei_ptr[1] * scale[1]));
            *(doutc2_ptr++) =
                saturate_cast<signed char>(roundf(din_hei_ptr[2] * scale[2]));
            *(doutc3_ptr++) =
                saturate_cast<signed char>(roundf(din_hei_ptr[3] * scale[3]));
            *(doutc4_ptr++) =
                saturate_cast<signed char>(roundf(din_hei_ptr[4] * scale[4]));
            *(doutc5_ptr++) =
                saturate_cast<signed char>(roundf(din_hei_ptr[5] * scale[5]));
            *(doutc6_ptr++) =
                saturate_cast<signed char>(roundf(din_hei_ptr[6] * scale[6]));
            *(doutc7_ptr++) =
                saturate_cast<signed char>(roundf(din_hei_ptr[7] * scale[7]));
            din_hei_ptr += 8;
          }
        }
      }
    }
  } else {
    LOG(ERROR) << "ERROR: unsupported input data type!!";
    return false;
  }
  return true;
}

/*
* din [n, hei_n, ch_n, w]
* dout [n, ch_n, hei_n, w]
*/
template <typename dtype>
static bool write_to_output_numc(const dtype* din, dtype* dout, int ch_n,
                                 int hei_n, int cs, int ce, int hs, int he,
                                 int ws, int we, int channel, int height,
                                 int width, bool flag_relu, dtype* trash_ptr) {
  if (ch_n <= 0 || hei_n <= 0) {
    LOG(ERROR) << "ch_n and hei_n are more than zero";
    return false;
  }
  int size_c_out = width * height;

  dtype* out_array[ch_n];
  out_array[0] = dout + cs * size_c_out + hs * width + ws;

  for (int i = 1; i < ch_n; i++) {
    out_array[i] = out_array[i - 1] + size_c_out;
  }

  const dtype* ptr_din = din;

  int cremain = ce - channel;
  for (int i = 1; i <= cremain; i++) {
    out_array[ch_n - i] = trash_ptr;
  }

  int size_h = (he > height ? height : he) - hs;  // size_h == hei_n

  int size_w = we - ws;

  int size_c_in = ch_n * size_w;

  size_t valid_w_byte = width * sizeof(dtype);

  if (flag_relu) {
    for (int h = 0; h < size_h; h++) {
      const dtype* din_ptr = din + h * size_c_in;
      for (int i = 0; i < ch_n; i++) {
        dtype* dout_ptr = out_array[i] + h * width;
        for (int k = 0; k < width; k++) {
          *(dout_ptr++) = AKMAX(din_ptr[k], 0);
        }
        din_ptr += size_w;
      }
    }
  } else {
    for (int h = 0; h < size_h; h++) {
      const dtype* din_ptr = din + h * size_c_in;
      for (int i = 0; i < ch_n; i++) {
        dtype* dout_ptr = out_array[i] + h * width;
        memcpy(dout_ptr, din_ptr, valid_w_byte);
        din_ptr += size_w;
      }
    }
  }
  return true;
}

/// ch_n == ce - cs ??
/// hei_n == he - hs ??
/// channel height width ? -> output
template <typename ditype, typename dotype>
static bool write2_to_output_numc(const ditype* din, dotype* dout, int ch_n,
                                  int hei_n, int cs, int ce, int hs, int he,
                                  int ws, int we, int channel, int height,
                                  int width, bool flag_relu, dotype* trash_ptr,
                                  float const* scales) {
  // static_assert(std::is_same<dotype, float>::value, "just support float");

  if (ch_n <= 0 || hei_n <= 0) {
    LOG(ERROR) << "ch_n and hei_n are more than zero";
    return false;
  }

  int size_c_out = width * height;

  dotype* out_array[ch_n];
  out_array[0] = dout + cs * size_c_out + hs * width + ws;

  for (int i = 1; i < ch_n; i++) {
    out_array[i] = out_array[i - 1] + size_c_out;
  }

  const ditype* ptr_din = din;

  int cremain = ce - channel;
  for (int i = 1; i <= cremain; i++) {
    out_array[ch_n - i] = trash_ptr;
  }

  int size_h = (he > height ? height : he) - hs;  // size_h == hei_n

  int size_w = we - ws;

  int size_c_in = ch_n * size_w;

  size_t valid_w_byte = width * sizeof(ditype);

  if (flag_relu) {
    for (int h = 0; h < size_h; h++) {
      ditype const* din_ptr = din + h * size_c_in;
      for (int i = 0; i < ch_n; i++) {
        float const ws = scales[(i + cs) % ch_n];
        dotype* dout_ptr = out_array[i] + h * width;
        for (int k = 0; k < width; k++) {
          *(dout_ptr++) = AKMAX(din_ptr[k] * ws, 0);
        }
        din_ptr += size_w;
      }
    }
  } else {
    for (int h = 0; h < size_h; h++) {
      ditype const* din_ptr = din + h * size_c_in;
      for (int i = 0; i < ch_n; i++) {
        dotype* dout_ptr = out_array[i] + h * width;

        float const* ws = &scales[(i + cs) % ch_n];
        int32_to_dtype(din_ptr, dout_ptr, ws, 1, 1, width);

        din_ptr += size_w;
      }
    }
  }
  return true;
}
/**
* innput din: nchwc(num)
*/
inline bool fill_packed_bias_nxmw_fp32(const float* bias, float* dout, int ch_n,
                                       int hei_n, int wround) {
  if (ch_n <= 0 || hei_n <= 0) {
    LOG(ERROR) << "ch_n and hei_n are more than zero";
    return false;
  }
  int cnt_ch = ch_n / 4;
  int size = wround * ch_n;
  for (int h = 0; h < hei_n; h++) {
    float* dout_ptr = dout + h * size;
    for (int i = 0; i < wround; i++) {
      const float* bias_ptr = bias;
      int j = 0;
      for (; j < cnt_ch; j++) {
        float32x4_t vb = vld1q_f32(bias_ptr);
        bias_ptr += 4;

        vst1q_f32(dout_ptr, vb);
        dout_ptr += 4;
      }
      j = j * 4;
      for (; j < ch_n; j++) {
        *dout_ptr = *bias_ptr;
        dout_ptr++;
        bias_ptr++;
      }
    }
  }
}

inline bool fill_packed_bias_nxmw_int8(const int* bias, int* dout, int ch_n,
                                       int hei_n, int wround) {
  if (ch_n <= 0 || hei_n <= 0) {
    LOG(ERROR) << "ch_n and hei_n are more than zero";
    return false;
  }
  int cnt_ch = ch_n / 4;
  int size = wround * ch_n;
  for (int h = 0; h < hei_n; h++) {
    int* dout_ptr = dout + h * size;
    for (int i = 0; i < wround; i++) {
      const int* bias_ptr = bias;
      int j = 0;
      for (; j < cnt_ch; j++) {
        int32x4_t vb = vld1q_s32(bias_ptr);
        bias_ptr += 4;

        vst1q_s32(dout_ptr, vb);
        dout_ptr += 4;
      }
      j = j * 4;
      for (; j < ch_n; j++) {
        *dout_ptr = *bias_ptr;
        dout_ptr++;
        bias_ptr++;
      }
    }
  }
  return true;
}

}  // namespace math
}  // namespace arm
}  // namespace lite
}  // namespace paddle
