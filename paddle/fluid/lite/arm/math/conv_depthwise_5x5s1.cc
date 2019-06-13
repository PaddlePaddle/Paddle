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

//!    weights layout
//!            *-----------------------*-----*
//!    w0  <-- | W0    W1    W2    W3  | W4  |
//!            *-----------------------*     |
//!    w1  <-- | W5    W6    W7    W8  | W9  |
//!            *-----------------------*     | -->  w5
//!    w2  <-- | W10   W11   W12   W13 | W14 |
//!            *-----------------------*     |
//!    w3  <-- | W15   W16   W17   W18 | W19 |
//!            *-----------------------*-----*
//!    w4  <-- | W20   W21   W22   W23 | W24 | -->  w6[0]
//!            *-----------------------*-----*

void conv_depthwise_5x5s1_impl(const float* din, float* dout, int num,
                               int ch_out, int h_out, int w_out, int ch_in,
                               int h_in, int w_in, const float* weights,
                               const float* bias, int pad, bool flag_bias,
                               bool flag_relu, ARMContext* ctx);

void conv_depthwise_5x5s1_small_impl(const float* din, float* dout, int num,
                                     int ch_out, int h_out, int w_out,
                                     int ch_in, int h_in, int w_in,
                                     const float* weights, const float* bias,
                                     int pad, bool flag_bias, bool flag_relu,
                                     ARMContext* ctx);

void conv_depthwise_5x5s1_relu_impl(const float* din, float* dout, int num,
                                    int ch_out, int h_out, int w_out, int ch_in,
                                    int h_in, int w_in, const float* weights,
                                    const float* bias, int pad, bool flag_bias,
                                    bool flag_relu, ARMContext* ctx);

void conv_depthwise_5x5s1_small_relu_impl(
    const float* din, float* dout, int num, int ch_out, int h_out, int w_out,
    int ch_in, int h_in, int w_in, const float* weights, const float* bias,
    int pad, bool flag_bias, bool flag_relu, ARMContext* ctx);

static float* prepad_input(const float* input, int num, int ch_in, int h_in,
                           int w_in, int pad) {
  int h_new = h_in + 2 * pad;
  int w_new = w_in + 2 * pad;
  float* new_input =
      static_cast<float*>(malloc(h_new * w_new * ch_in * num * sizeof(float)));
  float* new_input_ptr = new_input;
  for (int c = 0; c < num * ch_in; ++c) {
    memset(new_input_ptr, 0x00, w_new * pad * sizeof(float));
    new_input_ptr += w_new * pad;
    for (int i = 0; i < h_in; ++i) {
      memset(new_input_ptr, 0x00, pad * sizeof(float));
      new_input_ptr += pad;
      memcpy(new_input_ptr, input, w_in * sizeof(float));
      new_input_ptr += w_in;
      input += w_in;
      memset(new_input_ptr, 0x00, pad * sizeof(float));
      new_input_ptr += pad;
    }
    memset(new_input_ptr, 0x00, w_new * pad * sizeof(float));
    new_input_ptr += w_new * pad;
  }
  return new_input;
}

#ifdef __aarch64__

//! kernel for one out without extracting data mid
//! deal with four lines out
void compute_one_out_without_extract(
    const float* din0, const float* din1, const float* din2, const float* din3,
    const float* din4, const float* din5, const float* din6, const float* din7,
    float* dout0, float* dout1, float* dout2, float* dout3, float32x4_t w0,
    float32x4_t w1, float32x4_t w2, float32x4_t w3, float32x4_t w4,
    float32x4_t w5, float32x4_t w6, const float* bias) {
  //! din0 - din7: 0-4 v8-v15
  //! din0 - din7: 5   v20, v21
  //! dout0 - dout3: v16-v19
  asm volatile(
      "ld1 {v8.4s}, [%[din0]], #16  \n"
      "ld1 {v9.4s}, [%[din1]], #16  \n"
      "ld1 {v10.4s}, [%[din2]], #16  \n"
      "ld1 {v11.4s}, [%[din3]], #16  \n"
      "ld1 {v12.4s}, [%[din4]], #16  \n"
      "ld1 {v13.4s}, [%[din5]], #16  \n"

      // in row0
      "fmul v16.4s, %[w0].4s, v8.4s  \n"
      "fmul v17.4s, %[w0].4s, v9.4s  \n"
      "fmul v18.4s, %[w0].4s, v10.4s  \n"
      "fmul v19.4s, %[w0].4s, v11.4s  \n"

      "ld1 {v14.4s}, [%[din6]], #16  \n"
      "ld1 {v15.4s}, [%[din7]], #16  \n"

      // in row1
      "fmla v16.4s, %[w1].4s, v9.4s \n"
      "fmla v17.4s, %[w1].4s, v10.4s \n"
      "fmla v18.4s, %[w1].4s, v11.4s \n"
      "fmla v19.4s, %[w1].4s, v12.4s \n"

      "ld1 {v20.s}[0], [%[din0]]  \n"
      "ld1 {v21.s}[0], [%[din4]]  \n"
      "ld1 {v20.s}[1], [%[din1]]  \n"
      "ld1 {v21.s}[1], [%[din5]]  \n"

      // in row2
      "fmla v16.4s, %[w2].4s, v10.4s \n"
      "fmla v17.4s, %[w2].4s, v11.4s \n"
      "fmla v18.4s, %[w2].4s, v12.4s \n"
      "fmla v19.4s, %[w2].4s, v13.4s \n"

      "ld1 {v20.s}[2], [%[din2]]  \n"
      "ld1 {v21.s}[2], [%[din6]]  \n"
      "ld1 {v20.s}[3], [%[din3]]  \n"
      "ld1 {v21.s}[3], [%[din7]]  \n"

      // in row3
      "fmla v16.4s, %[w3].4s, v11.4s \n"
      "fmla v17.4s, %[w3].4s, v12.4s \n"
      "fmla v18.4s, %[w3].4s, v13.4s \n"
      "fmla v19.4s, %[w3].4s, v14.4s \n"

      // in row4
      "fmla v16.4s, %[w4].4s, v12.4s \n"
      "fmla v17.4s, %[w4].4s, v13.4s \n"
      "fmla v18.4s, %[w4].4s, v14.4s \n"
      "fmla v19.4s, %[w4].4s, v15.4s \n"

      // ext
      "ext v22.16b, v20.16b, v21.16b, #4  \n"  // 1 2 3 4
      "ext v23.16b, v20.16b, v21.16b, #8  \n"  // 2 3 4 5
      "ext v24.16b, v20.16b, v21.16b, #12 \n"  // 3 4 5 6

      // in col5
      "fmla v16.4s, %[w5].4s, v20.4s  \n"
      "fmla v17.4s, %[w5].4s, v22.4s  \n"
      "fmla v18.4s, %[w5].4s, v23.4s  \n"
      "fmla v19.4s, %[w5].4s, v24.4s  \n"

      "ld1 {v31.4s}, [%[bias]] \n"

      // add to out register v25
      "faddp v25.4s, v16.4s, v17.4s  \n"
      "faddp v26.4s, v18.4s, v19.4s  \n"
      "faddp v25.4s, v25.4s, v26.4s  \n"

      // in[24] * w6[0]
      "fmla v25.4s, v21.4s, %[w6].s[0]\n"
      "fadd v25.4s, v25.4s, v31.4s    \n"

      // write output
      "st1 {v25.s}[0], [%[dout0]]  \n"
      "st1 {v25.s}[1], [%[dout1]]  \n"
      "st1 {v25.s}[2], [%[dout2]]  \n"
      "st1 {v25.s}[3], [%[dout3]]  \n"
      : [din0] "+r"(din0), [din1] "+r"(din1), [din2] "+r"(din2),
        [din3] "+r"(din3), [din4] "+r"(din4), [din5] "+r"(din5),
        [din6] "+r"(din6), [din7] "+r"(din7)
      : [dout0] "r"(dout0), [dout1] "r"(dout1), [dout2] "r"(dout2),
        [dout3] "r"(dout3), [w0] "w"(w0), [w1] "w"(w1), [w2] "w"(w2),
        [w3] "w"(w3), [w4] "w"(w4), [w5] "w"(w5), [w6] "w"(w6), [bias] "r"(bias)
      : "memory", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16",
        "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25", "v26",
        "v31");
}

//! kernel for one out without extracting data mid
//! deal with four lines out
void compute_one_out_without_extract_relu(
    const float* din0, const float* din1, const float* din2, const float* din3,
    const float* din4, const float* din5, const float* din6, const float* din7,
    float* dout0, float* dout1, float* dout2, float* dout3, float32x4_t w0,
    float32x4_t w1, float32x4_t w2, float32x4_t w3, float32x4_t w4,
    float32x4_t w5, float32x4_t w6, const float* bias) {
  //! din0 - din7: 0-4 v8-v15
  //! din0 - din7: 5   v20, v21
  //! dout0 - dout3: v16-v19
  asm volatile(
      "ld1 {v8.4s}, [%[din0]], #16   \n"
      "ld1 {v9.4s}, [%[din1]], #16   \n"
      "ld1 {v10.4s}, [%[din2]], #16  \n"
      "ld1 {v11.4s}, [%[din3]], #16  \n"
      "ld1 {v12.4s}, [%[din4]], #16  \n"
      "ld1 {v13.4s}, [%[din5]], #16  \n"

      // in row0
      "fmul v16.4s, %[w0].4s, v8.4s   \n"
      "fmul v17.4s, %[w0].4s, v9.4s   \n"
      "fmul v18.4s, %[w0].4s, v10.4s  \n"
      "fmul v19.4s, %[w0].4s, v11.4s  \n"

      "ld1 {v14.4s}, [%[din6]], #16  \n"
      "ld1 {v15.4s}, [%[din7]], #16  \n"

      // in row1
      "fmla v16.4s, %[w1].4s, v9.4s  \n"
      "fmla v17.4s, %[w1].4s, v10.4s \n"
      "fmla v18.4s, %[w1].4s, v11.4s \n"
      "fmla v19.4s, %[w1].4s, v12.4s \n"

      "ld1 {v20.s}[0], [%[din0]]  \n"
      "ld1 {v21.s}[0], [%[din4]]  \n"
      "ld1 {v20.s}[1], [%[din1]]  \n"
      "ld1 {v21.s}[1], [%[din5]]  \n"

      // in row2
      "fmla v16.4s, %[w2].4s, v10.4s \n"
      "fmla v17.4s, %[w2].4s, v11.4s \n"
      "fmla v18.4s, %[w2].4s, v12.4s \n"
      "fmla v19.4s, %[w2].4s, v13.4s \n"

      "ld1 {v20.s}[2], [%[din2]]  \n"
      "ld1 {v21.s}[2], [%[din6]]  \n"
      "ld1 {v20.s}[3], [%[din3]]  \n"
      "ld1 {v21.s}[3], [%[din7]]  \n"

      // in row3
      "fmla v16.4s, %[w3].4s, v11.4s \n"
      "fmla v17.4s, %[w3].4s, v12.4s \n"
      "fmla v18.4s, %[w3].4s, v13.4s \n"
      "fmla v19.4s, %[w3].4s, v14.4s \n"

      // in row4
      "fmla v16.4s, %[w4].4s, v12.4s \n"
      "fmla v17.4s, %[w4].4s, v13.4s \n"
      "fmla v18.4s, %[w4].4s, v14.4s \n"
      "fmla v19.4s, %[w4].4s, v15.4s \n"

      // ext
      "ext v22.16b, v20.16b, v21.16b, #4  \n"  // 1 2 3 4
      "ext v23.16b, v20.16b, v21.16b, #8  \n"  // 2 3 4 5
      "ext v24.16b, v20.16b, v21.16b, #12 \n"  // 3 4 5 6

      // in col5
      "fmla v16.4s, %[w5].4s, v20.4s  \n"
      "fmla v17.4s, %[w5].4s, v22.4s  \n"
      "fmla v18.4s, %[w5].4s, v23.4s  \n"
      "fmla v19.4s, %[w5].4s, v24.4s  \n"

      "ld1 {v31.4s}, [%[bias]] \n"
      "movi v30.4s, #0  \n"

      // add to out register v25
      "faddp v25.4s, v16.4s, v17.4s  \n"
      "faddp v26.4s, v18.4s, v19.4s  \n"
      "faddp v25.4s, v25.4s, v26.4s  \n"

      // in[24] * w6[0]
      "fmla v25.4s, v21.4s, %[w6].s[0] \n"
      "fadd v25.4s, v25.4s, v31.4s     \n"
      "fmax v25.4s, v25.4s, v30.4s     \n"

      // write output
      "st1 {v25.s}[0], [%[dout0]]  \n"
      "st1 {v25.s}[1], [%[dout1]]  \n"
      "st1 {v25.s}[2], [%[dout2]]  \n"
      "st1 {v25.s}[3], [%[dout3]]  \n"

      : [din0] "+r"(din0), [din1] "+r"(din1), [din2] "+r"(din2),
        [din3] "+r"(din3), [din4] "+r"(din4), [din5] "+r"(din5),
        [din6] "+r"(din6), [din7] "+r"(din7)
      : [dout0] "r"(dout0), [dout1] "r"(dout1), [dout2] "r"(dout2),
        [dout3] "r"(dout3), [w0] "w"(w0), [w1] "w"(w1), [w2] "w"(w2),
        [w3] "w"(w3), [w4] "w"(w4), [w5] "w"(w5), [w6] "w"(w6), [bias] "r"(bias)
      : "memory", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16",
        "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25", "v26",
        "v30", "v31");
}

//! kernel for one out with extracting data pre
//! deal with four lines out
//! need extra load weights
void compute_one_out_extract_pre(const float* din0, const float* din1,
                                 const float* din2, const float* din3,
                                 const float* din4, const float* din5,
                                 const float* din6, const float* din7,
                                 float* dout0, float* dout1, float* dout2,
                                 float* dout3, const float* weights,
                                 const float* bias) {
  //! din0 - din7: 0-4 v8-v15
  //! dout0 - dout3: v16-v19
  //! weights: v0-v4
  asm volatile(
      // load weights
      "add %[wh], %[wh], #4  \n"
      "ldr q0, [%[wh]], #20  \n"
      "ldr q1, [%[wh]], #20  \n"
      "ldr q2, [%[wh]], #20  \n"
      "ldr q3, [%[wh]], #20  \n"
      "ldr q4, [%[wh]], #20  \n"

      "ld1 {v31.4s}, [%[bias]]  \n"
      "ld1 {v8.4s}, [%[din0]], #16   \n"
      "ld1 {v9.4s}, [%[din1]], #16   \n"
      "ld1 {v10.4s}, [%[din2]], #16  \n"
      "ld1 {v11.4s}, [%[din3]], #16  \n"
      "ld1 {v12.4s}, [%[din4]], #16  \n"
      "ld1 {v13.4s}, [%[din5]], #16  \n"

      // in row0
      "fmul v16.4s, v0.4s, v8.4s  \n"
      "fmul v17.4s, v0.4s, v9.4s  \n"
      "fmul v18.4s, v0.4s, v10.4s \n"
      "fmul v19.4s, v0.4s, v11.4s \n"

      "ld1 {v14.4s}, [%[din6]], #16  \n"
      "ld1 {v15.4s}, [%[din7]], #16  \n"

      // in row1
      "fmla v16.4s, v1.4s, v9.4s \n"
      "fmla v17.4s, v1.4s, v10.4s \n"
      "fmla v18.4s, v1.4s, v11.4s \n"
      "fmla v19.4s, v1.4s, v12.4s \n"

      // in row2
      "fmla v16.4s, v2.4s, v10.4s \n"
      "fmla v17.4s, v2.4s, v11.4s \n"
      "fmla v18.4s, v2.4s, v12.4s \n"
      "fmla v19.4s, v2.4s, v13.4s \n"

      // in row3
      "fmla v16.4s, v3.4s, v11.4s \n"
      "fmla v17.4s, v3.4s, v12.4s \n"
      "fmla v18.4s, v3.4s, v13.4s \n"
      "fmla v19.4s, v3.4s, v14.4s \n"

      // in row4
      "fmla v16.4s, v4.4s, v12.4s \n"
      "fmla v17.4s, v4.4s, v13.4s \n"
      "fmla v18.4s, v4.4s, v14.4s \n"
      "fmla v19.4s, v4.4s, v15.4s \n"

      // add to out register v25
      "faddp v25.4s, v16.4s, v17.4s \n"
      "faddp v26.4s, v18.4s, v19.4s \n"
      "faddp v25.4s, v25.4s, v26.4s \n"
      "fadd  v25.4s, v25.4s, v31.4s \n"

      // write output
      "st1 {v25.s}[0], [%[dout0]]  \n"
      "st1 {v25.s}[1], [%[dout1]]  \n"
      "st1 {v25.s}[2], [%[dout2]]  \n"
      "st1 {v25.s}[3], [%[dout3]]  \n"

      : [din0] "+r"(din0), [din1] "+r"(din1), [din2] "+r"(din2),
        [din3] "+r"(din3), [din4] "+r"(din4), [din5] "+r"(din5),
        [din6] "+r"(din6), [din7] "+r"(din7), [wh] "+r"(weights)
      : [dout0] "r"(dout0), [dout1] "r"(dout1), [dout2] "r"(dout2),
        [dout3] "r"(dout3), [bias] "r"(bias)
      : "memory", "v0", "v1", "v2", "v3", "v4", "v8", "v9", "v10", "v11", "v12",
        "v13", "v14", "v15", "v16", "v17", "v18", "v19", "v25", "v26", "v31");
}

//! kernel for one out with extracting data pre
//! deal with four lines out
//! need extra load weights
void compute_one_out_extract_pre_relu(const float* din0, const float* din1,
                                      const float* din2, const float* din3,
                                      const float* din4, const float* din5,
                                      const float* din6, const float* din7,
                                      float* dout0, float* dout1, float* dout2,
                                      float* dout3, const float* weights,
                                      const float* bias) {
  //! din0 - din7: 0-4 v8-v15
  //! dout0 - dout3: v16-v19
  //! weights: v0-v4
  asm volatile(
      // load weights
      "add %[wh], %[wh], #4  \n"
      "ldr q0, [%[wh]], #20  \n"
      "ldr q1, [%[wh]], #20  \n"
      "ldr q2, [%[wh]], #20  \n"
      "ldr q3, [%[wh]], #20  \n"
      "ldr q4, [%[wh]], #20  \n"

      "ld1 {v8.4s}, [%[din0]], #16   \n"
      "ld1 {v9.4s}, [%[din1]], #16   \n"
      "ld1 {v10.4s}, [%[din2]], #16  \n"
      "ld1 {v11.4s}, [%[din3]], #16  \n"
      "ld1 {v12.4s}, [%[din4]], #16  \n"
      "ld1 {v13.4s}, [%[din5]], #16  \n"

      // in row0
      "fmul v16.4s, v0.4s, v8.4s   \n"
      "fmul v17.4s, v0.4s, v9.4s   \n"
      "fmul v18.4s, v0.4s, v10.4s  \n"
      "fmul v19.4s, v0.4s, v11.4s  \n"

      "ld1 {v14.4s}, [%[din6]], #16  \n"
      "ld1 {v15.4s}, [%[din7]], #16  \n"

      // in row1
      "fmla v16.4s, v1.4s, v9.4s  \n"
      "fmla v17.4s, v1.4s, v10.4s \n"
      "fmla v18.4s, v1.4s, v11.4s \n"
      "fmla v19.4s, v1.4s, v12.4s \n"

      // in row2
      "fmla v16.4s, v2.4s, v10.4s \n"
      "fmla v17.4s, v2.4s, v11.4s \n"
      "fmla v18.4s, v2.4s, v12.4s \n"
      "fmla v19.4s, v2.4s, v13.4s \n"

      // in row3
      "fmla v16.4s, v3.4s, v11.4s \n"
      "fmla v17.4s, v3.4s, v12.4s \n"
      "fmla v18.4s, v3.4s, v13.4s \n"
      "fmla v19.4s, v3.4s, v14.4s \n"

      "ld1 {v31.4s}, [%[bias]]  \n"
      "movi v30.4s, #0  \n"

      // in row4
      "fmla v16.4s, v4.4s, v12.4s \n"
      "fmla v17.4s, v4.4s, v13.4s \n"
      "fmla v18.4s, v4.4s, v14.4s \n"
      "fmla v19.4s, v4.4s, v15.4s \n"

      // add to out register v25
      "faddp v25.4s, v16.4s, v17.4s  \n"
      "faddp v26.4s, v18.4s, v19.4s  \n"
      "faddp v25.4s, v25.4s, v26.4s  \n"
      "fadd  v25.4s, v25.4s, v31.4s  \n"
      "fmax  v25.4s, v25.4s, v30.4s  \n"

      // write output
      "st1 {v25.s}[0], [%[dout0]]  \n"
      "st1 {v25.s}[1], [%[dout1]]  \n"
      "st1 {v25.s}[2], [%[dout2]]  \n"
      "st1 {v25.s}[3], [%[dout3]]  \n"

      : [din0] "+r"(din0), [din1] "+r"(din1), [din2] "+r"(din2),
        [din3] "+r"(din3), [din4] "+r"(din4), [din5] "+r"(din5),
        [din6] "+r"(din6), [din7] "+r"(din7), [wh] "+r"(weights)
      : [dout0] "r"(dout0), [dout1] "r"(dout1), [dout2] "r"(dout2),
        [dout3] "r"(dout3), [bias] "r"(bias)
      : "memory", "v0", "v1", "v2", "v3", "v4", "v8", "v9", "v10", "v11", "v12",
        "v13", "v14", "v15", "v16", "v17", "v18", "v19", "v25", "v26", "v30",
        "v31");
}

//! kernel for one out with extracting data post
//! deal with four lines out
void compute_one_out_extract_post(const float* din0, const float* din1,
                                  const float* din2, const float* din3,
                                  const float* din4, const float* din5,
                                  const float* din6, const float* din7,
                                  float* dout0, float* dout1, float* dout2,
                                  float* dout3, float32x4_t w0, float32x4_t w1,
                                  float32x4_t w2, float32x4_t w3,
                                  float32x4_t w4, const float* bias) {
  //! din0 - din7: 0-4 v8-v15
  //! dout0 - dout3: v16-v19
  asm volatile(
      "ld1 {v31.4s}, [%[bias]]  \n"
      "ld1 {v8.4s}, [%[din0]], #16  \n"
      "ld1 {v9.4s}, [%[din1]], #16  \n"
      "ld1 {v10.4s}, [%[din2]], #16  \n"
      "ld1 {v11.4s}, [%[din3]], #16  \n"
      "ld1 {v12.4s}, [%[din4]], #16  \n"
      "ld1 {v13.4s}, [%[din5]], #16  \n"

      // in row0
      "fmul v16.4s, %[w0].4s, v8.4s   \n"
      "fmul v17.4s, %[w0].4s, v9.4s   \n"
      "fmul v18.4s, %[w0].4s, v10.4s  \n"
      "fmul v19.4s, %[w0].4s, v11.4s  \n"

      "ld1 {v14.4s}, [%[din6]], #16  \n"
      "ld1 {v15.4s}, [%[din7]], #16  \n"

      // in row1
      "fmla v16.4s, %[w1].4s, v9.4s  \n"
      "fmla v17.4s, %[w1].4s, v10.4s \n"
      "fmla v18.4s, %[w1].4s, v11.4s \n"
      "fmla v19.4s, %[w1].4s, v12.4s \n"

      // in row2
      "fmla v16.4s, %[w2].4s, v10.4s \n"
      "fmla v17.4s, %[w2].4s, v11.4s \n"
      "fmla v18.4s, %[w2].4s, v12.4s \n"
      "fmla v19.4s, %[w2].4s, v13.4s \n"

      // in row3
      "fmla v16.4s, %[w3].4s, v11.4s \n"
      "fmla v17.4s, %[w3].4s, v12.4s \n"
      "fmla v18.4s, %[w3].4s, v13.4s \n"
      "fmla v19.4s, %[w3].4s, v14.4s \n"

      // in row4
      "fmla v16.4s, %[w4].4s, v12.4s \n"
      "fmla v17.4s, %[w4].4s, v13.4s \n"
      "fmla v18.4s, %[w4].4s, v14.4s \n"
      "fmla v19.4s, %[w4].4s, v15.4s \n"

      // add to out register v25
      "faddp v25.4s, v16.4s, v17.4s \n"
      "faddp v26.4s, v18.4s, v19.4s \n"
      "faddp v25.4s, v25.4s, v26.4s \n"
      "fadd v25.4s, v25.4s, v31.4s  \n"

      // write output
      "st1 {v25.s}[0], [%[dout0]]  \n"
      "st1 {v25.s}[1], [%[dout1]]  \n"
      "st1 {v25.s}[2], [%[dout2]]  \n"
      "st1 {v25.s}[3], [%[dout3]]  \n"

      : [din0] "+r"(din0), [din1] "+r"(din1), [din2] "+r"(din2),
        [din3] "+r"(din3), [din4] "+r"(din4), [din5] "+r"(din5),
        [din6] "+r"(din6), [din7] "+r"(din7)
      : [dout0] "r"(dout0), [dout1] "r"(dout1), [dout2] "r"(dout2),
        [dout3] "r"(dout3), [w0] "w"(w0), [w1] "w"(w1), [w2] "w"(w2),
        [w3] "w"(w3), [w4] "w"(w4), [bias] "r"(bias)
      : "memory", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16",
        "v17", "v18", "v19", "v25", "v26", "v31");
}

//! kernel for one out with extracting data post
//! deal with four lines out
void compute_one_out_extract_post_relu(
    const float* din0, const float* din1, const float* din2, const float* din3,
    const float* din4, const float* din5, const float* din6, const float* din7,
    float* dout0, float* dout1, float* dout2, float* dout3, float32x4_t w0,
    float32x4_t w1, float32x4_t w2, float32x4_t w3, float32x4_t w4,
    const float* bias) {
  //! din0 - din7: 0-4 v8-v15
  //! dout0 - dout3: v16-v19
  asm volatile(
      "ld1 {v8.4s}, [%[din0]], #16  \n"
      "ld1 {v9.4s}, [%[din1]], #16  \n"
      "ld1 {v10.4s}, [%[din2]], #16  \n"
      "ld1 {v11.4s}, [%[din3]], #16  \n"
      "ld1 {v12.4s}, [%[din4]], #16  \n"
      "ld1 {v13.4s}, [%[din5]], #16  \n"

      // in row0
      "fmul v16.4s, %[w0].4s, v8.4s  \n"
      "fmul v17.4s, %[w0].4s, v9.4s  \n"
      "fmul v18.4s, %[w0].4s, v10.4s  \n"
      "fmul v19.4s, %[w0].4s, v11.4s  \n"

      "ld1 {v14.4s}, [%[din6]], #16  \n"
      "ld1 {v15.4s}, [%[din7]], #16  \n"

      // in row1
      "fmla v16.4s, %[w1].4s, v9.4s \n"
      "fmla v17.4s, %[w1].4s, v10.4s \n"
      "fmla v18.4s, %[w1].4s, v11.4s \n"
      "fmla v19.4s, %[w1].4s, v12.4s \n"

      // in row2
      "fmla v16.4s, %[w2].4s, v10.4s \n"
      "fmla v17.4s, %[w2].4s, v11.4s \n"
      "fmla v18.4s, %[w2].4s, v12.4s \n"
      "fmla v19.4s, %[w2].4s, v13.4s \n"

      // in row3
      "fmla v16.4s, %[w3].4s, v11.4s \n"
      "fmla v17.4s, %[w3].4s, v12.4s \n"
      "fmla v18.4s, %[w3].4s, v13.4s \n"
      "fmla v19.4s, %[w3].4s, v14.4s \n"

      "ld1 {v31.4s}, [%[bias]]  \n"
      "movi v30.4s, #0  \n"

      // in row4
      "fmla v16.4s, %[w4].4s, v12.4s \n"
      "fmla v17.4s, %[w4].4s, v13.4s \n"
      "fmla v18.4s, %[w4].4s, v14.4s \n"
      "fmla v19.4s, %[w4].4s, v15.4s \n"

      // add to out register v25
      "faddp v25.4s, v16.4s, v17.4s \n"
      "faddp v26.4s, v18.4s, v19.4s \n"
      "faddp v25.4s, v25.4s, v26.4s \n"
      "fadd v25.4s, v25.4s, v31.4s  \n"
      "fmax v25.4s, v25.4s, v30.4s  \n"

      // write output
      "st1 {v25.s}[0], [%[dout0]]  \n"
      "st1 {v25.s}[1], [%[dout1]]  \n"
      "st1 {v25.s}[2], [%[dout2]]  \n"
      "st1 {v25.s}[3], [%[dout3]]  \n"

      : [din0] "+r"(din0), [din1] "+r"(din1), [din2] "+r"(din2),
        [din3] "+r"(din3), [din4] "+r"(din4), [din5] "+r"(din5),
        [din6] "+r"(din6), [din7] "+r"(din7)
      : [dout0] "r"(dout0), [dout1] "r"(dout1), [dout2] "r"(dout2),
        [dout3] "r"(dout3), [w0] "w"(w0), [w1] "w"(w1), [w2] "w"(w2),
        [w3] "w"(w3), [w4] "w"(w4), [bias] "r"(bias)
      : "memory", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16",
        "v17", "v18", "v19", "v25", "v26", "v30", "v31");
}

//! kernel for two out with extracting data pre
//! deal with four lines out
//! need extra load weights
void compute_two_out_extract_pre(const float* din0, const float* din1,
                                 const float* din2, const float* din3,
                                 const float* din4, const float* din5,
                                 const float* din6, const float* din7,
                                 float* dout0, float* dout1, float* dout2,
                                 float* dout3, const float* weights,
                                 const float* bias) {
  //! din0 - din7: 0-4 v8-v15
  //! dout0 - dout3: v16-v19
  //! weights: v0-v4
  asm volatile(
      // load weights
      "movi v31.4s, #0  \n"
      "add %[wh], %[wh], #4  \n"
      "ldr q0, [%[wh]], #20  \n"  // 1, 2, 3, 4
      "ldr q1, [%[wh]], #20  \n"  // 6, 7, 8, 9
      "ldr q2, [%[wh]], #20  \n"  // 11, 12, 13, 14
      "ldr q3, [%[wh]], #20  \n"  // 16, 17, 18, 19
      "ldr q4, [%[wh]], #20  \n"  // 21, 22, 23, 24

      // load inputs
      "ld1 {v20.4s}, [%[bias]]  \n"
      "ld1 {v8.4s}, [%[din0]], #16   \n"
      "ld1 {v9.4s}, [%[din1]], #16   \n"
      "ld1 {v10.4s}, [%[din2]], #16  \n"
      "ld1 {v11.4s}, [%[din3]], #16  \n"
      "ld1 {v12.4s}, [%[din4]], #16  \n"
      "ld1 {v13.4s}, [%[din5]], #16  \n"

      // in row0
      "fmul v16.4s, v0.4s, v8.4s   \n"
      "fmul v17.4s, v0.4s, v9.4s   \n"
      "fmul v18.4s, v0.4s, v10.4s  \n"
      "fmul v19.4s, v0.4s, v11.4s  \n"

      "ld1 {v14.4s}, [%[din6]], #16  \n"
      "ld1 {v15.4s}, [%[din7]], #16  \n"

      // in row1
      "fmla v16.4s, v1.4s, v9.4s  \n"
      "fmla v17.4s, v1.4s, v10.4s \n"
      "fmla v18.4s, v1.4s, v11.4s \n"
      "fmla v19.4s, v1.4s, v12.4s \n"

      // in row2
      "fmla v16.4s, v2.4s, v10.4s \n"
      "fmla v17.4s, v2.4s, v11.4s \n"
      "fmla v18.4s, v2.4s, v12.4s \n"
      "fmla v19.4s, v2.4s, v13.4s \n"

      // in row3
      "fmla v16.4s, v3.4s, v11.4s \n"
      "fmla v17.4s, v3.4s, v12.4s \n"
      "fmla v18.4s, v3.4s, v13.4s \n"
      "fmla v19.4s, v3.4s, v14.4s \n"

      // in row4
      "fmla v16.4s, v4.4s, v12.4s \n"
      "fmla v17.4s, v4.4s, v13.4s \n"
      "fmla v18.4s, v4.4s, v14.4s \n"
      "fmla v19.4s, v4.4s, v15.4s \n"

      // add to out register v5
      "faddp v5.4s, v16.4s, v17.4s \n"
      "faddp v6.4s, v18.4s, v19.4s \n"
      "faddp v5.4s, v5.4s, v6.4s   \n"

      // ext weights
      "ext v0.16b, v0.16b, v31.16b, #4  \n"  // 2, 3, 4
      "ext v1.16b, v1.16b, v31.16b, #4  \n"  // 7, 8, 9
      "ext v2.16b, v2.16b, v31.16b, #4  \n"  // 12, 13, 14
      "ext v3.16b, v3.16b, v31.16b, #4  \n"  // 17, 18, 19
      "ext v4.16b, v4.16b, v31.16b, #4  \n"  // 22, 23, 24

      // in row0
      "fmul v16.4s, v0.4s, v8.4s   \n"
      "fmul v17.4s, v0.4s, v9.4s   \n"
      "fmul v18.4s, v0.4s, v10.4s  \n"
      "fmul v19.4s, v0.4s, v11.4s  \n"

      // in row1
      "fmla v16.4s, v1.4s, v9.4s  \n"
      "fmla v17.4s, v1.4s, v10.4s \n"
      "fmla v18.4s, v1.4s, v11.4s \n"
      "fmla v19.4s, v1.4s, v12.4s \n"

      // in row2
      "fmla v16.4s, v2.4s, v10.4s \n"
      "fmla v17.4s, v2.4s, v11.4s \n"
      "fmla v18.4s, v2.4s, v12.4s \n"
      "fmla v19.4s, v2.4s, v13.4s \n"

      // in row3
      "fmla v16.4s, v3.4s, v11.4s \n"
      "fmla v17.4s, v3.4s, v12.4s \n"
      "fmla v18.4s, v3.4s, v13.4s \n"
      "fmla v19.4s, v3.4s, v14.4s \n"

      // in row4
      "fmla v16.4s, v4.4s, v12.4s \n"
      "fmla v17.4s, v4.4s, v13.4s \n"
      "fmla v18.4s, v4.4s, v14.4s \n"
      "fmla v19.4s, v4.4s, v15.4s \n"

      // add to out register v7
      "faddp v7.4s, v16.4s, v17.4s \n"
      "faddp v8.4s, v18.4s, v19.4s \n"
      "faddp v7.4s, v7.4s, v8.4s   \n"

      // zip
      "zip1 v6.4s, v7.4s, v5.4s  \n"
      "zip2 v8.4s, v7.4s, v5.4s  \n"
      "fadd v6.4s, v6.4s, v20.4s \n"
      "fadd v8.4s, v8.4s, v20.4s \n"
      "ext v7.16b, v6.16b, v31.16b, #8  \n"
      "ext v9.16b, v8.16b, v31.16b, #8  \n"

      // write output
      "str d6, [%[dout0]]  \n"
      "str d7, [%[dout1]]  \n"
      "str d8, [%[dout2]]  \n"
      "str d9, [%[dout3]]  \n"

      : [din0] "+r"(din0), [din1] "+r"(din1), [din2] "+r"(din2),
        [din3] "+r"(din3), [din4] "+r"(din4), [din5] "+r"(din5),
        [din6] "+r"(din6), [din7] "+r"(din7), [wh] "+r"(weights)
      : [dout0] "r"(dout0), [dout1] "r"(dout1), [dout2] "r"(dout2),
        [dout3] "r"(dout3), [bias] "r"(bias)
      : "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9",
        "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19",
        "v20", "v31");
}

//! kernel for two out with extracting data pre
//! deal with four lines out
//! need extra load weights
void compute_two_out_extract_pre_relu(const float* din0, const float* din1,
                                      const float* din2, const float* din3,
                                      const float* din4, const float* din5,
                                      const float* din6, const float* din7,
                                      float* dout0, float* dout1, float* dout2,
                                      float* dout3, const float* weights,
                                      const float* bias) {
  //! din0 - din7: 0-4 v8-v15
  //! dout0 - dout3: v16-v19
  //! weights: v0-v4
  asm volatile(
      // load weights
      "movi v31.4s, #0  \n"
      "add %[wh], %[wh], #4  \n"
      "ldr q0, [%[wh]], #20  \n"  // 1, 2, 3, 4
      "ldr q1, [%[wh]], #20  \n"  // 6, 7, 8, 9
      "ldr q2, [%[wh]], #20  \n"  // 11, 12, 13, 14
      "ldr q3, [%[wh]], #20  \n"  // 16, 17, 18, 19
      "ldr q4, [%[wh]], #20  \n"  // 21, 22, 23, 24

      // load inputs
      "ld1 {v20.4s}, [%[bias]]      \n"
      "ld1 {v8.4s}, [%[din0]], #16  \n"
      "ld1 {v9.4s}, [%[din1]], #16  \n"
      "ld1 {v10.4s}, [%[din2]], #16  \n"
      "ld1 {v11.4s}, [%[din3]], #16  \n"
      "ld1 {v12.4s}, [%[din4]], #16  \n"
      "ld1 {v13.4s}, [%[din5]], #16  \n"

      // in row0
      "fmul v16.4s, v0.4s, v8.4s   \n"
      "fmul v17.4s, v0.4s, v9.4s   \n"
      "fmul v18.4s, v0.4s, v10.4s  \n"
      "fmul v19.4s, v0.4s, v11.4s  \n"

      "ld1 {v14.4s}, [%[din6]], #16  \n"
      "ld1 {v15.4s}, [%[din7]], #16  \n"

      // in row1
      "fmla v16.4s, v1.4s, v9.4s  \n"
      "fmla v17.4s, v1.4s, v10.4s \n"
      "fmla v18.4s, v1.4s, v11.4s \n"
      "fmla v19.4s, v1.4s, v12.4s \n"

      // in row2
      "fmla v16.4s, v2.4s, v10.4s \n"
      "fmla v17.4s, v2.4s, v11.4s \n"
      "fmla v18.4s, v2.4s, v12.4s \n"
      "fmla v19.4s, v2.4s, v13.4s \n"

      // in row3
      "fmla v16.4s, v3.4s, v11.4s \n"
      "fmla v17.4s, v3.4s, v12.4s \n"
      "fmla v18.4s, v3.4s, v13.4s \n"
      "fmla v19.4s, v3.4s, v14.4s \n"

      // in row4
      "fmla v16.4s, v4.4s, v12.4s \n"
      "fmla v17.4s, v4.4s, v13.4s \n"
      "fmla v18.4s, v4.4s, v14.4s \n"
      "fmla v19.4s, v4.4s, v15.4s \n"

      // add to out register v5
      "faddp v5.4s, v16.4s, v17.4s \n"
      "faddp v6.4s, v18.4s, v19.4s \n"
      "faddp v5.4s, v5.4s, v6.4s   \n"

      // ext weights
      "ext v0.16b, v0.16b, v31.16b, #4  \n"  // 2, 3, 4
      "ext v1.16b, v1.16b, v31.16b, #4  \n"  // 7, 8, 9
      "ext v2.16b, v2.16b, v31.16b, #4  \n"  // 12, 13, 14
      "ext v3.16b, v3.16b, v31.16b, #4  \n"  // 17, 18, 19
      "ext v4.16b, v4.16b, v31.16b, #4  \n"  // 22, 23, 24

      // in row0
      "fmul v16.4s, v0.4s, v8.4s  \n"
      "fmul v17.4s, v0.4s, v9.4s  \n"
      "fmul v18.4s, v0.4s, v10.4s  \n"
      "fmul v19.4s, v0.4s, v11.4s  \n"

      // in row1
      "fmla v16.4s, v1.4s, v9.4s  \n"
      "fmla v17.4s, v1.4s, v10.4s \n"
      "fmla v18.4s, v1.4s, v11.4s \n"
      "fmla v19.4s, v1.4s, v12.4s \n"

      // in row2
      "fmla v16.4s, v2.4s, v10.4s \n"
      "fmla v17.4s, v2.4s, v11.4s \n"
      "fmla v18.4s, v2.4s, v12.4s \n"
      "fmla v19.4s, v2.4s, v13.4s \n"

      // in row3
      "fmla v16.4s, v3.4s, v11.4s \n"
      "fmla v17.4s, v3.4s, v12.4s \n"
      "fmla v18.4s, v3.4s, v13.4s \n"
      "fmla v19.4s, v3.4s, v14.4s \n"

      // in row4
      "fmla v16.4s, v4.4s, v12.4s \n"
      "fmla v17.4s, v4.4s, v13.4s \n"
      "fmla v18.4s, v4.4s, v14.4s \n"
      "fmla v19.4s, v4.4s, v15.4s \n"

      // add to out register v7
      "faddp v7.4s, v16.4s, v17.4s \n"
      "faddp v8.4s, v18.4s, v19.4s \n"
      "faddp v7.4s, v7.4s, v8.4s   \n"

      // zip
      "zip1 v6.4s, v7.4s, v5.4s  \n"
      "zip2 v8.4s, v7.4s, v5.4s  \n"

      // add bias
      "fadd v6.4s, v6.4s, v20.4s \n"
      "fadd v8.4s, v8.4s, v20.4s \n"

      // relu
      "fmax v6.4s, v6.4s, v31.4s \n"
      "fmax v8.4s, v8.4s, v31.4s \n"

      "ext v7.16b, v6.16b, v31.16b, #8  \n"
      "ext v9.16b, v8.16b, v31.16b, #8  \n"

      // write output
      "str d6, [%[dout0]]  \n"
      "str d7, [%[dout1]]  \n"
      "str d8, [%[dout2]]  \n"
      "str d9, [%[dout3]]  \n"

      : [din0] "+r"(din0), [din1] "+r"(din1), [din2] "+r"(din2),
        [din3] "+r"(din3), [din4] "+r"(din4), [din5] "+r"(din5),
        [din6] "+r"(din6), [din7] "+r"(din7), [wh] "+r"(weights)
      : [dout0] "r"(dout0), [dout1] "r"(dout1), [dout2] "r"(dout2),
        [dout3] "r"(dout3), [bias] "r"(bias)
      : "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9",
        "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19",
        "v20", "v31");
}

//! kernel for two out with extracting data post
//! deal with four lines out
void compute_two_out_extract_post(const float* din0, const float* din1,
                                  const float* din2, const float* din3,
                                  const float* din4, const float* din5,
                                  const float* din6, const float* din7,
                                  float* dout0, float* dout1, float* dout2,
                                  float* dout3, float32x4_t w0, float32x4_t w1,
                                  float32x4_t w2, float32x4_t w3,
                                  float32x4_t w4, const float* bias) {
  //! din0 - din7: 0-4 v8-v15
  //! dout0 - dout3: v16-v19
  asm volatile(
      "movi v31.4s, #0  \n"

      // load inputs
      "ld1 {v20.4s}, [%[bias]]  \n"
      "ld1 {v8.4s}, [%[din0]], #16  \n"
      "ld1 {v9.4s}, [%[din1]], #16  \n"
      "ld1 {v10.4s}, [%[din2]], #16  \n"
      "ld1 {v11.4s}, [%[din3]], #16  \n"
      "ld1 {v12.4s}, [%[din4]], #16  \n"
      "ld1 {v13.4s}, [%[din5]], #16  \n"

      // in row0
      "fmul v16.4s, %[w0].4s, v8.4s   \n"
      "fmul v17.4s, %[w0].4s, v9.4s   \n"
      "fmul v18.4s, %[w0].4s, v10.4s  \n"
      "fmul v19.4s, %[w0].4s, v11.4s  \n"

      "ld1 {v14.4s}, [%[din6]], #16  \n"
      "ld1 {v15.4s}, [%[din7]], #16  \n"

      // in row1
      "fmla v16.4s, %[w1].4s, v9.4s  \n"
      "fmla v17.4s, %[w1].4s, v10.4s \n"
      "fmla v18.4s, %[w1].4s, v11.4s \n"
      "fmla v19.4s, %[w1].4s, v12.4s \n"

      // in row2
      "fmla v16.4s, %[w2].4s, v10.4s \n"
      "fmla v17.4s, %[w2].4s, v11.4s \n"
      "fmla v18.4s, %[w2].4s, v12.4s \n"
      "fmla v19.4s, %[w2].4s, v13.4s \n"

      // in row3
      "fmla v16.4s, %[w3].4s, v11.4s \n"
      "fmla v17.4s, %[w3].4s, v12.4s \n"
      "fmla v18.4s, %[w3].4s, v13.4s \n"
      "fmla v19.4s, %[w3].4s, v14.4s \n"

      // in row4
      "fmla v16.4s, %[w4].4s, v12.4s \n"
      "fmla v17.4s, %[w4].4s, v13.4s \n"
      "fmla v18.4s, %[w4].4s, v14.4s \n"
      "fmla v19.4s, %[w4].4s, v15.4s \n"

      // add to out register v5
      "faddp v5.4s, v16.4s, v17.4s \n"
      "faddp v6.4s, v18.4s, v19.4s \n"
      "faddp v5.4s, v5.4s, v6.4s   \n"

      // ext input
      "ext v8.16b, v8.16b, v31.16b, #4    \n"
      "ext v9.16b, v9.16b, v31.16b, #4    \n"
      "ext v10.16b, v10.16b, v31.16b, #4  \n"
      "ext v11.16b, v11.16b, v31.16b, #4  \n"
      "ext v12.16b, v12.16b, v31.16b, #4  \n"
      "ext v13.16b, v13.16b, v31.16b, #4  \n"
      "ext v14.16b, v14.16b, v31.16b, #4  \n"
      "ext v15.16b, v15.16b, v31.16b, #4  \n"

      // in row0
      "fmul v16.4s, %[w0].4s, v8.4s   \n"
      "fmul v17.4s, %[w0].4s, v9.4s   \n"
      "fmul v18.4s, %[w0].4s, v10.4s  \n"
      "fmul v19.4s, %[w0].4s, v11.4s  \n"

      // in row1
      "fmla v16.4s, %[w1].4s, v9.4s  \n"
      "fmla v17.4s, %[w1].4s, v10.4s \n"
      "fmla v18.4s, %[w1].4s, v11.4s \n"
      "fmla v19.4s, %[w1].4s, v12.4s \n"

      // in row2
      "fmla v16.4s, %[w2].4s, v10.4s \n"
      "fmla v17.4s, %[w2].4s, v11.4s \n"
      "fmla v18.4s, %[w2].4s, v12.4s \n"
      "fmla v19.4s, %[w2].4s, v13.4s \n"

      // in row3
      "fmla v16.4s, %[w3].4s, v11.4s \n"
      "fmla v17.4s, %[w3].4s, v12.4s \n"
      "fmla v18.4s, %[w3].4s, v13.4s \n"
      "fmla v19.4s, %[w3].4s, v14.4s \n"

      // in row4
      "fmla v16.4s, %[w4].4s, v12.4s \n"
      "fmla v17.4s, %[w4].4s, v13.4s \n"
      "fmla v18.4s, %[w4].4s, v14.4s \n"
      "fmla v19.4s, %[w4].4s, v15.4s \n"

      // add to out register v7
      "faddp v7.4s, v16.4s, v17.4s \n"
      "faddp v8.4s, v18.4s, v19.4s \n"
      "faddp v7.4s, v7.4s, v8.4s   \n"

      // zip
      "zip1 v6.4s, v5.4s, v7.4s  \n"
      "zip2 v8.4s, v5.4s, v7.4s  \n"
      "fadd v6.4s, v6.4s, v20.4s \n"
      "fadd v8.4s, v8.4s, v20.4s  \n"
      "ext v7.16b, v6.16b, v31.16b, #8  \n"
      "ext v9.16b, v8.16b, v31.16b, #8  \n"

      // write output
      "str d6, [%[dout0]]  \n"
      "str d7, [%[dout1]]  \n"
      "str d8, [%[dout2]]  \n"
      "str d9, [%[dout3]]  \n"

      : [din0] "+r"(din0), [din1] "+r"(din1), [din2] "+r"(din2),
        [din3] "+r"(din3), [din4] "+r"(din4), [din5] "+r"(din5),
        [din6] "+r"(din6), [din7] "+r"(din7)
      : [dout0] "r"(dout0), [dout1] "r"(dout1), [dout2] "r"(dout2),
        [dout3] "r"(dout3), [w0] "w"(w0), [w1] "w"(w1), [w2] "w"(w2),
        [w3] "w"(w3), [w4] "w"(w4), [bias] "r"(bias)
      : "memory", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13",
        "v14", "v15", "v16", "v17", "v18", "v19", "v20", "v31");
}

//! kernel for two out with extracting data post
//! deal with four lines out
void compute_two_out_extract_post_relu(
    const float* din0, const float* din1, const float* din2, const float* din3,
    const float* din4, const float* din5, const float* din6, const float* din7,
    float* dout0, float* dout1, float* dout2, float* dout3, float32x4_t w0,
    float32x4_t w1, float32x4_t w2, float32x4_t w3, float32x4_t w4,
    const float* bias) {
  //! din0 - din7: 0-4 v8-v15
  //! dout0 - dout3: v16-v19
  asm volatile(
      "movi v31.4s, #0  \n"

      // load inputs
      "ld1 {v20.4s}, [%[bias]]  \n"
      "ld1 {v8.4s}, [%[din0]], #16   \n"
      "ld1 {v9.4s}, [%[din1]], #16   \n"
      "ld1 {v10.4s}, [%[din2]], #16  \n"
      "ld1 {v11.4s}, [%[din3]], #16  \n"
      "ld1 {v12.4s}, [%[din4]], #16  \n"
      "ld1 {v13.4s}, [%[din5]], #16  \n"

      // in row0
      "fmul v16.4s, %[w0].4s, v8.4s   \n"
      "fmul v17.4s, %[w0].4s, v9.4s   \n"
      "fmul v18.4s, %[w0].4s, v10.4s  \n"
      "fmul v19.4s, %[w0].4s, v11.4s  \n"

      "ld1 {v14.4s}, [%[din6]], #16  \n"
      "ld1 {v15.4s}, [%[din7]], #16  \n"

      // in row1
      "fmla v16.4s, %[w1].4s, v9.4s  \n"
      "fmla v17.4s, %[w1].4s, v10.4s \n"
      "fmla v18.4s, %[w1].4s, v11.4s \n"
      "fmla v19.4s, %[w1].4s, v12.4s \n"

      // in row2
      "fmla v16.4s, %[w2].4s, v10.4s \n"
      "fmla v17.4s, %[w2].4s, v11.4s \n"
      "fmla v18.4s, %[w2].4s, v12.4s \n"
      "fmla v19.4s, %[w2].4s, v13.4s \n"

      // in row3
      "fmla v16.4s, %[w3].4s, v11.4s \n"
      "fmla v17.4s, %[w3].4s, v12.4s \n"
      "fmla v18.4s, %[w3].4s, v13.4s \n"
      "fmla v19.4s, %[w3].4s, v14.4s \n"

      // in row4
      "fmla v16.4s, %[w4].4s, v12.4s \n"
      "fmla v17.4s, %[w4].4s, v13.4s \n"
      "fmla v18.4s, %[w4].4s, v14.4s \n"
      "fmla v19.4s, %[w4].4s, v15.4s \n"

      // add to out register v5
      "faddp v5.4s, v16.4s, v17.4s \n"
      "faddp v6.4s, v18.4s, v19.4s \n"
      "faddp v5.4s, v5.4s, v6.4s \n"

      // ext input
      "ext v8.16b, v8.16b, v31.16b, #4    \n"
      "ext v9.16b, v9.16b, v31.16b, #4    \n"
      "ext v10.16b, v10.16b, v31.16b, #4  \n"
      "ext v11.16b, v11.16b, v31.16b, #4  \n"
      "ext v12.16b, v12.16b, v31.16b, #4  \n"
      "ext v13.16b, v13.16b, v31.16b, #4  \n"
      "ext v14.16b, v14.16b, v31.16b, #4  \n"
      "ext v15.16b, v15.16b, v31.16b, #4  \n"

      // in row0
      "fmul v16.4s, %[w0].4s, v8.4s   \n"
      "fmul v17.4s, %[w0].4s, v9.4s   \n"
      "fmul v18.4s, %[w0].4s, v10.4s  \n"
      "fmul v19.4s, %[w0].4s, v11.4s  \n"

      // in row1
      "fmla v16.4s, %[w1].4s, v9.4s  \n"
      "fmla v17.4s, %[w1].4s, v10.4s \n"
      "fmla v18.4s, %[w1].4s, v11.4s \n"
      "fmla v19.4s, %[w1].4s, v12.4s \n"

      // in row2
      "fmla v16.4s, %[w2].4s, v10.4s \n"
      "fmla v17.4s, %[w2].4s, v11.4s \n"
      "fmla v18.4s, %[w2].4s, v12.4s \n"
      "fmla v19.4s, %[w2].4s, v13.4s \n"

      // in row3
      "fmla v16.4s, %[w3].4s, v11.4s \n"
      "fmla v17.4s, %[w3].4s, v12.4s \n"
      "fmla v18.4s, %[w3].4s, v13.4s \n"
      "fmla v19.4s, %[w3].4s, v14.4s \n"

      // in row4
      "fmla v16.4s, %[w4].4s, v12.4s \n"
      "fmla v17.4s, %[w4].4s, v13.4s \n"
      "fmla v18.4s, %[w4].4s, v14.4s \n"
      "fmla v19.4s, %[w4].4s, v15.4s \n"

      // add to out register v7
      "faddp v7.4s, v16.4s, v17.4s \n"
      "faddp v8.4s, v18.4s, v19.4s \n"
      "faddp v7.4s, v7.4s, v8.4s \n"

      // zip
      "zip1 v6.4s, v5.4s, v7.4s  \n"
      "zip2 v8.4s, v5.4s, v7.4s  \n"

      // add bias
      "fadd v6.4s, v6.4s, v20.4s \n"
      "fadd v8.4s, v8.4s, v20.4s \n"

      // relu
      "fmax v6.4s, v6.4s, v31.4s  \n"
      "fmax v8.4s, v8.4s, v31.4s  \n"
      "ext v7.16b, v6.16b, v31.16b, #8  \n"
      "ext v9.16b, v8.16b, v31.16b, #8  \n"

      // write output
      "str d6, [%[dout0]]  \n"
      "str d7, [%[dout1]]  \n"
      "str d8, [%[dout2]]  \n"
      "str d9, [%[dout3]]  \n"

      : [din0] "+r"(din0), [din1] "+r"(din1), [din2] "+r"(din2),
        [din3] "+r"(din3), [din4] "+r"(din4), [din5] "+r"(din5),
        [din6] "+r"(din6), [din7] "+r"(din7)
      : [dout0] "r"(dout0), [dout1] "r"(dout1), [dout2] "r"(dout2),
        [dout3] "r"(dout3), [w0] "w"(w0), [w1] "w"(w1), [w2] "w"(w2),
        [w3] "w"(w3), [w4] "w"(w4), [bias] "r"(bias)
      : "memory", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13",
        "v14", "v15", "v16", "v17", "v18", "v19", "v20", "v31");
}

//! kernel for three out with extracting data pre
//! deal with four lines out
//! need extra load weights
void compute_three_out_extract_pre(const float* din0, const float* din1,
                                   const float* din2, const float* din3,
                                   const float* din4, const float* din5,
                                   const float* din6, const float* din7,
                                   float* dout0, float* dout1, float* dout2,
                                   float* dout3, const float* weights,
                                   const float* bias) {
  //! din0 - din7: 0-4 v8-v15
  //! dout0 - dout3: v16-v19
  //! weights: v0-v4
  asm volatile(
      // load weights
      "movi v31.4s, #0  \n"
      "add %[wh], %[wh], #4  \n"
      "ldr q0, [%[wh]], #20  \n"  // 1, 2, 3, 4
      "ldr q1, [%[wh]], #20  \n"  // 6, 7, 8, 9
      "ldr q2, [%[wh]], #20  \n"  // 11, 12, 13, 14
      "ldr q3, [%[wh]], #20  \n"  // 16, 17, 18, 19
      "ldr q4, [%[wh]], #20  \n"  // 21, 22, 23, 24

      // load inputs
      "ld1 {v20.4s}, [%[bias]]  \n"
      "ld1 {v8.4s}, [%[din0]], #16  \n"
      "ld1 {v9.4s}, [%[din1]], #16  \n"
      "ld1 {v10.4s}, [%[din2]], #16  \n"
      "ld1 {v11.4s}, [%[din3]], #16  \n"
      "ld1 {v12.4s}, [%[din4]], #16  \n"
      "ld1 {v13.4s}, [%[din5]], #16  \n"

      // in row0
      "fmul v16.4s, v0.4s, v8.4s   \n"
      "fmul v17.4s, v0.4s, v9.4s   \n"
      "fmul v18.4s, v0.4s, v10.4s  \n"
      "fmul v19.4s, v0.4s, v11.4s  \n"

      "ld1 {v14.4s}, [%[din6]], #16  \n"
      "ld1 {v15.4s}, [%[din7]], #16  \n"

      // in row1
      "fmla v16.4s, v1.4s, v9.4s  \n"
      "fmla v17.4s, v1.4s, v10.4s \n"
      "fmla v18.4s, v1.4s, v11.4s \n"
      "fmla v19.4s, v1.4s, v12.4s \n"

      // in row2
      "fmla v16.4s, v2.4s, v10.4s \n"
      "fmla v17.4s, v2.4s, v11.4s \n"
      "fmla v18.4s, v2.4s, v12.4s \n"
      "fmla v19.4s, v2.4s, v13.4s \n"

      // in row3
      "fmla v16.4s, v3.4s, v11.4s \n"
      "fmla v17.4s, v3.4s, v12.4s \n"
      "fmla v18.4s, v3.4s, v13.4s \n"
      "fmla v19.4s, v3.4s, v14.4s \n"

      // in row4
      "fmla v16.4s, v4.4s, v12.4s \n"
      "fmla v17.4s, v4.4s, v13.4s \n"
      "fmla v18.4s, v4.4s, v14.4s \n"
      "fmla v19.4s, v4.4s, v15.4s \n"

      // add to out register v5
      "faddp v5.4s, v16.4s, v17.4s \n"
      "faddp v6.4s, v18.4s, v19.4s \n"
      "faddp v5.4s, v5.4s, v6.4s \n"

      // ext weights
      "ext v0.16b, v0.16b, v31.16b, #4  \n"  // 2, 3, 4
      "ext v1.16b, v1.16b, v31.16b, #4  \n"  // 7, 8, 9
      "ext v2.16b, v2.16b, v31.16b, #4  \n"  // 12, 13, 14
      "ext v3.16b, v3.16b, v31.16b, #4  \n"  // 17, 18, 19
      "ext v4.16b, v4.16b, v31.16b, #4  \n"  // 22, 23, 24

      // in row0
      "fmul v16.4s, v0.4s, v8.4s   \n"
      "fmul v17.4s, v0.4s, v9.4s   \n"
      "fmul v18.4s, v0.4s, v10.4s  \n"
      "fmul v19.4s, v0.4s, v11.4s  \n"

      // in row1
      "fmla v16.4s, v1.4s, v9.4s  \n"
      "fmla v17.4s, v1.4s, v10.4s \n"
      "fmla v18.4s, v1.4s, v11.4s \n"
      "fmla v19.4s, v1.4s, v12.4s \n"

      // in row2
      "fmla v16.4s, v2.4s, v10.4s \n"
      "fmla v17.4s, v2.4s, v11.4s \n"
      "fmla v18.4s, v2.4s, v12.4s \n"
      "fmla v19.4s, v2.4s, v13.4s \n"

      // in row3
      "fmla v16.4s, v3.4s, v11.4s \n"
      "fmla v17.4s, v3.4s, v12.4s \n"
      "fmla v18.4s, v3.4s, v13.4s \n"
      "fmla v19.4s, v3.4s, v14.4s \n"

      // in row4
      "fmla v16.4s, v4.4s, v12.4s \n"
      "fmla v17.4s, v4.4s, v13.4s \n"
      "fmla v18.4s, v4.4s, v14.4s \n"
      "fmla v19.4s, v4.4s, v15.4s \n"

      // add to out register v7
      "faddp v7.4s, v16.4s, v17.4s \n"
      "faddp v6.4s, v18.4s, v19.4s \n"
      "faddp v7.4s, v7.4s, v6.4s \n"

      // ext weights
      "ext v0.16b, v0.16b, v31.16b, #4  \n"  // 3, 4
      "ext v1.16b, v1.16b, v31.16b, #4  \n"  // 8, 9
      "ext v2.16b, v2.16b, v31.16b, #4  \n"  // 13, 14
      "ext v3.16b, v3.16b, v31.16b, #4  \n"  // 18, 19
      "ext v4.16b, v4.16b, v31.16b, #4  \n"  // 23, 24

      // in row0
      "fmul v16.4s, v0.4s, v8.4s   \n"
      "fmul v17.4s, v0.4s, v9.4s   \n"
      "fmul v18.4s, v0.4s, v10.4s  \n"
      "fmul v19.4s, v0.4s, v11.4s  \n"

      // in row1
      "fmla v16.4s, v1.4s, v9.4s  \n"
      "fmla v17.4s, v1.4s, v10.4s \n"
      "fmla v18.4s, v1.4s, v11.4s \n"
      "fmla v19.4s, v1.4s, v12.4s \n"

      // in row2
      "fmla v16.4s, v2.4s, v10.4s \n"
      "fmla v17.4s, v2.4s, v11.4s \n"
      "fmla v18.4s, v2.4s, v12.4s \n"
      "fmla v19.4s, v2.4s, v13.4s \n"

      // in row3
      "fmla v16.4s, v3.4s, v11.4s \n"
      "fmla v17.4s, v3.4s, v12.4s \n"
      "fmla v18.4s, v3.4s, v13.4s \n"
      "fmla v19.4s, v3.4s, v14.4s \n"

      // in row4
      "fmla v16.4s, v4.4s, v12.4s \n"
      "fmla v17.4s, v4.4s, v13.4s \n"
      "fmla v18.4s, v4.4s, v14.4s \n"
      "fmla v19.4s, v4.4s, v15.4s \n"

      // add to out register v25
      "faddp v25.4s, v16.4s, v17.4s \n"
      "faddp v26.4s, v18.4s, v19.4s \n"
      "faddp v25.4s, v25.4s, v26.4s \n"
      "fadd  v25.4s, v25.4s, v20.4s \n"

      // zip
      "zip1 v6.4s, v7.4s, v5.4s  \n"
      "zip2 v8.4s, v7.4s, v5.4s  \n"
      "fadd v6.4s, v6.4s, v20.4s \n"
      "fadd v8.4s, v8.4s, v20.4s \n"
      "ext v7.16b, v6.16b, v31.16b, #8  \n"
      "ext v9.16b, v8.16b, v31.16b, #8  \n"

      // write output
      "st1 {v25.s}[0], [%[dout0]], #4  \n"
      "st1 {v25.s}[1], [%[dout1]], #4  \n"
      "st1 {v25.s}[2], [%[dout2]], #4  \n"
      "st1 {v25.s}[3], [%[dout3]], #4  \n"

      "str d6, [%[dout0]]  \n"
      "str d7, [%[dout1]]  \n"
      "str d8, [%[dout2]]  \n"
      "str d9, [%[dout3]]  \n"

      : [dout0] "+r"(dout0), [dout1] "+r"(dout1), [dout2] "+r"(dout2),
        [dout3] "+r"(dout3), [din0] "+r"(din0), [din1] "+r"(din1),
        [din2] "+r"(din2), [din3] "+r"(din3), [din4] "+r"(din4),
        [din5] "+r"(din5), [din6] "+r"(din6), [din7] "+r"(din7),
        [wh] "+r"(weights)
      : [bias] "r"(bias)
      : "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9",
        "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19",
        "v20", "v25", "v26", "v31");
}

//! kernel for three out with extracting data pre
//! deal with four lines out
//! need extra load weights
void compute_three_out_extract_pre_relu(
    const float* din0, const float* din1, const float* din2, const float* din3,
    const float* din4, const float* din5, const float* din6, const float* din7,
    float* dout0, float* dout1, float* dout2, float* dout3,
    const float* weights, const float* bias) {
  //! din0 - din7: 0-4 v8-v15
  //! dout0 - dout3: v16-v19
  //! weights: v0-v4
  asm volatile(
      // load weights
      "movi v31.4s, #0  \n"
      "add %[wh], %[wh], #4  \n"
      "ldr q0, [%[wh]], #20  \n"  // 1, 2, 3, 4
      "ldr q1, [%[wh]], #20  \n"  // 6, 7, 8, 9
      "ldr q2, [%[wh]], #20  \n"  // 11, 12, 13, 14
      "ldr q3, [%[wh]], #20  \n"  // 16, 17, 18, 19
      "ldr q4, [%[wh]], #20  \n"  // 21, 22, 23, 24

      // load inputs
      "ld1 {v20.4s}, [%[bias]]  \n"
      "ld1 {v8.4s}, [%[din0]], #16  \n"
      "ld1 {v9.4s}, [%[din1]], #16  \n"
      "ld1 {v10.4s}, [%[din2]], #16  \n"
      "ld1 {v11.4s}, [%[din3]], #16  \n"
      "ld1 {v12.4s}, [%[din4]], #16  \n"
      "ld1 {v13.4s}, [%[din5]], #16  \n"

      // in row0
      "fmul v16.4s, v0.4s, v8.4s   \n"
      "fmul v17.4s, v0.4s, v9.4s   \n"
      "fmul v18.4s, v0.4s, v10.4s  \n"
      "fmul v19.4s, v0.4s, v11.4s  \n"

      "ld1 {v14.4s}, [%[din6]], #16  \n"
      "ld1 {v15.4s}, [%[din7]], #16  \n"

      // in row1
      "fmla v16.4s, v1.4s, v9.4s  \n"
      "fmla v17.4s, v1.4s, v10.4s \n"
      "fmla v18.4s, v1.4s, v11.4s \n"
      "fmla v19.4s, v1.4s, v12.4s \n"

      // in row2
      "fmla v16.4s, v2.4s, v10.4s \n"
      "fmla v17.4s, v2.4s, v11.4s \n"
      "fmla v18.4s, v2.4s, v12.4s \n"
      "fmla v19.4s, v2.4s, v13.4s \n"

      // in row3
      "fmla v16.4s, v3.4s, v11.4s \n"
      "fmla v17.4s, v3.4s, v12.4s \n"
      "fmla v18.4s, v3.4s, v13.4s \n"
      "fmla v19.4s, v3.4s, v14.4s \n"

      // in row4
      "fmla v16.4s, v4.4s, v12.4s \n"
      "fmla v17.4s, v4.4s, v13.4s \n"
      "fmla v18.4s, v4.4s, v14.4s \n"
      "fmla v19.4s, v4.4s, v15.4s \n"

      // add to out register v5
      "faddp v5.4s, v16.4s, v17.4s \n"
      "faddp v6.4s, v18.4s, v19.4s \n"
      "faddp v5.4s, v5.4s, v6.4s \n"

      // ext weights
      "ext v0.16b, v0.16b, v31.16b, #4  \n"  // 2, 3, 4
      "ext v1.16b, v1.16b, v31.16b, #4  \n"  // 7, 8, 9
      "ext v2.16b, v2.16b, v31.16b, #4  \n"  // 12, 13, 14
      "ext v3.16b, v3.16b, v31.16b, #4  \n"  // 17, 18, 19
      "ext v4.16b, v4.16b, v31.16b, #4  \n"  // 22, 23, 24

      // in row0
      "fmul v16.4s, v0.4s, v8.4s   \n"
      "fmul v17.4s, v0.4s, v9.4s   \n"
      "fmul v18.4s, v0.4s, v10.4s  \n"
      "fmul v19.4s, v0.4s, v11.4s  \n"

      // in row1
      "fmla v16.4s, v1.4s, v9.4s  \n"
      "fmla v17.4s, v1.4s, v10.4s \n"
      "fmla v18.4s, v1.4s, v11.4s \n"
      "fmla v19.4s, v1.4s, v12.4s \n"

      // in row2
      "fmla v16.4s, v2.4s, v10.4s \n"
      "fmla v17.4s, v2.4s, v11.4s \n"
      "fmla v18.4s, v2.4s, v12.4s \n"
      "fmla v19.4s, v2.4s, v13.4s \n"

      // in row3
      "fmla v16.4s, v3.4s, v11.4s \n"
      "fmla v17.4s, v3.4s, v12.4s \n"
      "fmla v18.4s, v3.4s, v13.4s \n"
      "fmla v19.4s, v3.4s, v14.4s \n"

      // in row4
      "fmla v16.4s, v4.4s, v12.4s \n"
      "fmla v17.4s, v4.4s, v13.4s \n"
      "fmla v18.4s, v4.4s, v14.4s \n"
      "fmla v19.4s, v4.4s, v15.4s \n"

      // add to out register v7
      "faddp v7.4s, v16.4s, v17.4s \n"
      "faddp v6.4s, v18.4s, v19.4s \n"
      "faddp v7.4s, v7.4s, v6.4s \n"

      // ext weights
      "ext v0.16b, v0.16b, v31.16b, #4  \n"  // 3, 4
      "ext v1.16b, v1.16b, v31.16b, #4  \n"  // 8, 9
      "ext v2.16b, v2.16b, v31.16b, #4  \n"  // 13, 14
      "ext v3.16b, v3.16b, v31.16b, #4  \n"  // 18, 19
      "ext v4.16b, v4.16b, v31.16b, #4  \n"  // 23, 24

      // in row0
      "fmul v16.4s, v0.4s, v8.4s   \n"
      "fmul v17.4s, v0.4s, v9.4s   \n"
      "fmul v18.4s, v0.4s, v10.4s  \n"
      "fmul v19.4s, v0.4s, v11.4s  \n"

      // in row1
      "fmla v16.4s, v1.4s, v9.4s  \n"
      "fmla v17.4s, v1.4s, v10.4s \n"
      "fmla v18.4s, v1.4s, v11.4s \n"
      "fmla v19.4s, v1.4s, v12.4s \n"

      // in row2
      "fmla v16.4s, v2.4s, v10.4s \n"
      "fmla v17.4s, v2.4s, v11.4s \n"
      "fmla v18.4s, v2.4s, v12.4s \n"
      "fmla v19.4s, v2.4s, v13.4s \n"

      // in row3
      "fmla v16.4s, v3.4s, v11.4s \n"
      "fmla v17.4s, v3.4s, v12.4s \n"
      "fmla v18.4s, v3.4s, v13.4s \n"
      "fmla v19.4s, v3.4s, v14.4s \n"

      // in row4
      "fmla v16.4s, v4.4s, v12.4s \n"
      "fmla v17.4s, v4.4s, v13.4s \n"
      "fmla v18.4s, v4.4s, v14.4s \n"
      "fmla v19.4s, v4.4s, v15.4s \n"

      // add to out register v25
      "faddp v25.4s, v16.4s, v17.4s \n"
      "faddp v26.4s, v18.4s, v19.4s \n"
      "faddp v25.4s, v25.4s, v26.4s \n"
      "fadd  v25.4s, v25.4s, v20.4s \n"
      "fmax  v25.4s, v25.4s, v31.4s \n"

      // zip
      "zip1 v6.4s, v7.4s, v5.4s  \n"
      "zip2 v8.4s, v7.4s, v5.4s  \n"

      // add bias
      "fadd v6.4s, v6.4s, v20.4s \n"
      "fadd v8.4s, v8.4s, v20.4s \n"

      // relu
      "fmax v6.4s, v6.4s, v31.4s \n"
      "fmax v8.4s, v8.4s, v31.4s \n"

      "ext v7.16b, v6.16b, v31.16b, #8  \n"
      "ext v9.16b, v8.16b, v31.16b, #8  \n"

      // write output
      "st1 {v25.s}[0], [%[dout0]], #4  \n"
      "st1 {v25.s}[1], [%[dout1]], #4  \n"
      "st1 {v25.s}[2], [%[dout2]], #4  \n"
      "st1 {v25.s}[3], [%[dout3]], #4  \n"

      "str d6, [%[dout0]]  \n"
      "str d7, [%[dout1]]  \n"
      "str d8, [%[dout2]]  \n"
      "str d9, [%[dout3]]  \n"

      : [dout0] "+r"(dout0), [dout1] "+r"(dout1), [dout2] "+r"(dout2),
        [dout3] "+r"(dout3), [din0] "+r"(din0), [din1] "+r"(din1),
        [din2] "+r"(din2), [din3] "+r"(din3), [din4] "+r"(din4),
        [din5] "+r"(din5), [din6] "+r"(din6), [din7] "+r"(din7),
        [wh] "+r"(weights)
      : [bias] "r"(bias)
      : "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9",
        "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19",
        "v20", "v25", "v26", "v31");
}

//! kernel for three out with extracting data post
//! deal with four lines out
void compute_three_out_extract_post(
    const float* din0, const float* din1, const float* din2, const float* din3,
    const float* din4, const float* din5, const float* din6, const float* din7,
    float* dout0, float* dout1, float* dout2, float* dout3, float32x4_t w0,
    float32x4_t w1, float32x4_t w2, float32x4_t w3, float32x4_t w4,
    const float* bias) {
  //! din0 - din7: 0-4 v8-v15
  //! dout0 - dout3: v6, v8, v25
  asm volatile(
      "movi v31.4s, #0  \n"
      // load inputs
      "ld1 {v20.4s}, [%[bias]]  \n"
      "ld1 {v8.4s}, [%[din0]], #16  \n"
      "ld1 {v9.4s}, [%[din1]], #16  \n"
      "ld1 {v10.4s}, [%[din2]], #16  \n"
      "ld1 {v11.4s}, [%[din3]], #16  \n"
      "ld1 {v12.4s}, [%[din4]], #16  \n"
      "ld1 {v13.4s}, [%[din5]], #16  \n"

      // in row0
      "fmul v16.4s, %[w0].4s, v8.4s  \n"
      "fmul v17.4s, %[w0].4s, v9.4s  \n"
      "fmul v18.4s, %[w0].4s, v10.4s  \n"
      "fmul v19.4s, %[w0].4s, v11.4s  \n"

      "ld1 {v14.4s}, [%[din6]], #16  \n"
      "ld1 {v15.4s}, [%[din7]], #16  \n"

      // in row1
      "fmla v16.4s, %[w1].4s, v9.4s  \n"
      "fmla v17.4s, %[w1].4s, v10.4s \n"
      "fmla v18.4s, %[w1].4s, v11.4s \n"
      "fmla v19.4s, %[w1].4s, v12.4s \n"

      // in row2
      "fmla v16.4s, %[w2].4s, v10.4s \n"
      "fmla v17.4s, %[w2].4s, v11.4s \n"
      "fmla v18.4s, %[w2].4s, v12.4s \n"
      "fmla v19.4s, %[w2].4s, v13.4s \n"

      // in row3
      "fmla v16.4s, %[w3].4s, v11.4s \n"
      "fmla v17.4s, %[w3].4s, v12.4s \n"
      "fmla v18.4s, %[w3].4s, v13.4s \n"
      "fmla v19.4s, %[w3].4s, v14.4s \n"

      // in row4
      "fmla v16.4s, %[w4].4s, v12.4s \n"
      "fmla v17.4s, %[w4].4s, v13.4s \n"
      "fmla v18.4s, %[w4].4s, v14.4s \n"
      "fmla v19.4s, %[w4].4s, v15.4s \n"

      // add to out register v5
      "faddp v5.4s, v16.4s, v17.4s \n"
      "faddp v6.4s, v18.4s, v19.4s \n"
      "faddp v5.4s, v5.4s, v6.4s \n"

      // ext input
      "ext v8.16b, v8.16b, v31.16b, #4  \n"
      "ext v9.16b, v9.16b, v31.16b, #4  \n"
      "ext v10.16b, v10.16b, v31.16b, #4  \n"
      "ext v11.16b, v11.16b, v31.16b, #4  \n"
      "ext v12.16b, v12.16b, v31.16b, #4  \n"
      "ext v13.16b, v13.16b, v31.16b, #4  \n"
      "ext v14.16b, v14.16b, v31.16b, #4  \n"
      "ext v15.16b, v15.16b, v31.16b, #4  \n"

      // in row0
      "fmul v16.4s, %[w0].4s, v8.4s   \n"
      "fmul v17.4s, %[w0].4s, v9.4s   \n"
      "fmul v18.4s, %[w0].4s, v10.4s  \n"
      "fmul v19.4s, %[w0].4s, v11.4s  \n"

      // in row1
      "fmla v16.4s, %[w1].4s, v9.4s  \n"
      "fmla v17.4s, %[w1].4s, v10.4s \n"
      "fmla v18.4s, %[w1].4s, v11.4s \n"
      "fmla v19.4s, %[w1].4s, v12.4s \n"

      // in row2
      "fmla v16.4s, %[w2].4s, v10.4s \n"
      "fmla v17.4s, %[w2].4s, v11.4s \n"
      "fmla v18.4s, %[w2].4s, v12.4s \n"
      "fmla v19.4s, %[w2].4s, v13.4s \n"

      // in row3
      "fmla v16.4s, %[w3].4s, v11.4s \n"
      "fmla v17.4s, %[w3].4s, v12.4s \n"
      "fmla v18.4s, %[w3].4s, v13.4s \n"
      "fmla v19.4s, %[w3].4s, v14.4s \n"

      // in row4
      "fmla v16.4s, %[w4].4s, v12.4s \n"
      "fmla v17.4s, %[w4].4s, v13.4s \n"
      "fmla v18.4s, %[w4].4s, v14.4s \n"
      "fmla v19.4s, %[w4].4s, v15.4s \n"

      // add to out register v7
      "faddp v7.4s, v16.4s, v17.4s \n"
      "faddp v6.4s, v18.4s, v19.4s \n"
      "faddp v7.4s, v7.4s, v6.4s \n"

      // ext input
      "ext v8.16b, v8.16b, v31.16b, #4    \n"
      "ext v9.16b, v9.16b, v31.16b, #4    \n"
      "ext v10.16b, v10.16b, v31.16b, #4  \n"
      "ext v11.16b, v11.16b, v31.16b, #4  \n"
      "ext v12.16b, v12.16b, v31.16b, #4  \n"
      "ext v13.16b, v13.16b, v31.16b, #4  \n"
      "ext v14.16b, v14.16b, v31.16b, #4  \n"
      "ext v15.16b, v15.16b, v31.16b, #4  \n"

      // in row0
      "fmul v16.4s, %[w0].4s, v8.4s  \n"
      "fmul v17.4s, %[w0].4s, v9.4s  \n"
      "fmul v18.4s, %[w0].4s, v10.4s  \n"
      "fmul v19.4s, %[w0].4s, v11.4s  \n"

      // in row1
      "fmla v16.4s, %[w1].4s, v9.4s  \n"
      "fmla v17.4s, %[w1].4s, v10.4s \n"
      "fmla v18.4s, %[w1].4s, v11.4s \n"
      "fmla v19.4s, %[w1].4s, v12.4s \n"

      // in row2
      "fmla v16.4s, %[w2].4s, v10.4s \n"
      "fmla v17.4s, %[w2].4s, v11.4s \n"
      "fmla v18.4s, %[w2].4s, v12.4s \n"
      "fmla v19.4s, %[w2].4s, v13.4s \n"

      // in row3
      "fmla v16.4s, %[w3].4s, v11.4s \n"
      "fmla v17.4s, %[w3].4s, v12.4s \n"
      "fmla v18.4s, %[w3].4s, v13.4s \n"
      "fmla v19.4s, %[w3].4s, v14.4s \n"

      // in row4
      "fmla v16.4s, %[w4].4s, v12.4s \n"
      "fmla v17.4s, %[w4].4s, v13.4s \n"
      "fmla v18.4s, %[w4].4s, v14.4s \n"
      "fmla v19.4s, %[w4].4s, v15.4s \n"

      // add to out register v25
      "faddp v25.4s, v16.4s, v17.4s \n"
      "faddp v26.4s, v18.4s, v19.4s \n"
      "faddp v25.4s, v25.4s, v26.4s \n"
      "fadd  v25.4s, v25.4s, v20.4s \n"

      // zip
      "zip1 v6.4s, v5.4s, v7.4s  \n"
      "zip2 v8.4s, v5.4s, v7.4s  \n"
      "fadd v6.4s, v6.4s, v20.4s \n"
      "fadd v8.4s, v8.4s, v20.4s \n"
      "ext v7.16b, v6.16b, v31.16b, #8  \n"
      "ext v9.16b, v8.16b, v31.16b, #8  \n"

      // write output
      "str d6, [%[dout0]], #8  \n"
      "str d7, [%[dout1]], #8  \n"
      "str d8, [%[dout2]], #8  \n"
      "str d9, [%[dout3]], #8  \n"

      "st1 {v25.s}[0], [%[dout0]]  \n"
      "st1 {v25.s}[1], [%[dout1]]  \n"
      "st1 {v25.s}[2], [%[dout2]]  \n"
      "st1 {v25.s}[3], [%[dout3]]  \n"

      : [dout0] "+r"(dout0), [dout1] "+r"(dout1), [dout2] "+r"(dout2),
        [dout3] "+r"(dout3), [din0] "+r"(din0), [din1] "+r"(din1),
        [din2] "+r"(din2), [din3] "+r"(din3), [din4] "+r"(din4),
        [din5] "+r"(din5), [din6] "+r"(din6), [din7] "+r"(din7)
      : [w0] "w"(w0), [w1] "w"(w1), [w2] "w"(w2), [w3] "w"(w3), [w4] "w"(w4),
        [bias] "r"(bias)
      : "memory", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13",
        "v14", "v15", "v16", "v17", "v18", "v19", "v20", "v25", "v26", "v31");
}

//! kernel for three out with extracting data post
//! deal with four lines out
void compute_three_out_extract_post_relu(
    const float* din0, const float* din1, const float* din2, const float* din3,
    const float* din4, const float* din5, const float* din6, const float* din7,
    float* dout0, float* dout1, float* dout2, float* dout3, float32x4_t w0,
    float32x4_t w1, float32x4_t w2, float32x4_t w3, float32x4_t w4,
    const float* bias) {
  //! din0 - din7: 0-4 v8-v15
  //! dout0 - dout3: v6, v8, v25
  asm volatile(
      "movi v31.4s, #0  \n"

      // load inputs
      "ld1 {v20.4s}, [%[bias]]  \n"
      "ld1 {v8.4s}, [%[din0]], #16  \n"
      "ld1 {v9.4s}, [%[din1]], #16  \n"
      "ld1 {v10.4s}, [%[din2]], #16  \n"
      "ld1 {v11.4s}, [%[din3]], #16  \n"
      "ld1 {v12.4s}, [%[din4]], #16  \n"
      "ld1 {v13.4s}, [%[din5]], #16  \n"

      // in row0
      "fmul v16.4s, %[w0].4s, v8.4s  \n"
      "fmul v17.4s, %[w0].4s, v9.4s  \n"
      "fmul v18.4s, %[w0].4s, v10.4s  \n"
      "fmul v19.4s, %[w0].4s, v11.4s  \n"

      "ld1 {v14.4s}, [%[din6]], #16  \n"
      "ld1 {v15.4s}, [%[din7]], #16  \n"

      // in row1
      "fmla v16.4s, %[w1].4s, v9.4s  \n"
      "fmla v17.4s, %[w1].4s, v10.4s \n"
      "fmla v18.4s, %[w1].4s, v11.4s \n"
      "fmla v19.4s, %[w1].4s, v12.4s \n"

      // in row2
      "fmla v16.4s, %[w2].4s, v10.4s \n"
      "fmla v17.4s, %[w2].4s, v11.4s \n"
      "fmla v18.4s, %[w2].4s, v12.4s \n"
      "fmla v19.4s, %[w2].4s, v13.4s \n"

      // in row3
      "fmla v16.4s, %[w3].4s, v11.4s \n"
      "fmla v17.4s, %[w3].4s, v12.4s \n"
      "fmla v18.4s, %[w3].4s, v13.4s \n"
      "fmla v19.4s, %[w3].4s, v14.4s \n"

      // in row4
      "fmla v16.4s, %[w4].4s, v12.4s \n"
      "fmla v17.4s, %[w4].4s, v13.4s \n"
      "fmla v18.4s, %[w4].4s, v14.4s \n"
      "fmla v19.4s, %[w4].4s, v15.4s \n"

      // add to out register v5
      "faddp v5.4s, v16.4s, v17.4s \n"
      "faddp v6.4s, v18.4s, v19.4s \n"
      "faddp v5.4s, v5.4s, v6.4s \n"

      // ext input
      "ext v8.16b, v8.16b, v31.16b, #4    \n"
      "ext v9.16b, v9.16b, v31.16b, #4    \n"
      "ext v10.16b, v10.16b, v31.16b, #4  \n"
      "ext v11.16b, v11.16b, v31.16b, #4  \n"
      "ext v12.16b, v12.16b, v31.16b, #4  \n"
      "ext v13.16b, v13.16b, v31.16b, #4  \n"
      "ext v14.16b, v14.16b, v31.16b, #4  \n"
      "ext v15.16b, v15.16b, v31.16b, #4  \n"

      // in row0
      "fmul v16.4s, %[w0].4s, v8.4s   \n"
      "fmul v17.4s, %[w0].4s, v9.4s   \n"
      "fmul v18.4s, %[w0].4s, v10.4s  \n"
      "fmul v19.4s, %[w0].4s, v11.4s  \n"

      // in row1
      "fmla v16.4s, %[w1].4s, v9.4s  \n"
      "fmla v17.4s, %[w1].4s, v10.4s \n"
      "fmla v18.4s, %[w1].4s, v11.4s \n"
      "fmla v19.4s, %[w1].4s, v12.4s \n"

      // in row2
      "fmla v16.4s, %[w2].4s, v10.4s \n"
      "fmla v17.4s, %[w2].4s, v11.4s \n"
      "fmla v18.4s, %[w2].4s, v12.4s \n"
      "fmla v19.4s, %[w2].4s, v13.4s \n"

      // in row3
      "fmla v16.4s, %[w3].4s, v11.4s \n"
      "fmla v17.4s, %[w3].4s, v12.4s \n"
      "fmla v18.4s, %[w3].4s, v13.4s \n"
      "fmla v19.4s, %[w3].4s, v14.4s \n"

      // in row4
      "fmla v16.4s, %[w4].4s, v12.4s \n"
      "fmla v17.4s, %[w4].4s, v13.4s \n"
      "fmla v18.4s, %[w4].4s, v14.4s \n"
      "fmla v19.4s, %[w4].4s, v15.4s \n"

      // add to out register v7
      "faddp v7.4s, v16.4s, v17.4s \n"
      "faddp v6.4s, v18.4s, v19.4s \n"
      "faddp v7.4s, v7.4s, v6.4s \n"

      // ext input
      "ext v8.16b, v8.16b, v31.16b, #4    \n"
      "ext v9.16b, v9.16b, v31.16b, #4    \n"
      "ext v10.16b, v10.16b, v31.16b, #4  \n"
      "ext v11.16b, v11.16b, v31.16b, #4  \n"
      "ext v12.16b, v12.16b, v31.16b, #4  \n"
      "ext v13.16b, v13.16b, v31.16b, #4  \n"
      "ext v14.16b, v14.16b, v31.16b, #4  \n"
      "ext v15.16b, v15.16b, v31.16b, #4  \n"

      // in row0
      "fmul v16.4s, %[w0].4s, v8.4s   \n"
      "fmul v17.4s, %[w0].4s, v9.4s   \n"
      "fmul v18.4s, %[w0].4s, v10.4s  \n"
      "fmul v19.4s, %[w0].4s, v11.4s  \n"

      // in row1
      "fmla v16.4s, %[w1].4s, v9.4s  \n"
      "fmla v17.4s, %[w1].4s, v10.4s \n"
      "fmla v18.4s, %[w1].4s, v11.4s \n"
      "fmla v19.4s, %[w1].4s, v12.4s \n"

      // in row2
      "fmla v16.4s, %[w2].4s, v10.4s \n"
      "fmla v17.4s, %[w2].4s, v11.4s \n"
      "fmla v18.4s, %[w2].4s, v12.4s \n"
      "fmla v19.4s, %[w2].4s, v13.4s \n"

      // in row3
      "fmla v16.4s, %[w3].4s, v11.4s \n"
      "fmla v17.4s, %[w3].4s, v12.4s \n"
      "fmla v18.4s, %[w3].4s, v13.4s \n"
      "fmla v19.4s, %[w3].4s, v14.4s \n"

      // in row4
      "fmla v16.4s, %[w4].4s, v12.4s \n"
      "fmla v17.4s, %[w4].4s, v13.4s \n"
      "fmla v18.4s, %[w4].4s, v14.4s \n"
      "fmla v19.4s, %[w4].4s, v15.4s \n"

      // add to out register v25
      "faddp v25.4s, v16.4s, v17.4s \n"
      "faddp v26.4s, v18.4s, v19.4s \n"
      "faddp v25.4s, v25.4s, v26.4s \n"
      "fadd  v25.4s, v25.4s, v20.4s \n"
      "fmax  v25.4s, v25.4s, v31.4s \n"

      // zip
      "zip1 v6.4s, v5.4s, v7.4s  \n"
      "zip2 v8.4s, v5.4s, v7.4s  \n"

      // add bias
      "fadd v6.4s, v6.4s, v20.4s \n"
      "fadd v8.4s, v8.4s, v20.4s \n"

      // relu
      "fmax v6.4s, v6.4s, v31.4s \n"
      "fmax v8.4s, v8.4s, v31.4s \n"

      "ext v7.16b, v6.16b, v31.16b, #8  \n"
      "ext v9.16b, v8.16b, v31.16b, #8  \n"

      // write output
      "str d6, [%[dout0]], #8  \n"
      "str d7, [%[dout1]], #8  \n"
      "str d8, [%[dout2]], #8  \n"
      "str d9, [%[dout3]], #8  \n"

      "st1 {v25.s}[0], [%[dout0]]  \n"
      "st1 {v25.s}[1], [%[dout1]]  \n"
      "st1 {v25.s}[2], [%[dout2]]  \n"
      "st1 {v25.s}[3], [%[dout3]]  \n"

      : [dout0] "+r"(dout0), [dout1] "+r"(dout1), [dout2] "+r"(dout2),
        [dout3] "+r"(dout3), [din0] "+r"(din0), [din1] "+r"(din1),
        [din2] "+r"(din2), [din3] "+r"(din3), [din4] "+r"(din4),
        [din5] "+r"(din5), [din6] "+r"(din6), [din7] "+r"(din7)
      : [w0] "w"(w0), [w1] "w"(w1), [w2] "w"(w2), [w3] "w"(w3), [w4] "w"(w4),
        [bias] "r"(bias)
      : "memory", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13",
        "v14", "v15", "v16", "v17", "v18", "v19", "v20", "v25", "v26", "v31");
}

//! kernel for four out with extracting data pre
//! deal with four lines out
//! need extra load weights
void compute_four_out_extract_pre(const float* din0, const float* din1,
                                  const float* din2, const float* din3,
                                  const float* din4, const float* din5,
                                  const float* din6, const float* din7,
                                  float* dout0, float* dout1, float* dout2,
                                  float* dout3, const float* weights,
                                  const float* bias) {
  //! din0 - din7: 0-4 v8-v15
  //! dout0 - dout3: v0-v3
  //! weights: v0-v4, v5, v6
  asm volatile(
      // load weights
      "movi v31.4s, #0  \n"
      "mov x0, #20  \n"
      "add %[wh], %[wh], #4  \n"
      "ldr q0, [%[wh]], #20  \n"  // 1, 2, 3, 4
      "ldr q1, [%[wh]], #20  \n"  // 6, 7, 8, 9
      "ldr q2, [%[wh]], #20  \n"  // 11, 12, 13, 14
      "ldr q3, [%[wh]], #20  \n"  // 16, 17, 18, 19
      "ldr q4, [%[wh]]       \n"  // 21, 22, 23, 24
      "sub %[wh], %[wh], #68 \n"

      // load inputs
      "ld1 {v8.4s},  [%[din0]]  \n"
      "ld1 {v9.4s},  [%[din1]]  \n"
      "ld1 {v10.4s}, [%[din2]]  \n"
      "ld1 {v11.4s}, [%[din3]]  \n"
      "ld1 {v12.4s}, [%[din4]]  \n"
      "ld1 {v13.4s}, [%[din5]]  \n"

      // in row0
      "fmul v16.4s, v0.4s, v8.4s  \n"
      "fmul v17.4s, v0.4s, v9.4s  \n"
      "fmul v18.4s, v0.4s, v10.4s \n"
      "fmul v19.4s, v0.4s, v11.4s \n"

      "ld1 {v14.4s}, [%[din6]]  \n"
      "ld1 {v15.4s}, [%[din7]]  \n"

      // in row1
      "fmla v16.4s, v1.4s, v9.4s  \n"
      "fmla v17.4s, v1.4s, v10.4s \n"
      "fmla v18.4s, v1.4s, v11.4s \n"
      "fmla v19.4s, v1.4s, v12.4s \n"

      // in row2
      "fmla v16.4s, v2.4s, v10.4s \n"
      "fmla v17.4s, v2.4s, v11.4s \n"
      "fmla v18.4s, v2.4s, v12.4s \n"
      "fmla v19.4s, v2.4s, v13.4s \n"

      // in row3
      "fmla v16.4s, v3.4s, v11.4s \n"
      "fmla v17.4s, v3.4s, v12.4s \n"
      "fmla v18.4s, v3.4s, v13.4s \n"
      "fmla v19.4s, v3.4s, v14.4s \n"

      // in row4
      "fmla v16.4s, v4.4s, v12.4s \n"
      "fmla v17.4s, v4.4s, v13.4s \n"
      "fmla v18.4s, v4.4s, v14.4s \n"
      "fmla v19.4s, v4.4s, v15.4s \n"

      // add to out register v25
      "faddp v25.4s, v16.4s, v17.4s \n"
      "faddp v26.4s, v18.4s, v19.4s \n"
      "faddp v25.4s, v25.4s, v26.4s \n"

      // load weights col5
      "ld1 {v5.s}[0], [%[wh]], x0  \n"
      "ld1 {v5.s}[1], [%[wh]], x0  \n"
      "ld1 {v5.s}[2], [%[wh]], x0  \n"
      "ld1 {v5.s}[3], [%[wh]], x0  \n"
      "ld1 {v6.s}[0], [%[wh]]      \n"

      // ext weights
      "ext v0.16b, v0.16b, v31.16b, #4  \n"  // 2, 3, 4
      "ext v1.16b, v1.16b, v31.16b, #4  \n"  // 7, 8, 9
      "ext v2.16b, v2.16b, v31.16b, #4  \n"  // 12, 13, 14
      "ext v3.16b, v3.16b, v31.16b, #4  \n"  // 17, 18, 19
      "ext v4.16b, v4.16b, v31.16b, #4  \n"  // 22, 23, 24

      // in row0
      "fmul v16.4s, v0.4s, v8.4s   \n"
      "fmul v17.4s, v0.4s, v9.4s   \n"
      "fmul v18.4s, v0.4s, v10.4s  \n"
      "fmul v19.4s, v0.4s, v11.4s  \n"

      // in row1
      "fmla v16.4s, v1.4s, v9.4s  \n"
      "fmla v17.4s, v1.4s, v10.4s \n"
      "fmla v18.4s, v1.4s, v11.4s \n"
      "fmla v19.4s, v1.4s, v12.4s \n"

      // in row2
      "fmla v16.4s, v2.4s, v10.4s \n"
      "fmla v17.4s, v2.4s, v11.4s \n"
      "fmla v18.4s, v2.4s, v12.4s \n"
      "fmla v19.4s, v2.4s, v13.4s \n"

      // in row3
      "fmla v16.4s, v3.4s, v11.4s \n"
      "fmla v17.4s, v3.4s, v12.4s \n"
      "fmla v18.4s, v3.4s, v13.4s \n"
      "fmla v19.4s, v3.4s, v14.4s \n"

      // in row4
      "fmla v16.4s, v4.4s, v12.4s \n"
      "fmla v17.4s, v4.4s, v13.4s \n"
      "fmla v18.4s, v4.4s, v14.4s \n"
      "fmla v19.4s, v4.4s, v15.4s \n"

      // add to out register v27
      "faddp v27.4s, v16.4s, v17.4s \n"
      "faddp v26.4s, v18.4s, v19.4s \n"
      "faddp v27.4s, v27.4s, v26.4s \n"

      // load in col5
      "ld1 {v20.s}[0], [%[din0]] \n"
      "ld1 {v20.s}[1], [%[din1]] \n"
      "ld1 {v20.s}[2], [%[din2]] \n"
      "ld1 {v20.s}[3], [%[din3]] \n"

      // ext weights
      "ext v0.16b, v0.16b, v31.16b, #4  \n"  // 3, 4
      "ext v1.16b, v1.16b, v31.16b, #4  \n"  // 8, 9
      "ext v2.16b, v2.16b, v31.16b, #4  \n"  // 13, 14
      "ext v3.16b, v3.16b, v31.16b, #4  \n"  // 18, 19
      "ext v4.16b, v4.16b, v31.16b, #4  \n"  // 23, 24

      "ld1 {v21.s}[0], [%[din4]] \n"
      "ld1 {v21.s}[1], [%[din5]] \n"
      "ld1 {v21.s}[2], [%[din6]] \n"
      "ld1 {v21.s}[3], [%[din7]] \n"

      // in row0
      "fmul v16.4s, v0.4s, v8.4s  \n"
      "fmul v17.4s, v0.4s, v9.4s  \n"
      "fmul v18.4s, v0.4s, v10.4s \n"
      "fmul v19.4s, v0.4s, v11.4s \n"

      // in row1
      "fmla v16.4s, v1.4s, v9.4s  \n"
      "fmla v17.4s, v1.4s, v10.4s \n"
      "fmla v18.4s, v1.4s, v11.4s \n"
      "fmla v19.4s, v1.4s, v12.4s \n"

      // in row2
      "fmla v16.4s, v2.4s, v10.4s \n"
      "fmla v17.4s, v2.4s, v11.4s \n"
      "fmla v18.4s, v2.4s, v12.4s \n"
      "fmla v19.4s, v2.4s, v13.4s \n"

      // in row3
      "fmla v16.4s, v3.4s, v11.4s \n"
      "fmla v17.4s, v3.4s, v12.4s \n"
      "fmla v18.4s, v3.4s, v13.4s \n"
      "fmla v19.4s, v3.4s, v14.4s \n"

      // in row4
      "fmla v16.4s, v4.4s, v12.4s \n"
      "fmla v17.4s, v4.4s, v13.4s \n"
      "fmla v18.4s, v4.4s, v14.4s \n"
      "fmla v19.4s, v4.4s, v15.4s \n"

      // add to out register v26
      "faddp v26.4s, v16.4s, v17.4s \n"
      "faddp v28.4s, v18.4s, v19.4s \n"
      "faddp v26.4s, v26.4s, v28.4s \n"

      // ext input col5
      "ext v22.16b, v20.16b, v21.16b, #4  \n"
      "ext v23.16b, v20.16b, v21.16b, #8  \n"
      "ext v24.16b, v20.16b, v21.16b, #12 \n"

      // in col5
      "fmul v16.4s, v5.4s, v20.4s  \n"
      "fmul v17.4s, v5.4s, v22.4s  \n"
      "fmul v18.4s, v5.4s, v23.4s  \n"
      "fmul v19.4s, v5.4s, v24.4s  \n"

      // add to out register v28
      "faddp v28.4s, v16.4s, v17.4s \n"
      "faddp v29.4s, v18.4s, v19.4s \n"
      "faddp v28.4s, v28.4s, v29.4s \n"
      "fmla v28.4s, v21.4s, v6.s[0] \n"

      "ld1 {v8.4s}, [%[bias]]  \n"

      // zip
      "zip1 v0.4s, v28.4s, v26.4s  \n"
      "zip2 v2.4s, v28.4s, v26.4s  \n"
      "zip1 v4.4s, v27.4s, v25.4s  \n"
      "zip2 v6.4s, v27.4s, v25.4s  \n"

      "fadd v0.4s, v0.4s, v8.4s  \n"
      "fadd v2.4s, v2.4s, v8.4s  \n"
      "fadd v4.4s, v4.4s, v8.4s  \n"
      "fadd v6.4s, v6.4s, v8.4s  \n"

      "ext v1.16b, v0.16b, v31.16b, #8  \n"
      "ext v3.16b, v2.16b, v31.16b, #8  \n"
      "ext v5.16b, v4.16b, v31.16b, #8  \n"
      "ext v7.16b, v6.16b, v31.16b, #8  \n"

      // write output
      "str d0, [%[dout0]], #8  \n"
      "str d1, [%[dout1]], #8  \n"
      "str d2, [%[dout2]], #8  \n"
      "str d3, [%[dout3]], #8  \n"

      "str d4, [%[dout0]]  \n"
      "str d5, [%[dout1]]  \n"
      "str d6, [%[dout2]]  \n"
      "str d7, [%[dout3]]  \n"

      : [dout0] "+r"(dout0), [dout1] "+r"(dout1), [dout2] "+r"(dout2),
        [dout3] "+r"(dout3), [din0] "+r"(din0), [din1] "+r"(din1),
        [din2] "+r"(din2), [din3] "+r"(din3), [din4] "+r"(din4),
        [din5] "+r"(din5), [din6] "+r"(din6), [din7] "+r"(din7),
        [wh] "+r"(weights)
      : [bias] "r"(bias)
      : "memory", "x0", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8",
        "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18",
        "v19", "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27", "v28",
        "v29", "v31");
}

//! kernel for four out with extracting data pre
//! deal with four lines out
//! need extra load weights
void compute_four_out_extract_pre_relu(const float* din0, const float* din1,
                                       const float* din2, const float* din3,
                                       const float* din4, const float* din5,
                                       const float* din6, const float* din7,
                                       float* dout0, float* dout1, float* dout2,
                                       float* dout3, const float* weights,
                                       const float* bias) {
  //! din0 - din7: 0-4 v8-v15
  //! dout0 - dout3: v0-v3
  //! weights: v0-v4, v5, v6
  asm volatile(
      // load weights
      "movi v31.4s, #0  \n"
      "mov x0, #20  \n"
      "add %[wh], %[wh], #4  \n"
      "ldr q0, [%[wh]], #20  \n"  // 1, 2, 3, 4
      "ldr q1, [%[wh]], #20  \n"  // 6, 7, 8, 9
      "ldr q2, [%[wh]], #20  \n"  // 11, 12, 13, 14
      "ldr q3, [%[wh]], #20  \n"  // 16, 17, 18, 19
      "ldr q4, [%[wh]]       \n"  // 21, 22, 23, 24
      "sub %[wh], %[wh], #68 \n"

      // load inputs
      "ld1 {v8.4s}, [%[din0]]   \n"
      "ld1 {v9.4s}, [%[din1]]   \n"
      "ld1 {v10.4s}, [%[din2]]  \n"
      "ld1 {v11.4s}, [%[din3]]  \n"
      "ld1 {v12.4s}, [%[din4]]  \n"
      "ld1 {v13.4s}, [%[din5]]  \n"

      // in row0
      "fmul v16.4s, v0.4s, v8.4s  \n"
      "fmul v17.4s, v0.4s, v9.4s  \n"
      "fmul v18.4s, v0.4s, v10.4s \n"
      "fmul v19.4s, v0.4s, v11.4s \n"

      "ld1 {v14.4s}, [%[din6]]  \n"
      "ld1 {v15.4s}, [%[din7]]  \n"

      // in row1
      "fmla v16.4s, v1.4s, v9.4s  \n"
      "fmla v17.4s, v1.4s, v10.4s \n"
      "fmla v18.4s, v1.4s, v11.4s \n"
      "fmla v19.4s, v1.4s, v12.4s \n"

      // in row2
      "fmla v16.4s, v2.4s, v10.4s \n"
      "fmla v17.4s, v2.4s, v11.4s \n"
      "fmla v18.4s, v2.4s, v12.4s \n"
      "fmla v19.4s, v2.4s, v13.4s \n"

      // in row3
      "fmla v16.4s, v3.4s, v11.4s \n"
      "fmla v17.4s, v3.4s, v12.4s \n"
      "fmla v18.4s, v3.4s, v13.4s \n"
      "fmla v19.4s, v3.4s, v14.4s \n"

      // in row4
      "fmla v16.4s, v4.4s, v12.4s \n"
      "fmla v17.4s, v4.4s, v13.4s \n"
      "fmla v18.4s, v4.4s, v14.4s \n"
      "fmla v19.4s, v4.4s, v15.4s \n"

      // add to out register v25
      "faddp v25.4s, v16.4s, v17.4s \n"
      "faddp v26.4s, v18.4s, v19.4s \n"
      "faddp v25.4s, v25.4s, v26.4s \n"

      // load weights col5
      "ld1 {v5.s}[0], [%[wh]], x0  \n"
      "ld1 {v5.s}[1], [%[wh]], x0  \n"
      "ld1 {v5.s}[2], [%[wh]], x0  \n"
      "ld1 {v5.s}[3], [%[wh]], x0  \n"
      "ld1 {v6.s}[0], [%[wh]]      \n"

      // ext weights
      "ext v0.16b, v0.16b, v31.16b, #4  \n"  // 2, 3, 4
      "ext v1.16b, v1.16b, v31.16b, #4  \n"  // 7, 8, 9
      "ext v2.16b, v2.16b, v31.16b, #4  \n"  // 12, 13, 14
      "ext v3.16b, v3.16b, v31.16b, #4  \n"  // 17, 18, 19
      "ext v4.16b, v4.16b, v31.16b, #4  \n"  // 22, 23, 24

      // in row0
      "fmul v16.4s, v0.4s, v8.4s   \n"
      "fmul v17.4s, v0.4s, v9.4s   \n"
      "fmul v18.4s, v0.4s, v10.4s  \n"
      "fmul v19.4s, v0.4s, v11.4s  \n"

      // in row1
      "fmla v16.4s, v1.4s, v9.4s  \n"
      "fmla v17.4s, v1.4s, v10.4s \n"
      "fmla v18.4s, v1.4s, v11.4s \n"
      "fmla v19.4s, v1.4s, v12.4s \n"

      // in row2
      "fmla v16.4s, v2.4s, v10.4s \n"
      "fmla v17.4s, v2.4s, v11.4s \n"
      "fmla v18.4s, v2.4s, v12.4s \n"
      "fmla v19.4s, v2.4s, v13.4s \n"

      // in row3
      "fmla v16.4s, v3.4s, v11.4s \n"
      "fmla v17.4s, v3.4s, v12.4s \n"
      "fmla v18.4s, v3.4s, v13.4s \n"
      "fmla v19.4s, v3.4s, v14.4s \n"

      // in row4
      "fmla v16.4s, v4.4s, v12.4s \n"
      "fmla v17.4s, v4.4s, v13.4s \n"
      "fmla v18.4s, v4.4s, v14.4s \n"
      "fmla v19.4s, v4.4s, v15.4s \n"

      // add to out register v27
      "faddp v27.4s, v16.4s, v17.4s \n"
      "faddp v26.4s, v18.4s, v19.4s \n"
      "faddp v27.4s, v27.4s, v26.4s \n"

      // load in col5
      "ld1 {v20.s}[0], [%[din0]] \n"
      "ld1 {v20.s}[1], [%[din1]] \n"
      "ld1 {v20.s}[2], [%[din2]] \n"
      "ld1 {v20.s}[3], [%[din3]] \n"

      // ext weights
      "ext v0.16b, v0.16b, v31.16b, #4  \n"  // 3, 4
      "ext v1.16b, v1.16b, v31.16b, #4  \n"  // 8, 9
      "ext v2.16b, v2.16b, v31.16b, #4  \n"  // 13, 14
      "ext v3.16b, v3.16b, v31.16b, #4  \n"  // 18, 19
      "ext v4.16b, v4.16b, v31.16b, #4  \n"  // 23, 24

      "ld1 {v21.s}[0], [%[din4]] \n"
      "ld1 {v21.s}[1], [%[din5]] \n"
      "ld1 {v21.s}[2], [%[din6]] \n"
      "ld1 {v21.s}[3], [%[din7]] \n"

      // in row0
      "fmul v16.4s, v0.4s, v8.4s  \n"
      "fmul v17.4s, v0.4s, v9.4s  \n"
      "fmul v18.4s, v0.4s, v10.4s \n"
      "fmul v19.4s, v0.4s, v11.4s \n"

      // in row1
      "fmla v16.4s, v1.4s, v9.4s  \n"
      "fmla v17.4s, v1.4s, v10.4s \n"
      "fmla v18.4s, v1.4s, v11.4s \n"
      "fmla v19.4s, v1.4s, v12.4s \n"

      // in row2
      "fmla v16.4s, v2.4s, v10.4s \n"
      "fmla v17.4s, v2.4s, v11.4s \n"
      "fmla v18.4s, v2.4s, v12.4s \n"
      "fmla v19.4s, v2.4s, v13.4s \n"

      // in row3
      "fmla v16.4s, v3.4s, v11.4s \n"
      "fmla v17.4s, v3.4s, v12.4s \n"
      "fmla v18.4s, v3.4s, v13.4s \n"
      "fmla v19.4s, v3.4s, v14.4s \n"

      // in row4
      "fmla v16.4s, v4.4s, v12.4s \n"
      "fmla v17.4s, v4.4s, v13.4s \n"
      "fmla v18.4s, v4.4s, v14.4s \n"
      "fmla v19.4s, v4.4s, v15.4s \n"

      // add to out register v26
      "faddp v26.4s, v16.4s, v17.4s \n"
      "faddp v28.4s, v18.4s, v19.4s \n"
      "faddp v26.4s, v26.4s, v28.4s \n"

      // ext input col5
      "ext v22.16b, v20.16b, v21.16b, #4  \n"
      "ext v23.16b, v20.16b, v21.16b, #8  \n"
      "ext v24.16b, v20.16b, v21.16b, #12 \n"

      // in col5
      "fmul v16.4s, v5.4s, v20.4s  \n"
      "fmul v17.4s, v5.4s, v22.4s  \n"
      "fmul v18.4s, v5.4s, v23.4s  \n"
      "fmul v19.4s, v5.4s, v24.4s  \n"

      // add to out register v28
      "faddp v28.4s, v16.4s, v17.4s \n"
      "faddp v29.4s, v18.4s, v19.4s \n"
      "faddp v28.4s, v28.4s, v29.4s \n"
      "fmla v28.4s, v21.4s, v6.s[0] \n"

      "ld1 {v8.4s}, [%[bias]]  \n"

      // zip
      "zip1 v0.4s, v28.4s, v26.4s  \n"
      "zip2 v2.4s, v28.4s, v26.4s  \n"
      "zip1 v4.4s, v27.4s, v25.4s  \n"
      "zip2 v6.4s, v27.4s, v25.4s  \n"

      // add bias
      "fadd v0.4s, v0.4s, v8.4s  \n"
      "fadd v2.4s, v2.4s, v8.4s  \n"
      "fadd v4.4s, v4.4s, v8.4s  \n"
      "fadd v6.4s, v6.4s, v8.4s  \n"

      // relu
      "fmax v0.4s, v0.4s, v31.4s \n"
      "fmax v2.4s, v2.4s, v31.4s \n"
      "fmax v4.4s, v4.4s, v31.4s \n"
      "fmax v6.4s, v6.4s, v31.4s \n"

      "ext v1.16b, v0.16b, v31.16b, #8  \n"
      "ext v3.16b, v2.16b, v31.16b, #8  \n"
      "ext v5.16b, v4.16b, v31.16b, #8  \n"
      "ext v7.16b, v6.16b, v31.16b, #8  \n"

      // write output
      "str d0, [%[dout0]], #8  \n"
      "str d1, [%[dout1]], #8  \n"
      "str d2, [%[dout2]], #8  \n"
      "str d3, [%[dout3]], #8  \n"

      "str d4, [%[dout0]]  \n"
      "str d5, [%[dout1]]  \n"
      "str d6, [%[dout2]]  \n"
      "str d7, [%[dout3]]  \n"

      : [dout0] "+r"(dout0), [dout1] "+r"(dout1), [dout2] "+r"(dout2),
        [dout3] "+r"(dout3), [din0] "+r"(din0), [din1] "+r"(din1),
        [din2] "+r"(din2), [din3] "+r"(din3), [din4] "+r"(din4),
        [din5] "+r"(din5), [din6] "+r"(din6), [din7] "+r"(din7),
        [wh] "+r"(weights)
      : [bias] "r"(bias)
      : "memory", "x0", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8",
        "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18",
        "v19", "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27", "v28",
        "v29", "v31");
}

//! kernel for four out with extracting data post
//! deal with four lines out
void compute_four_out_extract_post(const float* din0, const float* din1,
                                   const float* din2, const float* din3,
                                   const float* din4, const float* din5,
                                   const float* din6, const float* din7,
                                   float* dout0, float* dout1, float* dout2,
                                   float* dout3, float32x4_t w0, float32x4_t w1,
                                   float32x4_t w2, float32x4_t w3,
                                   float32x4_t w4, const float* bias) {
  //! din0 - din7: 0-4 v8-v15
  //! dout0 - dout3: v0-v3
  const int64_t s_12 = 12;
  const float* doutl[4] = {dout0, dout1, dout2, dout3};
  void* doutl_ptr = reinterpret_cast<void*>(doutl);
  asm volatile(
      "movi v31.4s, #0  \n"
      "ldp x0, x1, [%[doutl]], #16  \n"
      "ldp x2, x3, [%[doutl]]  \n"

      // load inputs
      "ld1 {v8.4s}, [%[din0]], %[s_12]   \n"
      "ld1 {v9.4s}, [%[din1]], %[s_12]   \n"
      "ld1 {v10.4s}, [%[din2]], %[s_12]  \n"
      "ld1 {v11.4s}, [%[din3]], %[s_12]  \n"
      "ld1 {v12.4s}, [%[din4]], %[s_12]  \n"
      "ld1 {v13.4s}, [%[din5]], %[s_12]  \n"

      // in row0
      "fmul v16.4s, %[w0].4s, v8.4s  \n"
      "fmul v17.4s, %[w0].4s, v9.4s  \n"
      "fmul v18.4s, %[w0].4s, v10.4s \n"
      "fmul v19.4s, %[w0].4s, v11.4s \n"

      "ld1 {v14.4s}, [%[din6]], %[s_12]  \n"
      "ld1 {v15.4s}, [%[din7]], %[s_12]  \n"

      // in row1
      "fmla v16.4s, %[w1].4s, v9.4s  \n"
      "fmla v17.4s, %[w1].4s, v10.4s \n"
      "fmla v18.4s, %[w1].4s, v11.4s \n"
      "fmla v19.4s, %[w1].4s, v12.4s \n"

      // in row2
      "fmla v16.4s, %[w2].4s, v10.4s \n"
      "fmla v17.4s, %[w2].4s, v11.4s \n"
      "fmla v18.4s, %[w2].4s, v12.4s \n"
      "fmla v19.4s, %[w2].4s, v13.4s \n"

      // in row3
      "fmla v16.4s, %[w3].4s, v11.4s \n"
      "fmla v17.4s, %[w3].4s, v12.4s \n"
      "fmla v18.4s, %[w3].4s, v13.4s \n"
      "fmla v19.4s, %[w3].4s, v14.4s \n"

      // in row4
      "fmla v16.4s, %[w4].4s, v12.4s \n"
      "fmla v17.4s, %[w4].4s, v13.4s \n"
      "fmla v18.4s, %[w4].4s, v14.4s \n"
      "fmla v19.4s, %[w4].4s, v15.4s \n"

      // add to out register v25
      "faddp v25.4s, v16.4s, v17.4s \n"
      "faddp v26.4s, v18.4s, v19.4s \n"
      "faddp v25.4s, v25.4s, v26.4s \n"

      // load input col5
      "ld1 {v20.s}[0], [%[din0]]  \n"
      "ld1 {v20.s}[1], [%[din1]]  \n"
      "ld1 {v20.s}[2], [%[din2]]  \n"
      "ld1 {v20.s}[3], [%[din3]]  \n"

      // ext input
      "ext v8.16b, v8.16b, v31.16b, #4  \n"
      "ext v9.16b, v9.16b, v31.16b, #4  \n"
      "ext v10.16b, v10.16b, v31.16b, #4  \n"
      "ext v11.16b, v11.16b, v31.16b, #4  \n"
      "ext v12.16b, v12.16b, v31.16b, #4  \n"
      "ext v13.16b, v13.16b, v31.16b, #4  \n"
      "ext v14.16b, v14.16b, v31.16b, #4  \n"
      "ext v15.16b, v15.16b, v31.16b, #4  \n"

      // load input col5
      "ld1 {v21.s}[0], [%[din4]]  \n"
      "ld1 {v21.s}[1], [%[din5]]  \n"
      "ld1 {v21.s}[2], [%[din6]]  \n"
      "ld1 {v21.s}[3], [%[din7]]  \n"

      // in row0
      "fmul v16.4s, %[w0].4s, v8.4s  \n"
      "fmul v17.4s, %[w0].4s, v9.4s  \n"
      "fmul v18.4s, %[w0].4s, v10.4s \n"
      "fmul v19.4s, %[w0].4s, v11.4s \n"

      // in row1
      "fmla v16.4s, %[w1].4s, v9.4s  \n"
      "fmla v17.4s, %[w1].4s, v10.4s \n"
      "fmla v18.4s, %[w1].4s, v11.4s \n"
      "fmla v19.4s, %[w1].4s, v12.4s \n"

      // in row2
      "fmla v16.4s, %[w2].4s, v10.4s \n"
      "fmla v17.4s, %[w2].4s, v11.4s \n"
      "fmla v18.4s, %[w2].4s, v12.4s \n"
      "fmla v19.4s, %[w2].4s, v13.4s \n"

      // in row3
      "fmla v16.4s, %[w3].4s, v11.4s \n"
      "fmla v17.4s, %[w3].4s, v12.4s \n"
      "fmla v18.4s, %[w3].4s, v13.4s \n"
      "fmla v19.4s, %[w3].4s, v14.4s \n"

      // in row4
      "fmla v16.4s, %[w4].4s, v12.4s \n"
      "fmla v17.4s, %[w4].4s, v13.4s \n"
      "fmla v18.4s, %[w4].4s, v14.4s \n"
      "fmla v19.4s, %[w4].4s, v15.4s \n"

      // add to out register v27
      "faddp v27.4s, v16.4s, v17.4s \n"
      "faddp v26.4s, v18.4s, v19.4s \n"
      "faddp v27.4s, v27.4s, v26.4s \n"

      // ext input
      "ext v8.16b, v8.16b, v31.16b, #4    \n"
      "ext v9.16b, v9.16b, v31.16b, #4    \n"
      "ext v10.16b, v10.16b, v31.16b, #4  \n"
      "ext v11.16b, v11.16b, v31.16b, #4  \n"
      "ext v12.16b, v12.16b, v31.16b, #4  \n"
      "ext v13.16b, v13.16b, v31.16b, #4  \n"
      "ext v14.16b, v14.16b, v31.16b, #4  \n"
      "ext v15.16b, v15.16b, v31.16b, #4  \n"

      // in row0
      "fmul v16.4s, %[w0].4s, v8.4s  \n"
      "fmul v17.4s, %[w0].4s, v9.4s  \n"
      "fmul v18.4s, %[w0].4s, v10.4s \n"
      "fmul v19.4s, %[w0].4s, v11.4s \n"

      // in row1
      "fmla v16.4s, %[w1].4s, v9.4s  \n"
      "fmla v17.4s, %[w1].4s, v10.4s \n"
      "fmla v18.4s, %[w1].4s, v11.4s \n"
      "fmla v19.4s, %[w1].4s, v12.4s \n"

      // in row2
      "fmla v16.4s, %[w2].4s, v10.4s \n"
      "fmla v17.4s, %[w2].4s, v11.4s \n"
      "fmla v18.4s, %[w2].4s, v12.4s \n"
      "fmla v19.4s, %[w2].4s, v13.4s \n"

      // in row3
      "fmla v16.4s, %[w3].4s, v11.4s \n"
      "fmla v17.4s, %[w3].4s, v12.4s \n"
      "fmla v18.4s, %[w3].4s, v13.4s \n"
      "fmla v19.4s, %[w3].4s, v14.4s \n"

      // in row4
      "fmla v16.4s, %[w4].4s, v12.4s \n"
      "fmla v17.4s, %[w4].4s, v13.4s \n"
      "fmla v18.4s, %[w4].4s, v14.4s \n"
      "fmla v19.4s, %[w4].4s, v15.4s \n"

      // add to out register v26
      "faddp v26.4s, v16.4s, v17.4s \n"
      "faddp v28.4s, v18.4s, v19.4s \n"
      "faddp v26.4s, v26.4s, v28.4s \n"

      // ext input col5
      "ext v8.16b, v20.16b, v21.16b, #4  \n"
      "ext v9.16b, v20.16b, v21.16b, #8  \n"
      "ext v10.16b, v20.16b, v21.16b, #12  \n"

      // ext weights col0
      "ins v5.s[0], %[w0].s[0]  \n"
      "ins v5.s[1], %[w1].s[0]  \n"
      "ins v5.s[2], %[w2].s[0]  \n"
      "ins v5.s[3], %[w3].s[0]  \n"

      // in col5
      "fmul v16.4s, v5.4s, v20.4s \n"
      "fmul v17.4s, v5.4s, v8.4s  \n"
      "fmul v18.4s, v5.4s, v9.4s  \n"
      "fmul v19.4s, v5.4s, v10.4s \n"

      // add to out register v28
      "faddp v28.4s, v16.4s, v17.4s \n"
      "faddp v29.4s, v18.4s, v19.4s \n"
      "faddp v28.4s, v28.4s, v29.4s \n"
      "fmla v28.4s, v21.4s, %[w4].s[0]  \n"

      "ld1 {v8.4s}, [%[bias]]  \n"

      // zip
      "zip1 v0.4s, v25.4s, v27.4s  \n"
      "zip2 v2.4s, v25.4s, v27.4s  \n"
      "zip1 v4.4s, v26.4s, v28.4s  \n"
      "zip2 v6.4s, v26.4s, v28.4s  \n"

      "fadd v0.4s, v0.4s, v8.4s  \n"
      "fadd v2.4s, v2.4s, v8.4s  \n"
      "fadd v4.4s, v4.4s, v8.4s  \n"
      "fadd v6.4s, v6.4s, v8.4s  \n"

      "ext v1.16b, v0.16b, v31.16b, #8  \n"
      "ext v3.16b, v2.16b, v31.16b, #8  \n"
      "ext v5.16b, v4.16b, v31.16b, #8  \n"
      "ext v7.16b, v6.16b, v31.16b, #8  \n"

      // write output
      "str d0, [x0], #8  \n"
      "str d1, [x1], #8  \n"
      "str d2, [x2], #8  \n"
      "str d3, [x3], #8  \n"

      "str d4, [x0]  \n"
      "str d5, [x1]  \n"
      "str d6, [x2]  \n"
      "str d7, [x3]  \n"

      : [din0] "+r"(din0), [din1] "+r"(din1), [din2] "+r"(din2),
        [din3] "+r"(din3), [din4] "+r"(din4), [din5] "+r"(din5),
        [din6] "+r"(din6), [din7] "+r"(din7), [doutl] "+r"(doutl_ptr)
      : [s_12] "r"(s_12), [w0] "w"(w0), [w1] "w"(w1), [w2] "w"(w2),
        [w3] "w"(w3), [w4] "w"(w4), [bias] "r"(bias)
      : "memory", "x0", "x1", "x2", "x3", "v0", "v1", "v2", "v3", "v5", "v7",
        "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17",
        "v18", "v19", "v20", "v21", "v25", "v26", "v27", "v28", "v29", "v31");
}

//! kernel for four out with extracting data post
//! deal with four lines out
void compute_four_out_extract_post_relu(
    const float* din0, const float* din1, const float* din2, const float* din3,
    const float* din4, const float* din5, const float* din6, const float* din7,
    float* dout0, float* dout1, float* dout2, float* dout3, float32x4_t w0,
    float32x4_t w1, float32x4_t w2, float32x4_t w3, float32x4_t w4,
    const float* bias) {
  //! din0 - din7: 0-4 v8-v15
  //! dout0 - dout3: v0-v3
  const int64_t s_12 = 12;
  const float* doutl[4] = {dout0, dout1, dout2, dout3};
  void* doutl_ptr = reinterpret_cast<void*>(doutl);
  asm volatile(
      "movi v31.4s, #0  \n"
      "ldp x0, x1, [%[doutl]], #16  \n"
      "ldp x2, x3, [%[doutl]]  \n"

      // load inputs
      "ld1 {v8.4s}, [%[din0]], %[s_12]   \n"
      "ld1 {v9.4s}, [%[din1]], %[s_12]   \n"
      "ld1 {v10.4s}, [%[din2]], %[s_12]  \n"
      "ld1 {v11.4s}, [%[din3]], %[s_12]  \n"
      "ld1 {v12.4s}, [%[din4]], %[s_12]  \n"
      "ld1 {v13.4s}, [%[din5]], %[s_12]  \n"

      // in row0
      "fmul v16.4s, %[w0].4s, v8.4s   \n"
      "fmul v17.4s, %[w0].4s, v9.4s   \n"
      "fmul v18.4s, %[w0].4s, v10.4s  \n"
      "fmul v19.4s, %[w0].4s, v11.4s  \n"

      "ld1 {v14.4s}, [%[din6]], %[s_12]  \n"
      "ld1 {v15.4s}, [%[din7]], %[s_12]  \n"

      // in row1
      "fmla v16.4s, %[w1].4s, v9.4s  \n"
      "fmla v17.4s, %[w1].4s, v10.4s \n"
      "fmla v18.4s, %[w1].4s, v11.4s \n"
      "fmla v19.4s, %[w1].4s, v12.4s \n"

      // in row2
      "fmla v16.4s, %[w2].4s, v10.4s \n"
      "fmla v17.4s, %[w2].4s, v11.4s \n"
      "fmla v18.4s, %[w2].4s, v12.4s \n"
      "fmla v19.4s, %[w2].4s, v13.4s \n"

      // in row3
      "fmla v16.4s, %[w3].4s, v11.4s \n"
      "fmla v17.4s, %[w3].4s, v12.4s \n"
      "fmla v18.4s, %[w3].4s, v13.4s \n"
      "fmla v19.4s, %[w3].4s, v14.4s \n"

      // in row4
      "fmla v16.4s, %[w4].4s, v12.4s \n"
      "fmla v17.4s, %[w4].4s, v13.4s \n"
      "fmla v18.4s, %[w4].4s, v14.4s \n"
      "fmla v19.4s, %[w4].4s, v15.4s \n"

      // add to out register v25
      "faddp v25.4s, v16.4s, v17.4s \n"
      "faddp v26.4s, v18.4s, v19.4s \n"
      "faddp v25.4s, v25.4s, v26.4s \n"

      // load input col5
      "ld1 {v20.s}[0], [%[din0]]  \n"
      "ld1 {v20.s}[1], [%[din1]]  \n"
      "ld1 {v20.s}[2], [%[din2]]  \n"
      "ld1 {v20.s}[3], [%[din3]]  \n"

      // ext input
      "ext v8.16b, v8.16b, v31.16b, #4  \n"
      "ext v9.16b, v9.16b, v31.16b, #4  \n"
      "ext v10.16b, v10.16b, v31.16b, #4  \n"
      "ext v11.16b, v11.16b, v31.16b, #4  \n"
      "ext v12.16b, v12.16b, v31.16b, #4  \n"
      "ext v13.16b, v13.16b, v31.16b, #4  \n"
      "ext v14.16b, v14.16b, v31.16b, #4  \n"
      "ext v15.16b, v15.16b, v31.16b, #4  \n"

      // load input col5
      "ld1 {v21.s}[0], [%[din4]]  \n"
      "ld1 {v21.s}[1], [%[din5]]  \n"
      "ld1 {v21.s}[2], [%[din6]]  \n"
      "ld1 {v21.s}[3], [%[din7]]  \n"

      // in row0
      "fmul v16.4s, %[w0].4s, v8.4s  \n"
      "fmul v17.4s, %[w0].4s, v9.4s  \n"
      "fmul v18.4s, %[w0].4s, v10.4s \n"
      "fmul v19.4s, %[w0].4s, v11.4s \n"

      // in row1
      "fmla v16.4s, %[w1].4s, v9.4s  \n"
      "fmla v17.4s, %[w1].4s, v10.4s \n"
      "fmla v18.4s, %[w1].4s, v11.4s \n"
      "fmla v19.4s, %[w1].4s, v12.4s \n"

      // in row2
      "fmla v16.4s, %[w2].4s, v10.4s \n"
      "fmla v17.4s, %[w2].4s, v11.4s \n"
      "fmla v18.4s, %[w2].4s, v12.4s \n"
      "fmla v19.4s, %[w2].4s, v13.4s \n"

      // in row3
      "fmla v16.4s, %[w3].4s, v11.4s \n"
      "fmla v17.4s, %[w3].4s, v12.4s \n"
      "fmla v18.4s, %[w3].4s, v13.4s \n"
      "fmla v19.4s, %[w3].4s, v14.4s \n"

      // in row4
      "fmla v16.4s, %[w4].4s, v12.4s \n"
      "fmla v17.4s, %[w4].4s, v13.4s \n"
      "fmla v18.4s, %[w4].4s, v14.4s \n"
      "fmla v19.4s, %[w4].4s, v15.4s \n"

      // add to out register v27
      "faddp v27.4s, v16.4s, v17.4s \n"
      "faddp v26.4s, v18.4s, v19.4s \n"
      "faddp v27.4s, v27.4s, v26.4s \n"

      // ext input
      "ext v8.16b, v8.16b, v31.16b, #4  \n"
      "ext v9.16b, v9.16b, v31.16b, #4  \n"
      "ext v10.16b, v10.16b, v31.16b, #4  \n"
      "ext v11.16b, v11.16b, v31.16b, #4  \n"
      "ext v12.16b, v12.16b, v31.16b, #4  \n"
      "ext v13.16b, v13.16b, v31.16b, #4  \n"
      "ext v14.16b, v14.16b, v31.16b, #4  \n"
      "ext v15.16b, v15.16b, v31.16b, #4  \n"

      // in row0
      "fmul v16.4s, %[w0].4s, v8.4s  \n"
      "fmul v17.4s, %[w0].4s, v9.4s  \n"
      "fmul v18.4s, %[w0].4s, v10.4s \n"
      "fmul v19.4s, %[w0].4s, v11.4s \n"

      // in row1
      "fmla v16.4s, %[w1].4s, v9.4s  \n"
      "fmla v17.4s, %[w1].4s, v10.4s \n"
      "fmla v18.4s, %[w1].4s, v11.4s \n"
      "fmla v19.4s, %[w1].4s, v12.4s \n"

      // in row2
      "fmla v16.4s, %[w2].4s, v10.4s \n"
      "fmla v17.4s, %[w2].4s, v11.4s \n"
      "fmla v18.4s, %[w2].4s, v12.4s \n"
      "fmla v19.4s, %[w2].4s, v13.4s \n"

      // in row3
      "fmla v16.4s, %[w3].4s, v11.4s \n"
      "fmla v17.4s, %[w3].4s, v12.4s \n"
      "fmla v18.4s, %[w3].4s, v13.4s \n"
      "fmla v19.4s, %[w3].4s, v14.4s \n"

      // in row4
      "fmla v16.4s, %[w4].4s, v12.4s \n"
      "fmla v17.4s, %[w4].4s, v13.4s \n"
      "fmla v18.4s, %[w4].4s, v14.4s \n"
      "fmla v19.4s, %[w4].4s, v15.4s \n"

      // add to out register v26
      "faddp v26.4s, v16.4s, v17.4s \n"
      "faddp v28.4s, v18.4s, v19.4s \n"
      "faddp v26.4s, v26.4s, v28.4s \n"

      // ext input col5
      "ext v8.16b, v20.16b, v21.16b, #4   \n"
      "ext v9.16b, v20.16b, v21.16b, #8   \n"
      "ext v10.16b, v20.16b, v21.16b, #12 \n"

      // ext weights col0
      "ins v5.s[0], %[w0].s[0]  \n"
      "ins v5.s[1], %[w1].s[0]  \n"
      "ins v5.s[2], %[w2].s[0]  \n"
      "ins v5.s[3], %[w3].s[0]  \n"

      // in col5
      "fmul v16.4s, v5.4s, v20.4s \n"
      "fmul v17.4s, v5.4s, v8.4s  \n"
      "fmul v18.4s, v5.4s, v9.4s  \n"
      "fmul v19.4s, v5.4s, v10.4s \n"

      // add to out register v28
      "faddp v28.4s, v16.4s, v17.4s \n"
      "faddp v29.4s, v18.4s, v19.4s \n"
      "faddp v28.4s, v28.4s, v29.4s \n"
      "fmla v28.4s, v21.4s, %[w4].s[0]  \n"

      "ld1 {v8.4s}, [%[bias]]  \n"

      // zip
      "zip1 v0.4s, v25.4s, v27.4s  \n"
      "zip2 v2.4s, v25.4s, v27.4s  \n"
      "zip1 v4.4s, v26.4s, v28.4s  \n"
      "zip2 v6.4s, v26.4s, v28.4s  \n"

      // add bias
      "fadd v0.4s, v0.4s, v8.4s  \n"
      "fadd v2.4s, v2.4s, v8.4s  \n"
      "fadd v4.4s, v4.4s, v8.4s  \n"
      "fadd v6.4s, v6.4s, v8.4s  \n"

      // relu
      "fmax v0.4s, v0.4s, v31.4s \n"
      "fmax v2.4s, v2.4s, v31.4s \n"
      "fmax v4.4s, v4.4s, v31.4s \n"
      "fmax v6.4s, v6.4s, v31.4s \n"

      "ext v1.16b, v0.16b, v31.16b, #8  \n"
      "ext v3.16b, v2.16b, v31.16b, #8  \n"
      "ext v5.16b, v4.16b, v31.16b, #8  \n"
      "ext v7.16b, v6.16b, v31.16b, #8  \n"

      // write output
      "str d0, [x0], #8  \n"
      "str d1, [x1], #8  \n"
      "str d2, [x2], #8  \n"
      "str d3, [x3], #8  \n"

      "str d4, [x0]  \n"
      "str d5, [x1]  \n"
      "str d6, [x2]  \n"
      "str d7, [x3]  \n"

      : [din0] "+r"(din0), [din1] "+r"(din1), [din2] "+r"(din2),
        [din3] "+r"(din3), [din4] "+r"(din4), [din5] "+r"(din5),
        [din6] "+r"(din6), [din7] "+r"(din7), [doutl] "+r"(doutl_ptr)
      : [s_12] "r"(s_12), [w0] "w"(w0), [w1] "w"(w1), [w2] "w"(w2),
        [w3] "w"(w3), [w4] "w"(w4), [bias] "r"(bias)
      : "memory", "x0", "x1", "x2", "x3", "v0", "v1", "v2", "v3", "v5", "v7",
        "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17",
        "v18", "v19", "v20", "v21", "v25", "v26", "v27", "v28", "v29", "v31");
}

void conv_depthwise_5x5s1_impl(const float* din, float* dout, int num,
                               int ch_out, int h_out, int w_out, int ch_in,
                               int h_in, int w_in, const float* weights,
                               const float* bias, int pad, bool flag_bias,
                               bool flag_relu, ARMContext* ctx) {
  float* zero_ptr = ctx->workspace_data<float>();
  memset(zero_ptr, 0, w_in * sizeof(float));
  float* write_ptr = zero_ptr + w_in;
  int pad_new = pad > 4 ? 4 : pad;
  int pad_0 = pad - pad_new;
  int h_out_new = h_out - 2 * pad_0;
  int mid_out = w_out - 2 * pad;
  int mid_cnt = mid_out >> 2;
  int mid_remain = mid_out - (mid_cnt << 2);
  int pad_cnt = pad_0 >> 2;
  int pad_remain = pad_0 - (pad_cnt << 2);
  int bias_cnt = (w_out * pad_0) >> 2;
  int bias_remain = (w_out * pad_0) - (bias_cnt << 2);
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
      float bias_c = flag_bias ? bias[c] : 0.f;
      float vbias[4] = {bias_c, bias_c, bias_c, bias_c};
      float32x4_t vbias_c = vdupq_n_f32(bias_c);
      if (flag_bias) {
        //! deal with h_out pad_0 line with bias
        for (int i = 0; i < bias_cnt; ++i) {
          vst1q_f32(dout_ch, vbias_c);
          dout_ch += 4;
        }
        for (int i = 0; i < bias_remain; ++i) {
          *dout_ch++ = bias_c;
        }
      } else {
        //! deal with h_out pad_0 line without bias
        for (int i = 0; i < pad_0; ++i) {
          memset(dout_ch, 0x00, w_out * sizeof(float));
          dout_ch += w_out;
        }
      }
      const float* din_list[8];
      const float* dinl[8];
      //! set din ptr with zero buffer
      for (int i = 0; i < pad_new; ++i) {
        din_list[i] = zero_ptr;
      }
      //! set din ptr with input data
      for (int i = pad_new; i < 8; ++i) {
        din_list[i] = din_ch;
        din_ch += w_in;
      }

      //! every h loop, deal with 4 line output
      float* dout0 = dout_ch;
      float* dout1 = dout0 + w_out;
      float* dout2 = dout1 + w_out;
      float* dout3 = dout2 + w_out;

      //! load weights to neon register
      const float* weights_c = weights + c * weights_saptial_size;

      float32x4_t w5;
      float32x4_t w6;
      float32x4_t w0 = vld1q_f32(weights_c);
      float32x4_t w1 = vld1q_f32(weights_c + 5);
      float32x4_t w2 = vld1q_f32(weights_c + 10);
      float32x4_t w3 = vld1q_f32(weights_c + 15);
      float32x4_t w4 = vld1q_f32(weights_c + 20);
      w5 = vsetq_lane_f32(weights_c[4], w5, 0);
      w5 = vsetq_lane_f32(weights_c[9], w5, 1);
      w5 = vsetq_lane_f32(weights_c[14], w5, 2);
      w5 = vsetq_lane_f32(weights_c[19], w5, 3);
      w6 = vsetq_lane_f32(weights_c[24], w6, 0);

      //! h loop
      for (int h = 0; h < h_out_new; h += 4) {
        //! (h - pad_new) + 7 > h_in - 1
        if (h + 8 - pad_new > h_in) {
          switch (h + 8 - pad_new - h_in) {
            case 7:
              din_list[1] = zero_ptr;
            case 6:
              din_list[2] = zero_ptr;
            case 5:
              din_list[3] = zero_ptr;
            case 4:
              din_list[4] = zero_ptr;
            case 3:
              din_list[5] = zero_ptr;
            case 2:
              din_list[6] = zero_ptr;
            case 1:
              din_list[7] = zero_ptr;
            default:
              break;
          }
        }
        if (h + 4 > h_out_new) {
          switch (h + 4 - h_out_new) {
            case 3:
              dout1 = write_ptr;
            case 2:
              dout2 = write_ptr;
            case 1:
              dout3 = write_ptr;
            default:
              break;
          }
        }

        //! every h loop, deal with 8 line input
        dinl[0] = din_list[0];
        dinl[1] = din_list[1];
        dinl[2] = din_list[2];
        dinl[3] = din_list[3];
        dinl[4] = din_list[4];
        dinl[5] = din_list[5];
        dinl[6] = din_list[6];
        dinl[7] = din_list[7];

        const float* weights_ptr = weights_c;
        float* dout_ptr0 = dout0;
        float* dout_ptr1 = dout1;
        float* dout_ptr2 = dout2;
        float* dout_ptr3 = dout3;
        if (flag_bias) {
          //! deal with w_out pad_0 column pre with bias
          for (int i = 0; i < pad_cnt; i++) {
            vst1q_f32(dout_ptr0, vbias_c);
            vst1q_f32(dout_ptr1, vbias_c);
            vst1q_f32(dout_ptr2, vbias_c);
            vst1q_f32(dout_ptr3, vbias_c);
            dout_ptr0 += 4;
            dout_ptr1 += 4;
            dout_ptr2 += 4;
            dout_ptr3 += 4;
          }
          for (int i = 0; i < pad_remain; ++i) {
            *dout_ptr0++ = bias_c;
            *dout_ptr1++ = bias_c;
            *dout_ptr2++ = bias_c;
            *dout_ptr3++ = bias_c;
          }
        } else {
          //! deal with w_out pad_0 column pre without bias
          memset(dout_ptr0, 0x00, pad_0 * sizeof(float));
          memset(dout_ptr1, 0x00, pad_0 * sizeof(float));
          memset(dout_ptr2, 0x00, pad_0 * sizeof(float));
          memset(dout_ptr3, 0x00, pad_0 * sizeof(float));
          dout_ptr0 += pad_0;
          dout_ptr1 += pad_0;
          dout_ptr2 += pad_0;
          dout_ptr3 += pad_0;
        }
        //! deal with w_out pad_new column pre
        switch (pad_new) {
          case 4:
            compute_four_out_extract_pre(dinl[0], dinl[1], dinl[2], dinl[3],
                                         dinl[4], dinl[5], dinl[6], dinl[7],
                                         dout_ptr0, dout_ptr1, dout_ptr2,
                                         dout_ptr3, weights_ptr, vbias);
            dout_ptr0 += 4;
            dout_ptr1 += 4;
            dout_ptr2 += 4;
            dout_ptr3 += 4;
            break;
          case 3:
            compute_three_out_extract_pre(dinl[0], dinl[1], dinl[2], dinl[3],
                                          dinl[4], dinl[5], dinl[6], dinl[7],
                                          dout_ptr0, dout_ptr1, dout_ptr2,
                                          dout_ptr3, weights_ptr, vbias);
            dout_ptr0 += 3;
            dout_ptr1 += 3;
            dout_ptr2 += 3;
            dout_ptr3 += 3;
            break;
          case 2:
            compute_two_out_extract_pre(dinl[0], dinl[1], dinl[2], dinl[3],
                                        dinl[4], dinl[5], dinl[6], dinl[7],
                                        dout_ptr0, dout_ptr1, dout_ptr2,
                                        dout_ptr3, weights_ptr, vbias);
            dout_ptr0 += 2;
            dout_ptr1 += 2;
            dout_ptr2 += 2;
            dout_ptr3 += 2;
            break;
          case 1:
            compute_one_out_extract_pre(dinl[0], dinl[1], dinl[2], dinl[3],
                                        dinl[4], dinl[5], dinl[6], dinl[7],
                                        dout_ptr0, dout_ptr1, dout_ptr2,
                                        dout_ptr3, weights_ptr, vbias);
            dout_ptr0 += 1;
            dout_ptr1 += 1;
            dout_ptr2 += 1;
            dout_ptr3 += 1;
            break;
        }
        //! mid loop
        if (mid_cnt > 0) {
          void* dinl_ptr = reinterpret_cast<void*>(dinl);
          int mid_loop = mid_cnt;
          asm volatile(
              //! din: v7-v14
              //! dout: v15-v18
              "mov x0, #0  \n"
              "mov x1, #4  \n"
              "ldp x2, x3, [%[dinl]], #16  \n"
              "ldp x4, x5, [%[dinl]], #16  \n"
              "ldp x6, x7, [%[dinl]], #16  \n"
              "ldp x8, x9, [%[dinl]], #16  \n"

              "ld1 {v7.4s} , [x2], x1  \n"
              "ld1 {v8.4s} , [x3], x1  \n"
              "ld1 {v9.4s} , [x4], x1  \n"
              "ld1 {v10.4s}, [x5], x1  \n"
              "ld1 {v11.4s}, [x6], x1  \n"
              "ld1 {v12.4s}, [x7], x1  \n"
              "ld1 {v13.4s}, [x8], x1  \n"
              "ld1 {v14.4s}, [x9], x1  \n"

              //! load bias
              "ld1 {v19.4s}, [%[bias]]  \n"

              "1: \n"
              //! add bias to output
              "mov v15.16b, v19.16b  \n"
              "mov v16.16b, v19.16b  \n"
              "mov v17.16b, v19.16b  \n"
              "mov v18.16b, v19.16b  \n"

              //! loop cnt is even, prefetch 64 Byte to l1 cache
              "cmp x0, #1  \n"
              "bne 2f  \n"
              "mov x0, #0  \n"
              "prfm pldl1keep, [x2]  \n"
              "prfm pldl1keep, [x3]  \n"
              "prfm pldl1keep, [x4]  \n"
              "prfm pldl1keep, [x5]  \n"
              "prfm pldl1keep, [x6]  \n"
              "prfm pldl1keep, [x7]  \n"
              "prfm pldl1keep, [x8]  \n"
              "prfm pldl1keep, [x9]  \n"

              "2:  \n"
              // weights col 0
              "fmla v15.4s, v7.4s , %[w0].s[0]  \n"
              "fmla v16.4s, v8.4s , %[w0].s[0]  \n"
              "fmla v17.4s, v9.4s , %[w0].s[0]  \n"
              "fmla v18.4s, v10.4s, %[w0].s[0]  \n"

              "fmla v15.4s, v8.4s , %[w1].s[0]  \n"
              "fmla v16.4s, v9.4s , %[w1].s[0]  \n"
              "fmla v17.4s, v10.4s, %[w1].s[0]  \n"
              "fmla v18.4s, v11.4s, %[w1].s[0]  \n"

              "ld1 {v7.4s}, [x2], x1  \n"
              "ld1 {v8.4s}, [x3], x1  \n"

              "fmla v15.4s, v9.4s , %[w2].s[0]  \n"
              "fmla v16.4s, v10.4s, %[w2].s[0]  \n"
              "fmla v17.4s, v11.4s, %[w2].s[0]  \n"
              "fmla v18.4s, v12.4s, %[w2].s[0]  \n"

              "fmla v15.4s, v10.4s, %[w3].s[0]  \n"
              "fmla v16.4s, v11.4s, %[w3].s[0]  \n"
              "fmla v17.4s, v12.4s, %[w3].s[0]  \n"
              "fmla v18.4s, v13.4s, %[w3].s[0]  \n"

              "ld1 {v9.4s} , [x4], x1  \n"
              "ld1 {v10.4s}, [x5], x1  \n"

              "fmla v15.4s, v11.4s, %[w4].s[0]  \n"
              "fmla v16.4s, v12.4s, %[w4].s[0]  \n"
              "fmla v17.4s, v13.4s, %[w4].s[0]  \n"
              "fmla v18.4s, v14.4s, %[w4].s[0]  \n"

              "ld1 {v11.4s}, [x6], x1  \n"
              "ld1 {v12.4s}, [x7], x1  \n"

              // weights col 1
              "fmla v15.4s, v7.4s , %[w0].s[1]  \n"
              "fmla v16.4s, v8.4s , %[w0].s[1]  \n"
              "fmla v17.4s, v9.4s , %[w0].s[1]  \n"
              "fmla v18.4s, v10.4s, %[w0].s[1]  \n"

              "ld1 {v13.4s}, [x8], x1  \n"
              "ld1 {v14.4s}, [x9], x1  \n"

              "fmla v15.4s, v8.4s , %[w1].s[1]  \n"
              "fmla v16.4s, v9.4s , %[w1].s[1]  \n"
              "fmla v17.4s, v10.4s, %[w1].s[1]  \n"
              "fmla v18.4s, v11.4s, %[w1].s[1]  \n"

              "ld1 {v7.4s}, [x2], x1  \n"
              "ld1 {v8.4s}, [x3], x1  \n"

              "fmla v15.4s, v9.4s , %[w2].s[1]  \n"
              "fmla v16.4s, v10.4s, %[w2].s[1]  \n"
              "fmla v17.4s, v11.4s, %[w2].s[1]  \n"
              "fmla v18.4s, v12.4s, %[w2].s[1]  \n"

              "fmla v15.4s, v10.4s, %[w3].s[1]  \n"
              "fmla v16.4s, v11.4s, %[w3].s[1]  \n"
              "fmla v17.4s, v12.4s, %[w3].s[1]  \n"
              "fmla v18.4s, v13.4s, %[w3].s[1]  \n"

              "ld1 {v9.4s} , [x4], x1  \n"
              "ld1 {v10.4s}, [x5], x1  \n"

              "fmla v15.4s, v11.4s, %[w4].s[1]  \n"
              "fmla v16.4s, v12.4s, %[w4].s[1]  \n"
              "fmla v17.4s, v13.4s, %[w4].s[1]  \n"
              "fmla v18.4s, v14.4s, %[w4].s[1]  \n"

              "ld1 {v11.4s}, [x6], x1  \n"
              "ld1 {v12.4s}, [x7], x1  \n"

              // weights col 2
              "fmla v15.4s, v7.4s , %[w0].s[2]  \n"
              "fmla v16.4s, v8.4s , %[w0].s[2]  \n"
              "fmla v17.4s, v9.4s , %[w0].s[2]  \n"
              "fmla v18.4s, v10.4s, %[w0].s[2]  \n"

              "ld1 {v13.4s}, [x8], x1  \n"
              "ld1 {v14.4s}, [x9], x1  \n"

              "fmla v15.4s, v8.4s , %[w1].s[2]  \n"
              "fmla v16.4s, v9.4s , %[w1].s[2]  \n"
              "fmla v17.4s, v10.4s, %[w1].s[2]  \n"
              "fmla v18.4s, v11.4s, %[w1].s[2]  \n"

              "ld1 {v7.4s}, [x2], x1  \n"
              "ld1 {v8.4s}, [x3], x1  \n"

              "fmla v15.4s, v9.4s , %[w2].s[2]  \n"
              "fmla v16.4s, v10.4s, %[w2].s[2]  \n"
              "fmla v17.4s, v11.4s, %[w2].s[2]  \n"
              "fmla v18.4s, v12.4s, %[w2].s[2]  \n"

              "fmla v15.4s, v10.4s, %[w3].s[2]  \n"
              "fmla v16.4s, v11.4s, %[w3].s[2]  \n"
              "fmla v17.4s, v12.4s, %[w3].s[2]  \n"
              "fmla v18.4s, v13.4s, %[w3].s[2]  \n"

              "ld1 {v9.4s} , [x4], x1  \n"
              "ld1 {v10.4s}, [x5], x1  \n"

              "fmla v15.4s, v11.4s, %[w4].s[2]  \n"
              "fmla v16.4s, v12.4s, %[w4].s[2]  \n"
              "fmla v17.4s, v13.4s, %[w4].s[2]  \n"
              "fmla v18.4s, v14.4s, %[w4].s[2]  \n"

              "ld1 {v11.4s}, [x6], x1  \n"
              "ld1 {v12.4s}, [x7], x1  \n"

              // weights col 3
              "fmla v15.4s, v7.4s , %[w0].s[3] \n"
              "fmla v16.4s, v8.4s , %[w0].s[3] \n"
              "fmla v17.4s, v9.4s , %[w0].s[3] \n"
              "fmla v18.4s, v10.4s, %[w0].s[3] \n"

              "ld1 {v13.4s}, [x8], x1  \n"
              "ld1 {v14.4s}, [x9], x1  \n"

              "fmla v15.4s, v8.4s , %[w1].s[3]  \n"
              "fmla v16.4s, v9.4s , %[w1].s[3]  \n"
              "fmla v17.4s, v10.4s, %[w1].s[3]  \n"
              "fmla v18.4s, v11.4s, %[w1].s[3]  \n"

              "ld1 {v7.4s}, [x2], x1  \n"
              "ld1 {v8.4s}, [x3], x1  \n"

              "fmla v15.4s, v9.4s , %[w2].s[3]  \n"
              "fmla v16.4s, v10.4s, %[w2].s[3]  \n"
              "fmla v17.4s, v11.4s, %[w2].s[3]  \n"
              "fmla v18.4s, v12.4s, %[w2].s[3]  \n"

              "fmla v15.4s, v10.4s, %[w3].s[3]  \n"
              "fmla v16.4s, v11.4s, %[w3].s[3]  \n"
              "fmla v17.4s, v12.4s, %[w3].s[3]  \n"
              "fmla v18.4s, v13.4s, %[w3].s[3]  \n"

              "ld1 {v9.4s} , [x4], x1  \n"
              "ld1 {v10.4s}, [x5], x1  \n"

              "fmla v15.4s, v11.4s, %[w4].s[3]  \n"
              "fmla v16.4s, v12.4s, %[w4].s[3]  \n"
              "fmla v17.4s, v13.4s, %[w4].s[3]  \n"
              "fmla v18.4s, v14.4s, %[w4].s[3]  \n"

              "ld1 {v11.4s}, [x6], x1  \n"
              "ld1 {v12.4s}, [x7], x1  \n"

              // weights col 4
              "fmla v15.4s, v7.4s, %[w5].s[0]  \n"
              "fmla v16.4s, v8.4s, %[w5].s[0]  \n"
              "fmla v17.4s, v9.4s, %[w5].s[0]  \n"
              "fmla v18.4s, v10.4s, %[w5].s[0] \n"

              "ld1 {v13.4s}, [x8], x1  \n"
              "ld1 {v14.4s}, [x9], x1  \n"

              "fmla v15.4s, v8.4s, %[w5].s[1]   \n"
              "fmla v16.4s, v9.4s, %[w5].s[1]   \n"
              "fmla v17.4s, v10.4s, %[w5].s[1]  \n"
              "fmla v18.4s, v11.4s, %[w5].s[1]  \n"

              "fmla v15.4s, v9.4s , %[w5].s[2]  \n"
              "fmla v16.4s, v10.4s, %[w5].s[2]  \n"
              "fmla v17.4s, v11.4s, %[w5].s[2]  \n"
              "fmla v18.4s, v12.4s, %[w5].s[2]  \n"

              "fmla v15.4s, v10.4s, %[w5].s[3]  \n"
              "fmla v16.4s, v11.4s, %[w5].s[3]  \n"
              "fmla v17.4s, v12.4s, %[w5].s[3]  \n"
              "fmla v18.4s, v13.4s, %[w5].s[3]  \n"

              "fmla v15.4s, v11.4s, %[w6].s[0]  \n"
              "fmla v16.4s, v12.4s, %[w6].s[0]  \n"
              "fmla v17.4s, v13.4s, %[w6].s[0]  \n"
              "fmla v18.4s, v14.4s, %[w6].s[0]  \n"

              "st1 {v15.4s}, [%[dout0]], #16  \n"
              "st1 {v16.4s}, [%[dout1]], #16  \n"
              "st1 {v17.4s}, [%[dout2]], #16  \n"
              "st1 {v18.4s}, [%[dout3]], #16  \n"

              "subs %w[cnt], %w[cnt], #1  \n"
              "add x0, x0, #1  \n"
              "bne 1b  \n"

              : [dout0] "+r"(dout_ptr0), [dout1] "+r"(dout_ptr1),
                [dout2] "+r"(dout_ptr2), [dout3] "+r"(dout_ptr3),
                [cnt] "+r"(mid_loop), [dinl] "+r"(dinl_ptr)
              : [w0] "w"(w0), [w1] "w"(w1), [w2] "w"(w2), [w3] "w"(w3),
                [w4] "w"(w4), [w5] "w"(w5), [w6] "w"(w6), [bias] "r"(vbias)
              : "cc", "memory", "x0", "x1", "x2", "x3", "x4", "x5", "x6", "x7",
                "x8", "x9", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14",
                "v15", "v16", "v17", "v18", "v19");
        }
        dinl[0] += 4 * mid_cnt;
        dinl[1] += 4 * mid_cnt;
        dinl[2] += 4 * mid_cnt;
        dinl[3] += 4 * mid_cnt;
        dinl[4] += 4 * mid_cnt;
        dinl[5] += 4 * mid_cnt;
        dinl[6] += 4 * mid_cnt;
        dinl[7] += 4 * mid_cnt;
        //! deal with mid remain
        for (int i = 0; i < mid_remain; ++i) {
          compute_one_out_without_extract(
              dinl[0], dinl[1], dinl[2], dinl[3], dinl[4], dinl[5], dinl[6],
              dinl[7], dout_ptr0, dout_ptr1, dout_ptr2, dout_ptr3, w0, w1, w2,
              w3, w4, w5, w6, vbias);
          dinl[0]++;
          dinl[1]++;
          dinl[2]++;
          dinl[3]++;
          dinl[4]++;
          dinl[5]++;
          dinl[6]++;
          dinl[7]++;

          dout_ptr0++;
          dout_ptr1++;
          dout_ptr2++;
          dout_ptr3++;
        }
        //! deal with w_out pad_new column post
        switch (pad_new) {
          case 4:
            compute_four_out_extract_post(dinl[0], dinl[1], dinl[2], dinl[3],
                                          dinl[4], dinl[5], dinl[6], dinl[7],
                                          dout_ptr0, dout_ptr1, dout_ptr2,
                                          dout_ptr3, w0, w1, w2, w3, w4, vbias);
            dout_ptr0 += 4;
            dout_ptr1 += 4;
            dout_ptr2 += 4;
            dout_ptr3 += 4;
            break;
          case 3:
            compute_three_out_extract_post(
                dinl[0], dinl[1], dinl[2], dinl[3], dinl[4], dinl[5], dinl[6],
                dinl[7], dout_ptr0, dout_ptr1, dout_ptr2, dout_ptr3, w0, w1, w2,
                w3, w4, vbias);
            dout_ptr0 += 3;
            dout_ptr1 += 3;
            dout_ptr2 += 3;
            dout_ptr3 += 3;
            break;
          case 2:
            compute_two_out_extract_post(dinl[0], dinl[1], dinl[2], dinl[3],
                                         dinl[4], dinl[5], dinl[6], dinl[7],
                                         dout_ptr0, dout_ptr1, dout_ptr2,
                                         dout_ptr3, w0, w1, w2, w3, w4, vbias);
            dout_ptr0 += 2;
            dout_ptr1 += 2;
            dout_ptr2 += 2;
            dout_ptr3 += 2;
            break;
          case 1:
            compute_one_out_extract_post(dinl[0], dinl[1], dinl[2], dinl[3],
                                         dinl[4], dinl[5], dinl[6], dinl[7],
                                         dout_ptr0, dout_ptr1, dout_ptr2,
                                         dout_ptr3, w0, w1, w2, w3, w4, vbias);
            dout_ptr0 += 1;
            dout_ptr1 += 1;
            dout_ptr2 += 1;
            dout_ptr3 += 1;
            break;
        }

        if (flag_bias) {
          //! deal with w_out pad_0 column post with bias
          memcpy(dout_ptr0, dout0, pad_0 * sizeof(float));
          memcpy(dout_ptr1, dout1, pad_0 * sizeof(float));
          memcpy(dout_ptr2, dout2, pad_0 * sizeof(float));
          memcpy(dout_ptr3, dout3, pad_0 * sizeof(float));
        } else {
          //! deal with w_out pad_0 column post without bias
          memset(dout_ptr0, 0x00, pad_0 * sizeof(float));
          memset(dout_ptr1, 0x00, pad_0 * sizeof(float));
          memset(dout_ptr2, 0x00, pad_0 * sizeof(float));
          memset(dout_ptr3, 0x00, pad_0 * sizeof(float));
        }

        din_list[0] = din_list[4];
        din_list[1] = din_list[5];
        din_list[2] = din_list[6];
        din_list[3] = din_list[7];
        din_list[4] = din_list[3] + w_in;
        din_list[5] = din_list[4] + w_in;
        din_list[6] = din_list[5] + w_in;
        din_list[7] = din_list[6] + w_in;

        dout0 = dout3 + w_out;
        dout1 = dout0 + w_out;
        dout2 = dout1 + w_out;
        dout3 = dout2 + w_out;
      }
      float* dout_pad_end = dout_ch + h_out_new * w_out;
      if (flag_bias) {
        //! deal with h_out pad_0 line with bias
        memcpy(reinterpret_cast<void*>(dout_pad_end), dout_ch - pad_0 * w_out,
               pad_0 * w_out * sizeof(float));
      } else {
        //! deal with h_out pad_0 line without bias
        memset(reinterpret_cast<void*>(dout_pad_end), 0x00,
               pad_0 * w_out * sizeof(float));
      }
    }
  }
}

void conv_depthwise_5x5s1_relu_impl(const float* din, float* dout, int num,
                                    int ch_out, int h_out, int w_out, int ch_in,
                                    int h_in, int w_in, const float* weights,
                                    const float* bias, int pad, bool flag_bias,
                                    bool flag_relu, ARMContext* ctx) {
  float* zero_ptr = ctx->workspace_data<float>();
  memset(zero_ptr, 0, w_in * sizeof(float));
  float* write_ptr = zero_ptr + w_in;
  int pad_new = pad > 4 ? 4 : pad;
  int pad_0 = pad - pad_new;
  int h_out_new = h_out - 2 * pad_0;
  int mid_out = w_out - 2 * pad;
  int mid_cnt = mid_out >> 2;
  int mid_remain = mid_out - (mid_cnt << 2);
  int pad_cnt = pad_0 >> 2;
  int pad_remain = pad_0 - (pad_cnt << 2);
  int bias_cnt = (w_out * pad_0) >> 2;
  int bias_remain = (w_out * pad_0) - (bias_cnt << 2);
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
      float bias_c = flag_bias ? bias[c] : 0.f;
      float bias_relu = bias_c > 0.f ? bias_c : 0.f;
      float vbias[4] = {bias_c, bias_c, bias_c, bias_c};
      float32x4_t vbias_c = vdupq_n_f32(bias_relu);
      if (flag_bias) {
        //! deal with h_out pad_0 line with bias
        for (int i = 0; i < bias_cnt; ++i) {
          vst1q_f32(dout_ch, vbias_c);
          dout_ch += 4;
        }
        for (int i = 0; i < bias_remain; ++i) {
          *dout_ch++ = bias_relu;
        }
      } else {
        //! deal with h_out pad_0 line without bias
        for (int i = 0; i < pad_0; ++i) {
          memset(dout_ch, 0x00, w_out * sizeof(float));
          dout_ch += w_out;
        }
      }
      const float* din_list[8];
      const float* dinl[8];
      //! set din ptr with zero buffer
      for (int i = 0; i < pad_new; ++i) {
        din_list[i] = zero_ptr;
      }
      //! set din ptr with input data
      for (int i = pad_new; i < 8; ++i) {
        din_list[i] = din_ch;
        din_ch += w_in;
      }

      //! every h loop, deal with 4 line output
      float* dout0 = dout_ch;
      float* dout1 = dout0 + w_out;
      float* dout2 = dout1 + w_out;
      float* dout3 = dout2 + w_out;

      //! load weights to neon register
      const float* weights_c = weights + c * weights_saptial_size;

      float32x4_t w5;
      float32x4_t w6;
      float32x4_t w0 = vld1q_f32(weights_c);
      float32x4_t w1 = vld1q_f32(weights_c + 5);
      float32x4_t w2 = vld1q_f32(weights_c + 10);
      float32x4_t w3 = vld1q_f32(weights_c + 15);
      float32x4_t w4 = vld1q_f32(weights_c + 20);
      w5 = vsetq_lane_f32(weights_c[4], w5, 0);
      w5 = vsetq_lane_f32(weights_c[9], w5, 1);
      w5 = vsetq_lane_f32(weights_c[14], w5, 2);
      w5 = vsetq_lane_f32(weights_c[19], w5, 3);
      w6 = vsetq_lane_f32(weights_c[24], w6, 0);

      //! h loop
      for (int h = 0; h < h_out_new; h += 4) {
        //! (h - pad_new) + 7 > h_in - 1
        if (h + 8 - pad_new > h_in) {
          switch (h + 8 - pad_new - h_in) {
            case 7:
              din_list[1] = zero_ptr;
            case 6:
              din_list[2] = zero_ptr;
            case 5:
              din_list[3] = zero_ptr;
            case 4:
              din_list[4] = zero_ptr;
            case 3:
              din_list[5] = zero_ptr;
            case 2:
              din_list[6] = zero_ptr;
            case 1:
              din_list[7] = zero_ptr;
            default:
              break;
          }
        }
        if (h + 4 > h_out_new) {
          switch (h + 4 - h_out_new) {
            case 3:
              dout1 = write_ptr;
            case 2:
              dout2 = write_ptr;
            case 1:
              dout3 = write_ptr;
            default:
              break;
          }
        }

        //! every h loop, deal with 8 line input
        dinl[0] = din_list[0];
        dinl[1] = din_list[1];
        dinl[2] = din_list[2];
        dinl[3] = din_list[3];
        dinl[4] = din_list[4];
        dinl[5] = din_list[5];
        dinl[6] = din_list[6];
        dinl[7] = din_list[7];

        const float* weights_ptr = weights_c;
        float* dout_ptr0 = dout0;
        float* dout_ptr1 = dout1;
        float* dout_ptr2 = dout2;
        float* dout_ptr3 = dout3;
        if (flag_bias) {
          //! deal with w_out pad_0 column pre with bias
          for (int i = 0; i < pad_cnt; i++) {
            vst1q_f32(dout_ptr0, vbias_c);
            vst1q_f32(dout_ptr1, vbias_c);
            vst1q_f32(dout_ptr2, vbias_c);
            vst1q_f32(dout_ptr3, vbias_c);
            dout_ptr0 += 4;
            dout_ptr1 += 4;
            dout_ptr2 += 4;
            dout_ptr3 += 4;
          }
          for (int i = 0; i < pad_remain; ++i) {
            *dout_ptr0++ = bias_relu;
            *dout_ptr1++ = bias_relu;
            *dout_ptr2++ = bias_relu;
            *dout_ptr3++ = bias_relu;
          }
        } else {
          //! deal with w_out pad_0 column pre without bias
          memset(dout_ptr0, 0x00, pad_0 * sizeof(float));
          memset(dout_ptr1, 0x00, pad_0 * sizeof(float));
          memset(dout_ptr2, 0x00, pad_0 * sizeof(float));
          memset(dout_ptr3, 0x00, pad_0 * sizeof(float));
          dout_ptr0 += pad_0;
          dout_ptr1 += pad_0;
          dout_ptr2 += pad_0;
          dout_ptr3 += pad_0;
        }
        //! deal with w_out pad_new column pre
        switch (pad_new) {
          case 4:
            compute_four_out_extract_pre_relu(
                dinl[0], dinl[1], dinl[2], dinl[3], dinl[4], dinl[5], dinl[6],
                dinl[7], dout_ptr0, dout_ptr1, dout_ptr2, dout_ptr3,
                weights_ptr, vbias);
            dout_ptr0 += 4;
            dout_ptr1 += 4;
            dout_ptr2 += 4;
            dout_ptr3 += 4;
            break;
          case 3:
            compute_three_out_extract_pre_relu(
                dinl[0], dinl[1], dinl[2], dinl[3], dinl[4], dinl[5], dinl[6],
                dinl[7], dout_ptr0, dout_ptr1, dout_ptr2, dout_ptr3,
                weights_ptr, vbias);
            dout_ptr0 += 3;
            dout_ptr1 += 3;
            dout_ptr2 += 3;
            dout_ptr3 += 3;
            break;
          case 2:
            compute_two_out_extract_pre_relu(dinl[0], dinl[1], dinl[2], dinl[3],
                                             dinl[4], dinl[5], dinl[6], dinl[7],
                                             dout_ptr0, dout_ptr1, dout_ptr2,
                                             dout_ptr3, weights_ptr, vbias);
            dout_ptr0 += 2;
            dout_ptr1 += 2;
            dout_ptr2 += 2;
            dout_ptr3 += 2;
            break;
          case 1:
            compute_one_out_extract_pre_relu(dinl[0], dinl[1], dinl[2], dinl[3],
                                             dinl[4], dinl[5], dinl[6], dinl[7],
                                             dout_ptr0, dout_ptr1, dout_ptr2,
                                             dout_ptr3, weights_ptr, vbias);
            dout_ptr0 += 1;
            dout_ptr1 += 1;
            dout_ptr2 += 1;
            dout_ptr3 += 1;
            break;
        }
        //! mid loop
        if (mid_cnt > 0) {
          void* dinl_ptr = reinterpret_cast<void*>(dinl);
          int mid_loop = mid_cnt;
          asm volatile(
              //! din: v7-v14
              //! dout: v15-v18
              "mov x0, #0  \n"
              "mov x1, #4  \n"
              "movi v31.4s, #0  \n"
              "ldp x2, x3, [%[dinl]], #16  \n"
              "ldp x4, x5, [%[dinl]], #16  \n"
              "ldp x6, x7, [%[dinl]], #16  \n"
              "ldp x8, x9, [%[dinl]], #16  \n"

              "ld1 {v7.4s} , [x2], x1  \n"
              "ld1 {v8.4s} , [x3], x1  \n"
              "ld1 {v9.4s} , [x4], x1  \n"
              "ld1 {v10.4s}, [x5], x1  \n"
              "ld1 {v11.4s}, [x6], x1  \n"
              "ld1 {v12.4s}, [x7], x1  \n"
              "ld1 {v13.4s}, [x8], x1  \n"
              "ld1 {v14.4s}, [x9], x1  \n"

              //! load bias
              "ld1 {v19.4s}, [%[bias]]  \n"

              "1: \n"
              //! add bias to output
              "mov v15.16b, v19.16b  \n"
              "mov v16.16b, v19.16b  \n"
              "mov v17.16b, v19.16b  \n"
              "mov v18.16b, v19.16b  \n"

              //! loop cnt is even, prefetch 64 Byte to l1 cache
              "cmp x0, #1  \n"
              "bne 2f  \n"
              "mov x0, #0  \n"
              "prfm pldl1keep, [x2]  \n"
              "prfm pldl1keep, [x3]  \n"
              "prfm pldl1keep, [x4]  \n"
              "prfm pldl1keep, [x5]  \n"
              "prfm pldl1keep, [x6]  \n"
              "prfm pldl1keep, [x7]  \n"
              "prfm pldl1keep, [x8]  \n"
              "prfm pldl1keep, [x9]  \n"

              "2:  \n"
              // weights col 0
              "fmla v15.4s, v7.4s , %[w0].s[0]  \n"
              "fmla v16.4s, v8.4s , %[w0].s[0]  \n"
              "fmla v17.4s, v9.4s , %[w0].s[0]  \n"
              "fmla v18.4s, v10.4s, %[w0].s[0]  \n"

              "fmla v15.4s, v8.4s , %[w1].s[0]  \n"
              "fmla v16.4s, v9.4s , %[w1].s[0]  \n"
              "fmla v17.4s, v10.4s, %[w1].s[0]  \n"
              "fmla v18.4s, v11.4s, %[w1].s[0]  \n"

              "ld1 {v7.4s}, [x2], x1  \n"
              "ld1 {v8.4s}, [x3], x1  \n"

              "fmla v15.4s, v9.4s , %[w2].s[0]  \n"
              "fmla v16.4s, v10.4s, %[w2].s[0]  \n"
              "fmla v17.4s, v11.4s, %[w2].s[0]  \n"
              "fmla v18.4s, v12.4s, %[w2].s[0]  \n"

              "fmla v15.4s, v10.4s, %[w3].s[0]  \n"
              "fmla v16.4s, v11.4s, %[w3].s[0]  \n"
              "fmla v17.4s, v12.4s, %[w3].s[0]  \n"
              "fmla v18.4s, v13.4s, %[w3].s[0]  \n"

              "ld1 {v9.4s} , [x4], x1  \n"
              "ld1 {v10.4s}, [x5], x1  \n"

              "fmla v15.4s, v11.4s, %[w4].s[0]  \n"
              "fmla v16.4s, v12.4s, %[w4].s[0]  \n"
              "fmla v17.4s, v13.4s, %[w4].s[0]  \n"
              "fmla v18.4s, v14.4s, %[w4].s[0]  \n"

              "ld1 {v11.4s}, [x6], x1  \n"
              "ld1 {v12.4s}, [x7], x1  \n"

              // weights col 1
              "fmla v15.4s, v7.4s , %[w0].s[1]  \n"
              "fmla v16.4s, v8.4s , %[w0].s[1]  \n"
              "fmla v17.4s, v9.4s , %[w0].s[1]  \n"
              "fmla v18.4s, v10.4s, %[w0].s[1]  \n"

              "ld1 {v13.4s}, [x8], x1  \n"
              "ld1 {v14.4s}, [x9], x1  \n"

              "fmla v15.4s, v8.4s , %[w1].s[1]  \n"
              "fmla v16.4s, v9.4s , %[w1].s[1]  \n"
              "fmla v17.4s, v10.4s, %[w1].s[1]  \n"
              "fmla v18.4s, v11.4s, %[w1].s[1]  \n"

              "ld1 {v7.4s}, [x2], x1  \n"
              "ld1 {v8.4s}, [x3], x1  \n"

              "fmla v15.4s, v9.4s , %[w2].s[1]  \n"
              "fmla v16.4s, v10.4s, %[w2].s[1]  \n"
              "fmla v17.4s, v11.4s, %[w2].s[1]  \n"
              "fmla v18.4s, v12.4s, %[w2].s[1]  \n"

              "fmla v15.4s, v10.4s, %[w3].s[1]  \n"
              "fmla v16.4s, v11.4s, %[w3].s[1]  \n"
              "fmla v17.4s, v12.4s, %[w3].s[1]  \n"
              "fmla v18.4s, v13.4s, %[w3].s[1]  \n"

              "ld1 {v9.4s} , [x4], x1  \n"
              "ld1 {v10.4s}, [x5], x1  \n"

              "fmla v15.4s, v11.4s, %[w4].s[1]  \n"
              "fmla v16.4s, v12.4s, %[w4].s[1]  \n"
              "fmla v17.4s, v13.4s, %[w4].s[1]  \n"
              "fmla v18.4s, v14.4s, %[w4].s[1]  \n"

              "ld1 {v11.4s}, [x6], x1  \n"
              "ld1 {v12.4s}, [x7], x1  \n"

              // weights col 2
              "fmla v15.4s, v7.4s , %[w0].s[2]  \n"
              "fmla v16.4s, v8.4s , %[w0].s[2]  \n"
              "fmla v17.4s, v9.4s , %[w0].s[2]  \n"
              "fmla v18.4s, v10.4s, %[w0].s[2]  \n"

              "ld1 {v13.4s}, [x8], x1  \n"
              "ld1 {v14.4s}, [x9], x1  \n"

              "fmla v15.4s, v8.4s , %[w1].s[2]  \n"
              "fmla v16.4s, v9.4s , %[w1].s[2]  \n"
              "fmla v17.4s, v10.4s, %[w1].s[2]  \n"
              "fmla v18.4s, v11.4s, %[w1].s[2]  \n"

              "ld1 {v7.4s}, [x2], x1  \n"
              "ld1 {v8.4s}, [x3], x1  \n"

              "fmla v15.4s, v9.4s , %[w2].s[2]  \n"
              "fmla v16.4s, v10.4s, %[w2].s[2]  \n"
              "fmla v17.4s, v11.4s, %[w2].s[2]  \n"
              "fmla v18.4s, v12.4s, %[w2].s[2]  \n"

              "fmla v15.4s, v10.4s, %[w3].s[2]  \n"
              "fmla v16.4s, v11.4s, %[w3].s[2]  \n"
              "fmla v17.4s, v12.4s, %[w3].s[2]  \n"
              "fmla v18.4s, v13.4s, %[w3].s[2]  \n"

              "ld1 {v9.4s} , [x4], x1  \n"
              "ld1 {v10.4s}, [x5], x1  \n"

              "fmla v15.4s, v11.4s, %[w4].s[2]  \n"
              "fmla v16.4s, v12.4s, %[w4].s[2]  \n"
              "fmla v17.4s, v13.4s, %[w4].s[2]  \n"
              "fmla v18.4s, v14.4s, %[w4].s[2]  \n"

              "ld1 {v11.4s}, [x6], x1  \n"
              "ld1 {v12.4s}, [x7], x1  \n"

              // weights col 3
              "fmla v15.4s, v7.4s , %[w0].s[3] \n"
              "fmla v16.4s, v8.4s , %[w0].s[3] \n"
              "fmla v17.4s, v9.4s , %[w0].s[3] \n"
              "fmla v18.4s, v10.4s, %[w0].s[3] \n"

              "ld1 {v13.4s}, [x8], x1  \n"
              "ld1 {v14.4s}, [x9], x1  \n"

              "fmla v15.4s, v8.4s , %[w1].s[3]  \n"
              "fmla v16.4s, v9.4s , %[w1].s[3]  \n"
              "fmla v17.4s, v10.4s, %[w1].s[3]  \n"
              "fmla v18.4s, v11.4s, %[w1].s[3]  \n"

              "ld1 {v7.4s}, [x2], x1  \n"
              "ld1 {v8.4s}, [x3], x1  \n"

              "fmla v15.4s, v9.4s , %[w2].s[3]  \n"
              "fmla v16.4s, v10.4s, %[w2].s[3]  \n"
              "fmla v17.4s, v11.4s, %[w2].s[3]  \n"
              "fmla v18.4s, v12.4s, %[w2].s[3]  \n"

              "fmla v15.4s, v10.4s, %[w3].s[3]  \n"
              "fmla v16.4s, v11.4s, %[w3].s[3]  \n"
              "fmla v17.4s, v12.4s, %[w3].s[3]  \n"
              "fmla v18.4s, v13.4s, %[w3].s[3]  \n"

              "ld1 {v9.4s} , [x4], x1  \n"
              "ld1 {v10.4s}, [x5], x1  \n"

              "fmla v15.4s, v11.4s, %[w4].s[3]  \n"
              "fmla v16.4s, v12.4s, %[w4].s[3]  \n"
              "fmla v17.4s, v13.4s, %[w4].s[3]  \n"
              "fmla v18.4s, v14.4s, %[w4].s[3]  \n"

              "ld1 {v11.4s}, [x6], x1  \n"
              "ld1 {v12.4s}, [x7], x1  \n"

              // weights col 4
              "fmla v15.4s, v7.4s, %[w5].s[0]  \n"
              "fmla v16.4s, v8.4s, %[w5].s[0]  \n"
              "fmla v17.4s, v9.4s, %[w5].s[0]  \n"
              "fmla v18.4s, v10.4s, %[w5].s[0] \n"

              "ld1 {v13.4s}, [x8], x1  \n"
              "ld1 {v14.4s}, [x9], x1  \n"

              "fmla v15.4s, v8.4s, %[w5].s[1]   \n"
              "fmla v16.4s, v9.4s, %[w5].s[1]   \n"
              "fmla v17.4s, v10.4s, %[w5].s[1]  \n"
              "fmla v18.4s, v11.4s, %[w5].s[1]  \n"

              "fmla v15.4s, v9.4s , %[w5].s[2]  \n"
              "fmla v16.4s, v10.4s, %[w5].s[2]  \n"
              "fmla v17.4s, v11.4s, %[w5].s[2]  \n"
              "fmla v18.4s, v12.4s, %[w5].s[2]  \n"

              "fmla v15.4s, v10.4s, %[w5].s[3]  \n"
              "fmla v16.4s, v11.4s, %[w5].s[3]  \n"
              "fmla v17.4s, v12.4s, %[w5].s[3]  \n"
              "fmla v18.4s, v13.4s, %[w5].s[3]  \n"

              "fmla v15.4s, v11.4s, %[w6].s[0]  \n"
              "fmla v16.4s, v12.4s, %[w6].s[0]  \n"
              "fmla v17.4s, v13.4s, %[w6].s[0]  \n"
              "fmla v18.4s, v14.4s, %[w6].s[0]  \n"

              "fmax v15.4s, v15.4s, v31.4s  \n"
              "fmax v16.4s, v16.4s, v31.4s  \n"
              "fmax v17.4s, v17.4s, v31.4s  \n"
              "fmax v18.4s, v18.4s, v31.4s  \n"

              "st1 {v15.4s}, [%[dout0]], #16  \n"
              "st1 {v16.4s}, [%[dout1]], #16  \n"
              "st1 {v17.4s}, [%[dout2]], #16  \n"
              "st1 {v18.4s}, [%[dout3]], #16  \n"

              "subs %w[cnt], %w[cnt], #1  \n"
              "add x0, x0, #1  \n"
              "bne 1b  \n"

              : [dout0] "+r"(dout_ptr0), [dout1] "+r"(dout_ptr1),
                [dout2] "+r"(dout_ptr2), [dout3] "+r"(dout_ptr3),
                [cnt] "+r"(mid_loop), [dinl] "+r"(dinl_ptr)
              : [w0] "w"(w0), [w1] "w"(w1), [w2] "w"(w2), [w3] "w"(w3),
                [w4] "w"(w4), [w5] "w"(w5), [w6] "w"(w6), [bias] "r"(vbias)
              : "cc", "memory", "x0", "x1", "x2", "x3", "x4", "x5", "x6", "x7",
                "x8", "x9", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14",
                "v15", "v16", "v17", "v18", "v19", "v31");
        }
        dinl[0] += 4 * mid_cnt;
        dinl[1] += 4 * mid_cnt;
        dinl[2] += 4 * mid_cnt;
        dinl[3] += 4 * mid_cnt;
        dinl[4] += 4 * mid_cnt;
        dinl[5] += 4 * mid_cnt;
        dinl[6] += 4 * mid_cnt;
        dinl[7] += 4 * mid_cnt;
        //! deal with mid remain
        for (int i = 0; i < mid_remain; ++i) {
          compute_one_out_without_extract_relu(
              dinl[0], dinl[1], dinl[2], dinl[3], dinl[4], dinl[5], dinl[6],
              dinl[7], dout_ptr0, dout_ptr1, dout_ptr2, dout_ptr3, w0, w1, w2,
              w3, w4, w5, w6, vbias);
          dinl[0]++;
          dinl[1]++;
          dinl[2]++;
          dinl[3]++;
          dinl[4]++;
          dinl[5]++;
          dinl[6]++;
          dinl[7]++;

          dout_ptr0++;
          dout_ptr1++;
          dout_ptr2++;
          dout_ptr3++;
        }
        //! deal with w_out pad_new column post
        switch (pad_new) {
          case 4:
            compute_four_out_extract_post_relu(
                dinl[0], dinl[1], dinl[2], dinl[3], dinl[4], dinl[5], dinl[6],
                dinl[7], dout_ptr0, dout_ptr1, dout_ptr2, dout_ptr3, w0, w1, w2,
                w3, w4, vbias);
            dout_ptr0 += 4;
            dout_ptr1 += 4;
            dout_ptr2 += 4;
            dout_ptr3 += 4;
            break;
          case 3:
            compute_three_out_extract_post_relu(
                dinl[0], dinl[1], dinl[2], dinl[3], dinl[4], dinl[5], dinl[6],
                dinl[7], dout_ptr0, dout_ptr1, dout_ptr2, dout_ptr3, w0, w1, w2,
                w3, w4, vbias);
            dout_ptr0 += 3;
            dout_ptr1 += 3;
            dout_ptr2 += 3;
            dout_ptr3 += 3;
            break;
          case 2:
            compute_two_out_extract_post_relu(
                dinl[0], dinl[1], dinl[2], dinl[3], dinl[4], dinl[5], dinl[6],
                dinl[7], dout_ptr0, dout_ptr1, dout_ptr2, dout_ptr3, w0, w1, w2,
                w3, w4, vbias);
            dout_ptr0 += 2;
            dout_ptr1 += 2;
            dout_ptr2 += 2;
            dout_ptr3 += 2;
            break;
          case 1:
            compute_one_out_extract_post_relu(
                dinl[0], dinl[1], dinl[2], dinl[3], dinl[4], dinl[5], dinl[6],
                dinl[7], dout_ptr0, dout_ptr1, dout_ptr2, dout_ptr3, w0, w1, w2,
                w3, w4, vbias);
            dout_ptr0 += 1;
            dout_ptr1 += 1;
            dout_ptr2 += 1;
            dout_ptr3 += 1;
            break;
        }

        if (flag_bias) {
          //! deal with w_out pad_0 column post with bias
          memcpy(dout_ptr0, dout0, pad_0 * sizeof(float));
          memcpy(dout_ptr1, dout1, pad_0 * sizeof(float));
          memcpy(dout_ptr2, dout2, pad_0 * sizeof(float));
          memcpy(dout_ptr3, dout3, pad_0 * sizeof(float));
        } else {
          //! deal with w_out pad_0 column post without bias
          memset(dout_ptr0, 0x00, pad_0 * sizeof(float));
          memset(dout_ptr1, 0x00, pad_0 * sizeof(float));
          memset(dout_ptr2, 0x00, pad_0 * sizeof(float));
          memset(dout_ptr3, 0x00, pad_0 * sizeof(float));
        }

        din_list[0] = din_list[4];
        din_list[1] = din_list[5];
        din_list[2] = din_list[6];
        din_list[3] = din_list[7];
        din_list[4] = din_list[3] + w_in;
        din_list[5] = din_list[4] + w_in;
        din_list[6] = din_list[5] + w_in;
        din_list[7] = din_list[6] + w_in;

        dout0 = dout3 + w_out;
        dout1 = dout0 + w_out;
        dout2 = dout1 + w_out;
        dout3 = dout2 + w_out;
      }
      float* dout_pad_end = dout_ch + h_out_new * w_out;
      if (flag_bias) {
        //! deal with h_out pad_0 line with bias
        memcpy(reinterpret_cast<void*>(dout_pad_end), dout_ch - pad_0 * w_out,
               pad_0 * w_out * sizeof(float));
      } else {
        //! deal with h_out pad_0 line without bias
        memset(reinterpret_cast<void*>(dout_pad_end), 0x00,
               pad_0 * w_out * sizeof(float));
      }
    }
  }
}

void conv_depthwise_5x5s1_small_impl(const float* din, float* dout, int num,
                                     int ch_out, int h_out, int w_out,
                                     int ch_in, int h_in, int w_in,
                                     const float* weights, const float* bias,
                                     int pad, bool flag_bias, bool flag_relu,
                                     ARMContext* ctx) {
  int pad_new = pad > 4 ? 4 : pad;
  int pad_0 = pad - pad_new;
  int h_in_new = h_in + 2 * pad_new;
  int w_in_new = w_in + 2 * pad_new;
  int h_out_new = h_out - 2 * pad_0;
  int w_out_new = w_out - 2 * pad_0;
  float zero_ptr[w_in_new + w_out];
  memset(zero_ptr, 0, w_in_new * sizeof(float));
  float* write_ptr = zero_ptr + w_in_new;
  int pad_cnt = pad_0 >> 2;
  int pad_remain = pad_0 - (pad_cnt << 2);
  int bias_cnt = (w_out * pad_0) >> 2;
  int bias_remain = (w_out * pad_0) - (bias_cnt << 2);
  int in_spatial_size = w_in_new * h_in_new;
  int out_spatial_size = w_out * h_out;
  int weights_saptial_size = 25;

  float* din_new = prepad_input(din, num, ch_in, h_in, w_in, pad_new);
  for (int n = 0; n < num; ++n) {
    const float* din_batch = din_new + n * in_spatial_size * ch_in;
    float* dout_batch = dout + n * out_spatial_size * ch_out;
#pragma omp parallel for
    for (int c = 0; c < ch_in; ++c) {
      const float* din_ch = din_batch + c * in_spatial_size;
      float* dout_ch = dout_batch + c * out_spatial_size;
      float bias_c = flag_bias ? bias[c] : 0.f;
      float vbias[4] = {bias_c, bias_c, bias_c, bias_c};
      float32x4_t vbias_c = vdupq_n_f32(bias_c);
      if (flag_bias) {
        //! deal with h_out pad_0 line with bias
        for (int i = 0; i < bias_cnt; ++i) {
          vst1q_f32(dout_ch, vbias_c);
          dout_ch += 4;
        }
        for (int i = 0; i < bias_remain; ++i) {
          *dout_ch++ = bias_c;
        }
      } else {
        //! deal with h_out pad_0 line without bias
        for (int i = 0; i < pad_0; ++i) {
          memset(dout_ch, 0x00, w_out * sizeof(float));
          dout_ch += w_out;
        }
      }
      //! every h loop, deal with 8 line input
      const float* din0 = din_ch;
      const float* din1 = din0 + w_in_new;
      const float* din2 = din1 + w_in_new;
      const float* din3 = din2 + w_in_new;
      const float* din4 = din3 + w_in_new;
      const float* din5 = din4 + w_in_new;
      const float* din6 = din5 + w_in_new;
      const float* din7 = din6 + w_in_new;
      //! every h loop, deal with 4 line output
      float* dout0 = dout_ch;
      float* dout1 = dout0 + w_out;
      float* dout2 = dout1 + w_out;
      float* dout3 = dout2 + w_out;

      //! load weights to neon register
      const float* weights_c = weights + c * weights_saptial_size;

      float32x4_t w5;
      float32x4_t w6;
      float32x4_t w0 = vld1q_f32(weights_c);
      float32x4_t w1 = vld1q_f32(weights_c + 5);
      float32x4_t w2 = vld1q_f32(weights_c + 10);
      float32x4_t w3 = vld1q_f32(weights_c + 15);
      float32x4_t w4 = vld1q_f32(weights_c + 20);
      w5 = vsetq_lane_f32(weights_c[4], w5, 0);
      w5 = vsetq_lane_f32(weights_c[9], w5, 1);
      w5 = vsetq_lane_f32(weights_c[14], w5, 2);
      w5 = vsetq_lane_f32(weights_c[19], w5, 3);
      w6 = vsetq_lane_f32(weights_c[24], w6, 0);
      //! h loop
      for (int h = 0; h < h_out_new; h += 4) {
        //! (h - pad_new) + 7 > h_in - 1
        if (h + 8 > h_in_new) {
          switch (h + 8 - h_in_new) {
            case 7:
              din1 = zero_ptr;
            case 6:
              din2 = zero_ptr;
            case 5:
              din3 = zero_ptr;
            case 4:
              din4 = zero_ptr;
            case 3:
              din5 = zero_ptr;
            case 2:
              din6 = zero_ptr;
            case 1:
              din7 = zero_ptr;
            default:
              break;
          }
        }
        if (h + 4 > h_out_new) {
          switch (h + 4 - h_out_new) {
            case 3:
              dout1 = write_ptr;
            case 2:
              dout2 = write_ptr;
            case 1:
              dout3 = write_ptr;
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
        const float* din_ptr7 = din7;

        const float* weights_ptr = weights_c;
        float* dout_ptr0 = dout0;
        float* dout_ptr1 = dout1;
        float* dout_ptr2 = dout2;
        float* dout_ptr3 = dout3;

        if (flag_bias) {
          //! deal with w_out pad_0 column pre with bias
          for (int i = 0; i < pad_cnt; i++) {
            vst1q_f32(dout_ptr0, vbias_c);
            vst1q_f32(dout_ptr1, vbias_c);
            vst1q_f32(dout_ptr2, vbias_c);
            vst1q_f32(dout_ptr3, vbias_c);
            dout_ptr0 += 4;
            dout_ptr1 += 4;
            dout_ptr2 += 4;
            dout_ptr3 += 4;
          }
          for (int i = 0; i < pad_remain; ++i) {
            *dout_ptr0++ = bias_c;
            *dout_ptr1++ = bias_c;
            *dout_ptr2++ = bias_c;
            *dout_ptr3++ = bias_c;
          }
        } else {
          //! deal with w_out pad_0 column pre without bias
          memset(dout_ptr0, 0x00, pad_0 * sizeof(float));
          memset(dout_ptr1, 0x00, pad_0 * sizeof(float));
          memset(dout_ptr2, 0x00, pad_0 * sizeof(float));
          memset(dout_ptr3, 0x00, pad_0 * sizeof(float));
          dout_ptr0 += pad_0;
          dout_ptr1 += pad_0;
          dout_ptr2 += pad_0;
          dout_ptr3 += pad_0;
        }
        //! mid loop
        for (int i = 0; i < w_out_new; ++i) {
          compute_one_out_without_extract(
              din_ptr0, din_ptr1, din_ptr2, din_ptr3, din_ptr4, din_ptr5,
              din_ptr6, din_ptr7, dout_ptr0, dout_ptr1, dout_ptr2, dout_ptr3,
              w0, w1, w2, w3, w4, w5, w6, vbias);
          din_ptr0++;
          din_ptr1++;
          din_ptr2++;
          din_ptr3++;
          din_ptr4++;
          din_ptr5++;
          din_ptr6++;
          din_ptr7++;

          dout_ptr0++;
          dout_ptr1++;
          dout_ptr2++;
          dout_ptr3++;
        }
        if (flag_bias) {
          //! deal with w_out pad_0 column post with bias
          memcpy(dout_ptr0, dout0, pad_0 * sizeof(float));
          memcpy(dout_ptr1, dout1, pad_0 * sizeof(float));
          memcpy(dout_ptr2, dout2, pad_0 * sizeof(float));
          memcpy(dout_ptr3, dout3, pad_0 * sizeof(float));
        } else {
          //! deal with w_out pad_0 column post without bias
          memset(dout_ptr0, 0x00, pad_0 * sizeof(float));
          memset(dout_ptr1, 0x00, pad_0 * sizeof(float));
          memset(dout_ptr2, 0x00, pad_0 * sizeof(float));
          memset(dout_ptr3, 0x00, pad_0 * sizeof(float));
        }

        din0 = din4;
        din1 = din5;
        din2 = din6;
        din3 = din7;
        din4 = din3 + w_in_new;
        din5 = din4 + w_in_new;
        din6 = din5 + w_in_new;
        din7 = din6 + w_in_new;

        dout0 = dout3 + w_out;
        dout1 = dout0 + w_out;
        dout2 = dout1 + w_out;
        dout3 = dout2 + w_out;
      }
      float* dout_pad_end = dout_ch + h_out_new * w_out;
      if (flag_bias) {
        //! deal with h_out pad_0 line with bias
        memcpy(reinterpret_cast<void*>(dout_pad_end), dout_ch - pad_0 * w_out,
               pad_0 * w_out * sizeof(float));
      } else {
        //! deal with h_out pad_0 line without bias
        memset(reinterpret_cast<void*>(dout_pad_end), 0x00,
               pad_0 * w_out * sizeof(float));
      }
    }
  }
  free(din_new);
}

void conv_depthwise_5x5s1_small_relu_impl(
    const float* din, float* dout, int num, int ch_out, int h_out, int w_out,
    int ch_in, int h_in, int w_in, const float* weights, const float* bias,
    int pad, bool flag_bias, bool flag_relu, ARMContext* ctx) {
  int pad_new = pad > 4 ? 4 : pad;
  int pad_0 = pad - pad_new;
  int h_in_new = h_in + 2 * pad_new;
  int w_in_new = w_in + 2 * pad_new;
  float zero_ptr[w_in_new + w_out];
  memset(zero_ptr, 0, w_in_new * sizeof(float));
  float* write_ptr = zero_ptr + w_in_new;
  int h_out_new = h_out - 2 * pad_0;
  int w_out_new = w_out - 2 * pad_0;
  int pad_cnt = pad_0 >> 2;
  int pad_remain = pad_0 - (pad_cnt << 2);
  int bias_cnt = (w_out * pad_0) >> 2;
  int bias_remain = (w_out * pad_0) - (bias_cnt << 2);
  int in_spatial_size = w_in_new * h_in_new;
  int out_spatial_size = w_out * h_out;
  int weights_saptial_size = 25;

  float* din_new = prepad_input(din, num, ch_in, h_in, w_in, pad_new);
  for (int n = 0; n < num; ++n) {
    const float* din_batch = din_new + n * in_spatial_size * ch_in;
    float* dout_batch = dout + n * out_spatial_size * ch_out;
#pragma omp parallel for
    for (int c = 0; c < ch_in; ++c) {
      const float* din_ch = din_batch + c * in_spatial_size;
      float* dout_ch = dout_batch + c * out_spatial_size;
      float bias_c = flag_bias ? bias[c] : 0.f;
      float bias_relu = bias_c > 0.f ? bias_c : 0.f;
      float vbias[4] = {bias_c, bias_c, bias_c, bias_c};
      float32x4_t vbias_c = vdupq_n_f32(bias_relu);
      if (flag_bias) {
        //! deal with h_out pad_0 line with bias
        for (int i = 0; i < bias_cnt; ++i) {
          vst1q_f32(dout_ch, vbias_c);
          dout_ch += 4;
        }
        for (int i = 0; i < bias_remain; ++i) {
          *dout_ch++ = bias_relu;
        }
      } else {
        //! deal with h_out pad_0 line without bias
        for (int i = 0; i < pad_0; ++i) {
          memset(dout_ch, 0x00, w_out * sizeof(float));
          dout_ch += w_out;
        }
      }

      //! every h loop, deal with 8 line input
      const float* din0 = din_ch;
      const float* din1 = din0 + w_in_new;
      const float* din2 = din1 + w_in_new;
      const float* din3 = din2 + w_in_new;
      const float* din4 = din3 + w_in_new;
      const float* din5 = din4 + w_in_new;
      const float* din6 = din5 + w_in_new;
      const float* din7 = din6 + w_in_new;
      //! every h loop, deal with 4 line output
      float* dout0 = dout_ch;
      float* dout1 = dout0 + w_out;
      float* dout2 = dout1 + w_out;
      float* dout3 = dout2 + w_out;

      //! load weights to neon register
      const float* weights_c = weights + c * weights_saptial_size;

      float32x4_t w5;
      float32x4_t w6;
      float32x4_t w0 = vld1q_f32(weights_c);
      float32x4_t w1 = vld1q_f32(weights_c + 5);
      float32x4_t w2 = vld1q_f32(weights_c + 10);
      float32x4_t w3 = vld1q_f32(weights_c + 15);
      float32x4_t w4 = vld1q_f32(weights_c + 20);
      w5 = vsetq_lane_f32(weights_c[4], w5, 0);
      w5 = vsetq_lane_f32(weights_c[9], w5, 1);
      w5 = vsetq_lane_f32(weights_c[14], w5, 2);
      w5 = vsetq_lane_f32(weights_c[19], w5, 3);
      w6 = vsetq_lane_f32(weights_c[24], w6, 0);

      //! h loop
      for (int h = 0; h < h_out_new; h += 4) {
        //! (h - pad_new) + 7 > h_in - 1
        if (h + 8 > h_in_new) {
          switch (h + 8 - h_in_new) {
            case 7:
              din1 = zero_ptr;
            case 6:
              din2 = zero_ptr;
            case 5:
              din3 = zero_ptr;
            case 4:
              din4 = zero_ptr;
            case 3:
              din5 = zero_ptr;
            case 2:
              din6 = zero_ptr;
            case 1:
              din7 = zero_ptr;
            default:
              break;
          }
        }
        if (h + 4 > h_out_new) {
          switch (h + 4 - h_out_new) {
            case 3:
              dout1 = write_ptr;
            case 2:
              dout2 = write_ptr;
            case 1:
              dout3 = write_ptr;
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
        const float* din_ptr7 = din7;

        float* dout_ptr0 = dout0;
        float* dout_ptr1 = dout1;
        float* dout_ptr2 = dout2;
        float* dout_ptr3 = dout3;

        if (flag_bias) {
          //! deal with w_out pad_0 column pre with bias
          for (int i = 0; i < pad_cnt; i++) {
            vst1q_f32(dout_ptr0, vbias_c);
            vst1q_f32(dout_ptr1, vbias_c);
            vst1q_f32(dout_ptr2, vbias_c);
            vst1q_f32(dout_ptr3, vbias_c);
            dout_ptr0 += 4;
            dout_ptr1 += 4;
            dout_ptr2 += 4;
            dout_ptr3 += 4;
          }
          for (int i = 0; i < pad_remain; ++i) {
            *dout_ptr0++ = bias_relu;
            *dout_ptr1++ = bias_relu;
            *dout_ptr2++ = bias_relu;
            *dout_ptr3++ = bias_relu;
          }
        } else {
          //! deal with w_out pad_0 column pre without bias
          memset(dout_ptr0, 0x00, pad_0 * sizeof(float));
          memset(dout_ptr1, 0x00, pad_0 * sizeof(float));
          memset(dout_ptr2, 0x00, pad_0 * sizeof(float));
          memset(dout_ptr3, 0x00, pad_0 * sizeof(float));
          dout_ptr0 += pad_0;
          dout_ptr1 += pad_0;
          dout_ptr2 += pad_0;
          dout_ptr3 += pad_0;
        }

        //! mid loop
        for (int i = 0; i < w_out_new; ++i) {
          compute_one_out_without_extract_relu(
              din_ptr0, din_ptr1, din_ptr2, din_ptr3, din_ptr4, din_ptr5,
              din_ptr6, din_ptr7, dout_ptr0, dout_ptr1, dout_ptr2, dout_ptr3,
              w0, w1, w2, w3, w4, w5, w6, vbias);
          din_ptr0++;
          din_ptr1++;
          din_ptr2++;
          din_ptr3++;
          din_ptr4++;
          din_ptr5++;
          din_ptr6++;
          din_ptr7++;

          dout_ptr0++;
          dout_ptr1++;
          dout_ptr2++;
          dout_ptr3++;
        }

        if (flag_bias) {
          //! deal with w_out pad_0 column post with bias
          memcpy(dout_ptr0, dout0, pad_0 * sizeof(float));
          memcpy(dout_ptr1, dout1, pad_0 * sizeof(float));
          memcpy(dout_ptr2, dout2, pad_0 * sizeof(float));
          memcpy(dout_ptr3, dout3, pad_0 * sizeof(float));
        } else {
          //! deal with w_out pad_0 column post without bias
          memset(dout_ptr0, 0x00, pad_0 * sizeof(float));
          memset(dout_ptr1, 0x00, pad_0 * sizeof(float));
          memset(dout_ptr2, 0x00, pad_0 * sizeof(float));
          memset(dout_ptr3, 0x00, pad_0 * sizeof(float));
        }

        din0 = din4;
        din1 = din5;
        din2 = din6;
        din3 = din7;
        din4 = din3 + w_in_new;
        din5 = din4 + w_in_new;
        din6 = din5 + w_in_new;
        din7 = din6 + w_in_new;

        dout0 = dout3 + w_out;
        dout1 = dout0 + w_out;
        dout2 = dout1 + w_out;
        dout3 = dout2 + w_out;
      }
      float* dout_pad_end = dout_ch + h_out_new * w_out;
      if (flag_bias) {
        //! deal with h_out pad_0 line with bias
        memcpy(reinterpret_cast<void*>(dout_pad_end), dout_ch - pad_0 * w_out,
               pad_0 * w_out * sizeof(float));
      } else {
        //! deal with h_out pad_0 line without bias
        memset(reinterpret_cast<void*>(dout_pad_end), 0x00,
               pad_0 * w_out * sizeof(float));
      }
    }
  }
  free(din_new);
}

#else

//! kernel for one out without extracting data mid
//! deal with two lines out
void compute_one_out_without_extract(const float* din0, const float* din1,
                                     const float* din2, const float* din3,
                                     const float* din4, const float* din5,
                                     float* dout0, float* dout1,
                                     const float* weights, const float* bias) {
  asm volatile(
      "mov r0, #20  \n"
      "vld1.32 {d0-d1}, [%[wh]], r0  \n"
      "vld1.32 {d2-d3}, [%[wh]], r0  \n"

      "vld1.32 {d4-d5},   [%[din0]]!  \n"
      "vld1.32 {d6-d7},   [%[din1]]!  \n"
      "vld1.32 {d8-d9},   [%[din2]]!  \n"
      "vld1.32 {d10-d11}, [%[din3]]!  \n"
      "vld1.32 {d12-d13}, [%[din4]]!  \n"
      "vld1.32 {d14-d15}, [%[din5]]!  \n"

      // weights r0
      "vmul.f32 q9,  q0, q2  \n"
      "vmul.f32 q10, q0, q3  \n"

      "vld1.32 {d0-d1}, [%[wh]], r0  \n"

      // weights r1
      "vmla.f32 q9,  q1, q3  \n"
      "vmla.f32 q10, q1, q4  \n"

      "vld1.32 {d2-d3}, [%[wh]], r0  \n"
      "vld1.32 {d6[0]}, [%[din0]]  \n"
      "vld1.32 {d6[1]}, [%[din1]]  \n"
      "vld1.32 {d7[0]}, [%[din2]]  \n"
      "vld1.32 {d7[1]}, [%[din3]]  \n"

      // weights r2
      "vmla.f32 q9,  q0, q4  \n"
      "vmla.f32 q10, q0, q5  \n"

      "vld1.32 {d8[0]}, [%[din4]]  \n"
      "vld1.32 {d8[1]}, [%[din5]]  \n"

      "vld1.32 {d0-d1}, [%[wh]]  \n"

      // weights r3
      "vmla.f32 q9,  q1, q5  \n"
      "vmla.f32 q10, q1, q6  \n"

      // weights col4
      "sub %[wh], #64  \n"
      "vld1.32 {d4[0]}, [%[wh]], r0  \n"
      "vld1.32 {d4[1]}, [%[wh]], r0  \n"
      "vld1.32 {d5[0]}, [%[wh]], r0  \n"
      "vld1.32 {d5[1]}, [%[wh]], r0  \n"

      // weights r4
      "vmla.f32 q9,  q0, q6  \n"
      "vmla.f32 q10, q0, q7  \n"

      "vext.32 q5, q3, q4, #1  \n"

      "vmla.f32 q9,  q2, q3  \n"
      "vmla.f32 q10, q2, q5  \n"

      "vld1.32 {d4[0]}, [%[wh]]  \n"
      "vld1.32 {d6}, [%[bias]]  \n"

      "vpadd.f32 d18, d18, d19  \n"
      "vpadd.f32 d19, d20, d21  \n"
      "vpadd.f32 d18, d18, d19  \n"

      "vmla.f32 d18, d8, d4[0]  \n"

      // add bias
      "vadd.f32 d18, d18, d6  \n"

      "vst1.32 {d18[0]}, [%[dout0]]  \n"
      "vst1.32 {d18[1]}, [%[dout1]]  \n"

      : [din0] "+r"(din0), [din1] "+r"(din1), [din2] "+r"(din2),
        [din3] "+r"(din3), [din4] "+r"(din4), [din5] "+r"(din5),
        [wh] "+r"(weights)
      : [dout0] "r"(dout0), [dout1] "r"(dout1), [bias] "r"(bias)
      : "memory", "r0", "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7", "q8",
        "q9", "q10", "q11");
}

//! kernel for one out without extracting data mid
//! deal with two lines out
void compute_one_out_without_extract_relu(const float* din0, const float* din1,
                                          const float* din2, const float* din3,
                                          const float* din4, const float* din5,
                                          float* dout0, float* dout1,
                                          const float* weights,
                                          const float* bias) {
  asm volatile(
      "mov r0, #20  \n"
      "vmov.i32  q15, #0x0  \n"
      "vld1.32 {d0-d1}, [%[wh]], r0  \n"
      "vld1.32 {d2-d3}, [%[wh]], r0  \n"

      "vld1.32 {d4-d5},   [%[din0]]!  \n"
      "vld1.32 {d6-d7},   [%[din1]]!  \n"
      "vld1.32 {d8-d9},   [%[din2]]!  \n"
      "vld1.32 {d10-d11}, [%[din3]]!  \n"
      "vld1.32 {d12-d13}, [%[din4]]!  \n"
      "vld1.32 {d14-d15}, [%[din5]]!  \n"

      // weights r0
      "vmul.f32 q9,  q0, q2  \n"
      "vmul.f32 q10, q0, q3  \n"

      "vld1.32 {d0-d1}, [%[wh]], r0 \n"

      // weights r1
      "vmla.f32 q9,  q1, q3  \n"
      "vmla.f32 q10, q1, q4  \n"

      "vld1.32 {d2-d3}, [%[wh]], r0 \n"
      "vld1.32 {d6[0]}, [%[din0]]   \n"
      "vld1.32 {d6[1]}, [%[din1]]   \n"
      "vld1.32 {d7[0]}, [%[din2]]   \n"
      "vld1.32 {d7[1]}, [%[din3]]   \n"

      // weights r2
      "vmla.f32 q9,  q0, q4  \n"
      "vmla.f32 q10, q0, q5  \n"

      "vld1.32 {d8[0]}, [%[din4]]  \n"
      "vld1.32 {d8[1]}, [%[din5]]  \n"

      "vld1.32 {d0-d1}, [%[wh]]  \n"

      // weights r3
      "vmla.f32 q9,  q1, q5  \n"
      "vmla.f32 q10, q1, q6  \n"

      // weights col4
      "sub %[wh], #64  \n"
      "vld1.32 {d4[0]}, [%[wh]], r0  \n"
      "vld1.32 {d4[1]}, [%[wh]], r0  \n"
      "vld1.32 {d5[0]}, [%[wh]], r0  \n"
      "vld1.32 {d5[1]}, [%[wh]], r0  \n"

      // weights r4
      "vmla.f32 q9,  q0, q6  \n"
      "vmla.f32 q10, q0, q7  \n"

      "vext.32 q5, q3, q4, #1  \n"

      "vmla.f32 q9,  q2, q3  \n"
      "vmla.f32 q10, q2, q5  \n"

      "vld1.32 {d4[0]}, [%[wh]] \n"
      "vld1.32 {d6}, [%[bias]]  \n"

      "vpadd.f32 d18, d18, d19  \n"
      "vpadd.f32 d19, d20, d21  \n"
      "vpadd.f32 d18, d18, d19  \n"

      "vmla.f32 d18, d8, d4[0] \n"

      // add bias
      "vadd.f32 d18, d18, d6   \n"

      // relu
      "vmax.f32 d18, d18, d30  \n"

      "vst1.32 {d18[0]}, [%[dout0]]  \n"
      "vst1.32 {d18[1]}, [%[dout1]]  \n"

      : [din0] "+r"(din0), [din1] "+r"(din1), [din2] "+r"(din2),
        [din3] "+r"(din3), [din4] "+r"(din4), [din5] "+r"(din5),
        [wh] "+r"(weights)
      : [dout0] "r"(dout0), [dout1] "r"(dout1), [bias] "r"(bias)
      : "memory", "r0", "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7", "q8",
        "q9", "q10", "q11", "q15");
}

//! kernel for one out without extracting data pre
//! deal with two lines out
void compute_one_out_extract_pre(const float* din0, const float* din1,
                                 const float* din2, const float* din3,
                                 const float* din4, const float* din5,
                                 float* dout0, float* dout1,
                                 const float* weights, const float* bias) {
  asm volatile(
      "mov r0, #20  \n"
      "add %[wh], #4  \n"
      "vld1.32 {d0-d1}, [%[wh]], r0  \n"
      "vld1.32 {d2-d3}, [%[wh]], r0  \n"

      "vld1.32 {d4-d5},   [%[din0]]!  \n"
      "vld1.32 {d6-d7},   [%[din1]]!  \n"
      "vld1.32 {d8-d9},   [%[din2]]!  \n"
      "vld1.32 {d10-d11}, [%[din3]]!  \n"
      "vld1.32 {d12-d13}, [%[din4]]!  \n"
      "vld1.32 {d14-d15}, [%[din5]]!  \n"

      // weights r0
      "vmul.f32 q9,  q0, q2  \n"
      "vmul.f32 q10, q0, q3  \n"

      "vld1.32 {d0-d1}, [%[wh]], r0  \n"

      // weights r1
      "vmla.f32 q9,  q1, q3  \n"
      "vmla.f32 q10, q1, q4  \n"

      "vld1.32 {d2-d3}, [%[wh]], r0  \n"

      // weights r2
      "vmla.f32 q9,  q0, q4  \n"
      "vmla.f32 q10, q0, q5  \n"

      "vld1.32 {d0-d1}, [%[wh]]  \n"

      // weights r3
      "vmla.f32 q9,  q1, q5  \n"
      "vmla.f32 q10, q1, q6  \n"

      // weights r4
      "vmla.f32 q9,  q0, q6  \n"
      "vmla.f32 q10, q0, q7  \n"

      // load bias
      "vld1.32 {d0}, [%[bias]]  \n"

      "vpadd.f32 d18, d18, d19  \n"
      "vpadd.f32 d19, d20, d21  \n"
      "vpadd.f32 d18, d18, d19  \n"

      // add bias
      "vadd.f32 d18, d18, d0  \n"

      "vst1.32 {d18[0]}, [%[dout0]]  \n"
      "vst1.32 {d18[1]}, [%[dout1]]  \n"

      : [din0] "+r"(din0), [din1] "+r"(din1), [din2] "+r"(din2),
        [din3] "+r"(din3), [din4] "+r"(din4), [din5] "+r"(din5),
        [wh] "+r"(weights)
      : [dout0] "r"(dout0), [dout1] "r"(dout1), [bias] "r"(bias)
      : "memory", "r0", "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7", "q8",
        "q9", "q10", "q11");
}

//! kernel for one out without extracting data pre
//! deal with two lines out
void compute_one_out_extract_pre_relu(const float* din0, const float* din1,
                                      const float* din2, const float* din3,
                                      const float* din4, const float* din5,
                                      float* dout0, float* dout1,
                                      const float* weights, const float* bias) {
  asm volatile(
      "mov r0, #20  \n"
      "add %[wh], #4  \n"
      "vmov.i32  q15, #0x0  \n"
      "vld1.32 {d0-d1}, [%[wh]], r0  \n"
      "vld1.32 {d2-d3}, [%[wh]], r0  \n"

      "vld1.32 {d4-d5},   [%[din0]]!  \n"
      "vld1.32 {d6-d7},   [%[din1]]!  \n"
      "vld1.32 {d8-d9},   [%[din2]]!  \n"
      "vld1.32 {d10-d11}, [%[din3]]!  \n"
      "vld1.32 {d12-d13}, [%[din4]]!  \n"
      "vld1.32 {d14-d15}, [%[din5]]!  \n"

      // weights r0
      "vmul.f32 q9,  q0, q2  \n"
      "vmul.f32 q10, q0, q3  \n"

      "vld1.32 {d0-d1}, [%[wh]], r0  \n"

      // weights r1
      "vmla.f32 q9,  q1, q3  \n"
      "vmla.f32 q10, q1, q4  \n"

      "vld1.32 {d2-d3}, [%[wh]], r0  \n"

      // weights r2
      "vmla.f32 q9,  q0, q4  \n"
      "vmla.f32 q10, q0, q5  \n"

      "vld1.32 {d0-d1}, [%[wh]]  \n"

      // weights r3
      "vmla.f32 q9,  q1, q5  \n"
      "vmla.f32 q10, q1, q6  \n"

      // weights r4
      "vmla.f32 q9,  q0, q6  \n"
      "vmla.f32 q10, q0, q7  \n"

      // load bias
      "vld1.32 {d0}, [%[bias]]  \n"

      "vpadd.f32 d18, d18, d19  \n"
      "vpadd.f32 d19, d20, d21  \n"
      "vpadd.f32 d18, d18, d19  \n"

      // add bias
      "vadd.f32 d18, d18, d0  \n"

      // relu
      "vmax.f32 d18, d18, d30  \n"
      "vst1.32 {d18[0]}, [%[dout0]]  \n"
      "vst1.32 {d18[1]}, [%[dout1]]  \n"

      : [din0] "+r"(din0), [din1] "+r"(din1), [din2] "+r"(din2),
        [din3] "+r"(din3), [din4] "+r"(din4), [din5] "+r"(din5),
        [wh] "+r"(weights)
      : [dout0] "r"(dout0), [dout1] "r"(dout1), [bias] "r"(bias)
      : "memory", "r0", "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7", "q8",
        "q9", "q10", "q11", "q15");
}

//! kernel for one out with extracting data post
//! deal with two lines out
void compute_one_out_extract_post(const float* din0, const float* din1,
                                  const float* din2, const float* din3,
                                  const float* din4, const float* din5,
                                  float* dout0, float* dout1,
                                  const float* weights, const float* bias) {
  asm volatile(
      "mov r0, #20  \n"
      "vld1.32 {d0-d1}, [%[wh]], r0  \n"
      "vld1.32 {d2-d3}, [%[wh]], r0  \n"

      "vld1.32 {d4-d5},   [%[din0]]!  \n"
      "vld1.32 {d6-d7},   [%[din1]]!  \n"
      "vld1.32 {d8-d9},   [%[din2]]!  \n"
      "vld1.32 {d10-d11}, [%[din3]]!  \n"
      "vld1.32 {d12-d13}, [%[din4]]!  \n"
      "vld1.32 {d14-d15}, [%[din5]]!  \n"

      // weights r0
      "vmul.f32 q9,  q0, q2  \n"
      "vmul.f32 q10, q0, q3  \n"

      "vld1.32 {d0-d1}, [%[wh]], r0  \n"

      // weights r1
      "vmla.f32 q9,  q1, q3  \n"
      "vmla.f32 q10, q1, q4  \n"

      "vld1.32 {d2-d3}, [%[wh]], r0  \n"

      // weights r2
      "vmla.f32 q9,  q0, q4  \n"
      "vmla.f32 q10, q0, q5  \n"

      "vld1.32 {d0-d1}, [%[wh]]  \n"

      // weights r3
      "vmla.f32 q9,  q1, q5  \n"
      "vmla.f32 q10, q1, q6  \n"

      // weights r4
      "vmla.f32 q9,  q0, q6  \n"
      "vmla.f32 q10, q0, q7  \n"

      "vld1.32 {d0}, [%[bias]]  \n"

      "vpadd.f32 d18, d18, d19  \n"
      "vpadd.f32 d19, d20, d21  \n"
      "vpadd.f32 d18, d18, d19  \n"

      // add bias
      "vadd.f32 d18, d18, d0  \n"

      "vst1.32 {d18[0]}, [%[dout0]]  \n"
      "vst1.32 {d18[1]}, [%[dout1]]  \n"

      : [din0] "+r"(din0), [din1] "+r"(din1), [din2] "+r"(din2),
        [din3] "+r"(din3), [din4] "+r"(din4), [din5] "+r"(din5),
        [wh] "+r"(weights)
      : [dout0] "r"(dout0), [dout1] "r"(dout1), [bias] "r"(bias)
      : "memory", "r0", "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7", "q8",
        "q9", "q10", "q11");
}

//! kernel for one out with extracting data post
//! deal with two lines out
void compute_one_out_extract_post_relu(const float* din0, const float* din1,
                                       const float* din2, const float* din3,
                                       const float* din4, const float* din5,
                                       float* dout0, float* dout1,
                                       const float* weights,
                                       const float* bias) {
  asm volatile(
      "mov r0, #20  \n"
      "vmov.i32  q15, #0x0  \n"
      "vld1.32 {d0-d1}, [%[wh]], r0  \n"
      "vld1.32 {d2-d3}, [%[wh]], r0  \n"

      "vld1.32 {d4-d5},   [%[din0]]!  \n"
      "vld1.32 {d6-d7},   [%[din1]]!  \n"
      "vld1.32 {d8-d9},   [%[din2]]!  \n"
      "vld1.32 {d10-d11}, [%[din3]]!  \n"
      "vld1.32 {d12-d13}, [%[din4]]!  \n"
      "vld1.32 {d14-d15}, [%[din5]]!  \n"

      // weights r0
      "vmul.f32 q9,  q0, q2  \n"
      "vmul.f32 q10, q0, q3  \n"

      "vld1.32 {d0-d1}, [%[wh]], r0  \n"

      // weights r1
      "vmla.f32 q9,  q1, q3  \n"
      "vmla.f32 q10, q1, q4  \n"

      "vld1.32 {d2-d3}, [%[wh]], r0  \n"

      // weights r2
      "vmla.f32 q9,  q0, q4  \n"
      "vmla.f32 q10, q0, q5  \n"

      "vld1.32 {d0-d1}, [%[wh]]  \n"

      // weights r3
      "vmla.f32 q9,  q1, q5  \n"
      "vmla.f32 q10, q1, q6  \n"

      // weights r4
      "vmla.f32 q9,  q0, q6  \n"
      "vmla.f32 q10, q0, q7  \n"

      "vld1.32 {d0}, [%[bias]]  \n"

      "vpadd.f32 d18, d18, d19  \n"
      "vpadd.f32 d19, d20, d21  \n"
      "vpadd.f32 d18, d18, d19  \n"

      // add bias
      "vadd.f32 d18, d18, d0  \n"

      // relu
      "vmax.f32 d18, d18, d30  \n"

      "vst1.32 {d18[0]}, [%[dout0]]  \n"
      "vst1.32 {d18[1]}, [%[dout1]]  \n"

      : [din0] "+r"(din0), [din1] "+r"(din1), [din2] "+r"(din2),
        [din3] "+r"(din3), [din4] "+r"(din4), [din5] "+r"(din5),
        [wh] "+r"(weights)
      : [dout0] "r"(dout0), [dout1] "r"(dout1), [bias] "r"(bias)
      : "memory", "r0", "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7", "q8",
        "q9", "q10", "q11", "q15");
}

//! kernel for two out with extracting data pre
//! deal with two lines out
void compute_two_out_extract_pre(const float* din0, const float* din1,
                                 const float* din2, const float* din3,
                                 const float* din4, const float* din5,
                                 float* dout0, float* dout1,
                                 const float* weights, const float* bias) {
  asm volatile(
      "mov r0, #20  \n"
      "mov r1, #0  \n"
      "add %[wh], #8  \n"
      "vld1.32 {d0-d1}, [%[wh]], r0  \n"
      "vld1.32 {d2-d3}, [%[wh]], r0  \n"

      "vmov.32 d1[1], r1  \n"
      "vmov.32 d3[1], r1  \n"

      "vld1.32 {d4-d5},   [%[din0]]!  \n"
      "vld1.32 {d6-d7},   [%[din1]]!  \n"
      "vld1.32 {d8-d9},   [%[din2]]!  \n"
      "vld1.32 {d10-d11}, [%[din3]]!  \n"
      "vld1.32 {d12-d13}, [%[din4]]!  \n"
      "vld1.32 {d14-d15}, [%[din5]]!  \n"

      // weights r0
      "vmul.f32 q9,  q0, q2  \n"
      "vmul.f32 q10, q0, q3  \n"

      "vld1.32 {d24-d25}, [%[wh]], r0  \n"
      "vmov.32 d25[1], r1  \n"

      // weights r1
      "vmla.f32 q9,  q1, q3  \n"
      "vmla.f32 q10, q1, q4  \n"

      "vld1.32 {d26-d27}, [%[wh]], r0  \n"
      "vmov.32 d27[1], r1  \n"

      // weights r2
      "vmla.f32 q9,  q12, q4  \n"
      "vmla.f32 q10, q12, q5  \n"

      "vld1.32 {d28-d29}, [%[wh]]\n"
      "vmov.32 d29[1], r1  \n"

      // weights r3
      "vmla.f32 q9,  q13, q5  \n"
      "vmla.f32 q10, q13, q6  \n"

      // weights r4
      "vmla.f32 q9,  q14, q6  \n"
      "vmla.f32 q10, q14, q7  \n"

      "sub %[wh], #84  \n"
      "vld1.32 {d0-d1}, [%[wh]], r0  \n"
      "vld1.32 {d2-d3}, [%[wh]], r0  \n"
      "vld1.32 {d24-d25}, [%[wh]], r0  \n"

      "vpadd.f32 d22, d18, d19  \n"
      "vpadd.f32 d23, d20, d21  \n"

      "vld1.32 {d26-d27}, [%[wh]], r0  \n"
      "vld1.32 {d28-d29}, [%[wh]]\n"

      "vpadd.f32 d22, d22, d23  \n"

      // weights r0
      "vmul.f32 q9,  q0, q2  \n"
      "vmul.f32 q10, q0, q3  \n"

      // weights r1
      "vmla.f32 q9,  q1, q3  \n"
      "vmla.f32 q10, q1, q4  \n"

      // weights r2
      "vmla.f32 q9,  q12, q4  \n"
      "vmla.f32 q10, q12, q5  \n"

      // weights r3
      "vmla.f32 q9,  q13, q5  \n"
      "vmla.f32 q10, q13, q6  \n"

      // weights r4
      "vmla.f32 q9,  q14, q6  \n"
      "vmla.f32 q10, q14, q7  \n"
      "vld1.32 {d30-d31}, [%[bias]]  \n"

      "vpadd.f32 d18, d18, d19  \n"
      "vpadd.f32 d19, d20, d21  \n"
      "vpadd.f32 d23, d18, d19  \n"

      // trn out neon register
      "vtrn.32 d22, d23  \n"

      // add bias
      "vadd.f32 q11, q11, q15  \n"

      // store result
      "vst1.32 {d22}, [%[dout0]] \n"
      "vst1.32 {d23}, [%[dout1]]  \n"

      : [din0] "+r"(din0), [din1] "+r"(din1), [din2] "+r"(din2),
        [din3] "+r"(din3), [din4] "+r"(din4), [din5] "+r"(din5),
        [wh] "+r"(weights)
      : [dout0] "r"(dout0), [dout1] "r"(dout1), [bias] "r"(bias)
      : "memory", "r0", "r1", "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7",
        "q8", "q9", "q10", "q11", "q12", "q13", "q14", "q15");
}

//! kernel for two out with extracting data pre
//! deal with two lines out
void compute_two_out_extract_pre_relu(const float* din0, const float* din1,
                                      const float* din2, const float* din3,
                                      const float* din4, const float* din5,
                                      float* dout0, float* dout1,
                                      const float* weights, const float* bias) {
  asm volatile(
      "mov r0, #20  \n"
      "mov r1, #0  \n"
      "add %[wh], #8  \n"
      "vld1.32 {d0-d1}, [%[wh]], r0  \n"
      "vld1.32 {d2-d3}, [%[wh]], r0  \n"

      "vmov.32 d1[1], r1  \n"
      "vmov.32 d3[1], r1  \n"

      "vld1.32 {d4-d5},   [%[din0]]!  \n"
      "vld1.32 {d6-d7},   [%[din1]]!  \n"
      "vld1.32 {d8-d9},   [%[din2]]!  \n"
      "vld1.32 {d10-d11}, [%[din3]]!  \n"
      "vld1.32 {d12-d13}, [%[din4]]!  \n"
      "vld1.32 {d14-d15}, [%[din5]]!  \n"

      // weights r0
      "vmul.f32 q9,  q0, q2  \n"
      "vmul.f32 q10, q0, q3  \n"

      "vld1.32 {d24-d25}, [%[wh]], r0  \n"
      "vmov.32 d25[1], r1  \n"

      // weights r1
      "vmla.f32 q9,  q1, q3  \n"
      "vmla.f32 q10, q1, q4  \n"

      "vld1.32 {d26-d27}, [%[wh]], r0  \n"
      "vmov.32 d27[1], r1  \n"

      // weights r2
      "vmla.f32 q9,  q12, q4  \n"
      "vmla.f32 q10, q12, q5  \n"

      "vld1.32 {d28-d29}, [%[wh]]\n"
      "vmov.32 d29[1], r1  \n"

      // weights r3
      "vmla.f32 q9,  q13, q5  \n"
      "vmla.f32 q10, q13, q6  \n"

      // weights r4
      "vmla.f32 q9,  q14, q6  \n"
      "vmla.f32 q10, q14, q7  \n"

      "sub %[wh], #84  \n"
      "vld1.32 {d0-d1}, [%[wh]], r0  \n"
      "vld1.32 {d2-d3}, [%[wh]], r0  \n"
      "vld1.32 {d24-d25}, [%[wh]], r0  \n"

      "vpadd.f32 d22, d18, d19  \n"
      "vpadd.f32 d23, d20, d21  \n"

      "vld1.32 {d26-d27}, [%[wh]], r0  \n"
      "vld1.32 {d28-d29}, [%[wh]]\n"

      "vpadd.f32 d22, d22, d23  \n"

      // weights r0
      "vmul.f32 q9,  q0, q2  \n"
      "vmul.f32 q10, q0, q3  \n"

      // weights r1
      "vmla.f32 q9,  q1, q3  \n"
      "vmla.f32 q10, q1, q4  \n"

      // weights r2
      "vmla.f32 q9,  q12, q4  \n"
      "vmla.f32 q10, q12, q5  \n"

      // weights r3
      "vmla.f32 q9,  q13, q5  \n"
      "vmla.f32 q10, q13, q6  \n"

      // weights r4
      "vmla.f32 q9,  q14, q6  \n"
      "vmla.f32 q10, q14, q7  \n"
      "vld1.32 {d30-d31}, [%[bias]]  \n"

      "vpadd.f32 d18, d18, d19  \n"
      "vpadd.f32 d19, d20, d21  \n"
      "vpadd.f32 d23, d18, d19  \n"
      "vmov.i32  q9, #0x0  \n"

      // trn out neon register
      "vtrn.32 d22, d23  \n"

      // add bias
      "vadd.f32 q11, q11, q15  \n"

      // relu
      "vmax.f32 q11, q11, q9  \n"
      // store result
      "vst1.32 {d22}, [%[dout0]] \n"
      "vst1.32 {d23}, [%[dout1]]  \n"

      : [din0] "+r"(din0), [din1] "+r"(din1), [din2] "+r"(din2),
        [din3] "+r"(din3), [din4] "+r"(din4), [din5] "+r"(din5),
        [wh] "+r"(weights)
      : [dout0] "r"(dout0), [dout1] "r"(dout1), [bias] "r"(bias)
      : "memory", "r0", "r1", "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7",
        "q8", "q9", "q10", "q11", "q12", "q13", "q14", "q15");
}

//! kernel for two out with extracting data post
//! deal with two lines out
void compute_two_out_extract_post(const float* din0, const float* din1,
                                  const float* din2, const float* din3,
                                  const float* din4, const float* din5,
                                  float* dout0, float* dout1,
                                  const float* weights, const float* bias) {
  asm volatile(
      "mov r0, #20  \n"
      "vld1.32 {d0-d1}, [%[wh]], r0  \n"
      "vld1.32 {d2-d3}, [%[wh]], r0  \n"

      "vld1.32 {d4-d5},   [%[din0]]!  \n"
      "vld1.32 {d6-d7},   [%[din1]]!  \n"
      "vld1.32 {d8-d9},   [%[din2]]!  \n"
      "vld1.32 {d10-d11}, [%[din3]]!  \n"
      "vld1.32 {d12-d13}, [%[din4]]!  \n"
      "vld1.32 {d14-d15}, [%[din5]]!  \n"

      //! out zero
      // weights r0
      "vmul.f32 q9,  q0, q2  \n"
      "vmul.f32 q10, q0, q3  \n"

      "vld1.32 {d24-d25}, [%[wh]], r0  \n"

      // weights r1
      "vmla.f32 q9,  q1, q3  \n"
      "vmla.f32 q10, q1, q4  \n"

      "vld1.32 {d26-d27}, [%[wh]], r0  \n"

      // weights r2
      "vmla.f32 q9,  q12, q4  \n"
      "vmla.f32 q10, q12, q5  \n"

      "vld1.32 {d28-d29}, [%[wh]]\n"

      // weights r3
      "vmla.f32 q9,  q13, q5  \n"
      "vmla.f32 q10, q13, q6  \n"

      // weights r4
      "vmla.f32 q9,  q14, q6  \n"
      "vmla.f32 q10, q14, q7  \n"

      "vpadd.f32 d22, d18, d19  \n"
      "vpadd.f32 d23, d20, d21  \n"
      "vpadd.f32 d22, d22, d23  \n"

      "vmov.f32 q15, #0.0  \n"
      "vext.32 q2, q2, q15, #1 \n"
      "vext.32 q3, q3, q15, #1 \n"
      "vext.32 q4, q4, q15, #1 \n"
      "vext.32 q5, q5, q15, #1 \n"
      "vext.32 q6, q6, q15, #1 \n"
      "vext.32 q7, q7, q15, #1 \n"
      "vext.32 q8, q8, q15, #1 \n"

      //! out one
      // weights r0
      "vmul.f32 q9,  q0, q2  \n"
      "vmul.f32 q10, q0, q3  \n"

      // weights r1
      "vmla.f32 q9,  q1, q3  \n"
      "vmla.f32 q10, q1, q4  \n"

      // weights r2
      "vmla.f32 q9,  q12, q4  \n"
      "vmla.f32 q10, q12, q5  \n"

      // weights r3
      "vmla.f32 q9,  q13, q5  \n"
      "vmla.f32 q10, q13, q6  \n"

      // weights r4
      "vmla.f32 q9,  q14, q6  \n"
      "vmla.f32 q10, q14, q7  \n"
      "vld1.32 {d30-d31}, [%[bias]] \n"

      "vpadd.f32 d18, d18, d19  \n"
      "vpadd.f32 d19, d20, d21  \n"
      "vpadd.f32 d23, d18, d19  \n"

      // trn out neon register
      "vtrn.32 d22, d23  \n"

      // add bias
      "vadd.f32 q11, q11, q15  \n"

      // store result
      "vst1.32 {d22}, [%[dout0]] \n"
      "vst1.32 {d23}, [%[dout1]] \n"

      : [din0] "+r"(din0), [din1] "+r"(din1), [din2] "+r"(din2),
        [din3] "+r"(din3), [din4] "+r"(din4), [din5] "+r"(din5),
        [wh] "+r"(weights)
      : [dout0] "r"(dout0), [dout1] "r"(dout1), [bias] "r"(bias)
      : "memory", "r0", "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7", "q8",
        "q9", "q10", "q11", "q12", "q13", "q14", "q15");
}

//! kernel for two out with extracting data post
//! deal with two lines out
void compute_two_out_extract_post_relu(const float* din0, const float* din1,
                                       const float* din2, const float* din3,
                                       const float* din4, const float* din5,
                                       float* dout0, float* dout1,
                                       const float* weights,
                                       const float* bias) {
  asm volatile(
      "mov r0, #20  \n"
      "vld1.32 {d0-d1}, [%[wh]], r0  \n"
      "vld1.32 {d2-d3}, [%[wh]], r0  \n"

      "vld1.32 {d4-d5},   [%[din0]]!  \n"
      "vld1.32 {d6-d7},   [%[din1]]!  \n"
      "vld1.32 {d8-d9},   [%[din2]]!  \n"
      "vld1.32 {d10-d11}, [%[din3]]!  \n"
      "vld1.32 {d12-d13}, [%[din4]]!  \n"
      "vld1.32 {d14-d15}, [%[din5]]!  \n"

      //! out zero
      // weights r0
      "vmul.f32 q9,  q0, q2  \n"
      "vmul.f32 q10, q0, q3  \n"

      "vld1.32 {d24-d25}, [%[wh]], r0  \n"

      // weights r1
      "vmla.f32 q9,  q1, q3  \n"
      "vmla.f32 q10, q1, q4  \n"

      "vld1.32 {d26-d27}, [%[wh]], r0  \n"

      // weights r2
      "vmla.f32 q9,  q12, q4  \n"
      "vmla.f32 q10, q12, q5  \n"

      "vld1.32 {d28-d29}, [%[wh]]\n"

      // weights r3
      "vmla.f32 q9,  q13, q5  \n"
      "vmla.f32 q10, q13, q6  \n"

      // weights r4
      "vmla.f32 q9,  q14, q6  \n"
      "vmla.f32 q10, q14, q7  \n"

      "vpadd.f32 d22, d18, d19  \n"
      "vpadd.f32 d23, d20, d21  \n"
      "vpadd.f32 d22, d22, d23  \n"

      "vmov.f32 q15, #0.0  \n"
      "vext.32 q2, q2, q15, #1 \n"
      "vext.32 q3, q3, q15, #1 \n"
      "vext.32 q4, q4, q15, #1 \n"
      "vext.32 q5, q5, q15, #1 \n"
      "vext.32 q6, q6, q15, #1 \n"
      "vext.32 q7, q7, q15, #1 \n"
      "vext.32 q8, q8, q15, #1 \n"

      //! out one
      // weights r0
      "vmul.f32 q9,  q0, q2  \n"
      "vmul.f32 q10, q0, q3  \n"

      // weights r1
      "vmla.f32 q9,  q1, q3  \n"
      "vmla.f32 q10, q1, q4  \n"

      // weights r2
      "vmla.f32 q9,  q12, q4  \n"
      "vmla.f32 q10, q12, q5  \n"

      // weights r3
      "vmla.f32 q9,  q13, q5  \n"
      "vmla.f32 q10, q13, q6  \n"

      // weights r4
      "vmla.f32 q9,  q14, q6  \n"
      "vmla.f32 q10, q14, q7  \n"
      "vld1.32 {d30-d31}, [%[bias]] \n"

      "vpadd.f32 d18, d18, d19  \n"
      "vpadd.f32 d19, d20, d21  \n"
      "vpadd.f32 d23, d18, d19  \n"
      "vmov.i32  q9, #0x0  \n"

      // trn out neon register
      "vtrn.32 d22, d23  \n"

      // add bias
      "vadd.f32 q11, q11, q15  \n"

      // relu
      "vmax.f32 q11, q11, q9  \n"

      // store result
      "vst1.32 {d22}, [%[dout0]] \n"
      "vst1.32 {d23}, [%[dout1]] \n"

      : [din0] "+r"(din0), [din1] "+r"(din1), [din2] "+r"(din2),
        [din3] "+r"(din3), [din4] "+r"(din4), [din5] "+r"(din5),
        [wh] "+r"(weights)
      : [dout0] "r"(dout0), [dout1] "r"(dout1), [bias] "r"(bias)
      : "memory", "r0", "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7", "q8",
        "q9", "q10", "q11", "q12", "q13", "q14", "q15");
}

//! kernel for three out with extracting data pre
//! deal with two lines out
void compute_three_out_extract_pre(const float* din0, const float* din1,
                                   const float* din2, const float* din3,
                                   const float* din4, const float* din5,
                                   float* dout0, float* dout1,
                                   const float* weights, const float* bias) {
  asm volatile(
      "mov r0, #20  \n"
      "add %[wh], #12  \n"
      "vld1.32 {d0}, [%[wh]], r0  \n"
      "vld1.32 {d2}, [%[wh]], r0  \n"

      "vld1.32 {d4-d5},   [%[din0]]  \n"
      "vld1.32 {d6-d7},   [%[din1]]  \n"
      "vld1.32 {d8-d9},   [%[din2]]  \n"
      "vld1.32 {d10-d11}, [%[din3]]  \n"
      "vld1.32 {d12-d13}, [%[din4]]  \n"
      "vld1.32 {d14-d15}, [%[din5]]  \n"

      //! out zero
      // weights r0
      "vmul.f32 d18, d0, d4  \n"
      "vmul.f32 d20, d0, d6  \n"

      "vld1.32 {d24}, [%[wh]], r0  \n"

      // weights r1
      "vmla.f32 d18, d2, d6  \n"
      "vmla.f32 d20, d2, d8  \n"

      "vld1.32 {d26}, [%[wh]], r0  \n"

      // weights r2
      "vmla.f32 d18, d24, d8  \n"
      "vmla.f32 d20, d24, d10  \n"

      "vld1.32 {d28}, [%[wh]]  \n"

      // weights r3
      "vmla.f32 d18, d26, d10  \n"
      "vmla.f32 d20, d26, d12  \n"

      // load bias
      "vld1.32 {d30-d31}, [%[bias]]  \n"

      // weights r4
      "vmla.f32 d18, d28, d12  \n"
      "vmla.f32 d20, d28, d14  \n"
      "vpadd.f32 d22, d18, d20 \n"

      //! out one
      "mov r1, #0  \n"
      "sub %[wh], #84  \n"
      "vld1.32 {d0-d1}, [%[wh]], r0  \n"
      "vld1.32 {d2-d3}, [%[wh]], r0  \n"

      "vmov.32 d1[1], r1  \n"
      "vmov.32 d3[1], r1  \n"

      // weights r0
      "vmul.f32 q9,  q0, q2  \n"
      "vmul.f32 q10, q0, q3  \n"

      "vld1.32 {d24-d25}, [%[wh]], r0  \n"
      "vmov.32 d25[1], r1  \n"

      // weights r1
      "vmla.f32 q9,  q1, q3  \n"
      "vmla.f32 q10, q1, q4  \n"

      "vld1.32 {d26-d27}, [%[wh]], r0  \n"
      "vmov.32 d27[1], r1  \n"

      // weights r2
      "vmla.f32 q9,  q12, q4  \n"
      "vmla.f32 q10, q12, q5  \n"

      "vld1.32 {d28-d29}, [%[wh]]\n"
      "vmov.32 d29[1], r1  \n"

      // weights r3
      "vmla.f32 q9,  q13, q5  \n"
      "vmla.f32 q10, q13, q6  \n"

      // weights r4
      "vmla.f32 q9,  q14, q6  \n"
      "vmla.f32 q10, q14, q7  \n"

      "sub %[wh], #84  \n"
      "vld1.32 {d0-d1}, [%[wh]], r0  \n"
      "vld1.32 {d2-d3}, [%[wh]], r0  \n"
      "vld1.32 {d24-d25}, [%[wh]], r0  \n"

      "vpadd.f32 d18, d18, d19  \n"
      "vpadd.f32 d19, d20, d21  \n"

      "vld1.32 {d26-d27}, [%[wh]], r0  \n"
      "vld1.32 {d28-d29}, [%[wh]]\n"

      "vpadd.f32 d23, d18, d19  \n"

      // trn out neon register
      "vtrn.32 d22, d23  \n"

      // add bias
      "vadd.f32 q11, q11, q15  \n"

      // store result
      "vst1.32 {d22}, [%[dout0]]! \n"
      "vst1.32 {d23}, [%[dout1]]!  \n"

      //! out two
      // weights r0
      "vmul.f32 q9,  q0, q2  \n"
      "vmul.f32 q10, q0, q3  \n"

      // weights r1
      "vmla.f32 q9,  q1, q3  \n"
      "vmla.f32 q10, q1, q4  \n"

      // weights r2
      "vmla.f32 q9,  q12, q4  \n"
      "vmla.f32 q10, q12, q5  \n"

      // weights r3
      "vmla.f32 q9,  q13, q5  \n"
      "vmla.f32 q10, q13, q6  \n"

      // weights r4
      "vmla.f32 q9,  q14, q6  \n"
      "vmla.f32 q10, q14, q7  \n"

      "vpadd.f32 d18, d18, d19  \n"
      "vpadd.f32 d19, d20, d21  \n"
      "vpadd.f32 d18, d18, d19  \n"

      // add bias
      "vadd.f32 d18, d18, d30  \n"

      // store result
      "vst1.32 {d18[0]}, [%[dout0]] \n"
      "vst1.32 {d18[1]}, [%[dout1]] \n"

      : [dout0] "+r"(dout0), [dout1] "+r"(dout1), [wh] "+r"(weights)
      : [din0] "r"(din0), [din1] "r"(din1), [din2] "r"(din2), [din3] "r"(din3),
        [din4] "r"(din4), [din5] "r"(din5), [bias] "r"(bias)
      : "memory", "r0", "r1", "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7",
        "q8", "q9", "q10", "q11", "q12", "q13", "q14", "q15");
}

//! kernel for three out with extracting data pre
//! deal with two lines out
void compute_three_out_extract_pre_relu(const float* din0, const float* din1,
                                        const float* din2, const float* din3,
                                        const float* din4, const float* din5,
                                        float* dout0, float* dout1,
                                        const float* weights,
                                        const float* bias) {
  asm volatile(
      "mov r0, #20  \n"
      "add %[wh], #12  \n"
      "vld1.32 {d0}, [%[wh]], r0  \n"
      "vld1.32 {d2}, [%[wh]], r0  \n"

      "vld1.32 {d4-d5},   [%[din0]]  \n"
      "vld1.32 {d6-d7},   [%[din1]]  \n"
      "vld1.32 {d8-d9},   [%[din2]]  \n"
      "vld1.32 {d10-d11}, [%[din3]]  \n"
      "vld1.32 {d12-d13}, [%[din4]]  \n"
      "vld1.32 {d14-d15}, [%[din5]]  \n"

      //! out zero
      // weights r0
      "vmul.f32 d18, d0, d4  \n"
      "vmul.f32 d20, d0, d6  \n"

      "vld1.32 {d24}, [%[wh]], r0  \n"

      // weights r1
      "vmla.f32 d18, d2, d6  \n"
      "vmla.f32 d20, d2, d8  \n"

      "vld1.32 {d26}, [%[wh]], r0  \n"

      // weights r2
      "vmla.f32 d18, d24, d8  \n"
      "vmla.f32 d20, d24, d10  \n"

      "vld1.32 {d28}, [%[wh]]  \n"

      // weights r3
      "vmla.f32 d18, d26, d10  \n"
      "vmla.f32 d20, d26, d12  \n"

      // load bias
      "vld1.32 {d30-d31}, [%[bias]]  \n"

      // weights r4
      "vmla.f32 d18, d28, d12  \n"
      "vmla.f32 d20, d28, d14  \n"
      "vpadd.f32 d22, d18, d20 \n"

      //! out one
      "mov r1, #0  \n"
      "sub %[wh], #84  \n"
      "vld1.32 {d0-d1}, [%[wh]], r0  \n"
      "vld1.32 {d2-d3}, [%[wh]], r0  \n"

      "vmov.32 d1[1], r1  \n"
      "vmov.32 d3[1], r1  \n"

      // weights r0
      "vmul.f32 q9,  q0, q2  \n"
      "vmul.f32 q10, q0, q3  \n"

      "vld1.32 {d24-d25}, [%[wh]], r0  \n"
      "vmov.32 d25[1], r1  \n"

      // weights r1
      "vmla.f32 q9,  q1, q3  \n"
      "vmla.f32 q10, q1, q4  \n"

      "vld1.32 {d26-d27}, [%[wh]], r0  \n"
      "vmov.32 d27[1], r1  \n"

      // weights r2
      "vmla.f32 q9,  q12, q4  \n"
      "vmla.f32 q10, q12, q5  \n"

      "vld1.32 {d28-d29}, [%[wh]]\n"
      "vmov.32 d29[1], r1  \n"

      // weights r3
      "vmla.f32 q9,  q13, q5  \n"
      "vmla.f32 q10, q13, q6  \n"

      // weights r4
      "vmla.f32 q9,  q14, q6  \n"
      "vmla.f32 q10, q14, q7  \n"

      "sub %[wh], #84  \n"
      "vld1.32 {d0-d1}, [%[wh]], r0  \n"
      "vld1.32 {d2-d3}, [%[wh]], r0  \n"
      "vld1.32 {d24-d25}, [%[wh]], r0  \n"

      "vpadd.f32 d18, d18, d19  \n"
      "vpadd.f32 d19, d20, d21  \n"

      "vld1.32 {d26-d27}, [%[wh]], r0  \n"
      "vld1.32 {d28-d29}, [%[wh]]\n"

      "vpadd.f32 d23, d18, d19  \n"
      "vmov.i32  q8, #0x0  \n"

      // trn out neon register
      "vtrn.32 d22, d23  \n"

      // add bias
      "vadd.f32 q11, q11, q15  \n"

      // relu
      "vmax.f32 q11, q11, q8  \n"

      // store result
      "vst1.32 {d22}, [%[dout0]]! \n"
      "vst1.32 {d23}, [%[dout1]]!  \n"

      //! out two
      // weights r0
      "vmul.f32 q9,  q0, q2  \n"
      "vmul.f32 q10, q0, q3  \n"

      // weights r1
      "vmla.f32 q9,  q1, q3  \n"
      "vmla.f32 q10, q1, q4  \n"

      // weights r2
      "vmla.f32 q9,  q12, q4  \n"
      "vmla.f32 q10, q12, q5  \n"

      // weights r3
      "vmla.f32 q9,  q13, q5  \n"
      "vmla.f32 q10, q13, q6  \n"

      // weights r4
      "vmla.f32 q9,  q14, q6  \n"
      "vmla.f32 q10, q14, q7  \n"

      "vpadd.f32 d18, d18, d19  \n"
      "vpadd.f32 d19, d20, d21  \n"
      "vpadd.f32 d18, d18, d19  \n"

      // add bias
      "vadd.f32 d18, d18, d30  \n"

      // relu
      "vmax.f32 d18, d18, d16  \n"

      // store result
      "vst1.32 {d18[0]}, [%[dout0]] \n"
      "vst1.32 {d18[1]}, [%[dout1]] \n"

      : [dout0] "+r"(dout0), [dout1] "+r"(dout1), [wh] "+r"(weights)
      : [din0] "r"(din0), [din1] "r"(din1), [din2] "r"(din2), [din3] "r"(din3),
        [din4] "r"(din4), [din5] "r"(din5), [bias] "r"(bias)
      : "memory", "r0", "r1", "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7",
        "q8", "q9", "q10", "q11", "q12", "q13", "q14", "q15");
}

//! kernel for three out with extracting data post
//! deal with two lines out
void compute_three_out_extract_post(const float* din0, const float* din1,
                                    const float* din2, const float* din3,
                                    const float* din4, const float* din5,
                                    float* dout0, float* dout1,
                                    const float* weights, const float* bias) {
  asm volatile(
      "mov r0, #20  \n"
      "vld1.32 {d0-d1}, [%[wh]], r0  \n"
      "vld1.32 {d2-d3}, [%[wh]], r0  \n"

      "vld1.32 {d4-d5},   [%[din0]]  \n"
      "vld1.32 {d6-d7},   [%[din1]]  \n"
      "vld1.32 {d8-d9},   [%[din2]]  \n"
      "vld1.32 {d10-d11}, [%[din3]]  \n"
      "vld1.32 {d12-d13}, [%[din4]]  \n"
      "vld1.32 {d14-d15}, [%[din5]]  \n"

      //! out zero && two

      // weights r0
      "vmul.f32 q9,  q0, q2  \n"
      "vmul.f32 q10, q0, q3  \n"
      "vmul.f32 d16, d0, d5  \n"
      "vmul.f32 d17, d0, d7  \n"

      "vld1.32 {d24-d25}, [%[wh]], r0  \n"

      // weights r1
      "vmla.f32 q9,  q1, q3  \n"
      "vmla.f32 q10, q1, q4  \n"
      "vmla.f32 d16, d2, d7  \n"
      "vmla.f32 d17, d2, d9  \n"

      "vld1.32 {d26-d27}, [%[wh]], r0  \n"

      // weights r2
      "vmla.f32 q9,  q12, q4  \n"
      "vmla.f32 q10, q12, q5  \n"
      "vmla.f32 d16, d24, d9  \n"
      "vmla.f32 d17, d24, d11 \n"

      "vld1.32 {d28-d29}, [%[wh]]\n"

      // weights r3
      "vmla.f32 q9,  q13, q5  \n"
      "vmla.f32 q10, q13, q6  \n"
      "vmla.f32 d16, d26, d11 \n"
      "vmla.f32 d17, d26, d13 \n"

      // weights r4
      "vmla.f32 q9,  q14, q6  \n"
      "vmla.f32 q10, q14, q7  \n"
      "vmla.f32 d16, d28, d13 \n"
      "vmla.f32 d17, d28, d15 \n"

      "vpadd.f32 d18, d18, d19  \n"
      "vpadd.f32 d19, d20, d21  \n"
      "vpadd.f32 d16, d16, d17  \n"
      "vpadd.f32 d22, d18, d19  \n"

      "vmov.f32 q15, #0.0  \n"
      "vext.32 q2, q2, q15, #1 \n"
      "vext.32 q3, q3, q15, #1 \n"
      "vext.32 q4, q4, q15, #1 \n"
      "vext.32 q5, q5, q15, #1 \n"
      "vext.32 q6, q6, q15, #1 \n"
      "vext.32 q7, q7, q15, #1 \n"

      //! out one
      // weights r0
      "vmul.f32 q9,  q0, q2  \n"
      "vmul.f32 q10, q0, q3  \n"

      // weights r1
      "vmla.f32 q9,  q1, q3  \n"
      "vmla.f32 q10, q1, q4  \n"

      // weights r2
      "vmla.f32 q9,  q12, q4  \n"
      "vmla.f32 q10, q12, q5  \n"

      // weights r3
      "vmla.f32 q9,  q13, q5  \n"
      "vmla.f32 q10, q13, q6  \n"

      // weights r4
      "vmla.f32 q9,  q14, q6  \n"
      "vmla.f32 q10, q14, q7  \n"

      // load bias
      "vld1.32 {d30-d31}, [%[bias]] \n"

      "vpadd.f32 d18, d18, d19  \n"
      "vpadd.f32 d19, d20, d21  \n"
      "vpadd.f32 d23, d18, d19  \n"
      "vmov.i32  q9, #0x0  \n"

      // trn out neon register
      "vtrn.32 d22, d23  \n"

      // add bias
      "vadd.f32 q11, q11, q15  \n"
      "vadd.f32 d16, d16, d30  \n"

      "vst1.32 {d22},    [%[dout0]]!  \n"
      "vst1.32 {d23},    [%[dout1]]!  \n"
      "vst1.32 {d16[0]}, [%[dout0]]!  \n"
      "vst1.32 {d16[1]}, [%[dout1]]!  \n"

      : [dout0] "+r"(dout0), [dout1] "+r"(dout1), [wh] "+r"(weights)
      : [din0] "r"(din0), [din1] "r"(din1), [din2] "r"(din2), [din3] "r"(din3),
        [din4] "r"(din4), [din5] "r"(din5), [bias] "r"(bias)
      : "memory", "r0", "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7", "q8",
        "q9", "q10", "q11", "q12", "q13", "q14", "q15");
}

//! kernel for three out with extracting data post
//! deal with two lines out
void compute_three_out_extract_post_relu(const float* din0, const float* din1,
                                         const float* din2, const float* din3,
                                         const float* din4, const float* din5,
                                         float* dout0, float* dout1,
                                         const float* weights,
                                         const float* bias) {
  asm volatile(
      "mov r0, #20  \n"
      "vld1.32 {d0-d1}, [%[wh]], r0  \n"
      "vld1.32 {d2-d3}, [%[wh]], r0  \n"

      "vld1.32 {d4-d5},   [%[din0]]  \n"
      "vld1.32 {d6-d7},   [%[din1]]  \n"
      "vld1.32 {d8-d9},   [%[din2]]  \n"
      "vld1.32 {d10-d11}, [%[din3]]  \n"
      "vld1.32 {d12-d13}, [%[din4]]  \n"
      "vld1.32 {d14-d15}, [%[din5]]  \n"

      //! out zero && two

      // weights r0
      "vmul.f32 q9,  q0, q2  \n"
      "vmul.f32 q10, q0, q3  \n"
      "vmul.f32 d16, d0, d5  \n"
      "vmul.f32 d17, d0, d7  \n"

      "vld1.32 {d24-d25}, [%[wh]], r0  \n"

      // weights r1
      "vmla.f32 q9,  q1, q3  \n"
      "vmla.f32 q10, q1, q4  \n"
      "vmla.f32 d16, d2, d7  \n"
      "vmla.f32 d17, d2, d9  \n"

      "vld1.32 {d26-d27}, [%[wh]], r0  \n"

      // weights r2
      "vmla.f32 q9,  q12, q4  \n"
      "vmla.f32 q10, q12, q5  \n"
      "vmla.f32 d16, d24, d9  \n"
      "vmla.f32 d17, d24, d11 \n"

      "vld1.32 {d28-d29}, [%[wh]]\n"

      // weights r3
      "vmla.f32 q9,  q13, q5  \n"
      "vmla.f32 q10, q13, q6  \n"
      "vmla.f32 d16, d26, d11 \n"
      "vmla.f32 d17, d26, d13 \n"

      // weights r4
      "vmla.f32 q9,  q14, q6  \n"
      "vmla.f32 q10, q14, q7  \n"
      "vmla.f32 d16, d28, d13 \n"
      "vmla.f32 d17, d28, d15 \n"

      "vpadd.f32 d18, d18, d19  \n"
      "vpadd.f32 d19, d20, d21  \n"
      "vpadd.f32 d16, d16, d17  \n"
      "vpadd.f32 d22, d18, d19  \n"

      "vmov.f32 q15, #0.0  \n"
      "vext.32 q2, q2, q15, #1 \n"
      "vext.32 q3, q3, q15, #1 \n"
      "vext.32 q4, q4, q15, #1 \n"
      "vext.32 q5, q5, q15, #1 \n"
      "vext.32 q6, q6, q15, #1 \n"
      "vext.32 q7, q7, q15, #1 \n"

      //! out one
      // weights r0
      "vmul.f32 q9,  q0, q2  \n"
      "vmul.f32 q10, q0, q3  \n"

      // weights r1
      "vmla.f32 q9,  q1, q3  \n"
      "vmla.f32 q10, q1, q4  \n"

      // weights r2
      "vmla.f32 q9,  q12, q4  \n"
      "vmla.f32 q10, q12, q5  \n"

      // weights r3
      "vmla.f32 q9,  q13, q5  \n"
      "vmla.f32 q10, q13, q6  \n"

      // weights r4
      "vmla.f32 q9,  q14, q6  \n"
      "vmla.f32 q10, q14, q7  \n"

      // load bias
      "vld1.32 {d30-d31}, [%[bias]] \n"

      "vpadd.f32 d18, d18, d19  \n"
      "vpadd.f32 d19, d20, d21  \n"
      "vpadd.f32 d23, d18, d19  \n"
      "vmov.i32 q9, #0x0  \n"

      // trn out neon register
      "vtrn.32 d22, d23  \n"

      // add bias
      "vadd.f32 q11, q11, q15  \n"
      "vadd.f32 d16, d16, d30  \n"

      // relu
      "vmax.f32 q11, q11, q9  \n"
      "vmax.f32 d16, d16, d18 \n"

      "vst1.32 {d22},    [%[dout0]]!  \n"
      "vst1.32 {d23},    [%[dout1]]!  \n"
      "vst1.32 {d16[0]}, [%[dout0]]!  \n"
      "vst1.32 {d16[1]}, [%[dout1]]!  \n"

      : [dout0] "+r"(dout0), [dout1] "+r"(dout1), [wh] "+r"(weights)
      : [din0] "r"(din0), [din1] "r"(din1), [din2] "r"(din2), [din3] "r"(din3),
        [din4] "r"(din4), [din5] "r"(din5), [bias] "r"(bias)
      : "memory", "r0", "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7", "q8",
        "q9", "q10", "q11", "q12", "q13", "q14", "q15");
}

//! kernel for four out with extracting data pre
//! deal with two lines out
void compute_four_out_extract_pre(const float* din0, const float* din1,
                                  const float* din2, const float* din3,
                                  const float* din4, const float* din5,
                                  float* dout0, float* dout1,
                                  const float* weights, const float* bias) {
  asm volatile(
      "mov r0, #20  \n"
      "add %[wh], #16  \n"

      //! out zero
      // load input
      "vld1.32 {d4[0]}, [%[din0]]  \n"
      "vld1.32 {d4[1]}, [%[din1]]  \n"
      "vld1.32 {d5[0]}, [%[din2]]  \n"
      "vld1.32 {d5[1]}, [%[din3]]  \n"
      "vld1.32 {d6[0]}, [%[din4]]  \n"
      "vld1.32 {d6[1]}, [%[din5]]  \n"

      "vext.32 q4, q2, q3, #1  \n"

      // load weights
      "vld1.32 d0[0], [%[wh]], r0  \n"
      "vld1.32 d0[1], [%[wh]], r0 \n"
      "vld1.32 d1[0], [%[wh]], r0  \n"
      "vld1.32 d1[1], [%[wh]], r0  \n"
      "vld1.32 d2[0], [%[wh]]\n"

      "vmul.f32 q9, q0, q2  \n"
      "vmul.f32 q10, q0, q4 \n"

      "vld1.32 {d30-d31}, [%[bias]]  \n"

      "vpadd.f32 d18, d18, d19  \n"
      "vpadd.f32 d19, d20, d21  \n"
      "vpadd.f32 d22, d18, d19  \n"

      "vmla.f32 d22, d6, d2[0]  \n"

      "sub %[wh], #84  \n"
      "vld1.32 {d0}, [%[wh]], r0  \n"
      "vld1.32 {d2}, [%[wh]], r0  \n"

      "vld1.32 {d4-d5},   [%[din0]]  \n"
      "vld1.32 {d6-d7},   [%[din1]]  \n"
      "vld1.32 {d8-d9},   [%[din2]]  \n"
      "vld1.32 {d10-d11}, [%[din3]]  \n"
      "vld1.32 {d12-d13}, [%[din4]]  \n"
      "vld1.32 {d14-d15}, [%[din5]]  \n"

      //! out one
      // weights r0
      "vmul.f32 d18, d0, d4  \n"
      "vmul.f32 d20, d0, d6  \n"

      "vld1.32 {d24}, [%[wh]], r0  \n"

      // weights r1
      "vmla.f32 d18, d2, d6  \n"
      "vmla.f32 d20, d2, d8  \n"

      "vld1.32 {d26}, [%[wh]], r0  \n"

      // weights r2
      "vmla.f32 d18, d24, d8  \n"
      "vmla.f32 d20, d24, d10  \n"

      "vld1.32 {d28}, [%[wh]]  \n"

      // weights r3
      "vmla.f32 d18, d26, d10  \n"
      "vmla.f32 d20, d26, d12  \n"

      // weights r4
      "vmla.f32 d18, d28, d12  \n"
      "vmla.f32 d20, d28, d14  \n"

      "vpadd.f32 d23, d18, d20 \n"

      // trn out neon register
      "vtrn.32 d22, d23  \n"

      // add bias
      "vadd.f32 q11, q11, q15  \n"

      // store result
      "vst1.32 {d22}, [%[dout0]]!  \n"
      "vst1.32 {d23}, [%[dout1]]!  \n"

      //! out two
      "mov r1, #0  \n"
      "sub %[wh], #84  \n"
      "vld1.32 {d0-d1}, [%[wh]], r0  \n"
      "vld1.32 {d2-d3}, [%[wh]], r0  \n"

      "vmov.32 d1[1], r1  \n"
      "vmov.32 d3[1], r1  \n"

      // weights r0
      "vmul.f32 q9,  q0, q2  \n"
      "vmul.f32 q10, q0, q3  \n"

      "vld1.32 {d24-d25}, [%[wh]], r0  \n"
      "vmov.32 d25[1], r1  \n"

      // weights r1
      "vmla.f32 q9,  q1, q3  \n"
      "vmla.f32 q10, q1, q4  \n"

      "vld1.32 {d26-d27}, [%[wh]], r0  \n"
      "vmov.32 d27[1], r1  \n"

      // weights r2
      "vmla.f32 q9,  q12, q4  \n"
      "vmla.f32 q10, q12, q5  \n"

      "vld1.32 {d28-d29}, [%[wh]]\n"
      "vmov.32 d29[1], r1  \n"

      // weights r3
      "vmla.f32 q9,  q13, q5  \n"
      "vmla.f32 q10, q13, q6  \n"

      // weights r4
      "vmla.f32 q9,  q14, q6  \n"
      "vmla.f32 q10, q14, q7  \n"

      "sub %[wh], #84  \n"
      "vld1.32 {d0-d1}, [%[wh]], r0  \n"
      "vld1.32 {d2-d3}, [%[wh]], r0  \n"
      "vld1.32 {d24-d25}, [%[wh]], r0  \n"

      "vpadd.f32 d18, d18, d19  \n"
      "vpadd.f32 d19, d20, d21  \n"

      "vld1.32 {d26-d27}, [%[wh]], r0  \n"
      "vld1.32 {d28-d29}, [%[wh]]\n"

      "vpadd.f32 d22, d18, d19  \n"

      //! out three
      // weights r0
      "vmul.f32 q9,  q0, q2  \n"
      "vmul.f32 q10, q0, q3  \n"

      // weights r1
      "vmla.f32 q9,  q1, q3  \n"
      "vmla.f32 q10, q1, q4  \n"

      // weights r2
      "vmla.f32 q9,  q12, q4  \n"
      "vmla.f32 q10, q12, q5  \n"

      // weights r3
      "vmla.f32 q9,  q13, q5  \n"
      "vmla.f32 q10, q13, q6  \n"

      // weights r4
      "vmla.f32 q9,  q14, q6  \n"
      "vmla.f32 q10, q14, q7  \n"

      "vpadd.f32 d18, d18, d19  \n"
      "vpadd.f32 d19, d20, d21  \n"
      "vpadd.f32 d23, d18, d19  \n"

      // trn out neon register
      "vtrn.32 d22, d23  \n"

      // add bias
      "vadd.f32 q11, q11, q15  \n"

      // store result
      "vst1.32 {d22}, [%[dout0]]  \n"
      "vst1.32 {d23}, [%[dout1]]  \n"

      : [dout0] "+r"(dout0), [dout1] "+r"(dout1), [wh] "+r"(weights)
      : [din0] "r"(din0), [din1] "r"(din1), [din2] "r"(din2), [din3] "r"(din3),
        [din4] "r"(din4), [din5] "r"(din5), [bias] "r"(bias)
      : "memory", "r0", "r1", "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7",
        "q8", "q9", "q10", "q11", "q12", "q13", "q14", "q15");
}

//! kernel for four out with extracting data pre
//! deal with two lines out
void compute_four_out_extract_pre_relu(const float* din0, const float* din1,
                                       const float* din2, const float* din3,
                                       const float* din4, const float* din5,
                                       float* dout0, float* dout1,
                                       const float* weights,
                                       const float* bias) {
  asm volatile(
      "mov r0, #20  \n"
      "add %[wh], #16  \n"

      //! out zero
      // load input
      "vld1.32 {d4[0]}, [%[din0]]  \n"
      "vld1.32 {d4[1]}, [%[din1]]  \n"
      "vld1.32 {d5[0]}, [%[din2]]  \n"
      "vld1.32 {d5[1]}, [%[din3]]  \n"
      "vld1.32 {d6[0]}, [%[din4]]  \n"
      "vld1.32 {d6[1]}, [%[din5]]  \n"

      "vext.32 q4, q2, q3, #1  \n"

      // load weights
      "vld1.32 d0[0], [%[wh]], r0  \n"
      "vld1.32 d0[1], [%[wh]], r0 \n"
      "vld1.32 d1[0], [%[wh]], r0  \n"
      "vld1.32 d1[1], [%[wh]], r0  \n"
      "vld1.32 d2[0], [%[wh]]\n"

      "vmul.f32 q9, q0, q2  \n"
      "vmul.f32 q10, q0, q4 \n"

      "vld1.32 {d30-d31}, [%[bias]]  \n"

      "vpadd.f32 d18, d18, d19  \n"
      "vpadd.f32 d19, d20, d21  \n"
      "vpadd.f32 d22, d18, d19  \n"

      "vmla.f32 d22, d6, d2[0]  \n"

      "sub %[wh], #84  \n"
      "vld1.32 {d0}, [%[wh]], r0  \n"
      "vld1.32 {d2}, [%[wh]], r0  \n"

      "vld1.32 {d4-d5},   [%[din0]]  \n"
      "vld1.32 {d6-d7},   [%[din1]]  \n"
      "vld1.32 {d8-d9},   [%[din2]]  \n"
      "vld1.32 {d10-d11}, [%[din3]]  \n"
      "vld1.32 {d12-d13}, [%[din4]]  \n"
      "vld1.32 {d14-d15}, [%[din5]]  \n"

      //! out one
      // weights r0
      "vmul.f32 d18, d0, d4  \n"
      "vmul.f32 d20, d0, d6  \n"

      "vld1.32 {d24}, [%[wh]], r0  \n"

      // weights r1
      "vmla.f32 d18, d2, d6  \n"
      "vmla.f32 d20, d2, d8  \n"

      "vld1.32 {d26}, [%[wh]], r0  \n"

      // weights r2
      "vmla.f32 d18, d24, d8  \n"
      "vmla.f32 d20, d24, d10  \n"

      "vld1.32 {d28}, [%[wh]]  \n"

      // weights r3
      "vmla.f32 d18, d26, d10  \n"
      "vmla.f32 d20, d26, d12  \n"

      // weights r4
      "vmla.f32 d18, d28, d12  \n"
      "vmla.f32 d20, d28, d14  \n"

      "vpadd.f32 d23, d18, d20 \n"
      "vmov.i32 q8, #0x0  \n"

      // trn out neon register
      "vtrn.32 d22, d23  \n"

      // add bias
      "vadd.f32 q11, q11, q15  \n"

      // relu
      "vmax.f32 q11, q11, q8  \n"

      // store result
      "vst1.32 {d22}, [%[dout0]]!  \n"
      "vst1.32 {d23}, [%[dout1]]!  \n"

      //! out two
      "mov r1, #0  \n"
      "sub %[wh], #84  \n"
      "vld1.32 {d0-d1}, [%[wh]], r0  \n"
      "vld1.32 {d2-d3}, [%[wh]], r0  \n"

      "vmov.32 d1[1], r1  \n"
      "vmov.32 d3[1], r1  \n"

      // weights r0
      "vmul.f32 q9,  q0, q2  \n"
      "vmul.f32 q10, q0, q3  \n"

      "vld1.32 {d24-d25}, [%[wh]], r0  \n"
      "vmov.32 d25[1], r1  \n"

      // weights r1
      "vmla.f32 q9,  q1, q3  \n"
      "vmla.f32 q10, q1, q4  \n"

      "vld1.32 {d26-d27}, [%[wh]], r0  \n"
      "vmov.32 d27[1], r1  \n"

      // weights r2
      "vmla.f32 q9,  q12, q4  \n"
      "vmla.f32 q10, q12, q5  \n"

      "vld1.32 {d28-d29}, [%[wh]]\n"
      "vmov.32 d29[1], r1  \n"

      // weights r3
      "vmla.f32 q9,  q13, q5  \n"
      "vmla.f32 q10, q13, q6  \n"

      // weights r4
      "vmla.f32 q9,  q14, q6  \n"
      "vmla.f32 q10, q14, q7  \n"

      "sub %[wh], #84  \n"
      "vld1.32 {d0-d1}, [%[wh]], r0  \n"
      "vld1.32 {d2-d3}, [%[wh]], r0  \n"
      "vld1.32 {d24-d25}, [%[wh]], r0  \n"

      "vpadd.f32 d18, d18, d19  \n"
      "vpadd.f32 d19, d20, d21  \n"

      "vld1.32 {d26-d27}, [%[wh]], r0  \n"
      "vld1.32 {d28-d29}, [%[wh]]\n"

      "vpadd.f32 d22, d18, d19  \n"

      //! out three
      // weights r0
      "vmul.f32 q9,  q0, q2  \n"
      "vmul.f32 q10, q0, q3  \n"

      // weights r1
      "vmla.f32 q9,  q1, q3  \n"
      "vmla.f32 q10, q1, q4  \n"

      // weights r2
      "vmla.f32 q9,  q12, q4  \n"
      "vmla.f32 q10, q12, q5  \n"

      // weights r3
      "vmla.f32 q9,  q13, q5  \n"
      "vmla.f32 q10, q13, q6  \n"

      // weights r4
      "vmla.f32 q9,  q14, q6  \n"
      "vmla.f32 q10, q14, q7  \n"

      "vpadd.f32 d18, d18, d19  \n"
      "vpadd.f32 d19, d20, d21  \n"
      "vpadd.f32 d23, d18, d19  \n"

      // trn out neon register
      "vtrn.32 d22, d23  \n"

      // add bias
      "vadd.f32 q11, q11, q15  \n"

      // relu
      "vmax.f32 q11, q11, q8  \n"

      // store result
      "vst1.32 {d22}, [%[dout0]]  \n"
      "vst1.32 {d23}, [%[dout1]]  \n"

      : [dout0] "+r"(dout0), [dout1] "+r"(dout1), [wh] "+r"(weights)
      : [din0] "r"(din0), [din1] "r"(din1), [din2] "r"(din2), [din3] "r"(din3),
        [din4] "r"(din4), [din5] "r"(din5), [bias] "r"(bias)
      : "memory", "r0", "r1", "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7",
        "q8", "q9", "q10", "q11", "q12", "q13", "q14", "q15");
}

//! kernel for three out with extracting data post
//! deal with two lines out
void compute_four_out_extract_post(const float* din0, const float* din1,
                                   const float* din2, const float* din3,
                                   const float* din4, const float* din5,
                                   float* dout0, float* dout1,
                                   const float* weights, const float* bias) {
  asm volatile(
      "mov r0, #20  \n"
      "mov r1, #12  \n"
      "vld1.32 {d0-d1}, [%[wh]], r0  \n"
      "vld1.32 {d2-d3}, [%[wh]], r0  \n"

      "vld1.32 {d4-d5},   [%[din0]], r1  \n"
      "vld1.32 {d6-d7},   [%[din1]], r1  \n"
      "vld1.32 {d8-d9},   [%[din2]], r1  \n"
      "vld1.32 {d10-d11}, [%[din3]], r1  \n"
      "vld1.32 {d12-d13}, [%[din4]], r1  \n"
      "vld1.32 {d14-d15}, [%[din5]], r1  \n"

      //! out zero && two
      // weights r0
      "vmul.f32 q9,  q0, q2  \n"
      "vmul.f32 q10, q0, q3  \n"
      "vmul.f32 d16, d0, d5  \n"
      "vmul.f32 d17, d0, d7  \n"

      "vld1.32 {d24-d25}, [%[wh]], r0  \n"

      // weights r1
      "vmla.f32 q9,  q1, q3  \n"
      "vmla.f32 q10, q1, q4  \n"
      "vmla.f32 d16, d2, d7  \n"
      "vmla.f32 d17, d2, d9  \n"

      "vld1.32 {d26-d27}, [%[wh]], r0  \n"

      // weights r2
      "vmla.f32 q9,  q12, q4  \n"
      "vmla.f32 q10, q12, q5  \n"
      "vmla.f32 d16, d24, d9  \n"
      "vmla.f32 d17, d24, d11 \n"

      "vld1.32 {d28-d29}, [%[wh]]  \n"

      // weights r3
      "vmla.f32 q9,  q13, q5  \n"
      "vmla.f32 q10, q13, q6  \n"
      "vmla.f32 d16, d26, d11 \n"
      "vmla.f32 d17, d26, d13 \n"

      // weights r4
      "vmla.f32 q9,  q14, q6  \n"
      "vmla.f32 q10, q14, q7  \n"
      "vmla.f32 d16, d28, d13 \n"
      "vmla.f32 d17, d28, d15 \n"

      "vpadd.f32 d18, d18, d19  \n"
      "vpadd.f32 d19, d20, d21  \n"
      "vpadd.f32 d16, d16, d17  \n"
      "vpadd.f32 d22, d18, d19  \n"

      //! out one
      "vmov.f32 q15, #0.0  \n"
      "vext.32 q2, q2, q15, #1 \n"
      "vext.32 q3, q3, q15, #1 \n"
      "vext.32 q4, q4, q15, #1 \n"
      "vext.32 q5, q5, q15, #1 \n"
      "vext.32 q6, q6, q15, #1 \n"
      "vext.32 q7, q7, q15, #1 \n"

      // weights r0
      "vmul.f32 q9,  q0, q2  \n"
      "vmul.f32 q10, q0, q3  \n"

      // weights r1
      "vmla.f32 q9,  q1, q3  \n"
      "vmla.f32 q10, q1, q4  \n"

      // weights r2
      "vmla.f32 q9,  q12, q4  \n"
      "vmla.f32 q10, q12, q5  \n"

      // weights r3
      "vmla.f32 q9,  q13, q5  \n"
      "vmla.f32 q10, q13, q6  \n"

      // weights r4
      "vmla.f32 q9,  q14, q6  \n"
      "vmla.f32 q10, q14, q7  \n"

      "vld1.32 {d30-d31}, [%[bias]] \n"

      "vpadd.f32 d18, d18, d19  \n"
      "vpadd.f32 d19, d20, d21  \n"
      "vpadd.f32 d23, d18, d19  \n"

      // trn out neon register
      "vtrn.32 d22, d23  \n"

      // add bias
      "vadd.f32 q11, q11, q15  \n"

      // store result
      "vst1.32 {d22}, [%[dout0]]! \n"
      "vst1.32 {d23}, [%[dout1]]! \n"

      //! out three
      "sub %[wh], #80  \n"
      "vld1.32 {d4[0]}, [%[din0]]  \n"
      "vld1.32 {d4[1]}, [%[din1]]  \n"
      "vld1.32 {d5[0]}, [%[din2]]  \n"
      "vld1.32 {d5[1]}, [%[din3]]  \n"
      "vld1.32 {d6[0]}, [%[din4]]  \n"
      "vld1.32 {d6[1]}, [%[din5]]  \n"

      "vext.32 q4, q2, q3, #1  \n"

      "vld1.32 {d0[0]}, [%[wh]], r0  \n"
      "vld1.32 {d0[1]}, [%[wh]], r0  \n"
      "vld1.32 {d1[0]}, [%[wh]], r0  \n"
      "vld1.32 {d1[1]}, [%[wh]], r0  \n"
      "vld1.32 {d2[0]}, [%[wh]]      \n"

      "vmul.f32 q9, q0, q2   \n"
      "vmul.f32 q10, q0, q4  \n"

      "vpadd.f32 d18, d18, d19  \n"
      "vpadd.f32 d20, d20, d21  \n"
      "vpadd.f32 d17, d18, d20  \n"

      "vmla.f32 d17, d6, d2[0]  \n"

      // trn out neon register
      "vtrn.32 d16, d17  \n"

      // add bias
      "vadd.f32 q8, q8, q15  \n"

      // store result
      "vst1.32 {d16}, [%[dout0]]  \n"
      "vst1.32 {d17}, [%[dout1]]  \n"

      : [dout0] "+r"(dout0), [dout1] "+r"(dout1), [din0] "+r"(din0),
        [din1] "+r"(din1), [din2] "+r"(din2), [din3] "+r"(din3),
        [din4] "+r"(din4), [din5] "+r"(din5), [wh] "+r"(weights)
      : [bias] "r"(bias)
      : "memory", "r0", "r1", "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7",
        "q8", "q9", "q10", "q11", "q12", "q13", "q14", "q15");
}

//! kernel for three out with extracting data post
//! deal with two lines out
void compute_four_out_extract_post_relu(const float* din0, const float* din1,
                                        const float* din2, const float* din3,
                                        const float* din4, const float* din5,
                                        float* dout0, float* dout1,
                                        const float* weights,
                                        const float* bias) {
  asm volatile(
      "mov r0, #20  \n"
      "mov r1, #12  \n"
      "vld1.32 {d0-d1}, [%[wh]], r0  \n"
      "vld1.32 {d2-d3}, [%[wh]], r0  \n"

      "vld1.32 {d4-d5},   [%[din0]], r1  \n"
      "vld1.32 {d6-d7},   [%[din1]], r1  \n"
      "vld1.32 {d8-d9},   [%[din2]], r1  \n"
      "vld1.32 {d10-d11}, [%[din3]], r1  \n"
      "vld1.32 {d12-d13}, [%[din4]], r1  \n"
      "vld1.32 {d14-d15}, [%[din5]], r1  \n"

      //! out zero && two
      // weights r0
      "vmul.f32 q9,  q0, q2  \n"
      "vmul.f32 q10, q0, q3  \n"
      "vmul.f32 d16, d0, d5  \n"
      "vmul.f32 d17, d0, d7  \n"

      "vld1.32 {d24-d25}, [%[wh]], r0  \n"

      // weights r1
      "vmla.f32 q9,  q1, q3  \n"
      "vmla.f32 q10, q1, q4  \n"
      "vmla.f32 d16, d2, d7  \n"
      "vmla.f32 d17, d2, d9  \n"

      "vld1.32 {d26-d27}, [%[wh]], r0  \n"

      // weights r2
      "vmla.f32 q9,  q12, q4  \n"
      "vmla.f32 q10, q12, q5  \n"
      "vmla.f32 d16, d24, d9  \n"
      "vmla.f32 d17, d24, d11 \n"

      "vld1.32 {d28-d29}, [%[wh]]  \n"

      // weights r3
      "vmla.f32 q9,  q13, q5  \n"
      "vmla.f32 q10, q13, q6  \n"
      "vmla.f32 d16, d26, d11 \n"
      "vmla.f32 d17, d26, d13 \n"

      // weights r4
      "vmla.f32 q9,  q14, q6  \n"
      "vmla.f32 q10, q14, q7  \n"
      "vmla.f32 d16, d28, d13 \n"
      "vmla.f32 d17, d28, d15 \n"

      "vpadd.f32 d18, d18, d19  \n"
      "vpadd.f32 d19, d20, d21  \n"
      "vpadd.f32 d16, d16, d17  \n"
      "vpadd.f32 d22, d18, d19  \n"

      //! out one
      "vmov.f32 q15, #0.0  \n"
      "vext.32 q2, q2, q15, #1 \n"
      "vext.32 q3, q3, q15, #1 \n"
      "vext.32 q4, q4, q15, #1 \n"
      "vext.32 q5, q5, q15, #1 \n"
      "vext.32 q6, q6, q15, #1 \n"
      "vext.32 q7, q7, q15, #1 \n"

      // weights r0
      "vmul.f32 q9,  q0, q2  \n"
      "vmul.f32 q10, q0, q3  \n"

      // weights r1
      "vmla.f32 q9,  q1, q3  \n"
      "vmla.f32 q10, q1, q4  \n"

      // weights r2
      "vmla.f32 q9,  q12, q4  \n"
      "vmla.f32 q10, q12, q5  \n"

      // weights r3
      "vmla.f32 q9,  q13, q5  \n"
      "vmla.f32 q10, q13, q6  \n"

      // weights r4
      "vmla.f32 q9,  q14, q6  \n"
      "vmla.f32 q10, q14, q7  \n"

      "vld1.32 {d30-d31}, [%[bias]] \n"

      "vpadd.f32 d18, d18, d19  \n"
      "vpadd.f32 d19, d20, d21  \n"
      "vpadd.f32 d23, d18, d19  \n"
      "vmov.i32 q5, #0x0  \n"

      // trn out neon register
      "vtrn.32 d22, d23  \n"

      // add bias
      "vadd.f32 q11, q11, q15  \n"

      // relu
      "vmax.f32 q11, q11, q5  \n"

      // store result
      "vst1.32 {d22}, [%[dout0]]! \n"
      "vst1.32 {d23}, [%[dout1]]! \n"

      //! out three
      "sub %[wh], #80  \n"
      "vld1.32 {d4[0]}, [%[din0]]  \n"
      "vld1.32 {d4[1]}, [%[din1]]  \n"
      "vld1.32 {d5[0]}, [%[din2]]  \n"
      "vld1.32 {d5[1]}, [%[din3]]  \n"
      "vld1.32 {d6[0]}, [%[din4]]  \n"
      "vld1.32 {d6[1]}, [%[din5]]  \n"

      "vext.32 q4, q2, q3, #1  \n"

      "vld1.32 {d0[0]}, [%[wh]], r0  \n"
      "vld1.32 {d0[1]}, [%[wh]], r0  \n"
      "vld1.32 {d1[0]}, [%[wh]], r0  \n"
      "vld1.32 {d1[1]}, [%[wh]], r0  \n"
      "vld1.32 {d2[0]}, [%[wh]]      \n"

      "vmul.f32 q9, q0, q2   \n"
      "vmul.f32 q10, q0, q4  \n"

      "vpadd.f32 d18, d18, d19  \n"
      "vpadd.f32 d20, d20, d21  \n"
      "vpadd.f32 d17, d18, d20  \n"

      "vmla.f32 d17, d6, d2[0]  \n"

      // trn out neon register
      "vtrn.32 d16, d17  \n"

      // add bias
      "vadd.f32 q8, q8, q15  \n"

      // relu
      "vmax.f32 q8, q8, q5  \n"

      // store result
      "vst1.32 {d16}, [%[dout0]]  \n"
      "vst1.32 {d17}, [%[dout1]]  \n"

      : [dout0] "+r"(dout0), [dout1] "+r"(dout1), [din0] "+r"(din0),
        [din1] "+r"(din1), [din2] "+r"(din2), [din3] "+r"(din3),
        [din4] "+r"(din4), [din5] "+r"(din5), [wh] "+r"(weights)
      : [bias] "r"(bias)
      : "memory", "r0", "r1", "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7",
        "q8", "q9", "q10", "q11", "q12", "q13", "q14", "q15");
}

void conv_depthwise_5x5s1_impl(const float* din, float* dout, int num,
                               int ch_out, int h_out, int w_out, int ch_in,
                               int h_in, int w_in, const float* weights,
                               const float* bias, int pad, bool flag_bias,
                               bool flag_relu, ARMContext* ctx) {
  float* zero_ptr = ctx->workspace_data<float>();
  memset(zero_ptr, 0, w_in * sizeof(float));
  float* write_ptr = zero_ptr + w_in;
  int pad_new = pad > 4 ? 4 : pad;
  int pad_0 = pad - pad_new;
  int h_out_new = h_out - 2 * pad_0;
  int mid_out = w_out - 2 * pad;
  int mid_cnt = mid_out >> 2;
  int mid_remain = mid_out - (mid_cnt << 2);
  int pad_cnt = pad_0 >> 2;
  int pad_remain = pad_0 - (pad_cnt << 2);
  int bias_cnt = (w_out * pad_0) >> 2;
  int bias_remain = (w_out * pad_0) - (bias_cnt << 2);
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
      float bias_c = flag_bias ? bias[c] : 0.f;
      float vbias[4] = {bias_c, bias_c, bias_c, bias_c};
      float32x4_t vbias_c = vdupq_n_f32(bias_c);
      if (flag_bias) {
        //! deal with h_out pad_0 line with bias
        for (int i = 0; i < bias_cnt; ++i) {
          vst1q_f32(dout_ch, vbias_c);
          dout_ch += 4;
        }
        for (int i = 0; i < bias_remain; ++i) {
          *dout_ch++ = bias_c;
        }
      } else {
        //! deal with h_out pad_0 line without bias
        for (int i = 0; i < pad_0; ++i) {
          memset(dout_ch, 0x00, w_out * sizeof(float));
          dout_ch += w_out;
        }
      }
      const float* din_list[6];
      //! set din ptr with zero buffer
      for (int i = 0; i < pad_new; ++i) {
        din_list[i] = zero_ptr;
      }
      //! set din ptr with input data
      for (int i = pad_new; i < 6; ++i) {
        din_list[i] = din_ch;
        din_ch += w_in;
      }
      //! every h loop, deal with 6 line input
      const float* din0 = din_list[0];
      const float* din1 = din_list[1];
      const float* din2 = din_list[2];
      const float* din3 = din_list[3];
      const float* din4 = din_list[4];
      const float* din5 = din_list[5];

      //! every h loop, deal with 2 line output
      float* dout0 = dout_ch;
      float* dout1 = dout0 + w_out;

      //! load weights to neon register
      const float* weights_c = weights + c * weights_saptial_size;

      //! h loop
      for (int h = 0; h < h_out_new; h += 2) {
        //! (h - pad_new) + 7 > h_in - 1
        if (h + 6 - pad_new > h_in) {
          switch (h + 6 - pad_new - h_in) {
            case 5:
              din1 = zero_ptr;
            case 4:
              din2 = zero_ptr;
            case 3:
              din3 = zero_ptr;
            case 2:
              din4 = zero_ptr;
            case 1:
              din5 = zero_ptr;
            default:
              break;
          }
        }
        if (h + 2 > h_out_new) {
          dout1 = write_ptr;
        }
        const float* din_ptr0 = din0;
        const float* din_ptr1 = din1;
        const float* din_ptr2 = din2;
        const float* din_ptr3 = din3;
        const float* din_ptr4 = din4;
        const float* din_ptr5 = din5;

        float* dout_ptr0 = dout0;
        float* dout_ptr1 = dout1;
        if (flag_bias) {
          //! deal with w_out pad_0 column pre with bias
          for (int i = 0; i < pad_cnt; i++) {
            vst1q_f32(dout_ptr0, vbias_c);
            vst1q_f32(dout_ptr1, vbias_c);
            dout_ptr0 += 4;
            dout_ptr1 += 4;
          }
          for (int i = 0; i < pad_remain; ++i) {
            *dout_ptr0++ = bias_c;
            *dout_ptr1++ = bias_c;
          }
        } else {
          //! deal with w_out pad_0 column pre without bias
          memset(dout_ptr0, 0x00, pad_0 * sizeof(float));
          memset(dout_ptr1, 0x00, pad_0 * sizeof(float));
          dout_ptr0 += pad_0;
          dout_ptr1 += pad_0;
        }

        //! deal with w_out pad_new column pre
        switch (pad_new) {
          case 4:
            compute_four_out_extract_pre(din_ptr0, din_ptr1, din_ptr2, din_ptr3,
                                         din_ptr4, din_ptr5, dout_ptr0,
                                         dout_ptr1, weights_c, vbias);
            dout_ptr0 += 4;
            dout_ptr1 += 4;
            break;
          case 3:
            compute_three_out_extract_pre(
                din_ptr0, din_ptr1, din_ptr2, din_ptr3, din_ptr4, din_ptr5,
                dout_ptr0, dout_ptr1, weights_c, vbias);
            dout_ptr0 += 3;
            dout_ptr1 += 3;
            break;
          case 2:
            compute_two_out_extract_pre(din_ptr0, din_ptr1, din_ptr2, din_ptr3,
                                        din_ptr4, din_ptr5, dout_ptr0,
                                        dout_ptr1, weights_c, vbias);
            dout_ptr0 += 2;
            dout_ptr1 += 2;
            break;
          case 1:
            compute_one_out_extract_pre(din_ptr0, din_ptr1, din_ptr2, din_ptr3,
                                        din_ptr4, din_ptr5, dout_ptr0,
                                        dout_ptr1, weights_c, vbias);
            dout_ptr0 += 1;
            dout_ptr1 += 1;
            break;
        }

        //! mid loop
        if (mid_cnt > 0) {
          int mid_loop = mid_cnt;
          const float* weights_ptr = weights_c;
          asm volatile(
              //! din: q7-q12
              //! dout: q13, q14
              "mov r1, #20  \n"
              //! load weights
              "vld1.32 {d0-d1}, [%[wh]], r1  \n"
              "vld1.32 {d2-d3}, [%[wh]], r1  \n"
              "vld1.32 {d4-d5}, [%[wh]], r1  \n"
              "vld1.32 {d6-d7}, [%[wh]], r1  \n"
              "vld1.32 {d8-d9}, [%[wh]]  \n"

              "sub %[wh], #64  \n"
              "vld1.32 {d10[0]}, [%[wh]], r1  \n"
              "vld1.32 {d10[1]}, [%[wh]], r1  \n"
              "vld1.32 {d11[0]}, [%[wh]], r1  \n"
              "vld1.32 {d11[1]}, [%[wh]], r1  \n"
              "vld1.32 {d12[0]}, [%[wh]]      \n"

              //! load input
              "mov r1, #4  \n"
              "vld1.32 {d14-d15}, [%[din0]], r1  \n"
              "vld1.32 {d16-d17}, [%[din1]], r1  \n"
              "vld1.32 {d18-d19}, [%[din2]], r1  \n"
              "vld1.32 {d20-d21}, [%[din3]], r1  \n"
              "vld1.32 {d22-d23}, [%[din4]], r1  \n"
              "vld1.32 {d24-d25}, [%[din5]], r1  \n"

              //! load bias
              "vld1.32 {d30-d31}, [%[bias]]  \n"

              "1: \n"
              //! add bias to output
              "vmov.32 q13, q15 \n"
              "vmov.32 q14, q15 \n"

              "pld [%[din0]]  \n"
              "pld [%[din1]]  \n"
              "pld [%[din2]]  \n"
              "pld [%[din3]]  \n"
              "pld [%[din4]]  \n"
              "pld [%[din5]]  \n"

              // weights col 0
              "vmla.f32 q13, q7, d0[0]  \n"
              "vmla.f32 q14, q8, d0[0]  \n"

              "vmla.f32 q13, q8, d2[0]  \n"
              "vmla.f32 q14, q9, d2[0]  \n"

              "vld1.32 {d14-d15}, [%[din0]], r1  \n"
              "vld1.32 {d16-d17}, [%[din1]], r1  \n"

              "vmla.f32 q13, q9, d4[0]  \n"
              "vmla.f32 q14, q10, d4[0]  \n"

              "vmla.f32 q13, q10, d6[0]  \n"
              "vmla.f32 q14, q11, d6[0]  \n"

              "vld1.32 {d18-d19}, [%[din2]], r1  \n"
              "vld1.32 {d20-d21}, [%[din3]], r1  \n"

              "vmla.f32 q13, q11, d8[0]  \n"
              "vmla.f32 q14, q12, d8[0]  \n"

              "vld1.32 {d22-d23}, [%[din4]], r1  \n"
              "vld1.32 {d24-d25}, [%[din5]], r1  \n"

              // weights col 1
              "vmla.f32 q13, q7, d0[1]  \n"
              "vmla.f32 q14, q8, d0[1]  \n"

              "vmla.f32 q13, q8, d2[1]   \n"
              "vmla.f32 q14, q9, d2[1]   \n"

              "vld1.32 {d14-d15}, [%[din0]], r1  \n"
              "vld1.32 {d16-d17}, [%[din1]], r1  \n"

              "vmla.f32 q13, q9, d4[1]  \n"
              "vmla.f32 q14, q10, d4[1]  \n"

              "vmla.f32 q13, q10, d6[1]  \n"
              "vmla.f32 q14, q11, d6[1]  \n"

              "vld1.32 {d18-d19}, [%[din2]], r1  \n"
              "vld1.32 {d20-d21}, [%[din3]], r1  \n"

              "vmla.f32 q13, q11, d8[1]  \n"
              "vmla.f32 q14, q12, d8[1]  \n"

              "vld1.32 {d22-d23}, [%[din4]], r1  \n"
              "vld1.32 {d24-d25}, [%[din5]], r1  \n"

              // weights col 2
              "vmla.f32 q13, q7, d1[0]  \n"
              "vmla.f32 q14, q8, d1[0]  \n"

              "vmla.f32 q13, q8, d3[0]   \n"
              "vmla.f32 q14, q9, d3[0]   \n"

              "vld1.32 {d14-d15}, [%[din0]], r1  \n"
              "vld1.32 {d16-d17}, [%[din1]], r1  \n"

              "vmla.f32 q13, q9, d5[0]  \n"
              "vmla.f32 q14, q10, d5[0]  \n"

              "vmla.f32 q13, q10, d7[0]  \n"
              "vmla.f32 q14, q11, d7[0]  \n"

              "vld1.32 {d18-d19}, [%[din2]], r1  \n"
              "vld1.32 {d20-d21}, [%[din3]], r1  \n"

              "vmla.f32 q13, q11, d9[0]  \n"
              "vmla.f32 q14, q12, d9[0]  \n"

              "vld1.32 {d22-d23}, [%[din4]], r1  \n"
              "vld1.32 {d24-d25}, [%[din5]], r1  \n"

              // weights col 3
              "vmla.f32 q13, q7, d1[1]  \n"
              "vmla.f32 q14, q8, d1[1]  \n"

              "vmla.f32 q13, q8, d3[1]   \n"
              "vmla.f32 q14, q9, d3[1]   \n"

              "vld1.32 {d14-d15}, [%[din0]], r1  \n"
              "vld1.32 {d16-d17}, [%[din1]], r1  \n"

              "vmla.f32 q13, q9, d5[1]  \n"
              "vmla.f32 q14, q10, d5[1]  \n"

              "vmla.f32 q13, q10, d7[1]  \n"
              "vmla.f32 q14, q11, d7[1]  \n"

              "vld1.32 {d18-d19}, [%[din2]], r1  \n"
              "vld1.32 {d20-d21}, [%[din3]], r1  \n"

              "vmla.f32 q13, q11, d9[1]  \n"
              "vmla.f32 q14, q12, d9[1]  \n"

              "vld1.32 {d22-d23}, [%[din4]], r1  \n"
              "vld1.32 {d24-d25}, [%[din5]], r1  \n"

              // weights col 4
              "vmla.f32 q13, q7, d10[0]  \n"
              "vmla.f32 q14, q8, d10[0]  \n"

              "vmla.f32 q13, q8,  d10[1]   \n"
              "vmla.f32 q14, q9, d10[1]   \n"

              "vmla.f32 q13, q9, d11[0]  \n"
              "vmla.f32 q14, q10, d11[0]  \n"

              "vmla.f32 q13, q10, d11[1]  \n"
              "vmla.f32 q14, q11, d11[1]  \n"

              "vmla.f32 q13, q11, d12[0]   \n"
              "vmla.f32 q14, q12, d12[0]  \n"

              // store reslult
              "vst1.32 {d26-d27}, [%[out0]]! \n"
              "vst1.32 {d28-d29}, [%[out1]]! \n"

              "subs %[cnt], #1  \n"
              "bne 1b  \n"

              "sub %[din0], r1  \n"
              "sub %[din1], r1  \n"
              "sub %[din2], r1  \n"
              "sub %[din3], r1  \n"
              "sub %[din4], r1  \n"
              "sub %[din5], r1  \n"

              : [din0] "+r"(din_ptr0), [din1] "+r"(din_ptr1),
                [din2] "+r"(din_ptr2), [din3] "+r"(din_ptr3),
                [din4] "+r"(din_ptr4), [din5] "+r"(din_ptr5),
                [out0] "+r"(dout_ptr0), [out1] "+r"(dout_ptr1),
                [wh] "+r"(weights_ptr), [cnt] "+r"(mid_loop)
              : [bias] "r"(vbias)
              : "cc", "memory", "r1", "q0", "q1", "q2", "q3", "q4", "q5", "q6",
                "q7", "q8", "q9", "q10", "q11", "q12", "q13", "q14", "q15");
        }
        //! deal with mid remain
        for (int i = 0; i < mid_remain; ++i) {
          compute_one_out_without_extract(
              din_ptr0, din_ptr1, din_ptr2, din_ptr3, din_ptr4, din_ptr5,
              dout_ptr0, dout_ptr1, weights_c, vbias);
          din_ptr0++;
          din_ptr1++;
          din_ptr2++;
          din_ptr3++;
          din_ptr4++;
          din_ptr5++;

          dout_ptr0++;
          dout_ptr1++;
        }
        //! deal with w_out pad_new column post
        switch (pad_new) {
          case 4:
            compute_four_out_extract_post(
                din_ptr0, din_ptr1, din_ptr2, din_ptr3, din_ptr4, din_ptr5,
                dout_ptr0, dout_ptr1, weights_c, vbias);
            dout_ptr0 += 4;
            dout_ptr1 += 4;
            break;
          case 3:
            compute_three_out_extract_post(
                din_ptr0, din_ptr1, din_ptr2, din_ptr3, din_ptr4, din_ptr5,
                dout_ptr0, dout_ptr1, weights_c, vbias);
            dout_ptr0 += 3;
            dout_ptr1 += 3;
            break;
          case 2:
            compute_two_out_extract_post(din_ptr0, din_ptr1, din_ptr2, din_ptr3,
                                         din_ptr4, din_ptr5, dout_ptr0,
                                         dout_ptr1, weights_c, vbias);
            dout_ptr0 += 2;
            dout_ptr1 += 2;
            break;
          case 1:
            compute_one_out_extract_post(din_ptr0, din_ptr1, din_ptr2, din_ptr3,
                                         din_ptr4, din_ptr5, dout_ptr0,
                                         dout_ptr1, weights_c, vbias);
            dout_ptr0 += 1;
            dout_ptr1 += 1;
            break;
        }

        if (flag_bias) {
          //! deal with w_out pad_0 column post with bias
          memcpy(dout_ptr0, dout0, pad_0 * sizeof(float));
          memcpy(dout_ptr1, dout1, pad_0 * sizeof(float));
        } else {
          //! deal with w_out pad_0 column post without bias
          memset(dout_ptr0, 0x00, pad_0 * sizeof(float));
          memset(dout_ptr1, 0x00, pad_0 * sizeof(float));
        }

        din0 = din2;
        din1 = din3;
        din2 = din4;
        din3 = din5;
        din4 = din3 + w_in;
        din5 = din4 + w_in;

        dout0 = dout1 + w_out;
        dout1 = dout0 + w_out;
      }
      float* dout_pad_end = dout_ch + h_out_new * w_out;
      if (flag_bias) {
        //! deal with h_out pad_0 line with bias
        memcpy(reinterpret_cast<void*>(dout_pad_end), dout_ch - pad_0 * w_out,
               pad_0 * w_out * sizeof(float));
      } else {
        //! deal with h_out pad_0 line without bias
        memset(reinterpret_cast<void*>(dout_pad_end), 0x00,
               pad_0 * w_out * sizeof(float));
      }
    }
  }
}

void conv_depthwise_5x5s1_relu_impl(const float* din, float* dout, int num,
                                    int ch_out, int h_out, int w_out, int ch_in,
                                    int h_in, int w_in, const float* weights,
                                    const float* bias, int pad, bool flag_bias,
                                    bool flag_relu, ARMContext* ctx) {
  float* zero_ptr = ctx->workspace_data<float>();
  memset(zero_ptr, 0, w_in * sizeof(float));
  float* write_ptr = zero_ptr + w_in;
  int pad_new = pad > 4 ? 4 : pad;
  int pad_0 = pad - pad_new;
  int h_out_new = h_out - 2 * pad_0;
  int mid_out = w_out - 2 * pad;
  int mid_cnt = mid_out >> 2;
  int mid_remain = mid_out - (mid_cnt << 2);
  int pad_cnt = pad_0 >> 2;
  int pad_remain = pad_0 - (pad_cnt << 2);
  int bias_cnt = (w_out * pad_0) >> 2;
  int bias_remain = (w_out * pad_0) - (bias_cnt << 2);
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
      float bias_c = flag_bias ? bias[c] : 0.f;
      float bias_relu = bias_c > 0.f ? bias_c : 0.f;
      float vbias[4] = {bias_c, bias_c, bias_c, bias_c};
      float32x4_t vbias_c = vdupq_n_f32(bias_relu);
      if (flag_bias) {
        //! deal with h_out pad_0 line with bias
        for (int i = 0; i < bias_cnt; ++i) {
          vst1q_f32(dout_ch, vbias_c);
          dout_ch += 4;
        }
        for (int i = 0; i < bias_remain; ++i) {
          *dout_ch++ = bias_relu;
        }
      } else {
        //! deal with h_out pad_0 line without bias
        for (int i = 0; i < pad_0; ++i) {
          memset(dout_ch, 0x00, w_out * sizeof(float));
          dout_ch += w_out;
        }
      }
      const float* din_list[6];
      //! set din ptr with zero buffer
      for (int i = 0; i < pad_new; ++i) {
        din_list[i] = zero_ptr;
      }
      //! set din ptr with input data
      for (int i = pad_new; i < 6; ++i) {
        din_list[i] = din_ch;
        din_ch += w_in;
      }
      //! every h loop, deal with 6 line input
      const float* din0 = din_list[0];
      const float* din1 = din_list[1];
      const float* din2 = din_list[2];
      const float* din3 = din_list[3];
      const float* din4 = din_list[4];
      const float* din5 = din_list[5];

      //! every h loop, deal with 2 line output
      float* dout0 = dout_ch;
      float* dout1 = dout0 + w_out;

      //! load weights to neon register
      const float* weights_c = weights + c * weights_saptial_size;

      //! h loop
      for (int h = 0; h < h_out_new; h += 2) {
        //! (h - pad_new) + 7 > h_in - 1
        if (h + 6 - pad_new > h_in) {
          switch (h + 6 - pad_new - h_in) {
            case 5:
              din1 = zero_ptr;
            case 4:
              din2 = zero_ptr;
            case 3:
              din3 = zero_ptr;
            case 2:
              din4 = zero_ptr;
            case 1:
              din5 = zero_ptr;
            default:
              break;
          }
        }
        if (h + 2 > h_out_new) {
          dout1 = write_ptr;
        }
        const float* din_ptr0 = din0;
        const float* din_ptr1 = din1;
        const float* din_ptr2 = din2;
        const float* din_ptr3 = din3;
        const float* din_ptr4 = din4;
        const float* din_ptr5 = din5;

        float* dout_ptr0 = dout0;
        float* dout_ptr1 = dout1;
        if (flag_bias) {
          //! deal with w_out pad_0 column pre with bias
          for (int i = 0; i < pad_cnt; i++) {
            vst1q_f32(dout_ptr0, vbias_c);
            vst1q_f32(dout_ptr1, vbias_c);
            dout_ptr0 += 4;
            dout_ptr1 += 4;
          }
          for (int i = 0; i < pad_remain; ++i) {
            *dout_ptr0++ = bias_relu;
            *dout_ptr1++ = bias_relu;
          }
        } else {
          //! deal with w_out pad_0 column pre without bias
          memset(dout_ptr0, 0x00, pad_0 * sizeof(float));
          memset(dout_ptr1, 0x00, pad_0 * sizeof(float));
          dout_ptr0 += pad_0;
          dout_ptr1 += pad_0;
        }

        //! deal with w_out pad_new column pre
        switch (pad_new) {
          case 4:
            compute_four_out_extract_pre_relu(
                din_ptr0, din_ptr1, din_ptr2, din_ptr3, din_ptr4, din_ptr5,
                dout_ptr0, dout_ptr1, weights_c, vbias);
            dout_ptr0 += 4;
            dout_ptr1 += 4;
            break;
          case 3:
            compute_three_out_extract_pre_relu(
                din_ptr0, din_ptr1, din_ptr2, din_ptr3, din_ptr4, din_ptr5,
                dout_ptr0, dout_ptr1, weights_c, vbias);
            dout_ptr0 += 3;
            dout_ptr1 += 3;
            break;
          case 2:
            compute_two_out_extract_pre_relu(
                din_ptr0, din_ptr1, din_ptr2, din_ptr3, din_ptr4, din_ptr5,
                dout_ptr0, dout_ptr1, weights_c, vbias);
            dout_ptr0 += 2;
            dout_ptr1 += 2;
            break;
          case 1:
            compute_one_out_extract_pre_relu(
                din_ptr0, din_ptr1, din_ptr2, din_ptr3, din_ptr4, din_ptr5,
                dout_ptr0, dout_ptr1, weights_c, vbias);
            dout_ptr0 += 1;
            dout_ptr1 += 1;
            break;
        }

        //! mid loop
        if (mid_cnt > 0) {
          int mid_loop = mid_cnt;
          const float* weights_ptr = weights_c;
          asm volatile(
              //! din: q7-q12
              //! dout: q13, q14
              "mov r1, #20  \n"
              "vmov.i32 q15, #0x0  \n"
              //! load weights
              "vld1.32 {d0-d1}, [%[wh]], r1  \n"
              "vld1.32 {d2-d3}, [%[wh]], r1  \n"
              "vld1.32 {d4-d5}, [%[wh]], r1  \n"
              "vld1.32 {d6-d7}, [%[wh]], r1  \n"
              "vld1.32 {d8-d9}, [%[wh]]  \n"

              "sub %[wh], #64  \n"
              "vld1.32 {d10[0]}, [%[wh]], r1  \n"
              "vld1.32 {d10[1]}, [%[wh]], r1  \n"
              "vld1.32 {d11[0]}, [%[wh]], r1  \n"
              "vld1.32 {d11[1]}, [%[wh]], r1  \n"
              "vld1.32 {d12[0]}, [%[wh]]      \n"

              //! load input
              "mov r1, #4  \n"
              "vld1.32 {d14-d15}, [%[din0]], r1  \n"
              "vld1.32 {d16-d17}, [%[din1]], r1  \n"
              "vld1.32 {d18-d19}, [%[din2]], r1  \n"
              "vld1.32 {d20-d21}, [%[din3]], r1  \n"
              "vld1.32 {d22-d23}, [%[din4]], r1  \n"
              "vld1.32 {d24-d25}, [%[din5]], r1  \n"

              "1: \n"

              //! load bias to output
              "vld1.32 {d26-d27}, [%[bias]] \n"
              "vld1.32 {d28-d29}, [%[bias]] \n"

              "pld [%[din0]]  \n"
              "pld [%[din1]]  \n"
              "pld [%[din2]]  \n"
              "pld [%[din3]]  \n"
              "pld [%[din4]]  \n"
              "pld [%[din5]]  \n"

              // weights col 0
              "vmla.f32 q13, q7, d0[0]  \n"
              "vmla.f32 q14, q8, d0[0]  \n"

              "vmla.f32 q13, q8, d2[0]  \n"
              "vmla.f32 q14, q9, d2[0]  \n"

              "vld1.32 {d14-d15}, [%[din0]], r1  \n"
              "vld1.32 {d16-d17}, [%[din1]], r1  \n"

              "vmla.f32 q13, q9, d4[0]  \n"
              "vmla.f32 q14, q10, d4[0]  \n"

              "vmla.f32 q13, q10, d6[0]  \n"
              "vmla.f32 q14, q11, d6[0]  \n"

              "vld1.32 {d18-d19}, [%[din2]], r1  \n"
              "vld1.32 {d20-d21}, [%[din3]], r1  \n"

              "vmla.f32 q13, q11, d8[0]  \n"
              "vmla.f32 q14, q12, d8[0]  \n"

              "vld1.32 {d22-d23}, [%[din4]], r1  \n"
              "vld1.32 {d24-d25}, [%[din5]], r1  \n"

              // weights col 1
              "vmla.f32 q13, q7, d0[1]  \n"
              "vmla.f32 q14, q8, d0[1]  \n"

              "vmla.f32 q13, q8, d2[1]   \n"
              "vmla.f32 q14, q9, d2[1]   \n"

              "vld1.32 {d14-d15}, [%[din0]], r1  \n"
              "vld1.32 {d16-d17}, [%[din1]], r1  \n"

              "vmla.f32 q13, q9, d4[1]  \n"
              "vmla.f32 q14, q10, d4[1]  \n"

              "vmla.f32 q13, q10, d6[1]  \n"
              "vmla.f32 q14, q11, d6[1]  \n"

              "vld1.32 {d18-d19}, [%[din2]], r1  \n"
              "vld1.32 {d20-d21}, [%[din3]], r1  \n"

              "vmla.f32 q13, q11, d8[1]  \n"
              "vmla.f32 q14, q12, d8[1]  \n"

              "vld1.32 {d22-d23}, [%[din4]], r1  \n"
              "vld1.32 {d24-d25}, [%[din5]], r1  \n"

              // weights col 2
              "vmla.f32 q13, q7, d1[0]  \n"
              "vmla.f32 q14, q8, d1[0]  \n"

              "vmla.f32 q13, q8, d3[0]   \n"
              "vmla.f32 q14, q9, d3[0]   \n"

              "vld1.32 {d14-d15}, [%[din0]], r1  \n"
              "vld1.32 {d16-d17}, [%[din1]], r1  \n"

              "vmla.f32 q13, q9, d5[0]  \n"
              "vmla.f32 q14, q10, d5[0]  \n"

              "vmla.f32 q13, q10, d7[0]  \n"
              "vmla.f32 q14, q11, d7[0]  \n"

              "vld1.32 {d18-d19}, [%[din2]], r1  \n"
              "vld1.32 {d20-d21}, [%[din3]], r1  \n"

              "vmla.f32 q13, q11, d9[0]  \n"
              "vmla.f32 q14, q12, d9[0]  \n"

              "vld1.32 {d22-d23}, [%[din4]], r1  \n"
              "vld1.32 {d24-d25}, [%[din5]], r1  \n"

              // weights col 3
              "vmla.f32 q13, q7, d1[1]  \n"
              "vmla.f32 q14, q8, d1[1]  \n"

              "vmla.f32 q13, q8, d3[1]   \n"
              "vmla.f32 q14, q9, d3[1]   \n"

              "vld1.32 {d14-d15}, [%[din0]], r1  \n"
              "vld1.32 {d16-d17}, [%[din1]], r1  \n"

              "vmla.f32 q13, q9, d5[1]  \n"
              "vmla.f32 q14, q10, d5[1]  \n"

              "vmla.f32 q13, q10, d7[1]  \n"
              "vmla.f32 q14, q11, d7[1]  \n"

              "vld1.32 {d18-d19}, [%[din2]], r1  \n"
              "vld1.32 {d20-d21}, [%[din3]], r1  \n"

              "vmla.f32 q13, q11, d9[1]  \n"
              "vmla.f32 q14, q12, d9[1]  \n"

              "vld1.32 {d22-d23}, [%[din4]], r1  \n"
              "vld1.32 {d24-d25}, [%[din5]], r1  \n"

              // weights col 4
              "vmla.f32 q13, q7, d10[0]  \n"
              "vmla.f32 q14, q8, d10[0]  \n"

              "vmla.f32 q13, q8,  d10[1]   \n"
              "vmla.f32 q14, q9, d10[1]   \n"

              "vmla.f32 q13, q9, d11[0]  \n"
              "vmla.f32 q14, q10, d11[0]  \n"

              "vmla.f32 q13, q10, d11[1]  \n"
              "vmla.f32 q14, q11, d11[1]  \n"

              "vmla.f32 q13, q11, d12[0]   \n"
              "vmla.f32 q14, q12, d12[0]  \n"

              // relu
              "vmax.f32 q13, q13, q15  \n"
              "vmax.f32 q14, q14, q15  \n"

              // store result
              "vst1.32 {d26-d27}, [%[out0]]! \n"
              "vst1.32 {d28-d29}, [%[out1]]! \n"

              "subs %[cnt], #1  \n"
              "bne 1b  \n"

              "sub %[din0], r1  \n"
              "sub %[din1], r1  \n"
              "sub %[din2], r1  \n"
              "sub %[din3], r1  \n"
              "sub %[din4], r1  \n"
              "sub %[din5], r1  \n"

              : [din0] "+r"(din_ptr0), [din1] "+r"(din_ptr1),
                [din2] "+r"(din_ptr2), [din3] "+r"(din_ptr3),
                [din4] "+r"(din_ptr4), [din5] "+r"(din_ptr5),
                [out0] "+r"(dout_ptr0), [out1] "+r"(dout_ptr1),
                [wh] "+r"(weights_ptr), [cnt] "+r"(mid_loop)
              : [bias] "r"(vbias)
              : "cc", "memory", "r1", "q0", "q1", "q2", "q3", "q4", "q5", "q6",
                "q7", "q8", "q9", "q10", "q11", "q12", "q13", "q14", "q15");
        }
        //! deal with mid remain
        for (int i = 0; i < mid_remain; ++i) {
          compute_one_out_without_extract_relu(
              din_ptr0, din_ptr1, din_ptr2, din_ptr3, din_ptr4, din_ptr5,
              dout_ptr0, dout_ptr1, weights_c, vbias);
          din_ptr0++;
          din_ptr1++;
          din_ptr2++;
          din_ptr3++;
          din_ptr4++;
          din_ptr5++;

          dout_ptr0++;
          dout_ptr1++;
        }
        //! deal with w_out pad_new column post
        switch (pad_new) {
          case 4:
            compute_four_out_extract_post_relu(
                din_ptr0, din_ptr1, din_ptr2, din_ptr3, din_ptr4, din_ptr5,
                dout_ptr0, dout_ptr1, weights_c, vbias);
            dout_ptr0 += 4;
            dout_ptr1 += 4;
            break;
          case 3:
            compute_three_out_extract_post_relu(
                din_ptr0, din_ptr1, din_ptr2, din_ptr3, din_ptr4, din_ptr5,
                dout_ptr0, dout_ptr1, weights_c, vbias);
            dout_ptr0 += 3;
            dout_ptr1 += 3;
            break;
          case 2:
            compute_two_out_extract_post_relu(
                din_ptr0, din_ptr1, din_ptr2, din_ptr3, din_ptr4, din_ptr5,
                dout_ptr0, dout_ptr1, weights_c, vbias);
            dout_ptr0 += 2;
            dout_ptr1 += 2;
            break;
          case 1:
            compute_one_out_extract_post_relu(
                din_ptr0, din_ptr1, din_ptr2, din_ptr3, din_ptr4, din_ptr5,
                dout_ptr0, dout_ptr1, weights_c, vbias);
            dout_ptr0 += 1;
            dout_ptr1 += 1;
            break;
        }

        if (flag_bias) {
          //! deal with w_out pad_0 column post with bias
          memcpy(dout_ptr0, dout0, pad_0 * sizeof(float));
          memcpy(dout_ptr1, dout1, pad_0 * sizeof(float));
        } else {
          //! deal with w_out pad_0 column post without bias
          memset(dout_ptr0, 0x00, pad_0 * sizeof(float));
          memset(dout_ptr1, 0x00, pad_0 * sizeof(float));
        }

        din0 = din2;
        din1 = din3;
        din2 = din4;
        din3 = din5;
        din4 = din3 + w_in;
        din5 = din4 + w_in;

        dout0 = dout1 + w_out;
        dout1 = dout0 + w_out;
      }
      float* dout_pad_end = dout_ch + h_out_new * w_out;
      if (flag_bias) {
        //! deal with h_out pad_0 line with bias
        memcpy(reinterpret_cast<void*>(dout_pad_end), dout_ch - pad_0 * w_out,
               pad_0 * w_out * sizeof(float));
      } else {
        //! deal with h_out pad_0 line without bias
        memset(reinterpret_cast<void*>(dout_pad_end), 0x00,
               pad_0 * w_out * sizeof(float));
      }
    }
  }
}

void conv_depthwise_5x5s1_small_impl(const float* din, float* dout, int num,
                                     int ch_out, int h_out, int w_out,
                                     int ch_in, int h_in, int w_in,
                                     const float* weights, const float* bias,
                                     int pad, bool flag_bias, bool flag_relu,
                                     ARMContext* ctx) {
  int pad_new = pad > 4 ? 4 : pad;
  int pad_0 = pad - pad_new;
  int h_in_new = h_in + 2 * pad_new;
  int w_in_new = w_in + 2 * pad_new;
  int h_out_new = h_out - 2 * pad_0;
  int w_out_new = w_out - 2 * pad_0;
  float zero_ptr[w_in_new + w_out];
  memset(zero_ptr, 0, w_in_new * sizeof(float));
  float* write_ptr = zero_ptr + w_in_new;
  int pad_cnt = pad_0 >> 2;
  int pad_remain = pad_0 - (pad_cnt << 2);
  int bias_cnt = (w_out * pad_0) >> 2;
  int bias_remain = (w_out * pad_0) - (bias_cnt << 2);
  int in_spatial_size = w_in_new * h_in_new;
  int out_spatial_size = w_out * h_out;
  int weights_saptial_size = 25;

  float* din_new = prepad_input(din, num, ch_in, h_in, w_in, pad_new);
  for (int n = 0; n < num; ++n) {
    const float* din_batch = din_new + n * in_spatial_size * ch_in;
    float* dout_batch = dout + n * out_spatial_size * ch_out;
#pragma omp parallel for
    for (int c = 0; c < ch_in; ++c) {
      const float* din_ch = din_batch + c * in_spatial_size;
      float* dout_ch = dout_batch + c * out_spatial_size;
      float bias_c = flag_bias ? bias[c] : 0.f;
      float vbias[4] = {bias_c, bias_c, bias_c, bias_c};
      float32x4_t vbias_c = vdupq_n_f32(bias_c);
      if (flag_bias) {
        //! deal with h_out pad_0 line with bias
        for (int i = 0; i < bias_cnt; ++i) {
          vst1q_f32(dout_ch, vbias_c);
          dout_ch += 4;
        }
        for (int i = 0; i < bias_remain; ++i) {
          *dout_ch++ = bias_c;
        }
      } else {
        //! deal with h_out pad_0 line without bias
        for (int i = 0; i < pad_0; ++i) {
          memset(dout_ch, 0x00, w_out * sizeof(float));
          dout_ch += w_out;
        }
      }
      //! every h loop, deal with 6 line input
      const float* din0 = din_ch;
      const float* din1 = din0 + w_in_new;
      const float* din2 = din1 + w_in_new;
      const float* din3 = din2 + w_in_new;
      const float* din4 = din3 + w_in_new;
      const float* din5 = din4 + w_in_new;
      //! every h loop, deal with 2 line output
      float* dout0 = dout_ch;
      float* dout1 = dout0 + w_out;

      const float* weights_c = weights + c * weights_saptial_size;

      //! h loop
      for (int h = 0; h < h_out_new; h += 2) {
        //! (h - pad_new) + 6 > h_in - 1
        if (h + 6 > h_in_new) {
          switch (h + 6 - h_in_new) {
            case 5:
              din1 = zero_ptr;
            case 4:
              din2 = zero_ptr;
            case 3:
              din3 = zero_ptr;
            case 2:
              din4 = zero_ptr;
            case 1:
              din5 = zero_ptr;
            default:
              break;
          }
        }
        if (h + 2 > h_out_new) {
          dout1 = write_ptr;
        }
        const float* din_ptr0 = din0;
        const float* din_ptr1 = din1;
        const float* din_ptr2 = din2;
        const float* din_ptr3 = din3;
        const float* din_ptr4 = din4;
        const float* din_ptr5 = din5;

        float* dout_ptr0 = dout0;
        float* dout_ptr1 = dout1;

        if (flag_bias) {
          //! deal with w_out pad_0 column pre with bias
          for (int i = 0; i < pad_cnt; i++) {
            vst1q_f32(dout_ptr0, vbias_c);
            vst1q_f32(dout_ptr1, vbias_c);
            dout_ptr0 += 4;
            dout_ptr1 += 4;
          }
          for (int i = 0; i < pad_remain; ++i) {
            *dout_ptr0++ = bias_c;
            *dout_ptr1++ = bias_c;
          }
        } else {
          //! deal with w_out pad_0 column pre without bias
          memset(dout_ptr0, 0x00, pad_0 * sizeof(float));
          memset(dout_ptr1, 0x00, pad_0 * sizeof(float));
          dout_ptr0 += pad_0;
          dout_ptr1 += pad_0;
        }
        //! mid loop
        for (int i = 0; i < w_out_new; ++i) {
          compute_one_out_without_extract(
              din_ptr0, din_ptr1, din_ptr2, din_ptr3, din_ptr4, din_ptr5,
              dout_ptr0, dout_ptr1, weights_c, vbias);
          din_ptr0++;
          din_ptr1++;
          din_ptr2++;
          din_ptr3++;
          din_ptr4++;
          din_ptr5++;

          dout_ptr0++;
          dout_ptr1++;
        }
        if (flag_bias) {
          //! deal with w_out pad_0 column post with bias
          memcpy(dout_ptr0, dout0, pad_0 * sizeof(float));
          memcpy(dout_ptr1, dout1, pad_0 * sizeof(float));
        } else {
          //! deal with w_out pad_0 column post without bias
          memset(dout_ptr0, 0x00, pad_0 * sizeof(float));
          memset(dout_ptr1, 0x00, pad_0 * sizeof(float));
        }

        din0 = din2;
        din1 = din3;
        din2 = din4;
        din3 = din5;
        din4 = din3 + w_in_new;
        din5 = din4 + w_in_new;

        dout0 = dout1 + w_out;
        dout1 = dout0 + w_out;
      }
      float* dout_pad_end = dout_ch + h_out_new * w_out;
      if (flag_bias) {
        //! deal with h_out pad_0 line with bias
        memcpy(reinterpret_cast<void*>(dout_pad_end), dout_ch - pad_0 * w_out,
               pad_0 * w_out * sizeof(float));
      } else {
        //! deal with h_out pad_0 line without bias
        memset(reinterpret_cast<void*>(dout_pad_end), 0x00,
               pad_0 * w_out * sizeof(float));
      }
    }
  }
  free(din_new);
}

void conv_depthwise_5x5s1_small_relu_impl(
    const float* din, float* dout, int num, int ch_out, int h_out, int w_out,
    int ch_in, int h_in, int w_in, const float* weights, const float* bias,
    int pad, bool flag_bias, bool flag_relu, ARMContext* ctx) {
  int pad_new = pad > 4 ? 4 : pad;
  int pad_0 = pad - pad_new;
  int h_in_new = h_in + 2 * pad_new;
  int w_in_new = w_in + 2 * pad_new;
  int h_out_new = h_out - 2 * pad_0;
  int w_out_new = w_out - 2 * pad_0;
  float zero_ptr[w_in_new + w_out];
  memset(zero_ptr, 0, w_in_new * sizeof(float));
  float* write_ptr = zero_ptr + w_in_new;
  int pad_cnt = pad_0 >> 2;
  int pad_remain = pad_0 - (pad_cnt << 2);
  int bias_cnt = (w_out * pad_0) >> 2;
  int bias_remain = (w_out * pad_0) - (bias_cnt << 2);
  int in_spatial_size = w_in_new * h_in_new;
  int out_spatial_size = w_out * h_out;
  int weights_saptial_size = 25;

  float* din_new = prepad_input(din, num, ch_in, h_in, w_in, pad_new);
  for (int n = 0; n < num; ++n) {
    const float* din_batch = din_new + n * in_spatial_size * ch_in;
    float* dout_batch = dout + n * out_spatial_size * ch_out;
#pragma omp parallel for
    for (int c = 0; c < ch_in; ++c) {
      const float* din_ch = din_batch + c * in_spatial_size;
      float* dout_ch = dout_batch + c * out_spatial_size;
      float bias_c = flag_bias ? bias[c] : 0.f;
      float bias_relu = bias_c > 0.f ? bias_c : 0.f;
      float vbias[4] = {bias_c, bias_c, bias_c, bias_c};
      float32x4_t vbias_c = vdupq_n_f32(bias_relu);
      if (flag_bias) {
        //! deal with h_out pad_0 line with bias
        for (int i = 0; i < bias_cnt; ++i) {
          vst1q_f32(dout_ch, vbias_c);
          dout_ch += 4;
        }
        for (int i = 0; i < bias_remain; ++i) {
          *dout_ch++ = bias_relu;
        }
      } else {
        //! deal with h_out pad_0 line without bias
        for (int i = 0; i < pad_0; ++i) {
          memset(dout_ch, 0x00, w_out * sizeof(float));
          dout_ch += w_out;
        }
      }
      //! every h loop, deal with 6 line input
      const float* din0 = din_ch;
      const float* din1 = din0 + w_in_new;
      const float* din2 = din1 + w_in_new;
      const float* din3 = din2 + w_in_new;
      const float* din4 = din3 + w_in_new;
      const float* din5 = din4 + w_in_new;
      //! every h loop, deal with 2 line output
      float* dout0 = dout_ch;
      float* dout1 = dout0 + w_out;

      const float* weights_c = weights + c * weights_saptial_size;

      //! h loop
      for (int h = 0; h < h_out_new; h += 2) {
        //! (h - pad_new) + 6 > h_in - 1
        if (h + 6 > h_in_new) {
          switch (h + 6 - h_in_new) {
            case 5:
              din1 = zero_ptr;
            case 4:
              din2 = zero_ptr;
            case 3:
              din3 = zero_ptr;
            case 2:
              din4 = zero_ptr;
            case 1:
              din5 = zero_ptr;
            default:
              break;
          }
        }
        if (h + 2 > h_out_new) {
          dout1 = write_ptr;
        }
        const float* din_ptr0 = din0;
        const float* din_ptr1 = din1;
        const float* din_ptr2 = din2;
        const float* din_ptr3 = din3;
        const float* din_ptr4 = din4;
        const float* din_ptr5 = din5;

        const float* weights_ptr = weights_c;
        float* dout_ptr0 = dout0;
        float* dout_ptr1 = dout1;

        if (flag_bias) {
          //! deal with w_out pad_0 column pre with bias
          for (int i = 0; i < pad_cnt; i++) {
            vst1q_f32(dout_ptr0, vbias_c);
            vst1q_f32(dout_ptr1, vbias_c);
            dout_ptr0 += 4;
            dout_ptr1 += 4;
          }
          for (int i = 0; i < pad_remain; ++i) {
            *dout_ptr0++ = bias_relu;
            *dout_ptr1++ = bias_relu;
          }
        } else {
          //! deal with w_out pad_0 column pre without bias
          memset(dout_ptr0, 0x00, pad_0 * sizeof(float));
          memset(dout_ptr1, 0x00, pad_0 * sizeof(float));
          dout_ptr0 += pad_0;
          dout_ptr1 += pad_0;
        }
        //! mid loop
        for (int i = 0; i < w_out_new; ++i) {
          compute_one_out_without_extract_relu(
              din_ptr0, din_ptr1, din_ptr2, din_ptr3, din_ptr4, din_ptr5,
              dout_ptr0, dout_ptr1, weights_c, vbias);
          din_ptr0++;
          din_ptr1++;
          din_ptr2++;
          din_ptr3++;
          din_ptr4++;
          din_ptr5++;

          dout_ptr0++;
          dout_ptr1++;
        }
        if (flag_bias) {
          //! deal with w_out pad_0 column post with bias
          memcpy(dout_ptr0, dout0, pad_0 * sizeof(float));
          memcpy(dout_ptr1, dout1, pad_0 * sizeof(float));
        } else {
          //! deal with w_out pad_0 column post without bias
          memset(dout_ptr0, 0x00, pad_0 * sizeof(float));
          memset(dout_ptr1, 0x00, pad_0 * sizeof(float));
        }

        din0 = din2;
        din1 = din3;
        din2 = din4;
        din3 = din5;
        din4 = din3 + w_in_new;
        din5 = din4 + w_in_new;

        dout0 = dout1 + w_out;
        dout1 = dout0 + w_out;
      }
      float* dout_pad_end = dout_ch + h_out_new * w_out;
      if (flag_bias) {
        //! deal with h_out pad_0 line with bias
        memcpy(reinterpret_cast<void*>(dout_pad_end), dout_ch - pad_0 * w_out,
               pad_0 * w_out * sizeof(float));
      } else {
        //! deal with h_out pad_0 line without bias
        memset(reinterpret_cast<void*>(dout_pad_end), 0x00,
               pad_0 * w_out * sizeof(float));
      }
    }
  }
  free(din_new);
}
#endif  // __aarch64__

void conv_depthwise_5x5s1(const float* din, float* dout, int num, int chout,
                          int hout, int wout, int chin, int hin, int win,
                          const float* weights, const float* bias, int pad,
                          bool flag_bias, bool flag_relu, ARMContext* ctx) {
  if (win < 4) {
    if (flag_relu) {
      conv_depthwise_5x5s1_small_relu_impl(din, dout, num, chout, hout, wout,
                                           chin, hin, win, weights, bias, pad,
                                           flag_bias, flag_relu, ctx);
    } else {
      conv_depthwise_5x5s1_small_impl(din, dout, num, chout, hout, wout, chin,
                                      hin, win, weights, bias, pad, flag_bias,
                                      flag_relu, ctx);
    }
  } else {
    if (flag_relu) {
      conv_depthwise_5x5s1_relu_impl(din, dout, num, chout, hout, wout, chin,
                                     hin, win, weights, bias, pad, flag_bias,
                                     flag_relu, ctx);
    } else {
      conv_depthwise_5x5s1_impl(din, dout, num, chout, hout, wout, chin, hin,
                                win, weights, bias, pad, flag_bias, flag_relu,
                                ctx);
    }
  }
}

}  // namespace math
}  // namespace arm
}  // namespace lite
}  // namespace paddle
