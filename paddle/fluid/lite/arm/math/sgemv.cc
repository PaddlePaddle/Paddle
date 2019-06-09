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

#include "paddle/fluid/lite/arm/math/sgemv.h"
#include <arm_neon.h>
#include "paddle/fluid/lite/utils/cp_logging.h"

namespace paddle {
namespace lite {
namespace arm {
namespace math {

void sgemv(const bool transA, const int M, const int N, const float *A,
           const float *x, float *y);

void sgemv_relu(const bool transA, const int M, const int N, const float *A,
                const float *x, float *y);

void sgemv_bias(const bool transA, const int M, const int N, const float *A,
                const float *x, float *y, const float *bias);

void sgemv_bias_relu(const bool transA, const int M, const int N,
                     const float *A, const float *x, float *y,
                     const float *bias);

bool sgemv(const float *A, const float *x, float *y, bool transA, int M, int N,
           bool is_bias, const float *bias, bool is_relu) {
  if (transA) {
    LOG(ERROR) << " sgemv, transA is not supported now";
    return false;
  }
  if (is_bias) {
    //! with bias
    if (is_relu) {
      //! with relu
      sgemv_bias_relu(transA, M, N, A, x, y, bias);
    } else {
      //! without relu
      sgemv_bias(transA, M, N, A, x, y, bias);
    }
  } else {
    //! without bias
    if (is_relu) {
      //! with relu
      sgemv_relu(transA, M, N, A, x, y);
    } else {
      //! without relu
      sgemv(transA, M, N, A, x, y);
    }
  }
  return true;
}

//! define compute kernel
#ifdef __aarch64__
#define SGEMV_IN_8                                    \
  "prfm  pldl1keep, [%[in]]   \n" /* preload din */   \
  "prfm  pldl1keep, [%[w0]]   \n" /* preload w0 */    \
  "prfm  pldl1keep, [%[w1]]   \n" /* preload w1 */    \
  "prfm  pldl1keep, [%[w2]]   \n" /* preload w2 */    \
  "prfm  pldl1keep, [%[w3]]   \n" /* preload w3 */    \
  "prfm  pldl1keep, [%[w4]]   \n" /* preload w4 */    \
  "prfm  pldl1keep, [%[w5]]   \n" /* preload w5 */    \
  "prfm  pldl1keep, [%[w6]]   \n" /* preload w6 */    \
  "prfm  pldl1keep, [%[w7]]   \n" /* preload w7 */    \
  "movi   v0.4s,  #0          \n" /* set out0 to 0 */ \
  "movi   v1.4s,  #0          \n" /* set out1 to 0 */ \
  "movi   v2.4s,  #0          \n" /* set out2 to 0 */ \
  "movi   v3.4s,  #0          \n" /* set out3 to 0 */ \
  "movi   v4.4s,  #0          \n" /* set out4 to 0 */ \
  "movi   v5.4s,  #0          \n" /* set out5 to 0 */ \
  "movi   v6.4s,  #0          \n" /* set out6 to 0 */ \
  "movi   v7.4s,  #0          \n" /* set out7 to 0 */

#define SGEMV_IN_8_BIAS                                    \
  "ldp   q8, q9, [%[bias_ptr]]\n" /* load bias to q8, q9*/ \
  "prfm  pldl1keep, [%[in]]   \n" /* preload din */        \
  "prfm  pldl1keep, [%[w0]]   \n" /* preload w0 */         \
  "prfm  pldl1keep, [%[w1]]   \n" /* preload w1 */         \
  "prfm  pldl1keep, [%[w2]]   \n" /* preload w2 */         \
  "prfm  pldl1keep, [%[w3]]   \n" /* preload w3 */         \
  "prfm  pldl1keep, [%[w4]]   \n" /* preload w4 */         \
  "prfm  pldl1keep, [%[w5]]   \n" /* preload w5 */         \
  "prfm  pldl1keep, [%[w6]]   \n" /* preload w6 */         \
  "prfm  pldl1keep, [%[w7]]   \n" /* preload w7 */         \
  "movi   v0.4s,  #0          \n" /* set out0 to 0 */      \
  "movi   v1.4s,  #0          \n" /* set out1 to 0 */      \
  "movi   v2.4s,  #0          \n" /* set out2 to 0 */      \
  "movi   v3.4s,  #0          \n" /* set out3 to 0 */      \
  "movi   v4.4s,  #0          \n" /* set out4 to 0 */      \
  "movi   v5.4s,  #0          \n" /* set out5 to 0 */      \
  "movi   v6.4s,  #0          \n" /* set out6 to 0 */      \
  "movi   v7.4s,  #0          \n" /* set out7 to 0 */      \
  "ins    v0.s[0], v8.s[0]    \n" /* out0 = bias0 */       \
  "ins    v1.s[0], v8.s[1]    \n" /* out1 = bias1 */       \
  "ins    v2.s[0], v8.s[2]    \n" /* out2 = bias2 */       \
  "ins    v3.s[0], v8.s[3]    \n" /* out3 = bias3 */       \
  "ins    v4.s[0], v9.s[0]    \n" /* out4 = bias4 */       \
  "ins    v5.s[0], v9.s[1]    \n" /* out5 = bias5 */       \
  "ins    v6.s[0], v9.s[2]    \n" /* out6 = bias6 */       \
  "ins    v7.s[0], v9.s[3]    \n" /* out7 = bias7 */

#define SGEMV_IN_1                                    \
  "prfm  pldl1keep, [%[in]]   \n" /* preload din */   \
  "prfm  pldl1keep, [%[w0]]   \n" /* preload w0 */    \
  "movi   v0.4s,  #0          \n" /* set out0 to 0 */ \
  "movi   v1.4s,  #0          \n" /* set out0 to 0 */

#define SGEMV_IN_1_BIAS                               \
  "prfm  pldl1keep, [%[in]]   \n" /* preload din */   \
  "prfm  pldl1keep, [%[w0]]   \n" /* preload w0 */    \
  "movi   v0.4s,  #0          \n" /* set out0 to 0 */ \
  "movi   v1.4s,  #0          \n" /* set out0 to 0 */ \
  "fmov   s0,  %w[bias0]      \n" /* set out0 = bias0 */

#define SGEMV_KERNEL_8                                                         \
  /* check main loop */                                                        \
  "cmp %w[cnt], #1            \n" /* check whether has main loop */            \
  "blt  2f                    \n" /* jump to tail */ /* main loop */           \
  "1:                         \n"                    /* main loop */           \
  "ldp q8, q9, [%[in]], #32   \n"                    /* load input 8 float */  \
  "ldp q10, q11, [%[w0]], #32 \n"                    /* load w0 8 float */     \
  "ldp q12, q13, [%[w1]], #32 \n"                    /* load w1 8 float */     \
  "ldp q14, q15, [%[w2]], #32 \n"                    /* load w2 8 float */     \
  "ldp q16, q17, [%[w3]], #32 \n"                    /* load w3 8 float */     \
  "ldp q18, q19, [%[w4]], #32 \n"                    /* load w4 8 float */     \
  "ldp q20, q21, [%[w5]], #32 \n"                    /* load w5 8 float */     \
  "ldp q22, q23, [%[w6]], #32 \n"                    /* load w6 8 float */     \
  "ldp q24, q25, [%[w7]], #32 \n"                    /* load w7 8 float */     \
  "fmla v0.4s, v8.4s, v10.4s  \n"                    /* mul + add*/            \
  "fmla v1.4s, v8.4s, v12.4s  \n"                    /* mul + add*/            \
  "fmla v2.4s, v8.4s, v14.4s  \n"                    /* mul + add*/            \
  "fmla v3.4s, v8.4s, v16.4s  \n"                    /* mul + add*/            \
  "fmla v4.4s, v8.4s, v18.4s  \n"                    /* mul + add*/            \
  "fmla v5.4s, v8.4s, v20.4s  \n"                    /* mul + add*/            \
  "fmla v6.4s, v8.4s, v22.4s  \n"                    /* mul + add*/            \
  "fmla v7.4s, v8.4s, v24.4s  \n"                    /* mul + add*/            \
  "subs %w[cnt], %w[cnt], #1  \n"                    /* sub main loop count */ \
  "fmla v0.4s, v9.4s, v11.4s  \n"                    /* mul + add*/            \
  "fmla v1.4s, v9.4s, v13.4s  \n"                    /* mul + add*/            \
  "fmla v2.4s, v9.4s, v15.4s  \n"                    /* mul + add*/            \
  "fmla v3.4s, v9.4s, v17.4s  \n"                    /* mul + add*/            \
  "fmla v4.4s, v9.4s, v19.4s  \n"                    /* mul + add*/            \
  "fmla v5.4s, v9.4s, v21.4s  \n"                    /* mul + add*/            \
  "fmla v6.4s, v9.4s, v23.4s  \n"                    /* mul + add*/            \
  "fmla v7.4s, v9.4s, v25.4s  \n"                    /* mul + add*/            \
  "bne 1b                     \n" /* jump to main loop */ /* pair add to final \
                                                             result */         \
  "2:                         \n"  /* reduce to scale */                       \
  "faddp  v16.4s, v0.4s, v0.4s\n"  /* pair add to vector */                    \
  "faddp  s8, v16.2s          \n"  /* pair add to scale */                     \
  "faddp  v17.4s, v1.4s, v1.4s\n"  /* pair add to vector */                    \
  "faddp  s9, v17.2s          \n"  /* pair add to scale */                     \
  "faddp  v18.4s, v2.4s, v2.4s\n"  /* pair add to vector */                    \
  "faddp  s10, v18.2s         \n"  /* pair add to scale */                     \
  "faddp  v19.4s, v3.4s, v3.4s\n"  /* pair add to vector */                    \
  "faddp  s11, v19.2s         \n"  /* pair add to scale */                     \
  "faddp  v20.4s, v4.4s, v4.4s\n"  /* pair add to vector */                    \
  "faddp  s12, v20.2s         \n"  /* pair add to scale */                     \
  "faddp  v21.4s, v5.4s, v5.4s\n"  /* pair add to vector */                    \
  "faddp  s13, v21.2s         \n"  /* pair add to scale */                     \
  "faddp  v22.4s, v6.4s, v6.4s\n"  /* pair add to vector */                    \
  "faddp  s14, v22.2s          \n" /* pair add to scale */                     \
  "faddp  v23.4s, v7.4s, v7.4s\n"  /* pair add to vector */                    \
  "faddp  s15, v23.2s          \n" /* pair add to scale */                     \
  "cmp %w[tail], #1           \n"  /* check whether has tail */                \
  "blt  4f                    \n"  /* jump to end */                           \
  "3:                         \n"  /* tail loop */                             \
  "ldr     s16, [%[in]], #4   \n"  /* load in, 1 float */                      \
  "ldr     s17, [%[w0]], #4   \n"  /* load w0, 1 float */                      \
  "ldr     s18, [%[w1]], #4   \n"  /* load w1, 1 float */                      \
  "ldr     s19, [%[w2]], #4   \n"  /* load w2, 1 float */                      \
  "ldr     s20, [%[w3]], #4   \n"  /* load w3, 1 float */                      \
  "ldr     s21, [%[w4]], #4   \n"  /* load w4, 1 float */                      \
  "ldr     s22, [%[w5]], #4   \n"  /* load w5, 1 float */                      \
  "ldr     s23, [%[w6]], #4   \n"  /* load w6, 1 float */                      \
  "ldr     s24, [%[w7]], #4   \n"  /* load w7, 1 float */                      \
  "fmadd   s8, s16, s17, s8   \n"  /* mul + add */                             \
  "fmadd   s9, s16, s18, s9   \n"  /* mul + add */                             \
  "fmadd   s10, s16, s19, s10 \n"  /* mul + add */                             \
  "fmadd   s11, s16, s20, s11 \n"  /* mul + add */                             \
  "fmadd   s12, s16, s21, s12 \n"  /* mul + add */                             \
  "fmadd   s13, s16, s22, s13 \n"  /* mul + add */                             \
  "fmadd   s14, s16, s23, s14 \n"  /* mul + add */                             \
  "fmadd   s15, s16, s24, s15 \n"  /* mul + add */                             \
  "subs %w[tail], %w[tail], #1\n"  /* sub tail loop count */                   \
  "bne 3b                     \n"  /* jump to tail loop */

#define SGEMV_KERNEL_1                                                         \
  /* check main loop */                                                        \
  "cmp %w[cnt], #1            \n" /* check whether has main loop */            \
  "blt  2f                    \n" /* jump to tail */ /* main loop */           \
  "1:                         \n"                    /* main loop */           \
  "ldp q8, q9, [%[in]], #32   \n"                    /* load input 8 float */  \
  "ldp q10, q11, [%[w0]], #32 \n"                    /* load w0 8 float */     \
  "fmla v0.4s, v8.4s, v10.4s  \n"                    /* mul + add*/            \
  "subs %w[cnt], %w[cnt], #1  \n"                    /* sub main loop count */ \
  "fmla v1.4s, v9.4s, v11.4s  \n"                    /* mul + add*/            \
  "bne 1b                     \n" /* jump to main loop */ /* pair add to final \
                                                             result */         \
  "2:                         \n" /* reduce to scale */                        \
  "fadd   v9.4s, v0.4s, v1.4s \n" /* add 2 vector */                           \
  "faddp  v10.4s, v9.4s, v9.4s\n" /* pair add to vector */                     \
  "faddp  s8, v10.2s          \n" /* pair add to scale */ /* check tails */    \
  "cmp %w[tail], #1           \n" /* check whether has tail */                 \
  "blt  4f                    \n" /* jump to end */                            \
  "3:                         \n" /* tail loop */                              \
  "ldr     s16, [%[in]], #4   \n" /* load in, 1 float */                       \
  "ldr     s17, [%[w0]], #4   \n" /* load w0, 1 float */                       \
  "fmadd   s8, s16, s17, s8   \n" /* mul + add */                              \
  "subs %w[tail], %w[tail], #1\n" /* sub tail loop count */                    \
  "bne 3b                     \n" /* jump to tail loop */

#define SGEMV_OUT_8                                 \
  /* end */                                         \
  "4:                         \n" /* end */         \
  "stp s8, s9, [%[out]]       \n" /* save result */ \
  "stp s10, s11, [%[out], #8] \n" /* save result */ \
  "stp s12, s13, [%[out], #16]\n" /* save result */ \
  "stp s14, s15, [%[out], #24]\n" /* save result */

#define SGEMV_OUT_8_RELU                                   \
  /* end */                                                \
  "4:                         \n" /* end */                \
  "movi   d0, #0              \n" /* zero data for relu */ \
  "fmax   s8, s8, s0          \n" /* relu */               \
  "fmax   s9, s9, s0          \n" /* relu */               \
  "fmax   s10, s10, s0        \n" /* relu */               \
  "fmax   s11, s11, s0        \n" /* relu */               \
  "fmax   s12, s12, s0        \n" /* relu */               \
  "fmax   s13, s13, s0        \n" /* relu */               \
  "fmax   s14, s14, s0        \n" /* relu */               \
  "fmax   s15, s15, s0        \n" /* relu */               \
  "stp s8, s9, [%[out]]       \n" /* save result */        \
  "stp s10, s11, [%[out], #8] \n" /* save result */        \
  "stp s12, s13, [%[out], #16]\n" /* save result */        \
  "stp s14, s15, [%[out], #24]\n" /* save result */

#define SGEMV_OUT_1                         \
  /* end */                                 \
  "4:                         \n" /* end */ \
  "str s8, [%[out]]           \n" /* save result */

#define SGEMV_OUT_1_RELU                                   \
  /* end */                                                \
  "4:                         \n" /* end */                \
  "movi   d0, #0              \n" /* zero data for relu */ \
  "fmax   s8, s8, s0          \n" /* relu */               \
  "str s8, [%[out]]           \n" /* save result */

#else  //__aarch64__

#define SGEMV_IN_4                                                    \
  "pld [%[in]]                    @ preload cache line, input\n"      \
  "pld [%[w0]]                    @ preload cache line, weights r0\n" \
  "pld [%[w1]]                    @ preload cache line, weights r1\n" \
  "pld [%[w2]]                    @ preload cache line, weights r2\n" \
  "pld [%[w3]]                    @ preload cache line, weights r3\n" \
  "vmov.u32 q0, #0                @ set q0 to 0\n"                    \
  "vmov.u32 q1, #0                @ set q1 to 0\n"                    \
  "vmov.u32 q2, #0                @ set q2 to 0\n"                    \
  "vmov.u32 q3, #0                @ set q3 to 0\n"                    \
  "pld [%[w0], #64]               @ preload cache line, weights r0\n" \
  "pld [%[w1], #64]               @ preload cache line, weights r1\n" \
  "pld [%[w2], #64]               @ preload cache line, weights r2\n" \
  "pld [%[w3], #64]               @ preload cache line, weights r3\n"

#define SGEMV_IN_4_BIAS                                               \
  "pld [%[in]]                    @ preload cache line, input\n"      \
  "pld [%[w0]]                    @ preload cache line, weights r0\n" \
  "pld [%[w1]]                    @ preload cache line, weights r1\n" \
  "pld [%[w2]]                    @ preload cache line, weights r2\n" \
  "pld [%[w3]]                    @ preload cache line, weights r3\n" \
  "vmov.u32 q0, #0                @ set q0 to 0\n"                    \
  "vmov.u32 q1, #0                @ set q1 to 0\n"                    \
  "vmov.u32 q2, #0                @ set q2 to 0\n"                    \
  "vmov.u32 q3, #0                @ set q3 to 0\n"                    \
  "vmov s0, %[bias0]              @ set q0 to bias0\n"                \
  "vmov s4, %[bias1]              @ set q1 to bias1\n"                \
  "vmov s8, %[bias2]              @ set q2 to bias2\n"                \
  "vmov s12,%[bias3]              @ set q3 to bias3\n"                \
  "pld [%[w0], #64]               @ preload cache line, weights r0\n" \
  "pld [%[w1], #64]               @ preload cache line, weights r1\n" \
  "pld [%[w2], #64]               @ preload cache line, weights r2\n" \
  "pld [%[w3], #64]               @ preload cache line, weights r3\n"

#define SGEMV_IN_1                                                        \
  "pld [%[in]]                        @ preload cache line, input\n"      \
  "pld [%[w0]]                        @ preload cache line, weights r0\n" \
  "vmov.u32 q0, #0                    @ set q0 to 0\n"

#define SGEMV_IN_1_BIAS                                                   \
  "pld [%[in]]                        @ preload cache line, input\n"      \
  "pld [%[w0]]                        @ preload cache line, weights r0\n" \
  "vmov.u32 q0, #0                    @ set q0 to 0\n"                    \
  "vmov s0, %[bias0]                  @ set q0 to 0\n"

#define SGEMV_KERNEL_4                                                         \
  /* check main loop */                                                        \
  "cmp %[cnt], #1                 @ check whether has main loop\n"             \
  "blt  2f                        @ jump to tail\n"                            \
  "1:                             @ main loop\n"                               \
  "vld1.32 {d8-d11}, [%[in]]!     @ load input, q4, q5\n"                      \
  "vld1.32 {d12-d15}, [%[w0]]!    @ load weights r0, q6,q7\n"                  \
  "vld1.32 {d16-d19}, [%[w1]]!    @ load weights r1, q8,q9\n"                  \
  "vld1.32 {d20-d23}, [%[w2]]!    @ load weights r2, q10,q11\n"                \
  "vld1.32 {d24-d27}, [%[w3]]!    @ load weights r3, q12,q13\n"                \
  "vmla.f32 q0, q4, q6            @ mul add\n"                                 \
  "vmla.f32 q1, q4, q8            @ mul add\n"                                 \
  "vmla.f32 q2, q4, q10           @ mul add\n"                                 \
  "vmla.f32 q3, q4, q12           @ mul add\n"                                 \
  "subs %[cnt], #1                @ sub loop count \n"                         \
  "vmla.f32 q0, q5, q7            @ mul add\n"                                 \
  "vmla.f32 q1, q5, q9            @ mul add\n"                                 \
  "vmla.f32 q2, q5, q11           @ mul add\n"                                 \
  "vmla.f32 q3, q5, q13           @ mul add\n"                                 \
  "bne 1b                         @ jump to main loop\n" /* pair add to final  \
                                                            result */          \
  "2:                             @ pair add \n"                               \
  "vpadd.f32 d8, d0, d1           @ pair add, first step\n"                    \
  "vpadd.f32 d9, d2, d3           @ pair add, first step\n"                    \
  "vpadd.f32 d10, d4, d5          @ pair add, first step\n"                    \
  "vpadd.f32 d11, d6, d7          @ pair add, first step\n"                    \
  "vpadd.f32 d0, d8, d9           @ pair add, second step\n"                   \
  "vpadd.f32 d1, d10, d11         @ pair add, second step\n" /* check tails */ \
  "cmp %[tail], #1                @ check whether has tail\n"                  \
  "blt  4f                        @ jump to end\n"                             \
  "3:                             @ tail loop\n"                               \
  "vldm     %[in]!, {s16}         @ load 1 float\n"                            \
  "vldm     %[w0]!, {s17}         @ load 1 float\n"                            \
  "vldm     %[w1]!, {s18}         @ load 1 float\n"                            \
  "vldm     %[w2]!, {s19}         @ load 1 float\n"                            \
  "vldm     %[w3]!, {s20}         @ load 1 float\n"                            \
  "vmla.f32   s0, s16, s17        @ mul + add\n"                               \
  "vmla.f32   s1, s16, s18        @ mul + add\n"                               \
  "vmla.f32   s2, s16, s19        @ mul + add\n"                               \
  "vmla.f32   s3, s16, s20        @ mul + add\n"                               \
  "subs %[tail], #1               @ sub loop count \n"                         \
  "bne 3b                         @ jump to tail loop\n"

#define SGEMV_KERNEL_1                                                         \
  "cmp %[cnt], #1                     @ check whether has main loop\n"         \
  "blt  2f                            @ jump to tail\n"                        \
  "1:                                 @ main loop\n"                           \
  "vld1.32 {d24-d27}, [%[in]]!        @ load input, q12,q13\n"                 \
  "vld1.32 {d28-d31}, [%[w0]]!        @ load weights r0, q14, q15\n"           \
  "vmla.f32 q0, q12, q14              @ mul add\n"                             \
  "vmla.f32 q0, q13, q15              @ mul add\n"                             \
  "subs %[cnt] , #1                   @ sub loop count \n"                     \
  "bne 1b                             @ jump to main loop\n" /* pair add to    \
                                                                final result   \
                                                                */             \
  "2:                                 @ end processing\n"                      \
  "vpadd.f32 d2, d0, d1               @ pair add, first step\n"                \
  "vpadd.f32 d0, d2, d2               @ pair add, final step\n" /* check tails \
                                                                   */          \
  "cmp %[tail], #1                    @ check whether has mid cols\n"          \
  "blt  4f                            @ jump to end\n"                         \
  "3:                                 @ tail loop\n"                           \
  "vldm     %[in]!, {s16}             @ load 1 float\n"                        \
  "vldm     %[w0]!, {s17}             @ load 1 float\n"                        \
  "vmla.f32   s0, s16, s17            @ mul + add\n"                           \
  "subs %[tail], #1                   @ sub loop count \n"                     \
  "bne 3b                             @ jump to tail loop\n"

#define SGEMV_OUT_4                        \
  /* end */                                \
  "4:                             @ end\n" \
  "vst1.32 {d0-d1}, [%[out]]      @ save result\n"

#define SGEMV_OUT_4_RELU                             \
  /* end */                                          \
  "4:                             @ end\n"           \
  "vmov.i32   q1, #0              @ zero for relu\n" \
  "vmax.f32   q0, q0, q1          @ relu\n"          \
  "vst1.32 {d0-d1}, [%[out]]      @ save result\n"

#define SGEMV_OUT_1                        \
  /* end */                                \
  "4:                             @ end\n" \
  "vst1.32 {d0[0]}, [%[out]]      @ save result\n"

#define SGEMV_OUT_1_RELU                             \
  /* end */                                          \
  "4:                             @ end\n"           \
  "vmov.i32   d1, #0              @ zero for relu\n" \
  "vmax.f32   d0, d0, d1          @ relu\n"          \
  "vst1.32 {d0[0]}, [%[out]]      @ save result\n"
#endif

void sgemv(const bool transA, const int M, const int N, const float *A,
           const float *x, float *y) {
  float *data_out = y;
  const float *data_in = x;
  const float *weights_ptr = A;

  int cnt = N >> 3;
  int tail = N & 7;

#ifdef __aarch64__
  int out_cnt = M >> 3;

#pragma omp parallel for
  for (int j = 0; j < out_cnt; j++) {
    int out_idx = j * 8;
    float *ptr_out = data_out + out_idx;
    const float *ptr_in = data_in;
    const float *ptr_w0 = weights_ptr + (N * out_idx);
    const float *ptr_w1 = ptr_w0 + N;
    const float *ptr_w2 = ptr_w1 + N;
    const float *ptr_w3 = ptr_w2 + N;
    const float *ptr_w4 = ptr_w3 + N;
    const float *ptr_w5 = ptr_w4 + N;
    const float *ptr_w6 = ptr_w5 + N;
    const float *ptr_w7 = ptr_w6 + N;
    int cnt_loop = cnt;
    int tail_loop = tail;
    asm volatile(SGEMV_IN_8 SGEMV_KERNEL_8 SGEMV_OUT_8
                 : [in] "+r"(ptr_in), [w0] "+r"(ptr_w0), [w1] "+r"(ptr_w1),
                   [w2] "+r"(ptr_w2), [w3] "+r"(ptr_w3), [w4] "+r"(ptr_w4),
                   [w5] "+r"(ptr_w5), [w6] "+r"(ptr_w6), [w7] "+r"(ptr_w7),
                   [cnt] "+r"(cnt_loop), [tail] "+r"(tail_loop)
                 : [out] "r"(ptr_out)
                 : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9",
                   "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17",
                   "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25", "cc",
                   "memory");
  }
//! deal with remains
#pragma omp parallel for
  for (int j = out_cnt * 8; j < M; ++j) {
    float *ptr_out = data_out + j;
    const float *ptr_in = data_in;
    const float *ptr_w0 = weights_ptr + (N * j);
    int cnt_loop = cnt;
    int tail_loop = tail;
    float tmp[4];
    float tmp1[4];
    float tmp2[4];
    float tmp3[4];
    float tmp4[4];
    asm volatile(SGEMV_IN_1 SGEMV_KERNEL_1 SGEMV_OUT_1
                 : [in] "+r"(ptr_in), [w0] "+r"(ptr_w0), [cnt] "+r"(cnt_loop),
                   [tail] "+r"(tail_loop)
                 : [out] "r"(ptr_out), [tmp] "r"(tmp), [tmp1] "r"(tmp1),
                   [tmp2] "r"(tmp2), [tmp3] "r"(tmp3), [tmp4] "r"(tmp4)
                 : "v0", "v1", "v8", "v9", "v10", "v11", "v16", "v17", "cc",
                   "memory");
  }
#else  //__aarch64__
  int out_cnt = M >> 2;
#pragma omp parallel for
  for (int j = 0; j < out_cnt; j++) {
    int out_idx = j * 4;
    float *ptr_out = data_out + out_idx;
    const float *ptr_in = data_in;
    const float *ptr_w0 = weights_ptr + (N * out_idx);
    const float *ptr_w1 = ptr_w0 + N;
    const float *ptr_w2 = ptr_w1 + N;
    const float *ptr_w3 = ptr_w2 + N;

    int cnt_loop = cnt;
    int tail_loop = tail;
    asm volatile(SGEMV_IN_4 SGEMV_KERNEL_4 SGEMV_OUT_4
                 : [in] "+r"(ptr_in), [w0] "+r"(ptr_w0), [w1] "+r"(ptr_w1),
                   [w2] "+r"(ptr_w2), [w3] "+r"(ptr_w3), [cnt] "+r"(cnt_loop),
                   [tail] "+r"(tail_loop)
                 : [out] "r"(ptr_out)
                 : "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7", "q8", "q9",
                   "q10", "q11", "q12", "q13", "cc", "memory");
  }
//! deal with remains
#pragma omp parallel for
  for (int j = out_cnt * 4; j < M; ++j) {
    float *ptr_out = data_out + j;
    const float *ptr_in = data_in;
    const float *ptr_w0 = weights_ptr + (N * j);
    int cnt_loop = cnt;
    int tail_loop = tail;
    asm volatile(SGEMV_IN_1 SGEMV_KERNEL_1 SGEMV_OUT_1
                 : [in] "+r"(ptr_in), [w0] "+r"(ptr_w0), [cnt] "+r"(cnt_loop),
                   [tail] "+r"(tail_loop)
                 : [out] "r"(ptr_out)
                 : "q0", "q1", "q12", "q13", "q14", "q15", "cc", "memory");
  }
#endif  //__aarch64__
}

void sgemv_relu(const bool transA, const int M, const int N, const float *A,
                const float *x, float *y) {
  float *data_out = y;
  const float *data_in = x;
  const float *weights_ptr = A;

  int cnt = N >> 3;
  int tail = N & 7;

#ifdef __aarch64__
  int out_cnt = M >> 3;
#pragma omp parallel for
  for (int j = 0; j < out_cnt; j++) {
    int out_idx = j * 8;
    float *ptr_out = data_out + out_idx;
    const float *ptr_in = data_in;
    const float *ptr_w0 = weights_ptr + (N * out_idx);
    const float *ptr_w1 = ptr_w0 + N;
    const float *ptr_w2 = ptr_w1 + N;
    const float *ptr_w3 = ptr_w2 + N;
    const float *ptr_w4 = ptr_w3 + N;
    const float *ptr_w5 = ptr_w4 + N;
    const float *ptr_w6 = ptr_w5 + N;
    const float *ptr_w7 = ptr_w6 + N;
    int cnt_loop = cnt;
    int tail_loop = tail;
    asm volatile(SGEMV_IN_8 SGEMV_KERNEL_8 SGEMV_OUT_8_RELU
                 : [in] "+r"(ptr_in), [w0] "+r"(ptr_w0), [w1] "+r"(ptr_w1),
                   [w2] "+r"(ptr_w2), [w3] "+r"(ptr_w3), [w4] "+r"(ptr_w4),
                   [w5] "+r"(ptr_w5), [w6] "+r"(ptr_w6), [w7] "+r"(ptr_w7),
                   [cnt] "+r"(cnt_loop), [tail] "+r"(tail_loop)
                 : [out] "r"(ptr_out)
                 : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9",
                   "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17",
                   "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25", "cc",
                   "memory");
  }
//! deal with remains
#pragma omp parallel for
  for (int j = out_cnt * 8; j < M; ++j) {
    float *ptr_out = data_out + j;
    const float *ptr_in = data_in;
    const float *ptr_w0 = weights_ptr + (N * j);
    int cnt_loop = cnt;
    int tail_loop = tail;
    asm volatile(SGEMV_IN_1 SGEMV_KERNEL_1 SGEMV_OUT_1_RELU
                 : [in] "+r"(ptr_in), [w0] "+r"(ptr_w0), [cnt] "+r"(cnt_loop),
                   [tail] "+r"(tail_loop)
                 : [out] "r"(ptr_out)
                 : "v0", "v1", "v8", "v9", "v10", "v11", "v16", "v17", "cc",
                   "memory");
  }
#else  //__aarch64__
  int out_cnt = M >> 2;
#pragma omp parallel for
  for (int j = 0; j < out_cnt; j++) {
    int out_idx = j * 4;
    float *ptr_out = data_out + out_idx;
    const float *ptr_in = data_in;
    const float *ptr_w0 = weights_ptr + (N * out_idx);
    const float *ptr_w1 = ptr_w0 + N;
    const float *ptr_w2 = ptr_w1 + N;
    const float *ptr_w3 = ptr_w2 + N;

    int cnt_loop = cnt;
    int tail_loop = tail;
    asm volatile(SGEMV_IN_4 SGEMV_KERNEL_4 SGEMV_OUT_4_RELU
                 : [in] "+r"(ptr_in), [w0] "+r"(ptr_w0), [w1] "+r"(ptr_w1),
                   [w2] "+r"(ptr_w2), [w3] "+r"(ptr_w3), [cnt] "+r"(cnt_loop),
                   [tail] "+r"(tail_loop)
                 : [out] "r"(ptr_out)
                 : "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7", "q8", "q9",
                   "q10", "q11", "q12", "q13", "cc", "memory");
  }
//! deal with remains
#pragma omp parallel for
  for (int j = out_cnt * 4; j < M; ++j) {
    float *ptr_out = data_out + j;
    const float *ptr_in = data_in;
    const float *ptr_w0 = weights_ptr + (N * j);
    int cnt_loop = cnt;
    int tail_loop = tail;
    asm volatile(SGEMV_IN_1 SGEMV_KERNEL_1 SGEMV_OUT_1_RELU
                 : [in] "+r"(ptr_in), [w0] "+r"(ptr_w0), [cnt] "+r"(cnt_loop),
                   [tail] "+r"(tail_loop)
                 : [out] "r"(ptr_out)
                 : "q0", "q1", "q12", "q13", "q14", "q15", "cc", "memory");
  }
#endif  //__aarch64__
}

void sgemv_bias(const bool transA, const int M, const int N, const float *A,
                const float *x, float *y, const float *bias) {
  float *data_out = y;
  const float *data_in = x;
  const float *weights_ptr = A;

  int cnt = N >> 3;
  int tail = N & 7;

#ifdef __aarch64__
  int out_cnt = M >> 3;
#pragma omp parallel for
  for (int j = 0; j < out_cnt; j++) {
    int out_idx = j * 8;
    float *ptr_out = data_out + out_idx;
    const float *ptr_in = data_in;
    const float *ptr_w0 = weights_ptr + (N * out_idx);
    const float *ptr_w1 = ptr_w0 + N;
    const float *ptr_w2 = ptr_w1 + N;
    const float *ptr_w3 = ptr_w2 + N;
    const float *ptr_w4 = ptr_w3 + N;
    const float *ptr_w5 = ptr_w4 + N;
    const float *ptr_w6 = ptr_w5 + N;
    const float *ptr_w7 = ptr_w6 + N;
    const float *bias_ptr = bias + out_idx;
    int cnt_loop = cnt;
    int tail_loop = tail;
    asm volatile(SGEMV_IN_8_BIAS SGEMV_KERNEL_8 SGEMV_OUT_8
                 : [in] "+r"(ptr_in), [w0] "+r"(ptr_w0), [w1] "+r"(ptr_w1),
                   [w2] "+r"(ptr_w2), [w3] "+r"(ptr_w3), [w4] "+r"(ptr_w4),
                   [w5] "+r"(ptr_w5), [w6] "+r"(ptr_w6), [w7] "+r"(ptr_w7),
                   [cnt] "+r"(cnt_loop), [tail] "+r"(tail_loop)
                 : [out] "r"(ptr_out), [bias_ptr] "r"(bias_ptr)
                 : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9",
                   "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17",
                   "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25", "cc",
                   "memory");
  }
//! deal with remains
#pragma omp parallel for
  for (int j = out_cnt * 8; j < M; ++j) {
    float *ptr_out = data_out + j;
    const float *ptr_in = data_in;
    const float *ptr_w0 = weights_ptr + (N * j);
    int cnt_loop = cnt;
    int tail_loop = tail;
    float bias0 = bias[j];
    asm volatile(SGEMV_IN_1_BIAS SGEMV_KERNEL_1 SGEMV_OUT_1
                 : [in] "+r"(ptr_in), [w0] "+r"(ptr_w0), [cnt] "+r"(cnt_loop),
                   [tail] "+r"(tail_loop)
                 : [out] "r"(ptr_out), [bias0] "r"(bias0)
                 : "v0", "v1", "v8", "v9", "v10", "v11", "v16", "v17", "cc",
                   "memory");
  }
#else  //__aarch64__
  int out_cnt = M >> 2;
#pragma omp parallel for
  for (int j = 0; j < out_cnt; j++) {
    int out_idx = j * 4;
    float *ptr_out = data_out + out_idx;
    const float *ptr_in = data_in;
    const float *ptr_w0 = weights_ptr + (N * out_idx);
    const float *ptr_w1 = ptr_w0 + N;
    const float *ptr_w2 = ptr_w1 + N;
    const float *ptr_w3 = ptr_w2 + N;
    float bias0 = bias[out_idx];
    float bias1 = bias[out_idx + 1];
    float bias2 = bias[out_idx + 2];
    float bias3 = bias[out_idx + 3];

    int cnt_loop = cnt;
    int tail_loop = tail;
    asm volatile(SGEMV_IN_4_BIAS SGEMV_KERNEL_4 SGEMV_OUT_4
                 : [in] "+r"(ptr_in), [w0] "+r"(ptr_w0), [w1] "+r"(ptr_w1),
                   [w2] "+r"(ptr_w2), [w3] "+r"(ptr_w3), [cnt] "+r"(cnt_loop),
                   [tail] "+r"(tail_loop)
                 : [out] "r"(ptr_out), [bias0] "r"(bias0), [bias1] "r"(bias1),
                   [bias2] "r"(bias2), [bias3] "r"(bias3)
                 : "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7", "q8", "q9",
                   "q10", "q11", "q12", "q13", "cc", "memory");
  }
//! deal with remains
#pragma omp parallel for
  for (int j = out_cnt * 4; j < M; ++j) {
    float *ptr_out = data_out + j;
    const float *ptr_in = data_in;
    const float *ptr_w0 = weights_ptr + (N * j);
    int cnt_loop = cnt;
    int tail_loop = tail;
    float bias0 = bias[j];
    asm volatile(SGEMV_IN_1_BIAS SGEMV_KERNEL_1 SGEMV_OUT_1
                 : [in] "+r"(ptr_in), [w0] "+r"(ptr_w0), [cnt] "+r"(cnt_loop),
                   [tail] "+r"(tail_loop)
                 : [out] "r"(ptr_out), [bias0] "r"(bias0)
                 : "q0", "q1", "q12", "q13", "q14", "q15", "cc", "memory");
  }
#endif  //__aarch64__
}

void sgemv_bias_relu(const bool transA, const int M, const int N,
                     const float *A, const float *x, float *y,
                     const float *bias) {
  float *data_out = y;
  const float *data_in = x;
  const float *weights_ptr = A;
  int cnt = N >> 3;
  int tail = N & 7;
#ifdef __aarch64__
  int out_cnt = M >> 3;
#pragma omp parallel for
  for (int j = 0; j < out_cnt; j++) {
    int out_idx = j * 8;
    float *ptr_out = data_out + out_idx;
    const float *ptr_in = data_in;
    const float *ptr_w0 = weights_ptr + (N * out_idx);
    const float *ptr_w1 = ptr_w0 + N;
    const float *ptr_w2 = ptr_w1 + N;
    const float *ptr_w3 = ptr_w2 + N;
    const float *ptr_w4 = ptr_w3 + N;
    const float *ptr_w5 = ptr_w4 + N;
    const float *ptr_w6 = ptr_w5 + N;
    const float *ptr_w7 = ptr_w6 + N;
    const float *bias_ptr = bias + out_idx;
    int cnt_loop = cnt;
    int tail_loop = tail;
    asm volatile(SGEMV_IN_8_BIAS SGEMV_KERNEL_8 SGEMV_OUT_8_RELU
                 : [in] "+r"(ptr_in), [w0] "+r"(ptr_w0), [w1] "+r"(ptr_w1),
                   [w2] "+r"(ptr_w2), [w3] "+r"(ptr_w3), [w4] "+r"(ptr_w4),
                   [w5] "+r"(ptr_w5), [w6] "+r"(ptr_w6), [w7] "+r"(ptr_w7),
                   [cnt] "+r"(cnt_loop), [tail] "+r"(tail_loop)
                 : [out] "r"(ptr_out), [bias_ptr] "r"(bias_ptr)
                 : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9",
                   "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17",
                   "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25", "cc",
                   "memory");
  }
//! deal with remains
#pragma omp parallel for
  for (int j = out_cnt * 8; j < M; ++j) {
    float *ptr_out = data_out + j;
    const float *ptr_in = data_in;
    const float *ptr_w0 = weights_ptr + (N * j);
    int cnt_loop = cnt;
    int tail_loop = tail;
    float bias0 = bias[j];
    asm volatile(SGEMV_IN_1_BIAS SGEMV_KERNEL_1 SGEMV_OUT_1_RELU
                 : [in] "+r"(ptr_in), [w0] "+r"(ptr_w0), [cnt] "+r"(cnt_loop),
                   [tail] "+r"(tail_loop)
                 : [out] "r"(ptr_out), [bias0] "r"(bias0)
                 : "v0", "v1", "v8", "v9", "v10", "v11", "v16", "v17", "cc",
                   "memory");
  }
#else  //__aarch64__
  int out_cnt = M >> 2;
#pragma omp parallel for
  for (int j = 0; j < out_cnt; j++) {
    int out_idx = j * 4;
    float *ptr_out = data_out + out_idx;
    const float *ptr_in = data_in;
    const float *ptr_w0 = weights_ptr + (N * out_idx);
    const float *ptr_w1 = ptr_w0 + N;
    const float *ptr_w2 = ptr_w1 + N;
    const float *ptr_w3 = ptr_w2 + N;
    float bias0 = bias[out_idx];
    float bias1 = bias[out_idx + 1];
    float bias2 = bias[out_idx + 2];
    float bias3 = bias[out_idx + 3];

    int cnt_loop = cnt;
    int tail_loop = tail;
    asm volatile(SGEMV_IN_4_BIAS SGEMV_KERNEL_4 SGEMV_OUT_4_RELU
                 : [in] "+r"(ptr_in), [w0] "+r"(ptr_w0), [w1] "+r"(ptr_w1),
                   [w2] "+r"(ptr_w2), [w3] "+r"(ptr_w3), [cnt] "+r"(cnt_loop),
                   [tail] "+r"(tail_loop)
                 : [out] "r"(ptr_out), [bias0] "r"(bias0), [bias1] "r"(bias1),
                   [bias2] "r"(bias2), [bias3] "r"(bias3)
                 : "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7", "q8", "q9",
                   "q10", "q11", "q12", "q13", "cc", "memory");
  }
//! deal with remains
#pragma omp parallel for
  for (int j = out_cnt * 4; j < M; ++j) {
    float *ptr_out = data_out + j;
    const float *ptr_in = data_in;
    const float *ptr_w0 = weights_ptr + (N * j);
    int cnt_loop = cnt;
    int tail_loop = tail;
    float bias0 = bias[j];
    asm volatile(SGEMV_IN_1_BIAS SGEMV_KERNEL_1 SGEMV_OUT_1_RELU
                 : [in] "+r"(ptr_in), [w0] "+r"(ptr_w0), [cnt] "+r"(cnt_loop),
                   [tail] "+r"(tail_loop)
                 : [out] "r"(ptr_out), [bias0] "r"(bias0)
                 : "q0", "q1", "q12", "q13", "q14", "q15", "cc", "memory");
  }
#endif  //__aarch64__
}

}  // namespace math
}  // namespace arm
}  // namespace lite
}  // namespace paddle
