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

#include "paddle/fluid/lite/arm/math/gemm_prepacked_int8.h"
#include <arm_neon.h>

namespace paddle {
namespace lite {
namespace arm {
namespace math {

void prepackA_m4k2x2_int8(int8_t* out, const int8_t* in, const int ldin,
                          const int m0, const int mmax, const int k0,
                          const int kmax);
void prepackA_m4k2x2_trans_int8(int8_t* out, const int8_t* in, const int ldin,
                                const int m0, const int mmax, const int k0,
                                const int kmax);
void packb_int8(int8_t* out, const int8_t* in, const int ldin, const int k0,
                const int kmax, const int n0, const int nmax,
                const int8_t* zerobuf);
void packb_trans_int8(int8_t* out, const int8_t* in, const int ldin,
                      const int k0, const int kmax, const int n0,
                      const int nmax, const int8_t* zerobuf);

void prepackA_int8(void* out, const void* in, const int ldin, const int m0,
                   const int mmax, const int k0, const int kmax,
                   bool is_trans) {
  if (is_trans) {
    prepackA_m4k2x2_trans_int8(static_cast<int8_t*>(out),
                               static_cast<const int8_t*>(in), ldin, m0, mmax,
                               k0, kmax);
  } else {
    prepackA_m4k2x2_int8(static_cast<int8_t*>(out),
                         static_cast<const int8_t*>(in), ldin, m0, mmax, k0,
                         kmax);
  }
}
void prepackA_int8(TensorLite* tout, const TensorLite& tin, int m, int k,
                   int group, bool is_trans, ARMContext* ctx) {
  int hblock = get_hblock_int8(ctx->arch());
  int m_roundup = ROUNDUP(m, hblock);
  // round up to 128 bits
  int kup = ROUNDUP(k, KBLOCK_INT8);
  int group_size_round_up = ((m_roundup * kup + 15) / 16) * 16;

  if (tout->numel() < group_size_round_up * group) {
    tout->Resize({1, 1, 1, group_size_round_up * group});
  }
  int lda = k;
  if (is_trans) {
    lda = m;
  }
  for (int g = 0; g < group; ++g) {
    const char* weights_group = tin.data<char>() + g * m * k;
    char* weights_trans_ptr =
        tout->mutable_data<char>() + g * group_size_round_up;
    prepackA_int8(weights_trans_ptr, weights_group, lda, 0, m, 0, k, is_trans);
  }
}
template <typename Dtype>
inline void gemm_int8_kernel(const int8_t* a_ptr,
                             const int8_t*& b_ptr,                 // NOLINT
                             const int32_t* bias, Dtype*& c_ptr0,  // NOLINT
                             Dtype*& c_ptr1, Dtype*& c_ptr2,       // NOLINT
                             Dtype*& c_ptr3,                       // NOLINT
                             const float* scale, bool is_relu, int k, int rem);
#ifdef __aarch64__
#define GEMM_INT8_KERNEL                                                      \
  "ld1 {v0.16b}, [%[a_ptr]],#16\n"         /* load a to q0, q1 */             \
  "ld1 {v4.16b, v5.16b}, [%[b_ptr]],#32\n" /* load b to q4, q5 */             \
  "ld1 {v6.16b, v7.16b}, [%[b_ptr]],#32\n" /* load b to q6, q7 */             \
  "ldr    q8, [%[bias]]\n"                 /* load bias */                    \
  "ext    v9.16b, v8.16b, v8.16b, #4\n"    /* shift left 1s */                \
  "ext    v10.16b, v8.16b, v8.16b, #8\n"   /* shift left 2s */                \
  "ext    v11.16b, v8.16b, v8.16b, #12\n"  /* shift left 3s */                \
  "and v16.16b, v8.16b, v8.16b\n"          /* set bias0 to out00 */           \
  "and v17.16b, v9.16b, v9.16b\n"          /* set bias0 to out01 */           \
  "prfm   pldl1keep, [%[a_ptr], #64]\n"    /* preload a*/                     \
  "and v18.16b, v10.16b, v10.16b\n"        /* set bias0 to out02 */           \
  "and v19.16b, v11.16b, v11.16b\n"        /* set bias0 to out03 */           \
  "prfm   pldl1keep, [%[b_ptr], #64]\n"    /* preload b*/                     \
  "and v20.16b, v8.16b, v8.16b\n"          /* set bias0 to out10 */           \
  "and v21.16b, v9.16b, v9.16b\n"          /* set bias0 to out11 */           \
  "prfm   pldl1keep, [%[a_ptr], #128]\n"   /* preload a*/                     \
  "and v22.16b, v10.16b, v10.16b\n"        /* set bias0 to out12 */           \
  "and v23.16b, v11.16b, v11.16b\n"        /* set bias0 to out13 */           \
  "prfm   pldl1keep, [%[b_ptr], #128]\n"   /* preload b*/                     \
  "and v24.16b, v8.16b, v8.16b\n"          /* set bias0 to out20 */           \
  "and v25.16b, v9.16b, v9.16b\n"          /* set bias0 to out21 */           \
  "prfm   pldl1keep, [%[a_ptr], #192]\n"   /* preload a*/                     \
  "and v26.16b, v10.16b, v10.16b\n"        /* set bias0 to out22 */           \
  "and v27.16b, v11.16b, v11.16b\n"        /* set bias0 to out23 */           \
  "prfm   pldl1keep, [%[b_ptr], #192]\n"   /* preload b*/                     \
  "and v28.16b, v8.16b, v8.16b\n"          /* set bias0 to out30 */           \
  "and v29.16b, v9.16b, v9.16b\n"          /* set bias0 to out31 */           \
  "prfm   pldl1keep, [%[b_ptr], #256]\n"   /* preload b*/                     \
  "and v30.16b, v10.16b, v10.16b\n"        /* set bias0 to out32 */           \
  "and v31.16b, v11.16b, v11.16b\n"        /* set bias0 to out33 */           \
  "ext    v1.16b, v0.16b, v0.16b, #2\n"    /* shift left 2bytes */            \
  "ins    v1.h[3], v0.h[0]\n"              /* insert element */               \
  "ins    v1.h[7], v0.h[4]\n"              /* insert element */               \
  "rev64  v2.4s,  v0.4s\n" /* get low: 22,33,00,11; hi: 66,77,44,55 */        \
  "rev64  v3.4s,  v1.4s\n" /* get low: 33,00,11,22; hi: 77,44,55,66 */        \
  "prfm   pldl1keep, [%[b_ptr], #320]\n"                  /* preload a*/      \
  "prfm   pldl1keep, [%[b_ptr], #384]\n"                  /* preload b*/      \
  "cbz    %w[k],    3f\n" /* if k = 0, jump to remains */ /* 1st b0, b1 */    \
  "smull  v8.8h,   v0.8b, v4.8b\n"                        /* a0 * b0 = c00 */ \
  "smull  v12.8h,  v0.8b, v5.8b\n"                        /* a0 * b1 = c01 */ \
  "smull  v9.8h,   v1.8b, v4.8b\n"                        /* a1 * b0 = c10 */ \
  "smull  v13.8h,  v1.8b, v5.8b\n"                        /* a1 * b1 = c11 */ \
  "smull  v10.8h,  v2.8b, v4.8b\n"                        /* a2 * b0 = c20 */ \
  "smull  v14.8h,  v2.8b, v5.8b\n"                        /* a2 * b1 = c21 */ \
  "smull  v11.8h,  v3.8b, v4.8b\n"                        /* a3 * b0 = c30 */ \
  "smull  v15.8h,  v3.8b, v5.8b\n"                        /* a3 * b1 = c31 */ \
  "subs %w[k], %w[k], #1\n" /* loop count -1 */           /* 2nd b0, b1 */    \
  "smlal2  v8.8h,   v0.16b, v4.16b\n"                     /* a0 * b0 = c00 */ \
  "smlal2  v12.8h,  v0.16b, v5.16b\n"                     /* a0 * b1 = c01 */ \
  "smlal2  v9.8h,   v1.16b, v4.16b\n"                     /* a1 * b0 = c10 */ \
  "smlal2  v13.8h,  v1.16b, v5.16b\n"                     /* a1 * b1 = c11 */ \
  "smlal2  v10.8h,  v2.16b, v4.16b\n"                     /* a2 * b0 = c20 */ \
  "smlal2  v14.8h,  v2.16b, v5.16b\n"                     /* a2 * b1 = c21 */ \
  "smlal2  v11.8h,  v3.16b, v4.16b\n"                     /* a3 * b0 = c30 */ \
  "smlal2  v15.8h,  v3.16b, v5.16b\n"                     /* a3 * b1 = c31 */ \
  "beq    8f\n" /* skip main loop */                      /* main loop*/      \
  "0:\n"                                                  /* main loop */     \
  "ld1 {v4.16b, v5.16b}, [%[b_ptr]],#32\n" /* load b to q4, q5 */             \
  "sadalp  v16.4s, v8.8h\n"        /* pairwise accumulate to int32, out00 */  \
  "smull  v8.8h,   v0.8b, v6.8b\n" /* a0 * b2 = c02 */                        \
  "sadalp  v20.4s, v12.8h\n"       /* pairwise accumulate to int32, out01 */  \
  "smull  v12.8h,  v0.8b, v7.8b\n" /* a0 * b3 = c03 */                        \
  "sadalp  v17.4s, v9.8h\n"        /* pairwise accumulate to int32, out10 */  \
  "smull  v9.8h,   v1.8b, v6.8b\n" /* a1 * b2 = c12 */                        \
  "sadalp  v21.4s, v13.8h\n"       /* pairwise accumulate to int32, out11 */  \
  "smull  v13.8h,  v1.8b, v7.8b\n" /* a1 * b3 = c13 */                        \
  "sadalp  v18.4s, v10.8h\n"       /* pairwise accumulate to int32, out20 */  \
  "smull  v10.8h,  v2.8b, v6.8b\n" /* a2 * b2 = c22 */                        \
  "sadalp  v22.4s, v14.8h\n"       /* pairwise accumulate to int32, out21 */  \
  "smull  v14.8h,  v2.8b, v7.8b\n" /* a2 * b3 = c23 */                        \
  "sadalp  v19.4s, v11.8h\n"       /* pairwise accumulate to int32, out30 */  \
  "smlal2  v8.8h,   v0.16b, v6.16b\n" /* a0 * b2 = c02 */                     \
  "smlal2  v12.8h,  v0.16b, v7.16b\n" /* a0 * b3 = c03 */                     \
  "ld1 {v0.16b}, [%[a_ptr]],#16\n"    /* load a to q0, q1 */                  \
  "smull  v11.8h,  v3.8b, v6.8b\n"    /* a3 * b2 = c32 */                     \
  "sadalp  v23.4s, v15.8h\n" /* pairwise accumulate to int32, out31 */        \
  "smull  v15.8h,  v3.8b, v7.8b\n" /* a3 * b3 = c33 */ /* 2nd b2, b3 */       \
  "smlal2  v9.8h,   v1.16b, v6.16b\n"                  /* a1 * b2 = c12 */    \
  "smlal2  v13.8h,  v1.16b, v7.16b\n"                  /* a1 * b3 = c13 */    \
  "smlal2  v10.8h,  v2.16b, v6.16b\n"                  /* a2 * b2 = c22 */    \
  "ext    v1.16b, v0.16b, v0.16b, #2\n"                /* shift left 2bytes*/ \
  "ins    v1.h[3], v0.h[0]\n"                          /* insert element */   \
  "ins    v1.h[7], v0.h[4]\n"                          /* insert element */   \
  "smlal2  v14.8h,  v2.16b, v7.16b\n"                  /* a2 * b3 = c23 */    \
  "smlal2  v11.8h,  v3.16b, v6.16b\n"                  /* a3 * b2 = c32 */    \
  "smlal2  v15.8h,  v3.16b, v7.16b\n" /* a3 * b3 = c33 */ /* pre-process a */ \
  "rev64  v2.4s,  v0.4s\n" /* get low: 22,33,00,11; hi: 66,77,44,55 */        \
  "rev64  v3.4s,  v1.4s\n" /* get low: 33,00,11,22; hi: 77,44,55,66 */        \
  "ld1 {v6.16b, v7.16b}, [%[b_ptr]],#32\n" /* load b to q6, q7 */             \
  "sadalp  v24.4s, v8.8h\n"        /* pairwise accumulate to int32, out02 */  \
  "smull  v8.8h,   v0.8b, v4.8b\n" /* a0 * b0 = c00 */                        \
  "sadalp  v28.4s, v12.8h\n"       /* pairwise accumulate to int32, out03 */  \
  "smull  v12.8h,  v0.8b, v5.8b\n" /* a0 * b1 = c01 */                        \
  "sadalp  v25.4s, v9.8h\n"        /* pairwise accumulate to int32, out12 */  \
  "smull  v9.8h,   v1.8b, v4.8b\n" /* a1 * b0 = c00 */                        \
  "sadalp  v29.4s, v13.8h\n"       /* pairwise accumulate to int32, out13 */  \
  "smull  v13.8h,  v1.8b, v5.8b\n" /* a1 * b1 = c01 */                        \
  "sadalp  v26.4s, v10.8h\n"       /* pairwise accumulate to int32, out22 */  \
  "smull  v10.8h,  v2.8b, v4.8b\n" /* a2 * b0 = c00 */                        \
  "sadalp  v30.4s, v14.8h\n"       /* pairwise accumulate to int32, out23 */  \
  "smull  v14.8h,  v2.8b, v5.8b\n" /* a2 * b1 = c01 */                        \
  "sadalp  v27.4s, v11.8h\n"       /* pairwise accumulate to int32, out32 */  \
  "smull  v11.8h,  v3.8b, v4.8b\n" /* a3 * b0 = c00 */                        \
  "sadalp  v31.4s, v15.8h\n"       /* pairwise accumulate to int32, out33 */  \
  "smull  v15.8h,  v3.8b, v5.8b\n" /* a3 * b1 = c01 */                        \
  "subs %w[k], %w[k], #1\n" /* loop count -1 */ /* 2nd b0, b1 */              \
  "smlal2  v8.8h,   v0.16b, v4.16b\n"           /* a0 * b0 = c00 */           \
  "smlal2  v12.8h,  v0.16b, v5.16b\n"           /* a0 * b1 = c01 */           \
  "smlal2  v9.8h,   v1.16b, v4.16b\n"           /* a1 * b0 = c10 */           \
  "smlal2  v13.8h,  v1.16b, v5.16b\n"           /* a1 * b1 = c11 */           \
  "smlal2  v10.8h,  v2.16b, v4.16b\n"           /* a2 * b0 = c20 */           \
  "smlal2  v14.8h,  v2.16b, v5.16b\n"           /* a2 * b1 = c21 */           \
  "smlal2  v11.8h,  v3.16b, v4.16b\n"           /* a3 * b0 = c30 */           \
  "smlal2  v15.8h,  v3.16b, v5.16b\n"           /* a3 * b1 = c31 */           \
  "bgt 0b\n"                                    /* jump to main loop */       \
  "8:\n" /* finish main loop */                 /* 1st b2, b3 */              \
  "sadalp  v16.4s, v8.8h\n"        /* pairwise accumulate to int32, out00 */  \
  "smull  v8.8h,   v0.8b, v6.8b\n" /* a0 * b0 = c02 */                        \
  "sadalp  v20.4s, v12.8h\n"       /* pairwise accumulate to int32, out01 */  \
  "smull  v12.8h,  v0.8b, v7.8b\n" /* a0 * b1 = c03 */                        \
  "sadalp  v17.4s, v9.8h\n"        /* pairwise accumulate to int32, out10 */  \
  "smull  v9.8h,   v1.8b, v6.8b\n" /* a1 * b0 = c12 */                        \
  "sadalp  v21.4s, v13.8h\n"       /* pairwise accumulate to int32, out11 */  \
  "smull  v13.8h,  v1.8b, v7.8b\n" /* a1 * b1 = c13 */                        \
  "sadalp  v18.4s, v10.8h\n"       /* pairwise accumulate to int32, out20 */  \
  "smull  v10.8h,  v2.8b, v6.8b\n" /* a2 * b0 = c22 */                        \
  "sadalp  v22.4s, v14.8h\n"       /* pairwise accumulate to int32, out21 */  \
  "smull  v14.8h,  v2.8b, v7.8b\n" /* a2 * b1 = c23 */                        \
  "sadalp  v19.4s, v11.8h\n"       /* pairwise accumulate to int32, out30 */  \
  "smull  v11.8h,  v3.8b, v6.8b\n" /* a3 * b0 = c32 */                        \
  "sadalp  v23.4s, v15.8h\n"       /* pairwise accumulate to int32, out31 */  \
  "smull  v15.8h,  v3.8b, v7.8b\n" /* a3 * b1 = c33 */ /* 2nd b2, b3 */       \
  "smlal2  v8.8h,   v0.16b, v6.16b\n"                  /* a0 * b0 = c02 */    \
  "smlal2  v12.8h,  v0.16b, v7.16b\n"                  /* a0 * b1 = c03 */    \
  "smlal2  v9.8h,   v1.16b, v6.16b\n"                  /* a1 * b0 = c12 */    \
  "smlal2  v13.8h,  v1.16b, v7.16b\n"                  /* a1 * b1 = c23 */    \
  "smlal2  v10.8h,  v2.16b, v6.16b\n"                  /* a2 * b0 = c13 */    \
  "smlal2  v14.8h,  v2.16b, v7.16b\n"                  /* a2 * b1 = c32 */    \
  "smlal2  v11.8h,  v3.16b, v6.16b\n"                  /* a3 * b0 = c22 */    \
  "smlal2  v15.8h,  v3.16b, v7.16b\n"                  /* a3 * b1 = c33 */    \
  "cbz    %w[rem],    5f\n"                            /* skip remain */      \
  "ld1 {v0.8b}, [%[a_ptr]]\n"              /* load a to q0, final */          \
  "ld1 {v4.16b, v5.16b}, [%[b_ptr]],#32\n" /* load b to q4, q5 */             \
  "ld1 {v6.16b, v7.16b}, [%[b_ptr]],#32\n" /* load b to q6, q7 */             \
  "5:\n"                                   /* no remain */                    \
  "sadalp  v24.4s, v8.8h\n"  /* pairwise accumulate to int32, out02 */        \
  "sadalp  v28.4s, v12.8h\n" /* pairwise accumulate to int32, out03 */        \
  "sadalp  v25.4s, v9.8h\n"  /* pairwise accumulate to int32, out12 */        \
  "sadalp  v29.4s, v13.8h\n" /* pairwise accumulate to int32, out13 */        \
  "sadalp  v26.4s, v10.8h\n" /* pairwise accumulate to int32, out22 */        \
  "sadalp  v30.4s, v14.8h\n" /* pairwise accumulate to int32, out23 */        \
  "sadalp  v27.4s, v11.8h\n" /* pairwise accumulate to int32, out32 */        \
  "sadalp  v31.4s, v15.8h\n" /* pairwise accumulate to int32, out33 */        \
  "3: \n"                    /* process remains */                            \
  "cbz    %w[rem],    7f\n" /* skip remain */ /* process remain k */          \
  "4: \n"                                     /* remain = 1, 2 */             \
  "ext    v1.8b, v0.8b, v0.8b, #2\n"          /* shift left 2bytes */         \
  "ext    v2.8b, v0.8b, v0.8b, #4\n"          /* shift left 4bytes */         \
  "ext    v3.8b, v0.8b, v0.8b, #6\n" /* shift left 6bytes */ /* 1st b0, b1 */ \
  "smull  v8.8h,   v0.8b, v4.8b\n"                     /* a0 * b0 = c00 */    \
  "smull  v12.8h,  v0.8b, v5.8b\n"                     /* a0 * b1 = c01 */    \
  "smull  v9.8h,   v1.8b, v4.8b\n"                     /* a1 * b0 = c10 */    \
  "smull  v13.8h,  v1.8b, v5.8b\n"                     /* a1 * b1 = c11 */    \
  "smull  v10.8h,  v2.8b, v4.8b\n"                     /* a2 * b0 = c20 */    \
  "smull  v14.8h,  v2.8b, v5.8b\n"                     /* a2 * b1 = c21 */    \
  "smull  v11.8h,  v3.8b, v4.8b\n"                     /* a3 * b0 = c30 */    \
  "smull  v15.8h,  v3.8b, v5.8b\n" /* a3 * b1 = c31 */ /* 1st b2, b3 */       \
  "sadalp  v16.4s, v8.8h\n"        /* pairwise accumulate to int32, out00 */  \
  "smull  v8.8h,   v0.8b, v6.8b\n" /* a0 * b0 = c02 */                        \
  "sadalp  v20.4s, v12.8h\n"       /* pairwise accumulate to int32, out01 */  \
  "smull  v12.8h,  v0.8b, v7.8b\n" /* a0 * b1 = c03 */                        \
  "sadalp  v17.4s, v9.8h\n"        /* pairwise accumulate to int32, out10 */  \
  "smull  v9.8h,   v1.8b, v6.8b\n" /* a1 * b0 = c12 */                        \
  "sadalp  v21.4s, v13.8h\n"       /* pairwise accumulate to int32, out11 */  \
  "smull  v13.8h,  v1.8b, v7.8b\n" /* a1 * b1 = c13 */                        \
  "sadalp  v18.4s, v10.8h\n"       /* pairwise accumulate to int32, out20 */  \
  "smull  v10.8h,  v2.8b, v6.8b\n" /* a2 * b0 = c22 */                        \
  "sadalp  v22.4s, v14.8h\n"       /* pairwise accumulate to int32, out21 */  \
  "smull  v14.8h,  v2.8b, v7.8b\n" /* a2 * b1 = c23 */                        \
  "sadalp  v19.4s, v11.8h\n"       /* pairwise accumulate to int32, out30 */  \
  "smull  v11.8h,  v3.8b, v6.8b\n" /* a3 * b0 = c32 */                        \
  "sadalp  v23.4s, v15.8h\n"       /* pairwise accumulate to int32, out31 */  \
  "smull  v15.8h,  v3.8b, v7.8b\n" /* a3 * b1 = c33 */                        \
  "sadalp  v24.4s, v8.8h\n"        /* pairwise accumulate to int32, out02 */  \
  "sadalp  v28.4s, v12.8h\n"       /* pairwise accumulate to int32, out03 */  \
  "sadalp  v25.4s, v9.8h\n"        /* pairwise accumulate to int32, out12 */  \
  "sadalp  v29.4s, v13.8h\n"       /* pairwise accumulate to int32, out13 */  \
  "sadalp  v26.4s, v10.8h\n"       /* pairwise accumulate to int32, out22 */  \
  "sadalp  v30.4s, v14.8h\n"       /* pairwise accumulate to int32, out23 */  \
  "sadalp  v27.4s, v11.8h\n"       /* pairwise accumulate to int32, out32 */  \
  "sadalp  v31.4s, v15.8h\n"       /* pairwise accumulate to int32, out33 */  \
  "7: \n" /* do relu */            /* do relu */                              \
  "cbz    %w[is_relu],    9f\n"    /* not relu, jump to unpack */             \
  "movi   v0.4s, #0\n"             /* for relu */                             \
  "smax   v16.4s, v16.4s, v0.4s\n" /* relu */                                 \
  "smax   v17.4s, v17.4s, v0.4s\n" /* relu */                                 \
  "smax   v18.4s, v18.4s, v0.4s\n" /* relu */                                 \
  "smax   v19.4s, v19.4s, v0.4s\n" /* relu */                                 \
  "smax   v20.4s, v20.4s, v0.4s\n" /* relu */                                 \
  "smax   v21.4s, v21.4s, v0.4s\n" /* relu */                                 \
  "smax   v22.4s, v22.4s, v0.4s\n" /* relu */                                 \
  "smax   v23.4s, v23.4s, v0.4s\n" /* relu */                                 \
  "smax   v24.4s, v24.4s, v0.4s\n" /* relu */                                 \
  "smax   v25.4s, v25.4s, v0.4s\n" /* relu */                                 \
  "smax   v26.4s, v26.4s, v0.4s\n" /* relu */                                 \
  "smax   v27.4s, v27.4s, v0.4s\n" /* relu */                                 \
  "smax   v28.4s, v28.4s, v0.4s\n" /* relu */                                 \
  "smax   v29.4s, v29.4s, v0.4s\n" /* relu */                                 \
  "smax   v30.4s, v30.4s, v0.4s\n" /* relu */                                 \
  "smax   v31.4s, v31.4s, v0.4s\n" /* relu */ /* unpack the result */         \
  "9:\n" /* unpack */                         /* trans 1 */                   \
  "trn1   v0.4s,  v16.4s, v17.4s\n"           /* get a0,b0, a2,b2 */          \
  "trn2   v1.4s,  v16.4s, v17.4s\n"           /* get a1,b1, a3,b3 */          \
  "trn1   v2.4s,  v18.4s, v19.4s\n"           /* get c0,d0, c2,c2 */          \
  "trn2   v3.4s,  v18.4s, v19.4s\n"           /* get c1,d1, c3,d3 */          \
  "trn1   v4.4s,  v20.4s, v21.4s\n"                                           \
  "trn2   v5.4s,  v20.4s, v21.4s\n"                                           \
  "trn1   v6.4s,  v22.4s, v23.4s\n"                                           \
  "trn2   v7.4s,  v22.4s, v23.4s\n"                                           \
  "trn1   v8.4s,  v24.4s, v25.4s\n"                                           \
  "trn2   v9.4s,  v24.4s, v25.4s\n"                                           \
  "trn1  v10.4s,  v26.4s, v27.4s\n"                                           \
  "trn2  v11.4s,  v26.4s, v27.4s\n"                                           \
  "trn1  v12.4s,  v28.4s, v29.4s\n"                                           \
  "trn2  v13.4s,  v28.4s, v29.4s\n"                                           \
  "trn1  v14.4s,  v30.4s, v31.4s\n"                                           \
  "trn2  v15.4s,  v30.4s, v31.4s\n" /* trans 2 */                             \
  "trn1   v16.2d,  v0.2d, v2.2d\n"  /* get a0,b0, c0,d0 */                    \
  "trn2   v18.2d,  v0.2d, v2.2d\n"  /* get a2,b2, c2,d2 */                    \
  "trn1   v17.2d,  v1.2d, v3.2d\n"  /* get a1,b1, c1,d1 */                    \
  "trn2   v19.2d,  v1.2d, v3.2d\n"  /* get a3,b3, c3,d3 */                    \
  "trn1   v20.2d,  v4.2d, v6.2d\n"                                            \
  "trn2   v22.2d,  v4.2d, v6.2d\n"                                            \
  "trn1   v21.2d,  v5.2d, v7.2d\n"                                            \
  "trn2   v23.2d,  v5.2d, v7.2d\n"                                            \
  "trn1   v24.2d,  v8.2d, v10.2d\n"                                           \
  "trn2   v26.2d,  v8.2d, v10.2d\n"                                           \
  "trn1   v25.2d,  v9.2d, v11.2d\n"                                           \
  "trn2   v27.2d,  v9.2d, v11.2d\n"                                           \
  "trn1   v28.2d,  v12.2d, v14.2d\n"                                          \
  "trn2   v30.2d,  v12.2d, v14.2d\n"                                          \
  "trn1   v29.2d,  v13.2d, v15.2d\n"                                          \
  "trn2   v31.2d,  v13.2d, v15.2d\n"        /* shift */                       \
  "ext    v17.16b, v17.16b, v17.16b, #12\n" /* circular shift left 1 */       \
  "ext    v18.16b, v18.16b, v18.16b, #8\n"  /* circular shift left 2 */       \
  "ext    v19.16b, v19.16b, v19.16b, #4\n"  /* circular shift left 3 */       \
  "ext    v21.16b, v21.16b, v21.16b, #12\n" /* circular shift left 1 */       \
  "ext    v22.16b, v22.16b, v22.16b, #8\n"  /* circular shift left 2 */       \
  "ext    v23.16b, v23.16b, v23.16b, #4\n"  /* circular shift left 3 */       \
  "ext    v25.16b, v25.16b, v25.16b, #12\n" /* circular shift left 1 */       \
  "ext    v26.16b, v26.16b, v26.16b, #8\n"  /* circular shift left 2 */       \
  "ext    v27.16b, v27.16b, v27.16b, #4\n"  /* circular shift left 3 */       \
  "ext    v29.16b, v29.16b, v29.16b, #12\n" /* circular shift left 1 */       \
  "ext    v30.16b, v30.16b, v30.16b, #8\n"  /* circular shift left 2 */       \
  "ext    v31.16b, v31.16b, v31.16b, #4\n"  /* circular shift left 3 */       \
  "trn1   v0.4s,  v16.4s, v17.4s\n"         /* get a0,b0, a2,b2 */            \
  "trn2   v1.4s,  v16.4s, v17.4s\n"         /* get a1,b1, a3,b3 */            \
  "trn1   v2.4s,  v18.4s, v19.4s\n"         /* get c0,d0, c2,c2 */            \
  "trn2   v3.4s,  v18.4s, v19.4s\n"         /* get c1,d1, c3,d3 */            \
  "trn1   v4.4s,  v20.4s, v21.4s\n"                                           \
  "trn2   v5.4s,  v20.4s, v21.4s\n"                                           \
  "trn1   v6.4s,  v22.4s, v23.4s\n"                                           \
  "trn2   v7.4s,  v22.4s, v23.4s\n"                                           \
  "trn1   v8.4s,  v24.4s, v25.4s\n"                                           \
  "trn2   v9.4s,  v24.4s, v25.4s\n"                                           \
  "trn1  v10.4s,  v26.4s, v27.4s\n"                                           \
  "trn2  v11.4s,  v26.4s, v27.4s\n"                                           \
  "trn1  v12.4s,  v28.4s, v29.4s\n"                                           \
  "trn2  v13.4s,  v28.4s, v29.4s\n"                                           \
  "trn1  v14.4s,  v30.4s, v31.4s\n"                                           \
  "trn2  v15.4s,  v30.4s, v31.4s\n" /* trans 2 */                             \
  "trn1   v16.2d,  v0.2d, v2.2d\n"  /* get a0,b0, c0,d0 */                    \
  "trn2   v24.2d,  v0.2d, v2.2d\n"  /* get a2,b2, c2,d2 */                    \
  "trn1   v20.2d,  v1.2d, v3.2d\n"  /* get a1,b1, c1,d1 */                    \
  "trn2   v28.2d,  v1.2d, v3.2d\n"  /* get a3,b3, c3,d3 */                    \
  "trn1   v17.2d,  v4.2d, v6.2d\n"                                            \
  "trn2   v25.2d,  v4.2d, v6.2d\n"                                            \
  "trn1   v21.2d,  v5.2d, v7.2d\n"                                            \
  "trn2   v29.2d,  v5.2d, v7.2d\n"                                            \
  "trn1   v18.2d,  v8.2d, v10.2d\n"                                           \
  "trn2   v26.2d,  v8.2d, v10.2d\n"                                           \
  "trn1   v22.2d,  v9.2d, v11.2d\n"                                           \
  "trn2   v30.2d,  v9.2d, v11.2d\n"                                           \
  "trn1   v19.2d,  v12.2d, v14.2d\n"                                          \
  "trn2   v27.2d,  v12.2d, v14.2d\n"                                          \
  "trn1   v23.2d,  v13.2d, v15.2d\n"                                          \
  "trn2   v31.2d,  v13.2d, v15.2d\n"

// clang-format off
#define GEMM_INT8_INT32_OUT                                                   \
  /* store */                                                                 \
  "st1    {v16.4s, v17.4s, v18.4s, v19.4s},   [%[c_ptr0]], #64\n"   \
  "st1    {v20.4s, v21.4s, v22.4s, v23.4s},   [%[c_ptr1]], #64\n"   \
  "st1    {v24.4s, v25.4s, v26.4s, v27.4s},   [%[c_ptr2]], #64\n"   \
  "st1    {v28.4s, v29.4s, v30.4s, v31.4s},   [%[c_ptr3]], #64\n"
// clang-format on

#define GEMM_INT8_FP32_OUT                                                    \
  /* store */                                                                 \
  "ldr    q15, [%[scale]]\n"         /* load scale */                         \
  "scvtf  v0.4s , v16.4s\n"          /*  00, convert to fp32 */               \
  "scvtf  v1.4s , v17.4s\n"          /*  01, convert to fp32 */               \
  "scvtf  v2.4s , v18.4s\n"          /*  02, convert to fp32 */               \
  "scvtf  v3.4s , v19.4s\n"          /*  03, convert to fp32 */               \
  "scvtf  v4.4s , v20.4s\n"          /*  10, convert to fp32 */               \
  "scvtf  v5.4s , v21.4s\n"          /*  11, convert to fp32 */               \
  "scvtf  v6.4s , v22.4s\n"          /*  12, convert to fp32 */               \
  "scvtf  v7.4s , v23.4s\n"          /*  13, convert to fp32 */               \
  "fmul   v16.4s, v0.4s, v15.s[0]\n" /*  00, mul scale to get final result */ \
  "fmul   v17.4s, v1.4s, v15.s[0]\n" /*  01, mul scale to get final result */ \
  "fmul   v18.4s, v2.4s, v15.s[0]\n" /*  02, mul scale to get final result */ \
  "fmul   v19.4s, v3.4s, v15.s[0]\n" /*  03, mul scale to get final result */ \
  "fmul   v20.4s, v4.4s, v15.s[1]\n" /*  10, mul scale to get final result */ \
  "fmul   v21.4s, v5.4s, v15.s[1]\n" /*  11, mul scale to get final result */ \
  "fmul   v22.4s, v6.4s, v15.s[1]\n" /*  12, mul scale to get final result */ \
  "fmul   v23.4s, v7.4s, v15.s[1]\n" /*  13, mul scale to get final result */ \
  "scvtf  v0.4s , v24.4s\n"          /*  20, convert to fp32 */               \
  "scvtf  v1.4s , v25.4s\n"          /*  21, convert to fp32 */               \
  "stp    q16, q17, [%[c_ptr0]], #32\n" /*  write r0, 0,1 */                  \
  "scvtf  v2.4s , v26.4s\n"             /*  22, convert to fp32 */            \
  "scvtf  v3.4s , v27.4s\n"             /*  23, convert to fp32 */            \
  "stp    q18, q19, [%[c_ptr0]], #32\n" /*  write r0, 2,3 */                  \
  "scvtf  v4.4s , v28.4s\n"             /*  30, convert to fp32 */            \
  "scvtf  v5.4s , v29.4s\n"             /*  31, convert to fp32 */            \
  "stp    q20, q21, [%[c_ptr1]], #32\n" /*  write r1, 0,1 */                  \
  "scvtf  v6.4s , v30.4s\n"             /*  32, convert to fp32 */            \
  "scvtf  v7.4s , v31.4s\n"             /*  33, convert to fp32 */            \
  "stp    q22, q23, [%[c_ptr1]], #32\n" /*  write r1, 2,3 */                  \
  "fmul   v24.4s, v0.4s, v15.s[2]\n" /*  20, mul scale to get final result */ \
  "fmul   v25.4s, v1.4s, v15.s[2]\n" /*  21, mul scale to get final result */ \
  "fmul   v26.4s, v2.4s, v15.s[2]\n" /*  22, mul scale to get final result */ \
  "fmul   v27.4s, v3.4s, v15.s[2]\n" /*  23, mul scale to get final result */ \
  "fmul   v28.4s, v4.4s, v15.s[3]\n" /*  30, mul scale to get final result */ \
  "fmul   v29.4s, v5.4s, v15.s[3]\n" /*  31, mul scale to get final result */ \
  "stp    q24, q25, [%[c_ptr2]], #32\n" /*  write r2, 2,3 */                  \
  "fmul   v30.4s, v6.4s, v15.s[3]\n" /*  32, mul scale to get final result */ \
  "stp    q26, q27, [%[c_ptr2]], #32\n" /*  write r2, 2,3 */                  \
  "fmul   v31.4s, v7.4s, v15.s[3]\n" /*  33, mul scale to get final result */ \
  "stp    q28, q29, [%[c_ptr3]], #32\n" /*  write r3, 2,3 */                  \
  "stp    q30, q31, [%[c_ptr3]], #32\n" /*  write r3, 2,3 */

#define GEMM_INT8_INT8_OUT                                                    \
  /* store */                                                                 \
  "ldr    q15, [%[scale]]\n"         /* load scale */                         \
  "scvtf  v0.4s , v16.4s\n"          /*  00, convert to fp32 */               \
  "scvtf  v1.4s , v17.4s\n"          /*  01, convert to fp32 */               \
  "scvtf  v2.4s , v18.4s\n"          /*  02, convert to fp32 */               \
  "scvtf  v3.4s , v19.4s\n"          /*  03, convert to fp32 */               \
  "scvtf  v4.4s , v20.4s\n"          /*  10, convert to fp32 */               \
  "scvtf  v5.4s , v21.4s\n"          /*  11, convert to fp32 */               \
  "scvtf  v6.4s , v22.4s\n"          /*  12, convert to fp32 */               \
  "scvtf  v7.4s , v23.4s\n"          /*  13, convert to fp32 */               \
  "fmul   v16.4s, v0.4s, v15.s[0]\n" /*  00, mul scale to get final result */ \
  "fmul   v17.4s, v1.4s, v15.s[0]\n" /*  01, mul scale to get final result */ \
  "fmul   v18.4s, v2.4s, v15.s[0]\n" /*  02, mul scale to get final result */ \
  "fmul   v19.4s, v3.4s, v15.s[0]\n" /*  03, mul scale to get final result */ \
  "fmul   v20.4s, v4.4s, v15.s[1]\n" /*  20, mul scale to get final result */ \
  "fmul   v21.4s, v5.4s, v15.s[1]\n" /*  21, mul scale to get final result */ \
  "fmul   v22.4s, v6.4s, v15.s[1]\n" /*  22, mul scale to get final result */ \
  "fmul   v23.4s, v7.4s, v15.s[1]\n" /*  23, mul scale to get final result */ \
  "scvtf  v0.4s , v24.4s\n"          /*  20, convert to fp32 */               \
  "scvtf  v1.4s , v25.4s\n"          /*  21, convert to fp32 */               \
  "scvtf  v2.4s , v26.4s\n"          /*  22, convert to fp32 */               \
  "scvtf  v3.4s , v27.4s\n"          /*  23, convert to fp32 */               \
  "scvtf  v4.4s , v28.4s\n"          /*  30, convert to fp32 */               \
  "scvtf  v5.4s , v29.4s\n"          /*  31, convert to fp32 */               \
  "scvtf  v6.4s , v30.4s\n"          /*  32, convert to fp32 */               \
  "scvtf  v7.4s , v31.4s\n"          /*  33, convert to fp32 */               \
  "fmul   v24.4s, v0.4s, v15.s[2]\n" /*  20, mul scale to get final result */ \
  "fmul   v25.4s, v1.4s, v15.s[2]\n" /*  21, mul scale to get final result */ \
  "fmul   v26.4s, v2.4s, v15.s[2]\n" /*  22, mul scale to get final result */ \
  "fmul   v27.4s, v3.4s, v15.s[2]\n" /*  23, mul scale to get final result */ \
  "fmul   v28.4s, v4.4s, v15.s[3]\n" /*  30, mul scale to get final result */ \
  "fmul   v29.4s, v5.4s, v15.s[3]\n" /*  31, mul scale to get final result */ \
  "fmul   v30.4s, v6.4s, v15.s[3]\n" /*  32, mul scale to get final result */ \
  "fmul   v31.4s, v7.4s, v15.s[3]\n" /*  33, mul scale to get final result */ \
  "fcvtas v0.4s, v16.4s\n"           /*  00, cvt to int */                    \
  "fcvtas v1.4s, v17.4s\n"           /*  01, cvt to int */                    \
  "fcvtas v2.4s, v18.4s\n"           /*  02, cvt to int */                    \
  "fcvtas v3.4s, v19.4s\n"           /*  03, cvt to int */                    \
  "fcvtas v4.4s, v20.4s\n"           /*  10, cvt to int */                    \
  "fcvtas v5.4s, v21.4s\n"           /*  11, cvt to int */                    \
  "fcvtas v6.4s, v22.4s\n"           /*  12, cvt to int */                    \
  "fcvtas v7.4s, v23.4s\n"           /*  13, cvt to int */                    \
  "sqxtn  v16.4h, v0.4s\n"           /*  00, cvt int32 to int16 */            \
  "fcvtas v8.4s, v24.4s\n"           /*  20, cvt to int */                    \
  "sqxtn2 v16.8h, v1.4s\n"           /*  01, cvt int32 to int16 */            \
  "fcvtas v9.4s, v25.4s\n"           /*  21, cvt to int */                    \
  "sqxtn  v17.4h, v2.4s\n"           /*  02, cvt int32 to int16 */            \
  "fcvtas v10.4s, v26.4s\n"          /*  22, cvt to int */                    \
  "sqxtn2 v17.8h, v3.4s\n"           /*  03, cvt int32 to int16 */            \
  "fcvtas v11.4s, v27.4s\n"          /*  23, cvt to int */                    \
  "sqxtn  v18.4h, v4.4s\n"           /*  10, cvt int32 to int16 */            \
  "fcvtas v12.4s, v28.4s\n"          /*  30, cvt to int */                    \
  "sqxtn2 v18.8h, v5.4s\n"           /*  11, cvt int32 to int16 */            \
  "fcvtas v13.4s, v29.4s\n"          /*  31, cvt to int */                    \
  "sqxtn  v19.4h, v6.4s\n"           /*  12, cvt int32 to int16 */            \
  "fcvtas v14.4s, v30.4s\n"          /*  32, cvt to int */                    \
  "sqxtn2 v19.8h, v7.4s\n"           /*  13, cvt int32 to int16 */            \
  "fcvtas v15.4s, v31.4s\n"          /*  33, cvt to int */                    \
  "sqxtn  v0.8b, v16.8h\n"           /*  00, 01, cvt int16 to int8 */         \
  "sqxtn2 v0.16b, v17.8h\n"          /*  02, 03, cvt int16 to int8 */         \
  "sqxtn  v1.8b, v18.8h\n"           /*  10, 11, cvt int16 to int8 */         \
  "sqxtn2 v1.16b, v19.8h\n"          /*  12, 13, cvt int16 to int8 */         \
  "sqxtn  v20.4h, v8.4s\n"           /*  20, cvt int32 to int16 */            \
  "sqxtn2 v20.8h, v9.4s\n"           /*  21, cvt int32 to int16 */            \
  "sqxtn  v21.4h, v10.4s\n"          /*  22, cvt int32 to int16 */            \
  "sqxtn2 v21.8h, v11.4s\n"          /*  23, cvt int32 to int16 */            \
  "sqxtn  v22.4h, v12.4s\n"          /*  30, cvt int32 to int16 */            \
  "sqxtn2 v22.8h, v13.4s\n"          /*  31, cvt int32 to int16 */            \
  "sqxtn  v23.4h, v14.4s\n"          /*  32, cvt int32 to int16 */            \
  "sqxtn2 v23.8h, v15.4s\n"          /*  33, cvt int32 to int16 */            \
  "sqxtn  v2.8b, v20.8h\n"           /*  20, 21, cvt int16 to int8 */         \
  "sqxtn2 v2.16b, v21.8h\n"          /*  22, 23, cvt int16 to int8 */         \
  "sqxtn  v3.8b, v22.8h\n"           /*  30, 31, cvt int16 to int8 */         \
  "sqxtn2 v3.16b, v23.8h\n"          /*  32, 33, cvt int16 to int8 */         \
  "str    q0, [%[c_ptr0]], #16\n"    /*  write r0 */                          \
  "str    q1, [%[c_ptr1]], #16\n"    /*  write r1 */                          \
  "str    q2, [%[c_ptr2]], #16\n"    /*  write r2 */                          \
  "str    q3, [%[c_ptr3]], #16\n"    /*  write r3 */

template <>
inline void gemm_int8_kernel(const int8_t* a_ptr,
                             const int8_t*& b_ptr,                   // NOLINT
                             const int32_t* bias, int32_t*& c_ptr0,  // NOLINT
                             int32_t*& c_ptr1, int32_t*& c_ptr2,     // NOLINT
                             int32_t*& c_ptr3, const float* scale,   // NOLINT
                             bool is_relu,                           // NOLINT
                             int k, int rem) {
  asm volatile(GEMM_INT8_KERNEL GEMM_INT8_INT32_OUT
               : [a_ptr] "+r"(a_ptr), [b_ptr] "+r"(b_ptr),
                 [c_ptr0] "+r"(c_ptr0), [c_ptr1] "+r"(c_ptr1),
                 [c_ptr2] "+r"(c_ptr2), [c_ptr3] "+r"(c_ptr3), [k] "+r"(k)
               : [is_relu] "r"(is_relu), [bias] "r"(bias), [rem] "r"(rem)
               : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9",
                 "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18",
                 "v19", "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27",
                 "v28", "v29", "v30", "v31", "cc");
}
template <>
inline void gemm_int8_kernel(const int8_t* a_ptr,
                             const int8_t*& b_ptr,                 // NOLINT
                             const int32_t* bias, float*& c_ptr0,  // NOLINT
                             float*& c_ptr1, float*& c_ptr2,       // NOLINT
                             float*& c_ptr3,                       // NOLINT
                             const float* scale, bool is_relu, int k, int rem) {
  asm volatile(GEMM_INT8_KERNEL GEMM_INT8_FP32_OUT
               : [a_ptr] "+r"(a_ptr), [b_ptr] "+r"(b_ptr),
                 [c_ptr0] "+r"(c_ptr0), [c_ptr1] "+r"(c_ptr1),
                 [c_ptr2] "+r"(c_ptr2), [c_ptr3] "+r"(c_ptr3), [k] "+r"(k)
               : [is_relu] "r"(is_relu), [bias] "r"(bias), [rem] "r"(rem),
                 [scale] "r"(scale)
               : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9",
                 "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18",
                 "v19", "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27",
                 "v28", "v29", "v30", "v31", "cc");
}

template <>
inline void gemm_int8_kernel(const int8_t* a_ptr,
                             const int8_t*& b_ptr,                  // NOLINT
                             const int32_t* bias, int8_t*& c_ptr0,  // NOLINT
                             int8_t*& c_ptr1, int8_t*& c_ptr2,      // NOLINT
                             int8_t*& c_ptr3,                       // NOLINT
                             const float* scale, bool is_relu, int k, int rem) {
  asm volatile(GEMM_INT8_KERNEL GEMM_INT8_INT8_OUT
               : [a_ptr] "+r"(a_ptr), [b_ptr] "+r"(b_ptr),
                 [c_ptr0] "+r"(c_ptr0), [c_ptr1] "+r"(c_ptr1),
                 [c_ptr2] "+r"(c_ptr2), [c_ptr3] "+r"(c_ptr3), [k] "+r"(k)
               : [is_relu] "r"(is_relu), [bias] "r"(bias), [rem] "r"(rem),
                 [scale] "r"(scale)
               : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9",
                 "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18",
                 "v19", "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27",
                 "v28", "v29", "v30", "v31", "cc");
}

#else  // armv7
// clang-format off
#define GEMM_INT8_KERNEL                                                     \
  "vld1.8 {d0-d1},    [%[a_ptr]: 128]!\n" /* load 4x2x2 int8, A, k2x2 */     \
  "vld1.8 {d4-d7},    [%[b_ptr]: 128]!\n" /* load 8x2x2 int8, B, k2x2 */     \
  "vld1.8 {d8-d9},    [%[bias]]\n"        /* load int32x4 bias */            \
  "vext.8 q5, q4, q4, #4\n"               /* bias shift 1 int32 */           \
  "vext.8 q6, q4, q4, #8\n"               /* bias shift 2 int32 */           \
  "vext.8 q7, q4, q4, #12\n"              /* bias shift 3 int32 */           \
  "pld [%[a_ptr]]\n"                      /* preload A */                    \
  "vand   q8, q4, q4\n"                   /* set bias to out00 */            \
  "vand   q9, q4, q4\n"                   /* set bias to out01 */            \
  "pld [%[b_ptr]]\n"                      /* preload B */                    \
  "vand  q10, q5, q5\n"                   /* set bias to out10 */            \
  "vand  q11, q5, q5\n"                   /* set bias to out11 */            \
  "pld [%[b_ptr], #64]\n"                 /* preload B */                    \
  "vand  q12, q6, q6\n"                   /* set bias to out20 */            \
  "vand  q13, q6, q6\n"                   /* set bias to out21 */            \
  "pld [%[b_ptr], #128]\n"                /* preload B */                    \
  "vand  q14, q7, q7\n"                   /* set bias to out30 */            \
  "vand  q15, q7, q7\n"                   /* set bias to out31 */            \
  "pld [%[a_ptr], #64]\n"                 /* preload A */                    \
  "vext.8 d2, d0, d0, #2\n"               /* shift left circular by 2byte */ \
  "vext.8 d3, d1, d1, #2\n"               /* shift left circular by 2byte */ \
  "pld [%[b_ptr], #192]\n"                /* preload b */                    \
  "pld [%[b_ptr], #256]\n"                /* preload b */                    \
  "pld [%[a_ptr], #128]\n"                /* preload a */                    \
  "cmp    %[k],   #0\n"                   /* check main loop count */        \
  "beq    3f\n" /* if k = 0, jump to remains */ /* 1st r0, r1 */             \
  "vmull.s8  q4, d0, d4\n"                      /* a0 * b0 = c00 */          \
  "vmull.s8  q5, d0, d5\n"                      /* a0 * b1 = c01 */          \
  "vmull.s8  q6, d2, d4\n"                      /* a1 * b0 = c10 */          \
  "vmull.s8  q7, d2, d5\n"                      /* a1 * b1 = c11 */          \
  "subs %[k], %[k], #1\n" /* loop count -1 */   /* 2nd r0, r1 */             \
  "vmlal.s8  q4, d1, d6\n"                      /* a0 * b0 = c00 */          \
  "vmlal.s8  q5, d1, d7\n"                      /* a0 * b1 = c01 */          \
  "vrev64.32  q0, q0\n"              /* shift left circular by 4byte */      \
  "vmlal.s8  q6, d3, d6\n"           /* a1 * b0 = c10 */                     \
  "vmlal.s8  q7, d3, d7\n"           /* a1 * b1 = c11 */                     \
  "vrev64.32  q1, q1\n"              /* shift left circular by 4byte */      \
  "beq    8f\n" /* skip main loop */ /* main loop*/                          \
  "0:\n" /* main loop */             /* 1st r2, r3 */                        \
  "vpadal.s16 q8, q4\n"    /* pair add and accumulate to int32, c00 */       \
  "vmull.s8  q4, d0, d4\n" /* a2 * b0 = c20 */                               \
  "vpadal.s16 q9, q5\n"    /* pair add and accumulate to int32, c01 */       \
  "vmull.s8  q5, d0, d5\n" /* a2 * b1 = c21 */                               \
  "vpadal.s16 q10,q6\n"    /* pair add and accumulate to int32, c10 */       \
  "vmull.s8  q6, d2, d4\n" /* a3 * b0 = c30 */                               \
  "vpadal.s16 q11,q7\n"    /* pair add and accumulate to int32, c11 */       \
  "vmull.s8  q7, d2, d5\n" /* a3 * b1 = c31 */                               \
  "vld1.8 {d4-d5},    [%[b_ptr]: 128]!\n" /* load 4x2x2 int8, B, k2x2 */     \
  "vmlal.s8  q4, d1, d6\n"                /* a0 * b0 = c00 */                \
  "vmlal.s8  q5, d1, d7\n"                /* a0 * b1 = c01 */                \
  "vld1.8 {d0-d1},    [%[a_ptr]: 128]!\n" /* load 4x2x2 int8, A, k2x2 */     \
  "vmlal.s8  q6, d3, d6\n"                /* a1 * b0 = c10 */                \
  "vmlal.s8  q7, d3, d7\n"                /* a1 * b1 = c11 */                \
  "vld1.8 {d6-d7},    [%[b_ptr]: 128]!\n" /* load 4x2x2 int8, B, k2x2 */     \
  "vext.8 d2, d0, d0, #2\n"               /* shift left circular by 2byte */ \
  "vext.8 d3, d1, d1, #2\n" /* shift left circular by 2byte */               \
  "vpadal.s16 q12,q4\n"    /* pair add and accumulate to int32, c20 */       \
  "vmull.s8  q4, d0, d4\n" /* a0 * b0 = c00 */                               \
  "vpadal.s16 q13,q5\n"    /* pair add and accumulate to int32, c21 */       \
  "vmull.s8  q5, d0, d5\n" /* a0 * b1 = c01 */                               \
  "vpadal.s16 q14,q6\n"    /* pair add and accumulate to int32, c30 */       \
  "vmull.s8  q6, d2, d4\n" /* a1 * b0 = c10 */                               \
  "vpadal.s16 q15,q7\n"    /* pair add and accumulate to int32, c31 */       \
  "vmull.s8  q7, d2, d5\n" /* a1 * b1 = c11 */                               \
  "subs %[k], %[k], #1\n" /* loop count -1 */ /* 2nd r0, r1 */               \
  "vmlal.s8  q4, d1, d6\n"                    /* a0 * b0 = c00 */            \
  "vmlal.s8  q5, d1, d7\n"                    /* a0 * b1 = c01 */            \
  "vrev64.32  q0, q0\n"                       /* shift left circular by 2 */ \
  "vmlal.s8  q6, d3, d6\n"                    /* a1 * b0 = c10 */            \
  "vmlal.s8  q7, d3, d7\n"                    /* a1 * b1 = c11 */            \
  "vrev64.32  q1, q1\n"                       /* shift left circular by 2 */ \
  "bgt    0b\n"                               /* jump to main loop */        \
  "8:\n" /* end of main loop */               /* 1st r2, r3 */               \
  "vpadal.s16 q8, q4\n"    /* pair add and accumulate to int32, c00 */       \
  "vmull.s8  q4, d0, d4\n" /* a2 * b0 = c20 */                               \
  "vpadal.s16 q9, q5\n"    /* pair add and accumulate to int32, c01 */       \
  "vmull.s8  q5, d0, d5\n" /* a2 * b1 = c21 */                               \
  "vpadal.s16 q10,q6\n"    /* pair add and accumulate to int32, c10 */       \
  "vmull.s8  q6, d2, d4\n" /* a3 * b0 = c30 */                               \
  "vpadal.s16 q11,q7\n"    /* pair add and accumulate to int32, c11 */       \
  "vmull.s8  q7, d2, d5\n" /* a3 * b1 = c31 */ /* 2nd r2, r3 */              \
  "vmlal.s8  q4, d1, d6\n"                     /* a0 * b0 = c20 */           \
  "vmlal.s8  q5, d1, d7\n"                     /* a0 * b1 = c21 */           \
  "vmlal.s8  q6, d3, d6\n"                     /* a1 * b0 = c30 */           \
  "vmlal.s8  q7, d3, d7\n"                     /* a1 * b1 = c31 */           \
  "cmp    %[rem],    #0\n"                     /* skip remain */             \
  "beq    5f\n"                                                              \
  "mov r0,    #32\n"                 /* address offset */                    \
  "vld1.8 {d0}, [%[a_ptr]]\n"        /* load a to d0, final */               \
  "vld1.8 {d4-d5}, [%[b_ptr]], r0\n" /* load b to d4, d5 */                  \
  "5:\n"                             /* skip rem */                          \
  "vpadal.s16 q12, q4\n"    /* pair add and accumulate to int32, c20 */      \
  "vpadal.s16 q13, q5\n"    /* pair add and accumulate to int32, c21 */      \
  "vpadal.s16 q14, q6\n"    /* pair add and accumulate to int32, c30 */      \
  "vpadal.s16 q15, q7\n"    /* pair add and accumulate to int32, c31 */      \
  "3:\n"                    /* process remain k */                           \
  "cmp    %[rem],    #0\n"  /* skip remain */                                \
  "beq    7f\n"             /* process remain k */                           \
  "vext.8 d1, d0, d0, #2\n" /* shift left 2bytes */                          \
  "vext.8 d2, d0, d0, #4\n" /* shift left 4bytes */                          \
  "vext.8 d3, d0, d0, #6\n" /* shift left 6bytes */ /* 1st r0, r1 */         \
  "vmull.s8  q4, d0, d4\n"                          /* a0 * b0 = c00 */      \
  "vmull.s8  q5, d0, d5\n"                          /* a0 * b1 = c01 */      \
  "vmull.s8  q6, d1, d4\n"                          /* a1 * b0 = c10 */      \
  "vmull.s8  q7, d1, d5\n" /* a1 * b1 = c11 */      /* 1st r2, r3 */         \
  "vpadal.s16 q8, q4\n"        /* pair add and accumulate to int32, c00 */   \
  "vmull.s8  q4, d2, d4\n"     /* a2 * b0 = c20 */                           \
  "vpadal.s16 q9, q5\n"        /* pair add and accumulate to int32, c01 */   \
  "vmull.s8  q5, d2, d5\n"     /* a2 * b1 = c21 */                           \
  "vpadal.s16 q10,q6\n"        /* pair add and accumulate to int32, c10 */   \
  "vmull.s8  q6, d3, d4\n"     /* a3 * b0 = c30 */                           \
  "vpadal.s16 q11,q7\n"        /* pair add and accumulate to int32, c11 */   \
  "vmull.s8  q7, d3, d5\n"     /* a3 * b1 = c31 */                           \
  "vpadal.s16 q12, q4\n"       /* pair add and accumulate to int32, c20 */   \
  "vpadal.s16 q13, q5\n"       /* pair add and accumulate to int32, c21 */   \
  "vpadal.s16 q14, q6\n"       /* pair add and accumulate to int32, c30 */   \
  "vpadal.s16 q15, q7\n"       /* pair add and accumulate to int32, c31 */   \
  "7: \n" /* do relu */        /* do relu */                                 \
  "cmp    %[is_relu],    #0\n" /* skip relu */                               \
  "beq    9f\n"                /* skip relu */                               \
  "vmov.i32   q0, #0\n"        /* for relu */                                \
  "vmax.s32   q8, q8, q0\n"    /* relu */                                    \
  "vmax.s32   q9, q9, q0\n"    /* relu */                                    \
  "vmax.s32  q10,q10, q0\n"    /* relu */                                    \
  "vmax.s32  q11,q11, q0\n"    /* relu */                                    \
  "vmax.s32  q12,q12, q0\n"    /* relu */                                    \
  "vmax.s32  q13,q13, q0\n"    /* relu */                                    \
  "vmax.s32  q14,q14, q0\n"    /* relu */                                    \
  "vmax.s32  q15,q15, q0\n" /* relu */ /* unpack the result */               \
  "9:\n" /* unpack */                  /* trans 1 */                         \
  "vtrn.32    q8, q10\n" /* get q8 */                                    \
  "vtrn.32   q12, q14\n" /* get q12 */                                    \
  "vtrn.32    q9, q11\n" /* get q9 */                                    \
  "vtrn.32   q13, q15\n" /* get q13*/ \
  "vswp   d17,    d24\n" /* get q8*/                                    \
  "vswp   d21,    d28\n" /* get q10 */                                    \
  "vswp   d19,    d26\n" /* get q9 */                                    \
  "vswp   d23,    d30\n" /* get q11 */  \
  "vext.8 q0, q10, q10, #12\n" /* circular shift left 1 q0 */ \
  "vext.8 q2, q12, q12, #8\n"  /* circular shift left 2 q2 */ \
  "vext.8 q4, q14, q14, #4\n"  /* circular shift left 3 q4 */ \
  "vext.8 q1, q11, q11, #12\n" /* circular shift left 1 q1 */ \
  "vext.8 q3, q13, q13, #8\n"  /* circular shift left 2 q3 */ \
  "vext.8 q5, q15, q15, #4\n" /* circular shift left 3 q5 */ \
  "vtrn.32    q8, q0\n" /* get q8 */ \
  "vtrn.32    q2, q4\n" /* get q2 */ \
  "vtrn.32    q9, q1\n" /* get q9 */ \
  "vtrn.32    q3, q5\n" /* get q3 */ /* trans 2 */ \
  "vswp   d17,    d4\n" /* get q8 */ \
  "vswp   d1, d8\n" /* get q0: a1*/ \
  "vswp   d19,    d6\n" /* get q9: */ \
  "vswp   d3, d10\n" /* get q1: a3b3 */

// clang-format off

#define GEMM_INT8_INT32_OUT                                 \
  /* write output */                                        \
  "vst1.32    {d16-d19},  [%[c_ptr0]]!\n" /* write outr0 */ \
  "vst1.32    {d0-d3},    [%[c_ptr1]]!\n" /* write outr1 */ \
  "vst1.32    {d4-d7},    [%[c_ptr2]]!\n" /* write outr2 */ \
  "vst1.32    {d8-d11},   [%[c_ptr3]]!\n" /* write outr3 */

#define GEMM_INT8_FP32_OUT                                               \
  /* write output */                                                     \
  "vld1.32    {d12-d13},  [%[scale]]\n" /* load scale */                 \
  "vcvt.f32.s32   q10, q8\n"            /* r00, cvt int32 to fp32*/      \
  "vcvt.f32.s32   q11, q9\n"            /* r01, cvt int32 to fp32*/      \
  "vcvt.f32.s32   q12, q0\n"            /* r10, cvt int32 to fp32*/      \
  "vcvt.f32.s32   q13, q1\n"            /* r11, cvt int32 to fp32*/      \
  "vmul.f32 q8, q10, d12[0]\n" /*  r00, mul scale to get final result */ \
  "vmul.f32 q9, q11, d12[0]\n" /*  r01, mul scale to get final result */ \
  "vmul.f32 q0, q12, d12[1]\n" /*  r10, mul scale to get final result */ \
  "vmul.f32 q1, q13, d12[1]\n" /*  r11, mul scale to get final result */ \
  "vcvt.f32.s32   q10, q2\n"   /* r20, cvt int32 to fp32*/               \
  "vcvt.f32.s32   q11, q3\n"   /* r21, cvt int32 to fp32*/               \
  "vcvt.f32.s32   q12, q4\n"   /* r30, cvt int32 to fp32*/               \
  "vcvt.f32.s32   q13, q5\n"   /* r31, cvt int32 to fp32*/               \
  "vst1.32    {d16-d19},  [%[c_ptr0]]!\n" /* write r0, float32x4 x2 */   \
  "vmul.f32 q2, q10, d13[0]\n" /* r20, mul scale to get final result */  \
  "vmul.f32 q3, q11, d13[0]\n" /* r21, mul scale to get final result */  \
  "vst1.32    {d0-d3},    [%[c_ptr1]]!\n" /* write r1, float32x4 x2 */   \
  "vmul.f32 q4, q12, d13[1]\n" /* r30, mul scale to get final result */  \
  "vmul.f32 q5, q13, d13[1]\n" /* r31, mul scale to get final result */  \
  "vst1.32    {d4-d7},    [%[c_ptr2]]!\n" /* write r2, float32x4 x2 */   \
  "vst1.32    {d8-d11},   [%[c_ptr3]]!\n" /* write r3, float32x4 x2 */

#define GEMM_INT8_INT8_OUT                                                    \
  /* write output */                                                          \
  "vld1.32    {d12-d13},  [%[scale]]\n" /* load scale */                      \
  "vmov.f32  q7, #-0.5\n"               /* neg offset */                      \
  "vcvt.f32.s32   q10, q8\n"            /* r00, cvt int32 to fp32*/           \
  "vcvt.f32.s32   q11, q9\n"            /* r01, cvt int32 to fp32*/           \
  "vcvt.f32.s32   q12, q0\n"            /* r10, cvt int32 to fp32*/           \
  "vcvt.f32.s32   q13, q1\n"            /* r11, cvt int32 to fp32*/           \
  "vmov.f32  q8, #0.5\n"                /* pos offset */                      \
  "vmov.f32  q9, #0.5\n"                /* pos offset */                      \
  "vmov.f32  q0, #0.5\n"                /* pos offset */                      \
  "vmov.f32  q1, #0.5\n"                /* pos offset */                      \
  "vcgt.f32  q14, q10, #0\n"            /* get pos mask */                    \
  "vcgt.f32  q15, q11, #0\n"            /* get pos mask */                    \
  "vbif.f32  q8, q7, q14\n"             /* get right offset */                \
  "vbif.f32  q9, q7, q15\n"             /* get right offset */                \
  "vcgt.f32  q14, q12, #0\n"            /* get pos mask */                    \
  "vcgt.f32  q15, q13, #0\n"            /* get pos mask */                    \
  "vbif.f32  q0, q7, q14\n"             /* get right offset */                \
  "vbif.f32  q1, q7, q15\n"             /* get right offset */                \
  "vmla.f32 q8, q10, d12[0]\n"       /* r00, mul scale to get final result */ \
  "vmla.f32 q9, q11, d12[0]\n"       /* r01, mul scale to get final result */ \
  "vmla.f32 q0, q12, d12[1]\n"       /* r10, mul scale to get final result */ \
  "vmla.f32 q1, q13, d12[1]\n"       /* r11, mul scale to get final result */ \
  "vcvt.f32.s32   q10, q2\n"         /* r20, cvt int32 to fp32*/              \
  "vcvt.f32.s32   q11, q3\n"         /* r21, cvt int32 to fp32*/              \
  "vcvt.f32.s32   q12, q4\n"         /* r30, cvt int32 to fp32*/              \
  "vcvt.f32.s32   q13, q5\n"         /* r31, cvt int32 to fp32*/              \
  "vmov.f32  q2, #0.5\n"             /* pos offset */                         \
  "vmov.f32  q3, #0.5\n"             /* pos offset */                         \
  "vmov.f32  q4, #0.5\n"             /* pos offset */                         \
  "vmov.f32  q5, #0.5\n"             /* pos offset */                         \
  "vcgt.f32  q14, q10, #0\n"         /* get pos mask */                       \
  "vcgt.f32  q15, q11, #0\n"         /* get pos mask */                       \
  "vbif.f32  q2, q7, q14\n"          /* get right offset */                   \
  "vbif.f32  q3, q7, q15\n"          /* get right offset */                   \
  "vcgt.f32  q14, q12, #0\n"         /* get pos mask */                       \
  "vcgt.f32  q15, q13, #0\n"         /* get pos mask */                       \
  "vbif.f32  q4, q7, q14\n"          /* get right offset */                   \
  "vbif.f32  q5, q7, q15\n"          /* get right offset */                   \
  "vmla.f32 q2, q10, d13[0]\n"       /* r20, mul scale to get final result */ \
  "vmla.f32 q3, q11, d13[0]\n"       /* r21, mul scale to get final result */ \
  "vmla.f32 q4, q12, d13[1]\n"       /* r30, mul scale to get final result */ \
  "vmla.f32 q5, q13, d13[1]\n"       /* r31, mul scale to get final result */ \
  "vcvt.s32.f32   q6, q8\n"          /* r00, fp32->int32 */                   \
  "vcvt.s32.f32   q7, q9\n"          /* r01, fp32->int32 */                   \
  "vcvt.s32.f32   q10, q0\n"         /* r10, fp32->int32 */                   \
  "vcvt.s32.f32   q11, q1\n"         /* r11, fp32->int32 */                   \
  "vcvt.s32.f32   q12, q2\n"         /* r20, fp32->int32 */                   \
  "vcvt.s32.f32   q13, q3\n"         /* r21, fp32->int32 */                   \
  "vcvt.s32.f32   q14, q4\n"         /* r30, fp32->int32 */                   \
  "vcvt.s32.f32   q15, q5\n"         /* r31, fp32->int32 */                   \
  "vqmovn.s32 d0, q6\n"              /* r00, int32 -> int16 */                \
  "vqmovn.s32 d1, q7\n"              /* r01, int32 -> int16 */                \
  "vqmovn.s32 d2, q10\n"             /* r10, int32 -> int16 */                \
  "vqmovn.s32 d3, q11\n"             /* r11, int32 -> int16 */                \
  "vqmovn.s32 d4, q12\n"             /* r00, int32 -> int16 */                \
  "vqmovn.s32 d5, q13\n"             /* r01, int32 -> int16 */                \
  "vqmovn.s32 d6, q14\n"             /* r10, int32 -> int16 */                \
  "vqmovn.s32 d7, q15\n"             /* r11, int32 -> int16 */                \
  "vqmovn.s16 d8, q0\n"              /* 0, int16 -> int8 */                   \
  "vqmovn.s16 d9, q1\n"              /* 1, int16 -> int8 */                   \
  "vqmovn.s16 d10, q2\n"             /* 2, int16 -> int8 */                   \
  "vqmovn.s16 d11, q3\n"             /* 3, int16 -> int8 */                   \
  "vst1.32    {d8}, [%[c_ptr0]]!\n"  /* write r0*/                            \
  "vst1.32    {d9}, [%[c_ptr1]]!\n"  /* write r1*/                            \
  "vst1.32    {d10}, [%[c_ptr2]]!\n" /* write r2*/                            \
  "vst1.32    {d11}, [%[c_ptr3]]!\n" /* write r3*/

template <>
inline void gemm_int8_kernel(const int8_t* a_ptr, const int8_t*& b_ptr,    // NOLINT
                             const int32_t* bias, int32_t*& c_ptr0,        // NOLINT
                             int32_t*& c_ptr1, int32_t*& c_ptr2,           // NOLINT
                             int32_t*& c_ptr3, const float* scale, bool is_relu, // NOLINT
                             int k, int rem) {
  asm volatile(GEMM_INT8_KERNEL GEMM_INT8_INT32_OUT
               : [a_ptr] "+r"(a_ptr), [b_ptr] "+r"(b_ptr),
                 [c_ptr0] "+r"(c_ptr0), [c_ptr1] "+r"(c_ptr1),
                 [c_ptr2] "+r"(c_ptr2), [c_ptr3] "+r"(c_ptr3), [k] "+r"(k)
               : [is_relu] "r"(is_relu), [bias] "r"(bias), [rem] "r"(rem)
               : "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7", "q8", "q9",
                 "q10", "q11", "q12", "q13", "q14", "q15", "r0", "cc");
}

template <>
inline void gemm_int8_kernel(const int8_t* a_ptr, const int8_t*& b_ptr,  // NOLINT
                             const int32_t* bias, float*& c_ptr0,        // NOLINT
                             float*& c_ptr1, float*& c_ptr2, float*& c_ptr3, // NOLINT
                             const float* scale, bool is_relu, int k, int rem) {
  asm volatile(GEMM_INT8_KERNEL GEMM_INT8_FP32_OUT
               : [a_ptr] "+r"(a_ptr), [b_ptr] "+r"(b_ptr),
                 [c_ptr0] "+r"(c_ptr0), [c_ptr1] "+r"(c_ptr1),
                 [c_ptr2] "+r"(c_ptr2), [c_ptr3] "+r"(c_ptr3), [k] "+r"(k)
               : [is_relu] "r"(is_relu), [bias] "r"(bias), [rem] "r"(rem),
                 [scale] "r"(scale)
               : "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7", "q8", "q9",
                 "q10", "q11", "q12", "q13", "q14", "q15", "r0", "cc");
}

template <>
inline void gemm_int8_kernel(const int8_t* a_ptr, const int8_t*& b_ptr,   // NOLINT
                             const int32_t* bias, int8_t*& c_ptr0,        // NOLINT
                             int8_t*& c_ptr1, int8_t*& c_ptr2, int8_t*& c_ptr3, // NOLINT
                             const float* scale, bool is_relu, int k, int rem) {
  asm volatile(GEMM_INT8_KERNEL GEMM_INT8_INT8_OUT
               : [a_ptr] "+r"(a_ptr), [b_ptr] "+r"(b_ptr),
                 [c_ptr0] "+r"(c_ptr0), [c_ptr1] "+r"(c_ptr1),
                 [c_ptr2] "+r"(c_ptr2), [c_ptr3] "+r"(c_ptr3), [k] "+r"(k)
               : [is_relu] "r"(is_relu), [bias] "r"(bias), [rem] "r"(rem),
                 [scale] "r"(scale)
               : "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7", "q8", "q9",
                 "q10", "q11", "q12", "q13", "q14", "q15", "r0", "cc");
}
#endif                               //__aarch64__ // NOLINT

// gemm wrapper
template <typename Dtype>
void gemm_prepack_int8(const int8_t* A_packed, const int8_t* B, const int* bias,
                       Dtype* C, int M, int N, int K, bool is_bias,
                       bool is_relu, bool is_transB, const float* scale,
                       ARMContext* ctx) {
  const int MBLOCK = 4;
  const int KUP = ROUNDUP(K, KBLOCK_INT8);
  size_t l2_cache =
      ctx->l2_cache_size() > 0 ? ctx->l2_cache_size() : 512 * 1024;
  auto* workspace = ctx->workspace_data<int8_t>();
  int threads = ctx->threads();
  int x_block = l2_cache / (sizeof(int8_t) * (KUP + MBLOCK));
  x_block /= NBLOCK_INT8;
  x_block *= NBLOCK_INT8;
  int x_num = (N + (x_block - 1)) / x_block;
  x_block = (N + x_num - 1) / x_num;
  x_block = (x_block + NBLOCK_INT8 - 1) / NBLOCK_INT8;
  x_block *= NBLOCK_INT8;
  int k = K / KBLOCK_INT8;
  int k_rem = K & (KBLOCK_INT8 - 1);
  if (k_rem > KBLOCK_INT8 / 2) {
    k_rem = 0;
    k += 1;
  }
  int n_rem = N & (NBLOCK_INT8 - 1);
  bool flag_rem = n_rem > 0;

  auto* b_tmp = static_cast<int8_t*>(workspace);

  auto* zerobuf =
      static_cast<int8_t*>(malloc(x_block * (sizeof(int8_t) + sizeof(Dtype))));
  memset(zerobuf, 0, x_block * sizeof(int8_t));
  auto* trash_ptr =
      reinterpret_cast<Dtype*>(zerobuf + x_block * sizeof(int8_t));

  //! apanel is pre_compute outside gemm
  for (unsigned int x0 = 0; x0 < N; x0 += x_block) {
    unsigned int xmax = x0 + x_block;
    if (xmax > N) {
      xmax = N;
    }
    int bblocks = (xmax - x0 + NBLOCK_INT8 - 1) / NBLOCK_INT8;
    //! load bpanel
    int8_t* b_pannel = b_tmp;
    if (is_transB) {
      packb_trans_int8(b_pannel, B, K, 0, K, x0, xmax, zerobuf);
    } else {
      packb_int8(b_pannel, B, N, 0, K, x0, xmax, zerobuf);
    }

#pragma omp parallel for num_threads(threads)
    for (unsigned int y = 0; y < M; y += MBLOCK) {
      Dtype out0[NBLOCK_INT8] = {0};
      Dtype out1[NBLOCK_INT8] = {0};
      Dtype out2[NBLOCK_INT8] = {0};
      Dtype out3[NBLOCK_INT8] = {0};
      Dtype* c_ptr0 = C + y * N + x0;
      Dtype* c_ptr1 = c_ptr0 + N;
      Dtype* c_ptr2 = c_ptr1 + N;
      Dtype* c_ptr3 = c_ptr2 + N;
      Dtype* tmp0 = nullptr;
      Dtype* tmp1 = nullptr;
      Dtype* tmp2 = nullptr;
      Dtype* tmp3 = nullptr;
      float scale_local[4];
      int32_t bias_local[4] = {0, 0, 0, 0};
      if (is_bias) {
        bias_local[0] = bias[y];
        bias_local[1] = bias[y + 1];
        bias_local[2] = bias[y + 2];
        bias_local[3] = bias[y + 3];
      }
      if (scale) {
        scale_local[0] = scale[y];
        scale_local[1] = scale[y + 1];
        scale_local[2] = scale[y + 2];
        scale_local[3] = scale[y + 3];
      }
      if (y + MBLOCK > M) {
        switch (y + MBLOCK - M) {
          case 3:
            c_ptr1 = trash_ptr;
          case 2:
            c_ptr2 = trash_ptr;
          case 1:
            c_ptr3 = trash_ptr;
          default:
            break;
        }
      }
      const int8_t* a_ptr_l = A_packed + y * KUP;
      const int8_t* b_ptr = b_pannel;
      for (int xb = 0; xb < bblocks; xb++) {
        if (flag_rem && (xb == bblocks - 1)) {
          tmp0 = c_ptr0;
          tmp1 = c_ptr1;
          tmp2 = c_ptr2;
          tmp3 = c_ptr3;
          c_ptr0 = out0;
          c_ptr1 = out1;
          c_ptr2 = out2;
          c_ptr3 = out3;
        }
        gemm_int8_kernel<Dtype>(a_ptr_l, b_ptr, bias_local, c_ptr0, c_ptr1,
                                c_ptr2, c_ptr3, scale_local, is_relu, k, k_rem);
        if (flag_rem && (xb == bblocks - 1)) {
          for (int i = 0; i < n_rem; ++i) {
            *(tmp0++) = out0[i];
            *(tmp1++) = out1[i];
            *(tmp2++) = out2[i];
            *(tmp3++) = out3[i];
          }
        }
      }
    }
  }
  free(zerobuf);
}

template void gemm_prepack_int8<float>(const int8_t* A_packed, const int8_t* B,
                                       const int* bias, float* C, int M, int N,
                                       int K, bool is_bias, bool is_relu,
                                       bool is_transB, const float* scale,
                                       ARMContext* ctx);
template void gemm_prepack_int8<int32_t>(const int8_t* A_packed,
                                         const int8_t* B, const int* bias,
                                         int32_t* C, int M, int N, int K,
                                         bool is_bias, bool is_relu,
                                         bool is_transB, const float* scale,
                                         ARMContext* ctx);
template void gemm_prepack_int8<int8_t>(const int8_t* A_packed, const int8_t* B,
                                        const int* bias, int8_t* C, int M,
                                        int N, int K, bool is_bias,
                                        bool is_relu, bool is_transB,
                                        const float* scale, ARMContext* ctx);

/***********************************************************************/
// prepack A according to gemm kernel
// A block size: (<4x2>x1) x2, with unroll=2 can be described as below:
// origin A data:
// A_origin(no trans, m x k):
//      r0: ==>   a0, b0, c0, d0, e0, f0, g0, h0
//      r1: ==>   a1, b1, c1, d1, e1, f1, g1, h1
//      r2: ==>   a2, b2, c2, d2, e2, f2, g2, h2
//      r3: ==>   a3, b3, c3, d3, e3, f3, g3, h3
// packed A
//      a0,b0, a1,b1, a2,b2, a3,b3;
//      c0,d0, c1,d1, c2,d2, c3,d3;
//      e0,f0, e1,f1, e2,f2, e3,f3;
//      g0,h0, g1,h1, g2,h2, g3,h3;
/***********************************************************************/
void prepackA_m4k2x2_int8(int8_t* out, const int8_t* in, const int ldin,
                          const int m0, const int mmax, const int k0,
                          const int kmax) {
  int y_len = mmax - m0;
  int x_len = kmax - k0;
  int x_len_roundup = ROUNDUP(x_len, KBLOCK_INT8);
  int8_t* zerobuff = static_cast<int8_t*>(malloc(x_len_roundup * sizeof(char)));
  memset(zerobuff, 0, sizeof(char) * x_len_roundup);

  const int8_t* inptr = in + m0 * ldin + k0;
  uint8_t remain = static_cast<uint8_t>(x_len & (KBLOCK_INT8 - 1));

#pragma omp parallel for
  for (int y = 0; y < y_len; y += MBLOCK_INT8) {
    const int8_t* ptr0 = inptr + y * ldin;
    const int8_t* ptr1 = ptr0 + ldin;
    const int8_t* ptr2 = ptr1 + ldin;
    const int8_t* ptr3 = ptr2 + ldin;
    //! cope with row index exceed real size, set to zero buffer
    if ((y + MBLOCK_INT8) > y_len) {
      switch ((y + MBLOCK_INT8) - y_len) {
        case 3:
          ptr1 = zerobuff;
        case 2:
          ptr2 = zerobuff;
        case 1:
          ptr3 = zerobuff;
        default:
          break;
      }
    }
    int8_t* ptr_out = out + y * x_len_roundup;
    int i = 0;
    for (; i < x_len + 1 - 2 * KBLOCK_INT8; i += 2 * KBLOCK_INT8) {
#ifdef __aarch64__
      asm volatile(
          "ld1    {v0.8b}, [%[ptr0]], #8\n" /* load r0, 8 int8 */
          "ld1    {v1.8b}, [%[ptr1]], #8\n" /* load r1, 8 int8 */
          "ld1    {v2.8b}, [%[ptr2]], #8\n" /* load r2, 8 int8 */
          "ld1    {v3.8b}, [%[ptr3]], #8\n" /* load r3, 8 int8 */
          "trn1   v4.4h, v0.4h, v1.4h\n"    /* get a0,b0, a2,b2 */
          "trn2   v5.4h, v0.4h, v1.4h\n"    /* get a1,b1, a3,b3 */
          "trn1   v6.4h, v2.4h, v3.4h\n"    /* get c0,d0, c2,d2 */
          "trn2   v7.4h, v2.4h, v3.4h\n"    /* get c1,d1, c3,d3 */
          "trn1   v0.2s, v4.2s, v6.2s\n"    /* get a0,b0, c0,d0 */
          "trn2   v2.2s, v4.2s, v6.2s\n"    /* get a2,b2, c2,d2 */
          "trn1   v1.2s, v5.2s, v7.2s\n"    /* get a1,b1, c1,d1 */
          "trn2   v3.2s, v5.2s, v7.2s\n"    /* get a3,b3, c3,d3 */
          "st1    {v0.8b, v1.8b, v2.8b, v3.8b}, [%[ptr_out]], #32\n" /* write
                                                                        out*/
          : [ptr_out] "+r"(ptr_out), [ptr0] "+r"(ptr0), [ptr1] "+r"(ptr1),
            [ptr2] "+r"(ptr2), [ptr3] "+r"(ptr3)
          :
          : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "cc", "memory");
#else   // armv7
      asm volatile(
          "vld1.8 {d0}, [%[ptr0]]!\n" /* load r0, 8 int8,
                                         a0,b0,c0,d0,e0,f0,g0,h0 */
          "vld1.8 {d1}, [%[ptr1]]!\n" /* load r1, 8 int8,
                                         a1,b1,c1,d1,e1,f1,g1,h1 */
          "vld1.8 {d2}, [%[ptr2]]!\n" /* load r2, 8 int8,
                                         a2,b2,c2,d2,e2,f2,g2,h2 */
          "vld1.8 {d3}, [%[ptr3]]!\n" /* load r3, 8 int8,
                                         a3,b3,c3,d3,e3,f3,g3,h3 */
          "vtrn.16    d0, d1\n" /* trans, d0: a0,b0,a1,b1, e0,f0,e1,f1; d1:
                                   c0,d0,c1,d1, g0,h0,g1,h1 */
          "vtrn.16    d2, d3\n" /* trans, d2: a2,b2,a3,b3, e2,f2,e3,f3; d3:
                                   c2,d2,c3,d3, g2,h2,g3,h3 */
          "vtrn.32    d0, d2\n" /* trans, d0: a0,b0,a1,b1, a2,b2,a3,b3; d2:
                                   e0,f0,e1,f1, e2,f2,e3,f3 */
          "vtrn.32    d1, d3\n" /* trans, d1: c0,d0,c1,d1, e2,f2,e3,f3; d3:
                                   g0,h0,g1,h1, g2,h2,g3,h3 */
          "vst1.32 {d0-d3}, [%[outptr]]!\n" /* write to output ptr */
          : [outptr] "+r"(ptr_out), [ptr0] "+r"(ptr0), [ptr1] "+r"(ptr1),
            [ptr2] "+r"(ptr2), [ptr3] "+r"(ptr3)
          :
          : "q0", "q1", "cc", "memory");
#endif  //__aarch64 // NOLINT
    }
    if (i + KBLOCK_INT8 <= x_len) {
      ptr_out[0] = ptr0[0];
      ptr_out[1] = ptr0[1];
      ptr_out[2] = ptr1[0];
      ptr_out[3] = ptr1[1];
      ptr_out[4] = ptr2[0];
      ptr_out[5] = ptr2[1];
      ptr_out[6] = ptr3[0];
      ptr_out[7] = ptr3[1];
      // unroll
      ptr_out[8] = ptr0[2];
      ptr_out[9] = ptr0[3];
      ptr_out[10] = ptr1[2];
      ptr_out[11] = ptr1[3];
      ptr_out[12] = ptr2[2];
      ptr_out[13] = ptr2[3];
      ptr_out[14] = ptr3[2];
      ptr_out[15] = ptr3[3];
      ptr_out += 16;
      ptr0 += 4;
      ptr1 += 4;
      ptr2 += 4;
      ptr3 += 4;
    }
    switch (remain) {
      case 0:
        break;
      case 1:
        ptr_out[0] = ptr0[0];
        ptr_out[1] = 0;
        ptr_out[2] = ptr1[0];
        ptr_out[3] = 0;
        ptr_out[4] = ptr2[0];
        ptr_out[5] = 0;
        ptr_out[6] = ptr3[0];
        ptr_out[7] = 0;
        // unroll
        ptr_out[8] = 0;
        ptr_out[9] = 0;
        ptr_out[10] = 0;
        ptr_out[11] = 0;
        ptr_out[12] = 0;
        ptr_out[13] = 0;
        ptr_out[14] = 0;
        ptr_out[15] = 0;
        ptr_out += 16;
        break;
      case 2:
        ptr_out[0] = ptr0[0];
        ptr_out[1] = ptr0[1];
        ptr_out[2] = ptr1[0];
        ptr_out[3] = ptr1[1];
        ptr_out[4] = ptr2[0];
        ptr_out[5] = ptr2[1];
        ptr_out[6] = ptr3[0];
        ptr_out[7] = ptr3[1];
        // unroll
        ptr_out[8] = 0;
        ptr_out[9] = 0;
        ptr_out[10] = 0;
        ptr_out[11] = 0;
        ptr_out[12] = 0;
        ptr_out[13] = 0;
        ptr_out[14] = 0;
        ptr_out[15] = 0;
        ptr_out += 16;
        break;
      case 3:
        ptr_out[0] = ptr0[0];
        ptr_out[1] = ptr0[1];
        ptr_out[2] = ptr1[0];
        ptr_out[3] = ptr1[1];
        ptr_out[4] = ptr2[0];
        ptr_out[5] = ptr2[1];
        ptr_out[6] = ptr3[0];
        ptr_out[7] = ptr3[1];
        // unroll
        ptr_out[8] = ptr0[2];
        ptr_out[9] = 0;
        ptr_out[10] = ptr1[2];
        ptr_out[11] = 0;
        ptr_out[12] = ptr2[2];
        ptr_out[13] = 0;
        ptr_out[14] = ptr3[2];
        ptr_out[15] = 0;
        ptr_out += 16;
        break;
      default:
        break;
    }
  }
  free(zerobuff);
}

/***************************************************************************/
// prepack A according to gemm kernel
// A block size: <4x2>x2, unroll x4, can be described as below:
// origin A data:
// A_origin(no trans, k x m):
//      r0: ==>   a0, a1, a2, a3 .... a12, a13, a14, a15
//      r1: ==>   b0, b1, b2, b3 .... b12, b13, b14, b15
//      r2: ==>   c0, c1, c2, c3 .... c12, c13, c14, c15
//      r3: ==>   d0, d1, d2, d3 .... d12, d13, d14, d15
// packed A:
//      a0,b0, a1,b1, a2,b2, a3,b3;
//      c0,d0, c1,d1, c2,d2, c3,d3;----block0
//      a4,b4, a5,b5, a6,b6, a7,b7;
//      c4,d4, c5,d5, c6,d6, c7,d7;----block1
//      a8,b8, a9,b9, a10,b10, a11,b11;
//      c8,d8, c9,d9, c10,d10, c11,d11;----block2
//      a12,b12, a13,b13, a14,b14, a15,b15;
//      c12,d12, c13,d13, c14,d14, c15,d15;----block3
/***************************************************************************/
void prepackA_m4k2x2_trans_int8(int8_t* out, const int8_t* in, const int ldin,
                                const int m0, const int mmax, const int k0,
                                const int kmax) {
  int xlen = mmax - m0;
  int ylen = kmax - k0;
  int ylen_roundup = ROUNDUP(ylen, KBLOCK_INT8);
  int xlen_roundup = ROUNDUP(xlen, MBLOCK_INT8);

  const int MUNROLL = 4;
  int mcnt = xlen / (MUNROLL * MBLOCK_INT8);
  int x_rem = xlen & (MUNROLL * MBLOCK_INT8 - 1);
  int m_rem = (x_rem + MBLOCK_INT8 - 1) / MBLOCK_INT8;

  const uint8_t mask_buffer[16] = {0, 1, 2,  3,  4,  5,  6,  7,
                                   8, 9, 10, 11, 12, 13, 14, 15};
  int8x16_t vzero = vdupq_n_s8(0);
  uint8x16_t vmask = vcltq_u8(vld1q_u8(mask_buffer), vdupq_n_u8(x_rem));

  int stride_out = ylen_roundup * MBLOCK_INT8;

  int8_t* zerobuf = static_cast<int8_t*>(malloc(xlen_roundup));
  memset(zerobuf, 0, xlen_roundup);

  const int8_t* inr = in + ldin * k0 + m0;
#pragma omp parallel for
  for (int y = 0; y < ylen; y += KBLOCK_INT8) {
    const int8_t* ptr0 = inr + y * ldin;
    const int8_t* ptr1 = ptr0 + ldin;
    const int8_t* ptr2 = ptr1 + ldin;
    const int8_t* ptr3 = ptr2 + ldin;
    int8_t* ptr_out = out + MBLOCK_INT8 * y;
    if (y + KBLOCK_INT8 > ylen) {
      switch (y + KBLOCK_INT8 - ylen) {
        case 3:
          ptr1 = zerobuf;
        case 2:
          ptr2 = zerobuf;
        case 1:
          ptr3 = zerobuf;
        default:
          break;
      }
    }
    int k = mcnt;
    int rem = m_rem;
#ifdef __aarch64__
    asm volatile(
        "ld1    {v0.16b},   [%[ptr0]],  #16\n" /* load r0 */
        "ld1    {v1.16b},   [%[ptr1]],  #16\n" /* load r1 */
        "ld1    {v2.16b},   [%[ptr2]],  #16\n" /* load r2 */
        "ld1    {v3.16b},   [%[ptr3]],  #16\n" /* load r3 */
        "cbz    %w[k], 1f\n"                   /* jump to remain */
        "0:\n"                                 /* main loop */
        /* trans 16b */
        "trn1   v4.16b, v0.16b, v1.16b\n" /* get a0,b0, a2,b2, a4,b4, a6,b6,
                                             a8,b8, a10,b10, a12,b12, a14,b14 */
        "trn2   v5.16b, v0.16b, v1.16b\n" /* get a1,b1, a3,b3, a5,b5, a7,b7,
                                             a9,b9, a11,b11, a13,b13, a15,b15 */
        "trn1   v6.16b, v2.16b, v3.16b\n" /* get c0,d0, c2,d2, c4,d4, c6,d6,
                                             c8,d8, c10,d10, c12,d12, c14,d14 */
        "trn2   v7.16b, v2.16b, v3.16b\n" /* get c1,d1, c3,d3, c5,d5, c7,d7,
                                             c9,d9, c11,d11, c13,d13, c15,d15 */
        "ld1    {v0.16b},   [%[ptr0]],  #16\n" /* load r0 */
        "ld1    {v1.16b},   [%[ptr1]],  #16\n" /* load r1 */
        "subs   %w[k], %w[k], #1\n"            /* loop cnt -1 */
        /* trans 8h */
        "trn1   v8.8h, v4.8h, v5.8h\n" /* get a0,b0, a1,b1, a4,b4, a5,b5, a8,b8,
                                          a9,b9, a12,b12, a13,b13 */
        "trn2   v9.8h, v4.8h, v5.8h\n" /* get a2,b2, a3,b3, a6,b6, a7,b7,
                                          a10,b10, a11,b11, a14,b14, a15,b15 */
        "trn1   v10.8h, v6.8h, v7.8h\n" /* get c0,d0, c1,d1, c4,d4, c5,d5,
                                           c8,d8, c9,d9, c12,d12, c13,d13 */
        "trn2   v11.8h, v6.8h, v7.8h\n" /* get c2,d2, c3,d3, c6,d6, c7,d7,
                                           c10,d10, c11,d11, c14,d14, c15,d15 */
        /* trans 4s */
        "ld1    {v2.16b},   [%[ptr2]],  #16\n" /* load r2 */
        "trn1   v4.4s, v8.4s, v9.4s\n" /* get a0,b0, a1,b1, a2,b2, a3,b3, a8,b8,
                                          a9,b9, a10,b10, a11,b11 */
        "trn2   v5.4s, v8.4s, v9.4s\n" /* get a4,b4, a5,b5, a6,b6, a7,b7,
                                          a12,b12, a13,b13, a14,b14, a15,b15 */
        "trn1   v6.4s, v10.4s, v11.4s\n" /* get c0,d0, c1,d1, c2,d2, c3,d3,
                                            c8,d8, c9,d9, c10,d10, c11,d11 */
        "trn2   v7.4s, v10.4s, v11.4s\n" /* get c4,d4, c5,d5, c6,d6, c7,d7,
                                            c12,d12, c13,d13, c14,d14, c15,d15
                                            */
        /* trans 2d */
        "ld1    {v3.16b},   [%[ptr3]],  #16\n" /* load r3 */
        "trn1   v8.2d, v4.2d, v6.2d\n" /* get a0,b0, a1,b1, a2,b2, a3,b3, c0,d0,
                                          c1,d1, c2,d2, c3,d3 */
        "trn1   v9.2d, v5.2d, v7.2d\n" /* get a4,b4, a5,b5, a6,b6, a7,b7, c4,d4,
                                          c5,d5, c6,d6, c7,d7 */
        "trn2   v10.2d, v4.2d, v6.2d\n" /* get a8,b8, a9,b9, a10,b10, a11,b11,
                                           c8,d8, c9,d9, c10,d10, c11,d11 */
        "trn2   v11.2d, v5.2d, v7.2d\n" /* get a12,b12, a13,b13, a14,b14,
                                           a15,b15, c12,d12, c13,d13, c14,d14,
                                           c15,d15 */
        "st1    {v8.16b}, [%[ptr_out]], %[stride]\n" /* write block0, address +
                                                        stride */
        "st1    {v9.16b}, [%[ptr_out]], %[stride]\n" /* write block1, address +
                                                        stride */
        "st1   {v10.16b}, [%[ptr_out]], %[stride]\n" /* write block2, address +
                                                        stride */
        "st1   {v11.16b}, [%[ptr_out]], %[stride]\n" /* write block3, address +
                                                        stride */
        "bgt    0b\n"                                /* jump to main loop */
        "1:\n"                                       /* process remain */
        "cbz    %w[rem], 2f\n"                       /* skip to remain */
        /* bit select */
        "bif    v0.16b, %[vzero].16b, %[mask].16b\n" /* pad 0 */
        "bif    v1.16b, %[vzero].16b, %[mask].16b\n" /* pad 0 */
        "bif    v2.16b, %[vzero].16b, %[mask].16b\n" /* pad 0 */
        "bif    v3.16b, %[vzero].16b, %[mask].16b\n" /* pad 0 */
        /* trans 16b */
        "trn1   v4.16b, v0.16b, v1.16b\n" /* get a0,b0, a2,b2, a4,b4, a6,b6,
                                             a8,b8, a10,b10, a12,b12, a14,b14 */
        "trn2   v5.16b, v0.16b, v1.16b\n" /* get a1,b1, a3,b3, a5,b5, a7,b7,
                                             a9,b9, a11,b11, a13,b13, a15,b15 */
        "trn1   v6.16b, v2.16b, v3.16b\n" /* get c0,d0, c2,d2, c4,d4, c6,d6,
                                             c8,d8, c10,d10, c12,d12, c14,d14 */
        "trn2   v7.16b, v2.16b, v3.16b\n" /* get c1,d1, c3,d3, c5,d5, c7,d7,
                                             c9,d9, c11,d11, c13,d13, c15,d15 */
        /* trans 8h */
        "trn1   v8.8h, v4.8h, v5.8h\n" /* get a0,b0, a1,b1, a4,b4, a5,b5, a8,b8,
                                          a9,b9, a12,b12, a13,b13 */
        "trn2   v9.8h, v4.8h, v5.8h\n" /* get a2,b2, a3,b3, a6,b6, a7,b7,
                                          a10,b10, a11,b11, a14,b14, a15,b15 */
        "trn1   v10.8h, v6.8h, v7.8h\n" /* get c0,d0, c1,d1, c4,d4, c5,d5,
                                           c8,d8, c9,d9, c12,d12, c13,d13 */
        "trn2   v11.8h, v6.8h, v7.8h\n" /* get c2,d2, c3,d3, c6,d6, c7,d7,
                                           c10,d10, c11,d11, c14,d14, c15,d15 */
        /* trans 4s */
        "trn1   v4.4s, v8.4s, v9.4s\n" /* get a0,b0, a1,b1, a2,b2, a3,b3, a8,b8,
                                          a9,b9, a10,b10, a11,b11 */
        "trn2   v5.4s, v8.4s, v9.4s\n" /* get a4,b4, a5,b5, a6,b6, a7,b7,
                                          a12,b12, a13,b13, a14,b14, a15,b15 */
        "trn1   v6.4s, v10.4s, v11.4s\n" /* get c0,d0, c1,d1, c2,d2, c3,d3,
                                            c8,d8, c9,d9, c10,d10, c11,d11 */
        "trn2   v7.4s, v10.4s, v11.4s\n" /* get c4,d4, c5,d5, c6,d6, c7,d7,
                                            c12,d12, c13,d13, c14,d14, c15,d15
                                            */
        /* trans 2d */
        "trn1   v8.2d, v4.2d, v6.2d\n" /* get a0,b0, a1,b1, a2,b2, a3,b3, c0,d0,
                                          c1,d1, c2,d2, c3,d3 */
        "trn1   v9.2d, v5.2d, v7.2d\n" /* get a4,b4, a5,b5, a6,b6, a7,b7, c4,d4,
                                          c5,d5, c6,d6, c7,d7 */
        "trn2   v10.2d, v4.2d, v6.2d\n" /* get a8,b8, a9,b9, a10,b10, a11,b11,
                                           c8,d8, c9,d9, c10,d10, c11,d11 */
        "trn2   v11.2d, v5.2d, v7.2d\n" /* get a12,b12, a13,b13, a14,b14,
                                           a15,b15, c12,d12, c13,d13, c14,d14,
                                           c15,d15 */
        /* check remain size */
        "subs    %w[rem], %w[rem], #1\n"             /* check remain num */
        "st1    {v8.16b}, [%[ptr_out]], %[stride]\n" /* write 0 */
        "beq    2f\n"                                /* remain = 1 */
        "subs    %w[rem], %w[rem], #1\n"             /* check remain num */
        "st1    {v9.16b}, [%[ptr_out]], %[stride]\n" /* write 1 */
        "beq    2f\n"                                /* remain = 2 */
        "subs    %w[rem], %w[rem], #1\n"             /* check remain num */
        "st1   {v10.16b}, [%[ptr_out]], %[stride]\n" /* write 2 */
        "beq    2f\n"                                /* remain = 3 */
        "st1   {v11.16b}, [%[ptr_out]]\n"            /* write 3 */
        /* end */
        "2:\n" /* end */
        : [ptr0] "+r"(ptr0), [ptr1] "+r"(ptr1), [ptr2] "+r"(ptr2),
          [ptr3] "+r"(ptr3), [k] "+r"(k), [rem] "+r"(rem),
          [ptr_out] "+r"(ptr_out)
        : [mask] "w"(vmask), [vzero] "w"(vzero), [stride] "r"(stride_out)
        : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10",
          "v11", "cc");
#else   // armv7
    asm volatile(
        "vld1.8 {d0-d1},    [%[ptr0]]!\n" /* load r0 */
        "vld1.8 {d2-d3},    [%[ptr1]]!\n" /* load r1 */
        "vld1.8 {d4-d5},    [%[ptr2]]!\n" /* load r2 */
        "vld1.8 {d6-d7},    [%[ptr3]]!\n" /* load r3 */
        "cmp    %[k], #0\n"               /* check main loop */
        "beq    1f\n"                     /* jump to remain */
        "0:\n"                            /* main loop */
        /* trans 16b */
        "vtrn.8 q0, q1\n" /* get q0: a0,b0, a2,b2, a4,b4, a6,b6, a8,b8, a10,b10,
                             a12,b12, a14,b14; q1: a1,b1, a3,b3, a5,b5, a7,b7,
                             a9,b9, a11,b11, a13,b13, a15,b15 */
        "vtrn.8 q2, q3\n" /* get q2: c0,d0, c2,d2, c4,d4, c6,d6, c8,d8, c10,d10,
                             c12,d12, c14,d14; q3: c0,d0, c2,d2, c4,d4, c6,d6,
                             c8,d8, c10,d10, c12,d12, c14,d14 */
        "subs   %[k], %[k], #1\n" /* loop cnt -1 */
        /* trans 8h */
        "vtrn.16    q0, q1\n" /* get q0: a0,b0, a1,b1, a4,b4, a5,b5, a8,b8,
                                 a9,b9, a12,b12, a13,b13; q1: a2,b2, a3,b3,
                                 a6,b6, a7,b7, a10,b10, a11,b11, a14,b14,
                                 a15,b15 */
        "vtrn.16    q2, q3\n" /* get q2: c0,d0, c1,d1, c4,d4, c5,d5, c8,d8,
                                 c9,d9, c12,d12, c13,d13; q3: c2,d2, c3,d3,
                                 c6,d6, c7,d7, c10,d10, c11,d11, c14,d14,
                                 c15,d15 */
        /* trans 4s */
        "vtrn.32    q0, q1\n" /* get q0: a0,b0, a1,b1, a2,b2, a3,b3, a8,b8,
                                 a9,b9, a10,b10, a11,b11; q1: a4,b4, a5,b5,
                                 a6,b6, a7,b7, a12,b12, a13,b13, a14,b14,
                                 a15,b15 */
        "vtrn.32    q2, q3\n" /* get q2: c0,d0, c1,d1, c2,d2, c3,d3, c8,d8,
                                 c9,d9, c10,d10, c11,d11; q3: c4,d4, c5,d5,
                                 c6,d6, c7,d7, c12,d12, c13,d13, c14,d14,
                                 c15,d15 */
        /* trans 2d */
        "vswp   d1, d4\n" /* get q0: a0,b0, a1,b1, a2,b2, a3,b3, c0,d0, c1,d1,
                             c2,d2, c3,d3; q2: a8,b8, a9,b9, a10,b10, a11,b11,
                             c8,d8, c9,d9, c10,d10, c11,d11 */
        "vswp   d3, d6\n" /* get q1: a4,b4, a5,b5, a6,b6, a7,b7, c4,d4, c5,d5,
                             c6,d6, c7,d7; q3: a12,b12, a13,b13, a14,b14,
                             a15,b15, c12,d12, c13,d13, c14,d14, c15,d15 */
        "vst1.8 {d0-d1}, [%[ptr_out]], %[stride]\n" /* write block0, address +
                                                       stride */
        "vst1.8 {d2-d3}, [%[ptr_out]], %[stride]\n" /* write block1, address +
                                                       stride */
        "vst1.8 {d4-d5}, [%[ptr_out]], %[stride]\n" /* write block2, address +
                                                       stride */
        "vst1.8 {d6-d7}, [%[ptr_out]], %[stride]\n" /* write block3, address +
                                                       stride */
        "vld1.8 {d0-d1},    [%[ptr0]]!\n"           /* load r0 */
        "vld1.8 {d2-d3},    [%[ptr1]]!\n"           /* load r1 */
        "vld1.8 {d4-d5},    [%[ptr2]]!\n"           /* load r2 */
        "vld1.8 {d6-d7},    [%[ptr3]]!\n"           /* load r3 */
        "bgt    0b\n"                               /* jump to main loop */
        "1:\n"                                      /* process remain */
        "cmp    %[rem], #0\n"                       /* check remain */
        "beq    2f\n"                               /* skip to remain */
        /* bit select */
        "vbif   q0, %q[vzero], %q[mask]\n" /* pad 0 */
        "vbif   q1, %q[vzero], %q[mask]\n" /* pad 0 */
        "vbif   q2, %q[vzero], %q[mask]\n" /* pad 0 */
        "vbif   q3, %q[vzero], %q[mask]\n" /* pad 0 */
        /* trans 16b */
        "vtrn.8 q0, q1\n" /* get q0: a0,b0, a2,b2, a4,b4, a6,b6, a8,b8, a10,b10,
                             a12,b12, a14,b14; q1: a1,b1, a3,b3, a5,b5, a7,b7,
                             a9,b9, a11,b11, a13,b13, a15,b15 */
        "vtrn.8 q2, q3\n" /* get q2: c0,d0, c2,d2, c4,d4, c6,d6, c8,d8, c10,d10,
                             c12,d12, c14,d14; q3: c0,d0, c2,d2, c4,d4, c6,d6,
                             c8,d8, c10,d10, c12,d12, c14,d14 */
        /* trans 8h */
        "vtrn.16    q0, q1\n" /* get q0: a0,b0, a1,b1, a4,b4, a5,b5, a8,b8,
                                 a9,b9, a12,b12, a13,b13; q1: a2,b2, a3,b3,
                                 a6,b6, a7,b7, a10,b10, a11,b11, a14,b14,
                                 a15,b15 */
        "vtrn.16    q2, q3\n" /* get q2: c0,d0, c1,d1, c4,d4, c5,d5, c8,d8,
                                 c9,d9, c12,d12, c13,d13; q3: c2,d2, c3,d3,
                                 c6,d6, c7,d7, c10,d10, c11,d11, c14,d14,
                                 c15,d15 */
        /* trans 4s */
        "vtrn.32    q0, q1\n" /* get q0: a0,b0, a1,b1, a2,b2, a3,b3, a8,b8,
                                 a9,b9, a10,b10, a11,b11; q1: a4,b4, a5,b5,
                                 a6,b6, a7,b7, a12,b12, a13,b13, a14,b14,
                                 a15,b15 */
        "vtrn.32    q2, q3\n" /* get q2: c0,d0, c1,d1, c2,d2, c3,d3, c8,d8,
                                 c9,d9, c10,d10, c11,d11; q3: c4,d4, c5,d5,
                                 c6,d6, c7,d7, c12,d12, c13,d13, c14,d14,
                                 c15,d15 */
        /* trans 2d */
        "vswp   d1, d4\n" /* get q0: a0,b0, a1,b1, a2,b2, a3,b3, c0,d0, c1,d1,
                             c2,d2, c3,d3; q2: a8,b8, a9,b9, a10,b10, a11,b11,
                             c8,d8, c9,d9, c10,d10, c11,d11 */
        "vswp   d3, d6\n" /* get q1: a4,b4, a5,b5, a6,b6, a7,b7, c4,d4, c5,d5,
                             c6,d6, c7,d7; q3: a12,b12, a13,b13, a14,b14,
                             a15,b15, c12,d12, c13,d13, c14,d14, c15,d15 */
        /* check remain size */
        "subs    %[rem], %[rem], #1\n"              /* check remain num */
        "vst1.8 {d0-d1}, [%[ptr_out]], %[stride]\n" /* write 0 */
        "beq    2f\n"                               /* remain = 1 */
        "subs    %[rem], %[rem], #1\n"              /* check remain num */
        "vst1.8 {d2-d3}, [%[ptr_out]], %[stride]\n" /* write 1 */
        "beq    2f\n"                               /* remain = 2 */
        "subs    %[rem], %[rem], #1\n"              /* check remain num */
        "vst1.8 {d4-d5}, [%[ptr_out]], %[stride]\n" /* write 2 */
        "beq    2f\n"                               /* remain = 3 */
        "vst1.8 {d6-d7}, [%[ptr_out]], %[stride]\n" /* write 3 */
        /* end */
        "2:\n" /* end */
        : [ptr0] "+r"(ptr0), [ptr1] "+r"(ptr1), [ptr2] "+r"(ptr2),
          [ptr3] "+r"(ptr3), [k] "+r"(k), [rem] "+r"(rem),
          [ptr_out] "+r"(ptr_out)
        : [mask] "w"(vmask), [vzero] "w"(vzero), [stride] "r"(stride_out)
        : "q0", "q1", "q2", "q3", "cc");
#endif  //__aarch64__ // NOLINT
  }
  free(zerobuf);
}

/**************************************************************************/
// for armv8
// prepack B according to gemm kernel
// B block size: (<4x2>x4) x2, can be described as below:
// origin B data:
// B_origin(no trans, k x n):
//      r0: ==>   a0, a1, a2, a3 .... a12, a13, a14, a15
//      r1: ==>   b0, b1, b2, b3 .... b12, b13, b14, b15
//      r2: ==>   c0, c1, c2, c3 .... c12, c13, c14, c15
//      r3: ==>   d0, d1, d2, d3 .... d12, d13, d14, d15
// packed B:
//      a0,b0, a1,b1, a2,b2, a3,b3;
//      c0,d0, c1,d1, c2,d2, c3,d3;
//                   .
//                   .
//                   .
//      a12,b12, a13,b13, a14,b14, a15,b15;
//      c12,d12, c13,d13, c14,d14, c15,d15;
// for armv7
// prepack B according to gemm kernel
// B block size: (<4x2>x4) x2, can be described as below:
// origin B data:
// B_origin(no trans, k x n):
//      r0: ==>   a0, a1, a2, a3, a4, a5, a6, a7
//      r1: ==>   b0, b1, b2, b3, b4, b5, b6, b7
//      r2: ==>   c0, c1, c2, c3, c4, c5, c6, c7
//      r3: ==>   d0, d1, d2, d3, d4, d5, d6, d7
// packed B:
//      a0,b0, a1,b1, a2,b2, a3,b3;
//      a4,b4, a5,b5, a6,b6, a7,b7;
//      c0,d0, c1,d1, c2,d2, c3,d3;
//      c4,d4, c5,d5, c6,d6, c7,d7;
/***************************************************************************/
void packb_int8(int8_t* out, const int8_t* in, const int ldin, const int k0,
                const int kmax, const int n0, const int nmax,
                const int8_t* zerobuf) {
  const int8_t* inptr = in + k0 * ldin + n0;
  const uint8_t mask_buffer[16] = {0, 1, 2,  3,  4,  5,  6,  7,
                                   8, 9, 10, 11, 12, 13, 14, 15};
  int x_len = nmax - n0;
  int y_len = kmax - k0;
  int kup = ROUNDUP(y_len, KBLOCK_INT8);
  int kcnt = x_len / NBLOCK_INT8;
  int rem = x_len & (NBLOCK_INT8 - 1);
  int stride_out = NBLOCK_INT8 * kup;

  int8x16_t vzero = vdupq_n_s8(0);
  uint8x16_t vmask = vcltq_u8(vld1q_u8(mask_buffer), vdupq_n_u8(rem));
#pragma omp parallel for
  for (int y = 0; y < y_len; y += KBLOCK_INT8) {
    const int8_t* ptr0 = inptr + y * ldin;
    const int8_t* ptr1 = ptr0 + ldin;
    const int8_t* ptr2 = ptr1 + ldin;
    const int8_t* ptr3 = ptr2 + ldin;
    if (y + KBLOCK_INT8 > y_len) {
      switch (y + KBLOCK_INT8 - y_len) {
        case 3:
          ptr1 = zerobuf;
        case 2:
          ptr2 = zerobuf;
        case 1:
          ptr3 = zerobuf;
        default:
          break;
      }
    }
    int8_t* outptr_row_col = out + y * NBLOCK_INT8;
    int k = kcnt;
#ifdef __aarch64__
    asm volatile(
        "ld1    {v0.16b},   [%[ptr0]],  #16\n" /* load r0 */
        "ld1    {v1.16b},   [%[ptr1]],  #16\n" /* load r1 */
        "ld1    {v2.16b},   [%[ptr2]],  #16\n" /* load r2 */
        "ld1    {v3.16b},   [%[ptr3]],  #16\n" /* load r3 */
        "cbz    %w[k], 1f\n"                   /* jump to remain */
        "0:\n"                                 /* main loop */
        /* trans 16b */
        "trn1   v4.16b, v0.16b, v1.16b\n" /* get a0,b0, a2,b2, a4,b4, a6,b6,
                                             a8,b8, a10,b10, a12,b12, a14,b14 */
        "trn2   v5.16b, v0.16b, v1.16b\n" /* get a1,b1, a3,b3, a5,b5, a7,b7,
                                             a9,b9, a11,b11, a13,b13, a15,b15 */
        "trn1   v6.16b, v2.16b, v3.16b\n" /* get c0,d0, c2,d2, c4,d4, c6,d6,
                                             c8,d8, c10,d10, c12,d12, c14,d14 */
        "trn2   v7.16b, v2.16b, v3.16b\n" /* get c1,d1, c3,d3, c5,d5, c7,d7,
                                             c9,d9, c11,d11, c13,d13, c15,d15 */
        "ld1    {v0.16b},   [%[ptr0]],  #16\n" /* load r0 */
        "ld1    {v1.16b},   [%[ptr1]],  #16\n" /* load r1 */
        "subs   %w[k], %w[k], #1\n"            /* loop cnt -1 */
        /* trans 8h */
        "trn1   v8.8h, v4.8h, v5.8h\n" /* get a0,b0, a1,b1, a4,b4, a5,b5, a8,b8,
                                          a9,b9, a12,b12, a13,b13 */
        "trn2   v9.8h, v4.8h, v5.8h\n" /* get a2,b2, a3,b3, a6,b6, a7,b7,
                                          a10,b10, a11,b11, a14,b14, a15,b15 */
        "trn1   v10.8h, v6.8h, v7.8h\n" /* get c0,d0, c1,d1, c4,d4, c5,d5,
                                           c8,d8, c9,d9, c12,d12, c13,d13 */
        "trn2   v11.8h, v6.8h, v7.8h\n" /* get c2,d2, c3,d3, c6,d6, c7,d7,
                                           c10,d10, c11,d11, c14,d14, c15,d15 */
        /* trans 4s */
        "ld1    {v2.16b},   [%[ptr2]],  #16\n" /* load r2 */
        "trn1   v4.4s, v8.4s, v9.4s\n" /* get a0,b0, a1,b1, a2,b2, a3,b3, a8,b8,
                                          a9,b9, a10,b10, a11,b11 */
        "trn2   v5.4s, v8.4s, v9.4s\n" /* get a4,b4, a5,b5, a6,b6, a7,b7,
                                          a12,b12, a13,b13, a14,b14, a15,b15 */
        "trn1   v6.4s, v10.4s, v11.4s\n" /* get c0,d0, c1,d1, c2,d2, c3,d3,
                                            c8,d8, c9,d9, c10,d10, c11,d11 */
        "trn2   v7.4s, v10.4s, v11.4s\n" /* get c4,d4, c5,d5, c6,d6, c7,d7,
                                            c12,d12, c13,d13, c14,d14, c15,d15
                                            */
        /* trans 2d */
        "ld1    {v3.16b},   [%[ptr3]],  #16\n" /* load r3 */
        "trn1   v8.2d, v4.2d, v6.2d\n" /* get a0,b0, a1,b1, a2,b2, a3,b3, c0,d0,
                                          c1,d1, c2,d2, c3,d3 */
        "trn2   v10.2d, v4.2d, v6.2d\n" /* get a8,b8, a9,b9, a10,b10, a11,b11,
                                           c8,d8, c9,d9, c10,d10, c11,d11 */
        "trn1   v9.2d, v5.2d, v7.2d\n" /* get a4,b4, a5,b5, a6,b6, a7,b7, c4,d4,
                                          c5,d5, c6,d6, c7,d7 */
        "trn2   v11.2d, v5.2d, v7.2d\n" /* get a12,b12, a13,b13, a14,b14,
                                           a15,b15, c12,d12, c13,d13, c14,d14,
                                           c15,d15 */
        "st1    {v8.16b, v9.16b, v10.16b, v11.16b},   [%[ptr_out]], %[stride]\n"
        "bgt    0b\n"          /* jump to main loop */
        "1:\n"                 /* process remain */
        "cbz    %w[rem], 2f\n" /* jump to remain */
        /* bit select */
        "bif    v0.16b, %[vzero].16b, %[mask].16b\n" /* pad 0 */
        "bif    v1.16b, %[vzero].16b, %[mask].16b\n" /* pad 0 */
        "bif    v2.16b, %[vzero].16b, %[mask].16b\n" /* pad 0 */
        "bif    v3.16b, %[vzero].16b, %[mask].16b\n" /* pad 0 */
        /* trans 16b */
        "trn1   v4.16b, v0.16b, v1.16b\n" /* get a0,b0, a2,b2, a4,b4, a6,b6,
                                             a8,b8, a10,b10, a12,b12, a14,b14 */
        "trn2   v5.16b, v0.16b, v1.16b\n" /* get a1,b1, a3,b3, a5,b5, a7,b7,
                                             a9,b9, a11,b11, a13,b13, a15,b15 */
        "trn1   v6.16b, v2.16b, v3.16b\n" /* get c0,d0, c2,d2, c4,d4, c6,d6,
                                             c8,d8, c10,d10, c12,d12, c14,d14 */
        "trn2   v7.16b, v2.16b, v3.16b\n" /* get c1,d1, c3,d3, c5,d5, c7,d7,
                                             c9,d9, c11,d11, c13,d13, c15,d15 */
        /* trans 8h */
        "trn1   v8.8h, v4.8h, v5.8h\n" /* get a0,b0, a1,b1, a4,b4, a5,b5, a8,b8,
                                          a9,b9, a12,b12, a13,b13 */
        "trn2   v9.8h, v4.8h, v5.8h\n" /* get a2,b2, a3,b3, a6,b6, a7,b7,
                                          a10,b10, a11,b11, a14,b14, a15,b15 */
        "trn1   v10.8h, v6.8h, v7.8h\n" /* get c0,d0, c1,d1, c4,d4, c5,d5,
                                           c8,d8, c9,d9, c12,d12, c13,d13 */
        "trn2   v11.8h, v6.8h, v7.8h\n" /* get c2,d2, c3,d3, c6,d6, c7,d7,
                                           c10,d10, c11,d11, c14,d14, c15,d15 */
        /* trans 4s */
        "trn1   v4.4s, v8.4s, v9.4s\n" /* get a0,b0, a1,b1, a2,b2, a3,b3, a8,b8,
                                          a9,b9, a10,b10, a11,b11 */
        "trn2   v5.4s, v8.4s, v9.4s\n" /* get a4,b4, a5,b5, a6,b6, a7,b7,
                                          a12,b12, a13,b13, a14,b14, a15,b15 */
        "trn1   v6.4s, v10.4s, v11.4s\n" /* get c0,d0, c1,d1, c2,d2, c3,d3,
                                            c8,d8, c9,d9, c10,d10, c11,d11 */
        "trn2   v7.4s, v10.4s, v11.4s\n" /* get c4,d4, c5,d5, c6,d6, c7,d7,
                                            c12,d12, c13,d13, c14,d14, c15,d15
                                            */
        /* trans 2d */
        "trn1   v8.2d, v4.2d, v6.2d\n" /* get a0,b0, a1,b1, a2,b2, a3,b3, c0,d0,
                                          c1,d1, c2,d2, c3,d3 */
        "trn2   v10.2d, v4.2d, v6.2d\n" /* get a8,b8, a9,b9, a10,b10, a11,b11,
                                           c8,d8, c9,d9, c10,d10, c11,d11 */
        "trn1   v9.2d, v5.2d, v7.2d\n" /* get a4,b4, a5,b5, a6,b6, a7,b7, c4,d4,
                                          c5,d5, c6,d6, c7,d7 */
        "trn2   v11.2d, v5.2d, v7.2d\n" /* get a12,b12, a13,b13, a14,b14,
                                           a15,b15, c12,d12, c13,d13, c14,d14,
                                           c15,d15 */
        "st1    {v8.16b, v9.16b, v10.16b, v11.16b},   [%[ptr_out]]\n" /* save to
                                                                         memory
                                                                         */
        /* end */
        "2:\n" /* end */
        : [ptr0] "+r"(ptr0), [ptr1] "+r"(ptr1), [ptr2] "+r"(ptr2),
          [ptr3] "+r"(ptr3), [k] "+r"(k), [ptr_out] "+r"(outptr_row_col)
        : [rem] "r"(rem), [mask] "w"(vmask), [vzero] "w"(vzero),
          [stride] "r"(stride_out)
        : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10",
          "v11", "cc");
#else   // armv7
    asm volatile(
        "vld1.8 {d0},   [%[ptr0]]!\n" /* load r0, a0,a1,a2,a3,a4,a5,a6,a7 */
        "vld1.8 {d1},   [%[ptr1]]!\n" /* load r1, b0,b1,b2,b3,b4,b5,b6,b7 */
        "vld1.8 {d2},   [%[ptr2]]!\n" /* load r2, c0,c1,c2,c3,c4,c5,c6,c7 */
        "vld1.8 {d3},   [%[ptr3]]!\n" /* load r3, d0,d1,d2,d3,d4,d5,d6,d7 */
        "cmp    %[k], #0\n"           /* check main loop count */
        "beq    1f\n"                 /* jump to remain */
        "0:\n"                        /* main loop */
        /* trans 8b */
        "vtrn.8 d0, d1\n" /* get d0: a0,b0, a2,b2, a4,b4, a6,b6; d1: a1,b1,
                             a3,b3, a5,b5, a7,b7 */
        "vtrn.8 d2, d3\n" /* get d2: c0,d0, c2,d2, c4,d4, c6,d6; d3: c1,d1,
                             c3,d3, c5,d5, c7,d7 */
        /* trans 4h */
        "vtrn.16    d0, d1\n" /* get d0: a0,b0, a1,b1, a4,b4, a5,b5; d1: a2,b2,
                                 a3,b3, a6,b6, a7,b7 */
        "vtrn.16    d2, d3\n" /* get d2: c0,d0, c1,d1, c4,d4, c5,d5; d3: c2,d2,
                                 c3,d3, c6,d6, c7,d7 */
        "subs   %[k],   %[k],   #1\n" /* loop - 1 */
        /* trans 2s */
        "vtrn.32    d0, d1\n" /* get d0: a0,b0, a1,b1, a2,b2, a3,b3; d1: a4,b4,
                                 a5,b5, a6,b6, a7,b7 */
        "vtrn.32    d2, d3\n" /* get d2: c0,d0, c1,d1, c2,d2, c3,d3; d3: c4,d4,
                                 c5,d5, c6,d6, c7,d7 */
        "vst1.8 {d0-d3},   [%[ptr_out]], %[stride]\n" /* save to memory */
        "vld1.8 {d0},   [%[ptr0]]!\n" /* load r0, a0,a1,a2,a3,a4,a5,a6,a7 */
        "vld1.8 {d1},   [%[ptr1]]!\n" /* load r1, b0,b1,b2,b3,b4,b5,b6,b7 */
        "vld1.8 {d2},   [%[ptr2]]!\n" /* load r2, c0,c1,c2,c3,c4,c5,c6,c7 */
        "vld1.8 {d3},   [%[ptr3]]!\n" /* load r3, d0,d1,d2,d3,d4,d5,d6,d7 */
        "bgt    0b\n"                 /* jump to main loop */
        "1:\n"                        /* process remain */
        "cmp    %[rem], #0\n"         /* check remain size */
        "beq    2f\n"                 /* jump to end */
        /* bit select */
        "vbif    d0, %e[vzero], %e[mask]\n" /* pad 0 */
        "vbif    d1, %e[vzero], %e[mask]\n" /* pad 0 */
        "vbif    d2, %e[vzero], %e[mask]\n" /* pad 0 */
        "vbif    d3, %e[vzero], %e[mask]\n" /* pad 0 */
        /* trans 8b */
        "vtrn.8 d0, d1\n" /* get d0: a0,b0, a2,b2, a4,b4, a6,b6; d1: a1,b1,
                             a3,b3, a5,b5, a7,b7 */
        "vtrn.8 d2, d3\n" /* get d2: c0,d0, c2,d2, c4,d4, c6,d6; d3: c1,d1,
                             c3,d3, c5,d5, c7,d7 */
        /* trans 4h */
        "vtrn.16    d0, d1\n" /* get d0: a0,b0, a1,b1, a4,b4, a5,b5; d1: a2,b2,
                                 a3,b3, a6,b6, a7,b7 */
        "vtrn.16    d2, d3\n" /* get d2: c0,d0, c1,d1, c4,d4, c5,d5; d3: c2,d2,
                                 c3,d3, c6,d6, c7,d7 */
        /* trans 2s */
        "vtrn.32    d0, d1\n" /* get d0: a0,b0, a1,b1, a2,b2, a3,b3; d1: a4,b4,
                                 a5,b5, a6,b6, a7,b7 */
        "vtrn.32    d2, d3\n" /* get d2: c0,d0, c1,d1, c2,d2, c3,d3; d3: c4,d4,
                                 c5,d5, c6,d6, c7,d7 */
        "vst1.8 {d0-d3},   [%[ptr_out]]\n" /* save to memory */
        /* end */
        "2:\n" /* end */
        : [ptr0] "+r"(ptr0), [ptr1] "+r"(ptr1), [ptr2] "+r"(ptr2),
          [ptr3] "+r"(ptr3), [k] "+r"(k), [ptr_out] "+r"(outptr_row_col)
        : [rem] "r"(rem), [mask] "w"(vmask), [vzero] "w"(vzero),
          [stride] "r"(stride_out)
        : "q0", "q1", "cc");
#endif  //__aarch64__  // NOLINT
  }
}

/************************************************************************/
// prepack B according to gemm kernel
// origin B data:
// B_origin(transpose, n x k:
//      k unroll 2, a0=k0,k1
//      r0: ==>   a0, a1, a2, a3, a4, a5, a6, a7
//      r1: ==>   b0, b1, b2, b3, b4, b5, b6, b7
//      r2: ==>   c0, c1, c2, c3, c4, c5, c6, c7
//      r3: ==>   d0, d1, d2, d3, d4, d5, d6, d7
//      r4: ==>   e0, e1, e2, e3, e4, e5, e6, e7
//      r5: ==>   f0, f1, f2, f3, f4, f5, f6, f7
//      r6: ==>   g0, g1, g2, g3, g4, g5, g6, g7
//      r7: ==>   h0, h1, h2, h3, h4, h5, h6, h7
// for armv8:
// B block size: (<4x2>x4) x2, can be described as below:
// packed B:
//      a0,b0, c0,d0, a1,b1, c1,d1;
//      e0,f0, g0,h0, e1,f1, g1,h1;--block0, address+64
//                   .
//                   .
//                   .
//      a6,b6, c6,d6, a7,b7, c7,d7;
//      e6,f6, g6,h6, e7,f7, g7,h7;--block3, address+64
// for armv7:
// B block size: (<8x2>x1) x2, can be described as below:
// packed B:
//      a0,b0, c0,d0, e0,f0, g0,h0;
//      a1,b1, c1,d1, e1,f1, g1,h1;--block0, address+32
//                   .
//                   .
//                   .
//      a6,b6, c6,d6, e6,f6, g6,h6;
//      a7,b7, c7,d7, e7,f7, g7,h7;--block3, address+32
/*******************************************************************/
void packb_trans_int8(int8_t* out, const int8_t* in, const int ldin,
                      const int k0, const int kmax, const int n0,
                      const int nmax, const int8_t* zerobuf) {
  const int KUNROLL = 4;
  const int NUNROLL = 8;
  const int RATIO = NBLOCK_INT8 / NUNROLL;
  const int8_t* inptr = in + n0 * ldin + k0;
  const uint8_t mask_buffer[16] = {0, 1, 2,  3,  4,  5,  6,  7,
                                   8, 9, 10, 11, 12, 13, 14, 15};
  int y_len = nmax - n0;
  int x_len = kmax - k0;
  int yup = ROUNDUP(y_len, NBLOCK_INT8);
  const int kup = ROUNDUP(x_len, KBLOCK_INT8);
  const int KSTRIDE = KBLOCK_INT8 * KUNROLL;
  int kcnt = x_len / KSTRIDE;
  int x_rem = (x_len & (KSTRIDE - 1));
  int k_rem = (x_rem + KBLOCK_INT8 - 1) / KBLOCK_INT8;
  const int stride_inner = KBLOCK_INT8 * NUNROLL;
  const int stride_outer = kup * NBLOCK_INT8;
  const int ncnt = yup / NUNROLL;

  int8x16_t vzero = vdupq_n_s8(0);
  uint8x16_t vmask = vcltq_u8(vld1q_u8(mask_buffer), vdupq_n_u8(x_rem));

#pragma omp parallel for
  for (int y = 0; y < ncnt; y++) {
    int idx = y * NUNROLL;
    const int8_t* ptr0 = inptr + idx * ldin;
    const int8_t* ptr1 = ptr0 + ldin;
    const int8_t* ptr2 = ptr1 + ldin;
    const int8_t* ptr3 = ptr2 + ldin;
    const int8_t* ptr4 = ptr3 + ldin;
    const int8_t* ptr5 = ptr4 + ldin;
    const int8_t* ptr6 = ptr5 + ldin;
    const int8_t* ptr7 = ptr6 + ldin;
    // only for ratio = 0 or 1
    int8_t* ptr_out =
        out + (y & (RATIO - 1)) * stride_inner + (y / RATIO) * stride_outer;
    if (idx + NUNROLL > y_len) {
      switch (idx + NUNROLL - y_len) {
        case 8:
          ptr0 = zerobuf;
        case 7:
          ptr1 = zerobuf;
        case 6:
          ptr2 = zerobuf;
        case 5:
          ptr3 = zerobuf;
        case 4:
          ptr4 = zerobuf;
        case 3:
          ptr5 = zerobuf;
        case 2:
          ptr6 = zerobuf;
        case 1:
          ptr7 = zerobuf;
        default:
          break;
      }
    }
    int k = kcnt;
    int rem = k_rem;
#ifdef __aarch64__
    asm volatile(
        "cbz    %w[k], 1f\n" /* skip  main loop */
        /* main loop */
        "0:\n"                              /* main loop */
        "ld1    {v0.16b}, [%[ptr0]], #16\n" /* load n0, k0~k15 */
        "ld1    {v1.16b}, [%[ptr1]], #16\n" /* load n1, k0~k15 */
        "ld1    {v2.16b}, [%[ptr2]], #16\n" /* load n2, k0~k15 */
        "ld1    {v3.16b}, [%[ptr3]], #16\n" /* load n3, k0~k15 */
        "ld1    {v4.16b}, [%[ptr4]], #16\n" /* load n4, k0~k15 */
        "ld1    {v5.16b}, [%[ptr5]], #16\n" /* load n5, k0~k15 */
        "ld1    {v6.16b}, [%[ptr6]], #16\n" /* load n6, k0~k15 */
        "ld1    {v7.16b}, [%[ptr7]], #16\n" /* load n7, k0~k15 */
        /* trans, 8h */
        "trn1   v8.8h,  v0.8h,  v1.8h\n" /* trans, zip n0,n1 */
        "trn2   v9.8h,  v0.8h,  v1.8h\n" /* trans, zip n0,n1 */
        "trn1  v10.8h,  v2.8h,  v3.8h\n" /* trans, zip n2,n3 */
        "trn2  v11.8h,  v2.8h,  v3.8h\n" /* trans, zip n2,n3 */
        "trn1  v12.8h,  v4.8h,  v5.8h\n" /* trans, zip n4,n5 */
        "trn2  v13.8h,  v4.8h,  v5.8h\n" /* trans, zip n4,n5 */
        "trn1  v14.8h,  v6.8h,  v7.8h\n" /* trans, zip n6,n7 */
        "trn2  v15.8h,  v6.8h,  v7.8h\n" /* trans, zip n6,n7 */
        /* trans, 4s */
        "trn1  v16.4s,  v8.4s, v10.4s\n" /* trans, block 0 */
        "trn2  v17.4s,  v8.4s, v10.4s\n" /* trans, block 0 */
        "trn1  v18.4s,  v9.4s, v11.4s\n" /* trans, block 0 */
        "trn2  v19.4s,  v9.4s, v11.4s\n" /* trans, block 0 */
        "trn1  v20.4s, v12.4s, v14.4s\n" /* trans, block 1 */
        "trn2  v21.4s, v12.4s, v14.4s\n" /* trans, block 1 */
        "trn1  v22.4s, v13.4s, v15.4s\n" /* trans, block 1 */
        "trn2  v23.4s, v13.4s, v15.4s\n" /* trans, block 1 */
        "subs   %w[k],  %w[k],  #1\n"    /* loop count -1 */
        /* trans, 2d */
        "trn1   v8.2d, v16.2d, v18.2d\n" /* trans, block 0, out0 */
        "trn1   v9.2d, v20.2d, v22.2d\n" /* trans, block 1, out0 */
        "trn1  v10.2d, v17.2d, v19.2d\n" /* trans, block 0, out1 */
        "trn1  v11.2d, v21.2d, v23.2d\n" /* trans, block 1, out1 */
        "trn2  v12.2d, v16.2d, v18.2d\n" /* trans, block 0, out2 */
        "trn2  v13.2d, v20.2d, v22.2d\n" /* trans, block 1, out2 */
        "trn2  v14.2d, v17.2d, v19.2d\n" /* trans, block 0, out3 */
        "trn2  v15.2d, v21.2d, v23.2d\n" /* trans, block 1, out3 */
        /* store result */
        "stp    q8, q9,   [%[ptr_out]],#64\n" /* write 0 */
        "stp  q10, q11,   [%[ptr_out]],#64\n" /* write 1 */
        "stp  q12, q13,   [%[ptr_out]],#64\n" /* write 2 */
        "stp  q14, q15,   [%[ptr_out]],#64\n" /* write 3 */
        "bgt    0b\n"                         /* jump to main loop */
        /* process remain */
        "1:\n"                         /* process remains */
        "cbz    %w[rem], 2f\n"         /* no remain, jump to end */
        "ld1    {v0.16b}, [%[ptr0]]\n" /* load n0, k0~k15 */
        "ld1    {v1.16b}, [%[ptr1]]\n" /* load n1, k0~k15 */
        "ld1    {v2.16b}, [%[ptr2]]\n" /* load n2, k0~k15 */
        "ld1    {v3.16b}, [%[ptr3]]\n" /* load n3, k0~k15 */
        "ld1    {v4.16b}, [%[ptr4]]\n" /* load n4, k0~k15 */
        "ld1    {v5.16b}, [%[ptr5]]\n" /* load n5, k0~k15 */
        "ld1    {v6.16b}, [%[ptr6]]\n" /* load n6, k0~k15 */
        "ld1    {v7.16b}, [%[ptr7]]\n" /* load n7, k0~k15 */
        /* bit select */
        "bif    v0.16b, %[vzero].16b, %[mask].16b\n" /* pad 0 */
        "bif    v1.16b, %[vzero].16b, %[mask].16b\n" /* pad 0 */
        "bif    v2.16b, %[vzero].16b, %[mask].16b\n" /* pad 0 */
        "bif    v3.16b, %[vzero].16b, %[mask].16b\n" /* pad 0 */
        "bif    v4.16b, %[vzero].16b, %[mask].16b\n" /* pad 0 */
        "bif    v5.16b, %[vzero].16b, %[mask].16b\n" /* pad 0 */
        "bif    v6.16b, %[vzero].16b, %[mask].16b\n" /* pad 0 */
        "bif    v7.16b, %[vzero].16b, %[mask].16b\n" /* pad 0 */
        /* trans, 8h */
        "trn1   v8.8h,  v0.8h,  v1.8h\n" /* trans, zip n0,n1 */
        "trn2   v9.8h,  v0.8h,  v1.8h\n" /* trans, zip n0,n1 */
        "trn1  v10.8h,  v2.8h,  v3.8h\n" /* trans, zip n2,n3 */
        "trn2  v11.8h,  v2.8h,  v3.8h\n" /* trans, zip n2,n3 */
        "trn1  v12.8h,  v4.8h,  v5.8h\n" /* trans, zip n4,n5 */
        "trn2  v13.8h,  v4.8h,  v5.8h\n" /* trans, zip n4,n5 */
        "trn1  v14.8h,  v6.8h,  v7.8h\n" /* trans, zip n6,n7 */
        "trn2  v15.8h,  v6.8h,  v7.8h\n" /* trans, zip n6,n7 */
        /* trans, 4s */
        "trn1  v16.4s,  v8.4s, v10.4s\n" /* trans, block 0 */
        "trn2  v17.4s,  v8.4s, v10.4s\n" /* trans, block 0 */
        "trn1  v18.4s,  v9.4s, v11.4s\n" /* trans, block 0 */
        "trn2  v19.4s,  v9.4s, v11.4s\n" /* trans, block 0 */
        "trn1  v20.4s, v12.4s, v14.4s\n" /* trans, block 1 */
        "trn2  v21.4s, v12.4s, v14.4s\n" /* trans, block 1 */
        "trn1  v22.4s, v13.4s, v15.4s\n" /* trans, block 1 */
        "trn2  v23.4s, v13.4s, v15.4s\n" /* trans, block 1 */
        /* trans, 2d */
        "trn1   v8.2d, v16.2d, v18.2d\n" /* trans, block 0, out0 */
        "trn1   v9.2d, v20.2d, v22.2d\n" /* trans, block 1, out0 */
        "trn1  v10.2d, v17.2d, v19.2d\n" /* trans, block 0, out1 */
        "trn1  v11.2d, v21.2d, v23.2d\n" /* trans, block 1, out1 */
        "trn2  v12.2d, v16.2d, v18.2d\n" /* trans, block 0, out2 */
        "trn2  v13.2d, v20.2d, v22.2d\n" /* trans, block 1, out2 */
        "trn2  v14.2d, v17.2d, v19.2d\n" /* trans, block 0, out3 */
        "trn2  v15.2d, v21.2d, v23.2d\n" /* trans, block 1, out3 */
        /* check remain size */
        "subs    %w[rem], %w[rem], #1\n"      /* check remain num */
        "stp    q8, q9,   [%[ptr_out]],#64\n" /* write 0 */
        "beq    2f\n"                         /* remain = 1 */
        "subs    %w[rem], %w[rem], #1\n"      /* check remain num */
        "stp  q10, q11,   [%[ptr_out]],#64\n" /* write 1 */
        "beq    2f\n"                         /* remain = 2 */
        "subs    %w[rem], %w[rem], #1\n"      /* check remain num */
        "stp  q12, q13,   [%[ptr_out]],#64\n" /* write 2 */
        "beq    2f\n"                         /* remain = 3 */
        "stp  q14, q15,   [%[ptr_out]]\n"     /* write 3 */
        /* end */
        "2:\n" /* end */
        : [ptr0] "+r"(ptr0), [ptr1] "+r"(ptr1), [ptr2] "+r"(ptr2),
          [ptr3] "+r"(ptr3), [ptr4] "+r"(ptr4), [ptr5] "+r"(ptr5),
          [ptr6] "+r"(ptr6), [ptr7] "+r"(ptr7), [ptr_out] "+r"(ptr_out),
          [k] "+r"(k), [rem] "+r"(rem)
        : [mask] "w"(vmask), [vzero] "w"(vzero)
        : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10",
          "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19", "v20",
          "v21", "v22", "v23", "cc");
#else   // armv7
    asm volatile(
        "cmp    %[k], #0\n" /* check  main loop */
        "beq    1f\n"       /* skip  main loop */
        /* main loop */
        "0:\n"                           /* main loop */
        "vld1.8 {d0-d1}, [%[ptr0]]!\n"   /* load n0, a0~a7 */
        "vld1.8 {d2-d3}, [%[ptr1]]!\n"   /* load n1, b0~b7 */
        "vld1.8 {d4-d5}, [%[ptr2]]!\n"   /* load n2, c0~c7 */
        "vld1.8 {d6-d7}, [%[ptr3]]!\n"   /* load n3, d0~d7 */
        "vld1.8 {d8-d9}, [%[ptr4]]!\n"   /* load n4, e0~e7 */
        "vld1.8 {d10-d11}, [%[ptr5]]!\n" /* load n5, f0~f7 */
        "vld1.8 {d12-d13}, [%[ptr6]]!\n" /* load n6, g0~g7 */
        "vld1.8 {d14-d15}, [%[ptr7]]!\n" /* load n7, h0~h7 */
        /* trans, 8h */
        "vtrn.16    q0, q1\n" /* trans, zip n0,n1, q0: a0b0,a2b2, a4b4,a6b6, q1:
                                 a1b1,a3b3, a5b5,a7b7 */
        "vtrn.16    q2, q3\n" /* trans, zip n2,n3, q2: c0d0,c2d2, c4d4,c6d6, q3:
                                 c1d1,c3d3, c5d5,c7d7 */
        "vtrn.16    q4, q5\n" /* trans, zip n4,n5, q4: e0f0,e2f2, e4f4,e6f6, q5:
                                 e1f1,e3f3, e5f5,e7f7 */
        "vtrn.16    q6, q7\n" /* trans, zip n6,n7, q6: g0h0,g2h2, g4h4,g6h6, q7:
                                 g1h1,g3h3, g5h5,g7h7 */
        /* trans, 4s */
        "vtrn.32    q0, q2\n" /* trans, q0: a0b0,c0d0, a4b4,c4d4, q2: a2b2,c2d2,
                                 a6b6,c6d6 */
        "vtrn.32    q1, q3\n" /* trans, q1: a1b1,c1d1, a5b5,c5d5, q3: a3b3,c3d3,
                                 a7b7,c7d7 */
        "vtrn.32    q4, q6\n" /* trans, q4: e0f0,g0h0, e4f4,g4h4, q6: e2f2,g2h2,
                                 e6f6,g6h6 */
        "vtrn.32    q5, q7\n" /* trans, q5: e1f1,g1h1, e5f5,g5h5, q7: e3f3,g3h3,
                                 e7f7,g7h7 */
        "subs   %[k],  %[k],  #1\n" /* loop count -1 */
        /* trans, 2d */
        "vswp   d1, d8\n"  /* q0: a0b0,c0d0, e0f0,g0h0, q4: a4b4,c4d4, e4f4,g4h4
                              */
        "vswp   d3, d10\n" /* q1: a1b1,c1d1, e1f1,g1h1, q5: a5b5,c5d5, e5f5,g5h5
                              */
        "vswp   d5, d12\n" /* q2: a2b2,c2d2, e2f2,g2h2, q6: a6b6,c6d6, e6f6,g6h6
                              */
        "vswp   d7, d14\n" /* q3: a3b3,c3d3, e3f3,g3h3, q7: a7b7,c7d7, e7f7,g7h7
                              */
        /* store result */
        "vst1.8 {d0-d3},    [%[ptr_out]]!\n" /* write 0 */
        "vst1.8 {d4-d7},    [%[ptr_out]]!\n" /* write 1 */
        "vst1.8 {d8-d11},   [%[ptr_out]]!\n" /* write 2 */
        "vst1.8 {d12-d15},  [%[ptr_out]]!\n" /* write 3 */
        "bgt    0b\n"                        /* jump to main loop */
        /* process remain */
        "1:\n"                           /* process remains */
        "cmp    %[rem], #0\n"            /* check remain */
        "beq    2f\n"                    /* no remain, jump to end */
        "vld1.8 {d0-d1}, [%[ptr0]]!\n"   /* load n0, a0~a7 */
        "vld1.8 {d2-d3}, [%[ptr1]]!\n"   /* load n1, b0~b7 */
        "vld1.8 {d4-d5}, [%[ptr2]]!\n"   /* load n2, c0~c7 */
        "vld1.8 {d6-d7}, [%[ptr3]]!\n"   /* load n3, d0~d7 */
        "vld1.8 {d8-d9}, [%[ptr4]]!\n"   /* load n4, e0~e7 */
        "vld1.8 {d10-d11}, [%[ptr5]]!\n" /* load n5, f0~f7 */
        "vld1.8 {d12-d13}, [%[ptr6]]!\n" /* load n6, g0~g7 */
        "vld1.8 {d14-d15}, [%[ptr7]]!\n" /* load n7, h0~h7 */
        /* bit select */
        "vbif   q0, %q[vzero], %q[mask]\n" /* pad 0 */
        "vbif   q1, %q[vzero], %q[mask]\n" /* pad 0 */
        "vbif   q2, %q[vzero], %q[mask]\n" /* pad 0 */
        "vbif   q3, %q[vzero], %q[mask]\n" /* pad 0 */
        "vbif   q4, %q[vzero], %q[mask]\n" /* pad 0 */
        "vbif   q5, %q[vzero], %q[mask]\n" /* pad 0 */
        "vbif   q6, %q[vzero], %q[mask]\n" /* pad 0 */
        "vbif   q7, %q[vzero], %q[mask]\n" /* pad 0 */
        /* trans, 8h */
        "vtrn.16    q0, q1\n" /* trans, zip n0,n1, q0: a0b0,a2b2, a4b4,a6b6, q1:
                                 a1b1,a3b3, a5b5,a7b7 */
        "vtrn.16    q2, q3\n" /* trans, zip n2,n3, q2: c0d0,c2d2, c4d4,c6d6, q3:
                                 c1d1,c3d3, c5d5,c7d7 */
        "vtrn.16    q4, q5\n" /* trans, zip n4,n5, q4: e0f0,e2f2, e4f4,e6f6, q5:
                                 e1f1,e3f3, e5f5,e7f7 */
        "vtrn.16    q6, q7\n" /* trans, zip n6,n7, q6: g0h0,g2h2, g4h4,g6h6, q7:
                                 g1h1,g3h3, g5h5,g7h7 */
        /* trans, 4s */
        "vtrn.32    q0, q2\n" /* trans, q0: a0b0,c0d0, a4b4,c4d4, q2: a2b2,c2d2,
                                 a6b6,c6d6 */
        "vtrn.32    q1, q3\n" /* trans, q1: a1b1,c1d1, a5b5,c5d5, q3: a3b3,c3d3,
                                 a7b7,c7d7 */
        "vtrn.32    q4, q6\n" /* trans, q4: e0f0,g0h0, e4f4,g4h4, q6: e2f2,g2h2,
                                 e6f6,g6h6 */
        "vtrn.32    q5, q7\n" /* trans, q5: e1f1,g1h1, e5f5,g5h5, q7: e3f3,g3h3,
                                 e7f7,g7h7 */
        /* trans, 2d */
        "vswp   d1, d8\n"  /* q0: a0b0,c0d0, e0f0,g0h0, q4: a4b4,c4d4, e4f4,g4h4
                              */
        "vswp   d3, d10\n" /* q1: a1b1,c1d1, e1f1,g1h1, q5: a5b5,c5d5, e5f5,g5h5
                              */
        "vswp   d5, d12\n" /* q2: a2b2,c2d2, e2f2,g2h2, q6: a6b6,c6d6, e6f6,g6h6
                              */
        "vswp   d7, d14\n" /* q3: a3b3,c3d3, e3f3,g3h3, q7: a7b7,c7d7, e7f7,g7h7
                              */
        /* check remain size */
        "subs    %[rem], %[rem], #1\n"       /* check remain num */
        "vst1.8 {d0-d3},    [%[ptr_out]]!\n" /* write 0 */
        "beq    2f\n"                        /* remain = 1 */
        "subs    %[rem], %[rem], #1\n"       /* check remain num */
        "vst1.8 {d4-d7},    [%[ptr_out]]!\n" /* write 1 */
        "beq    2f\n"                        /* remain = 2 */
        "subs    %[rem], %[rem], #1\n"       /* check remain num */
        "vst1.8 {d8-d11},   [%[ptr_out]]!\n" /* write 2 */
        "beq    2f\n"                        /* remain = 3 */
        "vst1.8 {d12-d15},  [%[ptr_out]]!\n" /* write 3 */
        /* end */
        "2:\n" /* end */
        : [ptr0] "+r"(ptr0), [ptr1] "+r"(ptr1), [ptr2] "+r"(ptr2),
          [ptr3] "+r"(ptr3), [ptr4] "+r"(ptr4), [ptr5] "+r"(ptr5),
          [ptr6] "+r"(ptr6), [ptr7] "+r"(ptr7), [ptr_out] "+r"(ptr_out),
          [k] "+r"(k), [rem] "+r"(rem)
        : [mask] "w"(vmask), [vzero] "w"(vzero)
        : "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7", "cc");
#endif  //__aarch64__  // NOLINT
  }
}

}  // namespace math
}  // namespace arm
}  // namespace lite
}  // namespace paddle
