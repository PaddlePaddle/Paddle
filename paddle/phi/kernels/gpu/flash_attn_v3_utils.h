// Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

#ifdef PADDLE_WITH_FLASHATTN_V3
#include "paddle/phi/backends/dynload/flashattnv3.h"
#include "paddle/phi/backends/dynload/flashmaskv2.h"
#endif
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/platform/device_context.h"

namespace phi {
#ifdef PADDLE_WITH_FLASHATTN_V3

#define CHECK_DEVICE(x)         \
  PADDLE_ENFORCE_EQ(            \
      x.place().GetType(),      \
      phi::AllocationType::GPU, \
      common::errors::InvalidArgument(#x " must be on CUDA Device"))

#define CHECK_SHAPE(x, ...)                           \
  PADDLE_ENFORCE_EQ(x.dims(),                         \
                    common::make_ddim({__VA_ARGS__}), \
                    common::errors::InvalidArgument(  \
                        #x " must have shape (" #__VA_ARGS__ ")"))

#define CHECK_CONTIGUOUS(x)                   \
  PADDLE_ENFORCE_EQ(x.meta().is_contiguous(), \
                    true,                     \
                    common::errors::InvalidArgument(#x " must be contiguous"))

Flash_fwd_params *get_flash_fwd_params_handle();

Flash_bwd_params *get_flash_bwd_params_handle();

FlashMask_fwd_params *get_flashmask_fwd_params_handle();

FlashMask_bwd_params *get_flashmask_bwd_params_handle();

inline int get_max_headdim() {
#ifndef FLASHATTENTION_DISABLE_HDIM256
  return 256;
#endif
#ifndef FLASHATTENTION_DISABLE_HDIM192
  return 192;
#endif
#ifndef FLASHATTENTION_DISABLE_HDIM128
  return 128;
#endif
#ifndef FLASHATTENTION_DISABLE_HDIM96
  return 96;
#endif
#ifndef FLASHATTENTION_DISABLE_HDIM64
  return 64;
#endif
  return 0;
}

inline int round_up_headdim(int head_size) {
#ifndef FLASHATTENTION_DISABLE_HDIM64
  if (head_size <= 64) {
    return 64;
  }
#endif
#ifndef FLASHATTENTION_DISABLE_HDIM96
  if (head_size <= 96) {
    return 96;
  }
#endif
#ifndef FLASHATTENTION_DISABLE_HDIM128
  if (head_size <= 128) {
    return 128;
  }
#endif
#ifndef FLASHATTENTION_DISABLE_HDIM192
  if (head_size <= 192) {
    return 192;
  }
#endif
#ifndef FLASHATTENTION_DISABLE_HDIM256
  if (head_size <= 256) {
    return 256;
  }
#endif
  return 256;
}

void set_params_fprop(Flash_fwd_params *params_handle,
                      // sizes
                      const size_t b,
                      const size_t seqlen_q,
                      const size_t seqlen_k,
                      const size_t seqlen_q_rounded,
                      const size_t seqlen_k_rounded,
                      const size_t h,
                      const size_t h_k,
                      const size_t d,
                      const size_t d_rounded,
                      // device pointers
                      const DenseTensor &q,
                      const DenseTensor &k,
                      const DenseTensor &v,
                      const DenseTensor *out,
                      void *cu_seqlens_q_d,
                      void *cu_seqlens_k_d,
                      void *seqused_q,
                      void *seqused_k,
                      void *softmax_lse_d,
                      float p_dropout,
                      float softmax_scale,
                      int window_size_left,
                      int window_size_right,
                      const gpuDeviceProp &dprops,
                      const float softcap = 0.f,
                      const int sm_margin = 0);

void set_params_dgrad(Flash_bwd_params *params_handle,
                      // sizes
                      const size_t b,
                      const size_t seqlen_q,
                      const size_t seqlen_k,
                      const size_t seqlen_q_rounded,
                      const size_t seqlen_k_rounded,
                      const size_t h,
                      const size_t h_k,
                      const size_t d,
                      const size_t d_rounded,
                      // device pointers
                      const DenseTensor &q,
                      const DenseTensor &k,
                      const DenseTensor &v,
                      const DenseTensor &out,
                      const DenseTensor &dout,
                      DenseTensor *dq,
                      DenseTensor *dk,
                      DenseTensor *dv,
                      void *cu_seqlens_q_d,
                      void *cu_seqlens_k_d,
                      void *seqused_q,
                      void *seqused_k,
                      void *dq_accum_d,
                      void *dk_accum_d,
                      void *dv_accum_d,
                      void *softmax_lse_d,
                      void *dsoftmax_sum_d,
                      float p_dropout,
                      float softmax_scale,
                      int window_size_left,
                      int window_size_right,
                      const gpuDeviceProp &dprops,
                      const float softcap = 0.f,
                      bool deterministic = false,
                      int const sm_margin = 0);

void set_flashmaskv2_params_fprop(Flash_fwd_params *params_handle,
                                  // sizes
                                  const size_t b,
                                  const size_t seqlen_q,
                                  const size_t seqlen_k,
                                  const size_t seqlen_q_rounded,
                                  const size_t seqlen_k_rounded,
                                  const size_t h,
                                  const size_t h_k,
                                  const size_t d,
                                  const size_t d_rounded,
                                  // device pointers
                                  const DenseTensor &q,
                                  const DenseTensor &k,
                                  const DenseTensor &v,
                                  const DenseTensor *out,
                                  void *cu_seqlens_q_d,
                                  void *cu_seqlens_k_d,
                                  void *seqused_q,
                                  void *seqused_k,
                                  void *softmax_lse_d,
                                  float p_dropout,
                                  float softmax_scale,
                                  int window_size_left,
                                  int window_size_right,
                                  const gpuDeviceProp &dprops,
                                  const float softcap = 0.f,
                                  const int sm_margin = 0);

void set_flashmaskv2_params_dgrad(Flash_bwd_params *params_handle,
                                  // sizes
                                  const size_t b,
                                  const size_t seqlen_q,
                                  const size_t seqlen_k,
                                  const size_t seqlen_q_rounded,
                                  const size_t seqlen_k_rounded,
                                  const size_t h,
                                  const size_t h_k,
                                  const size_t d,
                                  const size_t d_rounded,
                                  // device pointers
                                  const DenseTensor &q,
                                  const DenseTensor &k,
                                  const DenseTensor &v,
                                  const DenseTensor &out,
                                  const DenseTensor &dout,
                                  DenseTensor *dq,
                                  DenseTensor *dk,
                                  DenseTensor *dv,
                                  void *cu_seqlens_q_d,
                                  void *cu_seqlens_k_d,
                                  void *seqused_q,
                                  void *seqused_k,
                                  void *dq_accum_d,
                                  void *dk_accum_d,
                                  void *dv_accum_d,
                                  void *softmax_lse_d,
                                  void *dsoftmax_sum_d,
                                  float p_dropout,
                                  float softmax_scale,
                                  int window_size_left,
                                  int window_size_right,
                                  const gpuDeviceProp &dprops,
                                  const float softcap = 0.f,
                                  bool deterministic = false,
                                  int const sm_margin = 0);
#endif

}  // namespace phi
