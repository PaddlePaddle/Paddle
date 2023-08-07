/* Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#pragma once

#include <mutex>  // NOLINT

#include "cuda_runtime.h"  // NOLINT
#include "paddle/phi/backends/dynload/dynamic_loader.h"
#include "paddle/phi/backends/dynload/port.h"

namespace phi {
namespace dynload {

extern std::once_flag flashattn_v1_dso_flag;
extern void *flashattn_v1_dso_handle;

using flash_attn_fwd_v1_func_t = bool (*)(
    const void * /*q*/,  // total_q x num_heads x head_size, total_q :=
                         // \sum_{i=0}^{b} s_i
    const void * /*k*/,  // total_k x num_heads x head_size, total_k :=
                         // \sum_{i=0}^{b} s_i
    const void * /*v*/,  // total_k x num_heads x head_size, total_k :=
                         // \sum_{i=0}^{b} s_i
    void * /*out*/,      // total_q x num_heads x head_size, total_k :=
                         // \sum_{i=0}^{b} s_i
    const void * /*cu_seqlens_q*/,  // int32, batch_size+1, starting offset of
                                    // each sequence
    const void * /*cu_seqlens_k*/,  // int32, batch_size+1, starting offset of
                                    // each sequence
    const int /*total_q*/,
    const int /*total_k*/,
    const int /*batch_size*/,
    const int /*num_heads*/,
    const int /*head_size*/,
    const int /*max_seqlen_q_*/,
    const int /*max_seqlen_k_*/,
    const float /*p_dropout*/,
    const float /*softmax_scale*/,
    const bool /*zero_tensors*/,
    const bool /*is_causal*/,
    const bool /*is_bf16*/,
    const int /*num_splits*/,    // SMs per attention matrix, can be 1
    void * /*softmax_lse_ptr*/,  // softmax log_sum_exp
    void * /*softmax_ptr*/,
    void * /*workspace_ptr*/,
    uint64_t * /*workspace_size*/,
    cudaStream_t /*stream*/,
    uint64_t /*seed*/,
    uint64_t /*offset*/
);

using flash_attn_bwd_v1_func_t = bool (*)(
    const void * /*q*/,     // total_q x num_heads x head_size, total_q :=
                            // \sum_{i=0}^{b} s_i
    const void * /*k*/,     // total_k x num_heads x head_size, total_k :=
                            // \sum_{i=0}^{b} s_i
    const void * /*v*/,     // total_k x num_heads x head_size, total_k :=
                            // \sum_{i=0}^{b} s_i
    void * /*dq*/,          // total_q x num_heads x head_size, total_q :=
                            // \sum_{i=0}^{b} s_i
    void * /*dk*/,          // total_k x num_heads x head_size, total_k :=
                            // \sum_{i=0}^{b} s_i
    void * /*dv*/,          // total_k x num_heads x head_size, total_k :=
                            // \sum_{i=0}^{b} s_i
    const void * /*out*/,   // total_q x num_heads x head_size, total_k :=
                            // \sum_{i=0}^{b} s_i
    const void * /*dout*/,  // total_q x num_heads, x head_size
    const void * /*cu_seqlens_q*/,  // int32, batch_size+1
    const void * /*cu_seqlens_k*/,  // int32, batch_size+1
    const int /*total_q*/,
    const int /*total_k*/,
    const int /*batch_size*/,
    const int /*num_heads*/,
    const int /*head_size*/,
    const int /*max_seqlen_q_*/,
    const int /*max_seqlen_k_*/,
    const float /*p_dropout*/,
    const float /*softmax_scale*/,
    const bool /*zero_tensors*/,
    const bool /*is_causal*/,
    const bool /*is_bf16*/,
    const int /*num_splits*/,
    void * /*softmax_lse_ptr*/,
    void * /*dsoftmax_ptr*/,
    void * /*workspace_ptr*/,
    uint64_t * /*workspace_size*/,
    cudaStream_t /*stream*/,
    uint64_t /*seed*/,
    uint64_t /*offset*/
);

using flash_attn_error_v1_func_t = const char *(*)();

#define DYNAMIC_LOAD_FLASHATTN_V1_WRAP(__name)                             \
  struct DynLoad__##__name##__v1 {                                         \
    template <typename... Args>                                            \
    auto operator()(Args... args) {                                        \
      using flashattnFunc = ::phi::dynload::__name##_v1_func_t;            \
      std::call_once(flashattn_v1_dso_flag, []() {                         \
        flashattn_v1_dso_handle = phi::dynload::GetFlashAttnV1DsoHandle(); \
      });                                                                  \
      static void *p_##__name = dlsym(flashattn_v1_dso_handle, #__name);   \
      return reinterpret_cast<flashattnFunc>(p_##__name)(args...);         \
    }                                                                      \
  };                                                                       \
  extern DynLoad__##__name##__v1 __name##_v1

#define DECLARE_DYNAMIC_LOAD_FLASHATTN_V1_WRAP(__name) \
  DYNAMIC_LOAD_FLASHATTN_V1_WRAP(__name)

#define FLASHATTN_V1_ROUTINE_EACH(__macro) \
  __macro(flash_attn_fwd);                 \
  __macro(flash_attn_bwd);                 \
  __macro(flash_attn_error);

FLASHATTN_V1_ROUTINE_EACH(DECLARE_DYNAMIC_LOAD_FLASHATTN_V1_WRAP);

#undef DYNAMIC_LOAD_FLASHATTN_V1_WRAP

}  // namespace dynload
}  // namespace phi
