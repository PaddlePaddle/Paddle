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

#include "flashattn/include/flashv3_api.h"
#include "paddle/phi/backends/dynload/dynamic_loader.h"
#include "paddle/phi/common/port.h"

namespace phi {
namespace dynload {

extern std::once_flag flashattnv3_dso_flag;
extern void *flashattnv3_dso_handle;

#define DYNAMIC_LOAD_FLASHATTN_V3_WRAP(__name)                            \
  struct DynLoad__##__name {                                              \
    template <typename... Args>                                           \
    auto operator()(Args... args) -> DECLARE_TYPE(__name, args...) {      \
      using flashattnFunc = decltype(&::__name);                          \
      std::call_once(flashattnv3_dso_flag, []() {                         \
        flashattnv3_dso_handle = phi::dynload::GetFlashAttnV3DsoHandle(); \
      });                                                                 \
      static void *p_##__name = dlsym(flashattnv3_dso_handle, #__name);   \
      return reinterpret_cast<flashattnFunc>(p_##__name)(args...);        \
    }                                                                     \
  };                                                                      \
  extern DynLoad__##__name __name

#define DECLARE_DYNAMIC_LOAD_FLASHATTN_V3_WRAP(__name) \
  DYNAMIC_LOAD_FLASHATTN_V3_WRAP(__name)

#ifdef PADDLE_WITH_CUDA
#define FLASHATTN_V3_ROUTINE_EACH(__macro) \
  __macro(fa3_create_fwd_params_handle);   \
  __macro(fa3_clear_fwd_params_handle);    \
  __macro(fa3_destroy_fwd_params_handle);  \
  __macro(fa3_create_bwd_params_handle);   \
  __macro(fa3_clear_bwd_params_handle);    \
  __macro(fa3_destroy_bwd_params_handle);  \
  __macro(fa3_cast_to_fwd_params_handle);  \
  __macro(fa3_run_mha_fwd_combine);        \
  __macro(fa3_run_mha_fwd);                \
  __macro(fa3_run_mha_bwd);                \
  __macro(fa3_get_pagedkv_tma);            \
  __macro(fa3_get_pack_gqa);               \
  __macro(fa3_get_num_splits);

FLASHATTN_V3_ROUTINE_EACH(DECLARE_DYNAMIC_LOAD_FLASHATTN_V3_WRAP)

#define FLASHATTN_V3_HANDLE_ROUTINE(member)                            \
  DECLARE_DYNAMIC_LOAD_FLASHATTN_V3_WRAP(fa3_fwd_params_get_##member); \
  DECLARE_DYNAMIC_LOAD_FLASHATTN_V3_WRAP(fa3_fwd_params_set_##member); \
  DECLARE_DYNAMIC_LOAD_FLASHATTN_V3_WRAP(fa3_bwd_params_get_##member); \
  DECLARE_DYNAMIC_LOAD_FLASHATTN_V3_WRAP(fa3_bwd_params_set_##member);

// The QKV matrices.
FLASHATTN_V3_HANDLE_ROUTINE(q_ptr)
FLASHATTN_V3_HANDLE_ROUTINE(k_ptr)
FLASHATTN_V3_HANDLE_ROUTINE(v_ptr)

// The stride between rows of the Q, K and V matrices.
FLASHATTN_V3_HANDLE_ROUTINE(q_batch_stride)
FLASHATTN_V3_HANDLE_ROUTINE(k_batch_stride)
FLASHATTN_V3_HANDLE_ROUTINE(v_batch_stride)
FLASHATTN_V3_HANDLE_ROUTINE(q_row_stride)
FLASHATTN_V3_HANDLE_ROUTINE(k_row_stride)
FLASHATTN_V3_HANDLE_ROUTINE(v_row_stride)
FLASHATTN_V3_HANDLE_ROUTINE(q_head_stride)
FLASHATTN_V3_HANDLE_ROUTINE(k_head_stride)
FLASHATTN_V3_HANDLE_ROUTINE(v_head_stride)
FLASHATTN_V3_HANDLE_ROUTINE(v_dim_stride)

// The number of heads.
FLASHATTN_V3_HANDLE_ROUTINE(h)
FLASHATTN_V3_HANDLE_ROUTINE(h_k)

// The O matrix (output).
FLASHATTN_V3_HANDLE_ROUTINE(o_ptr)
FLASHATTN_V3_HANDLE_ROUTINE(oaccum_ptr)

// The stride between rows of O.
FLASHATTN_V3_HANDLE_ROUTINE(o_batch_stride)
FLASHATTN_V3_HANDLE_ROUTINE(o_row_stride)
FLASHATTN_V3_HANDLE_ROUTINE(o_head_stride)

// The pointer to the softmax sum.
FLASHATTN_V3_HANDLE_ROUTINE(softmax_lse_ptr)
FLASHATTN_V3_HANDLE_ROUTINE(softmax_lseaccum_ptr)

// For FP8 scaling
FLASHATTN_V3_HANDLE_ROUTINE(q_descale_ptr)
FLASHATTN_V3_HANDLE_ROUTINE(k_descale_ptr)
FLASHATTN_V3_HANDLE_ROUTINE(v_descale_ptr)
FLASHATTN_V3_HANDLE_ROUTINE(q_descale_batch_stride)
FLASHATTN_V3_HANDLE_ROUTINE(q_descale_head_stride)
FLASHATTN_V3_HANDLE_ROUTINE(k_descale_batch_stride)
FLASHATTN_V3_HANDLE_ROUTINE(k_descale_head_stride)
FLASHATTN_V3_HANDLE_ROUTINE(v_descale_batch_stride)
FLASHATTN_V3_HANDLE_ROUTINE(v_descale_head_stride)

// The dimensions.
FLASHATTN_V3_HANDLE_ROUTINE(b)
FLASHATTN_V3_HANDLE_ROUTINE(seqlen_q)
FLASHATTN_V3_HANDLE_ROUTINE(seqlen_k)
FLASHATTN_V3_HANDLE_ROUTINE(seqlen_knew)
FLASHATTN_V3_HANDLE_ROUTINE(d)
FLASHATTN_V3_HANDLE_ROUTINE(seqlen_q_rounded)
FLASHATTN_V3_HANDLE_ROUTINE(seqlen_k_rounded)
FLASHATTN_V3_HANDLE_ROUTINE(d_rounded)
FLASHATTN_V3_HANDLE_ROUTINE(rotary_dim)
FLASHATTN_V3_HANDLE_ROUTINE(total_q)
FLASHATTN_V3_HANDLE_ROUTINE(total_k)
FLASHATTN_V3_HANDLE_ROUTINE(total_knew)
FLASHATTN_V3_HANDLE_ROUTINE(b_k)
FLASHATTN_V3_HANDLE_ROUTINE(dv)
FLASHATTN_V3_HANDLE_ROUTINE(dv_rounded)

// The scaling factors for the kernel.
FLASHATTN_V3_HANDLE_ROUTINE(scale_softmax)
FLASHATTN_V3_HANDLE_ROUTINE(softcap)

// array of length b+1 holding starting offset of each sequence.
FLASHATTN_V3_HANDLE_ROUTINE(cu_seqlens_q)
FLASHATTN_V3_HANDLE_ROUTINE(cu_seqlens_k)
FLASHATTN_V3_HANDLE_ROUTINE(cu_seqlens_knew)
FLASHATTN_V3_HANDLE_ROUTINE(leftpad_k)

// If provided, the actual length of each q/k sequence.
FLASHATTN_V3_HANDLE_ROUTINE(seqused_q)
FLASHATTN_V3_HANDLE_ROUTINE(seqused_k)

// The stride between rows of Oaccum.
FLASHATTN_V3_HANDLE_ROUTINE(oaccum_split_stride)
FLASHATTN_V3_HANDLE_ROUTINE(oaccum_batch_stride)
FLASHATTN_V3_HANDLE_ROUTINE(oaccum_row_stride)
FLASHATTN_V3_HANDLE_ROUTINE(oaccum_head_stride)

// The stride between rows of LSEaccum.
FLASHATTN_V3_HANDLE_ROUTINE(lseaccum_split_stride)
FLASHATTN_V3_HANDLE_ROUTINE(lseaccum_batch_stride)
FLASHATTN_V3_HANDLE_ROUTINE(lseaccum_head_stride)

// The K_new and V_new matrices.
FLASHATTN_V3_HANDLE_ROUTINE(knew_ptr)
FLASHATTN_V3_HANDLE_ROUTINE(vnew_ptr)

// The stride between rows of the Q, K and V matrices.
FLASHATTN_V3_HANDLE_ROUTINE(knew_batch_stride)
FLASHATTN_V3_HANDLE_ROUTINE(vnew_batch_stride)
FLASHATTN_V3_HANDLE_ROUTINE(knew_row_stride)
FLASHATTN_V3_HANDLE_ROUTINE(vnew_row_stride)
FLASHATTN_V3_HANDLE_ROUTINE(knew_head_stride)
FLASHATTN_V3_HANDLE_ROUTINE(vnew_head_stride)

FLASHATTN_V3_HANDLE_ROUTINE(qv_ptr)
FLASHATTN_V3_HANDLE_ROUTINE(qv_batch_stride)
FLASHATTN_V3_HANDLE_ROUTINE(qv_row_stride)
FLASHATTN_V3_HANDLE_ROUTINE(qv_head_stride)

// The cos and sin matrices for rotary embedding.
FLASHATTN_V3_HANDLE_ROUTINE(rotary_cos_ptr)
FLASHATTN_V3_HANDLE_ROUTINE(rotary_sin_ptr)

// The indices to index into the KV cache.
FLASHATTN_V3_HANDLE_ROUTINE(kv_batch_idx)

// Paged KV cache
FLASHATTN_V3_HANDLE_ROUTINE(page_table)
FLASHATTN_V3_HANDLE_ROUTINE(page_table_batch_stride)
FLASHATTN_V3_HANDLE_ROUTINE(page_size)
FLASHATTN_V3_HANDLE_ROUTINE(num_pages)
FLASHATTN_V3_HANDLE_ROUTINE(pagedkv_tma)

// The dropout probability (probability of keeping an activation).
FLASHATTN_V3_HANDLE_ROUTINE(p_dropout)
FLASHATTN_V3_HANDLE_ROUTINE(p_dropout_in_uint8_t)

// Scale factor of 1 / (1 - p_dropout).
FLASHATTN_V3_HANDLE_ROUTINE(rp_dropout)

// Local window size
FLASHATTN_V3_HANDLE_ROUTINE(window_size_left)
FLASHATTN_V3_HANDLE_ROUTINE(window_size_right)

// Pointer to the RNG seed (idx 0) and offset (idx 1).
FLASHATTN_V3_HANDLE_ROUTINE(rng_state)

FLASHATTN_V3_HANDLE_ROUTINE(is_bf16)
FLASHATTN_V3_HANDLE_ROUTINE(is_fp32)
FLASHATTN_V3_HANDLE_ROUTINE(is_e4m3)
FLASHATTN_V3_HANDLE_ROUTINE(is_causal)
FLASHATTN_V3_HANDLE_ROUTINE(is_local)

FLASHATTN_V3_HANDLE_ROUTINE(is_rotary_interleaved)

FLASHATTN_V3_HANDLE_ROUTINE(num_splits)  // For split-KV version
FLASHATTN_V3_HANDLE_ROUTINE(pack_gqa)

FLASHATTN_V3_HANDLE_ROUTINE(tile_count_semaphore)
FLASHATTN_V3_HANDLE_ROUTINE(num_splits_dynamic_ptr)
FLASHATTN_V3_HANDLE_ROUTINE(skip_scheduler_metadata_computation)

FLASHATTN_V3_HANDLE_ROUTINE(arch)
FLASHATTN_V3_HANDLE_ROUTINE(num_sm)

#define FLASHATTN_V3_BWD_HANDLE_ROUTINE(type, member)                  \
  DECLARE_DYNAMIC_LOAD_FLASHATTN_V3_WRAP(fa3_bwd_params_get_##member); \
  DECLARE_DYNAMIC_LOAD_FLASHATTN_V3_WRAP(fa3_bwd_params_set_##member);

// The dO and dQKV matrices.
FLASHATTN_V3_BWD_HANDLE_ROUTINE(void *, do_ptr)
FLASHATTN_V3_BWD_HANDLE_ROUTINE(void *, dq_ptr)
FLASHATTN_V3_BWD_HANDLE_ROUTINE(void *, dk_ptr)
FLASHATTN_V3_BWD_HANDLE_ROUTINE(void *, dv_ptr)

// To accumulate dQ
FLASHATTN_V3_BWD_HANDLE_ROUTINE(void *, dq_accum_ptr)
FLASHATTN_V3_BWD_HANDLE_ROUTINE(void *, dk_accum_ptr)
FLASHATTN_V3_BWD_HANDLE_ROUTINE(void *, dv_accum_ptr)

// // To accumulate dK and dV in case we're splitting the bwd along seqlen_q
// dimension void *__restrict__ dk_accum_ptr; void *__restrict__
// dv_accum_ptr;

// The stride between rows of the dO, dQ, dK and dV matrices.
FLASHATTN_V3_BWD_HANDLE_ROUTINE(int64_t, do_batch_stride)
FLASHATTN_V3_BWD_HANDLE_ROUTINE(int64_t, do_row_stride)
FLASHATTN_V3_BWD_HANDLE_ROUTINE(int64_t, do_head_stride)
FLASHATTN_V3_BWD_HANDLE_ROUTINE(int64_t, dq_batch_stride)
FLASHATTN_V3_BWD_HANDLE_ROUTINE(int64_t, dk_batch_stride)
FLASHATTN_V3_BWD_HANDLE_ROUTINE(int64_t, dv_batch_stride)
FLASHATTN_V3_BWD_HANDLE_ROUTINE(int64_t, dq_row_stride)
FLASHATTN_V3_BWD_HANDLE_ROUTINE(int64_t, dk_row_stride)
FLASHATTN_V3_BWD_HANDLE_ROUTINE(int64_t, dv_row_stride)
FLASHATTN_V3_BWD_HANDLE_ROUTINE(int64_t, dq_head_stride)
FLASHATTN_V3_BWD_HANDLE_ROUTINE(int64_t, dk_head_stride)
FLASHATTN_V3_BWD_HANDLE_ROUTINE(int64_t, dv_head_stride)

// The pointer to the softmax d sum.
FLASHATTN_V3_BWD_HANDLE_ROUTINE(void *, dsoftmax_sum)
FLASHATTN_V3_BWD_HANDLE_ROUTINE(void *, softmax_lse_log2_ptr)

FLASHATTN_V3_BWD_HANDLE_ROUTINE(int *, dq_semaphore)
FLASHATTN_V3_BWD_HANDLE_ROUTINE(int *, dk_semaphore)
FLASHATTN_V3_BWD_HANDLE_ROUTINE(int *, dv_semaphore)

FLASHATTN_V3_BWD_HANDLE_ROUTINE(bool, deterministic)
FLASHATTN_V3_BWD_HANDLE_ROUTINE(int64_t, dq_accum_split_stride)
#endif

#undef DYNAMIC_LOAD_FLASHATTN_V3_WRAP

}  // namespace dynload
}  // namespace phi
