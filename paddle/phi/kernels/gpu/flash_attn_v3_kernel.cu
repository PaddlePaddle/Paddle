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

#include "paddle/phi/kernels/flash_attn_kernel.h"

#include <cstddef>
#include "glog/logging.h"  // For VLOG()
#include "paddle/common/enforce.h"
#include "paddle/common/errors.h"
#include "paddle/common/flags.h"
#include "paddle/phi/common/data_type.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/core/platform/device_context.h"
#include "paddle/phi/core/tensor_utils.h"
#include "paddle/phi/core/utils/data_type.h"
#include "paddle/phi/kernels/empty_kernel.h"
#include "paddle/phi/kernels/funcs/elementwise_base.h"
#include "paddle/phi/kernels/slice_kernel.h"
#include "paddle/utils/none.h"

#include "paddle/phi/kernels/gpu/flash_attn_utils.h"
#include "paddle/phi/kernels/gpu/flash_attn_v3_utils.h"

#include "paddle/phi/kernels/gpu/flash_attn_v3_kernel.h"

namespace phi {

template <typename T, typename Context>
void FlashAttnV3BaseKernel(
    const Context &ctx,
    const DenseTensor &q,
    const DenseTensor &k,
    const DenseTensor &v,
    const paddle::optional<DenseTensor>
        &k_new_,  // (b, s_k_new, h_k, d) or (total_k_new, h_k, d) if there is
                  // cu_seqlens_k_new
    const paddle::optional<DenseTensor>
        &v_new_,  // (b, s_k_new, h_k, dv) or (total_k_new, h_k, dv) if there is
                  // cu_seqlens_k_new
    const paddle::optional<DenseTensor>
        &q_v_,  // (b, s_q, h, dv) or (total_q_new, h, dv) if there is
                // cu_seqlens_q
    const paddle::optional<DenseTensor>
        &out_,  // (b, s_q, h, dv) or (total_q, h, dv) if there is cu_seqlens_q
    const paddle::optional<DenseTensor> &cu_seqlens_q_,      // b+1
    const paddle::optional<DenseTensor> &cu_seqlens_k_,      // b+1
    const paddle::optional<DenseTensor> &cu_seqlens_k_new_,  // b+1
    const paddle::optional<DenseTensor>
        &seqused_q_,  // b. If given, only this many elements of each batch
                      // element's queries and outputs are used.
    const paddle::optional<DenseTensor>
        &seqused_k_,  // b. If given, only this many elements of each batch
                      // element's keys are used.
    const paddle::optional<DenseTensor>
        &page_table_,  // (b_k, max_num_pages_per_seq)
    const paddle::optional<DenseTensor>
        &kv_batch_idx_,  // b. indices to index into the KV cache
    const paddle::optional<DenseTensor> &leftpad_k_,  // b
    const paddle::optional<DenseTensor>
        &rotary_cos_,  // seqlen_ro x (rotary_dim / 2)
    const paddle::optional<DenseTensor>
        &rotary_sin_,  // seqlen_ro x (rotary_dim / 2)
    const paddle::optional<DenseTensor> &q_descale_,  // (b, h_k), not (b, h)
    const paddle::optional<DenseTensor> &k_descale_,  // (b, h_k)
    const paddle::optional<DenseTensor> &v_descale_,  // (b, h_k)
    const paddle::optional<DenseTensor> &scheduler_metadata_,  // (b + 1)
    const int
        max_seqlen_q_,  // if max_seqlen_q_ is set to 0, it indicates that it is
                        // uninitialized and should not be referenced
    // TODO(tridao): check if we need max_seqlen_k
    const int
        max_seqlen_k_,  // if max_seqlen_q_ is set to 0, it indicates that it is
                        // uninitialized and should not be referenced
    const float softmax_scale,
    bool is_causal,
    int window_size_left,
    int window_size_right,
    const float softcap,
    const bool is_rotary_interleaved,  // if true, rotary combines indices 0 &
                                       // 1, else indices 0 & rotary_dim / 2
    int num_splits,
    const bool manual_set_pack_gqa,
    const bool
        pack_gqa_,  // the pack_gqa_ will be used only if manual_set_pack_gqa is
                    // set to True; otherwise, the internal heuristic
                    // get_pack_gqa() from fa3 will decide whether to pack gqa
    const int sm_margin,
    DenseTensor *out,
    DenseTensor *softmax_lse,
    DenseTensor *out_accum,
    DenseTensor *softmax_lse_accum) {
#ifdef PADDLE_WITH_FLASHATTN_V3
  // TODO(umiswing): support ampere
  int device_id = ctx.GetPlace().GetDeviceId();
  auto dprops = paddle::platform::GetDeviceProperties(device_id);
  const bool is_sm90 = dprops.major == 9 && dprops.minor == 0;
  PADDLE_ENFORCE_EQ(is_sm90,
                    true,
                    common::errors::Unavailable(
                        "FlashAttention-3 only supports Hopper GPUs."));

  auto q_type = q.dtype();
  PADDLE_ENFORCE_EQ(
      (q_type == phi::DataType::FLOAT16 || q_type == phi::DataType::BFLOAT16 ||
       q_type == phi::DataType::FLOAT8_E4M3FN),
      true,
      common::errors::InvalidArgument(
          "FlashAttention-3 only supports fp16, bf16, and fp8_e4m3 data type"));

  PADDLE_ENFORCE_EQ(k.dtype(),
                    q_type,
                    common::errors::InvalidArgument(
                        "query and key must have the same dtype"));
  PADDLE_ENFORCE_EQ(v.dtype(),
                    q_type,
                    common::errors::InvalidArgument(
                        "query and value must have the same dtype"));

  CHECK_DEVICE(q);
  CHECK_DEVICE(k);
  CHECK_DEVICE(v);

  PADDLE_ENFORCE_EQ(q.strides()[q.strides().size() - 1],
                    1,
                    common::errors::InvalidArgument(
                        "Input tensor must have contiguous last dimension"));
  PADDLE_ENFORCE_EQ(k.strides()[k.strides().size() - 1],
                    1,
                    common::errors::InvalidArgument(
                        "Input tensor must have contiguous last dimension"));
  PADDLE_ENFORCE_EQ(v.strides()[v.strides().size() - 1],
                    1,
                    common::errors::InvalidArgument(
                        "Input tensor must have contiguous last dimension"));

  DenseTensor page_table;
  // const bool paged_KV = page_table_.has_value();
  // umiswing: this is stupid but idk how to use paddle::optional
  const bool paged_KV = page_table_.is_initialized();
  if (paged_KV) {
    page_table = page_table_.get();
    CHECK_DEVICE(page_table);
    PADDLE_ENFORCE_EQ(page_table.dtype(),
                      phi::DataType::INT32,
                      common::errors::InvalidArgument(
                          "page_table must have dtype paddle.int32"));
    PADDLE_ENFORCE_EQ(page_table.strides()[page_table.strides().size() - 1],
                      1,
                      common::errors::InvalidArgument(
                          "page_table must have contiguous last dimension"));
  }

  // TODO(umiswing): support cusum

  DenseTensor cu_seqlens_q;
  // bool const is_varlen_q = cu_seqlens_q_.has_value();
  // TODO(umiswing): this is stupid, must fix it (after understand
  // paddle::optional)
  const bool is_varlen_q = cu_seqlens_q_.is_initialized();
  if (is_varlen_q) {
    cu_seqlens_q = cu_seqlens_q_.get();
    CHECK_DEVICE(cu_seqlens_q);
    CHECK_CONTIGUOUS(cu_seqlens_q);
    PADDLE_ENFORCE_EQ(cu_seqlens_q.dtype(),
                      phi::DataType::INT32,
                      common::errors::InvalidArgument(
                          "cu_seqlens_q must have dtype paddle.int32"));
    PADDLE_ENFORCE_NE(
        max_seqlen_q_,
        0,
        common::errors::InvalidArgument(
            "max_seqlen_q must be provided if cu_seqlens_q is provided"));
  }

  DenseTensor cu_seqlens_k;
  const bool is_varlen_k = cu_seqlens_k_.is_initialized();
  if (is_varlen_k) {
    cu_seqlens_k = cu_seqlens_k_.get();
    CHECK_DEVICE(cu_seqlens_k);
    CHECK_CONTIGUOUS(cu_seqlens_k);
    PADDLE_ENFORCE_EQ(cu_seqlens_k.dtype(),
                      phi::DataType::INT32,
                      common::errors::InvalidArgument(
                          "cu_seqlens_k must have dtype paddle.int32"));
    PADDLE_ENFORCE_NE(
        max_seqlen_k_,
        0,
        common::errors::InvalidArgument(
            "max_seqlen_k must be provided if cu_seqlens_k is provided"));
    PADDLE_ENFORCE_EQ(
        !paged_KV,
        true,
        common::errors::InvalidArgument(
            "If cu_seqlens_k is passed in, then page table is not supported"));
    PADDLE_ENFORCE_EQ(
        !kv_batch_idx_,
        true,
        common::errors::InvalidArgument(
            "If cu_seqlens_k is passed in, then page table is not supported"));
  }

  auto const sizes = q.dims();
  const int batch_size = !is_varlen_q ? sizes[0] : cu_seqlens_q.dims()[0] - 1;
  int seqlen_q = !is_varlen_q ? sizes[1] : max_seqlen_q_;
  int total_q = !is_varlen_q ? batch_size * sizes[1] : sizes[0];
  int num_heads = q.dims()[q.dims().size() - 2];
  int const head_size = q.dims()[q.dims().size() - 1];
  int const head_size_v = v.dims()[v.dims().size() - 1];
  int const max_num_pages_per_seq = !paged_KV ? 0 : page_table.dims()[1];
  int const num_pages = !paged_KV ? 0 : k.dims()[0];
  int const page_size = !paged_KV ? 1 : k.dims()[1];
  int const seqlen_k =
      !is_varlen_k
          ? (!paged_KV ? k.dims()[1] : max_num_pages_per_seq * page_size)
          : max_seqlen_k_;
  int const total_k = !is_varlen_k ? batch_size * k.dims()[1] : k.dims()[0];
  int const num_heads_k = k.dims()[k.dims().size() - 2];
  int const batch_size_k =
      !paged_KV ? (!is_varlen_k ? k.dims()[0] : cu_seqlens_k.dims()[0] - 1)
                : page_table.dims()[0];
  if (!kv_batch_idx_.is_initialized()) {
    PADDLE_ENFORCE_EQ(batch_size,
                      batch_size_k,
                      common::errors::InvalidArgument(
                          "batch_size must be equal to batch_size_k"));
  }
  int const max_headdim = get_max_headdim();
  PADDLE_ENFORCE_LE(
      head_size,
      max_headdim,
      common::errors::InvalidArgument(
          "FlashAttention forward only supports head dimension at most %d",
          max_headdim));
  PADDLE_ENFORCE_EQ(
      num_heads % num_heads_k,
      0,
      common::errors::InvalidArgument(
          "Number of heads in key/value must divide number of heads in query"));
  if (head_size_v != head_size) {
    PADDLE_ENFORCE_EQ(
        ((head_size > 128 && head_size <= 192 && head_size_v > 96 &&
          head_size_v <= 128) ||
         (head_size <= 64 && head_size_v <= 512)),
        true,
        common::errors::InvalidArgument(
            "If V headdim is different from Q/K dim, we only support "
            "Q/K headdim in (128, 192] and V headdim in (96, 128], "
            "or (Q/K <= 64 and V <= 512)."));
    PADDLE_ENFORCE_EQ(dprops.major,
                      9,
                      common::errors::InvalidArgument(
                          "Only Hopper supports different V headdim"));
    if (head_size_v > 256) {
      PADDLE_ENFORCE_EQ((q_type == phi::DataType::FLOAT16 ||
                         q_type == phi::DataType::BFLOAT16),
                        true,
                        common::errors::InvalidArgument(
                            "HeaddimV > 256 requires fp16 and bf16 data type"));
    }
  }

  // This needs to go before kBlockM & kBlockN since we rely on the correct
  // window_size and is_causal to set kBlockM
  // TODO(tridao): check this
  if (window_size_left >= seqlen_k - 1) {
    window_size_left = -1;
  }
  if (window_size_right >= seqlen_q - 1) {
    window_size_right = -1;
  }
  // causal=true is the same as causal=false in this case
  if (seqlen_q == 1 && window_size_left == -1 && window_size_right == -1) {
    // Special case of hdim 128 where we want causal to have kBlockN=128, better
    // for pagedKV and TMA
    if ((head_size <= 64 || head_size > 128) || !paged_KV) {
      is_causal = false;
    }
  }
  if (is_causal) {
    window_size_right = 0;
  }
  // There's a case where is_causal=false, window_size=(-1, 0). Then
  // set_params_fprop will set params.is_causal=true. If we don't have is_causal
  // here matching params.is_causal, we might get the wrong kBlockM.
  is_causal = window_size_left < 0 && window_size_right == 0;

  if (!is_varlen_q) {
    CHECK_SHAPE(q, batch_size, seqlen_q, num_heads, head_size);
  } else {
    CHECK_SHAPE(q, total_q, num_heads, head_size);
    CHECK_SHAPE(cu_seqlens_q, batch_size + 1);
  }
  if (!paged_KV) {
    if (!is_varlen_k) {
      CHECK_SHAPE(k, batch_size_k, seqlen_k, num_heads_k, head_size);
      CHECK_SHAPE(v, batch_size_k, seqlen_k, num_heads_k, head_size_v);
    } else {
      CHECK_SHAPE(k, total_k, num_heads_k, head_size);
      CHECK_SHAPE(v, total_k, num_heads_k, head_size_v);
      CHECK_SHAPE(cu_seqlens_k, batch_size + 1);
    }
  } else {
    CHECK_SHAPE(k, num_pages, page_size, num_heads_k, head_size);
    CHECK_SHAPE(v, num_pages, page_size, num_heads_k, head_size_v);
    CHECK_SHAPE(page_table, batch_size_k, max_num_pages_per_seq);
  }

  if (seqused_q_.is_initialized()) {
    auto seqused_q = seqused_q_.get();
    PADDLE_ENFORCE_EQ(
        seqused_q.dtype(),
        phi::DataType::INT32,
        common::errors::InvalidArgument("seqused_q must have dtype int32"));
    CHECK_DEVICE(seqused_q);
    CHECK_CONTIGUOUS(seqused_q);
    CHECK_SHAPE(seqused_q, batch_size);
  }
  if (seqused_k_.is_initialized()) {
    auto seqused_k = seqused_k_.get();
    PADDLE_ENFORCE_EQ(
        seqused_k.dtype(),
        phi::DataType::INT32,
        common::errors::InvalidArgument("seqused_k must have dtype int32"));
    CHECK_DEVICE(seqused_k);
    CHECK_CONTIGUOUS(seqused_k);
    CHECK_SHAPE(seqused_k, batch_size);
  }

  if (leftpad_k_.is_initialized()) {
    auto leftpad_k = leftpad_k_.get();
    PADDLE_ENFORCE_EQ(
        leftpad_k.dtype(),
        phi::DataType::INT32,
        common::errors::InvalidArgument("leftpad_k must have dtype int32"));
    CHECK_DEVICE(leftpad_k);
    CHECK_CONTIGUOUS(leftpad_k);
    CHECK_SHAPE(leftpad_k, batch_size);
  }

  // This is what we will template on
  bool const is_varlen =
      is_varlen_q || is_varlen_k || seqused_q_.is_initialized() ||
      seqused_k_.is_initialized() || leftpad_k_.is_initialized();
#ifdef FLASHATTENTION_DISABLE_VARLEN
  PADDLE_ENFORCE_EQ(!is_varlen,
                    true,
                    common::errors::Unavailable(
                        "This flash attention build does not support varlen."));
#endif

  int const alignment = q_type == phi::DataType::FLOAT8_E4M3FN ? 16 : 8;
  PADDLE_ENFORCE_EQ(head_size % alignment,
                    0,
                    common::errors::InvalidArgument(
                        "head_size should be a multiple of %d", alignment));
  PADDLE_ENFORCE_EQ(head_size_v % alignment,
                    0,
                    common::errors::InvalidArgument(
                        "head_size_v should be a multiple of %d", alignment));

  auto out_type =
      q_type == phi::DataType::FLOAT8_E4M3FN ? phi::DataType::BFLOAT16 : q_type;
  if (out_.is_initialized()) {
    *out = out_.get();
    PADDLE_ENFORCE_EQ(
        out->dtype(),
        out_type,
        common::errors::InvalidArgument(
            "For FP16/BF16 input, output must have the same dtype as "
            "inputs. For FP8 input, output must have dtype BF16"));
    CHECK_DEVICE((*out));
    PADDLE_ENFORCE_EQ(out->strides()[out->strides().size() - 1],
                      1,
                      common::errors::InvalidArgument(
                          "Output tensor must have contiguous last dimension"));
    if (!is_varlen_q) {
      CHECK_SHAPE((*out), batch_size, seqlen_q, num_heads, head_size_v);
    } else {
      CHECK_SHAPE((*out), total_q, num_heads, head_size_v);
    }
  } else {
    if (!is_varlen_q) {
      out->Resize(
          common::make_ddim({batch_size, seqlen_q, num_heads, head_size_v}));
    } else {
      out->Resize(common::make_ddim({total_q, num_heads, head_size_v}));
    }
    if (q_type == phi::DataType::FLOAT8_E4M3FN) {
      ctx.template Alloc<phi::dtype::bfloat16>(out);
    } else {
      // umiswing: assuming T is Input Type
      ctx.template Alloc<T>(out);
    }
  }

  auto round_multiple = [](int x, int m) { return (x + m - 1) / m * m; };
  int const head_size_rounded = round_up_headdim(head_size);
  int const head_size_v_rounded = round_up_headdim(head_size_v);
  int const seqlen_q_rounded = round_multiple(seqlen_q, 128);
  int const seqlen_k_rounded = round_multiple(seqlen_k, 128);

  if (!is_varlen_q) {
    softmax_lse->Resize(common::make_ddim({batch_size, num_heads, seqlen_q}));
  } else {
    softmax_lse->Resize(common::make_ddim({num_heads, total_q}));
  }
  ctx.template Alloc<float>(softmax_lse);

  Flash_fwd_params *params_handle = get_flash_fwd_params_handle();
  dynload::fa3_clear_fwd_params_handle(params_handle);
  set_params_fprop(
      params_handle,
      batch_size,
      seqlen_q,
      seqlen_k,
      seqlen_q_rounded,
      seqlen_k_rounded,
      num_heads,
      num_heads_k,
      head_size,
      head_size_rounded,
      q,
      k,
      v,
      out,
      !is_varlen_q ? nullptr : cu_seqlens_q.data(),
      !is_varlen_k ? nullptr : cu_seqlens_k.data(),
      seqused_q_.is_initialized() ? const_cast<void *>(seqused_q_.get().data())
                                  : nullptr,
      seqused_k_.is_initialized() ? const_cast<void *>(seqused_k_.get().data())
                                  : nullptr,
      softmax_lse->data(),
      /*p_dropout=*/0.f,
      softmax_scale,
      window_size_left,
      window_size_right,
      dprops,
      softcap,
      sm_margin);
  phi::dynload::fa3_fwd_params_set_total_q(params_handle, total_q);
  phi::dynload::fa3_fwd_params_set_total_k(params_handle, total_k);
  phi::dynload::fa3_fwd_params_set_b_k(params_handle, batch_size_k);
  phi::dynload::fa3_fwd_params_set_dv(params_handle, head_size_v);
  phi::dynload::fa3_fwd_params_set_dv_rounded(params_handle,
                                              head_size_v_rounded);

  if (leftpad_k_
          .is_initialized()) {  // This needs to be set before get_pagedkv_tma
    phi::dynload::fa3_fwd_params_set_leftpad_k(params_handle,
                                               leftpad_k_.get().data<int>());
  }
  if (paged_KV) {
    phi::dynload::fa3_fwd_params_set_page_table(params_handle,
                                                page_table.data<int>());
    phi::dynload::fa3_fwd_params_set_page_table_batch_stride(
        params_handle, page_table.strides()[0]);
  }
  phi::dynload::fa3_fwd_params_set_page_size(params_handle, page_size);
  phi::dynload::fa3_fwd_params_set_num_pages(params_handle, num_pages);

  if (k_new_.is_initialized()) {  // This needs to be set before get_pagedkv_tma
    DenseTensor k_new, v_new;
    PADDLE_ENFORCE_EQ(
        v_new_.is_initialized(),
        true,
        common::errors::InvalidArgument(
            "If k_new is supplied, v_new must also be passed in"));
    PADDLE_ENFORCE_EQ(
        seqused_k_.is_initialized(),
        true,
        common::errors::InvalidArgument(
            "If k_new is supplied, seqlens_k must also be passed in"));
    PADDLE_ENFORCE_LE(
        seqlen_q,
        seqlen_k,
        common::errors::InvalidArgument(
            "If k_new is supplied, it must have seqlen <= the seqlen "
            "of the KV cache"));
    DenseTensor cu_seqlens_k_new;
    bool const is_varlen_k_new = cu_seqlens_k_new_.is_initialized();
    if (is_varlen_k_new) {
      cu_seqlens_k_new = cu_seqlens_k_new_.get();
      CHECK_DEVICE(cu_seqlens_k_new);
      CHECK_CONTIGUOUS(cu_seqlens_k_new);
      PADDLE_ENFORCE_EQ(cu_seqlens_k_new.dtype(),
                        phi::DataType::INT32,
                        common::errors::InvalidArgument(
                            "cu_seqlens_k_new must have dtype paddle.int32"));
    }
    k_new = k_new_.get();
    v_new = v_new_.get();
    PADDLE_ENFORCE_EQ(k_new.dtype(),
                      q_type,
                      common::errors::InvalidArgument(
                          "k_new must have the same dtype as query"));
    PADDLE_ENFORCE_EQ(v_new.dtype(),
                      q_type,
                      common::errors::InvalidArgument(
                          "v_new must have the same dtype as query"));
    CHECK_DEVICE(k_new);
    CHECK_DEVICE(v_new);
    PADDLE_ENFORCE_EQ(k_new.strides()[k_new.strides().size() - 1],
                      1,
                      common::errors::InvalidArgument(
                          "k_new tensor must have contiguous last dimension"));
    PADDLE_ENFORCE_EQ(v_new.strides()[v_new.strides().size() - 1],
                      1,
                      common::errors::InvalidArgument(
                          "v_new tensor must have contiguous last dimension"));
    // We don't need max_seqlen_k_new, so seqlen_k_new can be whatever when
    // is_varlen_k_new
    int seqlen_k_new = !is_varlen_k_new ? k_new.dims()[1] : 0;
    int total_k_new =
        !is_varlen_k_new ? batch_size * k_new.dims()[1] : k_new.dims()[0];
    if (!is_varlen_k_new) {
      CHECK_SHAPE(k_new, batch_size, seqlen_k_new, num_heads_k, head_size);
      CHECK_SHAPE(v_new, batch_size, seqlen_k_new, num_heads_k, head_size_v);
    } else {
      CHECK_SHAPE(k_new, total_k_new, num_heads_k, head_size);
      CHECK_SHAPE(v_new, total_k_new, num_heads_k, head_size_v);
      CHECK_SHAPE(cu_seqlens_k_new, batch_size + 1);
    }
    // umiswing: dump this to shared library
    phi::dynload::fa3_fwd_params_set_seqlen_knew(params_handle, seqlen_k_new);
    phi::dynload::fa3_fwd_params_set_total_knew(params_handle, total_k_new);
    phi::dynload::fa3_fwd_params_set_knew_ptr(params_handle,
                                              const_cast<void *>(k_new.data()));
    phi::dynload::fa3_fwd_params_set_vnew_ptr(params_handle,
                                              const_cast<void *>(v_new.data()));
    // All stride are in elements, not bytes.
    phi::dynload::fa3_fwd_params_set_knew_row_stride(
        params_handle, k_new.strides()[k_new.strides().size() - 3]);
    phi::dynload::fa3_fwd_params_set_vnew_row_stride(
        params_handle, v_new.strides()[v_new.strides().size() - 3]);
    phi::dynload::fa3_fwd_params_set_knew_head_stride(
        params_handle, k_new.strides()[k_new.strides().size() - 2]);
    phi::dynload::fa3_fwd_params_set_vnew_head_stride(
        params_handle, v_new.strides()[v_new.strides().size() - 2]);
    if (!is_varlen_k_new) {
      phi::dynload::fa3_fwd_params_set_knew_batch_stride(params_handle,
                                                         k_new.strides()[0]);
      phi::dynload::fa3_fwd_params_set_vnew_batch_stride(params_handle,
                                                         v_new.strides()[0]);
    }
    if (is_varlen_k_new) {
      phi::dynload::fa3_fwd_params_set_cu_seqlens_knew(
          params_handle, cu_seqlens_k_new.data<int>());
    }
  }

  // 992 = 32 * 31 is the max supported batch in prepare_varlen_num_blocks
  // kernel
  bool const use_dynamic_split =
      is_varlen && phi::dynload::fa3_fwd_params_get_b(params_handle) <= 992;
  // Temporarily set num_splits_dynamic_ptr to 1 since get_num_splits checks it
  phi::dynload::fa3_fwd_params_set_num_splits_dynamic_ptr(
      params_handle, !use_dynamic_split ? nullptr : reinterpret_cast<int *>(1));

  phi::dynload::fa3_fwd_params_set_pagedkv_tma(
      params_handle, phi::dynload::fa3_get_pagedkv_tma(params_handle));
  if (num_splits <= 0) {
    num_splits = phi::dynload::fa3_get_num_splits(params_handle);
  }
  phi::dynload::fa3_fwd_params_set_num_splits(params_handle, num_splits);

  // Always enable PackGQA for Split, and get_pack_gqa requires
  // params.num_splits to decide
  const bool pack_gqa = manual_set_pack_gqa
                            ? pack_gqa_
                            : phi::dynload::fa3_get_pack_gqa(params_handle);
  phi::dynload::fa3_fwd_params_set_pack_gqa(params_handle, pack_gqa);

  // This needs to be set after get_num_splits
  DenseTensor tile_count_semaphore;  // Contains the semaphore and optionally
                                     // num_splits_dynamic
  // We don't use the persistent scheduler if Split and not Varlen
  const bool params_is_causal =
      phi::dynload::fa3_fwd_params_get_is_causal(params_handle);
  const bool params_is_local =
      phi::dynload::fa3_fwd_params_get_is_local(params_handle);
  const int params_num_splits =
      phi::dynload::fa3_fwd_params_get_num_splits(params_handle);
  const int params_b = phi::dynload::fa3_fwd_params_get_b(params_handle);
  const int params_arch = phi::dynload::fa3_fwd_params_get_arch(params_handle);
  bool const scheduler_needs_semaphore =
      params_arch >= 90 ? (((params_is_causal || params_is_local) &&
                            (params_num_splits == 1)) ||
                           is_varlen)
                        : ((params_is_causal && !is_varlen) ||
                           (is_varlen && params_num_splits > 1));
  if (scheduler_needs_semaphore || use_dynamic_split) {
    int metadata_size = static_cast<int>(scheduler_needs_semaphore) +
                        static_cast<int>(use_dynamic_split) * params_b;
    phi::dynload::fa3_fwd_params_set_skip_scheduler_metadata_computation(
        params_handle, scheduler_metadata_.is_initialized());
    if (scheduler_metadata_.is_initialized()) {
      DenseTensor scheduler_metadata = scheduler_metadata_.get();
      CHECK_DEVICE(scheduler_metadata);
      CHECK_SHAPE(scheduler_metadata, metadata_size);
      CHECK_CONTIGUOUS(scheduler_metadata);
      PADDLE_ENFORCE_EQ(scheduler_metadata.dtype(),
                        phi::DataType::INT32,
                        common::errors::InvalidArgument(
                            "scheduler_metadata must have dtype int32"));
      tile_count_semaphore = scheduler_metadata;
    } else {
      tile_count_semaphore = phi::Empty<int32_t>(ctx, {metadata_size});
    }
    if (scheduler_needs_semaphore && !use_dynamic_split) {
      phi::funcs::SetConstant<Context, int32_t> set_zero;
      set_zero(ctx,
               &tile_count_semaphore,
               int32_t{0});  // If varlen we'll manually do the zero-ing
    }
    phi::dynload::fa3_fwd_params_set_tile_count_semaphore(
        params_handle,
        scheduler_needs_semaphore
            ? const_cast<int *>(tile_count_semaphore.data<int>())
            : nullptr);
    phi::dynload::fa3_fwd_params_set_num_splits_dynamic_ptr(
        params_handle,
        use_dynamic_split
            ? const_cast<int *>(tile_count_semaphore.data<int>()) + 1
            : nullptr);
  }

  if (q_v_.is_initialized()) {
    PADDLE_ENFORCE_LT(head_size,
                      64,
                      common::errors::InvalidArgument(
                          "q_v is only supported for head_size <= 64"));
    PADDLE_ENFORCE_EQ(
        (q_type == phi::DataType::FLOAT16 || q_type == phi::DataType::FLOAT16),
        true,
        common::errors::InvalidArgument(
            "q_v is only supported for fp16 and bf16 data type"));
    PADDLE_ENFORCE_EQ(params_arch,
                      90,
                      common::errors::InvalidArgument(
                          "q_v is only supported for Hopper GPUs"));
    DenseTensor q_v = q_v_.get();
    PADDLE_ENFORCE_EQ(q_v.dtype(),
                      q_type,
                      common::errors::InvalidArgument(
                          "q_v must have the same dtype as query"));
    CHECK_DEVICE(q_v);
    PADDLE_ENFORCE_EQ(q_v.strides()[q_v.strides().size() - 1],
                      1,
                      common::errors::InvalidArgument(
                          "q_v tensor must have contiguous last dimension"));
    if (!is_varlen_q) {
      CHECK_SHAPE(q_v, batch_size, seqlen_q, num_heads, head_size_v);
    } else {
      CHECK_SHAPE(q_v, total_q, num_heads, head_size_v);
    }
    phi::dynload::fa3_fwd_params_set_qv_ptr(params_handle,
                                            const_cast<void *>(q_v.data()));
    // All stride are in elements, not bytes.
    phi::dynload::fa3_fwd_params_set_qv_row_stride(
        params_handle, q_v.strides()[q_v.strides().size() - 3]);
    phi::dynload::fa3_fwd_params_set_qv_head_stride(
        params_handle, q_v.strides()[q_v.strides().size() - 2]);
    if (!is_varlen_q) {
      phi::dynload::fa3_fwd_params_set_qv_batch_stride(params_handle,
                                                       q_v.strides()[0]);
    }
  }

  if (rotary_cos_.is_initialized()) {
    PADDLE_ENFORCE_EQ(
        k_new_.is_initialized(),
        true,
        common::errors::InvalidArgument(
            "If rotary cos/sin are provided, new key / value to be "
            "appended to KV cache must also be provided"));
    DenseTensor rotary_cos = rotary_cos_.get();
    CHECK_DEVICE(rotary_cos);
    CHECK_CONTIGUOUS(rotary_cos);
    int params_rotary_dim = rotary_cos.dims()[1] * 2;
    phi::dynload::fa3_fwd_params_set_rotary_dim(params_handle,
                                                params_rotary_dim);
    PADDLE_ENFORCE_LE(
        params_rotary_dim,
        head_size,
        common::errors::InvalidArgument("rotary_dim must be <= headdim"));
    PADDLE_ENFORCE_EQ(
        params_rotary_dim % 16,
        0,
        common::errors::InvalidArgument(
            "Only rotary dimensions divisible by 16 are currently supported"));
    const int seqlen_ro = rotary_cos.dims()[0];
    if (paged_KV) {
      PADDLE_ENFORCE_GE(
          seqlen_ro,
          seqlen_k,
          common::errors::InvalidArgument(
              "cos/sin seqlen must be at least the seqlen of KV cache"));
    }
    CHECK_SHAPE(rotary_cos, seqlen_ro, params_rotary_dim / 2);
    PADDLE_ENFORCE_EQ(rotary_cos.dtype(),
                      q_type,
                      common::errors::InvalidArgument(
                          "rotary_cos must have the same dtype as query"));

    PADDLE_ENFORCE_EQ(
        rotary_sin_.is_initialized(),
        true,
        common::errors::InvalidArgument(
            "If rotary cos is provided, rotary sin must also be provided"));
    auto rotary_sin = rotary_sin_.get();
    CHECK_DEVICE(rotary_sin);
    CHECK_CONTIGUOUS(rotary_sin);
    CHECK_SHAPE(rotary_sin, seqlen_ro, params_rotary_dim / 2);
    PADDLE_ENFORCE_EQ(rotary_sin.dtype(),
                      q_type,
                      common::errors::InvalidArgument(
                          "rotary_cos must have the same dtype as query"));

    phi::dynload::fa3_fwd_params_set_rotary_cos_ptr(
        params_handle, const_cast<void *>(rotary_cos.data()));
    phi::dynload::fa3_fwd_params_set_rotary_sin_ptr(
        params_handle, const_cast<void *>(rotary_sin.data()));
    dynload::fa3_fwd_params_set_is_rotary_interleaved(params_handle,
                                                      is_rotary_interleaved);
  } else {
    phi::dynload::fa3_fwd_params_set_rotary_dim(params_handle, 0);
  }

  if (kv_batch_idx_.is_initialized()) {
    DenseTensor kv_batch_idx = kv_batch_idx_.get();
    CHECK_DEVICE(kv_batch_idx);
    CHECK_CONTIGUOUS(kv_batch_idx);
    PADDLE_ENFORCE_EQ(
        kv_batch_idx.dtype(),
        phi::DataType::INT32,
        common::errors::InvalidArgument("kv_batch_idx must have dtype int32"));
    phi::dynload::fa3_fwd_params_set_kv_batch_idx(
        params_handle, reinterpret_cast<int *>(kv_batch_idx.data()));
  }

  if (phi::dynload::fa3_fwd_params_get_num_splits(params_handle) > 1) {
    PADDLE_ENFORCE_LE(
        phi::dynload::fa3_fwd_params_get_num_splits(params_handle),
        256,
        common::errors::InvalidArgument("num_splits > 256 not supported"));
    if (!is_varlen_q) {
      out_accum->Resize(common::make_ddim(
          {phi::dynload::fa3_fwd_params_get_num_splits(params_handle),
           batch_size,
           num_heads,
           seqlen_q,
           head_size_v}));
      softmax_lse_accum->Resize(common::make_ddim(
          {phi::dynload::fa3_fwd_params_get_num_splits(params_handle),
           batch_size,
           num_heads,
           seqlen_q}));
      ctx.template Alloc<float>(out_accum);
      ctx.template Alloc<float>(softmax_lse_accum);
      phi::dynload::fa3_fwd_params_set_oaccum_batch_stride(
          params_handle, out_accum->strides()[1]);
      phi::dynload::fa3_fwd_params_set_lseaccum_batch_stride(
          params_handle, softmax_lse_accum->strides()[1]);
    } else {
      out_accum->Resize(common::make_ddim(
          {phi::dynload::fa3_fwd_params_get_num_splits(params_handle),
           num_heads,
           total_q,
           head_size_v}));
      softmax_lse_accum->Resize(common::make_ddim(
          {phi::dynload::fa3_fwd_params_get_num_splits(params_handle),
           num_heads,
           total_q}));
      ctx.template Alloc<float>(out_accum);
      ctx.template Alloc<float>(softmax_lse_accum);
    }
    phi::dynload::fa3_fwd_params_set_is_fp32(params_handle, false);
    phi::dynload::fa3_fwd_params_set_oaccum_ptr(
        params_handle, const_cast<void *>(out_accum->data()));
    phi::dynload::fa3_fwd_params_set_softmax_lseaccum_ptr(
        params_handle, const_cast<void *>(softmax_lse_accum->data()));
    phi::dynload::fa3_fwd_params_set_oaccum_split_stride(
        params_handle, out_accum->strides()[0]);
    phi::dynload::fa3_fwd_params_set_oaccum_row_stride(
        params_handle, out_accum->strides()[out_accum->strides().size() - 2]);
    phi::dynload::fa3_fwd_params_set_oaccum_head_stride(
        params_handle, out_accum->strides()[out_accum->strides().size() - 3]);
    phi::dynload::fa3_fwd_params_set_lseaccum_split_stride(
        params_handle, softmax_lse_accum->strides()[0]);
    phi::dynload::fa3_fwd_params_set_lseaccum_head_stride(
        params_handle,
        softmax_lse_accum->strides()[softmax_lse_accum->strides().size() - 2]);
  }

  if (q_type == phi::DataType::FLOAT8_E4M3FN) {
    if (q_descale_.is_initialized()) {
      DenseTensor q_descale = q_descale_.get();
      CHECK_DEVICE(q_descale);
      CHECK_SHAPE(q_descale, batch_size, num_heads_k);
      phi::dynload::fa3_fwd_params_set_q_descale_ptr(
          params_handle, const_cast<float *>(q_descale.data<float>()));
      phi::dynload::fa3_fwd_params_set_q_descale_batch_stride(
          params_handle, q_descale.strides()[0]);
      phi::dynload::fa3_fwd_params_set_q_descale_head_stride(
          params_handle, q_descale.strides()[1]);
    } else {
      phi::dynload::fa3_fwd_params_set_q_descale_ptr(params_handle, nullptr);
    }
    if (k_descale_.is_initialized()) {
      DenseTensor k_descale = k_descale_.get();
      CHECK_DEVICE(k_descale);
      CHECK_SHAPE(k_descale, batch_size, num_heads_k);
      phi::dynload::fa3_fwd_params_set_k_descale_ptr(
          params_handle, const_cast<float *>(k_descale.data<float>()));
      phi::dynload::fa3_fwd_params_set_k_descale_batch_stride(
          params_handle, k_descale.strides()[0]);
      phi::dynload::fa3_fwd_params_set_k_descale_head_stride(
          params_handle, k_descale.strides()[1]);
    } else {
      phi::dynload::fa3_fwd_params_set_k_descale_ptr(params_handle, nullptr);
    }
    if (v_descale_.is_initialized()) {
      DenseTensor v_descale = v_descale_.get();
      CHECK_DEVICE(v_descale);
      CHECK_SHAPE(v_descale, batch_size, num_heads_k);
      phi::dynload::fa3_fwd_params_set_v_descale_ptr(
          params_handle, const_cast<float *>(v_descale.data<float>()));
      phi::dynload::fa3_fwd_params_set_v_descale_batch_stride(
          params_handle, v_descale.strides()[0]);
      phi::dynload::fa3_fwd_params_set_v_descale_head_stride(
          params_handle, v_descale.strides()[1]);
    } else {
      phi::dynload::fa3_fwd_params_set_v_descale_ptr(params_handle, nullptr);
    }
  }

#ifdef FLASHATTENTION_DISABLE_LOCAL
  PADDLE_ENFORCE_EQ(
      !phi::dynload::fa3_fwd_params_get_is_local(params_handle),
      true,
      common::errors::InvalidArgument(
          "This flash attention build does not support local attention."));
#endif
#ifdef FLASHATTENTION_DISABLE_SOFTCAP
  PADDLE_ENFORCE_EQ(
      phi::dynload::fa3_fwd_params_get_softcap(params_handle),
      0.0,
      common::errors::InvalidArgument(
          "This flash attention build does not support tanh softcapping."));
#endif
#ifdef FLASHATTENTION_DISABLE_SPLIT
  PADDLE_ENFORCE_EQ(phi::dynload::fa3_fwd_params_get_num_splits(params_handle),
                    1,
                    common::errors::InvalidArgument(
                        "This flash attention build does not support splits."));
#endif
#ifdef FLASHATTENTION_DISABLE_PACKGQA
  PADDLE_ENFORCE_EQ(
      (!phi::dynload::fa3_fwd_params_get_pack_gqa(params_handle) ||
       phi::dynload::fa3_fwd_params_get_arch(params_handle) < 90 ||
       (phi::dynload::fa3_fwd_params_get_page_table(params_handle) &&
        !phi::dynload::fa3_fwd_params_get_pagedkv_tma(params_handle)) ||
       phi::dynload::fa3_fwd_params_get_num_splits(params_handle) > 1),
      true,
      common::errors::InvalidArgument(
          "This flash attention build does not support pack_gqa."));
#endif
#ifdef FLASHATTENTION_DISABLE_PAGEDKV
  PADDLE_ENFORCE_EQ(
      (!(phi::dynload::fa3_fwd_params_get_page_table(params_handle) &&
         !phi::dynload::fa3_fwd_params_get_pagedkv_tma(params_handle))),
      true,
      common::errors::InvalidArgument(
          "This flash attention build does not support paged KV."));
#endif
#ifdef FLASHATTENTION_DISABLE_APPENDKV
  PADDLE_ENFORCE_EQ(
      !k_new_.is_initialized(),
      true,
      common::errors::InvalidArgument(
          "This flash attention build does not support appending KV."));
#endif

  if (total_q > 0 &&
      (total_k + dynload::fa3_fwd_params_get_total_knew(params_handle)) > 0 &&
      num_heads_k > 0) {
    dynload::fa3_run_mha_fwd(params_handle, ctx.stream());
    if (dynload::fa3_fwd_params_get_num_splits(params_handle) > 1) {
      if (out_type == phi::DataType::BFLOAT16) {
        // Since we want output in BF16. Otherwise fwd_combine will output to
        // FP16
        dynload::fa3_fwd_params_set_is_bf16(params_handle, true);
      }
      // Unless there's seqused_q, for the purpose of attn_combine, we can just
      // treat it as batch=1 and seqlen = total_q, and don't need to dispatch to
      // Varlen there. However, with dynamic split, each row needs to know which
      // batch it belongs to to read the number of splits, so we just use the
      // varlen version of combine kernel. if (is_varlen_q &&
      // !seqused_q_.has_value()) { if (is_varlen_q) {
      //     params.b = 1;
      //     params.seqlen_q = total_q;
      // }
      // }
      dynload::fa3_run_mha_fwd_combine(
          params_handle, ctx.stream(), true /*enable_pdl*/);
    }
  } else if (total_q > 0 && num_heads_k > 0) {
    PADDLE_ENFORCE_EQ(
        (out->dtype() == phi::DataType::BFLOAT16 ||
         out->dtype() == phi::DataType::FLOAT16 ||
         out->dtype() == phi::DataType::FLOAT8_E4M3FN),
        true,
        common::errors::InvalidArgument("flash attention 3 supports bfloat16, "
                                        "float16 and float8_e4m3fn only."));
    // If seqlen_k == 0, then we have an empty tensor. We need to set the output
    // to 0.
    if (out->dtype() == phi::DataType::BFLOAT16) {
      phi::funcs::SetConstant<Context, phi::dtype::bfloat16> set_zero;
      set_zero(
          ctx,
          out,
          phi::dtype::bfloat16{0});  // If varlen we'll manually do the zero-ing
    } else if (out->dtype() == phi::DataType::FLOAT16) {
      phi::funcs::SetConstant<Context, phi::dtype::float16> set_zero;
      set_zero(
          ctx,
          out,
          phi::dtype::float16{0});  // If varlen we'll manually do the zero-ing
    } else if (out->dtype() == phi::DataType::FLOAT8_E4M3FN) {
      phi::funcs::SetConstant<Context, phi::dtype::float8_e4m3fn> set_zero;
      set_zero(ctx,
               out,
               phi::dtype::float8_e4m3fn{
                   0});  // If varlen we'll manually do the zero-ing
    }
    phi::funcs::SetConstant<Context, float> set_infinity;
    set_infinity(ctx, softmax_lse, std::numeric_limits<float>::infinity());
  }

#else
  RaiseNotSupportedError();
#endif
}

template <typename T, typename Context>
void FlashAttnV3Kernel(const Context &ctx,
                       const DenseTensor &q,
                       const DenseTensor &k,
                       const DenseTensor &v,
                       const paddle::optional<DenseTensor> &q_v_,
                       const paddle::optional<DenseTensor> &q_descale_,
                       const paddle::optional<DenseTensor> &k_descale_,
                       const paddle::optional<DenseTensor> &v_descale_,
                       const float softmax_scale,
                       bool is_causal,
                       int window_size_left,
                       int window_size_right,
                       const float softcap,
                       int num_splits,
                       const bool manual_set_pack_gqa,
                       const bool pack_gqa_,
                       const int sm_margin,
                       DenseTensor *out,
                       DenseTensor *softmax_lse) {
#ifdef PADDLE_WITH_FLASHATTN_V3
  // umiswing: the following options have not been fully tested
  PADDLE_ENFORCE_EQ(q_v_.is_initialized(),
                    false,
                    common::errors::InvalidArgument("q_v_ is not supported"));
  PADDLE_ENFORCE_EQ(
      q_descale_.is_initialized(),
      false,
      common::errors::InvalidArgument("q_descale_ is not supported"));
  PADDLE_ENFORCE_EQ(
      k_descale_.is_initialized(),
      false,
      common::errors::InvalidArgument("k_descale_ is not supported"));
  PADDLE_ENFORCE_EQ(
      v_descale_.is_initialized(),
      false,
      common::errors::InvalidArgument("v_descale_ is not supported"));
  PADDLE_ENFORCE_EQ(
      window_size_left,
      -1,
      common::errors::InvalidArgument("window_size is not supported, please "
                                      "set window_size_left/right to -1"));
  PADDLE_ENFORCE_EQ(
      window_size_right,
      -1,
      common::errors::InvalidArgument("window_size is not supported, please "
                                      "set window_size_left/right to -1"));
  PADDLE_ENFORCE_EQ(softcap,
                    0,
                    common::errors::InvalidArgument(
                        "softcap is not supported, please set softcap to 0"));
  PADDLE_ENFORCE_EQ(
      num_splits,
      1,
      common::errors::InvalidArgument(
          "num_splits is not supported, please set num_splits to 1"));
  PADDLE_ENFORCE_EQ(manual_set_pack_gqa,
                    false,
                    common::errors::InvalidArgument(
                        "manual_set_pack_gqa is not supported, please set "
                        "manual_set_pack_gqa to false"));
  PADDLE_ENFORCE_EQ(
      pack_gqa_,
      false,
      common::errors::InvalidArgument(
          "pack_gqa_ is not supported, please set pack_gqa_ to false"));
  PADDLE_ENFORCE_EQ(
      sm_margin,
      0,
      common::errors::InvalidArgument(
          "sm_margin is not supported, please set sm_margin to 0"));

  DenseTensor out_accum;
  DenseTensor softmax_lse_accum;
  FlashAttnV3BaseKernel<T, Context>(ctx,
                                    q,
                                    k,
                                    v,
                                    paddle::none,  // k_new_
                                    paddle::none,  // v_new_
                                    q_v_,
                                    paddle::none,  // out_
                                    paddle::none,  // cu_seqlens_q_
                                    paddle::none,  // cu_seqlens_k_
                                    paddle::none,  // cu_seqlens_k_new_
                                    paddle::none,  // seqused_q_
                                    paddle::none,  // seqused_k_
                                    paddle::none,  // page_table_
                                    paddle::none,  // kv_batch_idx_
                                    paddle::none,  // leftpad_k_
                                    paddle::none,  // rotary_cos_
                                    paddle::none,  // rotary_sin_
                                    q_descale_,
                                    k_descale_,
                                    v_descale_,
                                    paddle::none,  // scheduler_metadata
                                    0,             // max_seqlen_q_
                                    0,             // max_seqlen_k_
                                    softmax_scale,
                                    is_causal,
                                    window_size_left,
                                    window_size_right,
                                    softcap,
                                    true,  // is_rotary_interleaved
                                    num_splits,
                                    manual_set_pack_gqa,
                                    pack_gqa_,
                                    sm_margin,
                                    out,
                                    softmax_lse,
                                    &out_accum,
                                    &softmax_lse_accum);
#else
  RaiseNotSupportedError();
#endif
}

template <typename T, typename Context>
void FlashAttnV3VarLenKernel(const Context &ctx,
                             const DenseTensor &q,
                             const DenseTensor &k,
                             const DenseTensor &v,
                             const DenseTensor &cu_seqlens_q,
                             const DenseTensor &cu_seqlens_k,
                             const paddle::optional<DenseTensor> &q_v_,
                             const paddle::optional<DenseTensor> &q_descale_,
                             const paddle::optional<DenseTensor> &k_descale_,
                             const paddle::optional<DenseTensor> &v_descale_,
                             const float softmax_scale,
                             bool is_causal,
                             int window_size_left,
                             int window_size_right,
                             const float softcap,
                             int num_splits,
                             const bool manual_set_pack_gqa,
                             const bool pack_gqa_,
                             const int sm_margin,
                             const int max_seqlen_q,
                             const int max_seqlen_k,
                             DenseTensor *out,
                             DenseTensor *softmax_lse) {
#ifdef PADDLE_WITH_FLASHATTN_V3
  // umiswing: the following options have not been fully tested
  PADDLE_ENFORCE_EQ(q_v_.is_initialized(),
                    false,
                    common::errors::InvalidArgument("q_v_ is not supported"));
  PADDLE_ENFORCE_EQ(
      q_descale_.is_initialized(),
      false,
      common::errors::InvalidArgument("q_descale_ is not supported"));
  PADDLE_ENFORCE_EQ(
      k_descale_.is_initialized(),
      false,
      common::errors::InvalidArgument("k_descale_ is not supported"));
  PADDLE_ENFORCE_EQ(
      v_descale_.is_initialized(),
      false,
      common::errors::InvalidArgument("v_descale_ is not supported"));
  PADDLE_ENFORCE_EQ(
      window_size_left,
      -1,
      common::errors::InvalidArgument("window_size is not supported, please "
                                      "set window_size_left/right to -1"));
  PADDLE_ENFORCE_EQ(
      window_size_right,
      -1,
      common::errors::InvalidArgument("window_size is not supported, please "
                                      "set window_size_left/right to -1"));
  PADDLE_ENFORCE_EQ(softcap,
                    0,
                    common::errors::InvalidArgument(
                        "softcap is not supported, please set softcap to 0"));
  PADDLE_ENFORCE_EQ(
      num_splits,
      1,
      common::errors::InvalidArgument(
          "num_splits is not supported, please set num_splits to 1"));
  PADDLE_ENFORCE_EQ(manual_set_pack_gqa,
                    false,
                    common::errors::InvalidArgument(
                        "manual_set_pack_gqa is not supported, please set "
                        "manual_set_pack_gqa to false"));
  PADDLE_ENFORCE_EQ(
      pack_gqa_,
      false,
      common::errors::InvalidArgument(
          "pack_gqa_ is not supported, please set pack_gqa_ to false"));
  PADDLE_ENFORCE_EQ(
      sm_margin,
      0,
      common::errors::InvalidArgument(
          "sm_margin is not supported, please set sm_margin to 0"));

  DenseTensor out_accum;
  DenseTensor softmax_lse_accum;
  FlashAttnV3BaseKernel<T, Context>(ctx,
                                    q,
                                    k,
                                    v,
                                    paddle::none,  // k_new_
                                    paddle::none,  // v_new_
                                    q_v_,
                                    paddle::none,  // out_
                                    cu_seqlens_q,  // cu_seqlens_q_
                                    cu_seqlens_k,  // cu_seqlens_k_
                                    paddle::none,  // cu_seqlens_k_new_
                                    paddle::none,  // seqused_q_
                                    paddle::none,  // seqused_k_
                                    paddle::none,  // page_table_
                                    paddle::none,  // kv_batch_idx_
                                    paddle::none,  // leftpad_k_
                                    paddle::none,  // rotary_cos_
                                    paddle::none,  // rotary_sin_
                                    q_descale_,
                                    k_descale_,
                                    v_descale_,
                                    paddle::none,  // scheduler_metadata
                                    max_seqlen_q,  // max_seqlen_q_
                                    max_seqlen_k,  // max_seqlen_k_
                                    softmax_scale,
                                    is_causal,
                                    window_size_left,
                                    window_size_right,
                                    softcap,
                                    true,  // is_rotary_interleaved
                                    num_splits,
                                    manual_set_pack_gqa,
                                    pack_gqa_,
                                    sm_margin,
                                    out,
                                    softmax_lse,
                                    &out_accum,
                                    &softmax_lse_accum);
#else
  RaiseNotSupportedError();
#endif
}

}  // namespace phi

PD_REGISTER_KERNEL(flash_attn_v3,
                   GPU,
                   ALL_LAYOUT,
                   phi::FlashAttnV3Kernel,
                   phi::dtype::float16,
                   phi::dtype::bfloat16) {}

PD_REGISTER_KERNEL(flash_attn_v3_varlen,
                   GPU,
                   ALL_LAYOUT,
                   phi::FlashAttnV3VarLenKernel,
                   phi::dtype::float16,
                   phi::dtype::bfloat16) {}
