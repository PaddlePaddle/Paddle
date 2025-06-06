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

#include "paddle/common/enforce.h"
#include "paddle/common/flags.h"
#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/common/bfloat16.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/core/platform/device_context.h"
#include "paddle/phi/core/tensor_utils.h"
#include "paddle/phi/kernels/empty_kernel.h"
#include "paddle/phi/kernels/funcs/math_function.h"
#include "paddle/phi/kernels/gpu/flash_attn_utils.h"
#include "paddle/phi/kernels/gpu/flash_attn_v3_utils.h"

#include "paddle/phi/kernels/concat_kernel.h"
#include "paddle/phi/kernels/gpu/flash_attn_v3_grad_kernel.h"
#include "paddle/phi/kernels/slice_kernel.h"

COMMON_DECLARE_bool(cudnn_deterministic);

namespace phi {

// b: batch_size
// s_q: seqlen_q
// s_k: seqlen_k
// h: num_heads
// h_k: num_heads_k
// d: head_size
template <typename T, typename Context>
void FlashAttnV3GradBaseKernel(
    const Context &ctx,
    const DenseTensor
        &dout,  // (b, s_q, h, dv) or (total_q, h, dv) if there is cu_seqlens_q
    const DenseTensor
        &q,  // (b, s_q, h, d) or (total_q, h, d) if there is cu_seqlens_q
    const DenseTensor
        &k,  // (b, s_k, h_k, d) or (total_k, h_k, d) if there is cu_seqlens_k
    const DenseTensor
        &v,  // (b, s_k, h_k, dv) or (total_k, h_k, dv) if there is cu_seqlens_k
    const DenseTensor
        &out,  // (b, s_q, h, dv) or (total_q, h, dv) if there is cu_seqlens_q
    const DenseTensor
        &softmax_lse,  // (b, h, s_q) or (h, total_q) if there is cu_seqlens_q
    const paddle::optional<DenseTensor>
        &dq_,  // (b, s_q, h, d) or (total_q, h, d) if there is cu_seqlens_q
    const paddle::optional<DenseTensor>
        &dk_,  // (b, s_k, h_k, d) or (total_k, h_k, d) if there is cu_seqlens_k
    const paddle::optional<DenseTensor>
        &dv_,  // (b, s_k, h_k, dv) or (total_k, h_k, dv) if there is
               // cu_seqlens_k
    const paddle::optional<DenseTensor> &cu_seqlens_q_,  // b+1
    const paddle::optional<DenseTensor> &cu_seqlens_k_,  // b+1
    const paddle::optional<DenseTensor>
        &seqused_q_,  // b. If given, only this many elements of each batch
                      // element's queries and outputs are used.
    const paddle::optional<DenseTensor>
        &seqused_k_,  // b. If given, only this many elements of each batch
                      // element's keys are used.
    int max_seqlen_q_,
    int max_seqlen_k_,
    float const softmax_scale,
    bool is_causal,
    int window_size_left,
    int window_size_right,
    float const softcap,
    bool const deterministic,
    int const sm_margin,
    DenseTensor *dq,
    DenseTensor *dk,
    DenseTensor *dv,
    DenseTensor *softmax_d,
    DenseTensor *softmax_lse_log2,
    DenseTensor *dq_accum,
    DenseTensor *dk_accum,
    DenseTensor *dv_accum) {
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
      (q_type == phi::DataType::FLOAT16 || q_type == phi::DataType::BFLOAT16),
      true,
      common::errors::InvalidArgument(
          "FlashAttention-3 bwd only support fp16 and bf16 data type"));
  PADDLE_ENFORCE_EQ(k.dtype(),
                    q_type,
                    common::errors::InvalidArgument(
                        "query and key must have the same dtype"));
  PADDLE_ENFORCE_EQ(v.dtype(),
                    q_type,
                    common::errors::InvalidArgument(
                        "query and value must have the same dtype"));
  PADDLE_ENFORCE_EQ(out.dtype(),
                    q_type,
                    common::errors::InvalidArgument(
                        "query and out must have the same dtype"));
  PADDLE_ENFORCE_EQ(dout.dtype(),
                    q_type,
                    common::errors::InvalidArgument(
                        "query and dout must have the same dtype"));

  CHECK_DEVICE(q);
  CHECK_DEVICE(k);
  CHECK_DEVICE(v);
  CHECK_DEVICE(out);
  CHECK_DEVICE(dout);
  CHECK_DEVICE(softmax_lse);

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
  PADDLE_ENFORCE_EQ(out.strides()[out.strides().size() - 1],
                    1,
                    common::errors::InvalidArgument(
                        "out tensor must have contiguous last dimension"));
  PADDLE_ENFORCE_EQ(dout.strides()[dout.strides().size() - 1],
                    1,
                    common::errors::InvalidArgument(
                        "dout tensor must have contiguous last dimension"));

  DenseTensor cu_seqlens_q;
  bool const is_varlen_q = cu_seqlens_q_.is_initialized();
  if (is_varlen_q) {
    cu_seqlens_q = cu_seqlens_q_.get();
    CHECK_DEVICE(cu_seqlens_q);
    CHECK_CONTIGUOUS(cu_seqlens_q);
    PADDLE_ENFORCE_EQ(cu_seqlens_q.dtype(),
                      phi::DataType::INT32,
                      common::errors::InvalidArgument(
                          "cu_seqlens_q must have dtype paddle.int32"));
    PADDLE_ENFORCE_GT(
        max_seqlen_q_,
        0,
        common::errors::InvalidArgument(
            "max_seqlen_q must be provided if cu_seqlens_q is provided"));
  }
  DenseTensor cu_seqlens_k;
  bool const is_varlen_k = cu_seqlens_k_.is_initialized();
  if (is_varlen_k) {
    cu_seqlens_k = cu_seqlens_k_.get();
    CHECK_DEVICE(cu_seqlens_k);
    CHECK_CONTIGUOUS(cu_seqlens_k);
    PADDLE_ENFORCE_EQ(cu_seqlens_k.dtype(),
                      phi::DataType::INT32,
                      common::errors::InvalidArgument(
                          "cu_seqlens_k must have dtype paddle.int32"));
    PADDLE_ENFORCE_GT(
        max_seqlen_k_,
        0,
        common::errors::InvalidArgument(
            "max_seqlen_k must be provided if cu_seqlens_k is provided"));
  }
  // This is what we will template on
  bool const is_varlen = is_varlen_q || is_varlen_k ||
                         seqused_q_.is_initialized() ||
                         seqused_k_.is_initialized();
#ifdef FLASHATTENTION_DISABLE_VARLEN
  PADDLE_ENFORCE_EQ(!is_varlen,
                    true,
                    common::errors::Unavailable(
                        "This flash attention build does not support varlen."));
#endif

  auto const sizes = q.dims();
  int const batch_size = !is_varlen_q ? sizes[0] : cu_seqlens_q.dims()[0] - 1;
  int const seqlen_q = !is_varlen_q ? sizes[1] : max_seqlen_q_;
  int const total_q = !is_varlen_q ? batch_size * sizes[1] : sizes[0];
  int const num_heads = q.dims()[q.dims().size() - 2];
  int const head_size = q.dims()[q.dims().size() - 1];
  int const head_size_v = v.dims()[v.dims().size() - 1];
  int const seqlen_k = !is_varlen_k ? k.dims()[1] : max_seqlen_k_;
  int const total_k = !is_varlen_k ? batch_size * k.dims()[1] : k.dims()[0];
  int const num_heads_k = k.dims()[k.dims().size() - 2];
  PADDLE_ENFORCE_EQ(
      head_size % 8,
      0,
      common::errors::InvalidArgument("head_size should be a multiple of 8"));
  int const max_headdim = get_max_headdim();
  PADDLE_ENFORCE_EQ(
      head_size_v % 8,
      0,
      common::errors::InvalidArgument("head_size_v should be a multiple of 8"));
  PADDLE_ENFORCE_LE(
      std::max(head_size, head_size_v),
      max_headdim,
      common::errors::InvalidArgument(
          "FlashAttention forward only supports head dimension at most %d",
          max_headdim));
  PADDLE_ENFORCE_EQ(
      num_heads % num_heads_k,
      0,
      common::errors::InvalidArgument(
          "Number of heads in key/value must divide number of heads in query"));

  // This needs to go before kBlockM & kBlockN since we rely on the correct
  // window_size and is_causal to set kBlockM
  if (window_size_left >= seqlen_k - 1) {
    window_size_left = -1;
  }
  if (window_size_right >= seqlen_q - 1) {
    window_size_right = -1;
  }
  if (is_causal) {
    window_size_right = 0;
  }
  // There's a case where is_causal=false, window_size=(-1, 0). Then
  // set_params_bprop will set params.is_causal=true. If we don't have is_causal
  // here matching params.is_causal, we might get the wrong kBlockM (and cause
  // IMA).
  is_causal = window_size_left < 0 && window_size_right == 0;

  int const arch = dprops.major * 10 + dprops.minor;
  int const head_size_rounded =
      round_up_headdim(std::max(head_size, head_size_v));
  int const head_size_v_rounded = head_size_rounded;
  // Very important that these match the kernel configs
  bool const is_local =
      (window_size_left >= 0 || window_size_right >= 0) && !is_causal;
  int const kBlockM_sm90 =
      head_size_rounded <= 64
          ? (is_causal && softcap > 0.0 ? 96 : 128)
          : (head_size_rounded <= 96
                 ? 64
                 : (head_size_rounded <= 128
                        ? (is_causal || is_local || softcap > 0.0 ? 64 : 80)
                        : 64));
  int const kBlockM_sm80 = head_size_rounded <= 64 ? 128 : 64;
  int const kBlockM_sm86 = head_size_rounded <= 192 ? 64 : 32;
  int const kBlockM =
      arch >= 90 ? kBlockM_sm90
                 : (arch == 86 || arch == 89 ? kBlockM_sm86 : kBlockM_sm80);
  int const kBlockN_sm90 =
      head_size_rounded <= 128 ? 128 : (head_size_rounded <= 192 ? 96 : 80);
  int const kBlockN_sm80 =
      head_size_rounded <= 128 ? 128 : (head_size_rounded <= 192 ? 80 : 64);
  int const kBlockN_sm86 =
      head_size_rounded <= 64
          ? 128
          : (head_size_rounded <= 96
                 ? 128
                 : (head_size_rounded <= 128
                        ? 96
                        : (head_size_rounded <= 192 ? 64 : 64)));
  int const kBlockN =
      arch >= 90 ? kBlockN_sm90
                 : (arch == 86 || arch == 89 ? kBlockN_sm86 : kBlockN_sm80);
  auto round_multiple = [](int x, int m) { return (x + m - 1) / m * m; };
  int const seqlen_q_rounded = round_multiple(seqlen_q, kBlockM);
  int const seqlen_k_rounded = round_multiple(seqlen_k, kBlockN);
  int const total_q_padded_rounded =
      round_multiple(total_q + batch_size * kBlockM, kBlockM);
  int const total_k_padded_rounded =
      round_multiple(total_k + batch_size * kBlockN, kBlockN);

  if (!is_varlen_q) {
    CHECK_SHAPE(q, batch_size, seqlen_q, num_heads, head_size);
    CHECK_SHAPE(out, batch_size, seqlen_q, num_heads, head_size_v);
    CHECK_SHAPE(dout, batch_size, seqlen_q, num_heads, head_size_v);
  } else {
    CHECK_SHAPE(q, total_q, num_heads, head_size);
    CHECK_SHAPE(out, total_q, num_heads, head_size_v);
    CHECK_SHAPE(dout, total_q, num_heads, head_size_v);
    CHECK_SHAPE(cu_seqlens_q, batch_size + 1);
  }
  if (!is_varlen_k) {
    CHECK_SHAPE(k, batch_size, seqlen_k, num_heads_k, head_size);
    CHECK_SHAPE(v, batch_size, seqlen_k, num_heads_k, head_size_v);
  } else {
    CHECK_SHAPE(k, total_k, num_heads_k, head_size);
    CHECK_SHAPE(v, total_k, num_heads_k, head_size_v);
    CHECK_SHAPE(cu_seqlens_k, batch_size + 1);
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

  if (dq_.is_initialized()) {
    *dq = dq_.get();
    PADDLE_ENFORCE_EQ(
        dq->dtype(),
        q_type,
        common::errors::InvalidArgument("dq must have the same dtype as q"));
    CHECK_DEVICE((*dq));
    PADDLE_ENFORCE_EQ(dq->strides()[dq->strides().size() - 1],
                      1,
                      common::errors::InvalidArgument(
                          "dq must have contiguous last dimension"));
    if (!is_varlen_q) {
      CHECK_SHAPE((*dq), batch_size, seqlen_q, num_heads, head_size);
    } else {
      CHECK_SHAPE((*dq), total_q, num_heads, head_size);
    }
  } else {
    *dq = phi::EmptyLike<T, Context>(ctx, q);
  }
  if (dk_.is_initialized()) {
    *dk = dk_.get();
    PADDLE_ENFORCE_EQ(
        dk->dtype(),
        q_type,
        common::errors::InvalidArgument("dk must have the same dtype as q"));
    CHECK_DEVICE((*dk));
    PADDLE_ENFORCE_EQ(dk->strides()[dk->strides().size() - 1],
                      1,
                      common::errors::InvalidArgument(
                          "dk must have contiguous last dimension"));
    if (!is_varlen_k) {
      CHECK_SHAPE((*dk), batch_size, seqlen_k, num_heads_k, head_size);
    } else {
      CHECK_SHAPE((*dk), total_k, num_heads_k, head_size);
    }
  } else {
    *dk = phi::EmptyLike<T, Context>(ctx, k);
  }
  if (dv_.is_initialized()) {
    *dv = dv_.get();
    PADDLE_ENFORCE_EQ(
        dv->dtype(),
        q_type,
        common::errors::InvalidArgument("dv must have the same dtype as q"));
    CHECK_DEVICE((*dv));
    PADDLE_ENFORCE_EQ(dv->strides()[dv->strides().size() - 1],
                      1,
                      common::errors::InvalidArgument(
                          "dv must have contiguous last dimension"));
    if (!is_varlen_k) {
      CHECK_SHAPE((*dv), batch_size, seqlen_k, num_heads_k, head_size_v);
    } else {
      CHECK_SHAPE((*dv), total_k, num_heads_k, head_size_v);
    }
  } else {
    *dv = phi::EmptyLike<T, Context>(ctx, v);
  }

  // Otherwise the kernel will be launched from cuda:0 device
  // Cast to char to avoid compiler warning about narrowing

  // Need softmax_d to have total_q_padded_rounded since we want its address to
  // be aligned by 16/8 bytes for TMA / LDG.64
  if (!is_varlen) {
    if (softmax_d) {
      // Need softmax_d to have seqlen_q_rounded since we want its address to be
      // aligned by 16/8 bytes for TMA / LDG.64
      softmax_d->Resize(
          common::make_ddim({batch_size, num_heads, seqlen_q_rounded}));
    }
    if (softmax_lse_log2) {
      softmax_lse_log2->Resize(
          common::make_ddim({batch_size, num_heads, seqlen_q_rounded}));
    }
  } else {
    if (softmax_d) {
      softmax_d->Resize(common::make_ddim({num_heads, total_q_padded_rounded}));
    }
    if (softmax_lse_log2) {
      softmax_lse_log2->Resize(
          common::make_ddim({num_heads, total_q_padded_rounded}));
    }
  }
  if (softmax_d) {
    ctx.template Alloc<float>(softmax_d);
  }
  if (softmax_lse_log2) {
    ctx.template Alloc<float>(softmax_lse_log2);
  }
  if (dq_accum) {
    if (!is_varlen) {
      dq_accum->Resize(common::make_ddim(
          {batch_size, num_heads, seqlen_q_rounded * head_size_rounded}));
    } else {
      dq_accum->Resize(common::make_ddim(
          {num_heads, total_q_padded_rounded * head_size_rounded}));
    }
    ctx.template Alloc<float>(dq_accum);
  }
  if (num_heads_k != num_heads) {  // MQA / GQA
    if (!is_varlen) {
      if (dk_accum) {
        dk_accum->Resize(common::make_ddim(
            {batch_size, num_heads_k, seqlen_k_rounded * head_size_rounded}));
      }
      if (dv_accum) {
        dv_accum->Resize(common::make_ddim(
            {batch_size, num_heads_k, seqlen_k_rounded * head_size_v_rounded}));
      }
    } else {
      if (dk_accum) {
        dk_accum->Resize(common::make_ddim(
            {num_heads_k, total_k_padded_rounded, head_size_rounded}));
      }
      if (dv_accum) {
        dv_accum->Resize(common::make_ddim(
            {num_heads_k, total_k_padded_rounded, head_size_v_rounded}));
      }
    }
    if (dk_accum) {
      ctx.template Alloc<float>(dk_accum);
    }
    if (dv_accum) {
      ctx.template Alloc<float>(dv_accum);
    }
    phi::funcs::SetConstant<Context, float> set_zero;

    if (dk_accum) {
      set_zero(ctx, dk_accum, float{0});
    }
    if (dv_accum) {
      set_zero(ctx, dv_accum, float{0});
    }
  }

  Flash_bwd_params *params_handle = get_flash_bwd_params_handle();
  dynload::fa3_clear_bwd_params_handle(params_handle);
  set_params_dgrad(
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
      dout,
      dq,
      dk,
      dv,
      !is_varlen_q ? nullptr : cu_seqlens_q.data(),
      !is_varlen_k ? nullptr : cu_seqlens_k.data(),
      seqused_q_.is_initialized() ? const_cast<void *>(seqused_q_.get().data())
                                  : nullptr,
      seqused_k_.is_initialized() ? const_cast<void *>(seqused_k_.get().data())
                                  : nullptr,
      dq_accum ? dq_accum->data() : nullptr,
      num_heads_k != num_heads && dk_accum ? dk_accum->data() : nullptr,
      num_heads_k != num_heads && dv_accum ? dv_accum->data() : nullptr,
      const_cast<void *>(softmax_lse.data()),
      softmax_d ? const_cast<void *>(softmax_d->data()) : nullptr,
      /*p_dropout=*/0.f,
      softmax_scale,
      window_size_left,
      window_size_right,
      dprops,
      softcap,
      deterministic,
      sm_margin);
  dynload::fa3_bwd_params_set_total_q(params_handle, total_q);
  dynload::fa3_bwd_params_set_total_k(params_handle, total_k);
  dynload::fa3_bwd_params_set_softmax_lse_log2_ptr(
      params_handle, softmax_lse_log2 ? softmax_lse_log2->data() : nullptr);
  dynload::fa3_bwd_params_set_dv(params_handle, head_size_v);
  dynload::fa3_bwd_params_set_dv_rounded(params_handle, head_size_v_rounded);
  // auto tile_count_semaphore = (params.is_causal || params.is_local) ?
  // paddle::zeros({1}, opts.dtype(torch::kInt32)) : torch::empty({1},
  // opts.dtype(torch::kInt32)); params.tile_count_semaphore =
  // tile_count_semaphore.data_ptr<int>(); Will be zero'ed out in the backward
  // preprocess kernel
  DenseTensor dq_semaphore = phi::Empty<int32_t>(
      ctx, {(seqlen_q + kBlockM - 1) / kBlockM, batch_size, num_heads});
  dynload::fa3_bwd_params_set_dq_semaphore(params_handle,
                                           dq_semaphore.data<int>());
  if (num_heads_k != num_heads &&
      dynload::fa3_bwd_params_get_deterministic(params_handle)) {
    // TODO(tridao): do we need to zero them out?
    DenseTensor dk_semaphore = phi::Empty<int32_t>(
        ctx, {(seqlen_k + kBlockN - 1) / kBlockN, batch_size, num_heads_k});
    DenseTensor dv_semaphore = phi::Empty<int32_t>(
        ctx, {(seqlen_k + kBlockN - 1) / kBlockN, batch_size, num_heads_k});
    dynload::fa3_bwd_params_set_dk_semaphore(params_handle,
                                             dk_semaphore.data<int>());
    dynload::fa3_bwd_params_set_dv_semaphore(params_handle,
                                             dv_semaphore.data<int>());
  }

#ifdef FLASHATTENTION_DISABLE_LOCAL
  PADDLE_ENABLE_EQ(
      !dynload::fa3_bwd_params_get_is_local(params_handle),
      true,
      "This flash attention build does not support local attention.");
#endif
#ifdef FLASHATTENTION_DISABLE_SOFTCAP
  PADDLE_ENABLE_EQ(
      dynload::fa3_bwd_params_get_softcap(params_handle),
      0.0,
      "This flash attention build does not support tanh softcapping.");
#endif

  if (total_q > 0 && total_k > 0 && num_heads_k > 0) {
    dynload::fa3_run_mha_bwd(params_handle, ctx.stream());
  } else if (total_k > 0 && num_heads_k > 0) {
    // If seqlen_q == 0, then we have an empty tensor. We need to set the output
    // to 0.
    phi::funcs::SetConstant<Context, T> set_zero;
    set_zero(ctx, dk, T{0});
    set_zero(ctx, dv, T{0});
    if (softmax_d) {
      phi::funcs::SetConstant<Context, float> set_zero_fp32;
      set_zero_fp32(ctx, softmax_d, float{0});
    }
  } else if (total_q > 0 && num_heads_k > 0) {
    phi::funcs::SetConstant<Context, T> set_zero;
    set_zero(ctx, dq, T{0});
    if (softmax_d) {
      phi::funcs::SetConstant<Context, float> set_zero_fp32;
      set_zero_fp32(ctx, softmax_d, float{0});
    }
  }
#else
  RaiseNotSupportedError();
#endif
}

template <typename T, typename Context>
void FlashAttnV3GradKernel(const Context &ctx,
                           const DenseTensor &q,
                           const DenseTensor &k,
                           const DenseTensor &v,
                           const DenseTensor &out,
                           const DenseTensor &softmax_lse,
                           const DenseTensor &out_grad,
                           float const softmax_scale,
                           bool is_causal,
                           int window_size_left,
                           int window_size_right,
                           float const softcap,
                           int const sm_margin,
                           DenseTensor *dq,
                           DenseTensor *dk,
                           DenseTensor *dv) {
#ifdef PADDLE_WITH_FLASHATTN_V3
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
      sm_margin,
      0,
      common::errors::InvalidArgument(
          "sm_margin is not supported, please set sm_margin to 0"));
  PADDLE_ENFORCE_EQ(FLAGS_cudnn_deterministic,
                    false,
                    common::errors::InvalidArgument(
                        "deterministic is not supported in flash attention 3, "
                        "please set FLAGS_cudnn_deterministic to false"));
  // umiswing: fake grad tensor for FlashAttnV3GradBaseKernel
  DenseTensor softmax_d;
  DenseTensor softmax_lse_log2;
  DenseTensor dq_accum;
  DenseTensor dk_accum;
  DenseTensor dv_accum;
  const int64_t b = q.dims()[0];
  const int64_t s_q = q.dims()[1];
  const int64_t s_k = k.dims()[1];
  const int64_t h_q = q.dims()[2];
  const int64_t h_k = k.dims()[2];
  const int64_t d_q = q.dims()[3];
  const int64_t d_v = v.dims()[3];
  if (q.dims()[q.dims().size() - 1] > v.dims()[v.dims().size() - 1]) {
    PADDLE_ENFORCE_EQ(v.dims()[v.dims().size() - 1],
                      out.dims()[out.dims().size() - 1],
                      common::errors::InvalidArgument(
                          "head_dim_v and head_dim_o must be equal"));
    PADDLE_ENFORCE_EQ(v.dims()[v.dims().size() - 2],
                      out.dims()[out.dims().size() - 2],
                      common::errors::InvalidArgument(
                          "num_heads_v and num_heads_o must be equal"));
    PADDLE_ENFORCE_EQ(
        v.dims()[v.dims().size() - 3],
        out.dims()[out.dims().size() - 3],
        common::errors::InvalidArgument("seqlen_v and seqlen_o must be equal"));
  }
  FlashAttnV3GradBaseKernel<T, Context>(ctx,
                                        out_grad,
                                        q,
                                        k,
                                        v,
                                        out,
                                        softmax_lse,
                                        paddle::none,
                                        paddle::none,
                                        paddle::none,
                                        paddle::none,
                                        paddle::none,
                                        paddle::none,
                                        paddle::none,
                                        0,
                                        0,
                                        softmax_scale,
                                        is_causal,
                                        window_size_left,
                                        window_size_right,
                                        softcap,
                                        FLAGS_cudnn_deterministic,
                                        sm_margin,
                                        dq,
                                        dk,
                                        dv,
                                        &softmax_d,
                                        &softmax_lse_log2,
                                        &dq_accum,
                                        &dk_accum,
                                        &dv_accum);
#else
  RaiseNotSupportedError();
#endif
}

}  // namespace phi

PD_REGISTER_KERNEL(flash_attn_v3_grad,
                   GPU,
                   ALL_LAYOUT,
                   phi::FlashAttnV3GradKernel,
                   phi::dtype::float16,
                   phi::dtype::bfloat16) {}
