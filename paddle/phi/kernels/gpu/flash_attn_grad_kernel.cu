// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/phi/kernels/flash_attn_grad_kernel.h"
#include "glog/logging.h"  // For VLOG()
#include "paddle/common/flags.h"
#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/common/bfloat16.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/core/tensor_utils.h"
#include "paddle/phi/kernels/gpu/flash_attn_utils.h"
#include "paddle/phi/kernels/reduce_sum_kernel.h"

COMMON_DECLARE_bool(cudnn_deterministic);

namespace phi {

int get_num_split() {
  // 0 for an internal heuristic, which is optimal
  return FLAGS_cudnn_deterministic ? 1 : 0;
}

template <typename T, typename Context>
void FlashAttnUnpaddedGradKernel(const Context& ctx,
                                 const DenseTensor& q,
                                 const DenseTensor& k,
                                 const DenseTensor& v,
                                 const DenseTensor& cu_seqlens_q,
                                 const DenseTensor& cu_seqlens_k,
                                 const DenseTensor& out,
                                 const DenseTensor& softmax_lse,
                                 const DenseTensor& seed_offset,
                                 const paddle::optional<DenseTensor>& attn_mask,
                                 const DenseTensor& dout,
                                 int64_t max_seqlen_q,
                                 int64_t max_seqlen_k,
                                 float scale,
                                 float dropout,
                                 bool causal,
                                 DenseTensor* dq,
                                 DenseTensor* dk,
                                 DenseTensor* dv) {
#ifdef PADDLE_WITH_FLASHATTN
  // q,k,v [total_*, num_heads, head_dim]
  auto dims = q.dims();

  const int64_t batch_size = cu_seqlens_q.numel() - 1;
  const int64_t num_heads = dims[1];
  const int64_t head_size_og = dout.dims()[2];
  const int64_t head_size = dims[2];
  const int64_t total_k = k.dims()[0];
  const int64_t num_heads_k = k.dims()[1];

  bool is_mha = (num_heads == num_heads_k);

  void* dq_ptr = nullptr;
  void* dk_ptr = nullptr;
  void* dv_ptr = nullptr;

  DenseTensor dq_tmp;
  if (dq) {
    dq_ptr = ctx.template Alloc<T>(dq);
  } else {
    dq_tmp.Resize(dims);
    dq_ptr = ctx.template Alloc<T>(&dq_tmp);
  }

  std::initializer_list<int64_t> dk_dv_shape = {
      total_k, num_heads_k, num_heads / num_heads_k, head_size};

  DenseTensor dk_tmp;
  if (dk && is_mha) {
    ctx.template Alloc<T>(dk);
    dk_ptr = dk->data();
  } else {
    dk_tmp.Resize(dk_dv_shape);
    dk_ptr = ctx.template Alloc<T>(&dk_tmp);
  }

  DenseTensor dv_tmp;
  if (dv && is_mha) {
    ctx.template Alloc<T>(dv);
    dv_ptr = dv->data();
  } else {
    dv_tmp.Resize(dk_dv_shape);
    dv_ptr = ctx.template Alloc<T>(&dv_tmp);
  }

  const cudaStream_t stream = ctx.stream();

  int num_splits = get_num_split();

  // TODO(umiswing): add shape check
  PADDLE_ENFORCE_EQ(
      head_size_og,
      head_size,
      phi::errors::InvalidArgument(
          "flash_attn_bwd receive input with head_size_og == head_size"));

  FlashAttnBwdParamsV2 params =
      FlashAttnBwdParamsV2(ctx,
                           batch_size,
                           max_seqlen_q,
                           max_seqlen_k,
                           num_heads,
                           num_heads_k,
                           head_size,
                           dropout,
                           scale,
                           causal,
                           0,  // attn_mask_start_row
                           q.dtype(),
                           attn_mask,
                           nullptr,  // attn_mask_start_row_indices
                           seed_offset.data<int64_t>());

  VLOG(10) << "FlashAttn bwd seed: " << params.seed
           << ", offset: " << params.offset;

  bool succ = phi::dynload::flash_attn_varlen_bwd(
      dout.data(),
      q.data(),
      k.data(),
      v.data(),
      out.data(),
      params.softmax_d.data(),
      softmax_lse.data(),
      cu_seqlens_q.data<int32_t>(),
      cu_seqlens_k.data<int32_t>(),
      params.rng_state.data(),
      dq_ptr,
      dk_ptr,
      dv_ptr,
      params.dq_accum.data(),
      params.batch_size,
      params.max_seqlen_q,
      params.max_seqlen_k,
      params.seqlen_q_rounded,
      params.seqlen_k_rounded,
      params.num_heads,
      params.num_heads_k,
      params.head_size,
      params.head_size_rounded,
      params.dropout,
      params.softmax_scale,
      1.0f / params.softmax_scale,
      params.causal,
      params.is_bf16,
      num_splits,
      stream,
      params.seed,
      params.offset,
      params.attn_mask_tensor ? params.attn_mask_tensor->data() : nullptr,
      params.attn_mask_tensor ? params.mask_dims.data() : nullptr);
  CheckFlashAttnStatus(succ);
  if (!is_mha) {
    if (dk) {
      phi::SumKernel<T, Context>(ctx, dk_tmp, {2}, dk->type(), false, dk);
    }
    if (dv) {
      phi::SumKernel<T, Context>(ctx, dv_tmp, {2}, dv->type(), false, dv);
    }
  }
#else
  RaiseNotSupportedError();
#endif
}
template <typename T, typename Context>
void FlashAttnGradBaseKernel(
    const Context& ctx,
    const DenseTensor& q,
    const DenseTensor& k,
    const DenseTensor& v,
    const DenseTensor& out,
    const DenseTensor& softmax_lse,
    const DenseTensor& seed_offset,
    const paddle::optional<DenseTensor>& attn_mask,
    const paddle::optional<DenseTensor>& attn_mask_start_row_indices,
    const DenseTensor& dout,
    float dropout,
    bool causal,
    int attn_mask_start_row,
    DenseTensor* dq,
    DenseTensor* dk,
    DenseTensor* dv) {
#ifdef PADDLE_WITH_FLASHATTN
  // q, k, v [batch_size, seq_len, num_heads, head_dim]
  const auto& dims = q.dims();

  const int64_t batch_size = dims[0];
  const int64_t seqlen_q = dims[1];
  const int64_t num_heads = dims[2];
  const int64_t head_size_og = dout.dims()[3];
  const int64_t head_size = dims[3];
  const int64_t seqlen_k = k.dims()[1];
  const int64_t num_heads_k = k.dims()[2];

  bool is_mha = (num_heads == num_heads_k);

  void* dq_ptr = nullptr;
  void* dk_ptr = nullptr;
  void* dv_ptr = nullptr;

  DenseTensor dq_tmp;
  if (dq) {
    dq_ptr = ctx.template Alloc<T>(dq);
  } else {
    dq_tmp.Resize(dims);
    dq_ptr = ctx.template Alloc<T>(&dq_tmp);
  }

  DenseTensor dk_tmp;
  std::initializer_list<int64_t> dk_dv_shape = {
      batch_size, seqlen_k, num_heads_k, num_heads / num_heads_k, head_size};
  if (dk && is_mha) {
    ctx.template Alloc<T>(dk);
    dk_ptr = dk->data();
  } else {
    dk_tmp.Resize(dk_dv_shape);
    dk_ptr = ctx.template Alloc<T>(&dk_tmp);
  }

  DenseTensor dv_tmp;
  if (dv && is_mha) {
    ctx.template Alloc<T>(dv);
    dv_ptr = dv->data();
  } else {
    dv_tmp.Resize(dk_dv_shape);
    dv_ptr = ctx.template Alloc<T>(&dv_tmp);
  }

  const cudaStream_t stream = ctx.stream();

  // TODO(umiswing): add shape check
  PADDLE_ENFORCE_EQ(
      head_size_og,
      head_size,
      phi::errors::InvalidArgument(
          "flash_attn_bwd receive input with head_size_og == head_size"));

  const float softmax_scale = 1.0f / std::sqrt(head_size);
  const float softmax_unscale = std::sqrt(head_size);

  FlashAttnBwdParamsV2 params =
      FlashAttnBwdParamsV2(ctx,
                           batch_size,
                           seqlen_q,
                           seqlen_k,
                           num_heads,
                           num_heads_k,
                           head_size,
                           dropout,
                           softmax_scale,
                           causal,
                           attn_mask_start_row,
                           q.dtype(),
                           attn_mask,
                           attn_mask_start_row_indices,
                           seed_offset.data<int64_t>());

  VLOG(10) << "[FlashAttn Forward] q.shape=[" << q.dims() << "], k.shape=["
           << k.dims() << "], v.shape=[" << v.dims() << "]";
  VLOG(10) << "[FlashAttn Forward] dropout=" << dropout
           << ", seed=" << params.seed << ", offset=" << params.offset;
  VLOG(10) << "[FlashAttn Forward] softmax_scale=" << softmax_scale
           << ", softmax_unscale=" << softmax_unscale;
  if (attn_mask.get_ptr()) {
    VLOG(10) << "[FlashAttn Backward] attn_mask.shape=["
             << (attn_mask.get_ptr())->dims() << "]";
  }

  int num_splits = get_num_split();

  bool succ = phi::dynload::flash_attn_bwd(
      dout.data(),
      q.data(),
      k.data(),
      v.data(),
      out.data(),
      params.softmax_d.data(),
      softmax_lse.data(),
      params.rng_state.data(),
      dq_ptr,
      dk_ptr,
      dv_ptr,
      params.dq_accum.data(),
      params.batch_size,
      params.max_seqlen_q,
      params.max_seqlen_k,
      params.seqlen_q_rounded,
      params.seqlen_k_rounded,
      params.num_heads,
      params.num_heads_k,
      params.head_size,
      params.head_size_rounded,
      params.dropout,
      params.softmax_scale,
      softmax_unscale,
      params.causal,
      params.is_bf16,
      num_splits,
      stream,
      params.seed,
      params.offset,
      params.attn_mask_tensor ? params.attn_mask_tensor->data() : nullptr,
      params.attn_mask_tensor ? params.mask_dims.data() : nullptr,
      params.attn_mask_start_row_indices_tensor
          ? params.attn_mask_start_row_indices_tensor->data()
          : nullptr,
      params.attn_mask_start_row_indices_tensor
          ? params.attn_mask_start_row_indices_dims.data()
          : nullptr,
      params.attn_mask_start_row);
  CheckFlashAttnStatus(succ);
  if (!is_mha) {
    if (dk) {
      phi::SumKernel<T, Context>(ctx, dk_tmp, {3}, dk->type(), false, dk);
    }
    if (dv) {
      phi::SumKernel<T, Context>(ctx, dv_tmp, {3}, dv->type(), false, dv);
    }
  }
#else
  RaiseNotSupportedError();
#endif
}

template <typename T, typename Context>
void FlashAttnGradKernel(const Context& ctx,
                         const DenseTensor& q,
                         const DenseTensor& k,
                         const DenseTensor& v,
                         const DenseTensor& out,
                         const DenseTensor& softmax_lse,
                         const DenseTensor& seed_offset,
                         const paddle::optional<DenseTensor>& attn_mask,
                         const DenseTensor& dout,
                         float dropout,
                         bool causal,
                         DenseTensor* dq,
                         DenseTensor* dk,
                         DenseTensor* dv) {
  FlashAttnGradBaseKernel<T, Context>(ctx,
                                      q,
                                      k,
                                      v,
                                      out,
                                      softmax_lse,
                                      seed_offset,
                                      attn_mask,
                                      paddle::none,
                                      dout,
                                      dropout,
                                      causal,
                                      0,
                                      dq,
                                      dk,
                                      dv);
}

template <typename T, typename Context>
void FlashAttnWithSparseGradKernel(
    const Context& ctx,
    const DenseTensor& q,
    const DenseTensor& k,
    const DenseTensor& v,
    const DenseTensor& attn_mask_start_row_indices,
    const DenseTensor& out,
    const DenseTensor& softmax_lse,
    const DenseTensor& seed_offset,
    const DenseTensor& dout,
    float dropout,
    bool causal,
    int attn_mask_start_row,
    DenseTensor* dq,
    DenseTensor* dk,
    DenseTensor* dv) {
  FlashAttnGradBaseKernel<T, Context>(ctx,
                                      q,
                                      k,
                                      v,
                                      out,
                                      softmax_lse,
                                      seed_offset,
                                      paddle::none,
                                      attn_mask_start_row_indices,
                                      dout,
                                      dropout,
                                      causal,
                                      attn_mask_start_row,
                                      dq,
                                      dk,
                                      dv);
}
}  // namespace phi

PD_REGISTER_KERNEL(flash_attn_unpadded_grad,
                   GPU,
                   ALL_LAYOUT,
                   phi::FlashAttnUnpaddedGradKernel,
                   phi::dtype::float16,
                   phi::dtype::bfloat16) {
  kernel->InputAt(7).SetBackend(phi::Backend::ALL_BACKEND);  // seed_offset
}

PD_REGISTER_KERNEL(flash_attn_grad,
                   GPU,
                   ALL_LAYOUT,
                   phi::FlashAttnGradKernel,
                   phi::dtype::float16,
                   phi::dtype::bfloat16) {
  kernel->InputAt(5).SetBackend(phi::Backend::ALL_BACKEND);  // seed_offset
}

PD_REGISTER_KERNEL(flash_attn_with_sparse_mask_grad,
                   GPU,
                   ALL_LAYOUT,
                   phi::FlashAttnWithSparseGradKernel,
                   phi::dtype::float16,
                   phi::dtype::bfloat16) {
  kernel->InputAt(6).SetBackend(phi::Backend::ALL_BACKEND);  // seed_offset
}
