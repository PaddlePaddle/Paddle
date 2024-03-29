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
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/core/tensor_utils.h"
#include "paddle/phi/kernels/funcs/elementwise_base.h"
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
                           q.dtype(),
                           attn_mask,
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
      params.attn_mask_tensor ? params.mask_dims.data() : nullptr,
      q.strides()[0],
      k.strides()[0],
      v.strides()[0],
      q.strides()[1],
      k.strides()[1],
      v.strides()[1],
      out.strides()[0],
      out.strides()[1],
      max_seqlen_q * q.strides()[0],
      max_seqlen_k * k.strides()[0],
      max_seqlen_k * v.strides()[0],
      max_seqlen_q * out.strides()[0],
      dq->strides()[0],
      dk->strides()[0],
      dv->strides()[0],
      dq->strides()[1],
      dk->strides()[1],
      dv->strides()[1],
      dout.strides()[0],
      dout.strides()[1],
      0,
      0,
      0,
      0,
      false /*varlen_padded_input*/);
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

template <typename OutT>
struct ZeroFunctor {
  __device__ __forceinline__ OutT operator()() const {
    return static_cast<OutT>(0);
  }
};
template <typename T, typename Context>
void FlashAttnUnpaddedQKVPackedGradKernel(
    const Context& ctx,
    const DenseTensor& qkv,
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
    DenseTensor* dqkv) {
#ifdef PADDLE_WITH_FLASHATTN
  // q,k,v [total_*, num_heads, head_dim]
  const auto head_groupnum = qkv.dims()[1];  // nheads/nheads_k + 1 + 1
  DenseTensor q(
      qkv.Holder(),
      DenseTensorMeta{
          qkv.dtype(),
          DDim{qkv.dims()[0], qkv.dims()[2], qkv.dims()[3]},
          DDim{qkv.strides()[0], qkv.strides()[2], qkv.strides()[3]}});
  q.set_offset(qkv.offset());
  DenseTensor k(
      qkv.Holder(),
      DenseTensorMeta{
          qkv.dtype(),
          DDim{qkv.dims()[0], qkv.dims()[2], qkv.dims()[3]},
          DDim{qkv.strides()[0], qkv.strides()[2], qkv.strides()[3]}});
  k.set_offset(qkv.offset() +
               (head_groupnum - 2) * qkv.strides()[1] * SizeOf(qkv.dtype()));
  DenseTensor v(
      qkv.Holder(),
      DenseTensorMeta{
          qkv.dtype(),
          DDim{qkv.dims()[0], qkv.dims()[2], qkv.dims()[3]},
          DDim{qkv.strides()[0], qkv.strides()[2], qkv.strides()[3]}});
  v.set_offset(qkv.offset() +
               (head_groupnum - 1) * qkv.strides()[1] * SizeOf(qkv.dtype()));
  DenseTensor dqkv_tmp;
  if (!dqkv) {
    dqkv_tmp.Resize(qkv.dims());
    dqkv = &dqkv_tmp;
  }
  ctx.template Alloc<T>(dqkv);
  {
    std::vector<const DenseTensor*> inputs{};
    std::vector<DenseTensor*> outputs{dqkv};
    phi::funcs::ElementwiseKernel<T>(ctx, inputs, &outputs, ZeroFunctor<T>());
  }
  DenseTensor dq(
      dqkv->Holder(),
      DenseTensorMeta{
          dqkv->dtype(),
          DDim{dqkv->dims()[0], dqkv->dims()[2], dqkv->dims()[3]},
          DDim{dqkv->strides()[0], dqkv->strides()[2], dqkv->strides()[3]}});
  dq.set_offset(dqkv->offset());
  DenseTensor dk(
      dqkv->Holder(),
      DenseTensorMeta{
          dqkv->dtype(),
          DDim{dqkv->dims()[0], dqkv->dims()[2], dqkv->dims()[3]},
          DDim{dqkv->strides()[0], dqkv->strides()[2], dqkv->strides()[3]}});
  dk.set_offset(dqkv->offset() +
                (head_groupnum - 2) * qkv.strides()[1] * SizeOf(dqkv->dtype()));
  DenseTensor dv(
      dqkv->Holder(),
      DenseTensorMeta{
          dqkv->dtype(),
          DDim{dqkv->dims()[0], dqkv->dims()[2], dqkv->dims()[3]},
          DDim{dqkv->strides()[0], dqkv->strides()[2], dqkv->strides()[3]}});
  dv.set_offset(dqkv->offset() +
                (head_groupnum - 1) * qkv.strides()[1] * SizeOf(dqkv->dtype()));

  auto dims = q.dims();
  const int64_t batch_size = cu_seqlens_q.numel() - 1;
  const int64_t num_heads = dims[1];
  const int64_t head_size_og = dout.dims()[2];
  const int64_t head_size = dims[2];
  const int64_t total_k = k.dims()[0];
  const int64_t num_heads_k = k.dims()[1];

  bool is_mha = (num_heads == num_heads_k);

  std::initializer_list<int64_t> dk_dv_shape = {
      total_k, num_heads_k, num_heads / num_heads_k, head_size};

  DenseTensor *kdk = &dk, *kdv = &dv;
  DenseTensor dk_tmp;
  if (!is_mha) {
    dk_tmp.Resize(dk_dv_shape);
    ctx.template Alloc<T>(&dk_tmp);
    kdk = &dk_tmp;
  }

  DenseTensor dv_tmp;
  if (!is_mha) {
    dv_tmp.Resize(dk_dv_shape);
    ctx.template Alloc<T>(&dv_tmp);
    kdv = &dv_tmp;
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
                           q.dtype(),
                           attn_mask,
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
      dq.data(),
      kdk->data(),
      kdv->data(),
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
      params.attn_mask_tensor ? params.mask_dims.data() : nullptr,
      q.strides()[0],
      k.strides()[0],
      v.strides()[0],
      q.strides()[1],
      k.strides()[1],
      v.strides()[1],
      out.strides()[0],
      out.strides()[1],
      max_seqlen_q * q.strides()[0],
      max_seqlen_k * k.strides()[0],
      max_seqlen_k * v.strides()[0],
      max_seqlen_q * out.strides()[0],
      dq.strides()[0],
      kdk->strides()[0],
      kdv->strides()[0],
      dq.strides()[1],
      kdk->strides()[1],
      kdv->strides()[1],
      dout.strides()[0],
      dout.strides()[1],
      max_seqlen_q * dq.strides()[0],
      max_seqlen_k * kdk->strides()[0],
      max_seqlen_k * kdv->strides()[0],
      max_seqlen_q * dout.strides()[0],
      true /*varlen_padded_input*/);
  CheckFlashAttnStatus(succ);
  if (!is_mha) {
    phi::SumKernel<T, Context>(ctx, dk_tmp, {2}, dk.type(), false, &dk);
    phi::SumKernel<T, Context>(ctx, dv_tmp, {2}, dv.type(), false, &dv);
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
                           q.dtype(),
                           attn_mask,
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
      params.attn_mask_tensor ? params.mask_dims.data() : nullptr);
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

}  // namespace phi

PD_REGISTER_KERNEL(flash_attn_unpadded_grad,
                   GPU,
                   ALL_LAYOUT,
                   phi::FlashAttnUnpaddedGradKernel,
                   phi::dtype::float16,
                   phi::dtype::bfloat16) {
  kernel->InputAt(7).SetBackend(phi::Backend::ALL_BACKEND);  // seed_offset
}

PD_REGISTER_KERNEL(flash_attn_unpadded_qkvpacked_grad,
                   GPU,
                   ALL_LAYOUT,
                   phi::FlashAttnUnpaddedQKVPackedGradKernel,
                   phi::dtype::float16,
                   phi::dtype::bfloat16) {
  kernel->InputAt(5).SetBackend(phi::Backend::ALL_BACKEND);  // seed_offset
}

PD_REGISTER_KERNEL(flash_attn_grad,
                   GPU,
                   ALL_LAYOUT,
                   phi::FlashAttnGradKernel,
                   phi::dtype::float16,
                   phi::dtype::bfloat16) {
  kernel->InputAt(5).SetBackend(phi::Backend::ALL_BACKEND);  // seed_offset
}
