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
#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/common/bfloat16.h"
#include "paddle/phi/core/flags.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/core/tensor_utils.h"
#include "paddle/phi/kernels/arange_kernel.h"
#include "paddle/phi/kernels/empty_kernel.h"
#include "paddle/phi/kernels/reshape_kernel.h"

#ifdef PADDLE_WITH_FLASHATTN
#include "paddle/phi/backends/dynload/flashattn.h"
#include "paddle/phi/kernels/gpu/flash_attn_utils.h"
#endif

DECLARE_bool(cudnn_deterministic);

namespace phi {

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
  ctx.template Alloc<T>(dq);
  ctx.template Alloc<T>(dk);
  ctx.template Alloc<T>(dv);

  const cudaStream_t stream = ctx.stream();
#if 0
  const bool is_bf16 = q.dtype() == DataType::BFLOAT16 ? true : false;
#endif

  // q,k,v [total_*, num_heads, head_dim]

  auto dims = q.dims();
  const int64_t total_q = dims[0];
  const int batch_size = cu_seqlens_q.numel() - 1;
  const int num_heads = dims[1];
  const int head_size_og = dout.dims()[2];
  const int head_size = dims[2];
  const int total_k = k.dims()[0];
  const int num_heads_k = k.dims()[1];

  const bool zero_tensors = false;

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
                           seed_offset.data<int64_t>());

#if 0
  const int64_t* seed_offset_data = seed_offset.data<int64_t>();
  uint64_t seed = static_cast<uint64_t>(seed_offset_data[0]);
  uint64_t offset = static_cast<uint64_t>(seed_offset_data[1]);
#endif

  VLOG(4) << "FlashAttn bwd seed: " << params.seed
          << ", offset: " << params.offset;

#if 0
  auto round_multiple = [](int x, int m) { return (x + m - 1) / m * m; };
  const int head_size_rounded = round_multiple(head_size, 32);
  const int seqlen_q_rounded = round_multiple(max_seqlen_q, 128);
  const int seqlen_k_rounded = round_multiple(max_seqlen_k, 128);

  DenseTensor softmax_d =
      Empty<float>(ctx, {batch_size, num_heads, seqlen_q_rounded});
  DenseTensor dq_accum = Empty<float>(
      ctx, {batch_size, num_heads, seqlen_q_rounded, head_size_rounded});
#endif

  const bool succ =
      phi::dynload::flash_attn_varlen_bwd(dout.data(),
                                          q.data(),
                                          k.data(),
                                          v.data(),
                                          out.data(),
                                          params.softmax_d.data(),
                                          softmax_lse.data(),
                                          dq->data(),
                                          dk->data(),
                                          dv->data(),
                                          params.dq_accum.data(),
                                          cu_seqlens_q.data<int32_t>(),
                                          cu_seqlens_k.data<int32_t>(),
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
                                          params.scale,
                                          params.causal,
                                          params.is_bf16,
                                          stream,
                                          params.seed,
                                          params.offset);

  if (!succ) {
    PADDLE_THROW(phi::errors::External(phi::dynload::flash_attn_error()));
  }

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
                         const DenseTensor& dout,
                         float dropout,
                         bool causal,
                         DenseTensor* dq,
                         DenseTensor* dk,
                         DenseTensor* dv) {
#ifdef PADDLE_WITH_FLASHATTN
  // q,k,v [batch_size, seq_len, num_heads, head_dim]

  auto dims = q.dims();
  const int batch_size = dims[0];
  const int seqlen_q = dims[1];
  const int num_heads = dims[2];
  const int head_size_og = dout.dims()[3];
  const int head_size = dims[3];
  const int seqlen_k = k.dims()[1];
  const int num_heads_k = k.dims()[2];

#if 0
  auto round_multiple = [](int x, int m) { return (x + m - 1) / m * m; };
  const int head_size_rounded = round_multiple(head_size, 32);
  const int seqlen_q_rounded = round_multiple(seqlen_q, 128);
  const int seqlen_k_rounded = round_multiple(seqlen_k, 128);
#endif

  // TODO(umiswing): add shape check
  PADDLE_ENFORCE_EQ(
      head_size_og,
      head_size,
      phi::errors::InvalidArgument(
          "flash_attn_bwd receive input with head_size_og == head_size"));

  VLOG(4) << "FlashAttn bwd dims q[" << q.dims() << "], k[" << k.dims()
          << "], v[" << v.dims() << "]";

  const float scale = 1.0f / std::sqrt(head_size);

  FlashAttnBwdParamsV2 params =
      FlashAttnBwdParamsV2(ctx,
                           batch_size,
                           seqlen_q,
                           seqlen_k,
                           num_heads,
                           num_heads_k,
                           head_size,
                           dropout,
                           scale,
                           causal,
                           q.dtype(),
                           seed_offset.data<int64_t>());

  ctx.template Alloc<T>(dq);
  ctx.template Alloc<T>(dk);
  ctx.template Alloc<T>(dv);

  cudaStream_t stream = ctx.stream();
#if 0
  bool is_bf16 = q.dtype() == DataType::BFLOAT16 ? true : false;

  const int64_t* seed_offset_data = seed_offset.data<int64_t>();
  uint64_t seed = static_cast<uint64_t>(seed_offset_data[0]);
  uint64_t offset = static_cast<uint64_t>(seed_offset_data[1]);

  DenseTensor softmax_d =
      Empty<float>(ctx, {batch_size, num_heads, seqlen_q_rounded});
  DenseTensor dq_accum = Empty<float>(
      ctx, {batch_size, num_heads, seqlen_q_rounded, head_size_rounded});
#endif

  VLOG(4) << "FlashAttn bwd seed: " << params.seed
          << ", offset: " << params.offset;

  const bool succ = phi::dynload::flash_attn_bwd(dout.data(),
                                                 q.data(),
                                                 k.data(),
                                                 v.data(),
                                                 out.data(),
                                                 params.softmax_d.data(),
                                                 softmax_lse.data(),
                                                 dq->data(),
                                                 dk->data(),
                                                 dv->data(),
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
                                                 params.scale,
                                                 params.causal,
                                                 params.is_bf16,
                                                 stream,
                                                 params.seed,
                                                 params.offset);

  if (!succ) {
    PADDLE_THROW(phi::errors::External(phi::dynload::flash_attn_error()));
  }

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

PD_REGISTER_KERNEL(flash_attn_grad,
                   GPU,
                   ALL_LAYOUT,
                   phi::FlashAttnGradKernel,
                   phi::dtype::float16,
                   phi::dtype::bfloat16) {
  kernel->InputAt(5).SetBackend(phi::Backend::ALL_BACKEND);  // seed_offset
}
