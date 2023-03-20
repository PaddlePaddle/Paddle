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
#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/common/bfloat16.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/core/tensor_utils.h"
#include "paddle/phi/kernels/arange_kernel.h"
#include "paddle/phi/kernels/empty_kernel.h"
#include "paddle/phi/kernels/reshape_kernel.h"

#ifdef PADDLE_WITH_FLASHATTN
#include "paddle/phi/backends/dynload/flashattn.h"
#endif

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

  cudaStream_t stream = ctx.stream();
  bool is_bf16 = q.dtype() == DataType::BFLOAT16 ? true : false;

  // q,k,v [total_*, num_heads, head_dim]

  auto dims = q.dims();
  int64_t total_q = dims[0];
  int64_t num_heads = dims[1];
  int64_t head_size = dims[2];

  int64_t total_k = k.dims()[0];
  int64_t batch_size = cu_seqlens_q.numel() - 1;

  int num_splits = 0;  // 0 for an internal heuristic, which is optimal
  bool zero_tensors = false;

  const int64_t* seed_offset_data = seed_offset.data<int64_t>();
  uint64_t seed = static_cast<uint64_t>(seed_offset_data[0]);
  uint64_t offset = static_cast<uint64_t>(seed_offset_data[1]);

  int64_t seq_len_q = ((max_seqlen_q + 16 - 1) / 16) * 16;
  DenseTensor dsoftmax = Empty<float>(ctx, {batch_size, num_heads, seq_len_q});

  uint64_t workspace_size;

  // calculate workspace size before execution
  bool succ = phi::dynload::flash_attn_bwd(
      q.data(),
      k.data(),
      v.data(),
      dq->data(),
      dk->data(),
      dv->data(),
      nullptr,  // for calculation workspace size
      dout.data(),
      cu_seqlens_q.data(),
      cu_seqlens_k.data(),
      total_q,
      total_k,
      batch_size,
      num_heads,
      head_size,
      max_seqlen_q,
      max_seqlen_k,
      dropout,
      scale,
      zero_tensors,
      causal,
      is_bf16,
      num_splits,
      const_cast<float*>(softmax_lse.data<float>()),
      dsoftmax.data(),
      nullptr,
      &workspace_size,
      stream,
      seed,
      offset);

  if (!succ) {
    PADDLE_THROW(phi::errors::External(phi::dynload::flash_attn_error()));
  }

  DenseTensor workspace;
  if (workspace_size > 0) {
    workspace = Empty<float>(ctx, {int64_t(workspace_size / sizeof(float))});
  }

  succ = phi::dynload::flash_attn_bwd(
      q.data(),
      k.data(),
      v.data(),
      dq->data(),
      dk->data(),
      dv->data(),
      out.data(),
      dout.data(),
      cu_seqlens_q.data(),
      cu_seqlens_k.data(),
      total_q,
      total_k,
      batch_size,
      num_heads,
      head_size,
      max_seqlen_q,
      max_seqlen_k,
      dropout,
      scale,
      zero_tensors,
      causal,
      is_bf16,
      num_splits,
      const_cast<float*>(softmax_lse.data<float>()),
      dsoftmax.data(),
      workspace_size > 0 ? workspace.data() : nullptr,
      &workspace_size,
      stream,
      seed,
      offset);

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
  int64_t batch_size = dims[0];
  int64_t seq_len_q = dims[1];
  int64_t num_heads = dims[2];
  int64_t head_size = dims[3];

  int64_t seq_len_k = k.dims()[1];

  int64_t total_q = batch_size * seq_len_q;
  int64_t total_k = batch_size * seq_len_k;

  float scale = 1.0f / std::sqrt(head_size);

  DenseTensor q_t_s, k_t_s, v_t_s;
  q_t_s.ShareDataWith(q).Resize({total_q, num_heads, head_size});
  k_t_s.ShareDataWith(k).Resize({total_k, num_heads, head_size});
  v_t_s.ShareDataWith(v).Resize({total_k, num_heads, head_size});

  DenseTensor cu_seqlens_q;
  DenseTensor cu_seqlens_k;
  ArangeNullaryKernel<int32_t, Context>(
      ctx, 0, (batch_size + 1) * seq_len_q, seq_len_q, &cu_seqlens_q);
  ArangeNullaryKernel<int32_t, Context>(
      ctx, 0, (batch_size + 1) * seq_len_k, seq_len_k, &cu_seqlens_k);

  FlashAttnUnpaddedGradKernel<T, Context>(ctx,
                                          q_t_s,
                                          k_t_s,
                                          v_t_s,
                                          cu_seqlens_q,
                                          cu_seqlens_k,
                                          out,
                                          softmax_lse,
                                          seed_offset,
                                          dout,
                                          seq_len_q,
                                          seq_len_k,
                                          scale,
                                          dropout,
                                          causal,
                                          dq,
                                          dk,
                                          dv);

#endif
}

}  // namespace phi

PD_REGISTER_KERNEL(flash_attn_unpadded_grad,
                   GPU,
                   ALL_LAYOUT,
                   phi::FlashAttnUnpaddedGradKernel,
                   phi::dtype::float16,
                   phi::dtype::bfloat16) {
  kernel->InputAt(7).SetBackend(phi::Backend::CPU);  // seed_offset
}

PD_REGISTER_KERNEL(flash_attn_grad,
                   GPU,
                   ALL_LAYOUT,
                   phi::FlashAttnGradKernel,
                   phi::dtype::float16,
                   phi::dtype::bfloat16) {
  kernel->InputAt(5).SetBackend(phi::Backend::CPU);  // seed_offset
}
