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

#include "paddle/phi/kernels/flash_attn_kernel.h"

#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/common/data_type.h"
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
void FlashAttnKernel(const Context& ctx,
                     const DenseTensor& q,
                     const DenseTensor& k,
                     const DenseTensor& v,
                     float dropout,
                     bool causal,
                     bool return_softmax,
                     DenseTensor* out,
                     DenseTensor* softmax_lse,
                     DenseTensor* softmax,
                     DenseTensor* seed_offset) {
#ifdef PADDLE_WITH_FLASHATTN
  ctx.template Alloc<T>(out);

  cudaStream_t stream = ctx.stream();
  bool is_bf16 = q.dtype() == DataType::BFLOAT16 ? true : false;

  // q,k,v [batch_size, seq_len, num_heads, head_dim]

  auto dims = q.dims();
  int64_t batch_size = dims[0];
  int64_t seq_len_q = dims[1];
  int64_t num_heads = dims[2];
  int64_t head_size = dims[3];

  int64_t seq_len_k = k.dims()[1];

  int64_t total_q = batch_size * seq_len_q;
  int64_t total_k = batch_size * seq_len_k;

  DenseTensor q_t_s =
      Reshape<T, Context>(ctx, q, {total_q, num_heads, head_size});
  DenseTensor k_t_s =
      Reshape<T, Context>(ctx, k, {total_k, num_heads, head_size});
  DenseTensor v_t_s =
      Reshape<T, Context>(ctx, v, {total_k, num_heads, head_size});

  // q,k,v [total_*, num_heads, head_dim]

  DenseTensor cu_seqlens_q;
  DenseTensor cu_seqlens_k;
  ArangeNullaryKernel<int32_t, Context>(
      ctx, 0, (batch_size + 1) * seq_len_q, seq_len_q, &cu_seqlens_q);
  ArangeNullaryKernel<int32_t, Context>(
      ctx, 0, (batch_size + 1) * seq_len_k, seq_len_k, &cu_seqlens_k);

  float scale = 1.0f / std::sqrt(head_size);
  int num_splits = 0;  // 0 for an internal heuristic, which is optimal
  bool zero_tensors = false;

  auto gen = ctx.GetGenerator();
  uint64_t inc = batch_size * num_heads * 32;
  auto seed_offset_pair = gen->IncrementOffset(inc);
  uint64_t seed = seed_offset_pair.first;
  uint64_t offset = seed_offset_pair.second;

  std::vector<int64_t> seed_offset_vec{int64_t(seed), int64_t(offset)};
  phi::TensorFromVector<int64_t>(seed_offset_vec, ctx, seed_offset);

  softmax_lse->Resize({batch_size, num_heads, seq_len_q});
  ctx.template Alloc<float>(softmax_lse);

  if (return_softmax) {
    // may allocate more space than *seq_len_k*
    int64_t blocksize_c = head_size > 64 ? 128 : 256;
    int64_t max_len_k_ =
        ((seq_len_k + blocksize_c - 1) / blocksize_c) * blocksize_c;
    int64_t max_len_k =
        seq_len_k <= 128 ? 128 : (seq_len_k <= 256 ? 256 : max_len_k_);
    softmax->Resize({batch_size, num_heads, seq_len_q, max_len_k});
    ctx.template Alloc<T>(softmax);
  }

  uint64_t workspace_size;

  // calculate workspace size before execution
  bool succ =
      phi::dynload::flash_attn_fwd(q_t_s.data(),
                                   k_t_s.data(),
                                   v_t_s.data(),
                                   nullptr,  // for calculation workspace size
                                   cu_seqlens_q.data(),
                                   cu_seqlens_k.data(),
                                   total_q,
                                   total_k,
                                   batch_size,
                                   num_heads,
                                   head_size,
                                   seq_len_q,
                                   seq_len_k,
                                   dropout,
                                   scale,
                                   zero_tensors,
                                   causal,
                                   is_bf16,
                                   num_splits,
                                   softmax_lse->data(),
                                   return_softmax ? softmax->data() : nullptr,
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

  succ = phi::dynload::flash_attn_fwd(
      q_t_s.data(),
      k_t_s.data(),
      v_t_s.data(),
      out->data(),
      cu_seqlens_q.data(),
      cu_seqlens_k.data(),
      total_q,
      total_k,
      batch_size,
      num_heads,
      head_size,
      seq_len_q,
      seq_len_k,
      dropout,
      scale,
      zero_tensors,
      causal,
      is_bf16,
      num_splits,
      softmax_lse->data(),
      return_softmax ? softmax->data() : nullptr,
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

}  // namespace phi

PD_REGISTER_KERNEL(flash_attn,
                   GPU,
                   ALL_LAYOUT,
                   phi::FlashAttnKernel,
                   phi::dtype::float16,
                   phi::dtype::bfloat16) {}
