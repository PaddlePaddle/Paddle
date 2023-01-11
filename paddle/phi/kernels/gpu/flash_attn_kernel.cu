// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/infrt/dialect/phi/data_type.h"
#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/common/bfloat16.h"
#include "paddle/phi/core/kernel_registry.h"

#include "paddle/phi/kernels/arange_kernel.h"
#include "paddle/phi/kernels/reshape_kernel.h"
#include "paddle/phi/kernels/transpose_kernel.h"

#include "paddle/phi/backends/dynload/flashattn.h"

namespace phi {

template <typename T, typename Context>
void FlashAttnKernel(const Context& ctx,
                     const DenseTensor& q,
                     const DenseTensor& k,
                     const DenseTensor& v,
                     float dropout,
                     bool causal,
                     DenseTensor* out,
                     DenseTensor* softmax_lse,
                     DenseTensor* softmax) {
  cudaStream_t stream = ctx.stream();
  bool is_bf16 = q.dtype() == DataType::BFLOAT16 ? true : false;

  // q,k,v [batch_size, num_heads, seq_len, head_dim]

  auto dims = q.dims();
  auto batch_size = dims[0];
  auto num_heads = dims[1];
  auto head_size = dims[3];

  int64_t seq_len_q = dims[2];
  int64_t seq_len_k = k.dims()[2];

  auto q_t = Transpose<T, Context>(ctx, q, {0, 2, 1, 3});
  auto k_t = Transpose<T, Context>(ctx, k, {0, 2, 1, 3});
  auto v_t = Transpose<T, Context>(ctx, v, {0, 2, 1, 3});

  // q,k,v [batch_size, seq_len, num_heads, head_dim]

  int64_t total_q = batch_size * seq_len_q;
  int64_t total_k = batch_size * seq_len_k;

  DenseTensor q_t_s =
      Reshape<T, Context>(ctx, q_t, {total_q, num_heads, head_size});
  DenseTensor k_t_s =
      Reshape<T, Context>(ctx, k_t, {total_k, num_heads, head_size});
  DenseTensor v_t_s =
      Reshape<T, Context>(ctx, v_t, {total_k, num_heads, head_size});

  // q,k,v [total_*, num_heads, head_dim]

  DenseTensor cu_seqlens_q;
  DenseTensor cu_seqlens_k;
  ArangeNullaryKernel<int64_t, Context>(
      ctx, 0, (batch_size + 1) * seq_len_q, seq_len_q, &cu_seqlens_q);
  ArangeNullaryKernel<int64_t, Context>(
      ctx, 0, (batch_size + 1) * seq_len_k, seq_len_k, &cu_seqlens_k);

  float scale = 1.0f / std::sqrt(head_size);
  int num_splits = 1;  // always 1 for now

  // auto tp = is_bf16 ? phi::dtype::float16 : phi::dtype::bfloat16;

  if (is_bf16) {
    flash_attn_fwd(
        q_t_s.data<phi::dtype::bfloat16>(),  // const void *q,              //
                                             // total_q x num_heads x head_size,
                                             // total_q := \sum_{i=0}^{b} s_i
        k_t_s.data<phi::dtype::bfloat16>(),  // const void *k,              //
                                             // total_k x num_heads x head_size,
                                             // total_k := \sum_{i=0}^{b} s_i
        v_t_s.data<phi::dtype::bfloat16>(),  // const void *v,              //
                                             // total_k x num_heads x head_size,
                                             // total_k := \sum_{i=0}^{b} s_i
        out->data<phi::dtype::bfloat16>(),   // void *out,                  //
                                             // total_q x num_heads x head_size,
                                             // total_k := \sum_{i=0}^{b} s_i
        cu_seqlens_q
            .data<int64_t>(),  // const void *cu_seqlens_q,   // int32,
                               // batch_size+1, starting offset of each sequence
        cu_seqlens_k
            .data<int64_t>(),  // const void *cu_seqlens_k,   // int32,
                               // batch_size+1, starting offset of each sequence
        total_q,               // const int total_q,
        total_k,               // const int total_k,
        batch_size,            // const int batch_size,
        num_heads,             // const int num_heads,
        head_size,             // const int head_size,
        seq_len_q,             // const int max_seqlen_q_,
        seq_len_k,             // const int max_seqlen_k_,
        dropout,               // const float p_dropout,
        scale,                 // const float softmax_scale,
        true,                  // const bool zero_tensors,
        causal,                // const bool is_causal,
        is_bf16,               // const bool is_bf16,
        num_splits,  // const int num_splits,        // SMs per attention
                     // matrix, can be 1
        softmax_lse->data<float>(),  // void *softmax_lse_ptr,       // softmax
                                     // log_sum_exp
        softmax == nullptr
            ? nullptr
            : softmax->data<phi::dtype::float16>(),  // void *softmax_ptr,
        stream,                                      // cudaStream_t stream,
        0                                            // int seed // TODO
    );
  } else {
  }
}

}  // namespace phi

PD_REGISTER_KERNEL(flash_attn,
                   GPU,
                   ALL_LAYOUT,
                   phi::FlashAttnKernel,
                   phi::dtype::float16,
                   phi::dtype::bfloat16) {}
