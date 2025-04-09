// Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/phi/kernels/calc_reduced_attn_kernel.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/core/tensor_utils.h"
#include "paddle/phi/kernels/funcs/math_function.h"
#include "paddle/phi/kernels/gpu/flash_attn_utils.h"

namespace phi {

#if defined(PADDLE_WITH_FLASHATTN) && !defined(PADDLE_WITH_HIP)
struct CalcReducedAttnScoresParams : public FlashAttnParamsBase {
  bool return_softmax;
  DenseTensor* softmax;

  CalcReducedAttnScoresParams(const GPUContext& ctx,
                              const int _batch_size,
                              const int64_t _max_seqlen_q,
                              const int64_t _max_seqlen_k,
                              const int _num_heads,
                              const int _num_heads_k,
                              const int _head_size,
                              const float _scale,
                              const DataType q_dtype)
      : FlashAttnParamsBase(/*version=*/2,
                            /*is_fwd=*/true,
                            _batch_size,
                            _max_seqlen_q,
                            _max_seqlen_k,
                            _num_heads,
                            _num_heads_k,
                            _head_size,
                            _scale,
                            /*_causal=*/false,
                            q_dtype,
                            paddle::optional<DenseTensor>{},
                            paddle::optional<DenseTensor>{}) {}
};
#endif

template <typename T, typename Context>
void CalcReducedAttnScoresKernel(const Context& ctx,
                                 const DenseTensor& q,
                                 const DenseTensor& k,
                                 const DenseTensor& softmax_lse,
                                 DenseTensor* reduced_scores) {
#if defined(PADDLE_WITH_FLASHATTN) && !defined(PADDLE_WITH_HIP)
  PADDLE_ENFORCE_EQ(q.dims().size(),
                    4,
                    common::errors::InvalidArgument(
                        "calc_reduced_attention receive input with dim "
                        "[batch_size, seq_len, num_heads, head_dim]"));

  PADDLE_ENFORCE_EQ(k.dims().size(),
                    4,
                    common::errors::InvalidArgument(
                        "calc_reduced_attention receive input with dim "
                        "[batch_size, seq_len, num_heads, head_dim]"));

  if (!reduced_scores->IsInitialized())
    ctx.template Alloc<float>(reduced_scores);
  phi::funcs::SetConstant<Context, float> set_zero;
  set_zero(ctx, reduced_scores, 0.0f);
  // q, k, v [batch_size, seq_len, num_heads, head_dim]
  const int64_t batch_size = q.dims()[0];
  const int64_t seqlen_q = q.dims()[1];
  const int64_t num_heads = q.dims()[2];
  const int64_t head_size = q.dims()[3];
  const int64_t seqlen_k = k.dims()[1];
  const int64_t num_heads_k = k.dims()[2];

  const float softmax_scale = 1.0f / std::sqrt(head_size);
  const float softmax_unscale = std::sqrt(head_size);

  using Params = CalcReducedAttnScoresParams;

  Params params = Params(ctx,
                         batch_size,
                         seqlen_q,
                         seqlen_k,
                         num_heads,
                         num_heads_k,
                         head_size,
                         softmax_scale,
                         q.dtype());

  cudaStream_t stream = ctx.stream();

  bool succ =
      phi::dynload::calc_reduced_attn_scores(q.data(),
                                             k.data(),
                                             softmax_lse.data(),
                                             reduced_scores->data(),
                                             /*softmax_ptr=*/nullptr,
                                             params.batch_size,
                                             params.max_seqlen_q,
                                             params.max_seqlen_k,
                                             params.num_heads,
                                             params.num_heads_k,
                                             params.head_size,
                                             params.softmax_scale,
                                             /*return_softmax=*/false,
                                             params.is_bf16,
                                             /*num_splits=*/0,
                                             stream,
                                             q.strides()[1],
                                             k.strides()[1],
                                             reduced_scores->strides()[1],
                                             q.strides()[2],
                                             k.strides()[2],
                                             reduced_scores->strides()[2],
                                             q.strides()[0],
                                             k.strides()[0],
                                             reduced_scores->strides()[0]);
  CheckFlashAttnStatus(succ);
#else
  RaiseNotSupportedError();
#endif
}
}  // namespace phi

PD_REGISTER_KERNEL(calc_reduced_attn_scores,
                   GPU,
                   ALL_LAYOUT,
                   phi::CalcReducedAttnScoresKernel,
                   phi::dtype::float16,
                   phi::dtype::bfloat16) {}
