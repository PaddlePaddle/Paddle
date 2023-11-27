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

#include "glog/logging.h"  // For VLOG()
#include "paddle/phi/common/data_type.h"
#include "paddle/phi/core/flags.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/core/tensor_utils.h"
#include "paddle/phi/kernels/arange_kernel.h"
#include "paddle/phi/kernels/empty_kernel.h"
#include "paddle/phi/kernels/gpu/flash_attn_utils.h"
#include "paddle/phi/kernels/reshape_kernel.h"

PD_DECLARE_bool(cudnn_deterministic);

namespace phi {

template <typename T, typename Context>
void FlashAttnUnpaddedKernel(
    const Context& ctx,
    const DenseTensor& q,
    const DenseTensor& k,
    const DenseTensor& v,
    const DenseTensor& cu_seqlens_q,
    const DenseTensor& cu_seqlens_k,
    const paddle::optional<DenseTensor>& fixed_seed_offset,
    const paddle::optional<DenseTensor>& attn_mask,
    int64_t max_seqlen_q,
    int64_t max_seqlen_k,
    float scale,
    float dropout,
    bool causal,
    bool return_softmax,
    bool is_test,
    const std::string& rng_name,
    DenseTensor* out,
    DenseTensor* softmax,
    DenseTensor* softmax_lse,
    DenseTensor* seed_offset) {
#ifdef PADDLE_WITH_FLASHATTN
  ctx.template Alloc<T>(out);

  cudaStream_t stream = ctx.stream();

  // q,k,v [total_*, num_heads, head_dim]
  auto dims = q.dims();
  PADDLE_ENFORCE_EQ(
      dims.size(),
      3,
      phi::errors::InvalidArgument("flash_attn_raw receive input with dim "
                                   "[total_seq_len, num_heads, head_dim]"));

  const int64_t total_q = dims[0];
  const int64_t num_heads = dims[1];
  const int64_t head_size = dims[2];

  const int64_t total_k = k.dims()[0];
  const int64_t num_heads_k = k.dims()[1];
  const int64_t batch_size = cu_seqlens_q.numel() - 1;

  // TODO(umiswing): add shape check

  FlashAttnFwdParamsV2<T> params = FlashAttnFwdParamsV2<T>(ctx,
                                                           batch_size,
                                                           max_seqlen_q,
                                                           max_seqlen_k,
                                                           num_heads,
                                                           num_heads_k,
                                                           head_size,
                                                           dropout,
                                                           scale,
                                                           causal,
                                                           return_softmax,
                                                           q.dtype(),
                                                           is_test,
                                                           rng_name,
                                                           fixed_seed_offset,
                                                           attn_mask,
                                                           softmax,
                                                           softmax_lse,
                                                           seed_offset);

  VLOG(10) << "FlashAttn fwd seed: " << params.seed
           << ", offset: " << params.offset;

  bool succ = phi::dynload::flash_attn_varlen_fwd(
      q.data(),
      k.data(),
      v.data(),
      cu_seqlens_q.data<int32_t>(),
      cu_seqlens_k.data<int32_t>(),
      params.rng_state.data(),
      out->data(),
      params.return_softmax ? softmax->data() : nullptr,
      softmax_lse->data(),
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
      1.0f / params.scale,
      params.causal,
      params.return_softmax,
      params.is_bf16,
      stream,
      params.seed,
      params.offset,
      params.attn_mask_tensor ? params.attn_mask_tensor->data() : nullptr,
      params.mask_dims.data());
  CheckFlashAttnStatus(succ);
#else
  RaiseNotSupportedError();
#endif
}

template <typename T, typename Context>
void FlashAttnKernel(const Context& ctx,
                     const DenseTensor& q,
                     const DenseTensor& k,
                     const DenseTensor& v,
                     const paddle::optional<DenseTensor>& fixed_seed_offset,
                     const paddle::optional<DenseTensor>& attn_mask,
                     float dropout,
                     bool causal,
                     bool return_softmax,
                     bool is_test,
                     const std::string& rng_name,
                     DenseTensor* out,
                     DenseTensor* softmax,
                     DenseTensor* softmax_lse,
                     DenseTensor* seed_offset) {
#ifdef PADDLE_WITH_FLASHATTN
  // q,k,v [batch_size, seq_len, num_heads, head_dim]
  const auto& dims = q.dims();
  PADDLE_ENFORCE_EQ(dims.size(),
                    4,
                    phi::errors::InvalidArgument(
                        "flash_attn receive input with dim "
                        "[batch_size, seq_len, num_heads, head_dim]"));

  const int64_t batch_size = dims[0];
  const int64_t seqlen_q = dims[1];
  const int64_t num_heads = dims[2];
  const int64_t head_size = dims[3];
  const int64_t seqlen_k = k.dims()[1];
  const int64_t num_heads_k = k.dims()[2];

  const int64_t total_q = batch_size * seqlen_q;
  const int64_t total_k = batch_size * seqlen_k;

  // TODO(umiswing): Add check shape

  const float scale = 1.0f / std::sqrt(head_size);

  FlashAttnFwdParamsV2<T> params = FlashAttnFwdParamsV2<T>(ctx,
                                                           batch_size,
                                                           seqlen_q,
                                                           seqlen_k,
                                                           num_heads,
                                                           num_heads_k,
                                                           head_size,
                                                           dropout,
                                                           scale,
                                                           causal,
                                                           return_softmax,
                                                           q.dtype(),
                                                           is_test,
                                                           rng_name,
                                                           fixed_seed_offset,
                                                           attn_mask,
                                                           softmax,
                                                           softmax_lse,
                                                           seed_offset);

  VLOG(10) << "FlashAttn fwd dims: q[" << q.dims() << "], k[" << k.dims()
           << "], v[" << v.dims() << "]";
  VLOG(10) << "FlashAttn fwd seed: " << params.seed
           << ", offset: " << params.offset;

  ctx.template Alloc<T>(out);

  cudaStream_t stream = ctx.stream();

  bool succ = phi::dynload::flash_attn_fwd(
      q.data(),
      k.data(),
      v.data(),
      params.rng_state.data(),
      out->data(),
      params.return_softmax ? params.softmax->data() : nullptr,
      params.softmax_lse->data(),
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
      std::sqrt(head_size),  // for unscale
      params.causal,
      params.return_softmax,
      params.is_bf16,
      stream,
      params.seed,
      params.offset,
      params.attn_mask_tensor ? params.attn_mask_tensor->data() : nullptr,
      params.mask_dims.data());
  CheckFlashAttnStatus(succ);
#else
  RaiseNotSupportedError();
#endif
}

}  // namespace phi

PD_REGISTER_KERNEL(flash_attn_unpadded,
                   GPU,
                   ALL_LAYOUT,
                   phi::FlashAttnUnpaddedKernel,
                   phi::dtype::float16,
                   phi::dtype::bfloat16) {
  kernel->InputAt(5).SetBackend(
      phi::Backend::ALL_BACKEND);  // fixed_seed_offset
}

PD_REGISTER_KERNEL(flash_attn,
                   GPU,
                   ALL_LAYOUT,
                   phi::FlashAttnKernel,
                   phi::dtype::float16,
                   phi::dtype::bfloat16) {
  kernel->InputAt(3).SetBackend(
      phi::Backend::ALL_BACKEND);  // fixed_seed_offset
}
