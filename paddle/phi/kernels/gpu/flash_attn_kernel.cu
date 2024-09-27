// 2024 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.   
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
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/core/tensor_utils.h"
#include "paddle/phi/kernels/empty_kernel.h"
#include "paddle/phi/kernels/gpu/flash_attn_utils.h"

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

  // q, k, v [total_q/k/v, num_heads, head_dim]
  auto dims = q.dims();
  PADDLE_ENFORCE_EQ(
      dims.size(),
      3,
      phi::errors::InvalidArgument("flash_attn_raw receive input with dim "
                                   "[total_seq_len, num_heads, head_dim]"));

  const int64_t batch_size = cu_seqlens_q.numel() - 1;
  const int64_t num_heads = dims[1];
  const int64_t head_size = dims[2];
  const int64_t head_size_og = out->dims()[2];
  const int64_t num_heads_k = k.dims()[1];
  const int64_t total_q = dims[0];
  const int64_t total_k = k.dims()[0];
  FlashAttnParamsFwd<T> params = FlashAttnParamsFwd<T>(ctx,
                                                       attn_mask,
                                                       return_softmax,
                                                       *softmax,
                                                       q,
                                                       k,
                                                       v,
                                                       *out,
                                                       *softmax_lse,
                                                       is_test,
                                                       dropout,
                                                       causal,
                                                       fixed_seed_offset,
                                                       *seed_offset,
                                                       rng_name,
                                                       batch_size,
                                                       max_seqlen_q,
                                                       num_heads,
                                                       head_size,
                                                       max_seqlen_k,
                                                       num_heads_k);

  VLOG(10) << "FlashAttn fwd seed: " << params.seed_offset_data[0]
           << ", offset: " << params.seed_offset_data[1];
  auto flash_cu_seqlens_q = DenseTensorToMcFlashAttnTensor(cu_seqlens_q);
  auto flash_cu_seqlens_k = DenseTensorToMcFlashAttnTensor(cu_seqlens_k);
  mcflashattnStatus_t succ =
      phi::dynload::mha_varlen_fwd(params.batch_size,
                                   total_q,
                                   params.num_heads,
                                   total_k,
                                   params.num_heads_k,
                                   head_size_og,
                                   params.q,
                                   params.k,
                                   params.v,
                                   params.out,
                                   flash_cu_seqlens_q,
                                   flash_cu_seqlens_k,
                                   nullptr,
                                   params.alibi_slopes,
                                   params.softmax_lse,
                                   params.p,
                                   params.rng_state,
                                   params.seqlen_q,
                                   params.seqlen_k,
                                   params.p_dropout,
                                   params.softmax_scale,
                                   params.is_causal,
                                   params.window_size_left,
                                   params.window_size_right,
                                   params.stream,
                                   params.extend_parameter);
  phi::dynload::release_tensor(cu_seqlens_q);
  phi::dynload::release_tensor(cu_seqlens_k);
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
  ctx.template Alloc<T>(out);
  // q, k, v [batch_size, seq_len, num_heads, head_dim]
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
  FlashAttnParamsFwd<T> params = FlashAttnParamsFwd<T>(ctx,
                                                       attn_mask,
                                                       return_softmax,
                                                       *softmax,
                                                       q,
                                                       k,
                                                       v,
                                                       *out,
                                                       *softmax_lse,
                                                       is_test,
                                                       dropout,
                                                       causal,
                                                       fixed_seed_offset,
                                                       *seed_offset,
                                                       rng_name,
                                                       batch_size,
                                                       seqlen_q,
                                                       num_heads,
                                                       head_size,
                                                       seqlen_k,
                                                       num_heads_k);

  VLOG(10) << "[FlashAttn Forward] q.shape=[" << q.dims() << "], k.shape=["
           << k.dims() << "], v.shape=[" << v.dims() << "]";
  VLOG(10) << "[FlashAttn Forward] dropout=" << dropout
           << ", seed=" << params.seed_offset_data[0]
           << ", offset=" << params.seed_offset_data[1];
  VLOG(10) << "[FlashAttn Forward] softmax_scale=" << params.softmax_scale;
  if (attn_mask.get_ptr()) {
    VLOG(10) << "[FlashAttn Forward] attn_mask.shape=["
             << (attn_mask.get_ptr())->dims() << "]";
  }

  mcflashattnStatus_t succ = phi::dynload::mha_fwd(params.batch_size,
                                                   params.seqlen_q,
                                                   params.num_heads,
                                                   params.seqlen_k,
                                                   params.num_heads_k,
                                                   params.head_size,
                                                   params.q,
                                                   params.k,
                                                   params.v,
                                                   params.out,
                                                   params.alibi_slopes,
                                                   params.attn_mask,
                                                   params.softmax_lse,
                                                   params.p,  // return softmax
                                                   params.rng_state,
                                                   params.p_dropout,
                                                   params.softmax_scale,
                                                   params.is_causal,
                                                   params.window_size_left,
                                                   params.window_size_right,
                                                   params.stream,
                                                   params.extend_parameter);
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
