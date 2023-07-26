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
#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/common/data_type.h"
#include "paddle/phi/core/enforce.h"
#include "paddle/phi/core/flags.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/core/tensor_utils.h"
#include "paddle/phi/kernels/arange_kernel.h"
#include "paddle/phi/kernels/empty_kernel.h"
#include "paddle/phi/kernels/reshape_kernel.h"

#ifdef PADDLE_WITH_FLASHATTN
#include "paddle/phi/backends/dynload/flashattn.h"
#endif

DECLARE_bool(cudnn_deterministic);

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
    const int max_seqlen_q,
    const int max_seqlen_k,
    const float softmax_scale,
    const float p_dropout,
    const bool is_causal,
    const bool return_softmax,
    const bool is_test,
    const std::string& rng_name,
    DenseTensor* out,
    DenseTensor* softmax,
    DenseTensor* softmax_lse,
    DenseTensor* seed_offset) {
#ifdef PADDLE_WITH_FLASHATTN
#if 0
  if (is_test) p_dropout = 0.0f;
#endif

  ctx.template Alloc<T>(out);

  cudaStream_t stream = ctx.stream();
  bool is_bf16 = q.dtype() == DataType::BFLOAT16 ? true : false;

  // q,k,v [total_*, num_heads, head_dim]

  auto dims = q.dims();
  PADDLE_ENFORCE_EQ(
      dims.size(),
      3,
      phi::errors::InvalidArgument("flash_attn_raw receive input with dim "
                                   "[total_seq_len, num_heads, head_dim]"));

  const int total_q = dims[0];
  const int num_heads = dims[1];
  const int head_size_og = dims[2];

  const int total_k = k.dims()[0];
  const int num_heads_k = k.dims()[1];
  const int batch_size = cu_seqlens_q.numel() - 1;

  const bool zero_tensors = false;

  uint64_t seed;
  uint64_t offset;

  if (fixed_seed_offset.get_ptr()) {
    const int64_t* fixed_seed_offset_data =
        fixed_seed_offset.get_ptr()->data<int64_t>();
    seed = static_cast<uint64_t>(fixed_seed_offset_data[0]);
    offset = static_cast<uint64_t>(fixed_seed_offset_data[1]);
  } else {
    uint64_t inc = batch_size * num_heads * 32;
    std::pair<uint64_t, uint64_t> seed_offset_pair;
    if (rng_name != "") {
      auto gen = phi::GetRandomSeedGenerator(rng_name);
      seed_offset_pair = gen->IncrementOffset(inc);
    } else {
      auto* gen = ctx.GetGenerator();
      seed_offset_pair = gen->IncrementOffset(inc);
    }
    seed = seed_offset_pair.first;
    offset = seed_offset_pair.second;
  }

  VLOG(4) << "FlashAttn fwd seed: " << seed << ", offset: " << offset;

  seed_offset->Resize({2});
  int64_t* seed_offset_data = ctx.template HostAlloc<int64_t>(seed_offset);
  seed_offset_data[0] = static_cast<int64_t>(seed);
  seed_offset_data[1] = static_cast<int64_t>(offset);

  auto round_multiple = [](int x, int m) { return (x + m - 1) / m * m; };
  const int head_size = round_multiple(head_size_og, 8);
  const int head_size_rounded = round_multiple(head_size, 32);
  const int seqlen_q_rounded = round_multiple(max_seqlen_q, 128);
  const int seqlen_k_rounded = round_multiple(max_seqlen_k, 128);

  softmax_lse->Resize({batch_size, num_heads, max_seqlen_q});
  ctx.template Alloc<float>(softmax_lse);

  if (return_softmax) {
    // may allocate more space than *max_seqlen_k*
    // TODO(umiswing): return_softmax is only supported when p_dropout > 0.0
    softmax->Resize(
        {batch_size, num_heads, seqlen_q_rounded, seqlen_k_rounded});
    ctx.template Alloc<T>(softmax);
  }

  const bool succ = phi::dynload::flash_attn_varlen_fwd(
      const_cast<void*>(q.data()),
      const_cast<void*>(k.data()),
      const_cast<void*>(v.data()),
      out->data(),
      const_cast<void*>(cu_seqlens_q.data()),
      const_cast<void*>(cu_seqlens_k.data()),
      batch_size,
      max_seqlen_q,
      max_seqlen_k,
      seqlen_q_rounded,
      seqlen_k_rounded,
      num_heads,
      num_heads_k,
      head_size,
      head_size_rounded,
      p_dropout,
      softmax_scale,
      zero_tensors,
      is_causal,
      return_softmax,
      is_bf16,
      return_softmax ? softmax->data() : nullptr,
      softmax_lse->data(),
      stream,
      seed,
      offset);

  if (!succ) {
    PADDLE_THROW(phi::errors::External(phi::dynload::flash_attn_error()));
  }

#endif
}

template <typename T, typename Context>
void FlashAttnKernel(const Context& ctx,
                     const DenseTensor& q,
                     const DenseTensor& k,
                     const DenseTensor& v,
                     const paddle::optional<DenseTensor>& fixed_seed_offset,
                     const float p_dropout,
                     const bool is_causal,
                     const bool return_softmax,
                     const bool is_test,
                     const std::string& rng_name,
                     DenseTensor* out,
                     DenseTensor* softmax,
                     DenseTensor* softmax_lse,
                     DenseTensor* seed_offset) {
#ifdef PADDLE_WITH_FLASHATTN
  // q,k,v [batch_size, seq_len, num_heads, head_dim]

  auto dims = q.dims();
  PADDLE_ENFORCE_EQ(dims.size(),
                    4,
                    phi::errors::InvalidArgument(
                        "flash_attn receive input with dim "
                        "[batch_size, seq_len, num_heads, head_dim]"));

  const int batch_size = dims[0];
  const int seqlen_q = dims[1];
  const int num_heads = dims[2];
  const int head_size_og = dims[3];
  const int seqlen_k = k.dims()[1];
  const int num_heads_k = k.dims()[2];

  // should we use this scale??
  const float softmax_scale = 1.0f / std::sqrt(head_size_og);

  VLOG(4) << "FlashAttn fwd dims q[" << q.dims() << "], k[" << k.dims()
          << "], v[" << v.dims() << "]";

#if 0
  if (is_test) p_dropout = 0.0f;
#endif

  ctx.template Alloc<T>(out);

  cudaStream_t stream = ctx.stream();
  const bool is_bf16 = q.dtype() == DataType::BFLOAT16 ? true : false;

  uint64_t seed;
  uint64_t offset;

  if (fixed_seed_offset.get_ptr()) {
    const int64_t* fixed_seed_offset_data =
        fixed_seed_offset.get_ptr()->data<int64_t>();
    seed = static_cast<uint64_t>(fixed_seed_offset_data[0]);
    offset = static_cast<uint64_t>(fixed_seed_offset_data[1]);
  } else {
    uint64_t inc = batch_size * num_heads * 32;
    std::pair<uint64_t, uint64_t> seed_offset_pair;
    if (rng_name != "") {
      auto gen = phi::GetRandomSeedGenerator(rng_name);
      seed_offset_pair = gen->IncrementOffset(inc);
    } else {
      auto* gen = ctx.GetGenerator();
      seed_offset_pair = gen->IncrementOffset(inc);
    }
    seed = seed_offset_pair.first;
    offset = seed_offset_pair.second;
  }

  VLOG(4) << "FlashAttn fwd seed: " << seed << ", offset: " << offset;

  seed_offset->Resize({2});
  int64_t* seed_offset_data = ctx.template HostAlloc<int64_t>(seed_offset);
  seed_offset_data[0] = static_cast<int64_t>(seed);
  seed_offset_data[1] = static_cast<int64_t>(offset);

  auto round_multiple = [](int x, int m) { return (x + m - 1) / m * m; };

  // we just round the size but not doing any padding, it's dangerous.
  const int head_size = round_multiple(head_size_og, 8);
  const int head_size_rounded = round_multiple(head_size, 32);
  const int seqlen_q_rounded = round_multiple(seqlen_q, 128);
  const int seqlen_k_rounded = round_multiple(seqlen_k, 128);

  softmax_lse->Resize({batch_size, num_heads, seqlen_q_rounded});
  ctx.template Alloc<float>(softmax_lse);

  if (return_softmax) {
    softmax->Resize(
        {batch_size, num_heads, seqlen_q_rounded, seqlen_k_rounded});
    ctx.template Alloc<T>(softmax);
  }

  // it seems that we don't have to allocate workspace in fa-2

#if 0
  bool succ = phi::dynload::flash_attn_fwd();
#endif
  bool succ =
      phi::dynload::flash_attn_fwd(const_cast<void*>(q.data()),
                                   const_cast<void*>(k.data()),
                                   const_cast<void*>(v.data()),
                                   out->data(),
                                   batch_size,
                                   seqlen_q,
                                   seqlen_k,
                                   seqlen_q_rounded,
                                   seqlen_k_rounded,
                                   num_heads,
                                   num_heads_k,
                                   head_size,
                                   head_size_rounded,
                                   p_dropout,
                                   softmax_scale,
                                   is_causal,
                                   return_softmax,
                                   is_bf16,
                                   return_softmax ? softmax->data() : nullptr,
                                   softmax_lse->data(),
                                   stream,
                                   seed,
                                   offset);

  if (!succ) {
    PADDLE_THROW(phi::errors::External(phi::dynload::flash_attn_error()));
  }
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
