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
#include "paddle/phi/backends/gpu/gpu_launch_config.h"
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

template <typename T>
__global__ void SimleScaleWithMaskKernel(int64_t numel,
                                         float scale,
                                         const T* input,
                                         T* ouput) {
  CUDA_KERNEL_LOOP_TYPE(i, numel, int64_t) {
    ouput[i] = static_cast<T>(scale * static_cast<float>(input[i]));
  }
}

template <typename T, typename Context>
void ComputeScaleQ(
    const Context& ctx, int64_t numel, float scale, const T* input, T* output) {
  auto gpu_config = phi::backends::gpu::GetGpuLaunchConfig1D(ctx, numel, 1);
  SimleScaleWithMaskKernel<<<gpu_config.block_per_grid,
                             gpu_config.thread_per_block,
                             0,
                             ctx.stream()>>>(numel, scale, input, output);
}

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
  if (is_test) dropout = 0.0f;

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

  int64_t total_q = dims[0];
  int64_t num_heads = dims[1];
  int64_t head_size = dims[2];

  int64_t total_k = k.dims()[0];
  int64_t batch_size = cu_seqlens_q.numel() - 1;

  int num_splits = 0;  // 0 for an internal heuristic, which is optimal
  if (FLAGS_cudnn_deterministic) {
    num_splits = 1;
  }
  bool zero_tensors = false;

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

  VLOG(4) << "FlashAttn fwd seed: " << seed << ", offset: " << offset
          << ", num_splits:" << num_splits;

  seed_offset->Resize({2});
  int64_t* seed_offset_data = ctx.template HostAlloc<int64_t>(seed_offset);
  seed_offset_data[0] = static_cast<int64_t>(seed);
  seed_offset_data[1] = static_cast<int64_t>(offset);

  int64_t seq_len_q = ((max_seqlen_q + 16 - 1) / 16) * 16;

  softmax_lse->Resize({batch_size, num_heads, seq_len_q});
  ctx.template Alloc<float>(softmax_lse);

  if (return_softmax) {
    // may allocate more space than *max_seqlen_k*
    int64_t blocksize_c = head_size > 64 ? 128 : 256;
    int64_t seq_len_k =
        ((max_seqlen_k + blocksize_c - 1) / blocksize_c) * blocksize_c;
    if (max_seqlen_k <= 128) {
      seq_len_k = 128;
    } else if (max_seqlen_k <= 256) {
      seq_len_k = 256;
    }
    softmax->Resize({batch_size, num_heads, seq_len_q, seq_len_k});
    ctx.template Alloc<T>(softmax);
  }

  uint64_t workspace_size = 0;
  bool succ;
  DenseTensor workspace;

  if (attn_mask.get_ptr()) {
    PADDLE_ENFORCE_NE(causal,
                      true,
                      phi::errors::InvalidArgument(
                          "attn_mask is not nullptr, causal can not be true"));

    int64_t q_size = total_q * num_heads * head_size;
    DenseTensor* scale_q = new DenseTensor;
    scale_q->Resize({total_q, num_heads, head_size});
    ctx.template Alloc<T>(scale_q);
    // compute scale Q
    ComputeScaleQ(ctx, q_size, scale, q.data<T>(), scale_q->data<T>());

    float fa_with_mask_scale = 1.0f;
    std::vector<int64_t> temp_rand_mask_dim;
    const DenseTensor* attn_mask_ptr = attn_mask.get_ptr();
    int64_t first_dim = 1;
    const auto& origin_dims = attn_mask_ptr->dims();
    auto rank = origin_dims.size();
    for (int i = 0; i < rank - 3; i++) {
      first_dim *= origin_dims[i];
    }
    temp_rand_mask_dim = {first_dim,
                          origin_dims[rank - 3],
                          origin_dims[rank - 2],
                          origin_dims[rank - 1]};
    succ = phi::dynload::flash_attn_fwd_with_bias_and_mask(
        static_cast<const void*>(scale_q->data()),
        static_cast<const void*>(k.data()),
        static_cast<const void*>(v.data()),
        nullptr,  // for calculation workspace size
        static_cast<const int32_t*>(cu_seqlens_q.data()),
        static_cast<const int32_t*>(cu_seqlens_k.data()),
        total_q,
        total_k,
        batch_size,
        num_heads,
        head_size,
        max_seqlen_q,
        max_seqlen_k,
        dropout,
        fa_with_mask_scale,
        zero_tensors,
        is_bf16,
        num_splits,
        softmax_lse->data(),
        nullptr,
        &workspace_size,
        stream,
        seed,
        offset,
        attn_mask_ptr ? attn_mask_ptr->data() : nullptr,
        nullptr,
        temp_rand_mask_dim.data() ? temp_rand_mask_dim.data() : nullptr,
        nullptr);
    if (!succ) {
      PADDLE_THROW(phi::errors::External(phi::dynload::flash_attn_error()));
    }

    if (workspace_size > 0) {
      workspace = Empty<float>(
          ctx, {static_cast<int64_t>(workspace_size / sizeof(float))});
    }
    succ = phi::dynload::flash_attn_fwd_with_bias_and_mask(
        static_cast<const void*>(scale_q->data()),
        k.data(),
        v.data(),
        out->data(),  // set out to nullptr to calculate workspace size
        static_cast<const int32_t*>(cu_seqlens_q.data()),
        static_cast<const int32_t*>(cu_seqlens_k.data()),
        total_q,
        total_k,
        batch_size,
        num_heads,
        head_size,
        max_seqlen_q,
        max_seqlen_k,
        dropout,
        fa_with_mask_scale,
        zero_tensors,
        is_bf16,
        num_splits,
        softmax_lse->data(),
        workspace_size > 0 ? workspace.data() : nullptr,
        &workspace_size,
        stream,
        seed,
        offset,
        attn_mask_ptr ? attn_mask_ptr->data() : nullptr,
        nullptr,
        temp_rand_mask_dim.data() ? temp_rand_mask_dim.data() : nullptr,
        nullptr);
    if (!succ) {
      PADDLE_THROW(phi::errors::External(phi::dynload::flash_attn_error()));
    }
    delete scale_q;
  } else {
    succ =
        phi::dynload::flash_attn_fwd(q.data(),
                                     k.data(),
                                     v.data(),
                                     nullptr,  // for calculation workspace size
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
      workspace = Empty<float>(
          ctx, {static_cast<int64_t>(workspace_size / sizeof(float))});
    }

    succ = phi::dynload::flash_attn_fwd(
        q.data(),
        k.data(),
        v.data(),
        out->data(),
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
  }

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
  auto dims = q.dims();
  PADDLE_ENFORCE_EQ(dims.size(),
                    4,
                    phi::errors::InvalidArgument(
                        "flash_attn receive input with dim "
                        "[batch_size, seq_len, num_heads, head_dim]"));

  int64_t batch_size = dims[0];
  int64_t seq_len_q = dims[1];
  int64_t num_heads = dims[2];
  int64_t head_size = dims[3];

  int64_t seq_len_k = k.dims()[1];

  int64_t total_q = batch_size * seq_len_q;
  int64_t total_k = batch_size * seq_len_k;

  float scale = 1.0f / std::sqrt(head_size);

  VLOG(4) << "FlashAttn fwd dims q[" << q.dims() << "], k[" << k.dims()
          << "], v[" << v.dims() << "]";

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

  FlashAttnUnpaddedKernel<T, Context>(ctx,
                                      q_t_s,
                                      k_t_s,
                                      v_t_s,
                                      cu_seqlens_q,
                                      cu_seqlens_k,
                                      fixed_seed_offset,
                                      attn_mask,
                                      seq_len_q,
                                      seq_len_k,
                                      scale,
                                      dropout,
                                      causal,
                                      return_softmax,
                                      is_test,
                                      rng_name,
                                      out,
                                      softmax,
                                      softmax_lse,
                                      seed_offset);

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
