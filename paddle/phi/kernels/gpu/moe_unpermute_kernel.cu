// Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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
#include "paddle/phi/kernels/gpu/moe_unpermute_kernel.h"
#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/full_kernel.h"
#include "paddle/phi/kernels/gpu/moe_permute_utils.h"

namespace phi {

// Import MoE constants from shared header
using moe::kMaxNumExperts;

template <bool MP, bool WEIGHTED_TOKEN, int NUM_EXPERTS>
__global__ __launch_bounds__(256) void tokens_zip_kernel(
    const bfloat16 *__restrict__ unzipped_tokens_in,
    const int *__restrict__ zipped_expertwise_rowmap,
    const int *__restrict__ expert_routemap_topk,
    const float *__restrict__ unzipped_token_probs,
    bfloat16 *__restrict__ zipped_tokens_out,
    float *__restrict__ zipped_probs_topk,
    const int total_zipped_tokens_num,
    const int token_length,
    const int num_experts,
    const int topk) {
  const int this_row = blockIdx.x;

  if (this_row >= total_zipped_tokens_num) return;

  const __nv_bfloat16 *unzipped_tokens =
      reinterpret_cast<const __nv_bfloat16 *>(unzipped_tokens_in);
  __nv_bfloat16 *zipped_tokens =
      reinterpret_cast<__nv_bfloat16 *>(zipped_tokens_out);

  __shared__ int local_row_fetchlist[NUM_EXPERTS];
  __shared__ float local_row_weight[NUM_EXPERTS];

  // Strided load: blockDim.x may be < num_experts, so each thread
  // handles multiple slots to cover the full [0, num_experts) range.
  for (int i = threadIdx.x; i < num_experts; i += blockDim.x) {
    const int fetch_row = zipped_expertwise_rowmap[this_row * num_experts + i];
    local_row_fetchlist[i] = fetch_row;
    if constexpr (WEIGHTED_TOKEN) {
      local_row_weight[i] =
          ((fetch_row == -1) ? 0.0f : unzipped_token_probs[fetch_row]);
    }
  }

  __syncthreads();

#pragma unroll
  for (int k = 0; k < topk; ++k) {
    const int expert_idx = expert_routemap_topk[this_row * topk + k];
    if (expert_idx < 0) [[likely]]
      continue;
    const int expert_fetch_row = local_row_fetchlist[expert_idx];
    zipped_probs_topk[this_row * topk + k] =
        unzipped_token_probs[expert_fetch_row];
  }

  // only support VecSize = 8
  constexpr int VecSize = 8;
  // use bfloat162 to pack 2 bfloat16s
  constexpr int PACKED_VEC_SIZE = VecSize / 2;

  const int num_full_vec = token_length / VecSize;
  const int64_t thread_stride = static_cast<int64_t>(blockDim.x) * VecSize;

#pragma unroll 1
  for (int64_t x_offset = static_cast<int64_t>(threadIdx.x) * VecSize;
       x_offset < num_full_vec * VecSize;
       x_offset += thread_stride) {
    __nv_bfloat162 raw[PACKED_VEC_SIZE] = {{0.0f, 0.0f}};
    float2 sum[PACKED_VEC_SIZE] = {{0.0f, 0.0f}};

    int aggreg_cnt = 0;

#pragma unroll
    for (int expert = 0; expert < num_experts; ++expert) {
      float weight;
      const int fetch_row = local_row_fetchlist[expert];
      if (fetch_row < 0) continue;
      // Get weight of current copy of token.
      if constexpr (WEIGHTED_TOKEN) {
        weight = local_row_weight[expert];
      }
      aggreg_cnt++;

      const __nv_bfloat162 *base_ptr = reinterpret_cast<const __nv_bfloat162 *>(
          &unzipped_tokens[(int64_t)fetch_row * (int64_t)token_length +
                           x_offset]);

      // Cast the input pointer to uint4* to enforce a single 128-bit
      // vectorized load (LDG.E.128) for optimal memory bandwidth.
      uint4 packed_raw = *reinterpret_cast<const uint4 *>(base_ptr);

      const __nv_bfloat162 *raw_ptr =
          reinterpret_cast<const __nv_bfloat162 *>(&packed_raw);

#pragma unroll
      for (int i = 0; i < PACKED_VEC_SIZE; ++i) {
        raw[i] = raw_ptr[i];
        float2 token_vec = __bfloat1622float2(raw[i]);
        if constexpr (WEIGHTED_TOKEN) {
          sum[i].x = __fmaf_rn(token_vec.x, weight, sum[i].x);
          sum[i].y = __fmaf_rn(token_vec.y, weight, sum[i].y);
        } else {
          sum[i].x = __fadd_rn(token_vec.x, sum[i].x);
          sum[i].y = __fadd_rn(token_vec.y, sum[i].y);
        }
      }  // Pack loop
    }    // Expert loop

    __nv_bfloat162 results[PACKED_VEC_SIZE];
#pragma unroll
    for (int i = 0; i < PACKED_VEC_SIZE; ++i) {
      // Using raw if not aggregated, prevent submornal downcast.
      results[i] = (aggreg_cnt > 1) ? __float22bfloat162_rn(sum[i]) : raw[i];
    }

    __nv_bfloat162 *out_ptr = reinterpret_cast<__nv_bfloat162 *>(
        &zipped_tokens[(int64_t)this_row * (int64_t)token_length + x_offset]);

    // Cast the output pointer to uint4* to enforce a single 128-bit
    // vectorized store (STG.E.128) for optimal memory bandwidth.
    *reinterpret_cast<uint4 *>(out_ptr) = *reinterpret_cast<uint4 *>(results);
  }  // Vectorized token length loop

#pragma unroll 1
  for (int i = num_full_vec * VecSize + threadIdx.x; i < token_length;
       i += blockDim.x) {
    float sum = 0.0f;
    __nv_bfloat16 raw = 0.0f;
    int aggreg_cnt = 0;

#pragma unroll
    for (int expert = 0; expert < num_experts; ++expert) {
      int fetch_row = local_row_fetchlist[expert];
      float weight;
      if constexpr (WEIGHTED_TOKEN) {
        weight = local_row_weight[expert];
      }
      if (fetch_row < 0) continue;
      aggreg_cnt++;
      raw = unzipped_tokens[(int64_t)fetch_row * (int64_t)token_length + i];
      float token_val = static_cast<float>(raw);

      if constexpr (WEIGHTED_TOKEN) {
        sum = __fmaf_rn(token_val, weight, sum);
      } else {
        sum = __fadd_rn(token_val, sum);
      }
    }
    zipped_tokens[(int64_t)this_row * (int64_t)token_length + i] =
        (aggreg_cnt > 1) ? static_cast<__nv_bfloat16>(sum) : raw;
  }  // Trailing token length loop

  // Optimization: A dummy synchronization primitive is placed here to act as a
  // compiler barrier. This forces the compiler to shrink the live ranges of
  // variables and release registers earlier. This reduces peak register usage,
  // improving occupancy from 75% to 100% and yielding a significant performance
  // boost.
  __syncwarp();
}

template <typename T, typename Context>
void dispatch_tokens_zip(const Context &dev_ctx,
                         const DenseTensor &unzipped_tokens,
                         const DenseTensor &zipped_expertwise_rowmap,
                         const DenseTensor &expert_routemap_topk,
                         const DenseTensor &unzipped_token_probs,
                         DenseTensor *zipped_tokens,
                         DenseTensor *zipped_probs_topk,
                         const int total_zipped_tokens_num,
                         const int num_experts,
                         const int token_length,
                         const int topk,
                         const bool MP,
                         const bool using_weighted_combine) {
  dim3 grid, block;
  grid.x = total_zipped_tokens_num;
  block.x = 256;

  if (unzipped_token_probs.dtype() != paddle::DataType::FLOAT32) return;

  // Unified dispatch: MP x WEIGHTED x NUM_EXPERTS
  dispatch::Bools(
      [&](auto mp_tag, auto weighted_tag) {
        constexpr bool MP_CONST = decltype(mp_tag)::value;
        constexpr bool WEIGHTED_CONST = decltype(weighted_tag)::value;

        dispatch::NumExperts(num_experts, [&](auto ne_tag) {
          constexpr int NE = decltype(ne_tag)::value;

          tokens_zip_kernel<MP_CONST, WEIGHTED_CONST, NE>
              <<<grid, block, 0, dev_ctx.stream()>>>(
                  unzipped_tokens.data<bfloat16>(),
                  zipped_expertwise_rowmap.data<int>(),
                  expert_routemap_topk.data<int>(),
                  unzipped_token_probs.data<float>(),
                  zipped_tokens->data<bfloat16>(),
                  zipped_probs_topk->data<float>(),
                  total_zipped_tokens_num,
                  token_length,
                  num_experts,
                  topk);
        });
      },
      MP,
      using_weighted_combine);
}

template <typename T, typename Context>
void MoeUnpermuteKernel(const Context &dev_ctx,
                        const DenseTensor &unzipped_tokens,
                        const DenseTensor &zipped_expertwise_rowmap,
                        const DenseTensor &expert_routemap_topk,
                        const DenseTensor &unzipped_token_probs,
                        const int total_zipped_tokens_num,
                        const int num_experts,
                        const bool MP,
                        const bool using_weighted_combine,
                        DenseTensor *zipped_tokens,
                        DenseTensor *zipped_probs_topk) {
  const int64_t cols = unzipped_tokens.dims()[1];
  PADDLE_ENFORCE_LE(cols,
                    std::numeric_limits<int32_t>::max(),
                    common::errors::InvalidArgument(
                        "unzipped_tokens.dims()[1] should be less than "
                        "INT_MAX, received unzipped_tokens.dims()[1]: (%ld)",
                        cols));
  PADDLE_ENFORCE_LE(
      num_experts,
      kMaxNumExperts,
      common::errors::InvalidArgument(
          "Currently we support no more than (%ld), received num_expert: "
          "(%ld). Please check input "
          "value.",
          kMaxNumExperts,
          num_experts));
  const int64_t topk = expert_routemap_topk.dims()[1];
  PADDLE_ENFORCE_LE(
      topk,
      std::numeric_limits<int32_t>::max(),
      common::errors::InvalidArgument(
          "topk should be less than INT_MAX, received topk: (%ld)", topk));
  dev_ctx.template Alloc<T>(zipped_tokens);
  dev_ctx.template Alloc<float>(zipped_probs_topk);
  if (unzipped_tokens.numel() == 0) return;  // 0-size tensor
  void *zipped_probs_topk_ptr =
      reinterpret_cast<void *>(zipped_probs_topk->data<float>());
  PADDLE_ENFORCE_GPU_SUCCESS(
      cudaMemsetAsync(zipped_probs_topk_ptr,
                      0,
                      sizeof(float) * int64_t(total_zipped_tokens_num) * topk,
                      dev_ctx.stream()));

  dispatch_tokens_zip<T, Context>(dev_ctx,
                                  unzipped_tokens,
                                  zipped_expertwise_rowmap,
                                  expert_routemap_topk,
                                  unzipped_token_probs,
                                  zipped_tokens,
                                  zipped_probs_topk,
                                  total_zipped_tokens_num,
                                  num_experts,
                                  static_cast<int>(cols),
                                  static_cast<int>(topk),
                                  MP,
                                  using_weighted_combine);
}
}  // namespace phi

PD_REGISTER_KERNEL(
    moe_unpermute, GPU, ALL_LAYOUT, phi::MoeUnpermuteKernel, phi::bfloat16) {}
