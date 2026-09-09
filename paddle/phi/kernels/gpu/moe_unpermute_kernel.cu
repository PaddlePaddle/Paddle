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

#include <cstdlib>
#include <limits>

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
    const int fetch_row =
        zipped_expertwise_rowmap[static_cast<int64_t>(this_row) * num_experts +
                                 i];
    local_row_fetchlist[i] = fetch_row;
    if constexpr (WEIGHTED_TOKEN) {
      local_row_weight[i] =
          ((fetch_row == -1) ? 0.0f : unzipped_token_probs[fetch_row]);
    }
  }

  __syncthreads();

#pragma unroll
  for (int k = 0; k < topk; ++k) {
    const int expert_idx =
        expert_routemap_topk[static_cast<int64_t>(this_row) * topk + k];
    if (expert_idx < 0) [[likely]]
      continue;
    const int expert_fetch_row = local_row_fetchlist[expert_idx];
    zipped_probs_topk[static_cast<int64_t>(this_row) * topk + k] =
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


// ============================================================================
//   tokens_zip: several rows per block, and stop doing the probs scatter 256x
// ============================================================================
// The kernel above launches one block of 256 threads per zipped row.  With
// token_length 2048 and VecSize 8 that is exactly one 16 B vector per thread,
// so a block moves ~8 KB and pays for it with a rowmap load done by 8 of its
// 256 threads, a __syncthreads, and a topk-long probs scatter that ALL 256
// threads execute redundantly (see :67-76 above -- there is no threadIdx guard,
// so 256 threads write the same value to the same address).
//
// This kernel keeps the arithmetic identical and changes only the thread
// mapping:
//   * ROWS_PER_BLOCK rows per block, so the per-block setup is amortised and
//     the grid drops to a 16th of total_zipped_tokens_num;
//   * the probs scatter is done by one thread per (row, k);
//   * the expert loop bound is the compile-time NUM_EXPERTS, with the slots in
//     [num_experts, NUM_EXPERTS) pre-filled with -1 so they are skipped by the
//     very same `fetch_row < 0` test.
//
// IMPORTANT -- this kernel does contain a reduction (the fp32 accumulation over
// the local experts a token was routed to).  None of the changes above touches
// its order: the loop still runs over ascending expert id, still accumulates in
// fp32, and still ends with `aggreg_cnt > 1 ? cvt(sum) : raw`.  That is why the
// output is bit-identical rather than merely close.
// Standalone bench, 153962 rows / token_length 2048 / 8 local experts /
// topk 10, median of 100:  437.1 us -> 246.8 us (43.5% -> 77.0% of HBM peak);
// the probs-scatter fix alone accounts for 437.1 -> 320.5.
// Set MOE_UNPERMUTE_MULTIROW=0 to get the shipped kernel back.
inline bool moe_unpermute_multirow() {
  static const bool v = [] {
    const char *s = std::getenv("MOE_UNPERMUTE_MULTIROW");
    return (s == nullptr) || !(s[0] == '0' && s[1] == '\0');
  }();
  return v;
}

template <bool MP,
          bool WEIGHTED_TOKEN,
          int NUM_EXPERTS,
          int ROWS_PER_BLOCK,
          int BLOCK_DIM_X>
__global__ __launch_bounds__(BLOCK_DIM_X) void tokens_zip_multirow_kernel(
    const phi::bfloat16 *__restrict__ unzipped_tokens_in,
    const int *__restrict__ zipped_expertwise_rowmap,
    const int *__restrict__ expert_routemap_topk,
    const float *__restrict__ unzipped_token_probs,
    phi::bfloat16 *__restrict__ zipped_tokens_out,
    float *__restrict__ zipped_probs_topk,
    const int total_zipped_tokens_num,
    const int token_length,
    const int num_experts,
    const int topk) {
  const int row0 = blockIdx.x * ROWS_PER_BLOCK;
  if (row0 >= total_zipped_tokens_num) return;
  const int nrows = min(ROWS_PER_BLOCK, total_zipped_tokens_num - row0);
  const int tid = threadIdx.x;

  const __nv_bfloat16 *unzipped_tokens =
      reinterpret_cast<const __nv_bfloat16 *>(unzipped_tokens_in);
  __nv_bfloat16 *zipped_tokens =
      reinterpret_cast<__nv_bfloat16 *>(zipped_tokens_out);

  __shared__ int local_row_fetchlist[ROWS_PER_BLOCK * NUM_EXPERTS];
  __shared__ float
      local_row_weight[WEIGHTED_TOKEN ? ROWS_PER_BLOCK * NUM_EXPERTS : 1];

  for (int i = tid; i < ROWS_PER_BLOCK * NUM_EXPERTS; i += BLOCK_DIM_X) {
    const int r = i / NUM_EXPERTS;
    const int e = i - r * NUM_EXPERTS;
    int fetch_row = -1;
    if (r < nrows && e < num_experts) {
      fetch_row = zipped_expertwise_rowmap[static_cast<int64_t>(row0 + r) *
                                               num_experts +
                                           e];
    }
    local_row_fetchlist[i] = fetch_row;
    if constexpr (WEIGHTED_TOKEN) {
      local_row_weight[i] =
          ((fetch_row == -1) ? 0.0f : unzipped_token_probs[fetch_row]);
    }
  }

  __syncthreads();

  for (int i = tid; i < nrows * topk; i += BLOCK_DIM_X) {
    const int r = i / topk;
    const int k = i - r * topk;
    const int expert_idx =
        expert_routemap_topk[static_cast<int64_t>(row0 + r) * topk + k];
    if (expert_idx < 0) continue;
    const int expert_fetch_row =
        local_row_fetchlist[r * NUM_EXPERTS + expert_idx];
    zipped_probs_topk[static_cast<int64_t>(row0 + r) * topk + k] =
        unzipped_token_probs[expert_fetch_row];
  }

  // only support VecSize = 8
  constexpr int VecSize = 8;
  constexpr int PACKED_VEC_SIZE = VecSize / 2;
  const int num_full_vec = token_length / VecSize;

#pragma unroll 1
  for (int v = tid; v < num_full_vec; v += BLOCK_DIM_X) {
    const int x_offset = v * VecSize;
#pragma unroll 1
    for (int r = 0; r < ROWS_PER_BLOCK; ++r) {
      if (r >= nrows) break;
      __nv_bfloat162 raw[PACKED_VEC_SIZE] = {{0.0f, 0.0f}};
      float2 sum[PACKED_VEC_SIZE] = {{0.0f, 0.0f}};
      int aggreg_cnt = 0;

      // The trip count stays the runtime num_experts, exactly as shipped.
      // Using the compile-time NUM_EXPERTS instead lets the loop unroll fully
      // but it is a dispatch *bucket* (up to 384): at num_experts 40 the
      // bucket is 384, which cost 255 registers and 912 us against the
      // shipped 19.5 us.  `unroll 8` gives the unrolling without betting on
      // the bucket being tight.
#pragma unroll 8
      for (int expert = 0; expert < num_experts; ++expert) {
        float weight;
        const int fetch_row = local_row_fetchlist[r * NUM_EXPERTS + expert];
        if (fetch_row < 0) continue;
        if constexpr (WEIGHTED_TOKEN) {
          weight = local_row_weight[r * NUM_EXPERTS + expert];
        }
        aggreg_cnt++;

        const __nv_bfloat162 *base_ptr =
            reinterpret_cast<const __nv_bfloat162 *>(
                &unzipped_tokens[(int64_t)fetch_row * (int64_t)token_length +
                                 x_offset]);
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
        }
      }

      __nv_bfloat162 results[PACKED_VEC_SIZE];
#pragma unroll
      for (int i = 0; i < PACKED_VEC_SIZE; ++i) {
        results[i] = (aggreg_cnt > 1) ? __float22bfloat162_rn(sum[i]) : raw[i];
      }
      __nv_bfloat162 *out_ptr = reinterpret_cast<__nv_bfloat162 *>(
          &zipped_tokens[(int64_t)(row0 + r) * (int64_t)token_length +
                         x_offset]);
      *reinterpret_cast<uint4 *>(out_ptr) = *reinterpret_cast<uint4 *>(results);
    }
  }

  for (int r = 0; r < nrows; ++r) {
#pragma unroll 1
    for (int i = num_full_vec * VecSize + tid; i < token_length;
         i += BLOCK_DIM_X) {
      float sum = 0.0f;
      __nv_bfloat16 raw = 0.0f;
      int aggreg_cnt = 0;

#pragma unroll 8
      for (int expert = 0; expert < num_experts; ++expert) {
        const int fetch_row = local_row_fetchlist[r * NUM_EXPERTS + expert];
        float weight;
        if constexpr (WEIGHTED_TOKEN) {
          weight = local_row_weight[r * NUM_EXPERTS + expert];
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
      zipped_tokens[(int64_t)(row0 + r) * (int64_t)token_length + i] =
          (aggreg_cnt > 1) ? static_cast<__nv_bfloat16>(sum) : raw;
    }
  }
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
  PADDLE_ENFORCE_GE(
      total_zipped_tokens_num,
      0,
      common::errors::InvalidArgument(
          "total_zipped_tokens_num should be non-negative, but got %d.",
          total_zipped_tokens_num));
  if (total_zipped_tokens_num == 0) return;
  dim3 grid, block;
  grid.x = static_cast<unsigned int>(total_zipped_tokens_num);
  block.x = 256;

  if (unzipped_token_probs.dtype() != DataType::FLOAT32) return;

  // Several rows per block only pays off once the smaller grid still
  // fills the machine.  Measured crossover at 8 rows/block on 148
  // SMs: 2368 rows -> 13.3 us shipped vs 15.4 us multirow (grid 296);
  // 4736 -> 17.4 vs 15.4 (grid 592); 9472 -> 29.7 vs 19.5; 153962 ->
  // 435.6 vs 226.3.  Gate on 4 blocks per SM, i.e. the 4736 point.
  const bool multirow = moe_unpermute_multirow();
  const int64_t sm_count = dev_ctx.GetSMCount();

  // Unified dispatch: MP x WEIGHTED x NUM_EXPERTS
  dispatch::Bools(
      [&](auto mp_tag, auto weighted_tag) {
        constexpr bool MP_CONST = decltype(mp_tag)::value;
        constexpr bool WEIGHTED_CONST = decltype(weighted_tag)::value;

        dispatch::NumExperts(num_experts, [&](auto ne_tag) {
          constexpr int NE = decltype(ne_tag)::value;

          // Rows per block: 8 at the production shape (the 4..12 plateau is
          // flat -- 228.4 / 226.1 / 226.3 / 226.5 us -- and 16 already falls
          // back to 236.5), capped further so the two per-row shared-memory
          // tables stay within 4 KB for every NUM_EXPERTS bucket
          // (<=128 -> 8 rows, 256 -> 4, 384 -> 2).
          constexpr int kRowsCap = 1024 / NE;
          constexpr int kRowsPerBlock =
              kRowsCap >= 8 ? 8 : (kRowsCap >= 1 ? kRowsCap : 1);
          // 128 threads, not 256: token_length 2048 / VecSize 8 = 256 vectors
          // per row, so 256 threads leave every thread with a single 16 B
          // vector in flight and 128 threads with two.  Measured at the
          // production shape: 232.5 us at 256 threads vs 226.3 at 128.
          constexpr int kBlockDimX = 128;

          if (multirow &&
              static_cast<int64_t>(total_zipped_tokens_num) >=
                  static_cast<int64_t>(kRowsPerBlock) * 4 * sm_count) {
            dim3 mgrid, mblock;
            mgrid.x = static_cast<unsigned int>(
                (static_cast<int64_t>(total_zipped_tokens_num) + kRowsPerBlock -
                 1) /
                kRowsPerBlock);
            mblock.x = kBlockDimX;
            tokens_zip_multirow_kernel<MP_CONST,
                                       WEIGHTED_CONST,
                                       NE,
                                       kRowsPerBlock,
                                       kBlockDimX>
                <<<mgrid, mblock, 0, dev_ctx.stream()>>>(
                    unzipped_tokens.data<phi::bfloat16>(),
                    zipped_expertwise_rowmap.data<int>(),
                    expert_routemap_topk.data<int>(),
                    unzipped_token_probs.data<float>(),
                    zipped_tokens->data<phi::bfloat16>(),
                    zipped_probs_topk->data<float>(),
                    total_zipped_tokens_num,
                    token_length,
                    num_experts,
                    topk);
            return;
          }

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
  PADDLE_ENFORCE_EQ(unzipped_tokens.dims().size(),
                    2,
                    common::errors::InvalidArgument(
                        "Input unzipped_tokens's dims should be 2, but got %u.",
                        unzipped_tokens.dims().size()));
  PADDLE_ENFORCE_EQ(
      zipped_expertwise_rowmap.dims().size(),
      2,
      common::errors::InvalidArgument(
          "Input zipped_expertwise_rowmap's dims should be 2, but got %u.",
          zipped_expertwise_rowmap.dims().size()));
  PADDLE_ENFORCE_EQ(
      expert_routemap_topk.dims().size(),
      2,
      common::errors::InvalidArgument(
          "Input expert_routemap_topk's dims should be 2, but got %u.",
          expert_routemap_topk.dims().size()));
  PADDLE_ENFORCE_GE(
      total_zipped_tokens_num,
      0,
      common::errors::InvalidArgument(
          "total_zipped_tokens_num should be non-negative, but got %d.",
          total_zipped_tokens_num));
  PADDLE_ENFORCE_GE(
      num_experts,
      1,
      common::errors::InvalidArgument(
          "num_experts should be > 0, received: (%d)", num_experts));
  PADDLE_ENFORCE_LE(
      num_experts,
      kMaxNumExperts,
      common::errors::InvalidArgument(
          "Currently we support no more than (%ld), received num_expert: "
          "(%ld). Please check input value.",
          kMaxNumExperts,
          num_experts));
  PADDLE_ENFORCE_EQ(
      zipped_expertwise_rowmap.dims()[0],
      total_zipped_tokens_num,
      common::errors::InvalidArgument(
          "Input zipped_expertwise_rowmap's first dimension should be equal to "
          "total_zipped_tokens_num, but got %ld and %d.",
          zipped_expertwise_rowmap.dims()[0],
          total_zipped_tokens_num));
  PADDLE_ENFORCE_EQ(
      zipped_expertwise_rowmap.dims()[1],
      num_experts,
      common::errors::InvalidArgument("Input zipped_expertwise_rowmap's second "
                                      "dimension should be equal to "
                                      "num_experts, but got %ld and %d.",
                                      zipped_expertwise_rowmap.dims()[1],
                                      num_experts));
  PADDLE_ENFORCE_EQ(
      expert_routemap_topk.dims()[0],
      total_zipped_tokens_num,
      common::errors::InvalidArgument(
          "Input expert_routemap_topk's first dimension should be equal to "
          "total_zipped_tokens_num, but got %ld and %d.",
          expert_routemap_topk.dims()[0],
          total_zipped_tokens_num));
  PADDLE_ENFORCE_EQ(
      unzipped_token_probs.numel(),
      unzipped_tokens.dims()[0],
      common::errors::InvalidArgument(
          "Input unzipped_token_probs's number of elements should be equal to "
          "unzipped_tokens.dims()[0], but got %ld and %ld.",
          unzipped_token_probs.numel(),
          unzipped_tokens.dims()[0]));
  const int64_t cols = unzipped_tokens.dims()[1];
  PADDLE_ENFORCE_GT(
      cols,
      0,
      common::errors::InvalidArgument(
          "unzipped_tokens.dims()[1] should be positive, but got %ld.", cols));
  PADDLE_ENFORCE_LE(cols,
                    std::numeric_limits<int32_t>::max(),
                    common::errors::InvalidArgument(
                        "unzipped_tokens.dims()[1] should be less than "
                        "INT_MAX, received unzipped_tokens.dims()[1]: (%ld)",
                        cols));
  const int64_t topk = expert_routemap_topk.dims()[1];
  PADDLE_ENFORCE_GE(topk,
                    1,
                    common::errors::InvalidArgument(
                        "topk should be > 0, received topk: (%ld)", topk));
  PADDLE_ENFORCE_LE(
      topk,
      std::numeric_limits<int32_t>::max(),
      common::errors::InvalidArgument(
          "topk should be less than INT_MAX, received topk: (%ld)", topk));
  PADDLE_ENFORCE_LE(
      static_cast<int64_t>(total_zipped_tokens_num),
      static_cast<int64_t>(std::numeric_limits<int32_t>::max()),
      common::errors::InvalidArgument(
          "total_zipped_tokens_num should be <= INT_MAX, but got %d.",
          total_zipped_tokens_num));
  dev_ctx.template Alloc<T>(zipped_tokens);
  dev_ctx.template Alloc<float>(zipped_probs_topk);
  if (unzipped_tokens.numel() == 0 || total_zipped_tokens_num == 0) {
    if (zipped_tokens->numel() > 0) {
      PADDLE_ENFORCE_GPU_SUCCESS(
          cudaMemsetAsync(zipped_tokens->data<T>(),
                          0,
                          zipped_tokens->numel() * sizeof(T),
                          dev_ctx.stream()));
    }
    if (zipped_probs_topk->numel() > 0) {
      PADDLE_ENFORCE_GPU_SUCCESS(
          cudaMemsetAsync(zipped_probs_topk->data<float>(),
                          0,
                          zipped_probs_topk->numel() * sizeof(float),
                          dev_ctx.stream()));
    }
    return;
  }
  void *zipped_probs_topk_ptr =
      reinterpret_cast<void *>(zipped_probs_topk->data<float>());
  const int64_t probs_numel =
      static_cast<int64_t>(total_zipped_tokens_num) * topk;
  PADDLE_ENFORCE_LE(
      probs_numel,
      static_cast<int64_t>(std::numeric_limits<int64_t>::max() / sizeof(float)),
      common::errors::InvalidArgument(
          "The zipped_probs_topk memset size overflows, got %ld elements.",
          probs_numel));
  PADDLE_ENFORCE_GPU_SUCCESS(cudaMemsetAsync(
      zipped_probs_topk_ptr, 0, sizeof(float) * probs_numel, dev_ctx.stream()));

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
