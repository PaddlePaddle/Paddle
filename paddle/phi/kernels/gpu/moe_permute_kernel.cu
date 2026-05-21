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

#include "paddle/phi/backends/gpu/cuda/cuda_graph_with_memory_pool.h"
#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/full_kernel.h"
#include "paddle/phi/kernels/gpu/moe_permute_utils.h"
#include "paddle/utils/optional.h"

#if CUDA_VERSION >= 12080
#include <cooperative_groups.h>
#include <cooperative_groups/memcpy_async.h>
#include <cuda/barrier>
#include <cuda/pipeline>
namespace cg = cooperative_groups;
#endif

namespace phi {

using moe::kCumsumInvalidTag;
using moe::kMaxNumExperts;
using moe::kPermuteBlockDimX;
using moe::kPermuteBlockSize;

// ============================================================================
//                        Unified Permute Kernel
// ============================================================================
// Register-centric scheduling: metadata lives in registers, not shared memory.
// Shared memory layout (phases are non-overlapping):
//   Phase 1:   uint32_t[num_experts] expert bitmask
//   Phase 2:   int[ROWS_PER_BLOCK * TOPK] output_rows (reuses bitmask region)
//   TMA only:  ping/pong buffers + routemap/probs (after max(bitmask, outrows))
//
// Rowmap is pre-filled with -1 by host cudaMemsetAsync before kernel launch.
// Phase 1a: Scatter routemap->bitmask, cache expert/prob in registers
// Phase 1b: Progressive cumsum + direct global writes (routemap, probs,
// indices) Phase 2:  Flush output_rows to smem once, then zero-sync data
// movement
//
template <typename TokenT,
          typename IndexT,
          typename ProbT,
          typename ScaleT,
          bool has_scale,
          bool do_gather,
          bool return_expert_indices,
          int TOPK,
          bool USE_TMA,
          int ROWS_PER_BLOCK = kPermuteBlockSize,
          int BLOCK_DIM_X = kPermuteBlockDimX>
__global__ __launch_bounds__(BLOCK_DIM_X) void permute_kernel(
    const TokenT *__restrict__ X,
    const IndexT *__restrict__ routemap_topk,
    const ProbT *__restrict__ probs_topk,
    const ScaleT *__restrict__ XScale,
    const int *__restrict__ expert_base_offset,
    const int *__restrict__ expert_base_offset_end,
    TokenT *__restrict__ X_unzipped,
    int *__restrict__ zipped_expertwise_rowmap,
    ProbT *__restrict__ probs_unzipped,
    ScaleT *__restrict__ XScale_unzipped,
    int *global_expertwise_block_cumsum,
    int *__restrict__ expert_indices,
    const int total_zipped_tokens_num,
    const int token_length,
    const int scale_length,
    const int num_experts,
    const int topk) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
  static_assert(ROWS_PER_BLOCK == 32, "ROWS_PER_BLOCK must equal warp size");

  const int lane_id = threadIdx.x & 31;
  const int warp_id = threadIdx.x >> 5;
  constexpr int warp_num = BLOCK_DIM_X >> 5;

  // ===================== Block-to-data mapping (prefix/suffix) =============
  const bool use_prefix = blockIdx.x % 2 == 0;
  const int block_index_in_X =
      use_prefix ? blockIdx.x / 2 : gridDim.x - 1 - blockIdx.x / 2;
  const int block_row_base = block_index_in_X * ROWS_PER_BLOCK;
  const int block_row_end =
      min(block_row_base + ROWS_PER_BLOCK, total_zipped_tokens_num);

  // ===================== Shared memory layout =============================
  // Section 1 (Phase 1a-1b): uint32_t[num_experts] expert bitmask
  // Section 1 (Phase 2):     int[ROWS_PER_BLOCK * TOPK] output_rows (reuses)
  // Section 2 (USE_TMA):     ping/pong + routemap + probs
  extern __shared__ char smem_raw[];
  uint32_t *expert_bitmask = reinterpret_cast<uint32_t *>(smem_raw);

  // TMA region starts after max(bitmask, output_rows), 32-byte aligned
  constexpr int output_rows_bytes = ROWS_PER_BLOCK * TOPK * sizeof(int);
  [[maybe_unused]] char *tma_base =
      smem_raw + (((max(static_cast<int>(kMaxNumExperts * sizeof(uint32_t)),
                        output_rows_bytes)) +
                   31) &
                  ~31);

  // Initialize expert bitmask
  for (int i = threadIdx.x; i < num_experts; i += BLOCK_DIM_X) {
    expert_bitmask[i] = 0u;
  }

  // ===================== TMA setup ========================================
  [[maybe_unused]] TokenT *ping_buffer = nullptr;
  [[maybe_unused]] TokenT *pong_buffer = nullptr;
  [[maybe_unused]] IndexT *shared_routemap = nullptr;
  [[maybe_unused]] ProbT *shared_probs = nullptr;

#if CUDA_VERSION >= 12080
  [[maybe_unused]] cg::thread_block cg_block = cg::this_thread_block();
  constexpr auto scope = cuda::thread_scope_block;
  constexpr int stages = 2;

#pragma nv_diag_suppress 20054
  __shared__ cuda::pipeline_shared_state<scope, stages> pstate;
#pragma nv_diag_default 20054
  [[maybe_unused]] auto pipe = cuda::make_pipeline(cg_block, &pstate);

  if constexpr (USE_TMA) {
    ping_buffer = reinterpret_cast<TokenT *>(tma_base);
    pong_buffer =
        reinterpret_cast<TokenT *>(tma_base + token_length * sizeof(TokenT));
    shared_routemap = reinterpret_cast<IndexT *>(tma_base + 2 * token_length *
                                                                sizeof(TokenT));
    shared_probs =
        reinterpret_cast<ProbT *>(reinterpret_cast<char *>(shared_routemap) +
                                  ROWS_PER_BLOCK * topk * sizeof(IndexT));

    const int local_elems = (block_row_end - block_row_base) * topk;
    pipe.producer_acquire();
    cuda::memcpy_async(
        cg_block,
        shared_routemap,
        routemap_topk + static_cast<int64_t>(block_row_base) * topk,
        local_elems * sizeof(IndexT),
        pipe);
    cuda::memcpy_async(cg_block,
                       shared_probs,
                       probs_topk + static_cast<int64_t>(block_row_base) * topk,
                       local_elems * sizeof(ProbT),
                       pipe);
    pipe.producer_commit();

    if constexpr (do_gather) {
      pipe.producer_acquire();
      cuda::memcpy_async(
          cg_block,
          ping_buffer,
          X + static_cast<int64_t>(block_row_base) * token_length,
          token_length * sizeof(TokenT),
          pipe);
      pipe.producer_commit();
    }
  }
#endif

  if constexpr (USE_TMA) {
#if CUDA_VERSION >= 12080
    pipe.consumer_wait();
    __syncthreads();
    pipe.consumer_release();
#endif
  } else {
    __syncthreads();
  }

  // ===================== Phase 1a: Scatter into bitmasks ===================
  // Every lane loads ALL its topk columns into registers so that Phase 1b
  // can always match reg_expert[k] == expert_id regardless of warp id.
  int reg_expert[TOPK];
  ProbT reg_prob[TOPK];
#pragma unroll
  for (int k = 0; k < TOPK; k++) {
    reg_expert[k] = -1;
    reg_prob[k] = ProbT(0);
  }

  const int global_row = block_row_base + lane_id;
  const bool row_valid = global_row < total_zipped_tokens_num;

  // Each lane reads all its topk entries; warps collaborate on atomicOr.
  for (int col = 0; col < topk; col++) {
    int expert = -1;
    ProbT prob = ProbT(0);
    if (row_valid) {
      if constexpr (USE_TMA) {
        expert = shared_routemap[lane_id * topk + col];
        prob = shared_probs[lane_id * topk + col];
      } else {
        expert = routemap_topk[global_row * topk + col];
        prob = probs_topk[global_row * topk + col];
      }
    }
    if (expert >= 0 && expert < num_experts) {
      // Only one warp per column does atomicOr (idempotent, but reduces
      // traffic)
      if (col % warp_num == warp_id) {
        atomicOr(&expert_bitmask[expert], 1u << lane_id);
      }
      reg_expert[col] = expert;
      reg_prob[col] = prob;
    }
  }
  __syncthreads();

  // ===================== Phase 1b: Progressive cumsum + global writes =======
  // Chain layout (even blockIdx = prefix, odd = suffix):
  //   prefix: block 0 →  block 2 →  block 4 → ...  (recv from blockIdx-2)
  //   suffix: block 1 →  block 3 →  block 5 → ...  (recv from blockIdx-2)
  // Root blocks (prefix: blockIdx==0, suffix: blockIdx==1) have no predecessor.
  // When mask==0u for an expert, lane 0 still receives & forwards the offset
  // (with local_count==0) so the chain never breaks.
  int reg_output_row[TOPK];
#pragma unroll
  for (int k = 0; k < TOPK; k++) reg_output_row[k] = -1;

  const bool is_chain_root = (use_prefix ? blockIdx.x == 0 : blockIdx.x == 1);

  for (int expert_id = warp_id; expert_id < num_experts;
       expert_id += warp_num) {
    const uint32_t mask = expert_bitmask[expert_id];
    const int local_count = __popc(mask);

    // --- Inter-block cumsum: lane 0 receives from predecessor, sends to
    //     successor.  Always executes regardless of mask to keep chain alive.
    int chain_offset = 0;
    if (lane_id == 0) {
      if (!is_chain_root) {
        const int recv_idx = blockIdx.x * num_experts + expert_id;
        while ((chain_offset =
                    atomicAdd(&global_expertwise_block_cumsum[recv_idx], 0)) ==
               kCumsumInvalidTag) {
        }
      }
      const int send_idx = (blockIdx.x + 2) * num_experts + expert_id;
      atomicExch(&global_expertwise_block_cumsum[send_idx],
                 chain_offset + local_count);
    }

    // --- Intra-block position assignment (only when this expert has tokens)
    // ---
    int final_pos = -1;
    if (mask != 0u) {
      chain_offset = __shfl_sync(0xFFFFFFFF, chain_offset, 0);
      const bool lane_active = (mask & (1u << lane_id)) != 0;
      if (lane_active && row_valid) {
        if (use_prefix) {
          final_pos = expert_base_offset[expert_id] + chain_offset +
                      __popc(mask & ((1u << lane_id) - 1));
        } else {
          final_pos = expert_base_offset_end[expert_id] - chain_offset -
                      __popc((lane_id < 31) ? (mask >> (lane_id + 1)) : 0u);
        }

        zipped_expertwise_rowmap[global_row * num_experts + expert_id] =
            final_pos;
#pragma unroll
        for (int k = 0; k < TOPK; k++) {
          if (reg_expert[k] == expert_id) {
            reg_output_row[k] = final_pos;
            probs_unzipped[final_pos] = reg_prob[k];
            if constexpr (return_expert_indices) {
              expert_indices[final_pos] = expert_id;
            }
            break;
          }
        }
      }
    }
  }

  // ===================== Phase 2: Token data movement ======================
  // All warps must finish Phase 1b before shared memory is repurposed.
  __syncthreads();

  if constexpr (do_gather) {
    // Flush output_rows from registers to shared memory (reuse bitmask region).
    // reg_output_row[k] was set by whichever warp processed the matching
    // expert; other warps still hold -1 for that k.  Pre-fill with -1 then
    // only write non-(-1) values to avoid race conditions.
    int *shared_output_rows = reinterpret_cast<int *>(smem_raw);
    for (int i = threadIdx.x; i < ROWS_PER_BLOCK * TOPK; i += BLOCK_DIM_X) {
      shared_output_rows[i] = -1;
    }
    __syncthreads();
#pragma unroll
    for (int k = 0; k < TOPK; k++) {
      if (reg_output_row[k] >= 0) {
        shared_output_rows[lane_id * TOPK + k] = reg_output_row[k];
      }
    }
    __syncthreads();

    // Data movement loop — no per-row __syncthreads needed (output_rows
    // are already fully materialized in smem). Only TMA pipeline needs sync.
    for (int row = block_row_base; row < block_row_end; row++) {
      const int internal_row = row - block_row_base;

      if constexpr (USE_TMA) {
#if CUDA_VERSION >= 12080
        pipe.consumer_wait();
        cg_block.sync();
        if (row + 1 < block_row_end) {
          TokenT *prefetch_buffer =
              (internal_row % 2 == 0) ? pong_buffer : ping_buffer;
          pipe.producer_acquire();
          cuda::memcpy_async(cg_block,
                             prefetch_buffer,
                             X + static_cast<int64_t>(row + 1) * token_length,
                             token_length * sizeof(TokenT),
                             pipe);
          pipe.producer_commit();
        }
#endif
      }

      [[maybe_unused]] TokenT *current_buffer = nullptr;
      if constexpr (USE_TMA) {
        current_buffer = (internal_row % 2 == 0) ? ping_buffer : pong_buffer;
      }

      // Read output rows from shared memory (no sync needed)
#pragma unroll
      for (int k = 0; k < TOPK; k++) {
        const int out_row = shared_output_rows[internal_row * TOPK + k];
        if (out_row < 0) continue;

        if constexpr (USE_TMA) {
#if CUDA_VERSION >= 12080
          if (threadIdx.x == 0) {
            cuda::device::experimental::cp_async_bulk_shared_to_global(
                &X_unzipped[(int64_t)out_row * (int64_t)token_length],
                current_buffer,
                token_length * sizeof(TokenT));
            cuda::device::experimental::cp_async_bulk_commit_group();
          }
#endif
        } else {
          vectorized_memcpy(
              &X[(int64_t)row * (int64_t)token_length],
              &X_unzipped[(int64_t)out_row * (int64_t)token_length],
              token_length);
        }

        if constexpr (has_scale) {
          try_vectorized_memcpy(
              &XScale[(int64_t)row * (int64_t)scale_length],
              &XScale_unzipped[(int64_t)out_row * (int64_t)scale_length],
              scale_length);
        }
      }

      if constexpr (USE_TMA) {
#if CUDA_VERSION >= 12080
        if (threadIdx.x == 0) {
          cuda::device::experimental::cp_async_bulk_wait_group_read<0>();
        }
        pipe.consumer_release();
#endif
      }
    }
  }
#endif  // __CUDA_ARCH__ >= 900
}

// ============================================================================
//                            Kernel launcher
// ============================================================================
template <typename TokenT,
          typename ProbT,
          typename IntT,
          typename ScaleT,
          bool HasScale,
          bool DoGather,
          bool ReturnIndices,
          int TOPK>
void launch_permute_kernel(const GPUContext &dev_ctx,
                           const DenseTensor &X,
                           const DenseTensor &expert_routemap_topk,
                           const DenseTensor &expert_prob_topk,
                           const paddle::optional<DenseTensor> &XScale,
                           const DenseTensor &expert_offsets,
                           const DenseTensor &expert_offset_end,
                           DenseTensor *X_unzipped,
                           DenseTensor *zipped_expertwise_rowmap,
                           DenseTensor *token_prob_unzipped,
                           DenseTensor *XScale_unzipped,
                           DenseTensor *global_expertwise_block_cumsum,
                           DenseTensor *expert_indices,
                           int total_zipped_tokens_num,
                           int token_length,
                           int scale_length,
                           int num_experts,
                           int topk,
                           int capability) {
  constexpr int ROWS_PER_BLOCK = kPermuteBlockSize;
  constexpr int BLOCK_DIM_X = kPermuteBlockDimX;

  dim3 grid((total_zipped_tokens_num + ROWS_PER_BLOCK - 1) / ROWS_PER_BLOCK);
  dim3 block(BLOCK_DIM_X);

  const TokenT *x_ptr = X.data<TokenT>();
  const IntT *routemap_ptr = expert_routemap_topk.data<IntT>();
  const ProbT *prob_ptr = expert_prob_topk.data<ProbT>();
  const ScaleT *scale_ptr = XScale ? XScale.get_ptr()->data<ScaleT>() : nullptr;
  const int *offset_ptr = expert_offsets.data<int>();
  const int *offset_end_ptr = expert_offset_end.data<int>();
  TokenT *x_out_ptr = X_unzipped->data<TokenT>();
  IntT *rowmap_out_ptr = zipped_expertwise_rowmap->data<IntT>();
  ProbT *prob_out_ptr = token_prob_unzipped->data<ProbT>();
  ScaleT *scale_out_ptr = XScale_unzipped->data<ScaleT>();
  int *cumsum_ptr = global_expertwise_block_cumsum->data<int>();
  int *expert_indices_ptr =
      (ReturnIndices && expert_indices) ? expert_indices->data<int>() : nullptr;

  [[maybe_unused]] bool use_tma = false;
#if CUDA_VERSION >= 12080
  use_tma = capability >= 90 &&
            is_aligned_in_bytes(token_length * sizeof(TokenT)) &&
            is_aligned_in_bytes(sizeof(IntT) * topk * ROWS_PER_BLOCK);
#endif

  // Shared memory: max(bitmask, output_rows) + optional TMA buffers
  constexpr int output_rows_bytes = ROWS_PER_BLOCK * TOPK * sizeof(int);
  const int base_smem = max(static_cast<int>(kMaxNumExperts * sizeof(uint32_t)),
                            output_rows_bytes);

  dispatch::Bool(use_tma, [&](auto tma_tag) {
    constexpr bool UseTMA = decltype(tma_tag)::value;

    int smem = base_smem;
    if constexpr (UseTMA) {
      smem = ((smem + 31) & ~31);
      smem += 2 * token_length * sizeof(TokenT) +
              sizeof(IntT) * TOPK * ROWS_PER_BLOCK +
              sizeof(ProbT) * TOPK * ROWS_PER_BLOCK;
    }

    auto kernel_ptr = permute_kernel<TokenT,
                                     IntT,
                                     ProbT,
                                     ScaleT,
                                     HasScale,
                                     DoGather,
                                     ReturnIndices,
                                     TOPK,
                                     UseTMA,
                                     ROWS_PER_BLOCK,
                                     BLOCK_DIM_X>;
    if (smem > 48 * 1024) {
      PADDLE_ENFORCE_GPU_SUCCESS(cudaFuncSetAttribute(
          kernel_ptr, cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
    }
    kernel_ptr<<<grid, block, smem, dev_ctx.stream()>>>(x_ptr,
                                                        routemap_ptr,
                                                        prob_ptr,
                                                        scale_ptr,
                                                        offset_ptr,
                                                        offset_end_ptr,
                                                        x_out_ptr,
                                                        rowmap_out_ptr,
                                                        prob_out_ptr,
                                                        scale_out_ptr,
                                                        cumsum_ptr,
                                                        expert_indices_ptr,
                                                        total_zipped_tokens_num,
                                                        token_length,
                                                        scale_length,
                                                        num_experts,
                                                        topk);
  });
}

// ============================================================================
//                               Dispatchers
// ============================================================================
template <typename T, typename Context>
void dispatch_permute_kernel(const Context &dev_ctx,
                             const DenseTensor &X,
                             const DenseTensor &expert_routemap_topk,
                             const DenseTensor &expert_prob_topk,
                             const paddle::optional<DenseTensor> &XScale,
                             const DenseTensor &expert_offsets,
                             const DenseTensor &expert_offset_end,
                             DenseTensor *X_unzipped,
                             DenseTensor *zipped_expertwise_rowmap,
                             DenseTensor *token_prob_unzipped,
                             DenseTensor *XScale_unzipped,
                             DenseTensor *global_expertwise_block_cumsum,
                             DenseTensor *expert_indices,
                             int total_zipped_tokens_num,
                             int token_length,
                             int topk,
                             int num_experts,
                             int scale_length,
                             bool do_gather,
                             bool using_ue8m0_scale,
                             bool return_expert_indices) {
  static int capability = dev_ctx.GetComputeCapability();

  dispatch::TokenType(X.dtype(), [&](auto token_tag, auto has_scale_tag) {
    using TokenT = typename decltype(token_tag)::type;
    constexpr bool HasScale = decltype(has_scale_tag)::value;

    dispatch::ProbType(expert_prob_topk.dtype(), [&](auto prob_tag) {
      using ProbT = typename decltype(prob_tag)::type;

      dispatch::ScaleType(using_ue8m0_scale, [&](auto scale_tag) {
        using ScaleT = typename decltype(scale_tag)::type;

        dispatch::Bools(
            [&](auto do_gather_tag, auto return_indices_tag) {
              constexpr bool DoGather = decltype(do_gather_tag)::value;
              constexpr bool ReturnIndices =
                  decltype(return_indices_tag)::value;

              dispatch::TopK(topk, [&](auto topk_tag) {
                constexpr int TK = decltype(topk_tag)::value;

                launch_permute_kernel<TokenT,
                                      ProbT,
                                      int,
                                      ScaleT,
                                      HasScale,
                                      DoGather,
                                      ReturnIndices,
                                      TK>(dev_ctx,
                                          X,
                                          expert_routemap_topk,
                                          expert_prob_topk,
                                          XScale,
                                          expert_offsets,
                                          expert_offset_end,
                                          X_unzipped,
                                          zipped_expertwise_rowmap,
                                          token_prob_unzipped,
                                          XScale_unzipped,
                                          global_expertwise_block_cumsum,
                                          expert_indices,
                                          total_zipped_tokens_num,
                                          token_length,
                                          scale_length,
                                          num_experts,
                                          topk,
                                          capability);
              });
            },
            do_gather,
            return_expert_indices);
      });
    });
  });
}

// ============================================================================
//                        Preprocessing helpers
// ============================================================================
template <typename TokenT, typename ScaleT, typename Context>
void dispatch_preprocess(const Context &dev_ctx,
                         TokenT *X_unzipped_ptr,
                         ScaleT *XScale_unzipped_ptr,
                         float *token_prob_unzipped_ptr,
                         int *expert_indices_ptr,
                         int cols,
                         int quanted_cols,
                         const std::vector<int> &padding_rows) {
  if (padding_rows.empty()) return;

  DenseTensor padding_tokens_tensor;
  padding_tokens_tensor.Resize({static_cast<int64_t>(padding_rows.size())});
  dev_ctx.template Alloc<int>(&padding_tokens_tensor);

  auto *stable_padding_rows = backends::gpu::RestoreHostMemIfCapturingCUDAGraph(
      const_cast<int *>(padding_rows.data()), padding_rows.size());
  PADDLE_ENFORCE_GPU_SUCCESS(cudaMemcpyAsync(padding_tokens_tensor.data<int>(),
                                             stable_padding_rows,
                                             sizeof(int) * padding_rows.size(),
                                             cudaMemcpyHostToDevice,
                                             dev_ctx.stream()));

  dim3 grid{static_cast<unsigned>(padding_rows.size())};
  dim3 block{512};
  const int *padding_ptr = padding_tokens_tensor.data<int>();

  dispatch::Bools(
      [&](auto fill_x_tag, auto fill_scale_tag, auto fill_indices_tag) {
        constexpr bool FillX = decltype(fill_x_tag)::value;
        constexpr bool FillScale = decltype(fill_scale_tag)::value;
        constexpr bool FillIndices = decltype(fill_indices_tag)::value;

        filling_padding_rows_kernel<TokenT,
                                    ScaleT,
                                    FillX,
                                    FillScale,
                                    FillIndices>
            <<<grid, block, 0, dev_ctx.stream()>>>(X_unzipped_ptr,
                                                   XScale_unzipped_ptr,
                                                   token_prob_unzipped_ptr,
                                                   expert_indices_ptr,
                                                   cols,
                                                   quanted_cols,
                                                   padding_ptr);
      },
      X_unzipped_ptr != nullptr,
      XScale_unzipped_ptr != nullptr,
      expert_indices_ptr != nullptr);
}

template <typename Context>
void dispatch_preprocess_w_override(const Context &dev_ctx,
                                    const DenseTensor &expert_routemap_topk,
                                    const int num_experts,
                                    const int padding_alignment,
                                    const int override_buffer_size,
                                    const bool return_expert_indices,
                                    DenseTensor *expert_offset,
                                    DenseTensor *expert_offset_end,
                                    DenseTensor *expert_indices) {
  constexpr int BLOCK_SIZE = 1024;

  dispatch::Bools(
      [&](auto fill_expert_indices_tag) {
        constexpr bool FillExpertIndices =
            decltype(fill_expert_indices_tag)::value;
        const int smem_bytes =
            static_cast<int>(sizeof(int32_t)) * num_experts * 2;
        routemap_digest_kernel<FillExpertIndices, BLOCK_SIZE>
            <<<1, BLOCK_SIZE, smem_bytes, dev_ctx.stream()>>>(
                expert_routemap_topk.data<int32_t>(),
                expert_offset->data<int32_t>(),
                expert_offset_end->data<int32_t>(),
                expert_indices->data<int32_t>(),
                expert_routemap_topk.numel(),
                num_experts,
                padding_alignment);
      },
      return_expert_indices);
}

// ============================================================================
//                               CPP Interface
// ============================================================================
template <typename T, typename Context>
void MoePermuteKernel(const Context &dev_ctx,
                      const DenseTensor &X,
                      const paddle::optional<DenseTensor> &XScale,
                      const DenseTensor &expert_routemap_topk,
                      const DenseTensor &expert_prob_topk,
                      const int num_experts,
                      const std::vector<int> &tokens_per_expert,
                      const int padding_alignment,
                      const bool do_gather,
                      const bool using_ue8m0_scale,
                      const bool return_expert_indices,
                      const int override_buffer_size,
                      DenseTensor *X_unzipped,
                      DenseTensor *zipped_expertwise_rowmap,
                      DenseTensor *token_prob_unzipped,
                      DenseTensor *XScale_unzipped,
                      DenseTensor *expert_indices) {
  const int64_t rows = X.dims()[0];
  const int64_t cols = X.dims()[1];
  const int64_t topk = expert_routemap_topk.dims()[1];
  const int64_t quanted_cols = (XScale) ? XScale.get_ptr()->dims()[1] : 0;
  const bool is_buffer_overridden = (override_buffer_size > -1);

  PADDLE_ENFORCE_LE(
      rows,
      std::numeric_limits<int32_t>::max(),
      common::errors::InvalidArgument(
          "X.dims()[0] should be less than INT_MAX, received: (%ld)", rows));
  PADDLE_ENFORCE_LE(
      cols,
      std::numeric_limits<int32_t>::max(),
      common::errors::InvalidArgument(
          "X.dims()[1] should be less than INT_MAX, received: (%ld)", cols));
  PADDLE_ENFORCE_LE(topk,
                    16,
                    common::errors::InvalidArgument(
                        "topk should be <= 16, received: (%ld)", topk));
  PADDLE_ENFORCE_LE(num_experts,
                    kMaxNumExperts,
                    common::errors::InvalidArgument(
                        "num_experts should be <= %d, received: (%d)",
                        kMaxNumExperts,
                        num_experts));
  PADDLE_ENFORCE_LE(
      quanted_cols,
      std::numeric_limits<int32_t>::max(),
      common::errors::InvalidArgument(
          "quanted_cols should be less than INT_MAX, received: (%ld)",
          quanted_cols));

  // Output allocation
  void *XScale_unzipped_ptr = nullptr;
  dev_ctx.template Alloc<T>(X_unzipped);
  dev_ctx.template Alloc<int>(zipped_expertwise_rowmap);
  dev_ctx.template Alloc<float>(token_prob_unzipped);
  dev_ctx.template Alloc<int>(expert_indices);
  auto X_unzipped_ptr = reinterpret_cast<void *>(X_unzipped->data<T>());
  auto token_prob_unzipped_ptr =
      reinterpret_cast<void *>(token_prob_unzipped->data<float>());
  if (using_ue8m0_scale) {
    dev_ctx.template Alloc<int32_t>(XScale_unzipped);
    XScale_unzipped_ptr =
        reinterpret_cast<void *>(XScale_unzipped->data<int32_t>());
  } else {
    dev_ctx.template Alloc<float>(XScale_unzipped);
    XScale_unzipped_ptr =
        reinterpret_cast<void *>(XScale_unzipped->data<float>());
  }
  // Pre-fill expert_indices with -1 via hardware DMA engine (cudaMemsetAsync).
  // (Even if input is 0-size)
  // 0xFF byte-pattern on int32 = 0xFFFFFFFF = -1 in two's complement.
  // This offloads the bulk -1 fill (~10K-500K int32s) from SM compute to the
  // DMA copy engine, running in parallel with subsequent kernel execution.
  if (is_buffer_overridden && return_expert_indices) {
    PADDLE_ENFORCE_GPU_SUCCESS(cudaMemsetAsync(
        expert_indices->data<int32_t>(),
        0xFF,
        static_cast<size_t>(override_buffer_size) * sizeof(int32_t),
        dev_ctx.stream()));
  }

  // Handle empty input: initialize all outputs properly
  if (X.numel() == 0) return;

  // Preprocess
  constexpr int kEffectiveBlockSize = kPermuteBlockSize;
  const int cumsum_blocknum =
      (rows + kEffectiveBlockSize - 1) / kEffectiveBlockSize;

  DenseTensor expert_offset_tensor;
  DenseTensor expert_offset_end_tensor;
  DenseTensor global_expertwise_block_cumsum;

  expert_offset_tensor.Resize({kMaxNumExperts});
  expert_offset_end_tensor.Resize({kMaxNumExperts});
  global_expertwise_block_cumsum.Resize(
      {static_cast<int64_t>(cumsum_blocknum + 2),
       static_cast<int64_t>(num_experts)});

  dev_ctx.template Alloc<int>(&expert_offset_tensor);
  dev_ctx.template Alloc<int>(&expert_offset_end_tensor);
  dev_ctx.template Alloc<int>(&global_expertwise_block_cumsum);

  // Pre-fill rowmap with -1 via bulk DMA (replaces scattered per-block writes)
  PADDLE_ENFORCE_GPU_SUCCESS(
      cudaMemsetAsync(zipped_expertwise_rowmap->data<int>(),
                      -1,
                      zipped_expertwise_rowmap->numel() * sizeof(int),
                      dev_ctx.stream()));

  if (cumsum_blocknum > 1) {
    PADDLE_ENFORCE_GPU_SUCCESS(
        cudaMemsetAsync(global_expertwise_block_cumsum.data<int>(),
                        -1,
                        global_expertwise_block_cumsum.numel() * sizeof(int),
                        dev_ctx.stream()));
  }

  if (is_buffer_overridden) {
    dispatch_preprocess_w_override(dev_ctx,
                                   expert_routemap_topk,
                                   num_experts,
                                   padding_alignment,
                                   override_buffer_size,
                                   return_expert_indices,
                                   &expert_offset_tensor,
                                   &expert_offset_end_tensor,
                                   expert_indices);
  } else {
    int tokens_cumulated = 0;
    std::vector<int> padding_rows;
    int expert_offset[kMaxNumExperts];
    int expert_offset_end[kMaxNumExperts];
    for (int i = 0; i < kMaxNumExperts; i++) {
      if (i < num_experts) {
        expert_offset[i] = tokens_cumulated;
        expert_offset_end[i] = expert_offset[i] + tokens_per_expert[i] - 1;
        tokens_cumulated += ((tokens_per_expert[i] + padding_alignment - 1) /
                             padding_alignment) *
                            padding_alignment;
      } else {
        expert_offset[i] = 0;
      }
    }
    auto *stable_expert_offset =
        backends::gpu::RestoreHostMemIfCapturingCUDAGraph(expert_offset,
                                                          kMaxNumExperts);
    PADDLE_ENFORCE_GPU_SUCCESS(cudaMemcpyAsync(expert_offset_tensor.data<int>(),
                                               stable_expert_offset,
                                               sizeof(int) * kMaxNumExperts,
                                               cudaMemcpyHostToDevice,
                                               dev_ctx.stream()));
    auto *stable_expert_offset_end =
        backends::gpu::RestoreHostMemIfCapturingCUDAGraph(expert_offset_end,
                                                          kMaxNumExperts);
    PADDLE_ENFORCE_GPU_SUCCESS(
        cudaMemcpyAsync(expert_offset_end_tensor.data<int>(),
                        stable_expert_offset_end,
                        sizeof(int) * kMaxNumExperts,
                        cudaMemcpyHostToDevice,
                        dev_ctx.stream()));
    for (int i = 0; i < num_experts; i++) {
      int64_t next_expert_offset =
          i < num_experts - 1 ? expert_offset[i + 1] : tokens_cumulated;
      int64_t invalid_rows =
          next_expert_offset - expert_offset[i] - tokens_per_expert[i];
      int64_t cur_expert_end = expert_offset[i] + tokens_per_expert[i];
      for (int j = 0; j < invalid_rows; ++j) {
        padding_rows.push_back(cur_expert_end + j);
      }
    }
    if (using_ue8m0_scale) {
      dispatch_preprocess(dev_ctx,
                          do_gather ? X_unzipped->data<T>() : nullptr,
                          XScale ? XScale_unzipped->data<int32_t>() : nullptr,
                          token_prob_unzipped->data<float>(),
                          expert_indices->data<int>(),
                          cols,
                          quanted_cols,
                          padding_rows);
    } else {
      dispatch_preprocess(dev_ctx,
                          do_gather ? X_unzipped->data<T>() : nullptr,
                          XScale ? XScale_unzipped->data<float>() : nullptr,
                          token_prob_unzipped->data<float>(),
                          expert_indices->data<int>(),
                          cols,
                          quanted_cols,
                          padding_rows);
    }
  }

  // Kernel dispatch
  dispatch_permute_kernel<T, Context>(
      dev_ctx,
      X,
      expert_routemap_topk,
      expert_prob_topk,
      XScale,
      expert_offset_tensor,
      expert_offset_end_tensor,
      X_unzipped,
      zipped_expertwise_rowmap,
      token_prob_unzipped,
      XScale_unzipped,
      &global_expertwise_block_cumsum,
      expert_indices,
      static_cast<int>(rows),
      static_cast<int>(cols),
      static_cast<int>(topk),
      num_experts,
      static_cast<int>(quanted_cols),
      do_gather,
      using_ue8m0_scale,
      !is_buffer_overridden && return_expert_indices);
}

}  // namespace phi

PD_REGISTER_KERNEL(moe_permute,
                   GPU,
                   ALL_LAYOUT,
                   phi::MoePermuteKernel,
                   phi::float8_e4m3fn,
                   phi::bfloat16) {}
