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

// Import MoE constants from shared header
using moe::kCumsumBlockSize;
using moe::kCumsumInvalidTag;
using moe::kMaxNumExperts;
using moe::kMaxNumExpertsForOptKernel;

// ============================================================================
//                                Kernels
// ============================================================================

// Generic permute kernel for large num_experts or Hopper architecture.
// ROWS_PER_BLOCK : token rows each block processes (compile-time).
// BLOCK_DIM_X    : threads per block (compile-time, also drives
// __launch_bounds__). For FP8, both can be tuned via moe::kFp8CumsumBlockSize /
// kFp8BlockDimX.
template <typename TokenT,
          typename IndexT,
          typename ProbT,
          typename ScaleT,
          bool has_scale,
          bool do_gather,
          bool return_expert_indices,
          int max_num_experts,
          int ROWS_PER_BLOCK = moe::kCumsumBlockSize,
          int BLOCK_DIM_X = 512>
__global__ __launch_bounds__(BLOCK_DIM_X) void permute_generic_kernel(
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
  using SlotInfo = ExpertSlotInfo<ProbT>;
  const int block_row_base = blockIdx.x * ROWS_PER_BLOCK;

  // Dynamic shared memory layout:
  //   [0, ROWS_PER_BLOCK * max_num_experts * sizeof(SlotInfo))  :
  //   shared_slot_info [above, +max_num_experts * sizeof(uint32_t)) : expert
  //   mask (only when ROWS_PER_BLOCK==32)
  extern __shared__ char smem_raw[];
  SlotInfo(*shared_slot_info)[max_num_experts] =
      reinterpret_cast<SlotInfo(*)[max_num_experts]>(smem_raw);

  // Init shared memory (row_idx=-1, prob=0)
#pragma unroll
  for (int i = threadIdx.x; i < ROWS_PER_BLOCK * max_num_experts;
       i += blockDim.x) {
    shared_slot_info[i / max_num_experts][i % max_num_experts] = SlotInfo();
  }
  __syncthreads();

  // ---------------Expertwise deterministic job scheduling ---------------
  if constexpr (ROWS_PER_BLOCK == 32) {
    // -- Warp-level ballot/popc optimization (ROWS_PER_BLOCK == warp_size) --
    //
    // Phase-1a: Each warp handles one topk column.  For each lane's token,
    //   atomicOr the expert bit into a 32-bit mask (one per expert).
    // Phase-1b: Each warp handles one expert.  __popc gives the per-block
    //   count and per-token position in O(1) hardware ops.
    //
    // Key advantage: blockDim.x is fully decoupled from num_experts.
    const int lane_id = threadIdx.x & 31;
    const int warp_id = threadIdx.x >> 5;
    const int warp_num = BLOCK_DIM_X >> 5;

    // Expert bitmask: bit i = "token i in this block is assigned to expert"
    uint32_t *block_expert_mask = reinterpret_cast<uint32_t *>(
        smem_raw + ROWS_PER_BLOCK * max_num_experts * sizeof(SlotInfo));
    for (int i = threadIdx.x; i < num_experts; i += blockDim.x) {
      block_expert_mask[i] = 0u;
    }
    __syncthreads();

    // Phase 1a: scatter token→expert assignments into bitmasks + capture probs
    const int global_row = block_row_base + lane_id;
    const bool row_valid = global_row < total_zipped_tokens_num;
    for (int col = warp_id; col < topk; col += warp_num) {
      int expert = -1;
      ProbT prob = ProbT(0);
      if (row_valid) {
        expert = routemap_topk[global_row * topk + col];
        prob = probs_topk[global_row * topk + col];
      }
      if (expert >= 0 && expert < num_experts) {
        atomicOr(&block_expert_mask[expert], 1u << lane_id);
        shared_slot_info[lane_id][expert].prob = prob;
      }
    }
    __syncthreads();

    // Phase 1b: per-expert offset via popc + inter-block cumsum chain
    for (int expert_id = warp_id; expert_id < num_experts;
         expert_id += warp_num) {
      const uint32_t mask = block_expert_mask[expert_id];
      const int local_count = __popc(mask);

      // Warp leader: inter-block cumsum + fold expert base offset
      int cumsum_offset = 0;
      if (lane_id == 0) {
        const int base = expert_base_offset[expert_id];
        const int recv_idx = blockIdx.x * num_experts + expert_id;
        const int send_idx = (blockIdx.x + 1) * num_experts + expert_id;
        if (blockIdx.x != 0) {
          while ((cumsum_offset = atomicAdd(
                      &global_expertwise_block_cumsum[recv_idx], 0)) ==
                 kCumsumInvalidTag) {
          }
        }
        // Propagate cumulative count (without base) to next block
        global_expertwise_block_cumsum[send_idx] = cumsum_offset + local_count;
        // Fold base into offset for this block's position calculation
        cumsum_offset += base;
      }
      cumsum_offset = __shfl_sync(0xFFFFFFFF, cumsum_offset, 0);

      // Each lane: 0-based position = popc of bits below this lane
      if (mask & (1u << lane_id)) {
        const int local_pos = __popc(mask & ((1u << lane_id) - 1));
        shared_slot_info[lane_id][expert_id].row_idx =
            cumsum_offset + local_pos;
      }
    }
  } else {
    // Original Phase-1: one thread per expert, sequential cumsum.
    // Requires BLOCK_DIM_X >= num_experts.
    int local_cumsum = 0;
    int cumsum_offset = (blockIdx.x != 0) * kCumsumInvalidTag;
    if (threadIdx.x < num_experts) {
      const int expert_id = threadIdx.x;
      const int local_expert_offsets = expert_base_offset[expert_id];

      for (int row = block_row_base; row < block_row_base + ROWS_PER_BLOCK;
           row++) {
        if (row >= total_zipped_tokens_num) break;
        const int internal_row = row - block_row_base;
#pragma unroll
        for (int k = 0; k < topk; k++) {
          SlotInfo proposed = {routemap_topk[row * topk + k],
                               probs_topk[row * topk + k]};
          if (proposed.row_idx == -1) continue;
          if (threadIdx.x == proposed.row_idx) {
            shared_slot_info[internal_row][expert_id] = {
                local_cumsum + local_expert_offsets, proposed.prob};
            local_cumsum += 1;
          }
        }
      }
      // Inter-block communication
      const int recv_signal_idx = blockIdx.x * num_experts + threadIdx.x;
      const int send_signal_idx = (blockIdx.x + 1) * num_experts + threadIdx.x;
      if (blockIdx.x != 0) {
        while ((cumsum_offset = atomicAdd(
                    &global_expertwise_block_cumsum[recv_signal_idx], 0)) ==
               kCumsumInvalidTag) {
        }
      }
      const int proposed_offset = cumsum_offset + local_cumsum;
      global_expertwise_block_cumsum[send_signal_idx] = proposed_offset;
      // Intra-block communication: apply cumsum offset to all rows
#pragma unroll
      for (int i = 0; i < ROWS_PER_BLOCK; i++) {
        shared_slot_info[i][expert_id].row_idx =
            (shared_slot_info[i][expert_id].row_idx == -1)
                ? -1
                : shared_slot_info[i][expert_id].row_idx + cumsum_offset;
      }
    }
  }

  // --------------------------- Jobs schedule done -------------------------
  __syncthreads();
  const int block_row_end =
      min(block_row_base + ROWS_PER_BLOCK, total_zipped_tokens_num);
  for (int row = block_row_base; row < block_row_end; row++) {
    // OOB check
    if (row >= total_zipped_tokens_num) return;
    const int internal_row = row - block_row_base;
    int hits = 0;
    for (int expert = 0; expert < num_experts; expert++) {
      const SlotInfo slot = shared_slot_info[internal_row][expert];
      const int output_row = slot.row_idx;
      if (threadIdx.x == 0)
        zipped_expertwise_rowmap[row * num_experts + expert] = output_row;
      if (output_row == -1) continue;  // no memcpy
      if (threadIdx.x == 0) {
        probs_unzipped[output_row] = slot.prob;
        if constexpr (return_expert_indices) {
          expert_indices[output_row] = expert;
        }
      }

      if constexpr (do_gather) {
        // vec copy
        if constexpr (has_scale) {
          // src or dst may be unaligned with 128bits
          try_vectorized_memcpy(
              &XScale[(int64_t)row * (int64_t)scale_length],
              &XScale_unzipped[(int64_t)output_row * (int64_t)scale_length],
              scale_length);
        }
        vectorized_memcpy(
            &X[(int64_t)row * (int64_t)token_length],
            &X_unzipped[(int64_t)output_row * (int64_t)token_length],
            token_length);
      }
    }
  }
}

// Optimized kernel for blackwell+ architecture, in small num_experts
template <typename TokenT,
          typename IndexT,
          typename ProbT,
          typename ScaleT,
          bool has_scale,
          bool do_gather,
          bool return_expert_indices,
          int max_num_experts>
__global__ __launch_bounds__(256) void permute_opt_kernel(
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
// This kernel need TMA support, so it only be compiled on Hopper or above
// architecture
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
  using SlotInfo = ExpertSlotInfo<ProbT>;
  int local_cumsum = 0;
  int local_expert_offsets;
  int local_expert_end_offsets;
  // clang-format off
  // The task allocation for each block is as follows:
  //blockIdx.x    0   2  4   6  ..   5    3    1      // NOLINT
  // X          ｜X0｜X1｜X2｜X3|...|Xn-2|Xn-1| Xn |   // NOLINT
  // clang-format on
  const bool use_cumsum = blockIdx.x % 2 == 0;
  int block_index_in_X = 0;
  if (use_cumsum) {
    block_index_in_X = blockIdx.x / 2;
  } else {
    block_index_in_X = gridDim.x - 1 - blockIdx.x / 2;
  }
  const int block_row_base = block_index_in_X * kCumsumBlockSize;
  const int block_row_end =
      min(block_row_base + kCumsumBlockSize, total_zipped_tokens_num);
  int cumsum_offset = (blockIdx.x != 0) * kCumsumInvalidTag;
  __shared__ SlotInfo shared_slot_info[kCumsumBlockSize][max_num_experts];

  constexpr int stages = 2;
  // Preload the tokens to smem
  extern __shared__ float
      smem_fp32[];  // NVCC do not support extern __shared__ TokenT smem[];
  TokenT *smem = reinterpret_cast<TokenT *>(smem_fp32);
  TokenT *ping_buffer = smem + 0 * token_length;
  TokenT *pong_buffer = smem + 1 * token_length;

  IndexT *__restrict__ shared_routemap_topk =
      reinterpret_cast<IndexT *>(smem + token_length * stages);
  ProbT *__restrict__ shared_probs_topk =
      reinterpret_cast<ProbT *>(shared_routemap_topk + kCumsumBlockSize * topk);

  cg::thread_block block = cg::this_thread_block();
  constexpr auto scope = cuda::thread_scope_block;

// Suppress NVCC warning about dynamic initialization -
// cuda::pipeline_shared_state is trivially initializable and designed for use
// with __shared__ memory
#pragma nv_diag_suppress 20054
  __shared__ cuda::pipeline_shared_state<scope, stages> pstate;
#pragma nv_diag_default 20054
  auto pipe = cuda::make_pipeline(block, &pstate);

  const int local_rowmap_size = (block_row_end - block_row_base) * topk;
  // Asynchronously read the topk map and probs for current block into smem
  pipe.producer_acquire();
  cuda::memcpy_async(
      block,
      shared_routemap_topk,
      routemap_topk + static_cast<int64_t>(block_row_base) * topk,
      cuda::aligned_size_t<32>(local_rowmap_size * sizeof(IndexT)),
      pipe);

  cuda::memcpy_async(
      block,
      shared_probs_topk,
      probs_topk + static_cast<int64_t>(block_row_base) * topk,
      cuda::aligned_size_t<32>(local_rowmap_size * sizeof(ProbT)),
      pipe);
  pipe.producer_commit();

  if constexpr (do_gather) {
    // Prime stage 0 for tokens prefetch read
    pipe.producer_acquire();
    cuda::memcpy_async(block,
                       ping_buffer,
                       X + static_cast<int64_t>(block_row_base) * token_length,
                       cuda::aligned_size_t<32>(token_length * sizeof(TokenT)),
                       pipe);
    pipe.producer_commit();
  }

  // Init shared memory
#pragma unroll
  for (int i = threadIdx.x; i < kCumsumBlockSize * max_num_experts;
       i += blockDim.x) {
    shared_slot_info[i / max_num_experts][i % max_num_experts] = SlotInfo();
  }
  // Waiting for the shared_routemap_topk and shared_probs_topk async loading
  pipe.consumer_wait();
  __syncthreads();
  // Finishing  shared_routemap_topk and shared_probs_topk loading, we should
  // release the stage
  pipe.consumer_release();

  // ---------------Expertwise deterministic job scheduling ---------------
  if (threadIdx.x < num_experts) {
    const int expert_id = threadIdx.x;
    local_expert_offsets = expert_base_offset[expert_id];
    local_expert_end_offsets = expert_base_offset_end[expert_id];
    // clang-format off
    // If blockid.x is even, we will use cumsum to compute the position in X_unzipped,          // NOLINT
    // the dependency relationship between blocks is here:                                      // NOLINT
    //    ----------<------- ---------<-------                   -----------<-----------        // NOLINT
    //    |                 |                 |                 |                       |       // NOLINT
    // | block0 | block1 | block2 | block3 | block4 | ... | block2n -2 | block2n-1 |  block2n | // NOLINT
    // clang-format on
    // Compute the cumsum from the block with a smaller idx to the block with a
    // larger idx  which block idx is even
    if (use_cumsum) {
      for (int row = block_row_base; row < block_row_end; row++) {
        const int internal_row = row - block_row_base;
#pragma unroll
        for (int k = 0; k < topk; k++) {
          SlotInfo proposed = {shared_routemap_topk[internal_row * topk + k],
                               shared_probs_topk[internal_row * topk + k]};
          // const SlotInfo proposed = {routemap_topk[row * topk + k],
          //                     probs_topk[row * topk + k]};
          if (proposed.row_idx == -1) continue;
          if (threadIdx.x == proposed.row_idx) {
            shared_slot_info[internal_row][expert_id] = {
                local_cumsum + local_expert_offsets, proposed.prob};
            local_cumsum += 1;
          }
        }
      }
      // Inter-block communication
      const int recv_signal_idx = blockIdx.x * num_experts + threadIdx.x;
      const int send_signal_idx = (blockIdx.x + 2) * num_experts + threadIdx.x;
      if (blockIdx.x != 0) {
        // signal receive from previous block, using light-weight
        // atomicAdd(check) this will not change any data, only do fetch in
        // low-cost
        while ((cumsum_offset = atomicAdd(
                    &global_expertwise_block_cumsum[recv_signal_idx], 0)) ==
               kCumsumInvalidTag) {
        }
      }
      // signal send for next block, with current cumsum
      const int proposed_offset = cumsum_offset + local_cumsum;
      global_expertwise_block_cumsum[send_signal_idx] = proposed_offset;
// Intra-block communication;
#pragma unroll
      for (int i = 0; i < kCumsumBlockSize; i++) {
        shared_slot_info[i][expert_id].row_idx =
            (shared_slot_info[i][expert_id].row_idx == -1)
                ? -1
                : shared_slot_info[i][expert_id].row_idx + cumsum_offset;
      }
    } else {
      // clang-format off
      // If blockid.x is odd, we will use suffix sum to compute the position in X_unzipped,         // NOLINT
      // the dependency relationship between blocks is here:                                        // NOLINT
      //               ----------<-------                           --------<---------------        // NOLINT
      //              |                 |                         |                         |       // NOLINT
      // | block0 | block1 | block2 | block3 | block4 | ... | block2n - 1 | block2n | block2n+1 |   // NOLINT
      // clang-format on
      int local_suffixsum = 0;
      for (int row = block_row_end - 1; row >= block_row_base; --row) {
        const int internal_row = row - block_row_base;
#pragma unroll
        for (int k = 0; k < topk; k++) {
          SlotInfo proposed = {shared_routemap_topk[internal_row * topk + k],
                               shared_probs_topk[internal_row * topk + k]};
          if (proposed.row_idx == -1) continue;
          if (threadIdx.x == proposed.row_idx) {
            shared_slot_info[internal_row][expert_id] = {local_suffixsum,
                                                         proposed.prob};
            local_suffixsum += 1;
          }
        }
      }
      // Inter-block communication
      const int recv_signal_idx = (blockIdx.x) * num_experts + threadIdx.x;
      const int send_signal_idx = (blockIdx.x + 2) * num_experts + threadIdx.x;
      int suffixsum_offset = 0;
      if (blockIdx.x != 1) {
        // signal receive from previous block, using light-weight
        // atomicAdd(check) this will not change any data, only do fetch in
        // low-cost
        while ((suffixsum_offset = atomicAdd(
                    &global_expertwise_block_cumsum[recv_signal_idx], 0)) ==
               kCumsumInvalidTag) {
        }
      }
      // signal send for next block, with current cumsum
      const int proposed_offset = suffixsum_offset + local_suffixsum;
      global_expertwise_block_cumsum[send_signal_idx] = proposed_offset;
// Intra-block communication;
#pragma unroll
      for (int i = 0; i < kCumsumBlockSize; i++) {
        shared_slot_info[i][expert_id].row_idx =
            (shared_slot_info[i][expert_id].row_idx == -1)
                ? -1
                : local_expert_end_offsets -
                      (shared_slot_info[i][expert_id].row_idx +
                       suffixsum_offset);
      }
    }
  }
  // --------------------------- Jobs schedule done -------------------------
  __syncthreads();

  for (int row = block_row_base; row < block_row_end; row++) {
    const int internal_row = row - block_row_base;
    TokenT *current_buffer =
        (internal_row % 2 == 0) ? ping_buffer : pong_buffer;
    TokenT *prefetch_buffer =
        (internal_row % 2 == 0) ? pong_buffer : ping_buffer;
    if constexpr (do_gather) {
      // wait async loading to SMEM(current_buffer) done
      pipe.consumer_wait();
      block.sync();  // ensure shared memory is ready

      // start next stage
      if (row + 1 < block_row_end) {
        pipe.producer_acquire();
        // Start loading the next rows token into SMEM(prefetch_buffer)
        cuda::memcpy_async(
            block,
            prefetch_buffer,
            X + static_cast<int64_t>(row + 1) * token_length,
            cuda::aligned_size_t<32>(token_length * sizeof(TokenT)),
            pipe);

        pipe.producer_commit();
      }
    }

    int hits = 0;
    for (int expert = 0; expert < num_experts; expert++) {
      const SlotInfo slot = shared_slot_info[internal_row][expert];
      const int output_row = slot.row_idx;
      if (threadIdx.x == 0)
        zipped_expertwise_rowmap[row * num_experts + expert] = output_row;
      if (output_row == -1) continue;  // no memcpy
      if (threadIdx.x == 0) {
        probs_unzipped[output_row] = slot.prob;
        if constexpr (return_expert_indices) {
          expert_indices[output_row] = expert;
        }
      }

      if constexpr (do_gather) {
#if CUDA_VERSION >= 12080
        // Using TMA copy data from SMEM to GMEM
        if (threadIdx.x == 0) {
          cuda::device::experimental::cp_async_bulk_shared_to_global(
              &X_unzipped[(int64_t)output_row * (int64_t)token_length],
              current_buffer,
              token_length * sizeof(TokenT));
          cuda::device::experimental::cp_async_bulk_commit_group();
        }
#else
        vectorized_memcpy(
            current_buffer,
            &X_unzipped[(int64_t)output_row * (int64_t)token_length],
            token_length);
#endif
        // vec copy
        if constexpr (has_scale) {
          // src or dst may be unaligned with 128bits
          try_vectorized_memcpy(
              &XScale[(int64_t)row * (int64_t)scale_length],
              &XScale_unzipped[(int64_t)output_row * (int64_t)scale_length],
              scale_length);
        }
      }

      // Early exit: each row has at most topk expert assignments.
      // Remaining entries are guaranteed to be -1; flush them and break.
      if (++hits >= topk) {
        if (threadIdx.x == 0) {
          for (int e = expert + 1; e < num_experts; e++) {
            zipped_expertwise_rowmap[row * num_experts + e] = -1;
          }
        }
        break;
      }
    }
    if constexpr (do_gather) {
#if CUDA_VERSION >= 12080
      // waiting async writing from SMEM to GMEM done
      if (threadIdx.x == 0) {
        cuda::device::experimental::cp_async_bulk_wait_group_read<0>();
      }
#endif
      // release stage for SMEM (current_buffer)
      pipe.consumer_release();
    }
  }
#endif
}

// ============================================================================
//                                Kernel launcher
// ============================================================================
template <typename TokenT,
          typename ProbT,
          typename IntT,
          typename ScaleT,
          bool HasScale,
          bool DoGather,
          bool ReturnIndices,
          int MaxNumExperts>
void launch_permute_kernel(const phi::GPUContext &dev_ctx,
                           const phi::DenseTensor &X,
                           const phi::DenseTensor &expert_routemap_topk,
                           const phi::DenseTensor &expert_prob_topk,
                           const paddle::optional<phi::DenseTensor> &XScale,
                           const phi::DenseTensor &expert_offsets,
                           const phi::DenseTensor &expert_offset_end,
                           phi::DenseTensor *X_unzipped,
                           phi::DenseTensor *zipped_expertwise_rowmap,
                           phi::DenseTensor *token_prob_unzipped,
                           phi::DenseTensor *XScale_unzipped,
                           phi::DenseTensor *global_expertwise_block_cumsum,
                           phi::DenseTensor *expert_indices,
                           int total_zipped_tokens_num,
                           int token_length,
                           int scale_length,
                           int num_experts,
                           int topk,
                           int capability) {
  dim3 grid, block;

  // Determine whether to use optimized kernel (Hopper+, aligned, small experts)
  [[maybe_unused]] bool use_opt_kernel = false;
#if CUDA_VERSION >= 12080
  use_opt_kernel = capability >= 100 &&
                   num_experts <= kMaxNumExpertsForOptKernel &&
                   is_aligned_in_bytes(token_length * sizeof(TokenT)) &&
                   is_aligned_in_bytes(sizeof(IntT) * topk * kCumsumBlockSize);
#endif

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

#if CUDA_VERSION >= 12080
  if (use_opt_kernel) {
    grid.x =
        (total_zipped_tokens_num + kCumsumBlockSize - 1) / kCumsumBlockSize;
    block.x = 256;
    const int smem = 2 * token_length * sizeof(TokenT) +
                     (sizeof(IntT) + sizeof(ProbT)) * topk * kCumsumBlockSize;
    permute_opt_kernel<TokenT,
                       IntT,
                       ProbT,
                       ScaleT,
                       HasScale,
                       DoGather,
                       ReturnIndices,
                       kMaxNumExpertsForOptKernel>
        <<<grid, block, smem, dev_ctx.stream()>>>(x_ptr,
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
    return;
  }
#endif
  // Fallback to generic kernel
  //
  // FP8 uses warp-ballot Phase-1 optimization when kFp8CumsumBlockSize == 32,
  // which fully decouples blockDim.x from num_experts — any power-of-2 works.
  // BF16 uses the original one-thread-per-expert Phase-1 (blockDim.x >=
  // num_experts).
  constexpr int rows_per_block =
      (sizeof(TokenT) == 1) ? moe::kFp8CumsumBlockSize : kCumsumBlockSize;
  constexpr int block_dim_x = (sizeof(TokenT) == 1) ? moe::kFp8BlockDimX : 512;

  // Compile-time sanity checks
  static_assert(rows_per_block > 0, "rows_per_block must be positive");
  static_assert(block_dim_x >= 32 && block_dim_x <= 1024 &&
                    (block_dim_x & (block_dim_x - 1)) == 0,
                "block_dim_x must be a power-of-2 in [32, 1024]");
  // Original Phase-1 fallback requires blockDim.x >= MaxNumExperts
  static_assert(
      rows_per_block == 32 || block_dim_x >= MaxNumExperts,
      "Non-warp-optimized path requires block_dim_x >= MaxNumExperts");

  grid.x = (total_zipped_tokens_num + rows_per_block - 1) / rows_per_block;
  block.x = block_dim_x;

  // Shared memory: slot_info + expert bitmask (warp-ballot path only)
  constexpr int generic_smem =
      rows_per_block * MaxNumExperts * sizeof(ExpertSlotInfo<ProbT>) +
      (rows_per_block == 32 ? MaxNumExperts * sizeof(uint32_t) : 0);
  auto generic_kernel_ptr = permute_generic_kernel<TokenT,
                                                   IntT,
                                                   ProbT,
                                                   ScaleT,
                                                   HasScale,
                                                   DoGather,
                                                   ReturnIndices,
                                                   MaxNumExperts,
                                                   rows_per_block,
                                                   block_dim_x>;
  if constexpr (generic_smem > 48 * 1024) {
    PADDLE_ENFORCE_GPU_SUCCESS(
        cudaFuncSetAttribute(generic_kernel_ptr,
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             generic_smem));
  }
  generic_kernel_ptr<<<grid, block, generic_smem, dev_ctx.stream()>>>(
      x_ptr,
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
}

// ============================================================================
//                               Dispatchers
// ============================================================================
template <typename T, typename Context>
void dispatch_permute_kernel(const Context &dev_ctx,
                             const phi::DenseTensor &X,
                             const phi::DenseTensor &expert_routemap_topk,
                             const phi::DenseTensor &expert_prob_topk,
                             const paddle::optional<phi::DenseTensor> &XScale,
                             const phi::DenseTensor &expert_offsets,
                             const phi::DenseTensor &expert_offset_end,
                             phi::DenseTensor *X_unzipped,
                             phi::DenseTensor *zipped_expertwise_rowmap,
                             phi::DenseTensor *token_prob_unzipped,
                             phi::DenseTensor *XScale_unzipped,
                             phi::DenseTensor *global_expertwise_block_cumsum,
                             phi::DenseTensor *expert_indices,
                             int total_zipped_tokens_num,
                             int token_length,
                             int topk,
                             int num_experts,
                             int scale_length,
                             bool do_gather,
                             bool using_ue8m0_scale,
                             bool return_expert_indices) {
  static int capability = dev_ctx.GetComputeCapability();

  // Inlined high performance & extensible dispatch route
  // Token type dispatch: dtype -> (TokenT, HasScale)
  dispatch::TokenType(X.dtype(), [&](auto token_tag, auto has_scale_tag) {
    using TokenT = typename decltype(token_tag)::type;
    constexpr bool HasScale = decltype(has_scale_tag)::value;

    // Prob type dispatch
    dispatch::ProbType(expert_prob_topk.dtype(), [&](auto prob_tag) {
      using ProbT = typename decltype(prob_tag)::type;

      // Scale type dispatch
      dispatch::ScaleType(using_ue8m0_scale, [&](auto scale_tag) {
        using ScaleT = typename decltype(scale_tag)::type;

        // Boolean flags compile-time recursive dispatch
        dispatch::Bools(
            [&](auto do_gather_tag, auto return_indices_tag) {
              constexpr bool DoGather = decltype(do_gather_tag)::value;
              constexpr bool ReturnIndices =
                  decltype(return_indices_tag)::value;

              // Bucketed num_experts dispatch
              dispatch::NumExperts(num_experts, [&](auto ne_tag) {
                constexpr int NE = decltype(ne_tag)::value;

                launch_permute_kernel<TokenT,
                                      ProbT,
                                      int,
                                      ScaleT,
                                      HasScale,
                                      DoGather,
                                      ReturnIndices,
                                      NE>(dev_ctx,
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

  // Allocate GPU memory for padding_rows
  DenseTensor padding_tokens_tensor;
  padding_tokens_tensor.Resize({static_cast<int64_t>(padding_rows.size())});
  dev_ctx.template Alloc<int>(&padding_tokens_tensor);

  PADDLE_ENFORCE_GPU_SUCCESS(cudaMemcpyAsync(padding_tokens_tensor.data<int>(),
                                             padding_rows.data(),
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
  constexpr int GRID_SIZE = 1;

  dim3 grid{GRID_SIZE};  // using 1 block to avoid unnecessary global atomics
                         // and sync.
  dim3 block{BLOCK_SIZE};
  dispatch::Bools(
      [&](auto fill_expert_indices_tag) {
        constexpr bool FillExpertIndices =
            decltype(fill_expert_indices_tag)::value;
        const int smem_bytes =
            static_cast<int>(sizeof(int32_t)) * num_experts * 2;
        routemap_digest_kernel<FillExpertIndices, BLOCK_SIZE>
            <<<grid, block, smem_bytes, dev_ctx.stream()>>>(
                expert_routemap_topk.data<int32_t>(),
                expert_offset->data<int32_t>(),
                expert_offset_end->data<int32_t>(),
                expert_indices->data<int32_t>(),
                expert_routemap_topk.numel(),
                num_experts,
                padding_alignment,
                override_buffer_size);
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
  // ====================================================================
  //                            Input checks
  // ====================================================================
  const int64_t rows = X.dims()[0];
  const int64_t cols = X.dims()[1];
  const int64_t topk = expert_routemap_topk.dims()[1];
  const int64_t quanted_cols = (XScale) ? XScale.get_ptr()->dims()[1] : 0;
  const bool is_buffer_overridden = (override_buffer_size > -1);
  PADDLE_ENFORCE_LE(
      rows,
      std::numeric_limits<int32_t>::max(),
      common::errors::InvalidArgument("X.dims()[0] should be less than "
                                      "INT_MAX, received X.dims()[0]: (%ld)",
                                      rows));
  PADDLE_ENFORCE_LE(
      cols,
      std::numeric_limits<int32_t>::max(),
      common::errors::InvalidArgument("X.dims()[1] should be less than "
                                      "INT_MAX, received X.dims()[1]: (%ld)",
                                      cols));
  PADDLE_ENFORCE_LE(
      topk,
      std::numeric_limits<int32_t>::max(),
      common::errors::InvalidArgument(
          "topk should be less than INT_MAX, received topk: (%ld)", topk));
  PADDLE_ENFORCE_LE(
      num_experts,
      kMaxNumExperts,
      common::errors::InvalidArgument(
          "Currently we support no more than (%ld), received num_expert: "
          "(%ld). Please check input "
          "value.",
          kMaxNumExperts,
          num_experts));
  PADDLE_ENFORCE_LE(
      quanted_cols,
      std::numeric_limits<int32_t>::max(),
      common::errors::InvalidArgument("quanted_cols should be less than "
                                      "INT_MAX, received quanted_cols: (%ld)",
                                      quanted_cols));

  // ====================================================================
  //             Output resource allocation & 0-size handling
  // ====================================================================
  void *XScale_unzipped_ptr = nullptr;

  dev_ctx.template Alloc<T>(X_unzipped);
  dev_ctx.template Alloc<int>(zipped_expertwise_rowmap);
  dev_ctx.template Alloc<float>(token_prob_unzipped);
  dev_ctx.template Alloc<int>(expert_indices);
  auto X_unzipped_ptr = reinterpret_cast<void *>(X_unzipped->data<T>());
  auto token_prob_unzipped_ptr =
      reinterpret_cast<void *>(token_prob_unzipped->data<float>());
  if (using_ue8m0_scale) {
    // if using the ue8m0 scale, four ue8m0 scale will be packed into one int32
    dev_ctx.template Alloc<int32_t>(XScale_unzipped);
    XScale_unzipped_ptr =
        reinterpret_cast<void *>(XScale_unzipped->data<int32_t>());
  } else {
    dev_ctx.template Alloc<float>(XScale_unzipped);
    XScale_unzipped_ptr =
        reinterpret_cast<void *>(XScale_unzipped->data<float>());
  }
  // Handle 0-size input
  if (X.numel() == 0) return;

  // ====================================================================
  //                    Preprocess helper tensors
  // ====================================================================
  // The cumsum buffer must accommodate the largest possible grid.x across
  // all kernel paths.  The opt kernel always uses kCumsumBlockSize; the
  // generic kernel uses kFp8CumsumBlockSize for FP8 or kCumsumBlockSize
  // for BF16.  Take the minimum to guarantee the buffer is large enough.
  constexpr int kEffectiveBlockSize =
      (sizeof(T) == 1 && moe::kFp8CumsumBlockSize < kCumsumBlockSize)
          ? moe::kFp8CumsumBlockSize
          : kCumsumBlockSize;
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
  // 1.Semaphore initialization — only needed when there are multiple blocks
  //   that require inter-block cumsum communication.
  if (cumsum_blocknum > 1) {
    PADDLE_ENFORCE_GPU_SUCCESS(
        cudaMemsetAsync(global_expertwise_block_cumsum.data<int>(),
                        -1,
                        global_expertwise_block_cumsum.numel() * sizeof(int),
                        dev_ctx.stream()));
  }

  // 2.expert_offset initialization
  if (is_buffer_overridden) {
    // Note that when using override, the padding_rows is not needed, and
    // this helper kernel will do expert_indices filling as well.
    // Input: expert_routemap_topk, num_experts, padding_alignment,
    // return_expert_indices
    // Output: expert_offset, expert_offset_end, expert_indices
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
    // plain non-override mode
    int tokens_cumulated = 0;
    std::vector<int> padding_rows;
    // Using CPU vec to calculate the expert_offset and expert_offset_end
    // with extra alloc and memcpy
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
    PADDLE_ENFORCE_GPU_SUCCESS(cudaMemcpyAsync(expert_offset_tensor.data<int>(),
                                               expert_offset,
                                               sizeof(int) * kMaxNumExperts,
                                               cudaMemcpyHostToDevice,
                                               dev_ctx.stream()));

    PADDLE_ENFORCE_GPU_SUCCESS(
        cudaMemcpyAsync(expert_offset_end_tensor.data<int>(),
                        expert_offset_end,
                        sizeof(int) * kMaxNumExperts,
                        cudaMemcpyHostToDevice,
                        dev_ctx.stream()));
    for (int i = 0; i < num_experts; i++) {
      int64_t next_expert_offset =
          i < num_experts - 1 ? expert_offset[i + 1] : tokens_cumulated;
      int64_t invalid_rows =
          next_expert_offset - expert_offset[i] - tokens_per_expert[i];
      int64_t cur_expert_end = expert_offset[i] + tokens_per_expert[i];
      for (int i = 0; i < invalid_rows; ++i) {
        padding_rows.push_back(cur_expert_end + i);
      }
    }
    // padding rows fill-zeros in non-override mode
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
    }  // if (using_ue8m0_scale)
  }    // if (is_buffer_overridden)

  // ====================================================================
  //                    Kernel dispatch
  // ====================================================================
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
      /*If buffer overridden, the expert_indices is computed in pre-process*/
      !is_buffer_overridden && return_expert_indices);
}

}  // namespace phi

PD_REGISTER_KERNEL(moe_permute,
                   GPU,
                   ALL_LAYOUT,
                   phi::MoePermuteKernel,
                   phi::float8_e4m3fn,
                   phi::bfloat16) {}
