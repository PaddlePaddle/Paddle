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

#pragma once

#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/common/amp_type_traits.h"

namespace phi {
namespace funcs {

template <typename T, typename IdT, int WarpSize, int BlockDimY, bool UseLimit>
__global__ void EmbeddingGradDeterministicKernel(T* table,
                                                 const T* output,
                                                 const IdT* ids,
                                                 const int64_t K,
                                                 const int64_t D,
                                                 const int64_t start_idx,
                                                 const int64_t end_idx) {
  using MT = typename dtype::MPTypeTrait<T>::Type;
  constexpr int64_t kInvalidId = -1;
  extern __shared__ char buf[];
  MT* smem = reinterpret_cast<MT*>(buf);
  MT* my_s = smem + WarpSize * threadIdx.y;
  IdT* indices_batch =
      reinterpret_cast<IdT*>(buf + sizeof(MT) * WarpSize * BlockDimY);

  const int stride = static_cast<int>(D);

  const int feature = threadIdx.x + blockIdx.x * WarpSize;

  // To ensure determinism. If any other warps pulled grad data targeting
  // dst_row, we elect the first warp in each matching group as the leader.
  // Each leader warp serializes the accumulates targeting dst_row in shared
  // memory, then adding the accumulated buffer to dst_row in table.
  for (int batch_start = 0; batch_start < K;
       batch_start += WarpSize * BlockDimY) {
    int tid = threadIdx.x + threadIdx.y * WarpSize;
    if (batch_start + tid < K) {
      int64_t cur_id = static_cast<int64_t>(ids[batch_start + tid]);
      if (UseLimit) {
        if (cur_id >= start_idx && cur_id < end_idx) {
          cur_id -= start_idx;
        } else {
          cur_id = kInvalidId;
        }
      }
      indices_batch[tid] = cur_id;
    }

    int batch_end =
        min(static_cast<int64_t>(batch_start + WarpSize * BlockDimY), K);

    // Loop over the batch of <= 1024 loaded indices in chunks of BLOCKDIMY
    for (int chunk_start = batch_start; chunk_start < batch_end;
         chunk_start += BlockDimY) {
      // This sync makes sure that indices_batch is ready and match-group
      // leaders are done with their accumulates before other warps start
      // loading again.
      __syncthreads();

      int n_this_chunk = min(batch_end - chunk_start, BlockDimY);

      int64_t src_row = static_cast<int64_t>(chunk_start + threadIdx.y);
      int64_t dst_row = indices_batch[src_row - batch_start];
      if (src_row < K && feature < stride) {
        if (UseLimit && dst_row == kInvalidId) {
          my_s[threadIdx.x] = static_cast<MT>(0);
        } else {
          my_s[threadIdx.x] = static_cast<MT>(output[src_row * D + feature]);
        }
      }

      __syncthreads();

      if (src_row < K) {
        int match_found_this_thread = 0;
        if (threadIdx.x < n_this_chunk &&
            (!UseLimit || dst_row != kInvalidId)) {
          match_found_this_thread =
              (dst_row ==
               indices_batch[chunk_start - batch_start + threadIdx.x]);
        }
#ifdef PADDLE_WITH_HIP
        unsigned long long int matchmask =      // NOLINT
            __ballot(match_found_this_thread);  // NOLINT
        int first_remaining_peer = __ffsll(matchmask) - 1;
#else
        // If and only if match_found_this_thread of the Nth thread is non-zero,
        // set the Nth bit of matchmask to 1.
        unsigned int matchmask =
            __ballot_sync(0xffffffff, match_found_this_thread);
        // Find the position of the first bit set to 1 in matchmask.
        int first_remaining_peer = __ffs(matchmask) - 1;
#endif

        // select lowest-indexed warp as the leader
        if (threadIdx.y == first_remaining_peer) {
          // Set the first bit 1 in matchmask to 0.
          matchmask ^= (1 << first_remaining_peer);
          while (matchmask) {
#ifdef PADDLE_WITH_HIP
            first_remaining_peer = __ffsll(matchmask) - 1;
#else
            first_remaining_peer = __ffs(matchmask) - 1;
#endif
            my_s[threadIdx.x] +=
                smem[threadIdx.x + WarpSize * first_remaining_peer];
            matchmask ^= (1 << first_remaining_peer);
          }
          if (feature < stride && (!UseLimit || dst_row != kInvalidId)) {
            auto table_idx = dst_row * D + feature;
            table[table_idx] = static_cast<T>(
                static_cast<MT>(table[table_idx]) + my_s[threadIdx.x]);
          }
        }
      }
    }
  }
}

template <typename T, typename IdT>
void LaunchEmbeddingGradDeterministicKernel(const GPUContext& ctx,
                                            const IdT* ids,
                                            const T* d_out,
                                            T* d_table,
                                            int64_t N,
                                            int64_t D,
                                            int64_t K,
                                            int64_t start_idx = -1) {
#ifdef PADDLE_WITH_HIP
  constexpr int kWarpSize = 64;
  constexpr int kBlockDimY = 16;
#else
  constexpr int kWarpSize = 32;
  constexpr int kBlockDimY = 32;
#endif
  dim3 threads(kWarpSize, kBlockDimY);
  dim3 grids(static_cast<int>((D + kWarpSize - 1) / kWarpSize));
  using MT = typename dtype::MPTypeTrait<T>::Type;
  constexpr auto kSharedMemSize = sizeof(MT) * kWarpSize * kBlockDimY +
                                  sizeof(IdT) * kWarpSize * kBlockDimY;
  if (start_idx < 0) {
    EmbeddingGradDeterministicKernel<T, IdT, kWarpSize, kBlockDimY, false>
        <<<grids, threads, kSharedMemSize, ctx.stream()>>>(
            d_table, d_out, ids, K, D, -1, -1);
  } else {
    int64_t end_idx = start_idx + N;
    EmbeddingGradDeterministicKernel<T, IdT, kWarpSize, kBlockDimY, true>
        <<<grids, threads, kSharedMemSize, ctx.stream()>>>(
            d_table, d_out, ids, K, D, start_idx, end_idx);
  }
}

}  // namespace funcs
}  // namespace phi
