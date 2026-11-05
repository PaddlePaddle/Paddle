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

#include <algorithm>

#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/common/amp_type_traits.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/kernels/funcs/cub.h"
#include "paddle/phi/kernels/funcs/radix_sort.h"

namespace phi {
namespace funcs {

template <typename T, typename IdT, int WarpSize, int BlockDimY, bool UseLimit>
__global__ void EmbeddingGradDeterministicKernel(T* table,
                                                 const T* output,
                                                 const IdT* ids,
                                                 const int64_t K,
                                                 const int64_t D,
                                                 const int64_t start_idx,
                                                 const int64_t end_idx,
                                                 const bool round_acc_first) {
  using MT = typename MPTypeTrait<T>::Type;
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
            // round_acc_first matches torch's embedding_backward_feature_kernel
            // (`grad_weight += scalar_t(acc)`), which rounds acc to T before
            // adding, i.e. one extra rounding for low precision types.
            MT acc = round_acc_first
                         ? static_cast<MT>(static_cast<T>(my_s[threadIdx.x]))
                         : my_s[threadIdx.x];
            table[table_idx] =
                static_cast<T>(static_cast<MT>(table[table_idx]) + acc);
          }
        }
      }
    }
  }
}

template <typename T, typename IdT>
void LaunchEmbeddingGradDeterministicKernel(const GPUContext& dev_ctx,
                                            const IdT* ids,
                                            const T* d_out,
                                            T* d_table,
                                            int64_t N,
                                            int64_t D,
                                            int64_t K,
                                            int64_t start_idx = -1,
                                            bool round_acc_first = false) {
#ifdef PADDLE_WITH_HIP
  constexpr int kWarpSize = 64;
  constexpr int kBlockDimY = 16;
#else
  constexpr int kWarpSize = 32;
  constexpr int kBlockDimY = 32;
#endif
  dim3 threads(kWarpSize, kBlockDimY);
  dim3 grids(static_cast<int>((D + kWarpSize - 1) / kWarpSize));
  using MT = typename MPTypeTrait<T>::Type;
  constexpr auto kSharedMemSize = sizeof(MT) * kWarpSize * kBlockDimY +
                                  sizeof(IdT) * kWarpSize * kBlockDimY;
  if (start_idx < 0) {
    EmbeddingGradDeterministicKernel<T, IdT, kWarpSize, kBlockDimY, false>
        <<<grids, threads, kSharedMemSize, dev_ctx.stream()>>>(
            d_table, d_out, ids, K, D, -1, -1, round_acc_first);
  } else {
    int64_t end_idx = start_idx + N;
    EmbeddingGradDeterministicKernel<T, IdT, kWarpSize, kBlockDimY, true>
        <<<grids, threads, kSharedMemSize, dev_ctx.stream()>>>(
            d_table, d_out, ids, K, D, start_idx, end_idx, round_acc_first);
  }
}

// The sort based backward below follows torch's embedding_dense_backward_cuda
// (num_indices > 3072): stable sort ids, split rows sharing one id (a segment)
// into partials of kEmbeddingGradRowsPerPartial rows, sum each partial in MT,
// then sum the partials of a segment in MT and round to T only once.
// Accumulation order and grouping are kept identical to torch so the result
// matches bitwise. RadixSortPairs is not available on HIP/ROCm.
#if defined(PADDLE_WITH_CUDA) && !defined(PADDLE_WITH_HIP)
constexpr int64_t kEmbeddingGradRowsPerPartial = 10;

template <typename IdT>
__global__ void EmbeddingGradSortInitKernel(const IdT* ids,
                                            const int64_t K,
                                            int64_t* keys,
                                            int64_t* positions) {
  int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < K) {
    keys[i] = static_cast<int64_t>(ids[i]);
    positions[i] = i;
  }
}

static __global__ void EmbeddingGradPartialsPerSegmentKernel(
    const int64_t* segment_offsets,
    const int64_t* num_segments_ptr,
    const int64_t max_segments,
    const int64_t K,
    int64_t* partials_per_segment) {
  int64_t s = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (s < max_segments) {
    const int64_t num_segments = *num_segments_ptr;
    int64_t size = 0;
    if (s < num_segments) {
      int64_t end = s == num_segments - 1 ? K : segment_offsets[s + 1];
      size = end - segment_offsets[s];
    }
    // Zero the tail so that the exclusive sum of partials_per_segment also
    // gives the total number of partials at its last element.
    partials_per_segment[s] = (size + kEmbeddingGradRowsPerPartial - 1) /
                              kEmbeddingGradRowsPerPartial;
  }
}

static __global__ void EmbeddingGradPartialOffsetKernel(
    const int64_t* segment_offsets,
    const int64_t* partials_per_segment,
    const int64_t* partials_per_segment_offset,
    const int64_t* num_segments_ptr,
    int64_t* partial_offsets) {
  int64_t s = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (s < *num_segments_ptr) {
    int64_t p = partials_per_segment_offset[s];
    for (int64_t i = 0; i < partials_per_segment[s]; ++i) {
      partial_offsets[p + i] =
          segment_offsets[s] + i * kEmbeddingGradRowsPerPartial;
    }
  }
}
template <typename T, typename MT>
__global__ void EmbeddingGradPartialSumKernel(
    const T* output,
    const int64_t* sorted_positions,
    const int64_t K,
    const int64_t D,
    const int64_t* partial_offsets,
    const int64_t* partials_per_segment,
    const int64_t* partials_per_segment_offset,
    const int64_t max_segments,
    MT* partial_sums) {
  const int64_t num_partials = partials_per_segment_offset[max_segments - 1] +
                               partials_per_segment[max_segments - 1];
  int64_t gid = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  int64_t p = gid / D;
  int64_t feature = gid % D;
  if (p >= num_partials) return;
  int64_t begin = partial_offsets[p];
  int64_t end = p == num_partials - 1 ? K : partial_offsets[p + 1];
  MT sum = static_cast<MT>(0);
  for (int64_t i = begin; i < end; ++i) {
    sum += static_cast<MT>(output[sorted_positions[i] * D + feature]);
  }
  partial_sums[p * D + feature] = sum;
}

template <typename T, typename MT>
__global__ void EmbeddingGradSumAndScatterKernel(
    const MT* partial_sums,
    const int64_t* unique_ids,
    const int64_t* partials_per_segment,
    const int64_t* partials_per_segment_offset,
    const int64_t* num_segments_ptr,
    const int64_t D,
    T* table) {
  int64_t gid = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  int64_t s = gid / D;
  int64_t feature = gid % D;
  if (s >= *num_segments_ptr) return;
  int64_t begin = partials_per_segment_offset[s];
  int64_t end = begin + partials_per_segment[s];
  MT sum = static_cast<MT>(0);
  for (int64_t p = begin; p < end; ++p) {
    sum += partial_sums[p * D + feature];
  }
  table[unique_ids[s] * D + feature] = static_cast<T>(sum);
}
// d_table must be zero filled. Extra device memory is about
// (K / kEmbeddingGradRowsPerPartial + min(K, N)) * D * sizeof(MT).
template <typename T, typename IdT>
void LaunchEmbeddingGradSortKernel(const GPUContext& dev_ctx,
                                   const IdT* ids,
                                   const T* d_out,
                                   T* d_table,
                                   int64_t N,
                                   int64_t D,
                                   int64_t K) {
  using MT = typename MPTypeTrait<T>::Type;
  if (K == 0 || D == 0) return;
  auto stream = dev_ctx.stream();
  constexpr int kThreads = 256;
  auto ceil_div = [](int64_t a, int64_t b) {
    return static_cast<int>((a + b - 1) / b);
  };
  auto alloc_int64 = [&](DenseTensor* t, int64_t n) {
    t->Resize({n});
    return dev_ctx.template Alloc<int64_t>(t);
  };
  auto alloc_temp = [&](DenseTensor* t, size_t bytes) {
    t->Resize({static_cast<int64_t>(bytes)});
    return dev_ctx.template Alloc<uint8_t>(t);
  };

  // 1. Stable radix sort of (id, position) pairs, the same as torch.
  DenseTensor keys_in_t, keys_t, positions_in_t, positions_t;
  int64_t* keys_in = alloc_int64(&keys_in_t, K);
  int64_t* keys = alloc_int64(&keys_t, K);
  int64_t* positions_in = alloc_int64(&positions_in_t, K);
  int64_t* positions = alloc_int64(&positions_t, K);
  EmbeddingGradSortInitKernel<IdT>
      <<<ceil_div(K, kThreads), kThreads, 0, stream>>>(
          ids, K, keys_in, positions_in);
  RadixSortPairs<int64_t, int64_t>(
      dev_ctx, keys_in, keys, positions_in, positions, K);
  // 2. Segments: unique ids and their start offsets in sorted order, the same
  // as torch's unique_by_key(sorted_indices, counting_iterator(0)).
  const int64_t max_segments = std::min(K, N);
  DenseTensor unique_ids_t, segment_offsets_t, num_segments_t, unique_temp_t;
  int64_t* unique_ids = alloc_int64(&unique_ids_t, max_segments);
  int64_t* segment_offsets = alloc_int64(&segment_offsets_t, max_segments);
  int64_t* num_segments = alloc_int64(&num_segments_t, 1);
  cub::CountingInputIterator<int64_t> counting_iter(0);
  size_t temp_bytes = 0;
  cub::DeviceSelect::UniqueByKey(nullptr,
                                 temp_bytes,
                                 keys,
                                 counting_iter,
                                 unique_ids,
                                 segment_offsets,
                                 num_segments,
                                 K,
                                 stream);
  cub::DeviceSelect::UniqueByKey(alloc_temp(&unique_temp_t, temp_bytes),
                                 temp_bytes,
                                 keys,
                                 counting_iter,
                                 unique_ids,
                                 segment_offsets,
                                 num_segments,
                                 K,
                                 stream);

  // 3. Partials of at most kEmbeddingGradRowsPerPartial rows per segment.
  DenseTensor partials_per_segment_t, partials_per_segment_offset_t,
      scan_temp_t;
  int64_t* partials_per_segment =
      alloc_int64(&partials_per_segment_t, max_segments);
  int64_t* partials_per_segment_offset =
      alloc_int64(&partials_per_segment_offset_t, max_segments);
  EmbeddingGradPartialsPerSegmentKernel<<<ceil_div(max_segments, kThreads),
                                          kThreads,
                                          0,
                                          stream>>>(
      segment_offsets, num_segments, max_segments, K, partials_per_segment);
  temp_bytes = 0;
  cub::DeviceScan::ExclusiveSum(nullptr,
                                temp_bytes,
                                partials_per_segment,
                                partials_per_segment_offset,
                                max_segments,
                                stream);
  cub::DeviceScan::ExclusiveSum(alloc_temp(&scan_temp_t, temp_bytes),
                                temp_bytes,
                                partials_per_segment,
                                partials_per_segment_offset,
                                max_segments,
                                stream);
  const int64_t max_partials = K / kEmbeddingGradRowsPerPartial + max_segments;
  DenseTensor partial_offsets_t;
  int64_t* partial_offsets = alloc_int64(&partial_offsets_t, max_partials);
  EmbeddingGradPartialOffsetKernel<<<ceil_div(max_segments, kThreads),
                                     kThreads,
                                     0,
                                     stream>>>(segment_offsets,
                                               partials_per_segment,
                                               partials_per_segment_offset,
                                               num_segments,
                                               partial_offsets);

  // 4. Sum every partial, then the partials of every segment, both in MT.
  DenseTensor partial_sums_t;
  partial_sums_t.Resize({max_partials, D});
  MT* partial_sums = dev_ctx.template Alloc<MT>(&partial_sums_t);
  EmbeddingGradPartialSumKernel<T, MT>
      <<<ceil_div(max_partials * D, kThreads), kThreads, 0, stream>>>(
          d_out,
          positions,
          K,
          D,
          partial_offsets,
          partials_per_segment,
          partials_per_segment_offset,
          max_segments,
          partial_sums);
  EmbeddingGradSumAndScatterKernel<T, MT>
      <<<ceil_div(max_segments * D, kThreads), kThreads, 0, stream>>>(
          partial_sums,
          unique_ids,
          partials_per_segment,
          partials_per_segment_offset,
          num_segments,
          D,
          d_table);
}
#endif

}  // namespace funcs
}  // namespace phi
