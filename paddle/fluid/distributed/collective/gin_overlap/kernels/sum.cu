// Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
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

// sum.cu: the sum kernel declared in sum.h. It has two uses:
//   - intra-node sum: SumKernel<false, true, true>, 512 threads, through
//     LaunchIntraNodeSum; the sources are the ranks of the node.
//   - final sum: SumKernel<true, false, false>, 256 threads, through
//     LaunchFinalSum; the sources are the nodes (dense bucket,
//     self_index = node index) or the ranks of the expert group (expert
//     bucket, self_index = rank in the group).
//
// Constraints
//   - Summation order: ring order, starting at self_index + 1 and going once
//     around, from sum = 0.
//   - kScale = true: each term is __fadd_rn(sum, __fmul_rn(value, scale)).
//     kScale = false: each term is sum += value.
//   - The float4 path and the scalar path give the same bits: the float4 path
//     loads up to 8 sources first and then adds them in source order, one
//     component at a time.
#include "paddle/fluid/distributed/collective/gin_overlap/kernels/sum.h"

#include <cstdint>

namespace paddle {
namespace distributed {
namespace {

constexpr int kSourcesPerBatch = 8;

template <bool kScale>
__device__ __forceinline__ void Accumulate(float* sum,
                                           float value,
                                           float scale) {
  if constexpr (kScale) {
    *sum = __fadd_rn(*sum, __fmul_rn(value, scale));
  } else {
    *sum += value;
  }
}

template <bool kScale>
__device__ __forceinline__ void Accumulate4(float4* sum,
                                            const float4& value,
                                            float scale) {
  Accumulate<kScale>(&sum->x, value.x, scale);
  Accumulate<kScale>(&sum->y, value.y, scale);
  Accumulate<kScale>(&sum->z, value.z, scale);
  Accumulate<kScale>(&sum->w, value.w, scale);
}

// Row of recv that holds `source`.
template <bool kSkipSelf>
__device__ __forceinline__ size_t RecvRow(int source, int self_index) {
  if constexpr (kSkipSelf) {
    return static_cast<size_t>(source - (source > self_index));
  } else {
    return static_cast<size_t>(source);
  }
}

// Whether the float4 path applies: numel and every stride that is read are
// multiples of 4, and self_values and recv are 16-byte aligned.
// size_bits / address_bits are the OR of those sizes / addresses, so their low
// bits are 0 only when they are 0 in every one.
template <bool kMultiShard>
__device__ __forceinline__ bool CanUseFloat4(const float* self_values,
                                             const float* recv,
                                             const SumLayout& layout) {
  size_t size_bits = layout.numel | layout.recv_row_stride;
  if constexpr (kMultiShard) {
    size_bits |= layout.self_shard_stride | layout.recv_shard_stride;
  }
  const uintptr_t address_bits = reinterpret_cast<uintptr_t>(self_values) |
                                 reinterpret_cast<uintptr_t>(recv);
  return (size_bits & 3) == 0 && (address_bits & 15) == 0;
}

template <bool kScale, bool kSkipSelf, bool kMultiShard>
__global__ void SumKernel(float* self_values,
                          const float* __restrict__ recv,
                          SumLayout layout,
                          float scale) {
  const size_t grid_stride = static_cast<size_t>(gridDim.x) * blockDim.x;
  const size_t first_index =
      static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int num_sources = layout.num_sources;
  const int self_index = layout.self_index;
  const size_t num_shards =
      kMultiShard ? static_cast<size_t>(layout.num_shards) : 1;

  if (CanUseFloat4<kMultiShard>(self_values, recv, layout)) {
    // in float4 units
    const size_t numel4 = layout.numel / 4;
    const size_t self_shard_stride4 = layout.self_shard_stride / 4;
    const size_t recv_row_stride4 = layout.recv_row_stride / 4;
    const size_t recv_shard_stride4 = layout.recv_shard_stride / 4;
    float4* self_values4 = reinterpret_cast<float4*>(self_values);
    const float4* recv4 = reinterpret_cast<const float4*>(recv);
    const size_t total_numel4 = num_shards * numel4;
    for (size_t index = first_index; index < total_numel4;
         index += grid_stride) {
      size_t shard = 0;
      size_t element = index;
      if constexpr (kMultiShard) {
        shard = index / numel4;
        element = index - shard * numel4;
      }
      float4* self_element =
          self_values4 + shard * self_shard_stride4 + element;
      const float4* recv_element = recv4 + shard * recv_shard_stride4 + element;
      float4 sum = make_float4(0.f, 0.f, 0.f, 0.f);
      for (int batch_start = 0; batch_start < num_sources;
           batch_start += kSourcesPerBatch) {
        float4 batch_values[kSourcesPerBatch];
#pragma unroll
        for (int batch_offset = 0; batch_offset < kSourcesPerBatch;
             ++batch_offset) {
          const int ring_step = batch_start + batch_offset;
          if (ring_step < num_sources) {
            const int source = (self_index + 1 + ring_step) % num_sources;
            batch_values[batch_offset] =
                *((source == self_index)
                      ? self_element
                      : recv_element + RecvRow<kSkipSelf>(source, self_index) *
                                           recv_row_stride4);
          }
        }
#pragma unroll
        for (int batch_offset = 0; batch_offset < kSourcesPerBatch;
             ++batch_offset) {
          if (batch_start + batch_offset < num_sources)
            Accumulate4<kScale>(&sum, batch_values[batch_offset], scale);
        }
      }
      *self_element = sum;
    }
    return;
  }

  const size_t numel = layout.numel;
  const size_t total_numel = num_shards * numel;
  for (size_t index = first_index; index < total_numel; index += grid_stride) {
    size_t shard = 0;
    size_t element = index;
    if constexpr (kMultiShard) {
      shard = index / numel;
      element = index - shard * numel;
    }
    float* self_element =
        self_values + shard * layout.self_shard_stride + element;
    const float* recv_element =
        recv + shard * layout.recv_shard_stride + element;
    float sum = 0.f;
    for (int ring_step = 0; ring_step < num_sources; ++ring_step) {
      const int source = (self_index + 1 + ring_step) % num_sources;
      const float value =
          (source == self_index)
              ? *self_element
              : recv_element[RecvRow<kSkipSelf>(source, self_index) *
                             layout.recv_row_stride];
      Accumulate<kScale>(&sum, value, scale);
    }
    *self_element = sum;
  }
}

}  // namespace

template <bool kScale, bool kSkipSelf, bool kMultiShard>
cudaError_t LaunchSum(float* self_values,
                      const float* recv,
                      const SumLayout& layout,
                      float scale,
                      int num_blocks,
                      cudaStream_t stream,
                      int num_threads) {
  if (layout.num_sources < 1 || layout.self_index < 0 ||
      layout.self_index >= layout.num_sources || layout.num_shards < 1 ||
      (!kMultiShard && layout.num_shards != 1)) {
    return cudaErrorInvalidValue;
  }
  SumKernel<kScale, kSkipSelf, kMultiShard>
      <<<num_blocks, num_threads, 0, stream>>>(
          self_values, recv, layout, scale);
  return cudaGetLastError();
}

template cudaError_t LaunchSum<false, true, true>(
    float*, const float*, const SumLayout&, float, int, cudaStream_t, int);
template cudaError_t LaunchSum<true, false, false>(
    float*, const float*, const SumLayout&, float, int, cudaStream_t, int);

cudaError_t LaunchIntraNodeSum(float* grad,
                               const float* intra_recv,
                               size_t numel,
                               int ranks_per_node,
                               int num_nodes,
                               int rank_in_node,
                               int num_blocks,
                               cudaStream_t stream,
                               int num_threads) {
  SumLayout layout;
  layout.num_sources = ranks_per_node;
  layout.self_index = rank_in_node;
  layout.num_shards = num_nodes;
  layout.numel = numel;
  layout.self_shard_stride = static_cast<size_t>(ranks_per_node) * numel;
  layout.recv_row_stride = static_cast<size_t>(num_nodes) * numel;
  layout.recv_shard_stride = numel;
  float* self_values = grad + static_cast<size_t>(rank_in_node) * numel;
  return LaunchSum<false, true, true>(
      self_values, intra_recv, layout, 1.f, num_blocks, stream, num_threads);
}

cudaError_t LaunchFinalSum(float* shard,
                           const float* recv,
                           size_t numel,
                           int num_sources,
                           int self_index,
                           float scale,
                           int num_blocks,
                           cudaStream_t stream,
                           int num_threads) {
  SumLayout layout;
  layout.num_sources = num_sources;
  layout.self_index = self_index;
  layout.num_shards = 1;
  layout.numel = numel;
  layout.self_shard_stride = 0;
  layout.recv_row_stride = numel;
  layout.recv_shard_stride = 0;
  return LaunchSum<true, false, false>(
      shard, recv, layout, scale, num_blocks, stream, num_threads);
}

}  // namespace distributed
}  // namespace paddle
