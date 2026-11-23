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

// sum.cu: the sum kernel declared in sum.h.
//
// Instantiations that match the kernels of gin_overlap/kernels/reduce.cu:
//   k_reduce_intra_node  SumKernel<false, true, true>, 512 threads,
//                        through LaunchIntraNodeSum with
//                        numel = numel4 * 4.
//   k_reduce_dense_shard SumKernel<true, false, false>, 256 threads,
//                        through LaunchFinalSum with count = num_nodes,
//                        self_index = node_index.
//   k_reduce_moe_shard   SumKernel<true, false, false>, 256 threads,
//                        through LaunchFinalSum with
//                        count = expert_group_size,
//                        self_index = expert_group_rank.
//
// Constraints
//   - Summation order: ring order, starting at self_index + 1 and going once
//     around, from acc = 0.
//   - kScale = true: each term is __fadd_rn(acc, __fmul_rn(value, scale)).
//     kScale = false: each term is acc += value.
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
__device__ __forceinline__ void Accumulate(float* acc,
                                           float value,
                                           float scale) {
  if constexpr (kScale) {
    *acc = __fadd_rn(*acc, __fmul_rn(value, scale));
  } else {
    *acc += value;
  }
}

template <bool kScale>
__device__ __forceinline__ void Accumulate4(float4* acc,
                                            const float4& value,
                                            float scale) {
  Accumulate<kScale>(&acc->x, value.x, scale);
  Accumulate<kScale>(&acc->y, value.y, scale);
  Accumulate<kScale>(&acc->z, value.z, scale);
  Accumulate<kScale>(&acc->w, value.w, scale);
}

// Row of recv that holds source src.
template <bool kSkipSelf>
__device__ __forceinline__ size_t RecvRow(int src, int self_index) {
  if constexpr (kSkipSelf) {
    return static_cast<size_t>(src - (src > self_index));
  } else {
    return static_cast<size_t>(src);
  }
}

// Whether the float4 path applies: numel and every stride a multiple of 4,
// own and recv 16-byte aligned.
template <bool kMultiSegment>
__device__ __forceinline__ bool CanUseFloat4(const float* own,
                                             const float* recv,
                                             const SumLayout& layout) {
  size_t counts = layout.numel | layout.recv_row_stride;
  if constexpr (kMultiSegment) {
    counts |= layout.own_segment_stride | layout.recv_segment_stride;
  }
  const uintptr_t addresses =
      reinterpret_cast<uintptr_t>(own) | reinterpret_cast<uintptr_t>(recv);
  return (counts & 3) == 0 && (addresses & 15) == 0;
}

template <bool kScale, bool kSkipSelf, bool kMultiSegment>
__global__ void SumKernel(float* own,
                          const float* __restrict__ recv,
                          SumLayout layout,
                          float scale) {
  const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
  const size_t thread_first =
      static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int count = layout.count;
  const int self_index = layout.self_index;
  const size_t num_segments =
      kMultiSegment ? static_cast<size_t>(layout.num_segments) : 1;

  if (CanUseFloat4<kMultiSegment>(own, recv, layout)) {
    const size_t numel4 = layout.numel / 4;
    const size_t own_segment_stride4 = layout.own_segment_stride / 4;
    const size_t recv_row_stride4 = layout.recv_row_stride / 4;
    const size_t recv_segment_stride4 = layout.recv_segment_stride / 4;
    float4* own4 = reinterpret_cast<float4*>(own);
    const float4* recv4 = reinterpret_cast<const float4*>(recv);
    const size_t total = num_segments * numel4;
    for (size_t i = thread_first; i < total; i += stride) {
      size_t segment = 0;
      size_t elem = i;
      if constexpr (kMultiSegment) {
        segment = i / numel4;
        elem = i - segment * numel4;
      }
      float4* own_elem = own4 + segment * own_segment_stride4 + elem;
      const float4* recv_elem = recv4 + segment * recv_segment_stride4 + elem;
      float4 acc = make_float4(0.f, 0.f, 0.f, 0.f);
      for (int batch = 0; batch < count; batch += kSourcesPerBatch) {
        float4 vals[kSourcesPerBatch];
#pragma unroll
        for (int j = 0; j < kSourcesPerBatch; ++j) {
          const int k = batch + j;
          if (k < count) {
            const int src = (self_index + 1 + k) % count;
            vals[j] = *((src == self_index)
                            ? own_elem
                            : recv_elem + RecvRow<kSkipSelf>(src, self_index) *
                                              recv_row_stride4);
          }
        }
#pragma unroll
        for (int j = 0; j < kSourcesPerBatch; ++j) {
          if (batch + j < count) Accumulate4<kScale>(&acc, vals[j], scale);
        }
      }
      *own_elem = acc;
    }
    return;
  }

  const size_t numel = layout.numel;
  const size_t total = num_segments * numel;
  for (size_t i = thread_first; i < total; i += stride) {
    size_t segment = 0;
    size_t elem = i;
    if constexpr (kMultiSegment) {
      segment = i / numel;
      elem = i - segment * numel;
    }
    float* own_elem = own + segment * layout.own_segment_stride + elem;
    const float* recv_elem = recv + segment * layout.recv_segment_stride + elem;
    float acc = 0.f;
    for (int k = 0; k < count; ++k) {
      const int src = (self_index + 1 + k) % count;
      const float value = (src == self_index)
                              ? *own_elem
                              : recv_elem[RecvRow<kSkipSelf>(src, self_index) *
                                          layout.recv_row_stride];
      Accumulate<kScale>(&acc, value, scale);
    }
    *own_elem = acc;
  }
}

}  // namespace

template <bool kScale, bool kSkipSelf, bool kMultiSegment>
cudaError_t LaunchSum(float* own,
                      const float* recv,
                      const SumLayout& layout,
                      float scale,
                      int num_blocks,
                      int num_threads,
                      cudaStream_t stream) {
  if (layout.count < 1 || layout.self_index < 0 ||
      layout.self_index >= layout.count || layout.num_segments < 1 ||
      (!kMultiSegment && layout.num_segments != 1)) {
    return cudaErrorInvalidValue;
  }
  SumKernel<kScale, kSkipSelf, kMultiSegment>
      <<<num_blocks, num_threads, 0, stream>>>(own, recv, layout, scale);
  return cudaGetLastError();
}

template cudaError_t LaunchSum<false, false, false>(
    float*, const float*, const SumLayout&, float, int, int, cudaStream_t);
template cudaError_t LaunchSum<false, false, true>(
    float*, const float*, const SumLayout&, float, int, int, cudaStream_t);
template cudaError_t LaunchSum<false, true, false>(
    float*, const float*, const SumLayout&, float, int, int, cudaStream_t);
template cudaError_t LaunchSum<false, true, true>(
    float*, const float*, const SumLayout&, float, int, int, cudaStream_t);
template cudaError_t LaunchSum<true, false, false>(
    float*, const float*, const SumLayout&, float, int, int, cudaStream_t);
template cudaError_t LaunchSum<true, false, true>(
    float*, const float*, const SumLayout&, float, int, int, cudaStream_t);
template cudaError_t LaunchSum<true, true, false>(
    float*, const float*, const SumLayout&, float, int, int, cudaStream_t);
template cudaError_t LaunchSum<true, true, true>(
    float*, const float*, const SumLayout&, float, int, int, cudaStream_t);

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
  layout.count = ranks_per_node;
  layout.self_index = rank_in_node;
  layout.num_segments = num_nodes;
  layout.numel = numel;
  layout.own_segment_stride = static_cast<size_t>(ranks_per_node) * numel;
  layout.recv_row_stride = static_cast<size_t>(num_nodes) * numel;
  layout.recv_segment_stride = numel;
  float* own = grad + static_cast<size_t>(rank_in_node) * numel;
  return LaunchSum<false, true, true>(
      own, intra_recv, layout, 1.f, num_blocks, num_threads, stream);
}

cudaError_t LaunchFinalSum(float* shard,
                           const float* recv,
                           size_t numel,
                           int count,
                           int self_index,
                           float scale,
                           int num_blocks,
                           cudaStream_t stream,
                           int num_threads) {
  SumLayout layout;
  layout.count = count;
  layout.self_index = self_index;
  layout.num_segments = 1;
  layout.numel = numel;
  layout.own_segment_stride = 0;
  layout.recv_row_stride = numel;
  layout.recv_segment_stride = 0;
  return LaunchSum<true, false, false>(
      shard, recv, layout, scale, num_blocks, num_threads, stream);
}

}  // namespace distributed
}  // namespace paddle
