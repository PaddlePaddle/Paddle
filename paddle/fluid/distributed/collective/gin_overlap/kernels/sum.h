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

// sum.h: in-place sum kernel on one GPU for gradient buckets.
// own = sum over the count sources, taken in ring order starting at
// self_index + 1 and going once around. Source self_index is own itself;
// every other source is a row of recv.
//
// Conventions
//   - Element counts and strides count floats, not bytes.
//   - Every launch function returns cudaGetLastError() after the launch, or
//     cudaErrorInvalidValue without launching when the layout is out of range
//     (see SumLayout).
//   - Block and thread counts do not change any result bit: they only decide
//     which elements a thread handles.
#pragma once

#include <cuda_runtime.h>

#include <cstddef>

namespace paddle {
namespace distributed {

// Memory layout of one sum.
//
// The data is num_shards shards of numel elements each; a shard is one
// slice of this rank's column per node. Element e of shard s is at
//   own:                  own + s * own_shard_stride + e
//   recv, source src:     recv + RecvRow(src) * recv_row_stride
//                              + s * recv_shard_stride + e
// RecvRow(src) is src when recv has count rows (the row of self_index is not
// read), or src - (src > self_index) when recv has count - 1 rows that skip
// self_index (template parameter kSkipSelf). The shard strides are not read
// when kMultiShard is false.
struct SumLayout {
  int count;       // number of sources, this rank included (>= 1)
  int self_index;  // index of this rank among the sources, in [0, count)
  int num_shards;  // >= 1; must be 1 when kMultiShard is false
  size_t numel;    // elements per shard
  size_t own_shard_stride;
  size_t recv_row_stride;
  size_t recv_shard_stride;
};

// Generic sum. <<<num_blocks, num_threads>>>
//   kScale        false: acc += value.
//                 true:  acc = __fadd_rn(acc, __fmul_rn(value, scale)).
//   kSkipSelf     recv has count - 1 rows and no row for self_index (true),
//                 or count rows (false).
//   kMultiShard   layout.num_shards may be greater than 1.
// acc starts at 0. A float4 path is taken when numel and every stride that is
// read are multiples of 4 and own and recv are 16-byte aligned; otherwise a
// scalar path is taken. Both paths use the same summation order and float
// operations. scale is ignored when kScale is false. Only two
// instantiations exist: <false, true, true> (LaunchIntraNodeSum) and
// <true, false, false> (LaunchFinalSum).
template <bool kScale, bool kSkipSelf, bool kMultiShard>
cudaError_t LaunchSum(float* own,
                      const float* recv,
                      const SumLayout& layout,
                      float scale,
                      int num_blocks,
                      int num_threads,
                      cudaStream_t stream);

// Sum over the ranks of one node, for a bucket with one block per node.
// LaunchSum<false, true, true>.
//   grad:       [num_nodes][ranks_per_node][numel]. Block (node, rank_in_node)
//               of every node is summed in place.
//   intra_recv: [ranks_per_node - 1][num_nodes][numel], the rows pushed by the
//               other ranks of the node; rank l is in row
//               l - (l > rank_in_node).
cudaError_t LaunchIntraNodeSum(float* grad,
                               const float* intra_recv,
                               size_t numel,
                               int ranks_per_node,
                               int num_nodes,
                               int rank_in_node,
                               int num_blocks,
                               cudaStream_t stream,
                               int num_threads = 512);

// Final sum of one shard with a coefficient, in place:
//   shard = sum over k of (k == self_index ? shard : recv[k]) * scale.
// LaunchSum<true, false, false>.
//   recv: [count][numel]; row self_index is not read.
cudaError_t LaunchFinalSum(float* shard,
                           const float* recv,
                           size_t numel,
                           int count,
                           int self_index,
                           float scale,
                           int num_blocks,
                           cudaStream_t stream,
                           int num_threads = 256);

}  // namespace distributed
}  // namespace paddle
