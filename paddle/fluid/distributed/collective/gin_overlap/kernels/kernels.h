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

// kernels.h: the launch wrappers of every CUDA kernel and the small structs
// shared with device code. Terms (window, LSA, GIN, credit, ...) are defined
// in the gin_engine.h file header.
//
// Conventions
//   - An offset argument is a byte offset in some window.
//   - numel counts elements (floats), not bytes; numel4 counts float4
//     (= numel / 4).
//   - Host code (gin_engine.cc / release.cu) only calls the launch_* /
//     preload_* functions here and never writes <<<>>> itself.
//   - The thread count of each launch_* is fixed (given in its comment). The
//     block count of the reduce kernels comes from the caller and decides how
//     many SMs the reduce takes.
//   - preload_* loads the module of its kernels at init time, so that the
//     first launch does not land on the critical path of the backward pass.
//     What a first module load does while a kernel of DeepEP / NCCL spins on
//     the GPU in the same process has not been verified.
//   - Every function returns a cudaError_t: the result of cudaGetLastError()
//     (cudaSuccess = success).
#pragma once

#include <cstddef>

#include <cuda_runtime.h>
#include <nccl.h>
#include <nccl_device.h>

namespace gin_overlap {

// Addresses of "the same u64 counter" (an intra-node arrival counter or a
// credit counter) in the host windows of the node's ranks, indexed by the
// node-local rank (lsaRank). A node has at most 16 ranks.
struct LsaSlotPointers {
  unsigned long long* per_rank[16];
};

// Block count from the element count (256 threads per block, at least 1, at
// most 4096)
inline int blocks_for_numel(size_t numel) {
  size_t num_blocks = (numel + 255) / 256;
  if (num_blocks < 1) num_blocks = 1;
  if (num_blocks > 4096) num_blocks = 4096;
  return (int)num_blocks;
}

// ----------------------------------------------------------------
// reduce.cu: reduces
// intra-node reduce: each block of this rank's column = sum of the matching
// blocks of the node's ranks (written back into grad in place).
// <<<num_blocks, 512>>>
cudaError_t launch_reduce_intra_node(float4* grad,
                                     const float4* intra_recv,
                                     size_t numel4,
                                     int ranks_per_node,
                                     int num_nodes,
                                     int rank_in_node,
                                     int num_blocks,
                                     cudaStream_t stream);
// dense shard reduce: shard = sum over nodes of the partial sums * scale (in
// place). <<<num_blocks, 256>>>
cudaError_t launch_reduce_dense_shard(float* shard,
                                      const float* inter_recv,
                                      size_t numel,
                                      int num_nodes,
                                      int node_index,
                                      float scale,
                                      int num_blocks,
                                      cudaStream_t stream);
// expert shard reduce: shard = sum over peers * scale (in place).
// <<<num_blocks, 256>>>
cudaError_t launch_reduce_moe_shard(float* shard,
                                    const float* recv,
                                    size_t numel,
                                    int expert_group_size,
                                    int expert_group_rank,
                                    float scale,
                                    int num_blocks,
                                    cudaStream_t stream);
cudaError_t preload_reduce_kernels();

// ----------------------------------------------------------------
// put.cu: GIN one-sided puts
// dense inter-node put: to the ranks with the same rank_in_node on the other
// num_nodes-1 nodes.
// <<<num_blocks, 256>>>
cudaError_t launch_put_to_node_peers(ncclDevComm dev_comm,
                                     ncclWindow_t recv_window,
                                     ncclWindow_t send_window,
                                     ncclWindow_t signal_window,
                                     size_t send_offset,
                                     size_t inter_recv_offset,
                                     size_t signal_offset,
                                     size_t numel,
                                     int ranks_per_node,
                                     int num_nodes,
                                     int node_index,
                                     int rank_in_node,
                                     int num_gin_contexts,
                                     int num_blocks,
                                     cudaStream_t stream);
// expert put: to the other expert_group_size-1 ranks of the expert group (by
// group rank).
// <<<num_blocks, 256>>>
cudaError_t launch_put_to_expert_peers(ncclDevComm dev_comm,
                                       ncclWindow_t recv_window,
                                       ncclWindow_t send_window,
                                       ncclWindow_t signal_window,
                                       size_t send_offset,
                                       size_t recv_offset,
                                       size_t signal_offset,
                                       size_t numel,
                                       int expert_group_size,
                                       int expert_group_rank,
                                       int num_gin_contexts,
                                       int num_blocks,
                                       cudaStream_t stream);
// 2D-bucket gather: this rank's whole shard in one put to the owner.
// <<<1, 256>>>
cudaError_t launch_put_to_owner(ncclDevComm dev_comm,
                                ncclWindow_t dst_window,
                                size_t dst_offset,
                                ncclWindow_t src_window,
                                size_t src_offset,
                                ncclWindow_t signal_window,
                                size_t signal_offset,
                                size_t numel,
                                int owner,
                                int num_gin_contexts,
                                cudaStream_t stream);
// Flushes every GIN context, i.e. waits until this step's puts have been sent;
// call it before the next step reuses the send window. One block per context.
// <<<num_gin_contexts, 128>>>
cudaError_t launch_gin_flush(ncclDevComm dev_comm,
                             int num_gin_contexts,
                             cudaStream_t stream);
cudaError_t preload_put_kernels();

// ----------------------------------------------------------------
// signal.cu: counters
// Adds 1 to the intra-node credit counters of the other ranks of the node
// (system-scope atomic add). <<<1, 32>>>
cudaError_t launch_signal_intra_node(LsaSlotPointers counter_slots,
                                     int ranks_per_node,
                                     int skip_rank,
                                     cudaStream_t stream);
// Returns a credit: adds 1 to the credit counter of the other num_peers-1
// ranks of the group (GIN signal only, no data). <<<num_blocks, 128>>>
// The group rank of peer k is peer_base + k * peer_stride: a dense credit goes
// to the ranks with the same rank_in_node on the other nodes, with
// peer_base = rank_in_node and peer_stride = ranks_per_node; for the expert
// group peer_base = 0 and peer_stride = 1. k == skip_rank is skipped.
cudaError_t launch_signal_credit(ncclDevComm dev_comm,
                                 ncclWindow_t signal_window,
                                 size_t signal_offset,
                                 int num_peers,
                                 int skip_rank,
                                 int peer_base,
                                 int peer_stride,
                                 int num_gin_contexts,
                                 int num_blocks,
                                 cudaStream_t stream);
// Writes seq into the intra-node arrival counters of the other ranks of the
// node (each sender writes only its own counter).
// <<<1, 32>>>
cudaError_t launch_signal_intra_arrive(LsaSlotPointers arrive_slots,
                                       int ranks_per_node,
                                       int skip_rank,
                                       unsigned long long seq,
                                       cudaStream_t stream);
// Writes one u64 with a release store from one thread (the intra-node arrival
// counter of the 2D gather). <<<1, 32>>>
cudaError_t launch_store_release(unsigned long long* dst,
                                 unsigned long long value,
                                 cudaStream_t stream);
// Gets the device-side addresses of the host windows of the node's ranks into
// out_pointers (device memory). Called once at init, on the default stream.
// <<<1, 1>>>
cudaError_t launch_host_window_lsa_pointers(ncclWindow_t window,
                                            int ranks_per_node,
                                            void** out_pointers);
cudaError_t preload_signal_kernels();

// ---------------------------------------------------------------- scale.cu
// [data, data + numel) *= scale, element by element. <<<num_blocks, 256>>>
cudaError_t launch_scale_range(float* data,
                               size_t numel,
                               float scale,
                               int num_blocks,
                               cudaStream_t stream);
cudaError_t preload_scale_kernels();

}  // namespace gin_overlap
