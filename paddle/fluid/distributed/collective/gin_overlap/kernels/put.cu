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

// put.cu: the four GIN one-sided put kernels: the dense inter-node put, the
// expert put, the 2D-bucket gather put to the owner, and the end-of-step
// flush.
//   - Every put carries an ncclGin_VASignalInc: the data lands in the peer's
//     recv / send window, and the signal counter in the peer's host window
//     goes up by 1 (NIC atomic add). Once the peer's CPU sees the counter
//     reach its target, the data has arrived.
//   - One put per peer, no chunking.
#include "paddle/fluid/distributed/collective/gin_overlap/kernels/kernels.h"

namespace gin_overlap {
namespace {

// dense inter-node put: block peer_node of this rank's column (a partial sum)
// -> row node_index of the inter-node segment of the rank with the same
// rank_in_node on node peer_node. One block per peer; with fewer blocks than
// peers a block sends several puts in turn.
__global__ void k_put_to_node_peers(ncclDevComm dev_comm,
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
                                    int num_gin_contexts) {
  ncclGin gin(
      dev_comm,
      static_cast<int>(blockIdx.x % static_cast<unsigned>(num_gin_contexts)));
  const ncclTeam team = ncclTeamWorld(dev_comm);
  for (int peer_index = blockIdx.x; peer_index < num_nodes - 1;
       peer_index += gridDim.x) {
    const int peer_node = (node_index + 1 + peer_index) % num_nodes;
    gin.put(team,
            peer_node * ranks_per_node + rank_in_node,
            recv_window,
            inter_recv_offset +
                static_cast<size_t>(node_index) * numel * sizeof(float),
            send_window,
            send_offset + (static_cast<size_t>(peer_node) * ranks_per_node +
                           rank_in_node) *
                              numel * sizeof(float),
            numel * sizeof(float),
            ncclGin_VASignalInc{signal_window, signal_offset},
            ncclGin_None{},
            ncclCoopCta());
  }
}

// expert put: this rank's block `peer` -> row expert_group_rank of the expert
// segment of the rank with group rank `peer`. Blocks are assigned as in
// k_put_to_node_peers
__global__ void k_put_to_expert_peers(ncclDevComm dev_comm,
                                      ncclWindow_t recv_window,
                                      ncclWindow_t send_window,
                                      ncclWindow_t signal_window,
                                      size_t send_offset,
                                      size_t recv_offset,
                                      size_t signal_offset,
                                      size_t numel,
                                      int expert_group_size,
                                      int expert_group_rank,
                                      int num_gin_contexts) {
  ncclGin gin(
      dev_comm,
      static_cast<int>(blockIdx.x % static_cast<unsigned>(num_gin_contexts)));
  const ncclTeam team = ncclTeamWorld(dev_comm);
  for (int peer_index = blockIdx.x; peer_index < expert_group_size - 1;
       peer_index += gridDim.x) {
    const int peer = (expert_group_rank + 1 + peer_index) % expert_group_size;
    gin.put(team,
            peer,
            recv_window,
            recv_offset +
                static_cast<size_t>(expert_group_rank) * numel * sizeof(float),
            send_window,
            send_offset + static_cast<size_t>(peer) * numel * sizeof(float),
            numel * sizeof(float),
            ncclGin_VASignalInc{signal_window, signal_offset},
            ncclGin_None{},
            ncclCoopCta());
  }
}

// 2D-bucket gather (Muon): this rank's whole shard in one put to the owner,
// adding 1 to the arrival counter in the owner's host window. 1 block
__global__ void k_put_to_owner(ncclDevComm dev_comm,
                               ncclWindow_t dst_window,
                               size_t dst_offset,
                               ncclWindow_t src_window,
                               size_t src_offset,
                               ncclWindow_t signal_window,
                               size_t signal_offset,
                               size_t numel,
                               int owner,
                               int num_gin_contexts) {
  ncclGin gin(
      dev_comm,
      static_cast<int>(blockIdx.x % static_cast<unsigned>(num_gin_contexts)));
  const ncclTeam team = ncclTeamWorld(dev_comm);
  gin.put(team,
          owner,
          dst_window,
          dst_offset,
          src_window,
          src_offset,
          numel * sizeof(float),
          ncclGin_VASignalInc{signal_window, signal_offset},
          ncclGin_None{},
          ncclCoopCta());
}

// flush: one block per GIN context. Call it before the next step reuses the
// send window
__global__ void k_gin_flush(ncclDevComm dev_comm, int num_gin_contexts) {
  if (blockIdx.x >= static_cast<unsigned>(num_gin_contexts)) return;
  ncclGin gin(dev_comm, static_cast<int>(blockIdx.x));
  gin.flush(ncclCoopCta());
}

}  // namespace

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
                                     cudaStream_t stream) {
  k_put_to_node_peers<<<num_blocks, 256, 0, stream>>>(dev_comm,
                                                      recv_window,
                                                      send_window,
                                                      signal_window,
                                                      send_offset,
                                                      inter_recv_offset,
                                                      signal_offset,
                                                      numel,
                                                      ranks_per_node,
                                                      num_nodes,
                                                      node_index,
                                                      rank_in_node,
                                                      num_gin_contexts);
  return cudaGetLastError();
}

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
                                       cudaStream_t stream) {
  k_put_to_expert_peers<<<num_blocks, 256, 0, stream>>>(dev_comm,
                                                        recv_window,
                                                        send_window,
                                                        signal_window,
                                                        send_offset,
                                                        recv_offset,
                                                        signal_offset,
                                                        numel,
                                                        expert_group_size,
                                                        expert_group_rank,
                                                        num_gin_contexts);
  return cudaGetLastError();
}

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
                                cudaStream_t stream) {
  k_put_to_owner<<<1, 256, 0, stream>>>(dev_comm,
                                        dst_window,
                                        dst_offset,
                                        src_window,
                                        src_offset,
                                        signal_window,
                                        signal_offset,
                                        numel,
                                        owner,
                                        num_gin_contexts);
  return cudaGetLastError();
}

cudaError_t launch_gin_flush(ncclDevComm dev_comm,
                             int num_gin_contexts,
                             cudaStream_t stream) {
  k_gin_flush<<<num_gin_contexts, 128, 0, stream>>>(dev_comm, num_gin_contexts);
  return cudaGetLastError();
}

cudaError_t preload_put_kernels() {
  cudaFuncAttributes fa;
  cudaError_t err;
  if ((err = cudaFuncGetAttributes(&fa, k_put_to_node_peers)) != cudaSuccess)
    return err;
  if ((err = cudaFuncGetAttributes(&fa, k_put_to_expert_peers)) != cudaSuccess)
    return err;
  if ((err = cudaFuncGetAttributes(&fa, k_put_to_owner)) != cudaSuccess)
    return err;
  if ((err = cudaFuncGetAttributes(&fa, k_gin_flush)) != cudaSuccess)
    return err;
  return cudaSuccess;
}

}  // namespace gin_overlap
