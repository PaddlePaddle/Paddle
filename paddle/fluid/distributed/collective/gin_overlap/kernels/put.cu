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

// put.cu —— GIN 单边写（put）的四个 kernel：dense 机间 put、专家 put、
// 2D 桶 gather 给 owner 的 put，以及每步末尾的 flush。
//   - 每笔 put 带一个 ncclGin_VASignalInc：数据写进对端 recv / send window，
//     同时对端 host window 里的 signal 计数 +1（网卡原子加）。对端 CPU 看到
//     计数达标，就知道数据已经到了。
//   - 每个 peer 一笔 put，不切片。
#include "paddle/fluid/distributed/collective/gin_overlap/kernels/kernels.h"

namespace gin_overlap {
namespace {

// dense 机间 put：本列第 peer_node 块的部分和 → 节点 peer_node 上同卡号那张
// 卡的机间段第 node_index 行。一个 block 负责一个 peer；block 数少于 peer 数时
// 一个 block 依次发多笔。
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
  ncclGin gin(dev_comm, (int)(blockIdx.x % (unsigned)num_gin_contexts));
  const ncclTeam team = ncclTeamWorld(dev_comm);
  for (int peer_index = blockIdx.x; peer_index < num_nodes - 1;
       peer_index += gridDim.x) {
    const int peer_node = (node_index + 1 + peer_index) % num_nodes;
    gin.put(team,
            peer_node * ranks_per_node + rank_in_node,
            recv_window,
            inter_recv_offset + (size_t)node_index * numel * sizeof(float),
            send_window,
            send_offset + ((size_t)peer_node * ranks_per_node + rank_in_node) *
                              numel * sizeof(float),
            numel * sizeof(float),
            ncclGin_VASignalInc{signal_window, signal_offset},
            ncclGin_None{},
            ncclCoopCta());
  }
}

// 专家 put：本卡第 peer 块 → 专家组里组内序号为 peer 的卡的专家段第
// expert_group_rank 行。block 分配同 k_put_to_node_peers
__global__ void k_put_to_expert_peers(ncclDevComm dev_comm,
                                      ncclWindow_t recv_window,
                                      ncclWindow_t send_window,
                                      ncclWindow_t signal_window,
                                      size_t recv_offset,
                                      size_t send_offset,
                                      size_t signal_offset,
                                      size_t numel,
                                      int expert_group_size,
                                      int expert_group_rank,
                                      int num_gin_contexts) {
  ncclGin gin(dev_comm, (int)(blockIdx.x % (unsigned)num_gin_contexts));
  const ncclTeam team = ncclTeamWorld(dev_comm);
  for (int peer_index = blockIdx.x; peer_index < expert_group_size - 1;
       peer_index += gridDim.x) {
    const int peer = (expert_group_rank + 1 + peer_index) % expert_group_size;
    gin.put(team,
            peer,
            recv_window,
            recv_offset + (size_t)expert_group_rank * numel * sizeof(float),
            send_window,
            send_offset + (size_t)peer * numel * sizeof(float),
            numel * sizeof(float),
            ncclGin_VASignalInc{signal_window, signal_offset},
            ncclGin_None{},
            ncclCoopCta());
  }
}

// 2D 桶 gather（Muon）：本卡分片整段一笔 put 给 owner，同时 owner host window
// 的到达格 +1。1 个 block
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
  ncclGin gin(dev_comm, (int)(blockIdx.x % (unsigned)num_gin_contexts));
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

// flush：一个 block 处理一个 GIN context。下一步复用 send window 之前必须调
__global__ void k_gin_flush(ncclDevComm dev_comm, int num_gin_contexts) {
  if (blockIdx.x >= (unsigned)num_gin_contexts) return;
  ncclGin gin(dev_comm, (int)blockIdx.x);
  gin.flush(ncclCoopCta());
}

}  // namespace

int launch_put_to_node_peers(ncclDevComm dev_comm,
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
  return (int)cudaGetLastError();
}

int launch_put_to_expert_peers(ncclDevComm dev_comm,
                               ncclWindow_t recv_window,
                               ncclWindow_t send_window,
                               ncclWindow_t signal_window,
                               size_t recv_offset,
                               size_t send_offset,
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
                                                        recv_offset,
                                                        send_offset,
                                                        signal_offset,
                                                        numel,
                                                        expert_group_size,
                                                        expert_group_rank,
                                                        num_gin_contexts);
  return (int)cudaGetLastError();
}

int launch_put_to_owner(ncclDevComm dev_comm,
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
  return (int)cudaGetLastError();
}

int launch_gin_flush(ncclDevComm dev_comm,
                     int num_gin_contexts,
                     cudaStream_t stream) {
  k_gin_flush<<<num_gin_contexts, 128, 0, stream>>>(dev_comm, num_gin_contexts);
  return (int)cudaGetLastError();
}

int preload_put_kernels() {
  cudaFuncAttributes fa;
  cudaError_t err;
  if ((err = cudaFuncGetAttributes(&fa, k_put_to_node_peers)) != cudaSuccess)
    return (int)err;
  if ((err = cudaFuncGetAttributes(&fa, k_put_to_expert_peers)) != cudaSuccess)
    return (int)err;
  if ((err = cudaFuncGetAttributes(&fa, k_put_to_owner)) != cudaSuccess)
    return (int)err;
  if ((err = cudaFuncGetAttributes(&fa, k_gin_flush)) != cudaSuccess)
    return (int)err;
  return 0;
}

}  // namespace gin_overlap
