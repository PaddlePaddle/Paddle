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

// kernels.h —— 这套代码用到的全部 CUDA kernel 的 launch 包装，以及 device
// 侧共用的小结构。
//
// 读之前要知道的两件事：
//   window：注册给 NCCL 的对称显存 / 内存块（send / recv / signal /
//   host），对端可以单边读写；下面的
//           offset 参数都是某个 window 里的字节偏移。
//   numel：元素数（float 个数），不是字节数；带 4 后缀的（numel4）是 float4
//   个数 = numel / 4。
//
// host 侧（gin_engine.cc / release.cu）只调下面的 launch_* /
// preload_*，自己不写
// <<<>>>：
//   launch_* 里的 block / thread 数是定好的，不许改 —— 求和 kernel 的 block
//   数直接决定它占多少 SM， 调大了会把 DeepEP
//   挤慢，调小了一步之内放不完（PaddleFleet 的 gin_overlap/scheduler.py 里
//   REDUCE_BLOCKS / FINALIZE_BLOCKS
//   有实测依据）。 preload_* 在初始化时把 kernel
//   的模块加载起来（CUDA_MODULE_LOADING=LAZY 下首次 launch 才加载，
//   而首次加载和别的流上的 memop 等待可能互相等死）。
//   返回值统一是 cudaGetLastError() 的结果（0 = 成功）。
#pragma once

#include <cstddef>

#include <cuda_runtime.h>
#include <nccl.h>
#include <nccl_device.h>

namespace gin_overlap {

// 同一个节点内 ranks_per_node 张卡上「同一个 u64 格子」的地址（机内 signal /
// 到达格 / credit 格）， 下标 = 那张卡在节点内的序号（lsa rank）。16
// 是节点内卡数的上限。
struct LsaSlotPointers {
  unsigned long long* per_rank[16];
};

// 按元素数算 block 数（每 block 256 线程，上限 4096）；host 侧用
inline int blocks_for_numel(size_t numel) {
  size_t num_blocks = (numel + 255) / 256;
  if (num_blocks < 1) num_blocks = 1;
  if (num_blocks > 4096) num_blocks = 4096;
  return (int)num_blocks;
}

// ----------------------------------------------------------------
// reduce.cu（求和） 机内求和：grad 本列块 = 本节点 ranks_per_node
// 张卡那一列之和（原地）：<<<num_blocks, 512>>>
int launch_reduce_intra_node(float4* grad,
                             const float4* intra_recv,
                             size_t numel4,
                             int ranks_per_node,
                             int num_nodes,
                             int rank_in_node,
                             int num_blocks,
                             cudaStream_t stream);
// dense 最终求和：shard = Σ_节点 的部分和 × scale（原地）：<<<num_blocks,
// 256>>>
int launch_reduce_dense_shard(float* shard,
                              const float* inter_recv,
                              size_t numel,
                              int num_nodes,
                              int node_index,
                              float scale,
                              int num_blocks,
                              cudaStream_t stream);
// 专家最终求和：shard = Σ_peer × scale（原地）：<<<num_blocks, 256>>>
int launch_reduce_moe_shard(float* shard,
                            const float* recv,
                            size_t numel,
                            int expert_group_size,
                            int expert_group_rank,
                            float scale,
                            int num_blocks,
                            cudaStream_t stream);
int preload_reduce_kernels();

// ---------------------------------------------------------------- put.cu（GIN
// 单边写） dense 机间 put：发给其余 num_nodes-1 个节点上同卡号的
// rank：<<<num_blocks, 256>>>
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
                             cudaStream_t stream);
// 专家 put：发给专家组其余 expert_group_size-1 个 peer（peer =
// 专家组内序号）：<<<num_blocks, 256>>>
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
                               cudaStream_t stream);
// 2D 桶 gather：本卡这一片整段一笔 put 给 owner：<<<1, 256>>>
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
                        cudaStream_t stream);
// send window 下一步才能复用（本步的 put 都真正发完）：flush 是按 GIN context
// 来的，一个 block 刷一个
//   ：<<<num_gin_contexts, 128>>>
int launch_gin_flush(ncclDevComm dev_comm,
                     int num_gin_contexts,
                     cudaStream_t stream);
int preload_put_kernels();

// ----------------------------------------------------------------
// signal.cu（计数 / 小信号） 给同节点其余 peer 的机内计数格 +1（NVLink
// 系统级原子）：<<<1, 32>>>
int launch_signal_intra_node(LsaSlotPointers counter_slots,
                             int ranks_per_node,
                             int skip_rank,
                             cudaStream_t stream);
// 回 credit：给组内其余 num_peers-1 个 peer 的 credit 格 +1（GIN 只发
// signal、不带数据）：<<<num_blocks, 128>>>。
//   peer 的组内序号 = peer_base + k * peer_stride（dense 要发给同卡号跨节点的
//   rank ⇒ peer_base = rank_in_node、 peer_stride =
//   ranks_per_node；专家组是组内连续的 ⇒ peer_base = 0、peer_stride = 1）。
int launch_signal_credit(ncclDevComm dev_comm,
                         ncclWindow_t signal_window,
                         size_t signal_offset,
                         int num_peers,
                         int skip_rank,
                         int peer_base,
                         int peer_stride,
                         int num_gin_contexts,
                         int num_blocks,
                         cudaStream_t stream);
// 写机内到达格（每个发送方一格、单写者，写的是序号）：<<<1, 32>>>
int launch_signal_intra_arrive(LsaSlotPointers arrive_slots,
                               int ranks_per_node,
                               int skip_rank,
                               unsigned long long seq,
                               cudaStream_t stream);
// 1 个线程用 release store 写一个 u64（2D gather 的同节点到达格）：<<<1, 32>>>
int launch_store_release(unsigned long long* dst,
                         unsigned long long value,
                         cudaStream_t stream);
// 取本节点各卡 host window 的设备侧地址（初始化时调一次）：<<<1, 1>>>，默认流
int launch_host_window_lsa_pointers(ncclWindow_t window,
                                    int ranks_per_node,
                                    void** out_pointers);
int preload_signal_kernels();

// ---------------------------------------------------------------- scale.cu
// [data, data + numel) 每个元素 *= scale：<<<num_blocks, 256>>>
int launch_scale_range(float* data,
                       size_t numel,
                       float scale,
                       int num_blocks,
                       cudaStream_t stream);
int preload_scale_kernels();

}  // namespace gin_overlap
