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

// kernels.h —— 全部 CUDA kernel 的 launch 包装，以及设备侧共用的小结构。
// 术语（window、LSA、GIN、credit 等）见 gin_engine.h 文件头。
//
// 约定
//   - offset 参数都是某个 window 里的字节偏移。
//   - numel 是元素数（float 个数），不是字节数；numel4 是 float4 个数
//     （= numel / 4）。
//   - host 侧（gin_engine.cc / release.cu）只调这里的 launch_* / preload_*，
//     不直接写 <<<>>>。
//   - 每个 launch_* 的 thread 数是固定的（写在各函数的注释里）。求和 kernel
//     的 block 数由调用方传入，它决定求和占多少 SM。
//   - preload_* 在初始化时把 kernel 所在的模块提前加载，免得第一次 launch
//     落在反向的关键路径上。同一进程里有在 GPU 上自旋等待的 kernel（DeepEP /
//     NCCL）时，第一次加载模块会怎样没有验证过。
//   - 返回值都是 cudaGetLastError() 的结果（0 = 成功）。
#pragma once

#include <cstddef>

#include <cuda_runtime.h>
#include <nccl.h>
#include <nccl_device.h>

namespace gin_overlap {

// 同一节点内各卡 host window 上「同一个 u64 格子」的地址（机内到达格或
// credit 格），下标 = 那张卡的节点内序号（lsaRank）。节点内最多 16 张卡。
struct LsaSlotPointers {
  unsigned long long* per_rank[16];
};

// 按元素数算 block 数（每 block 256 线程，最少 1，最多 4096）
inline int blocks_for_numel(size_t numel) {
  size_t num_blocks = (numel + 255) / 256;
  if (num_blocks < 1) num_blocks = 1;
  if (num_blocks > 4096) num_blocks = 4096;
  return (int)num_blocks;
}

// ----------------------------------------------------------------
// reduce.cu：求和
// 机内求和：本列各块 = 本节点各卡对应块之和（原地写回 grad）。
// <<<num_blocks, 512>>>
int launch_reduce_intra_node(float4* grad,
                             const float4* intra_recv,
                             size_t numel4,
                             int ranks_per_node,
                             int num_nodes,
                             int rank_in_node,
                             int num_blocks,
                             cudaStream_t stream);
// dense 分片求和：shard = Σ_节点 部分和 × scale（原地）。<<<num_blocks, 256>>>
int launch_reduce_dense_shard(float* shard,
                              const float* inter_recv,
                              size_t numel,
                              int num_nodes,
                              int node_index,
                              float scale,
                              int num_blocks,
                              cudaStream_t stream);
// 专家分片求和：shard = Σ_peer × scale（原地）。<<<num_blocks, 256>>>
int launch_reduce_moe_shard(float* shard,
                            const float* recv,
                            size_t numel,
                            int expert_group_size,
                            int expert_group_rank,
                            float scale,
                            int num_blocks,
                            cudaStream_t stream);
int preload_reduce_kernels();

// ----------------------------------------------------------------
// put.cu：GIN 单边写
// dense 机间 put：发给其余 num_nodes-1 个节点上同卡号的卡。
// <<<num_blocks, 256>>>
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
// 专家 put：发给专家组其余 expert_group_size-1 张卡（按组内序号）。
// <<<num_blocks, 256>>>
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
// 2D 桶 gather：本卡分片整段一笔 put 给 owner。<<<1, 256>>>
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
// flush 全部 GIN context，等本步的 put 都发完；下一步复用 send window 之前
// 必须调。一个 block 处理一个 context。<<<num_gin_contexts, 128>>>
int launch_gin_flush(ncclDevComm dev_comm,
                     int num_gin_contexts,
                     cudaStream_t stream);
int preload_put_kernels();

// ----------------------------------------------------------------
// signal.cu：计数格
// 同节点其余卡的机内 credit 格 +1（系统级原子加）。<<<1, 32>>>
int launch_signal_intra_node(LsaSlotPointers counter_slots,
                             int ranks_per_node,
                             int skip_rank,
                             cudaStream_t stream);
// 发 credit：组内其余 num_peers-1 张卡的 credit 格 +1（只发 GIN signal，不带
// 数据）。<<<num_blocks, 128>>>
// 第 k 个 peer 的组内序号 = peer_base + k * peer_stride：dense 发给其余节点
// 同卡号的卡，peer_base = rank_in_node、peer_stride = ranks_per_node；专家组
// peer_base = 0、peer_stride = 1。跳过 k == skip_rank。
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
// 把 seq 写进同节点其余卡的机内到达格（每张发送卡只写自己那一格）。
// <<<1, 32>>>
int launch_signal_intra_arrive(LsaSlotPointers arrive_slots,
                               int ranks_per_node,
                               int skip_rank,
                               unsigned long long seq,
                               cudaStream_t stream);
// 用一个线程以 release store 写一个 u64（2D gather 的同节点到达格）。
// <<<1, 32>>>
int launch_store_release(unsigned long long* dst,
                         unsigned long long value,
                         cudaStream_t stream);
// 取同节点各卡 host window 的设备侧地址，写进 out_pointers（设备内存）。
// 初始化时调一次，在默认流上。<<<1, 1>>>
int launch_host_window_lsa_pointers(ncclWindow_t window,
                                    int ranks_per_node,
                                    void** out_pointers);
int preload_signal_kernels();

// ---------------------------------------------------------------- scale.cu
// [data, data + numel) 每个元素 *= scale。<<<num_blocks, 256>>>
int launch_scale_range(float* data,
                       size_t numel,
                       float scale,
                       int num_blocks,
                       cudaStream_t stream);
int preload_scale_kernels();

}  // namespace gin_overlap
