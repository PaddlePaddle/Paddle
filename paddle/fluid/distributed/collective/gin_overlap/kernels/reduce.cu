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

// reduce.cu —— 三个求和 kernel：
//   - 机内求和：同节点其余卡推来的行加上本卡的块。
//   - dense 分片求和：各节点的部分和相加，再乘 scale。
//   - 专家分片求和：专家组各卡的那一份相加，再乘 scale。
//
// 约束
//   - 求和顺序不能改：都按 ring 序，从本卡序号 +1 开始绕一圈。
//   - 每个元素必须先 __fmul_rn 再 __fadd_rn，次序不能改。
//   - 改了上面任何一条，结果就不再和 reduce_scatter 逐位一致。
//   - block / thread 数不影响结果：它只决定每个线程处理哪些下标，不改变每个
//     元素的求和顺序。
#include "paddle/fluid/distributed/collective/gin_overlap/kernels/kernels.h"

namespace gin_overlap {
namespace {

// 两个分片求和 kernel 都有两条路径，结果逐位相同：
//   - float4 路径（元素数是 4 的倍数且两个指针都 16 字节对齐）：每个线程一次
//     处理 4 个元素；来源每 8 个一批，先发出这一批的全部 load，再按来源顺序
//     依次乘加。
//   - 标量路径：其余情况。
// fma_rn4：对 float4 的 4 个分量各做一次 __fadd_rn(acc, __fmul_rn(value,
// scale))，和标量路径的运算次序相同。
__device__ __forceinline__ void fma_rn4(float4& acc,
                                        const float4& value,
                                        float scale) {
  acc.x = __fadd_rn(acc.x, __fmul_rn(value.x, scale));
  acc.y = __fadd_rn(acc.y, __fmul_rn(value.y, scale));
  acc.z = __fadd_rn(acc.z, __fmul_rn(value.z, scale));
  acc.w = __fadd_rn(acc.w, __fmul_rn(value.w, scale));
}
// 能否走 float4 路径：元素数是 4 的倍数，且两个指针都 16 字节对齐
__device__ __forceinline__ bool can_use_float4(const void* dst,
                                               const void* src,
                                               size_t numel) {
  return (numel & 3) == 0 && ((reinterpret_cast<uintptr_t>(dst) |
                               reinterpret_cast<uintptr_t>(src)) &
                              15) == 0;
}

// 机内求和。dense 机内段里一个桶的布局是
// intra_recv[ranks_per_node-1][num_nodes][numel4]，只放同节点其余卡推来的行：
// 发送卡 l 在本卡上占第 (l < rank_in_node ? l : l-1) 行。
//   grad 本列第 node 块 = Σ_l（l == rank_in_node ? grad 本列第 node 块
//                              : intra_recv[行(l)][node]）
// l 按 ring 序从 rank_in_node+1 起。来源每 8 个一批，先发出这一批的全部 load，
// 再按来源顺序依次相加。
__global__ void k_reduce_intra_node(float4* grad,
                                    const float4* __restrict__ intra_recv,
                                    size_t numel4,
                                    int ranks_per_node,
                                    int num_nodes,
                                    int rank_in_node) {
  const size_t total = (size_t)num_nodes * numel4,
               stride = static_cast<size_t>(gridDim.x) * blockDim.x;
  for (size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       i < total;
       i += stride) {
    const size_t node = i / numel4, elem = i - node * numel4;
    const float4* own_block =
        grad + ((size_t)node * ranks_per_node + rank_in_node) * numel4 + elem;
    const float4* recv_base = intra_recv + (size_t)node * numel4 + elem;
    float4 acc = make_float4(0.f, 0.f, 0.f, 0.f);
    for (int batch = 0; batch < ranks_per_node; batch += 8) {
      float4 vals[8];
#pragma unroll
      for (int j = 0; j < 8; ++j) {
        const int k = batch + j;
        if (k < ranks_per_node) {
          const int l = (rank_in_node + 1 + k) % ranks_per_node;
          const int row = l - (l > rank_in_node);
          vals[j] = *((l == rank_in_node)
                          ? own_block
                          : recv_base + (size_t)row * num_nodes * numel4);
        }
      }
#pragma unroll
      for (int j = 0; j < 8; ++j) {
        if (batch + j < ranks_per_node) {
          acc.x += vals[j].x;
          acc.y += vals[j].y;
          acc.z += vals[j].z;
          acc.w += vals[j].w;
        }
      }
    }
    grad[((size_t)node * ranks_per_node + rank_in_node) * numel4 + elem] = acc;
  }
}

// dense 分片求和（原地）：shard = Σ_node（node == node_index ? shard
// : inter_recv[node]）× scale，node 按 ring 序从 node_index+1 起
__global__ void k_reduce_dense_shard(float* shard,
                                     const float* __restrict__ inter_recv,
                                     size_t numel,
                                     int num_nodes,
                                     int node_index,
                                     float scale) {
  const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
  const size_t thread_first =
      static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (can_use_float4(shard, inter_recv, numel)) {
    const size_t numel4 = numel / 4;
    float4* shard4 = reinterpret_cast<float4*>(shard);
    const float4* recv4 = reinterpret_cast<const float4*>(inter_recv);
    for (size_t elem = thread_first; elem < numel4; elem += stride) {
      float4 acc = make_float4(0.f, 0.f, 0.f, 0.f);
      for (int batch = 1; batch <= num_nodes; batch += 8) {
        float4 vals[8];
#pragma unroll
        for (int j = 0; j < 8; ++j) {
          const int k = batch + j;
          if (k <= num_nodes) {
            const int node = (node_index + k) % num_nodes;
            vals[j] = *((node == node_index)
                            ? (shard4 + elem)
                            : (recv4 + (size_t)node * numel4 + elem));
          }
        }
#pragma unroll
        for (int j = 0; j < 8; ++j)
          if (batch + j <= num_nodes) fma_rn4(acc, vals[j], scale);
      }
      shard4[elem] = acc;
    }
    return;
  }
  for (size_t elem = thread_first; elem < numel; elem += stride) {
    float acc = 0.f;
    for (int k = 1; k <= num_nodes; ++k) {
      const int node = (node_index + k) % num_nodes;
      const float value = (node == node_index)
                              ? shard[elem]
                              : inter_recv[(size_t)node * numel + elem];
      acc = __fadd_rn(acc, __fmul_rn(value, scale));
    }
    shard[elem] = acc;
  }
}

// 专家分片求和（原地）：shard = Σ_k（k == expert_group_rank ? shard
// : recv[k]）× scale，k 按 ring 序从 expert_group_rank+1 起
__global__ void k_reduce_moe_shard(float* shard,
                                   const float* __restrict__ recv,
                                   size_t numel,
                                   int expert_group_size,
                                   int expert_group_rank,
                                   float scale) {
  const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
  const size_t thread_first =
      static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (can_use_float4(shard, recv, numel)) {
    const size_t numel4 = numel / 4;
    float4* shard4 = reinterpret_cast<float4*>(shard);
    const float4* recv4 = reinterpret_cast<const float4*>(recv);
    for (size_t i = thread_first; i < numel4; i += stride) {
      float4 acc = make_float4(0.f, 0.f, 0.f, 0.f);
      for (int batch = 0; batch < expert_group_size; batch += 8) {
        float4 vals[8];
#pragma unroll
        for (int j = 0; j < 8; ++j) {
          const int k = batch + j;
          if (k < expert_group_size) {
            const int peer = (expert_group_rank + 1 + k) % expert_group_size;
            vals[j] = *((peer == expert_group_rank)
                            ? (shard4 + i)
                            : (recv4 + (size_t)peer * numel4 + i));
          }
        }
#pragma unroll
        for (int j = 0; j < 8; ++j)
          if (batch + j < expert_group_size) fma_rn4(acc, vals[j], scale);
      }
      shard4[i] = acc;
    }
    return;
  }
  for (size_t i = thread_first; i < numel; i += stride) {
    float acc = 0.f;
    for (int k = 0; k < expert_group_size; ++k) {
      const int peer = (expert_group_rank + 1 + k) % expert_group_size;
      const float value = (peer == expert_group_rank)
                              ? shard[i]
                              : recv[(size_t)peer * numel + i];
      acc = __fadd_rn(acc, __fmul_rn(value, scale));
    }
    shard[i] = acc;
  }
}

}  // namespace

int launch_reduce_intra_node(float4* grad,
                             const float4* intra_recv,
                             size_t numel4,
                             int ranks_per_node,
                             int num_nodes,
                             int rank_in_node,
                             int num_blocks,
                             cudaStream_t stream) {
  k_reduce_intra_node<<<num_blocks, 512, 0, stream>>>(
      grad, intra_recv, numel4, ranks_per_node, num_nodes, rank_in_node);
  return (int)cudaGetLastError();
}

int launch_reduce_dense_shard(float* shard,
                              const float* inter_recv,
                              size_t numel,
                              int num_nodes,
                              int node_index,
                              float scale,
                              int num_blocks,
                              cudaStream_t stream) {
  k_reduce_dense_shard<<<num_blocks, 256, 0, stream>>>(
      shard, inter_recv, numel, num_nodes, node_index, scale);
  return (int)cudaGetLastError();
}

int launch_reduce_moe_shard(float* shard,
                            const float* recv,
                            size_t numel,
                            int expert_group_size,
                            int expert_group_rank,
                            float scale,
                            int num_blocks,
                            cudaStream_t stream) {
  k_reduce_moe_shard<<<num_blocks, 256, 0, stream>>>(
      shard, recv, numel, expert_group_size, expert_group_rank, scale);
  return (int)cudaGetLastError();
}

// 预加载本文件的 kernel 模块（原因见 kernels.h 文件头）
int preload_reduce_kernels() {
  cudaFuncAttributes fa;
  cudaError_t err;
  if ((err = cudaFuncGetAttributes(&fa, k_reduce_intra_node)) != cudaSuccess)
    return (int)err;
  if ((err = cudaFuncGetAttributes(&fa, k_reduce_dense_shard)) != cudaSuccess)
    return (int)err;
  if ((err = cudaFuncGetAttributes(&fa, k_reduce_moe_shard)) != cudaSuccess)
    return (int)err;
  return 0;
}

}  // namespace gin_overlap
