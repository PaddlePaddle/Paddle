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

// reduce.cu —— 三个求和
// kernel：机内求和（把同节点其余卡推来的那几行和本卡的加起来）、dense 最终求和
//   （把各节点的部分和加起来）、专家最终求和（把专家组各 peer
//   的那一份加起来）。
//
// 数值上必须一字不差地保持：求和顺序（都按 ring 序，从本卡序号 +1
// 开始绕一圈）、以及每个元素先 __fmul_rn 再
//   __fadd_rn 的次序。改动其中任何一项，各卡算出的梯度就不再逐位相同，
//   多机训练会慢慢发散（这套代码是按「和 reduce_scatter 逐位一致」验收的）。
//   block / thread 数不在这张清单里：它只影响每个 block 处理哪些下标，每个
//   元素的来源顺序和运算次序都不变 —— 放行层在同一次运行里就按窗口换求和
//   kernel 的 block 数（scheduler 的 REDUCE_BLOCKS / FINALIZE_BLOCKS），而
//   逐位一致的验收一直是过的。
#include "paddle/fluid/distributed/collective/gin_overlap/kernels/kernels.h"

namespace gin_overlap {
namespace {

// 下面两个最终求和 kernel 的 float4 主体：元素数是 4 的倍数、指针又 16B
// 对齐时，每个线程一次处理 4 个元素，
//   而且每 8 个来源一批：先把这一批的 load 全部发出（先选指针再
//   load），再按原来的来源顺序依次
//   __fmul_rn →
//   __fadd_rn。所以每个元素的结果和逐元素写法逐位相同，只是同一个线程能有多条
//   load 同时在路上。 为什么要这样：原来每个线程每轮只有 1 个 4B
//   访存在途，block 少（32 / 64 个）时实测带宽只有 ~200–400 GB/s。
//   不满足条件时走下面的标量循环（结果一样）。
// fma_rn4 = 一次 4 个 float（float4）的乘加，_rn 是
// round-to-nearest-even：逐元素 __fmul_rn 再 __fadd_rn，
//   次序和标量路径里的 __fadd_rn(acc, __fmul_rn(value, scale)) 完全一样。
__device__ __forceinline__ void fma_rn4(float4& acc,
                                        const float4& value,
                                        float scale) {
  acc.x = __fadd_rn(acc.x, __fmul_rn(value.x, scale));
  acc.y = __fadd_rn(acc.y, __fmul_rn(value.y, scale));
  acc.z = __fadd_rn(acc.z, __fmul_rn(value.z, scale));
  acc.w = __fadd_rn(acc.w, __fmul_rn(value.w, scale));
}
// 能不能走上面 float4 的那条路：元素数是 4 的倍数、而且两个指针都 16B
// 对齐。两条路的结果逐位相同，只差快慢。
__device__ __forceinline__ bool can_use_float4(const void* dst,
                                               const void* src,
                                               size_t numel) {
  return (numel & 3) == 0 && ((reinterpret_cast<uintptr_t>(dst) |
                               reinterpret_cast<uintptr_t>(src)) &
                              15) == 0;
}

// 机内求和（接收池）：dense 机内段里一个桶 =
// intra_recv[ranks_per_node-1][num_nodes][numel]，
//   只放另外 ranks_per_node-1 张卡推来的行：发送卡 l 在接收卡 rank_in_node
//   上占第 (l < rank_in_node ? l : l-1) 行。 grad 本列块 node = Σ_l（l ==
//   rank_in_node ? grad 本列块 node : intra_recv[行(l)][node]）， ring 序从
//   rank_in_node+1 起。float4。 行号映射写成无分支的 l - (l >
//   rank_in_node)，并且每 8 个来源一批：先把这一批的 load 全部发出（先选指针再
//   load）， 再按原顺序依次相加，所以结果和「每次 load
//   之间隔着一个分支」的写法逐位相同，但同一个线程能有多条 LDG.128
//   同时在路上（带分支的写法实测慢约 1.6 倍，看 SASS 得到的）。
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

// dense 最终求和：shard（= 本节点部分和，原地）= Σ_节点（本节点 shard :
// inter_recv[node]）× scale，ring 序从 node_index+1 起
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

// 专家最终求和：本 rank 块 = Σ_k（k == expert_group_rank ? 本块 : recv[k]）×
// scale，ring 序
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

// 预加载（CUDA_MODULE_LOADING=LAZY 下首次 launch 才加载模块，首次 launch
// 和别的流上的 memop 等待可能互相等死）
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
