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

// signal.cu —— 写计数格的小 kernel：机内 credit、发 credit、机内到达格、
// release store，以及初始化时取同节点各卡 host window 的设备侧地址。
// 这些格子都在对端的 host window 上（布局见 gin_engine.h 文件头），写法有两种：
//   - 同节点：经 LSA 指针直接写对端卡的 host window。
//   - 跨节点：GIN signal，由网卡做原子加。
#include "paddle/fluid/distributed/collective/gin_overlap/kernels/kernels.h"

namespace gin_overlap {
namespace {

// 同节点其余卡（跳过 skip_rank）的机内 credit 格 +1（系统级原子加）
__global__ void k_signal_intra_node(LsaSlotPointers counter_slots,
                                    int ranks_per_node,
                                    int skip_rank) {
  const int l = threadIdx.x;
  if (l < ranks_per_node && l != skip_rank) {
    __threadfence_system();
    atomicAdd_system(counter_slots.per_rank[l], 1ULL);
  }
}

// 发 credit：第 k 个 peer（组内序号 peer_base + k*peer_stride，k 跳过
// skip_rank）的 signal_window 在 signal_offset 处的 VA signal +1，只发 signal
// 不带数据（ncclGin::signal）。一个线程负责一个 peer。signal_window 传 host
// window。
__global__ void k_signal_credit(ncclDevComm dev_comm,
                                ncclWindow_t signal_window,
                                size_t signal_offset,
                                int num_peers,
                                int skip_rank,
                                int peer_base,
                                int peer_stride,
                                int num_gin_contexts) {
  const int k = blockIdx.x * blockDim.x + threadIdx.x;
  if (k >= num_peers || k == skip_rank) return;
  ncclGin gin(dev_comm, k % num_gin_contexts);
  gin.signal(ncclTeamWorld(dev_comm),
             peer_base + k * peer_stride,
             ncclGin_VASignalInc{signal_window, signal_offset},
             ncclCoopThread());
}

// 写机内到达格。线程 l（l < ranks_per_node，跳过 skip_rank = 本卡）把 seq
// 写进 arrive_slots.per_rank[l]，即接收卡 l 的 host window 里「本桶 × 本卡」
// 那一格。
//   - seq = 本卡这个桶已提交的机内拷贝次数，从 1 开始单调递增。
//   - 每张发送卡只写自己那一格，同一格不能有多个写者；写的是绝对值，不是
//     加 1。
//   - 写法：__threadfence_system()，再 st.release.sys（system scope 的
//     release store，generic 地址）。
//   - 机内拷贝的数据由同一条流上前面的 cudaMemcpyBatchAsync 写入，靠流顺序
//     保证本 kernel 开始前拷贝已完成。拷贝引擎写的数据对接收端 CPU 的可见性
//     没有验证过。
__global__ void k_signal_intra_arrive(LsaSlotPointers arrive_slots,
                                      int ranks_per_node,
                                      int skip_rank,
                                      unsigned long long seq) {
  const int l = threadIdx.x;
  if (l < ranks_per_node && l != skip_rank) {
    __threadfence_system();
    asm volatile("st.release.sys.u64 [%0], %1;" ::"l"(arrive_slots.per_rank[l]),
                 "l"(seq)
                 : "memory");
  }
}

// 一个线程把 value 用 st.release.sys 写进 dst。用于 2D 桶 gather：同节点
// 发送方 CE 拷完本卡分片之后，写 owner host window 里「本桶 × 本卡」的到达格。
// 单写者，写法和数据顺序同 k_signal_intra_arrive。
__global__ void k_store_release(unsigned long long* dst,
                                unsigned long long value) {
  if (threadIdx.x || blockIdx.x) return;
  __threadfence_system();
  asm volatile("st.release.sys.u64 [%0], %1;" ::"l"(dst), "l"(value)
               : "memory");
}

// 取同节点各卡 host window 的设备侧地址（只在 configure 里调一次）：
// out_pointers[l] = 节点内序号为 l 的卡的 host window 起点（设备侧
// ncclGetLsaPointer）。
__global__ void k_host_window_lsa_pointers(ncclWindow_t window,
                                           int ranks_per_node,
                                           void** out_pointers) {
  if (threadIdx.x || blockIdx.x) return;
  for (int l = 0; l < ranks_per_node; ++l)
    out_pointers[l] = ncclGetLsaPointer(window, 0, l);
}

}  // namespace

int launch_signal_intra_node(LsaSlotPointers counter_slots,
                             int ranks_per_node,
                             int skip_rank,
                             cudaStream_t stream) {
  k_signal_intra_node<<<1, 32, 0, stream>>>(
      counter_slots, ranks_per_node, skip_rank);
  return (int)cudaGetLastError();
}

int launch_signal_credit(ncclDevComm dev_comm,
                         ncclWindow_t signal_window,
                         size_t signal_offset,
                         int num_peers,
                         int skip_rank,
                         int peer_base,
                         int peer_stride,
                         int num_gin_contexts,
                         int num_blocks,
                         cudaStream_t stream) {
  k_signal_credit<<<num_blocks, 128, 0, stream>>>(dev_comm,
                                                  signal_window,
                                                  signal_offset,
                                                  num_peers,
                                                  skip_rank,
                                                  peer_base,
                                                  peer_stride,
                                                  num_gin_contexts);
  return (int)cudaGetLastError();
}

int launch_signal_intra_arrive(LsaSlotPointers arrive_slots,
                               int ranks_per_node,
                               int skip_rank,
                               unsigned long long seq,
                               cudaStream_t stream) {
  k_signal_intra_arrive<<<1, 32, 0, stream>>>(
      arrive_slots, ranks_per_node, skip_rank, seq);
  return (int)cudaGetLastError();
}

int launch_store_release(unsigned long long* dst,
                         unsigned long long value,
                         cudaStream_t stream) {
  k_store_release<<<1, 32, 0, stream>>>(dst, value);
  return (int)cudaGetLastError();
}

int launch_host_window_lsa_pointers(ncclWindow_t window,
                                    int ranks_per_node,
                                    void** out_pointers) {
  k_host_window_lsa_pointers<<<1, 1>>>(window, ranks_per_node, out_pointers);
  return (int)cudaGetLastError();
}

int preload_signal_kernels() {
  cudaFuncAttributes fa;
  cudaError_t err;
  if ((err = cudaFuncGetAttributes(&fa, k_signal_intra_node)) != cudaSuccess)
    return (int)err;
  if ((err = cudaFuncGetAttributes(&fa, k_signal_credit)) != cudaSuccess)
    return (int)err;
  if ((err = cudaFuncGetAttributes(&fa, k_signal_intra_arrive)) != cudaSuccess)
    return (int)err;
  if ((err = cudaFuncGetAttributes(&fa, k_store_release)) != cudaSuccess)
    return (int)err;
  return 0;
}

}  // namespace gin_overlap
