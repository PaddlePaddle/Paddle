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

// signal.cu —— 几个只写几个 u64 的小 kernel：机内计数、回
// credit、写到达格、release store，
//   以及初始化时取本节点各卡 host window 的设备侧地址。
//
// 这些格子都在 host window（CPU 内存）上：本卡 GPU 或对端网卡往里写，本卡 CPU
// 读。写法上有两种：
//   同节点：经 LSA 指针直接写对端那张卡的 host window（NVLink）。
//   跨节点：GIN 的 signal，由网卡做原子加。
#include "paddle/fluid/distributed/collective/gin_overlap/kernels/kernels.h"

namespace gin_overlap {
namespace {

// 给同机 peer（跳过 skip_rank）的机内计数 +1（NVLink 系统级原子）
__global__ void k_signal_intra_node(LsaSlotPointers counter_slots,
                                    int ranks_per_node,
                                    int skip_rank) {
  const int l = threadIdx.x;
  if (l < ranks_per_node && l != skip_rank) {
    __threadfence_system();
    atomicAdd_system(counter_slots.per_rank[l], 1ULL);
  }
}

// 发 credit：给 num_peers 个 peer（world rank peer_base + k*peer_stride，跳过
// skip_rank）window signal_window 偏移 signal_offset 处的 VA signal +1，只发
// signal 不带数据（ncclGin::signal，nccl_device/gin.h:315）。 一个线程一个
// peer。signal_window 传的是本卡的 host window。
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

// 机内到达格「每个发送方一格、单写者」。线程 l（l < ranks_per_node，跳过
// skip_rank = 本卡）把 seq（本卡这个桶
//   已经交出过几次 CE，从 1 开始单调递增）用一个普通 64 位 store 写进
//   arrive_slots.per_rank[l] （接收卡 l 的 host window 里「本桶 ×
//   本卡」那一格）。 为什么一个发送方一格、而不是大家对同一个计数器
//   atomicAdd：这批机器的 CU_DEVICE_ATTRIBUTE_HOST_NATIVE_ATOMIC_SUPPORTED =
//   0，原来让同节点 ranks_per_node-1 张卡对同一个 u64 做
//   atomicAdd_system，实测丢过加
//   1。一人一格、而且写的是绝对序号，就没有竞争了。 写法：先
//   __threadfence_system()，再 st.release.sys（PTX 内存模型里的 release
//   store，system scope）。 release 本身就保证本线程之前的读写先于这次 store
//   对别人可见，fence 是多加的一道保险（和 k_signal_intra_node
//   的写法保持一致）；用 release store 而不是 volatile store，是因为 volatile
//   在 PTX 里只是 relaxed 语义、 得靠前面的 fence 保序，release store
//   自带顺序。 数据本身是同一条流上前面那次
//   cudaMemcpyBatchAsync（拷贝引擎）写的，靠流顺序保证这个 kernel
//   开始前拷贝已完成； 但「拷贝引擎写的数据对接收端 CPU
//   的可见性」这一点**没有实测过**（和改造前一样，是继承下来的假设）。 st
//   用的是 generic 地址（没写 .global）：LSA 指针指向 cuMemMap 出来的 host
//   内存，不假设它落在哪个 state space。
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

// 1 个线程把 value 用 st.release.sys 写进 dst：2D 桶 gather 的同节点发送方 CE
// 拷完本卡这一片之后，
//   写 owner host window 里「本桶 × 本卡」那个到达格。单写者，写法同
//   k_signal_intra_arrive； 数据是同一条流上前面那次 cudaMemcpyBatchAsync
//   写的，靠流顺序。
__global__ void k_store_release(unsigned long long* dst,
                                unsigned long long value) {
  if (threadIdx.x || blockIdx.x) return;
  __threadfence_system();
  asm volatile("st.release.sys.u64 [%0], %1;" ::"l"(dst), "l"(value)
               : "memory");
}

// 取本节点各卡 host window 的设备侧地址（只在初始化 configure 里调一次）：
//   out_pointers[l] = 节点内序号为 l 的那张卡的 host window
//   起点（ncclGetLsaPointer，nccl_device/core.h）。 用的是设备侧 API：host 侧的
//   ncclGetLsaDevicePointer 对 host window 能不能用没验证过，所以这里也在
//   kernel 里取。
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
