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

// gin_engine.cc —— host 侧编排：建 window、devcomm、侧流和 event，把一个
// 桶的某个阶段提交到 GPU（submit_*），以及 2D 桶 gather。
// 这里只有 host 代码，kernel 一律经 kernels/kernels.h 的 launch_* 发射。
// 术语、host window 布局和约束见 gin_engine.h 文件头。
#include "paddle/fluid/distributed/collective/gin_overlap/gin_engine.h"

#include <cstring>

namespace gin_overlap {

CommGroup g_groups[NUM_GROUPS] = {};
int g_device_id = 0;
cudaStream_t g_stream_intra_node_copy = nullptr,
             g_stream_intra_node_reduce = nullptr, g_stream_put = nullptr,
             g_stream_shard_reduce = nullptr;
std::vector<cudaEvent_t> g_event_intra_node_copy, g_event_intra_node_reduce,
    g_event_dense_put, g_event_moe_put;
std::vector<cudaEvent_t> g_event_dense_inter_node_reduce, g_event_moe_reduce;
std::vector<cudaEvent_t> g_window_marks;
size_t g_credit_intra_offset = 0, g_credit_inter_offset = 0,
       g_credit_moe_offset = 0;
int g_buckets_ready = 0;

// 集合操作：组内所有 rank 必须以相同的大小、相同的顺序调用
static int alloc_symmetric_window(ncclComm_t comm,
                                  size_t bytes,
                                  char** out_ptr,
                                  ncclWindow_t* out_window) {
  if (bytes < 4096) bytes = 4096;
  NCCL_CHECK(ncclMemAlloc((void**)out_ptr, bytes));
  CUDA_CHECK(cudaMemset(*out_ptr, 0, bytes));
  NCCL_CHECK(ncclCommWindowRegister(
      comm, *out_ptr, bytes, out_window, NCCL_WIN_COLL_SYMMETRIC));
  return 0;
}

// ---------------------------------------------------------------- 初始化
const char* GradCommEngine::error_string(int code) {
  if (code >= 10000) return ncclGetErrorString((ncclResult_t)(code - 10000));
  return cudaGetErrorString((cudaError_t)code);
}

int GradCommEngine::nccl_runtime_version() {
  int version = 0;
  ncclGetVersion(&version);
  return version;
}

// 编译时用的 NCCL 头文件版本。GIN 设备 API 内联在头文件里，所以它必须和运行时
// 加载的 libnccl 同版本
int GradCommEngine::nccl_build_version() { return NCCL_VERSION_CODE; }

// 两个组都取 Paddle 已有的 ncclComm（paddle_nccl_comm）。expert_ring_id < 0
// = 没有专家桶，不取专家组。取回的组大小 / 组内序号必须和传入的一致，否则
// 报错（ring id 填错时在这里报出）。set_comms 只能调一次。
int GradCommEngine::set_comms(int dense_ring_id,
                              int expert_ring_id,
                              int device_id,
                              int dense_size,
                              int dense_rank,
                              int expert_size,
                              int expert_rank,
                              std::string& err) {
  err.clear();
  CUDA_CHECK(cudaSetDevice(device_id));
  g_device_id = device_id;
  const int ring_ids[NUM_GROUPS] = {dense_ring_id, expert_ring_id};
  const int want_size[NUM_GROUPS] = {dense_size, expert_size};
  const int want_rank[NUM_GROUPS] = {dense_rank, expert_rank};
  const char* group_names[NUM_GROUPS] = {"sharding", "moe_sharding"};
  for (int group_index = 0; group_index < NUM_GROUPS; ++group_index) {
    if (ring_ids[group_index] < 0) continue;
    CommGroup& comm_group = group_at(group_index);
    if (comm_group.comm) {
      err = std::string(group_names[group_index]) +
            " 组的 comm 已经设置过了（set_comms 只能调一次）";
      return (int)cudaErrorNotReady;
    }
    ncclComm_t comm = nullptr;
    int size = 0, rank = 0;
    std::string reason;
    if (paddle_nccl_comm(
            ring_ids[group_index], device_id, &comm, &size, &rank, &reason)) {
      err = std::string(group_names[group_index]) + " 组（ring id " +
            std::to_string(ring_ids[group_index]) + "）：" + reason;
      return (int)cudaErrorInvalidValue;
    }
    int comm_rank = -1, comm_size = -1;
    if (ncclCommUserRank(comm, &comm_rank) != ncclSuccess ||
        ncclCommCount(comm, &comm_size) != ncclSuccess) {
      err = std::string(group_names[group_index]) +
            " 组：ncclCommUserRank / ncclCommCount 失败";
      return (int)cudaErrorInvalidValue;
    }
    if (size != want_size[group_index] || rank != want_rank[group_index] ||
        comm_size != want_size[group_index] ||
        comm_rank != want_rank[group_index]) {
      err = std::string(group_names[group_index]) + " 组（ring id " +
            std::to_string(ring_ids[group_index]) +
            "）的大小 / 序号对不上：Python 侧 " +
            std::to_string(want_rank[group_index]) + "/" +
            std::to_string(want_size[group_index]) + "，Paddle ProcessGroup " +
            std::to_string(rank) + "/" + std::to_string(size) + "，ncclComm " +
            std::to_string(comm_rank) + "/" + std::to_string(comm_size) +
            "（ring id 是不是填成别的组了？）";
      return (int)cudaErrorInvalidValue;
    }
    comm_group.comm = comm;
    comm_group.size = want_size[group_index];
    comm_group.rank = want_rank[group_index];
  }
  return 0;
}

int GradCommEngine::alloc_windows(int group,
                                  size_t send_bytes,
                                  size_t recv_bytes,
                                  size_t signal_bytes) {
  if (!is_valid_group(group)) return (int)cudaErrorInvalidValue;
  CommGroup& comm_group = group_at(group);
  if (!comm_group.comm || comm_group.send) return (int)cudaErrorNotReady;
  int rc;
  if ((rc = alloc_symmetric_window(comm_group.comm,
                                   send_bytes,
                                   &comm_group.send,
                                   &comm_group.win_send)))
    return rc;
  comm_group.send_bytes = send_bytes < 4096 ? 4096 : send_bytes;
  if ((rc = alloc_symmetric_window(comm_group.comm,
                                   recv_bytes,
                                   &comm_group.recv,
                                   &comm_group.win_recv)))
    return rc;
  if ((rc = alloc_symmetric_window(comm_group.comm,
                                   signal_bytes,
                                   &comm_group.signal,
                                   &comm_group.win_signal))) {
    return rc;
  }
  CUDA_CHECK(cudaDeviceSynchronize());
  return 0;
}

// 建本组的 devcomm（集合操作）。必须在本组所有 window（含 host window）都注册
// 之后调。出参：GIN context 数、lsaRank、lsaSize（lsaSize 必须在 1..16）
int GradCommEngine::create_dev_comm(int group,
                                    int num_gin_contexts_hint,
                                    int* num_contexts,
                                    int* lsa_rank,
                                    int* lsa_size) {
  if (!is_valid_group(group)) return (int)cudaErrorInvalidValue;
  CommGroup& comm_group = group_at(group);
  if (!comm_group.comm || comm_group.dev_comm_ready)
    return (int)cudaErrorNotReady;
  ncclDevCommRequirements_t req = NCCL_DEV_COMM_REQUIREMENTS_INITIALIZER;
  req.ginConnectionType =
      NCCL_GIN_CONNECTION_FULL;  // 专家 put 和 dense 机间 put 都要发给任意 peer
  req.ginContextCount = num_gin_contexts_hint;  // 只是建议值，以实际值为准
  req.ginSignalCount = 1;                       // 用 VA signal
  req.ginCounterCount = 0;
  NCCL_CHECK(ncclDevCommCreate(comm_group.comm, &req, &comm_group.dev_comm));
  comm_group.num_gin_contexts = (int)comm_group.dev_comm.ginContextCount;
  *num_contexts = comm_group.num_gin_contexts;
  *lsa_rank = comm_group.dev_comm.lsaRank;
  *lsa_size = comm_group.dev_comm.lsaSize;
  if (comm_group.dev_comm.lsaSize > 16 || comm_group.dev_comm.lsaSize < 1)
    return (int)cudaErrorInvalidValue;
  for (int l = 0; l < comm_group.dev_comm.lsaSize; ++l) {
    NCCL_CHECK(ncclGetLsaDevicePointer(
        comm_group.win_recv, 0, l, (void**)&comm_group.lsa_recv[l]));
    NCCL_CHECK(ncclGetLsaDevicePointer(
        comm_group.win_signal, 0, l, (void**)&comm_group.lsa_signal[l]));
  }
  comm_group.ranks_per_node = (int)comm_group.dev_comm.lsaSize;
  comm_group.rank_in_node = (int)comm_group.dev_comm.lsaRank;
  comm_group.dev_comm_ready = 1;
  return 0;
}

// 设置组内节点分块：num_nodes 个节点，本卡在第 node_index 个。必须满足
// num_nodes * ranks_per_node == 组大小，且 node_index == rank / ranks_per_node
int GradCommEngine::set_group_topology(int group,
                                       int num_nodes,
                                       int node_index) {
  if (!is_valid_group(group)) return (int)cudaErrorInvalidValue;
  CommGroup& comm_group = group_at(group);
  if (!comm_group.dev_comm_ready || num_nodes < 1 || node_index < 0 ||
      node_index >= num_nodes) {
    return (int)cudaErrorInvalidValue;
  }
  if (num_nodes * comm_group.ranks_per_node != comm_group.size ||
      node_index != comm_group.rank / comm_group.ranks_per_node) {
    return (int)cudaErrorInvalidValue;
  }
  comm_group.num_nodes = num_nodes;
  comm_group.node_index = node_index;
  return 0;
}

// 建四条侧流、窗口 mark 环和每桶 event，收下 host window 里三块 credit 区的
// 字节偏移（credit_intra / credit_inter 在 dense 组，credit_moe 在专家组），
// 取同节点各卡 host window 的 LSA 指针，并预加载全部 kernel。
// 预加载让模块提前加载，免得第一次 launch 落在反向的关键路径上。同一进程里有
// 在 GPU 上自旋等待的 kernel（DeepEP / NCCL）时，第一次加载模块会怎样没有
// 验证过。
// 桶数只在这里传入，release_setup 从 g_event_* 的长度读桶数。
int GradCommEngine::configure(int num_dense_buckets,
                              int num_moe_buckets,
                              int num_window_marks,
                              size_t credit_intra_offset,
                              size_t credit_inter_offset,
                              size_t credit_moe_offset) {
  if (!dense_group().dev_comm_ready || g_buckets_ready || !dense_group().host)
    return (int)cudaErrorNotReady;
  if (num_window_marks <= 0 || num_moe_buckets < 0 || num_dense_buckets < 0)
    return (int)cudaErrorInvalidValue;
  if (num_moe_buckets > 0 &&
      (!expert_group().dev_comm_ready || !expert_group().host))
    return (int)cudaErrorNotReady;
  g_credit_intra_offset = credit_intra_offset;
  g_credit_inter_offset = credit_inter_offset;
  g_credit_moe_offset = credit_moe_offset;
  CUDA_CHECK(cudaStreamCreateWithFlags(&g_stream_intra_node_copy,
                                       cudaStreamNonBlocking));
  CUDA_CHECK(cudaStreamCreateWithFlags(&g_stream_intra_node_reduce,
                                       cudaStreamNonBlocking));
  CUDA_CHECK(cudaStreamCreateWithFlags(&g_stream_put, cudaStreamNonBlocking));
  CUDA_CHECK(
      cudaStreamCreateWithFlags(&g_stream_shard_reduce, cudaStreamNonBlocking));
  g_event_moe_put.assign(num_moe_buckets, nullptr);
  g_event_moe_reduce.assign(num_moe_buckets, nullptr);
  g_event_dense_put.assign(num_dense_buckets, nullptr);
  g_event_intra_node_copy.assign(num_dense_buckets, nullptr);
  g_event_intra_node_reduce.assign(num_dense_buckets, nullptr);
  g_event_dense_inter_node_reduce.assign(num_dense_buckets, nullptr);
  g_window_marks.assign(num_window_marks, nullptr);
  for (auto* events : {&g_event_moe_put,
                       &g_event_moe_reduce,
                       &g_event_dense_put,
                       &g_event_intra_node_copy,
                       &g_event_intra_node_reduce,
                       &g_event_dense_inter_node_reduce,
                       &g_window_marks})
    for (auto& ev : *events)
      CUDA_CHECK(cudaEventCreateWithFlags(&ev, cudaEventDisableTiming));
  // 取同节点各卡 host window 的 LSA 指针，本卡 GPU 用它直接写对端的计数格。
  // dense 组用于机内到达格、机内 credit 和 2D gather 的同节点 CE；专家组只用于
  // 2D gather
  void** lsa_pointers_device = nullptr;
  CUDA_CHECK(cudaMalloc((void**)&lsa_pointers_device, 16 * sizeof(void*)));
  for (int group_index = 0; group_index < NUM_GROUPS; ++group_index) {
    CommGroup& comm_group = group_at(group_index);
    if (!comm_group.host) continue;
    {
      const int rc = launch_host_window_lsa_pointers(
          comm_group.win_host, comm_group.ranks_per_node, lsa_pointers_device);
      if (rc) return rc;
    }
    void* lsa_pointers_host[16] = {};
    CUDA_CHECK(cudaMemcpy(lsa_pointers_host,
                          lsa_pointers_device,
                          (size_t)comm_group.ranks_per_node * sizeof(void*),
                          cudaMemcpyDeviceToHost));
    for (int l = 0; l < comm_group.ranks_per_node; ++l) {
      comm_group.lsa_host[l] = (char*)lsa_pointers_host[l];
      if (l != comm_group.rank_in_node && !comm_group.lsa_host[l])
        return (int)cudaErrorInvalidValue;
    }
  }
  CUDA_CHECK(cudaFree(lsa_pointers_device));
  {
    const int rc = preload_signal_kernels();
    if (rc) return rc;
  }
  {
    const int rc = preload_put_kernels();
    if (rc) return rc;
  }
  {
    const int rc = preload_reduce_kernels();
    if (rc) return rc;
  }
  {
    const int rc = preload_scale_kernels();
    if (rc) return rc;
  }
  CUDA_CHECK(cudaDeviceSynchronize());
  g_buckets_ready = 1;
  return 0;
}

// ----------------------------------------------------------------
// 分配本组的 host window（集合操作，组内所有 rank 都要调）：一块 HOST_NUMA
// 内存，本卡 GPU 和 CPU 都可读写，注册成本组的 NCCL window。
//   - 必须在本组 alloc_windows 之后、create_dev_comm 之前调，否则 devcomm
//     里没有这块 window。
//   - 注册时必须带 NCCL_WIN_STRICT_ORDERING，否则 CPU 读不到对端网卡写入的
//     计数。
//   - *cpu_ptr = CPU 读计数用的地址；*nbytes = 实际大小（≥ want_bytes，按
//     分配粒度向上取整）。
int GradCommEngine::alloc_host_counters(int group,
                                        size_t want_bytes,
                                        uintptr_t* cpu_ptr,
                                        size_t* nbytes) {
  *cpu_ptr = 0;
  *nbytes = 0;
  if (!is_valid_group(group)) return (int)cudaErrorInvalidValue;
  CommGroup& comm_group = group_at(group);
  if (!comm_group.comm || comm_group.host) return (int)cudaErrorNotReady;
  int device_id = 0;
  CUDA_CHECK(cudaGetDevice(&device_id));
  CUdevice cu_device;
  CUDA_DRIVER_CHECK(cuDeviceGet(&cu_device, device_id));
  int numa_id = -1;
  if (cuDeviceGetAttribute(&numa_id,
                           CU_DEVICE_ATTRIBUTE_HOST_NUMA_ID,
                           cu_device) != CUDA_SUCCESS ||
      numa_id < 0) {
    numa_id = 0;
  }
  CUmemAllocationProp alloc_prop;
  memset(&alloc_prop, 0, sizeof(alloc_prop));
  alloc_prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
  alloc_prop.location.type = CU_MEM_LOCATION_TYPE_HOST_NUMA;
  alloc_prop.location.id = numa_id;
  alloc_prop.requestedHandleTypes =
      CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;  // 同节点的卡要映射这块内存，
                                                 // 必须用这种 handle
  size_t granularity = 0;
  CUDA_DRIVER_CHECK(cuMemGetAllocationGranularity(
      &granularity, &alloc_prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM));
  size_t bytes = want_bytes > (size_t)NCCL_WIN_REQUIRED_ALIGNMENT
                     ? want_bytes
                     : (size_t)NCCL_WIN_REQUIRED_ALIGNMENT;
  bytes = (bytes + granularity - 1) / granularity * granularity;
  CUDA_DRIVER_CHECK(
      cuMemCreate(&comm_group.host_handle, bytes, &alloc_prop, 0));
  CUdeviceptr base_ptr = 0;
  CUDA_DRIVER_CHECK(cuMemAddressReserve(&base_ptr, bytes, granularity, 0, 0));
  CUDA_DRIVER_CHECK(cuMemMap(base_ptr, bytes, 0, comm_group.host_handle, 0));
  CUmemAccessDesc access_desc;
  memset(&access_desc, 0, sizeof(access_desc));
  access_desc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  access_desc.location.id = device_id;
  access_desc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
  CUDA_DRIVER_CHECK(cuMemSetAccess(base_ptr, bytes, &access_desc, 1));
  access_desc.location.type = CU_MEM_LOCATION_TYPE_HOST_NUMA;
  access_desc.location.id = numa_id;
  CUDA_DRIVER_CHECK(cuMemSetAccess(base_ptr, bytes, &access_desc, 1));
  comm_group.host = (char*)base_ptr;
  memset(comm_group.host, 0, bytes);  // 必须在注册之前清零
  NCCL_CHECK(ncclCommWindowRegister(
      comm_group.comm,
      comm_group.host,
      bytes,
      &comm_group.win_host,
      NCCL_WIN_COLL_SYMMETRIC | NCCL_WIN_STRICT_ORDERING));
  comm_group.host_bytes = bytes;
  *cpu_ptr = (uintptr_t)comm_group.host;
  *nbytes = bytes;
  return 0;
}

}  // namespace gin_overlap
