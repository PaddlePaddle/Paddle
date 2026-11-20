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
int g_window_mark_index = 0;
unsigned long long g_inter_node_bytes = 0, g_intra_node_bytes = 0;
size_t g_credit_intra_offset = 0, g_credit_inter_offset = 0,
       g_credit_moe_offset = 0;
int g_buckets_ready = 0;
cudaStream_t g_stream_gather_put = nullptr;
cudaStream_t g_stream_gather_gate = nullptr;
std::vector<cudaEvent_t> g_event_gather_source, g_event_gather_put;
int g_gather_ready = 0;

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

// 在 Paddle 当前计算流的当前位置记一个窗口 mark，返回槽号（<0 = 出错）
int GradCommEngine::record_window_mark() {
  if (g_window_marks.empty()) return -1;
  cudaStream_t compute_stream = nullptr;
  if (paddle_compute_stream(&compute_stream)) return -1;
  const int i = g_window_mark_index;
  g_window_mark_index = (g_window_mark_index + 1) % (int)g_window_marks.size();
  if (cudaEventRecord(g_window_marks[i], compute_stream) != cudaSuccess)
    return -1;
  return i;
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

// ----------------------------------------------------------------
// dense 桶的四个阶段（含义见 gin_engine.h 文件头）。调用前放行层已经确认
// 远端条件满足；这里提交到 GPU 的工作只等 mark_event 和本卡的 event。

// 1. 机内拷贝（g_stream_intra_node_copy）：等 mark → CE 把本卡在第 l 列的
//    各块写进同节点卡 l 的机内段（recv_offset）第 row 行 → 把 seq 写进卡 l
//    host window 的 intra_arrive_offset（本卡那一格，各接收卡上偏移相同）
//    → 记 event。seq 必须 > 0，intra_arrive_offset 必须 8 字节对齐。
int submit_dense_intra_node_copy(int bucket,
                                 size_t recv_offset,
                                 size_t send_offset,
                                 long long shard_numel,
                                 cudaEvent_t mark_event,
                                 size_t intra_arrive_offset,
                                 unsigned long long seq) {
  const size_t numel = (size_t)shard_numel;
  CommGroup& comm_group = dense_group();
  const int ranks_per_node = comm_group.ranks_per_node,
            num_nodes = comm_group.num_nodes,
            rank_in_node = comm_group.rank_in_node;
  if (!g_buckets_ready) return (int)cudaErrorNotReady;
  if (ranks_per_node < 2) return 0;
  if (seq == 0 || (intra_arrive_offset & 7)) return (int)cudaErrorInvalidValue;
  const float* grad = (const float*)(comm_group.send + send_offset);
  CUDA_CHECK(cudaStreamWaitEvent(g_stream_intra_node_copy, mark_event, 0));
  std::vector<void*> dst_pointers;
  std::vector<const void*> src_pointers;
  std::vector<size_t> sizes;
  LsaSlotPointers arrive_slots;
  memset(&arrive_slots, 0, sizeof(arrive_slots));
  for (int l = 0; l < ranks_per_node; ++l) {
    if (l == rank_in_node) continue;
    const int row = l > rank_in_node
                        ? rank_in_node
                        : rank_in_node - 1;  // 本卡在接收卡 l 上的行号
    for (int n = 0; n < num_nodes; ++n) {
      dst_pointers.push_back(comm_group.lsa_recv[l] + recv_offset +
                             ((size_t)row * num_nodes + n) * numel * 4);
      src_pointers.push_back(grad + ((size_t)n * ranks_per_node + l) * numel);
      sizes.push_back(numel * 4);
    }
    // 对端 host window 里本卡那一格
    arrive_slots.per_rank[l] =
        (unsigned long long*)(comm_group.lsa_host[l] + intra_arrive_offset);
  }
  cudaMemcpyAttributes copy_attrs;
  memset(&copy_attrs, 0, sizeof(copy_attrs));
  copy_attrs.srcAccessOrder = cudaMemcpySrcAccessOrderStream;
  size_t fail_index = 0;
  CUDA_CHECK(cudaMemcpyBatchAsync(dst_pointers.data(),
                                  src_pointers.data(),
                                  sizes.data(),
                                  dst_pointers.size(),
                                  &copy_attrs,
                                  &fail_index,
                                  1,
                                  g_stream_intra_node_copy));
  {
    const int rc = launch_signal_intra_arrive(arrive_slots,
                                              ranks_per_node,
                                              rank_in_node,
                                              seq,
                                              g_stream_intra_node_copy);
    if (rc) return rc;
  }
  CUDA_CHECK(cudaEventRecord(g_event_intra_node_copy[bucket],
                             g_stream_intra_node_copy));
  g_intra_node_bytes +=
      (unsigned long long)(ranks_per_node - 1) * num_nodes * numel * 4;
  return 0;
}

// 2. 机内求和（g_stream_intra_node_reduce）：等 mark → k_reduce_intra_node
//    → 给同节点其余卡发机内 credit（它们 host window 的
//    [credit_intra + 8*rank_in_node] +1）→ 记 event。shard_numel 必须是 4 的
//    倍数。
int submit_dense_intra_node_reduce(int bucket,
                                   size_t recv_offset,
                                   size_t send_offset,
                                   long long shard_numel,
                                   int num_blocks,
                                   cudaEvent_t mark_event) {
  const size_t numel = (size_t)shard_numel;
  if (!g_buckets_ready) return (int)cudaErrorNotReady;
  if (numel & 3) return (int)cudaErrorInvalidValue;
  CommGroup& comm_group = dense_group();
  const int ranks_per_node = comm_group.ranks_per_node,
            rank_in_node = comm_group.rank_in_node;
  CUDA_CHECK(cudaStreamWaitEvent(g_stream_intra_node_reduce, mark_event, 0));
  {
    const int rc =
        launch_reduce_intra_node((float4*)(comm_group.send + send_offset),
                                 (const float4*)(comm_group.recv + recv_offset),
                                 numel / 4,
                                 ranks_per_node,
                                 comm_group.num_nodes,
                                 rank_in_node,
                                 num_blocks,
                                 g_stream_intra_node_reduce);
    if (rc) return rc;
  }
  if (ranks_per_node > 1) {
    LsaSlotPointers credit_slots;
    memset(&credit_slots, 0, sizeof(credit_slots));
    for (int l = 0; l < ranks_per_node; ++l)
      if (l != rank_in_node)
        credit_slots.per_rank[l] =
            (unsigned long long*)(comm_group.lsa_host[l] +
                                  g_credit_intra_offset +
                                  8 * (size_t)rank_in_node);
    const int rc = launch_signal_intra_node(
        credit_slots, ranks_per_node, rank_in_node, g_stream_intra_node_reduce);
    if (rc) return rc;
  }
  CUDA_CHECK(cudaEventRecord(g_event_intra_node_reduce[bucket],
                             g_stream_intra_node_reduce));
  return 0;
}

// 3. dense put（g_stream_put）：等本桶机内求和的 event → 等 mark → 发给其余
//    节点同卡号的卡，写进它们机间段（recv_offset）第 node_index 行，同时它们
//    host window 的 [signal_offset] +1 → 记 event。
//    *num_signals = 对端 signal 计数这次增加多少。
int submit_dense_inter_node_put(int bucket,
                                size_t recv_offset,
                                size_t send_offset,
                                size_t signal_offset,
                                long long shard_numel,
                                int num_blocks,
                                cudaEvent_t mark_event,
                                int* num_signals) {
  const size_t numel = (size_t)shard_numel;
  *num_signals = 0;
  if (!g_buckets_ready) return (int)cudaErrorNotReady;
  CommGroup& comm_group = dense_group();
  CUDA_CHECK(
      cudaStreamWaitEvent(g_stream_put, g_event_intra_node_reduce[bucket], 0));
  CUDA_CHECK(cudaStreamWaitEvent(g_stream_put, mark_event, 0));
  if (comm_group.num_nodes > 1) {
    // 对端在 host window 上收 signal，所以 signal_window 传 host window
    const int rc = launch_put_to_node_peers(comm_group.dev_comm,
                                            comm_group.win_recv,
                                            comm_group.win_send,
                                            comm_group.win_host,
                                            send_offset,
                                            recv_offset,
                                            signal_offset,
                                            numel,
                                            comm_group.ranks_per_node,
                                            comm_group.num_nodes,
                                            comm_group.node_index,
                                            comm_group.rank_in_node,
                                            comm_group.num_gin_contexts,
                                            num_blocks,
                                            g_stream_put);
    if (rc) return rc;
    *num_signals = comm_group.num_nodes - 1;
    g_inter_node_bytes +=
        (unsigned long long)(comm_group.num_nodes - 1) * numel * 4;
  }
  CUDA_CHECK(cudaEventRecord(g_event_dense_put[bucket], g_stream_put));
  return 0;
}

// 专家 put（g_stream_put，和 dense put 同一条流）：等 mark → 发给专家组其余
// 各卡（组内序号 k），写进它们专家段（recv_offset）第 rank 行，同时它们
// host window 的 [signal_offset] +1 → 记 event。组大小 / 本卡组内序号取自
// expert_group()。
int submit_moe_put(int bucket,
                   size_t recv_offset,
                   size_t send_offset,
                   size_t signal_offset,
                   long long shard_numel,
                   int num_blocks,
                   cudaEvent_t mark_event,
                   int* num_signals) {
  const size_t numel = (size_t)shard_numel;
  *num_signals = 0;
  if (!g_buckets_ready) return (int)cudaErrorNotReady;
  CommGroup& comm_group = expert_group();
  const int expert_group_size = comm_group.size,
            expert_group_rank = comm_group.rank;
  CUDA_CHECK(cudaStreamWaitEvent(g_stream_put, mark_event, 0));
  if (expert_group_size > 1) {
    const int rc = launch_put_to_expert_peers(comm_group.dev_comm,
                                              comm_group.win_recv,
                                              comm_group.win_send,
                                              comm_group.win_host,
                                              recv_offset,
                                              send_offset,
                                              signal_offset,
                                              numel,
                                              expert_group_size,
                                              expert_group_rank,
                                              comm_group.num_gin_contexts,
                                              num_blocks,
                                              g_stream_put);
    if (rc) return rc;
    *num_signals = expert_group_size - 1;
    g_inter_node_bytes +=
        (unsigned long long)(expert_group_size - 1) * numel * 4;
  }
  CUDA_CHECK(cudaEventRecord(g_event_moe_put[bucket], g_stream_put));
  return 0;
}

// 4. dense 分片求和（g_stream_shard_reduce）：等本桶机内求和的 event → 等
//    mark → 本卡分片 = Σ_节点 部分和 × scale → 给其余节点同卡号的卡发机间
//    credit（它们 host window 的 [credit_inter + 8*node_index] +1）→ 记
//    event。num_blocks <= 0 表示按 numel 自动取。
int submit_dense_inter_node_reduce(int bucket,
                                   size_t recv_offset,
                                   size_t send_offset,
                                   long long shard_numel,
                                   float scale,
                                   int num_blocks,
                                   cudaEvent_t mark_event) {
  const size_t numel = (size_t)shard_numel;
  if (!g_buckets_ready) return (int)cudaErrorNotReady;
  CommGroup& comm_group = dense_group();
  CUDA_CHECK(cudaStreamWaitEvent(
      g_stream_shard_reduce, g_event_intra_node_reduce[bucket], 0));
  CUDA_CHECK(cudaStreamWaitEvent(g_stream_shard_reduce, mark_event, 0));
  float* shard = (float*)(comm_group.send + send_offset) +
                 (size_t)(comm_group.node_index * comm_group.ranks_per_node +
                          comm_group.rank_in_node) *
                     numel;
  {
    const int rc = launch_reduce_dense_shard(
        shard,
        (const float*)(comm_group.recv + recv_offset),
        numel,
        comm_group.num_nodes,
        comm_group.node_index,
        scale,
        num_blocks > 0 ? num_blocks : blocks_for_numel(numel),
        g_stream_shard_reduce);
    if (rc) return rc;
  }
  if (comm_group.num_nodes > 1) {
    const int rc = launch_signal_credit(
        comm_group.dev_comm,
        comm_group.win_host,
        g_credit_inter_offset + 8 * (size_t)comm_group.node_index,
        comm_group.num_nodes,
        comm_group.node_index,
        comm_group.rank_in_node,
        comm_group.ranks_per_node,
        comm_group.num_gin_contexts,
        (comm_group.num_nodes + 127) / 128,
        g_stream_shard_reduce);
    if (rc) return rc;
  }
  CUDA_CHECK(cudaEventRecord(g_event_dense_inter_node_reduce[bucket],
                             g_stream_shard_reduce));
  return 0;
}

// 专家分片求和（g_stream_shard_reduce，和 dense 分片求和同一条流）：等 mark
// → 本卡分片 = Σ_k × scale → 给专家组其余各卡发专家 credit（它们 host window
// 的 [credit_moe + 8*rank] +1）→ 记 event。组大小 / 本卡组内序号取自
// expert_group()。
int submit_moe_reduce(int bucket,
                      size_t recv_offset,
                      size_t send_offset,
                      long long shard_numel,
                      float scale,
                      int num_blocks,
                      cudaEvent_t mark_event) {
  const size_t numel = (size_t)shard_numel;
  if (!g_buckets_ready) return (int)cudaErrorNotReady;
  CommGroup& comm_group = expert_group();
  const int expert_group_size = comm_group.size,
            expert_group_rank = comm_group.rank;
  CUDA_CHECK(cudaStreamWaitEvent(g_stream_shard_reduce, mark_event, 0));
  float* shard = (float*)(comm_group.send + send_offset) +
                 (size_t)expert_group_rank * numel;
  {
    const int rc = launch_reduce_moe_shard(
        shard,
        (const float*)(comm_group.recv + recv_offset),
        numel,
        expert_group_size,
        expert_group_rank,
        scale,
        num_blocks > 0 ? num_blocks : blocks_for_numel(numel),
        g_stream_shard_reduce);
    if (rc) return rc;
  }
  if (expert_group_size > 1) {
    const int rc = launch_signal_credit(
        comm_group.dev_comm,
        comm_group.win_host,
        g_credit_moe_offset + 8 * (size_t)expert_group_rank,
        expert_group_size,
        expert_group_rank,
        0,
        1,
        comm_group.num_gin_contexts,
        (expert_group_size + 127) / 128,
        g_stream_shard_reduce);
    if (rc) return rc;
  }
  CUDA_CHECK(
      cudaEventRecord(g_event_moe_reduce[bucket], g_stream_shard_reduce));
  return 0;
}

// ----------------------------------------------------------------
// 窗口 mark 环里的槽号是否有效
static inline bool is_valid_mark(int mark) {
  return mark >= 0 && mark < (int)g_window_marks.size();
}

// 让 stream 等一个桶各阶段侧流的 event：dense 等机内拷贝 / put / 分片求和，
// 专家等 put / 分片求和
static int join_bucket_streams_on(int is_dense,
                                  int bucket,
                                  cudaStream_t stream) {
  if (!g_buckets_ready) return (int)cudaErrorNotReady;
  if (bucket < 0) return (int)cudaErrorInvalidValue;
  if (is_dense) {
    if (bucket >= (int)g_event_intra_node_copy.size() ||
        bucket >= (int)g_event_dense_put.size() ||
        bucket >= (int)g_event_dense_inter_node_reduce.size())
      return (int)cudaErrorInvalidValue;
    CUDA_CHECK(cudaStreamWaitEvent(stream, g_event_intra_node_copy[bucket], 0));
    CUDA_CHECK(cudaStreamWaitEvent(stream, g_event_dense_put[bucket], 0));
    CUDA_CHECK(cudaStreamWaitEvent(
        stream, g_event_dense_inter_node_reduce[bucket], 0));
  } else {
    if (bucket >= (int)g_event_moe_put.size() ||
        bucket >= (int)g_event_moe_reduce.size())
      return (int)cudaErrorInvalidValue;
    CUDA_CHECK(cudaStreamWaitEvent(stream, g_event_moe_put[bucket], 0));
    CUDA_CHECK(cudaStreamWaitEvent(stream, g_event_moe_reduce[bucket], 0));
  }
  return 0;
}

// 计算流等本桶各阶段的 event（reduce_gradients 里逐桶调）
int GradCommEngine::join_bucket_streams(int is_dense, int bucket) {
  GIN_COMPUTE_STREAM(compute_stream);
  return join_bucket_streams_on(is_dense, bucket, compute_stream);
}

// 2D gather 的门流等本桶各阶段的 event
int GradCommEngine::gather_wait_bucket(int is_dense, int bucket) {
  if (!g_gather_ready || !g_stream_gather_gate) return (int)cudaErrorNotReady;
  return join_bucket_streams_on(is_dense, bucket, g_stream_gather_gate);
}

// 本组 send window [send_offset, send_offset + 4*numel) *= scale（计算流）
int GradCommEngine::scale_range(int group,
                                size_t send_offset,
                                long long numel,
                                float scale) {
  if (!is_valid_group(group)) return (int)cudaErrorInvalidValue;
  CommGroup& comm_group = group_at(group);
  if (!comm_group.send ||
      send_offset + (size_t)numel * 4 > comm_group.send_bytes) {
    return (int)cudaErrorInvalidValue;
  }
  GIN_COMPUTE_STREAM(compute_stream);
  return launch_scale_range((float*)(comm_group.send + send_offset),
                            (size_t)numel,
                            scale,
                            blocks_for_numel((size_t)numel),
                            compute_stream);
}

// 每步末尾在计算流上 flush 两个组的全部 GIN context。下一步复用 send window
// 之前必须 flush
int GradCommEngine::end_step() {
  GIN_COMPUTE_STREAM(compute_stream);
  for (int group_index = 0; group_index < NUM_GROUPS; ++group_index) {
    CommGroup& comm_group = group_at(group_index);
    if (!comm_group.dev_comm_ready || comm_group.num_gin_contexts < 1) continue;
    const int rc = launch_gin_flush(
        comm_group.dev_comm, comm_group.num_gin_contexts, compute_stream);
    if (rc) return rc;
  }
  return 0;
}

// [机间 put 的字节数, 机内拷贝的字节数, 保留位]
void GradCommEngine::stats(unsigned long long* out3) {
  out3[0] = g_inter_node_bytes;
  out3[1] = g_intra_node_bytes;
  out3[2] = 0;
}

// ----------------------------------------------------------------
// Muon 2D 桶 gather 到 owner。每张卡做完本卡分片 seg(k) 的分片求和后，把它写进
// owner send window 里同一个桶的同一偏移，owner 拼出整桶。各卡的 send_offset
// 必须一致。
//   - 发送方在 g_stream_gather_put 上：等本卡 event → 跨节点用 GIN put
//     （k_put_to_owner，signal 写 owner host window 的到达格）/ 同节点用 CE
//     （LSA send 指针）+ k_store_release 写 owner host window 里本卡那一格
//     → 记 g_event_gather_put。
//   - owner 端的「收齐了没有」由 CPU 读 host window 判断。
// slot = 2D 桶的 event 下标：dense 桶用 dense 桶号，专家桶用 dense 桶数 +
// 专家桶号。
int GradCommEngine::gather_setup(int num_slots) {
  if (!g_buckets_ready || g_gather_ready || num_slots < 0)
    return (int)cudaErrorNotReady;
  CUDA_CHECK(
      cudaStreamCreateWithFlags(&g_stream_gather_put, cudaStreamNonBlocking));
  CUDA_CHECK(
      cudaStreamCreateWithFlags(&g_stream_gather_gate, cudaStreamNonBlocking));
  g_event_gather_source.assign(num_slots, nullptr);
  g_event_gather_put.assign(num_slots, nullptr);
  for (auto* events : {&g_event_gather_source, &g_event_gather_put})
    for (auto& ev : *events)
      CUDA_CHECK(cudaEventCreateWithFlags(&ev, cudaEventDisableTiming));
  // 同节点 CE 要写 owner 的 send window（两个组各自的 LSA 指针）
  for (int group_index = 0; group_index < NUM_GROUPS; ++group_index) {
    CommGroup& comm_group = group_at(group_index);
    if (!comm_group.dev_comm_ready) continue;
    for (int l = 0; l < comm_group.ranks_per_node; ++l)
      NCCL_CHECK(ncclGetLsaDevicePointer(
          comm_group.win_send, 0, l, (void**)&comm_group.lsa_send[l]));
  }
  {
    const int rc = preload_put_kernels();
    if (rc) return rc;
  }
  {
    const int rc = preload_signal_kernels();
    if (rc) return rc;
  }
  CUDA_CHECK(cudaDeviceSynchronize());
  g_gather_ready = 1;
  return 0;
}

// 2D gather 的门流等窗口 mark：之后排在门流上的工作不会早于 GPU 走到这个 mark
int GradCommEngine::gather_gate(int mark) {
  if (!g_gather_ready || !g_stream_gather_gate) return (int)cudaErrorNotReady;
  if (mark < 0 || mark >= (int)g_window_marks.size())
    return (int)cudaErrorInvalidValue;
  CUDA_CHECK(
      cudaStreamWaitEvent(g_stream_gather_gate, g_window_marks[mark], 0));
  return 0;
}

// 在 2D gather 门流的当前位置记 g_event_gather_source[slot]。调用前门流必须
// 已经等过窗口 mark（gather_gate）和本桶的 event（gather_wait_bucket）
int GradCommEngine::gather_mark_ready(int slot) {
  if (!g_gather_ready || !g_stream_gather_gate) return (int)cudaErrorNotReady;
  if (slot < 0 || slot >= (int)g_event_gather_source.size())
    return (int)cudaErrorInvalidValue;
  return (int)cudaEventRecord(g_event_gather_source[slot],
                              g_stream_gather_gate);
}

// 发送方：等 wait_event → 本卡 send window [send_offset, send_offset +
// 4*numel)（本卡分片）写到 owner（组内序号）send window 的同一偏移 → 记
// g_event_gather_put[slot]。
//   - 跨节点：k_put_to_owner 一笔，owner host window 的 arrive_offset +1
//     （*num_signals = 1）。
//   - 同节点：cudaMemcpyBatchAsync（LSA send 指针）→ k_store_release 把 seq
//     写进 owner host window 的 intra_arrive_offset（*num_signals = 0）。
// wait_event：主线程（release::gather_put_entry）传
// g_event_gather_source[slot]；尾部线程传本桶分片求和之后记的
// g_event_dense_inter_node_reduce / g_event_moe_reduce[bucket]。
int gather_put_to_owner(int group,
                        int slot,
                        size_t send_offset,
                        long long shard_numel,
                        int owner,
                        size_t arrive_offset,
                        size_t intra_arrive_offset,
                        unsigned long long seq,
                        int* num_signals,
                        cudaEvent_t wait_event) {
  const size_t numel = (size_t)shard_numel;
  *num_signals = 0;
  if (!g_gather_ready || !is_valid_group(group)) return (int)cudaErrorNotReady;
  CommGroup& comm_group = group_at(group);
  if (slot < 0 || slot >= (int)g_event_gather_put.size() || owner < 0 ||
      owner >= comm_group.size || owner == comm_group.rank ||
      (send_offset & 3) || send_offset + numel * 4 > comm_group.send_bytes)
    return (int)cudaErrorInvalidValue;
  CUDA_CHECK(cudaStreamWaitEvent(g_stream_gather_put, wait_event, 0));
  if (owner / comm_group.ranks_per_node == comm_group.node_index) {
    const int l = owner % comm_group.ranks_per_node;
    if (!comm_group.lsa_send[l] || !comm_group.lsa_host[l] || seq == 0 ||
        (intra_arrive_offset & 7)) {
      return (int)cudaErrorInvalidValue;
    }
    void* dst = comm_group.lsa_send[l] + send_offset;
    const void* src = comm_group.send + send_offset;
    size_t bytes = numel * 4;
    cudaMemcpyAttributes copy_attrs;
    memset(&copy_attrs, 0, sizeof(copy_attrs));
    copy_attrs.srcAccessOrder = cudaMemcpySrcAccessOrderStream;
    size_t fail_index = 0;
    CUDA_CHECK(cudaMemcpyBatchAsync(&dst,
                                    &src,
                                    &bytes,
                                    1,
                                    &copy_attrs,
                                    &fail_index,
                                    1,
                                    g_stream_gather_put));
    const int rc = launch_store_release(
        (unsigned long long*)(comm_group.lsa_host[l] + intra_arrive_offset),
        seq,
        g_stream_gather_put);
    if (rc) return rc;
    g_intra_node_bytes += bytes;
  } else {
    if (arrive_offset & 7) return (int)cudaErrorInvalidValue;
    const int rc = launch_put_to_owner(comm_group.dev_comm,
                                       comm_group.win_send,
                                       send_offset,
                                       comm_group.win_send,
                                       send_offset,
                                       comm_group.win_host,
                                       arrive_offset,
                                       numel,
                                       owner,
                                       comm_group.num_gin_contexts,
                                       g_stream_gather_put);
    if (rc) return rc;
    *num_signals = 1;
    g_inter_node_bytes += (unsigned long long)numel * 4;
  }
  CUDA_CHECK(cudaEventRecord(g_event_gather_put[slot], g_stream_gather_put));
  return 0;
}

// 主线程发 2D gather put：按表项下标查表后调 gather_put_to_owner，等的 event
// 是 g_event_gather_source[slot]（实现在 release.cu）
int GradCommEngine::gather_put_entry(int entry_index,
                                     unsigned long long seq,
                                     int* num_signals) {
  *num_signals = 0;
  if (!g_gather_ready) return (int)cudaErrorNotReady;
  return release::gather_put_entry(entry_index, seq, num_signals);
}

// 计算流等本卡的 g_event_gather_put[slot]。必须在 end_step 的 flush 之前调，
// flush 才覆盖到这些 put
int GradCommEngine::gather_join(int slot) {
  if (!g_gather_ready) return (int)cudaErrorNotReady;
  if (slot < 0 || slot >= (int)g_event_gather_put.size())
    return (int)cudaErrorInvalidValue;
  GIN_COMPUTE_STREAM(compute_stream);
  return (int)cudaStreamWaitEvent(compute_stream, g_event_gather_put[slot], 0);
}

// ----------------------------------------------------------------
// 放行层的入口（实现在 kernels/release.cu）
int GradCommEngine::release_setup(const std::vector<int>& dense_order,
                                  const std::vector<int>& moe_order,
                                  const std::vector<long long>& dense_table,
                                  const std::vector<long long>& moe_table,
                                  int put_blocks,
                                  size_t intra_arrive_offset,
                                  int num_marks) {
  return release::setup(dense_order,
                        moe_order,
                        dense_table,
                        moe_table,
                        put_blocks,
                        intra_arrive_offset,
                        num_marks);
}
int GradCommEngine::release_step_reset(long long step) {
  return release::step_reset(step);
}
int GradCommEngine::advance(int mark,
                            int stages,
                            int intra_node_reduce_blocks,
                            int shard_reduce_blocks,
                            std::vector<int>& released_shard_reduce) {
  released_shard_reduce.clear();
  if (!is_valid_mark(mark)) return (int)cudaErrorInvalidValue;
  return release::advance(g_window_marks[mark],
                          stages,
                          intra_node_reduce_blocks,
                          shard_reduce_blocks,
                          released_shard_reduce);
}
int GradCommEngine::bucket_ready(int kind, int bucket, float scale, int mark) {
  return release::bucket_ready(kind, bucket, scale, mark);
}
int GradCommEngine::release_status(std::vector<int>& positions,
                                   std::vector<int>& counts,
                                   std::vector<int>& posted,
                                   int* done) {
  return release::status(positions, counts, posted, done);
}
int GradCommEngine::release_stuck_report(std::vector<long long>& flat) {
  return release::stuck_report(flat);
}
int GradCommEngine::release_error(int* code, int* stage, int* bucket) {
  int out3[3] = {0, 0, -1};
  const int rc = release::get_last_error(out3);
  *code = out3[0];
  *stage = out3[1];
  *bucket = out3[2];
  return rc;
}
int GradCommEngine::release_activity() {
  return release::get_current_activity();
}
int GradCommEngine::tail_mark() {
  cudaStream_t compute_stream = nullptr;
  if (paddle_compute_stream(&compute_stream)) return -1;
  return release::record_tail_mark(compute_stream);
}
int GradCommEngine::tail_start(int intra_node_reduce_blocks,
                               int shard_reduce_blocks,
                               int mark,
                               int all_posted) {
  return release::tail_start(
      intra_node_reduce_blocks, shard_reduce_blocks, mark, all_posted);
}
int GradCommEngine::tail_all_posted() { return release::set_all_posted(); }
int GradCommEngine::tail_wait(double timeout_s) {
  return release::wait_for_step_end(timeout_s);
}
int GradCommEngine::tail_abort(double wait_s) {
  return release::abort_step(wait_s);
}
int GradCommEngine::tail_finish() { return release::tail_finish(); }
int GradCommEngine::tail_gather_setup(const std::vector<long long>& table) {
  return release::setup_gather_table(table);
}
int GradCommEngine::tail_gather_set_sequences(
    const std::vector<unsigned long long>& seqs) {
  return release::set_gather_sequences(seqs);
}
int GradCommEngine::tail_gather_get_sent(std::vector<long long>& flat) {
  return release::get_gather_sent(flat);
}

}  // namespace gin_overlap
