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

// gin_engine.cc —— host 侧的编排：建 window / devcomm / 侧流 /
// event，把一个桶的某个阶段交给 GPU，以及 2D 桶 gather。 纯 host
// 代码：这里不出现 <<<>>>，kernel 一律经 kernels/kernels.h 的 launch_* 发射。
// 规则：交给 GPU 的工作只等「本卡已交出的东西」—— 计算流上的 mark、本卡侧流的
// event。不等任何远端计数，
//   也不等 busy。远端计数（credit / 机内到达计数 / signal）由对端直接写进本卡的
//   host window，host（Python 侧） 直接读这块内存，看到条件满足才调下面的
//   submit_* 把对应阶段交出去。
// host window（**每个通信组**一块 HOST_NUMA 内存，注册成本组的 NCCL
// window，偏移和本组 signal window 相同）：
//   dense 组（sharding 组）：
//     [signal_offset + 0]   每个 dense 桶：机间 GIN signal（对端 put 带的
//     VASignalInc，网卡 atomic 写） [intra_arrive + (bucket*16 + l)*8]：dense
//     桶 bucket 的机内到达，发送卡 l 一格、单写者
//                                        （l 的 submit_dense_intra_node_copy 写
//                                        seq）
//     [credit_intra + 8l]：本机卡 l 做完一个 dense
//     桶的机内求和后打（k_signal_intra_node，LSA 指针） [credit_inter +
//     8n]：节点 n 同卡号 rank 做完一个 dense
//     桶的最终求和后打（k_signal_credit，GIN signal）
//   专家组（moe_sharding 组）：
//     [signal_offset + 0]   每个专家桶：专家 GIN signal
//     [credit_moe + 8k] ：专家组第 k 个 rank
//     做完一个专家桶的最终求和后打（k_signal_credit，GIN signal）
//   每格 8B，跨步累计只增不减（CPU 不写、不清零：网卡 atomic 只和网卡 atomic
//   互相原子）。 网卡 atomic 和 GPU 写不落同一个 u64（[0]
//   只有网卡写；机内到达区、credit 机内块只有 GPU 写，机间 /
//   专家块只有网卡写）。
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

// 集合操作：组内所有 rank 同大小同顺序调用
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

// 编译时用的 NCCL 头文件版本（GIN device API 内联在头文件里，必须和运行时加载的
// libnccl 同版本）
int GradCommEngine::nccl_build_version() { return NCCL_VERSION_CODE; }

// 两个组的 comm 都复用 Paddle
// 已经建好的那个（paddle_comm.cc：ProcessGroupMapFromGid → ProcessGroupNCCL
//   → NCCLCommContext::GetNcclComm）。expert_ring_id < 0 =
//   没有专家桶，不取专家组。 回读的组大小 / 组内序号要和 Python
//   侧传来的一致（ring id 填错能在这里报出来），不一致报错。
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
  g_device_id = device_id;  // 之后取当前计算流 / 包 send window 张量都用它
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

// out: [GIN ctx 数, lsaRank, lsaSize]
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
      NCCL_GIN_CONNECTION_FULL;  // 专家组和 dense 机外都要打任意 peer
  req.ginContextCount = num_gin_contexts_hint;  // 只是 hint，按实际回读值轮转
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

// 组内机器分块（Python 侧按组的 rank 列表算好、并和上面回读的 lsaRank / lsaSize
// 核对过）
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

// 建四条侧流 / 窗口 mark 环 / 每桶 event，收下 host window 里三块 credit 的字节
// 偏移，取本机同组各卡 host window 的 LSA 指针，并预加载全部
// kernel（CUDA_MODULE_LOADING=LAZY 下，首次 launch 和别的流上的 memop
// 等待可能互相等死）。组内机器分块在 create_dev_comm / set_group_topology
// 里就定好了。
//   桶数只在这里收一次（放行层的 release_setup 不再带桶数，它从 g_event_* 的
//   长度读），所以「Python 侧桶数」和「引擎里按桶数开的那些
//   event」不可能对不上。 credit_intra / credit_inter 是 dense 组 host window
//   的偏移，credit_moe 是专家组的。
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
  // 取本节点同组各卡 host window 的 LSA 指针：本卡 GPU 用它直接写对端的 host
  // 计数格。
  //   dense 组给机内到达 / 机内 credit / 2D gather 的同节点 CE 用；专家组只给
  //   2D gather 用（专家组内同机成员通常只有本卡）
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

// 在 Paddle 当前计算流的当前位置记一个 mark，返回槽号（<0 = 出错）
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

// ---------------------------------------------------------------- 放计数的
// host window 集合操作（**组内**所有 rank 都要调，而且要在本组 alloc_windows
// 之后、本组 create_dev_comm 之前：
//   必须先把本组所有 window 都注册完，再 ncclDevCommCreate，否则 devcomm
//   里没有这块 window）。 每组一块 HOST_NUMA 内存并注册成
//   window：cuMemCreate（location = HOST_NUMA，numa id 取
//   CU_DEVICE_ATTRIBUTE_HOST_NUMA_ID）→ 按 granularity 向上取整 → cuMemMap →
//   给本卡和 HOST_NUMA 都设 READWRITE → CPU memset 清零 →
//   ncclCommWindowRegister。 注册时带
//   NCCL_WIN_STRICT_ORDERING：实测不带（默认）时对端网卡写的计数读出来会是
//   0，带上才对。 *cpu_ptr = CPU 读计数用的地址；*nbytes = 实际大小（≥
//   want_bytes，按粒度取整，实测粒度 2 MB）。
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
      CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;  // 同节点的卡要把这块内存映射进来，必须用这种
                                                 // handle
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
  memset(
      comm_group.host, 0, bytes);  // 初值清零（注册前，此时还没有任何对端能写）
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
// 一个桶的四个阶段 ① CE（g_stream_intra_node_copy）：等 mark → 推进本机
// ranks_per_node-1 张接收卡机内段 recv_offset 的第 row 行
//    （row = l > rank_in_node ? rank_in_node : rank_in_node-1）
//    → 同流 k_signal_intra_arrive：把 seq 写进接收卡 l 的 host window 偏移
//    intra_arrive_offset
//      （= intra_arrive + (bucket*16 +
//      rank_in_node)*8，本卡那一格，所有接收卡偏移相同）→ event。
//    接收卡机内段腾空（机内 credit 达标）由 host 读本卡 host window
//    判断后才调本函数，GPU 上不等。
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

// ② 机内求和（g_stream_intra_node_reduce）：等 mark → k_reduce_intra_node →
// 给本机 ranks_per_node-1 张发送卡 host window 的
//    credit 槽 [credit_intra + 8*rank_in_node] +1（LSA 指针）→ event。
//    本桶机内到达计数达标（ranks_per_node-1 张卡的 CE 都已落地）由 host 判断。
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

// ③ dense put（g_stream_put）：等本桶机内求和 event（本卡已交出的工作）→ 等
// mark → put 到对端机间段 recv_offset 的
//    第 node_index 行（数据写对端显存 recv window），每个 put 带的 signal
//    打到对端 host window 的 [signal_offset] → event。num_nodes-1
//    个接收卡机间段腾空（机间 credit 达标）由 host 判断。*num_signals = 对端
//    signal 增量
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
    // signal_window 参数传 host window：对端收 signal 的是它的 host window
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

// ③' 专家 put（g_stream_put，和 dense put 同一条流）：等 mark → put
// 到专家组其余 size-1 个 peer
//     （组内序号 k）专家段 recv_offset 的第 rank
//     行（数据写对端显存），signal 打到对端 host window 的 [signal_offset] →
//     event。专家 credit 达标由 host 判断。
//     组大小 / 本卡组内序号取自 expert_group()（set_comms 时和 Python
//     侧核对过的那一份），调用方不再传。
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

// ④ dense 最终求和（g_stream_shard_reduce）：等本桶机内求和 event → 等 mark →
// 本卡 shard = Σ_节点 × scale
//    → 给 num_nodes-1 个远端发送卡 host window 的 credit 槽 [credit_inter +
//    8*node_index] +1 → event。 机间 signal 达标由 host 判断。num_blocks <= 0 =
//    按 numel 取
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

// ④' 专家最终求和（g_stream_shard_reduce，和 dense 最终求和同一条流）：等 mark
// → 本卡块 = Σ_k × scale
//     → 给专家组其余 size-1 个 peer（组内序号 k）host window 的
//     credit 槽 [credit_moe + 8*rank] +1 → event。专家 signal
//     达标由 host 判断。组大小 / 本卡组内序号同 submit_moe_put，取自
//     expert_group()
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
// 窗口 mark 环里第 mark 个 event（放行层的 advance 用；桶号 /
// 其它参数的范围检查在 release.cu 里）
static inline bool is_valid_mark(int mark) {
  return mark >= 0 && mark < (int)g_window_marks.size();
}

// 一个桶的 CE / put / 最终求和三条侧流的 event（都是本卡已交出的工作）排到
// stream 上
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

// reduce_gradients（计算流）：等本桶 CE / put / 最终求和三条侧流的 event
int GradCommEngine::join_bucket_streams(int is_dense, int bucket) {
  GIN_COMPUTE_STREAM(compute_stream);
  return join_bucket_streams_on(is_dense, bucket, compute_stream);
}

// 2D gather 的门流等本桶那三个 event（同上，只是流不同）
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

// 两个组各自 flush 自己的 GIN ctx（send window 下一步才能复用）
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

// ---------------------------------------------------------------- Muon 2D 桶
// gather 到 owner（GIN put / 同节点 CE） 2D 桶（Muon
// act=REDUCE）每卡先得到本卡分片 seg(k) 的最终和，本节把它直接写进 owner send
// window 里同一个桶的 seg(k)
//   （同一偏移：Python 侧校验 send_offset 在组内一致），owner 拼成整桶：
//   发送方（g_stream_gather_put）：等本卡本桶最终求和之后记的 event → 跨节点
//   GIN put（k_put_to_owner，send→send，
//     signal 落 owner host window 的到达格，网卡 atomic）/ 同节点 CE（LSA send
//     指针）+ k_store_release 写 owner host window 里本卡那一格 →
//     g_event_gather_put。
//   远端条件（其余分片到没到）由 host 读 host window 判断；GPU
//   上只等本卡计算流上记的 event。
// slot = 2D 桶在本表里的序号（dense 用 dense 桶号，专家用 dense 桶数 +
// 专家桶号），只用来选 event。
int GradCommEngine::gather_setup(int num_slots) {
  if (!g_buckets_ready || g_gather_ready || num_slots < 0)
    return (int)cudaErrorNotReady;
  CUDA_CHECK(
      cudaStreamCreateWithFlags(&g_stream_gather_put, cudaStreamNonBlocking));
  // 门流：Python 侧原来是 paddle.device.Stream()，同样是 cudaStreamNonBlocking
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

// 2D gather 的门流等这个 mark：之后排在门流上的工作，GPU 一定走到了记 mark
// 的那一刻之后
int GradCommEngine::gather_gate(int mark) {
  if (!g_gather_ready || !g_stream_gather_gate) return (int)cudaErrorNotReady;
  if (mark < 0 || mark >= (int)g_window_marks.size())
    return (int)cudaErrorInvalidValue;
  CUDA_CHECK(
      cudaStreamWaitEvent(g_stream_gather_gate, g_window_marks[mark], 0));
  return 0;
}

// 2D gather 的门流当前位置（它已经等过窗口 mark 和本卡本桶那三个 event）记
// g_event_gather_source[slot]
int GradCommEngine::gather_mark_ready(int slot) {
  if (!g_gather_ready || !g_stream_gather_gate) return (int)cudaErrorNotReady;
  if (slot < 0 || slot >= (int)g_event_gather_source.size())
    return (int)cudaErrorInvalidValue;
  return (int)cudaEventRecord(g_event_gather_source[slot],
                              g_stream_gather_gate);
}

// 发送方：等 wait_event → 本卡 send window [send_offset, send_offset +
// 4*numel)（本卡分片 seg(k)）
//   → owner（world rank）send window 同一偏移。
//   跨节点：k_put_to_owner 一笔，VASignalInc 打 owner host window 的
//   arrive_offset（*num_signals = 1）； 同节点：cudaMemcpyBatchAsync（LSA send
//   指针）→ k_store_release 把 seq 写 owner host window 的
//           intra_arrive_offset（*num_signals = 0）。
//   → g_event_gather_put[slot]。
// wait_event：主线程（release::gather_put_entry）传
// g_event_gather_source[slot]；尾部 proxy 传本桶最终求和之后记的
//   g_event_dense_inter_node_reduce / g_event_moe_reduce[bucket]。
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

// 主线程那条路：表项下标 → 放行层查表（gather_entries）→ gather_put_to_owner，
//   等的 event 是 g_event_gather_source[slot]（实现在 release.cu，表在那边）
int GradCommEngine::gather_put_entry(int entry_index,
                                     unsigned long long seq,
                                     int* num_signals) {
  *num_signals = 0;
  if (!g_gather_ready) return (int)cudaErrorNotReady;
  return release::gather_put_entry(entry_index, seq, num_signals);
}

// 计算流等本卡 g_event_gather_put[slot]
//   （本卡交出的 put / CE；end_step 的 flush 之前调，flush 才覆盖到这些 put）
int GradCommEngine::gather_join(int slot) {
  if (!g_gather_ready) return (int)cudaErrorNotReady;
  if (slot < 0 || slot >= (int)g_event_gather_put.size())
    return (int)cudaErrorInvalidValue;
  GIN_COMPUTE_STREAM(compute_stream);
  return (int)cudaStreamWaitEvent(compute_stream, g_event_gather_put[slot], 0);
}

// ---------------------------------------------------------------- 放行层
// （实现在 kernels/release.cu）
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
