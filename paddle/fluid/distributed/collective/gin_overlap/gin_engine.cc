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

// gin_engine.cc: host-side orchestration. Creates the windows, devcomms, side
// streams and events, submits one stage of one bucket to the GPU (submit_*),
// and runs the 2D-bucket gather. Host code only: every kernel is launched
// through a launch_* function of kernels/kernels.h. Terms, the host window
// layout and the constraints are in the gin_engine.h file header.
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

// Collective: every rank of the group must call it with the same size, in the
// same order
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

// ---------------------------------------------------------------- init
const char* GradCommEngine::error_string(int code) {
  if (code >= 10000) return ncclGetErrorString((ncclResult_t)(code - 10000));
  return cudaGetErrorString((cudaError_t)code);
}

int GradCommEngine::nccl_runtime_version() {
  int version = 0;
  ncclGetVersion(&version);
  return version;
}

// Version of the NCCL headers used at build time. The GIN device API is inlined
// from those headers, so it must match the libnccl loaded at run time
int GradCommEngine::nccl_build_version() { return NCCL_VERSION_CODE; }

// Takes Paddle's existing ncclComm for both groups (paddle_nccl_comm).
// expert_ring_id < 0 means there are no expert buckets and the expert group is
// skipped. The size / group rank found must match the values passed in;
// otherwise this returns an error (a wrong ring id is caught here). Call
// set_comms only once.
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
            " group: comm is already set (set_comms may be called only once)";
      return (int)cudaErrorNotReady;
    }
    ncclComm_t comm = nullptr;
    int size = 0, rank = 0;
    std::string reason;
    if (paddle_nccl_comm(
            ring_ids[group_index], device_id, &comm, &size, &rank, &reason)) {
      err = std::string(group_names[group_index]) + " group (ring id " +
            std::to_string(ring_ids[group_index]) + "): " + reason;
      return (int)cudaErrorInvalidValue;
    }
    int comm_rank = -1, comm_size = -1;
    if (ncclCommUserRank(comm, &comm_rank) != ncclSuccess ||
        ncclCommCount(comm, &comm_size) != ncclSuccess) {
      err = std::string(group_names[group_index]) +
            " group: ncclCommUserRank / ncclCommCount failed";
      return (int)cudaErrorInvalidValue;
    }
    if (size != want_size[group_index] || rank != want_rank[group_index] ||
        comm_size != want_size[group_index] ||
        comm_rank != want_rank[group_index]) {
      err = std::string(group_names[group_index]) + " group (ring id " +
            std::to_string(ring_ids[group_index]) +
            "): size / rank mismatch: Python side " +
            std::to_string(want_rank[group_index]) + "/" +
            std::to_string(want_size[group_index]) + ", Paddle ProcessGroup " +
            std::to_string(rank) + "/" + std::to_string(size) + ", ncclComm " +
            std::to_string(comm_rank) + "/" + std::to_string(comm_size) +
            " (is the ring id that of another group?)";
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

// Creates the group's devcomm (collective). Call it after every window of the
// group, the host window included, is registered. Out-params: number of GIN
// contexts, lsaRank, lsaSize (lsaSize must be in 1..16)
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
  // the expert put and the dense inter-node put both go to arbitrary peers
  req.ginConnectionType = NCCL_GIN_CONNECTION_FULL;
  req.ginContextCount = num_gin_contexts_hint;  // a hint; the result may differ
  req.ginSignalCount = 1;                       // VA signals
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

// Sets the node layout of the group: num_nodes nodes, this rank on node
// node_index. Requires num_nodes * ranks_per_node == group size and
// node_index == rank / ranks_per_node
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

// Creates the four side streams, the window mark ring and the per-bucket
// events, takes the byte offsets of the three credit areas in the host windows
// (credit_intra / credit_inter in the dense group, credit_moe in the expert
// group), gets the LSA pointers of the host windows of the node's ranks, and
// preloads every kernel.
// Preloading loads the kernel modules up front so that the first launch does
// not land on the critical path of the backward pass. What a first module load
// does while a kernel of DeepEP / NCCL spins on the GPU in the same process has
// not been verified.
// The bucket counts are passed only here; release_setup reads them from the
// sizes of g_event_*.
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
  // LSA pointers of the host windows of the node's ranks: this GPU writes the
  // peers' counters through them. The dense group uses them for intra-node
  // arrival counters, intra-node credits and the intra-node CE of the 2D
  // gather; the expert group only for the 2D gather
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
// Allocates the group's host window (collective, every rank of the group must
// call it): a HOST_NUMA allocation that both this GPU and the CPU can read and
// write, registered as an NCCL window of the group.
//   - Call it after alloc_windows of the group and before create_dev_comm;
//     otherwise the devcomm does not contain this window.
//   - It must be registered with NCCL_WIN_STRICT_ORDERING, or the CPU may not
//     see the counters a peer NIC writes.
//   - *cpu_ptr = address the CPU reads the counters from; *nbytes = actual size
//     (>= want_bytes, rounded up to the allocation granularity).
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
  // ranks of the same node map this memory, which needs this handle type
  alloc_prop.requestedHandleTypes = CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;
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
  memset(comm_group.host, 0, bytes);  // must be cleared before registration
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
