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
int g_window_mark_index = 0;
unsigned long long g_inter_node_bytes = 0, g_intra_node_bytes = 0;
size_t g_credit_intra_offset = 0, g_credit_inter_offset = 0,
       g_credit_moe_offset = 0;
int g_buckets_ready = 0;
cudaStream_t g_stream_gather_put = nullptr;
cudaStream_t g_stream_gather_gate = nullptr;
std::vector<cudaEvent_t> g_event_gather_source, g_event_gather_put;
int g_gather_ready = 0;

// Collective: every rank of the group must call it with the same size, in the
// same order. Allocates at least 4096 bytes; *out_bytes = the size actually
// allocated and registered.
static int alloc_symmetric_window(ncclComm_t comm,
                                  size_t bytes,
                                  char** out_ptr,
                                  ncclWindow_t* out_window,
                                  size_t* out_bytes) {
  if (bytes < 4096) bytes = 4096;
  NCCL_CHECK(ncclMemAlloc((void**)out_ptr, bytes));
  CUDA_CHECK(cudaMemset(*out_ptr, 0, bytes));
  NCCL_CHECK(ncclCommWindowRegister(
      comm, *out_ptr, bytes, out_window, NCCL_WIN_COLL_SYMMETRIC));
  *out_bytes = bytes;
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
// set_comms only once. On failure no group is left set, so it can be retried.
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
  // Groups set by this call; a failure clears them again, so set_comms can be
  // retried.
  bool set_here[NUM_GROUPS] = {false, false};
  auto fail = [&](int rc) {
    for (int group_index = 0; group_index < NUM_GROUPS; ++group_index) {
      if (!set_here[group_index]) continue;
      CommGroup& comm_group = group_at(group_index);
      comm_group.comm = nullptr;
      comm_group.size = comm_group.rank = 0;
    }
    return rc;
  };
  for (int group_index = 0; group_index < NUM_GROUPS; ++group_index) {
    if (ring_ids[group_index] < 0) continue;
    CommGroup& comm_group = group_at(group_index);
    if (comm_group.comm) {
      err = std::string(group_names[group_index]) +
            " group: comm is already set (set_comms may be called only once)";
      return fail((int)cudaErrorNotReady);
    }
    ncclComm_t comm = nullptr;
    int size = 0, rank = 0;
    std::string reason;
    if (paddle_nccl_comm(
            ring_ids[group_index], device_id, &comm, &size, &rank, &reason)) {
      err = std::string(group_names[group_index]) + " group (ring id " +
            std::to_string(ring_ids[group_index]) + "): " + reason;
      return fail((int)cudaErrorInvalidValue);
    }
    int comm_rank = -1, comm_size = -1;
    if (ncclCommUserRank(comm, &comm_rank) != ncclSuccess ||
        ncclCommCount(comm, &comm_size) != ncclSuccess) {
      err = std::string(group_names[group_index]) +
            " group: ncclCommUserRank / ncclCommCount failed";
      return fail((int)cudaErrorInvalidValue);
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
      return fail((int)cudaErrorInvalidValue);
    }
    comm_group.comm = comm;
    comm_group.size = want_size[group_index];
    comm_group.rank = want_rank[group_index];
    set_here[group_index] = true;
  }
  return 0;
}

// Allocates and registers the group's send / recv / signal windows
// (collective). A failure cannot be rolled back (window registration is
// collective) and alloc_windows cannot be retried: restart the process.
int GradCommEngine::alloc_windows(int group,
                                  size_t send_bytes,
                                  size_t recv_bytes,
                                  size_t signal_bytes) {
  if (!is_valid_group(group)) return (int)cudaErrorInvalidValue;
  CommGroup& comm_group = group_at(group);
  if (!comm_group.comm || comm_group.send) return (int)cudaErrorNotReady;
  size_t send_actual = 0, recv_actual = 0, signal_actual = 0;
  GIN_CHECK(alloc_symmetric_window(comm_group.comm,
                                   send_bytes,
                                   &comm_group.send,
                                   &comm_group.win_send,
                                   &send_actual));
  comm_group.send_bytes = send_actual;
  GIN_CHECK(alloc_symmetric_window(comm_group.comm,
                                   recv_bytes,
                                   &comm_group.recv,
                                   &comm_group.win_recv,
                                   &recv_actual));
  comm_group.recv_bytes = recv_actual;
  GIN_CHECK(alloc_symmetric_window(comm_group.comm,
                                   signal_bytes,
                                   &comm_group.signal,
                                   &comm_group.win_signal,
                                   &signal_actual));
  CUDA_CHECK(cudaDeviceSynchronize());
  return 0;
}

// Creates the group's devcomm (collective). Call it after every window of the
// group, the host window included, is registered. Out-params: number of GIN
// contexts, lsaRank, lsaSize (lsaSize must be in 1..16). A failure, the
// lsaSize check included, cannot be retried: restart the process.
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

// configure, step by step. On failure configure calls
// release_configure_resources, which undoes everything done here.
static int configure_resources(int num_dense_buckets,
                               int num_moe_buckets,
                               int num_window_marks,
                               size_t credit_intra_offset,
                               size_t credit_inter_offset,
                               size_t credit_moe_offset,
                               void*** lsa_pointers_out) {
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
  void**& lsa_pointers_device = *lsa_pointers_out;
  CUDA_CHECK(cudaMalloc((void**)&lsa_pointers_device, 16 * sizeof(void*)));
  for (int group_index = 0; group_index < NUM_GROUPS; ++group_index) {
    CommGroup& comm_group = group_at(group_index);
    if (!comm_group.host) continue;
    GIN_CHECK(launch_host_window_lsa_pointers(
        comm_group.win_host, comm_group.ranks_per_node, lsa_pointers_device));
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
  lsa_pointers_device = nullptr;
  GIN_CHECK(preload_signal_kernels());
  GIN_CHECK(preload_put_kernels());
  GIN_CHECK(preload_reduce_kernels());
  GIN_CHECK(preload_scale_kernels());
  CUDA_CHECK(cudaDeviceSynchronize());
  return 0;
}

// Destroys the streams and events configure created, frees its temporary
// device buffer, and resets the globals it set, so that a failed configure
// leaves the engine as it was before the call (configure can be retried).
static void release_configure_resources(void*** lsa_pointers_device) {
  for (cudaStream_t* stream : {&g_stream_intra_node_copy,
                               &g_stream_intra_node_reduce,
                               &g_stream_put,
                               &g_stream_shard_reduce}) {
    if (*stream) cudaStreamDestroy(*stream);
    *stream = nullptr;
  }
  for (auto* events : {&g_event_moe_put,
                       &g_event_moe_reduce,
                       &g_event_dense_put,
                       &g_event_intra_node_copy,
                       &g_event_intra_node_reduce,
                       &g_event_dense_inter_node_reduce,
                       &g_window_marks}) {
    for (cudaEvent_t ev : *events)
      if (ev) cudaEventDestroy(ev);
    events->clear();
  }
  if (*lsa_pointers_device) cudaFree(*lsa_pointers_device);
  *lsa_pointers_device = nullptr;
  for (int group_index = 0; group_index < NUM_GROUPS; ++group_index)
    for (char*& pointer : group_at(group_index).lsa_host) pointer = nullptr;
  g_credit_intra_offset = g_credit_inter_offset = g_credit_moe_offset = 0;
  cudaGetLastError();  // drop the error the failed call left behind
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
  void** lsa_pointers_device = nullptr;
  const int rc = configure_resources(num_dense_buckets,
                                     num_moe_buckets,
                                     num_window_marks,
                                     credit_intra_offset,
                                     credit_inter_offset,
                                     credit_moe_offset,
                                     &lsa_pointers_device);
  if (rc) {
    release_configure_resources(&lsa_pointers_device);
    return rc;
  }
  g_buckets_ready = 1;
  return 0;
}

// Records a window mark at the current position of Paddle's compute stream and
// returns its slot (< 0 = error)
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
// Allocates the group's host window (collective, every rank of the group must
// call it): a HOST_NUMA allocation that both this GPU and the CPU can read and
// write, registered as an NCCL window of the group.
//   - Call it after alloc_windows of the group and before create_dev_comm;
//     otherwise the devcomm does not contain this window.
//   - It must be registered with NCCL_WIN_STRICT_ORDERING, or the CPU may not
//     see the counters a peer NIC writes.
//   - *cpu_ptr = address the CPU reads the counters from; *nbytes = actual size
//     (>= want_bytes, rounded up to the allocation granularity).
//   - A failure cannot be retried: restart the process.
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

// ----------------------------------------------------------------
// The four stages of a dense bucket (see the gin_engine.h file header). The
// release layer has already checked the remote conditions before these are
// called; the work submitted here waits only on mark_event and local events.

// 1. intra-node copy (g_stream_intra_node_copy): wait on the mark -> the CE
//    writes this rank's blocks of column l into row `row` of the intra-node
//    segment (recv_offset) of node-local rank l -> writes seq into
//    intra_arrive_offset of rank l's host window (this rank's counter; the
//    offset is the same on every receiver) -> records the event. seq must be
//    > 0 and intra_arrive_offset 8-byte aligned.
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
    // this rank's row on receiver l
    const int row = l > rank_in_node ? rank_in_node : rank_in_node - 1;
    for (int n = 0; n < num_nodes; ++n) {
      dst_pointers.push_back(comm_group.lsa_recv[l] + recv_offset +
                             ((size_t)row * num_nodes + n) * numel * 4);
      src_pointers.push_back(grad + ((size_t)n * ranks_per_node + l) * numel);
      sizes.push_back(numel * 4);
    }
    // this rank's counter in the peer's host window
    arrive_slots.per_rank[l] =
        (unsigned long long*)(comm_group.lsa_host[l] + intra_arrive_offset);
  }
  cudaMemcpyAttributes copy_attrs;
  memset(&copy_attrs, 0, sizeof(copy_attrs));
  copy_attrs.srcAccessOrder = cudaMemcpySrcAccessOrderStream;
  // CUDA 13 signature: (dsts, srcs, sizes, count, attrs, attrsIdxs, numAttrs,
  // stream). One attribute set, applied from copy attrs_idx = 0 on, covers
  // every copy.
  size_t attrs_idx = 0;
  CUDA_CHECK(cudaMemcpyBatchAsync(dst_pointers.data(),
                                  src_pointers.data(),
                                  sizes.data(),
                                  dst_pointers.size(),
                                  &copy_attrs,
                                  &attrs_idx,
                                  1,
                                  g_stream_intra_node_copy));
  GIN_CHECK(launch_signal_intra_arrive(arrive_slots,
                                       ranks_per_node,
                                       rank_in_node,
                                       seq,
                                       g_stream_intra_node_copy));
  CUDA_CHECK(cudaEventRecord(g_event_intra_node_copy[bucket],
                             g_stream_intra_node_copy));
  g_intra_node_bytes +=
      (unsigned long long)(ranks_per_node - 1) * num_nodes * numel * 4;
  return 0;
}

// 2. intra-node reduce (g_stream_intra_node_reduce): wait on the mark ->
//    k_reduce_intra_node -> return an intra-node credit to the other ranks of
//    the node ([credit_intra + 8*rank_in_node] of their host windows +1) ->
//    record the event. shard_numel must be a multiple of 4.
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
  GIN_CHECK(
      launch_reduce_intra_node((float4*)(comm_group.send + send_offset),
                               (const float4*)(comm_group.recv + recv_offset),
                               numel / 4,
                               ranks_per_node,
                               comm_group.num_nodes,
                               rank_in_node,
                               num_blocks,
                               g_stream_intra_node_reduce));
  if (ranks_per_node > 1) {
    LsaSlotPointers credit_slots;
    memset(&credit_slots, 0, sizeof(credit_slots));
    for (int l = 0; l < ranks_per_node; ++l)
      if (l != rank_in_node)
        credit_slots.per_rank[l] =
            (unsigned long long*)(comm_group.lsa_host[l] +
                                  g_credit_intra_offset +
                                  8 * (size_t)rank_in_node);
    GIN_CHECK(launch_signal_intra_node(credit_slots,
                                       ranks_per_node,
                                       rank_in_node,
                                       g_stream_intra_node_reduce));
  }
  CUDA_CHECK(cudaEventRecord(g_event_intra_node_reduce[bucket],
                             g_stream_intra_node_reduce));
  return 0;
}

// 3. dense put (g_stream_put): wait on this bucket's intra-node reduce event ->
//    wait on the mark -> send to the ranks with the same rank_in_node on the
//    other nodes, into row node_index of their inter-node segment
//    (recv_offset), adding 1 to [signal_offset] of their host windows ->
//    record the event. *num_signals = how much the peers' signal counters grow.
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
    // the peers receive the signal in their host window, so the host window is
    // passed as signal_window
    GIN_CHECK(launch_put_to_node_peers(comm_group.dev_comm,
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
                                       g_stream_put));
    *num_signals = comm_group.num_nodes - 1;
    g_inter_node_bytes +=
        (unsigned long long)(comm_group.num_nodes - 1) * numel * 4;
  }
  CUDA_CHECK(cudaEventRecord(g_event_dense_put[bucket], g_stream_put));
  return 0;
}

// expert put (g_stream_put, the same stream as the dense put): wait on the
// mark -> send to every other rank of the expert group (group rank k), into
// row `rank` of their expert segment (recv_offset), adding 1 to
// [signal_offset] of their host windows -> record the event. Group size and
// this rank's group rank come from expert_group().
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
    GIN_CHECK(launch_put_to_expert_peers(comm_group.dev_comm,
                                         comm_group.win_recv,
                                         comm_group.win_send,
                                         comm_group.win_host,
                                         send_offset,
                                         recv_offset,
                                         signal_offset,
                                         numel,
                                         expert_group_size,
                                         expert_group_rank,
                                         comm_group.num_gin_contexts,
                                         num_blocks,
                                         g_stream_put));
    *num_signals = expert_group_size - 1;
    g_inter_node_bytes +=
        (unsigned long long)(expert_group_size - 1) * numel * 4;
  }
  CUDA_CHECK(cudaEventRecord(g_event_moe_put[bucket], g_stream_put));
  return 0;
}

// 4. dense shard reduce (g_stream_shard_reduce): wait on this bucket's
//    intra-node reduce event -> wait on the mark -> this rank's shard =
//    sum over nodes of the partial sums * scale -> return an inter-node credit
//    to the ranks with the same rank_in_node on the other nodes
//    ([credit_inter + 8*node_index] of their host windows +1) -> record the
//    event. num_blocks <= 0 means: derive it from numel.
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
  GIN_CHECK(launch_reduce_dense_shard(
      shard,
      (const float*)(comm_group.recv + recv_offset),
      numel,
      comm_group.num_nodes,
      comm_group.node_index,
      scale,
      num_blocks > 0 ? num_blocks : blocks_for_numel(numel),
      g_stream_shard_reduce));
  if (comm_group.num_nodes > 1) {
    GIN_CHECK(launch_signal_credit(
        comm_group.dev_comm,
        comm_group.win_host,
        g_credit_inter_offset + 8 * (size_t)comm_group.node_index,
        comm_group.num_nodes,
        comm_group.node_index,
        comm_group.rank_in_node,
        comm_group.ranks_per_node,
        comm_group.num_gin_contexts,
        (comm_group.num_nodes + 127) / 128,
        g_stream_shard_reduce));
  }
  CUDA_CHECK(cudaEventRecord(g_event_dense_inter_node_reduce[bucket],
                             g_stream_shard_reduce));
  return 0;
}

// expert shard reduce (g_stream_shard_reduce, the same stream as the dense
// shard reduce): wait on the mark -> this rank's shard = sum over k * scale ->
// return an expert credit to every other rank of the expert group
// ([credit_moe + 8*rank] of their host windows +1) -> record the event. Group
// size and this rank's group rank come from expert_group().
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
  GIN_CHECK(launch_reduce_moe_shard(
      shard,
      (const float*)(comm_group.recv + recv_offset),
      numel,
      expert_group_size,
      expert_group_rank,
      scale,
      num_blocks > 0 ? num_blocks : blocks_for_numel(numel),
      g_stream_shard_reduce));
  if (expert_group_size > 1) {
    GIN_CHECK(launch_signal_credit(
        comm_group.dev_comm,
        comm_group.win_host,
        g_credit_moe_offset + 8 * (size_t)expert_group_rank,
        expert_group_size,
        expert_group_rank,
        0,
        1,
        comm_group.num_gin_contexts,
        (expert_group_size + 127) / 128,
        g_stream_shard_reduce));
  }
  CUDA_CHECK(
      cudaEventRecord(g_event_moe_reduce[bucket], g_stream_shard_reduce));
  return 0;
}

// ----------------------------------------------------------------
// Whether a slot of the window mark ring is valid
static inline bool is_valid_mark(int mark) {
  return mark >= 0 && mark < (int)g_window_marks.size();
}

// Makes `stream` wait on the side-stream events of one bucket: intra-node
// copy / put / shard reduce for a dense bucket, put / shard reduce for an
// expert bucket
static int join_bucket_streams_on(int kind, int bucket, cudaStream_t stream) {
  if (!g_buckets_ready) return (int)cudaErrorNotReady;
  if ((kind != 0 && kind != 1) || bucket < 0) return (int)cudaErrorInvalidValue;
  if (kind == 0) {
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

// The compute stream waits on this bucket's stage events (called per bucket in
// reduce_gradients)
int GradCommEngine::join_bucket_streams(int kind, int bucket) {
  GIN_COMPUTE_STREAM(compute_stream);
  return join_bucket_streams_on(kind, bucket, compute_stream);
}

// The gate stream of the 2D gather waits on this bucket's stage events
int GradCommEngine::gather_wait_bucket(int kind, int bucket) {
  if (!g_gather_ready || !g_stream_gather_gate) return (int)cudaErrorNotReady;
  return join_bucket_streams_on(kind, bucket, g_stream_gather_gate);
}

// [send_offset, send_offset + 4*numel) of the group's send window *= scale (on
// the compute stream)
int GradCommEngine::scale_range(int group,
                                size_t send_offset,
                                long long numel,
                                float scale) {
  if (!is_valid_group(group)) return (int)cudaErrorInvalidValue;
  CommGroup& comm_group = group_at(group);
  if (!comm_group.send) return (int)cudaErrorNotReady;
  // written so that nothing can wrap around: numel < 0 and offsets past the
  // window are rejected before the subtraction
  if (numel < 0 || send_offset > comm_group.send_bytes ||
      (size_t)numel > (comm_group.send_bytes - send_offset) / 4) {
    return (int)cudaErrorInvalidValue;
  }
  GIN_COMPUTE_STREAM(compute_stream);
  return launch_scale_range((float*)(comm_group.send + send_offset),
                            (size_t)numel,
                            scale,
                            blocks_for_numel((size_t)numel),
                            compute_stream);
}

// At the end of each step, flushes every GIN context of both groups on the
// compute stream. The send window must not be reused by the next step before
// this flush
int GradCommEngine::end_step() {
  GIN_COMPUTE_STREAM(compute_stream);
  for (int group_index = 0; group_index < NUM_GROUPS; ++group_index) {
    CommGroup& comm_group = group_at(group_index);
    if (!comm_group.dev_comm_ready || comm_group.num_gin_contexts < 1) continue;
    GIN_CHECK(launch_gin_flush(
        comm_group.dev_comm, comm_group.num_gin_contexts, compute_stream));
  }
  return 0;
}

// [inter-node put bytes, intra-node copy bytes, reserved]
void GradCommEngine::stats(unsigned long long* out3) {
  out3[0] = g_inter_node_bytes;
  out3[1] = g_intra_node_bytes;
  out3[2] = 0;
}

// ----------------------------------------------------------------
// Muon 2D-bucket gather to the owner. After the shard reduce of its shard
// seg(k), every rank writes the shard to the same offset of the same bucket in
// the owner's send window, so the owner ends up with the whole bucket. All
// ranks must use the same send_offset.
//   - The sender, on g_stream_gather_put: wait on a local event -> GIN put
//     across nodes (k_put_to_owner; the signal goes to an arrival counter in
//     the owner's host window) / CE within the node (LSA send pointer) +
//     k_store_release into this rank's counter in the owner's host window ->
//     record g_event_gather_put.
//   - The owner decides "everything arrived" on the CPU, from its host window.
// slot = the event index of the 2D bucket: the bucket id for a dense bucket,
// number of dense buckets + bucket id for an expert bucket.
int GradCommEngine::gather_setup(int num_slots) {
  if (!g_buckets_ready || g_gather_ready) return (int)cudaErrorNotReady;
  if (num_slots < 0) return (int)cudaErrorInvalidValue;
  CUDA_CHECK(
      cudaStreamCreateWithFlags(&g_stream_gather_put, cudaStreamNonBlocking));
  CUDA_CHECK(
      cudaStreamCreateWithFlags(&g_stream_gather_gate, cudaStreamNonBlocking));
  g_event_gather_source.assign(num_slots, nullptr);
  g_event_gather_put.assign(num_slots, nullptr);
  for (auto* events : {&g_event_gather_source, &g_event_gather_put})
    for (auto& ev : *events)
      CUDA_CHECK(cudaEventCreateWithFlags(&ev, cudaEventDisableTiming));
  // the intra-node CE writes the owner's send window (each group's LSA
  // pointers)
  for (int group_index = 0; group_index < NUM_GROUPS; ++group_index) {
    CommGroup& comm_group = group_at(group_index);
    if (!comm_group.dev_comm_ready) continue;
    for (int l = 0; l < comm_group.ranks_per_node; ++l)
      NCCL_CHECK(ncclGetLsaDevicePointer(
          comm_group.win_send, 0, l, (void**)&comm_group.lsa_send[l]));
  }
  GIN_CHECK(preload_put_kernels());
  GIN_CHECK(preload_signal_kernels());
  CUDA_CHECK(cudaDeviceSynchronize());
  g_gather_ready = 1;
  return 0;
}

// The gate stream of the 2D gather waits on a window mark: work queued on it
// afterwards does not start before the GPU reaches that mark
int GradCommEngine::gather_gate(int mark) {
  if (!g_gather_ready || !g_stream_gather_gate) return (int)cudaErrorNotReady;
  if (mark < 0 || mark >= (int)g_window_marks.size())
    return (int)cudaErrorInvalidValue;
  CUDA_CHECK(
      cudaStreamWaitEvent(g_stream_gather_gate, g_window_marks[mark], 0));
  return 0;
}

// Records g_event_gather_source[slot] at the current position of the 2D-gather
// gate stream. The gate stream must already have waited on the window mark
// (gather_gate) and on this bucket's events (gather_wait_bucket)
int GradCommEngine::gather_mark_ready(int slot) {
  if (!g_gather_ready || !g_stream_gather_gate) return (int)cudaErrorNotReady;
  if (slot < 0 || slot >= (int)g_event_gather_source.size())
    return (int)cudaErrorInvalidValue;
  return (int)cudaEventRecord(g_event_gather_source[slot],
                              g_stream_gather_gate);
}

// Sender: wait on wait_event -> write [send_offset, send_offset + 4*numel) of
// this rank's send window (this rank's shard) to the same offset of the
// owner's (group rank) send window -> record g_event_gather_put[slot].
//   - across nodes: one k_put_to_owner, adding 1 to arrive_offset of the
//     owner's host window (*num_signals = 1).
//   - within the node: cudaMemcpyBatchAsync (LSA send pointer) ->
//     k_store_release writes seq into intra_arrive_offset of the owner's host
//     window (*num_signals = 0).
// wait_event: the main thread (release::gather_put_entry) passes
// g_event_gather_source[slot]; the tail thread passes
// g_event_dense_inter_node_reduce / g_event_moe_reduce[bucket], recorded after
// this bucket's shard reduce.
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
  *num_signals = 0;
  if (!g_gather_ready) return (int)cudaErrorNotReady;
  if (!is_valid_group(group) || shard_numel < 0)
    return (int)cudaErrorInvalidValue;
  const size_t numel = (size_t)shard_numel;
  CommGroup& comm_group = group_at(group);
  // bounds written so that nothing can wrap around
  if (slot < 0 || slot >= (int)g_event_gather_put.size() || owner < 0 ||
      owner >= comm_group.size || owner == comm_group.rank ||
      (send_offset & 3) || send_offset > comm_group.send_bytes ||
      numel > (comm_group.send_bytes - send_offset) / 4)
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
    size_t attrs_idx = 0;
    CUDA_CHECK(cudaMemcpyBatchAsync(&dst,
                                    &src,
                                    &bytes,
                                    1,
                                    &copy_attrs,
                                    &attrs_idx,
                                    1,
                                    g_stream_gather_put));
    GIN_CHECK(launch_store_release(
        (unsigned long long*)(comm_group.lsa_host[l] + intra_arrive_offset),
        seq,
        g_stream_gather_put));
    g_intra_node_bytes += bytes;
  } else {
    if (arrive_offset & 7) return (int)cudaErrorInvalidValue;
    GIN_CHECK(launch_put_to_owner(comm_group.dev_comm,
                                  comm_group.win_send,
                                  send_offset,
                                  comm_group.win_send,
                                  send_offset,
                                  comm_group.win_host,
                                  arrive_offset,
                                  numel,
                                  owner,
                                  comm_group.num_gin_contexts,
                                  g_stream_gather_put));
    *num_signals = 1;
    g_inter_node_bytes += (unsigned long long)numel * 4;
  }
  CUDA_CHECK(cudaEventRecord(g_event_gather_put[slot], g_stream_gather_put));
  return 0;
}

// Main thread sends a 2D-gather put: looks up the table entry and calls
// gather_put_to_owner, waiting on g_event_gather_source[slot] (implemented in
// release.cc)
int GradCommEngine::gather_put_entry(int entry_index,
                                     unsigned long long seq,
                                     int* num_signals) {
  *num_signals = 0;
  if (!g_gather_ready) return (int)cudaErrorNotReady;
  return release::gather_put_entry(entry_index, seq, num_signals);
}

// The compute stream waits on this rank's g_event_gather_put[slot]. Call it
// before the flush in end_step so that the flush covers these puts
int GradCommEngine::gather_join(int slot) {
  if (!g_gather_ready) return (int)cudaErrorNotReady;
  if (slot < 0 || slot >= (int)g_event_gather_put.size())
    return (int)cudaErrorInvalidValue;
  GIN_COMPUTE_STREAM(compute_stream);
  return (int)cudaStreamWaitEvent(compute_stream, g_event_gather_put[slot], 0);
}

// ----------------------------------------------------------------
// Entry points of the release layer (implemented in release.cc)
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
