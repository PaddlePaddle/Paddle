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

// signal.cu: small kernels that write counters: intra-node credits, credits,
// intra-node arrival counters, a release store, and, at init, the
// device-side addresses of the host windows of the node's ranks.
// All of these counters live in a peer's host window (layout in the
// gin_engine.h file header) and are written in one of two ways:
//   - within the node: directly through the LSA pointer of the peer's host
//     window.
//   - across nodes: as a GIN signal, an atomic add done by the NIC.
#include "paddle/fluid/distributed/collective/gin_overlap/kernels/kernels.h"

namespace gin_overlap {
namespace {

// The 64-bit type of CUDA's atomicAdd_system prototype (uint64_t is unsigned
// long on Linux, a different type)
using cuda_u64 = unsigned long long;  // NOLINT(runtime/int)

// Adds 1 to the intra-node credit counter of every other rank of the node
// (skipping skip_rank) (system-scope atomic add)
__global__ void k_signal_intra_node(LsaSlotPointers counter_slots,
                                    int ranks_per_node,
                                    int skip_rank) {
  const int l = threadIdx.x;
  if (l < ranks_per_node && l != skip_rank) {
    __threadfence_system();
    atomicAdd_system(reinterpret_cast<cuda_u64*>(counter_slots.per_rank[l]),
                     1ULL);
  }
}

// Returns a credit: adds 1 to the VA signal at signal_offset of signal_window
// of peer k (group rank peer_base + k*peer_stride; k == skip_rank is
// skipped), signal only, no data (ncclGin::signal). One thread per peer.
// signal_window is the host window.
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

// Writes intra-node arrival counters. Thread l (l < ranks_per_node, skipping
// skip_rank = this rank) writes seq into arrive_slots.per_rank[l], the
// "this bucket x this rank" counter in the host window of receiver l.
//   - seq = how many intra-node copies of this bucket this rank has
//     submitted, starting at 1 and strictly increasing.
//   - Each sender writes only its own counter, so a counter has exactly one
//     writer; the value written is absolute, not an increment.
//   - How: __threadfence_system(), then st.release.sys (system-scope release
//     store on a generic address).
//   - The copied data is written by the cudaMemcpyBatchAsync queued before
//     this kernel on the same stream; stream order makes the copy finish
//     before this kernel starts. Whether data written by the copy engine is
//     visible to the receiving CPU has not been verified.
__global__ void k_signal_intra_arrive(LsaSlotPointers arrive_slots,
                                      int ranks_per_node,
                                      int skip_rank,
                                      uint64_t seq) {
  const int l = threadIdx.x;
  if (l < ranks_per_node && l != skip_rank) {
    __threadfence_system();
    asm volatile("st.release.sys.u64 [%0], %1;" ::"l"(arrive_slots.per_rank[l]),
                 "l"(seq)
                 : "memory");
  }
}

// One thread writes value into dst with st.release.sys. Used by the 2D-bucket
// gather: after the CE of an intra-node sender has copied its shard, this
// writes the "this bucket x this rank" arrival counter in the owner's host
// window. Single writer; same store and data ordering as
// k_signal_intra_arrive.
__global__ void k_store_release(uint64_t* dst, uint64_t value) {
  if (threadIdx.x || blockIdx.x) return;
  __threadfence_system();
  asm volatile("st.release.sys.u64 [%0], %1;" ::"l"(dst), "l"(value)
               : "memory");
}

// Device-side addresses of the host windows of the node's ranks (called only
// once, in configure): out_pointers[l] = start of the host window of the rank
// with node-local index l (device-side ncclGetLsaPointer).
__global__ void k_host_window_lsa_pointers(ncclWindow_t window,
                                           int ranks_per_node,
                                           void** out_pointers) {
  if (threadIdx.x || blockIdx.x) return;
  for (int l = 0; l < ranks_per_node; ++l)
    out_pointers[l] = ncclGetLsaPointer(window, 0, l);
}

}  // namespace

cudaError_t launch_signal_intra_node(LsaSlotPointers counter_slots,
                                     int ranks_per_node,
                                     int skip_rank,
                                     cudaStream_t stream) {
  k_signal_intra_node<<<1, 32, 0, stream>>>(
      counter_slots, ranks_per_node, skip_rank);
  return cudaGetLastError();
}

cudaError_t launch_signal_credit(ncclDevComm dev_comm,
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
  return cudaGetLastError();
}

cudaError_t launch_signal_intra_arrive(LsaSlotPointers arrive_slots,
                                       int ranks_per_node,
                                       int skip_rank,
                                       uint64_t seq,
                                       cudaStream_t stream) {
  k_signal_intra_arrive<<<1, 32, 0, stream>>>(
      arrive_slots, ranks_per_node, skip_rank, seq);
  return cudaGetLastError();
}

cudaError_t launch_store_release(uint64_t* dst,
                                 uint64_t value,
                                 cudaStream_t stream) {
  k_store_release<<<1, 32, 0, stream>>>(dst, value);
  return cudaGetLastError();
}

cudaError_t launch_host_window_lsa_pointers(ncclWindow_t window,
                                            int ranks_per_node,
                                            void** out_pointers) {
  k_host_window_lsa_pointers<<<1, 1>>>(window, ranks_per_node, out_pointers);
  return cudaGetLastError();
}

cudaError_t preload_signal_kernels() {
  cudaFuncAttributes fa;
  cudaError_t err;
  if ((err = cudaFuncGetAttributes(&fa, k_signal_intra_node)) != cudaSuccess)
    return err;
  if ((err = cudaFuncGetAttributes(&fa, k_signal_credit)) != cudaSuccess)
    return err;
  if ((err = cudaFuncGetAttributes(&fa, k_signal_intra_arrive)) != cudaSuccess)
    return err;
  if ((err = cudaFuncGetAttributes(&fa, k_store_release)) != cudaSuccess)
    return err;
  return cudaSuccess;
}

}  // namespace gin_overlap
