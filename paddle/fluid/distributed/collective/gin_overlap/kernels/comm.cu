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

// Device kernels of the GIN primitives of ProcessGroupNCCL. The
// launch wrappers are declared in comm.h.
#include "paddle/fluid/distributed/collective/gin_overlap/kernels/comm.h"

#include <nccl.h>
#include <nccl_device.h>

#if NCCL_VERSION_CODE < 23007
#error "The GIN primitives need NCCL >= 2.30.7 (nccl_device.h)"
#endif

namespace paddle {
namespace distributed {
namespace gin_overlap {
namespace {

// Device code is generated for sm_90 and newer only. For an older arch in the
// build every kernel body is replaced by a trap.
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 900
#define GIN_OVERLAP_COMM_DEVICE_CODE 0
#else
#define GIN_OVERLAP_COMM_DEVICE_CODE 1
#endif

// The 64-bit type of the atomicAdd_system prototype (uint64_t is unsigned long
// on Linux, a different type).
using cuda_u64 = unsigned long long;  // NOLINT(runtime/int)

__global__ void PutKernel(ncclDevComm dev_comm,
                          PutBatch batch,
                          int num_gin_contexts) {
#if GIN_OVERLAP_COMM_DEVICE_CODE
  ncclGin gin(
      dev_comm,
      static_cast<int>(blockIdx.x % static_cast<unsigned>(num_gin_contexts)));
  const ncclTeam team = ncclTeamWorld(dev_comm);
  for (int i = blockIdx.x; i < batch.count; i += gridDim.x) {
    const PutDesc& desc = batch.descs[i];
    gin.put(team,
            desc.peer,
            static_cast<ncclWindow_t>(desc.dst_window),
            desc.dst_offset,
            static_cast<ncclWindow_t>(desc.src_window),
            desc.src_offset,
            desc.bytes,
            ncclGin_VASignalInc{static_cast<ncclWindow_t>(desc.counter_window),
                                desc.counter_offset},
            ncclGin_None{},
            ncclCoopCta());
  }
#else
  __trap();
#endif
}

__global__ void IncrementPeerCounterKernel(ncclDevComm dev_comm,
                                           IncrementPeerCounterBatch batch,
                                           int num_gin_contexts) {
#if GIN_OVERLAP_COMM_DEVICE_CODE
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= batch.count) return;
  const IncrementPeerCounterDesc& desc = batch.descs[i];
  if (desc.lsa_counter != nullptr) {
    __threadfence_system();
    atomicAdd_system(reinterpret_cast<cuda_u64*>(desc.lsa_counter), 1ULL);
    return;
  }
  ncclGin gin(dev_comm, i % num_gin_contexts);
  gin.signal(ncclTeamWorld(dev_comm),
             desc.peer,
             ncclGin_VASignalInc{static_cast<ncclWindow_t>(desc.counter_window),
                                 desc.counter_offset},
             ncclCoopThread());
#else
  __trap();
#endif
}

__global__ void SetPeerCounterKernel(SetPeerCounterBatch batch) {
#if GIN_OVERLAP_COMM_DEVICE_CODE
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= batch.count) return;
  __threadfence_system();
  asm volatile("st.release.sys.u64 [%0], %1;" ::"l"(batch.descs[i].counter),
               "l"(batch.descs[i].value)
               : "memory");
#else
  __trap();
#endif
}

__global__ void FlushKernel(ncclDevComm dev_comm, int num_gin_contexts) {
#if GIN_OVERLAP_COMM_DEVICE_CODE
  if (blockIdx.x >= static_cast<unsigned>(num_gin_contexts)) return;
  ncclGin gin(dev_comm, static_cast<int>(blockIdx.x));
  gin.flush(ncclCoopCta());
#else
  __trap();
#endif
}

}  // namespace

cudaError_t LaunchPut(const void* dev_comm,
                      const PutBatch& batch,
                      int num_gin_contexts,
                      int num_blocks,
                      cudaStream_t stream) {
  PutKernel<<<num_blocks, 256, 0, stream>>>(
      *static_cast<const ncclDevComm*>(dev_comm), batch, num_gin_contexts);
  return cudaGetLastError();
}

cudaError_t LaunchIncrementPeerCounter(const void* dev_comm,
                                       const IncrementPeerCounterBatch& batch,
                                       int num_gin_contexts,
                                       cudaStream_t stream) {
  IncrementPeerCounterKernel<<<1, kMaxDescsPerLaunch, 0, stream>>>(
      *static_cast<const ncclDevComm*>(dev_comm), batch, num_gin_contexts);
  return cudaGetLastError();
}

cudaError_t LaunchSetPeerCounter(const SetPeerCounterBatch& batch,
                                 cudaStream_t stream) {
  SetPeerCounterKernel<<<1, kMaxDescsPerLaunch, 0, stream>>>(batch);
  return cudaGetLastError();
}

cudaError_t LaunchFlush(const void* dev_comm,
                        int num_gin_contexts,
                        cudaStream_t stream) {
  FlushKernel<<<num_gin_contexts, 128, 0, stream>>>(
      *static_cast<const ncclDevComm*>(dev_comm), num_gin_contexts);
  return cudaGetLastError();
}

}  // namespace gin_overlap
}  // namespace distributed
}  // namespace paddle
