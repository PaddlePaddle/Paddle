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

// Launch wrappers of the device kernels behind the GIN primitives of
// ProcessGroupNCCL. This header does not include nccl_device.h: a device
// communicator is passed as a pointer to an ncclDevComm, and a window as its
// ncclWindow_t handle cast to void*.
//
// A batch holds at most kMaxDescsPerLaunch descriptors and is passed to the
// kernel by value, so a launch does not read host or device memory that the
// caller may reuse after the launch returns. The largest kernel parameter list
// (ncclDevComm + PutBatch) stays below 4 KB.
//
// Every function returns the result of cudaGetLastError() after the launch.
#pragma once

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace paddle {
namespace distributed {
namespace gin_overlap {

constexpr int kMaxDescsPerLaunch = 32;

// One GIN put: `bytes` bytes from `src_offset` of this rank's `src_window` to
// `dst_offset` of `dst_window` on rank `peer`, then a VA signal increment of
// the u64 at `counter_offset` of `counter_window` on rank `peer`.
struct PutDesc {
  void* src_window;
  size_t src_offset;
  void* dst_window;
  size_t dst_offset;
  size_t bytes;
  void* counter_window;
  size_t counter_offset;
  int peer;
};

struct PutBatch {
  PutDesc descs[kMaxDescsPerLaunch];
  int count;
};

// One counter increment by 1. When `lsa_counter` is not null it is the LSA
// address of the peer's counter and the increment is a system-scope atomic
// add; otherwise it is a GIN VA signal increment of the u64 at
// `counter_offset` of `counter_window` on rank `peer`.
struct IncrementPeerCounterDesc {
  uint64_t* lsa_counter;
  void* counter_window;
  size_t counter_offset;
  int peer;
};

struct IncrementPeerCounterBatch {
  IncrementPeerCounterDesc descs[kMaxDescsPerLaunch];
  int count;
};

// One counter store: `value` is written to the LSA address `counter`.
struct SetPeerCounterDesc {
  uint64_t* counter;
  uint64_t value;
};

struct SetPeerCounterBatch {
  SetPeerCounterDesc descs[kMaxDescsPerLaunch];
  int count;
};

// GIN puts, <<<num_blocks, 256>>>. Block b uses GIN context
// b % num_gin_contexts and sends descriptors b, b + num_blocks, ... in turn.
// Every thread of the block takes part in each put (ncclCoopCta).
cudaError_t LaunchPut(const void* dev_comm,
                      const PutBatch& batch,
                      int num_gin_contexts,
                      int num_blocks,
                      cudaStream_t stream);

// Counter increments, <<<1, kMaxDescsPerLaunch>>>, one thread per descriptor.
// LSA: __threadfence_system() then atomicAdd_system. GIN: ncclGin::signal on
// GIN context (descriptor index % num_gin_contexts), one thread
// (ncclCoopThread).
cudaError_t LaunchIncrementPeerCounter(const void* dev_comm,
                                       const IncrementPeerCounterBatch& batch,
                                       int num_gin_contexts,
                                       cudaStream_t stream);

// Counter stores, <<<1, kMaxDescsPerLaunch>>>, one thread per descriptor:
// __threadfence_system() then st.release.sys.u64.
cudaError_t LaunchSetPeerCounter(const SetPeerCounterBatch& batch,
                                 cudaStream_t stream);

// Flushes every GIN context: returns once the puts issued on them so far have
// been sent, so their source buffers can be reused.
// <<<num_gin_contexts, 128>>>, one block per context.
cudaError_t LaunchFlush(const void* dev_comm,
                        int num_gin_contexts,
                        cudaStream_t stream);

}  // namespace gin_overlap
}  // namespace distributed
}  // namespace paddle
