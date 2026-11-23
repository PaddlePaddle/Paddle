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
// A batch holds at most kMaxOpsPerLaunch operations and is passed to the
// kernel by value, so a launch does not read host or device memory that the
// caller may reuse after the launch returns. The largest kernel parameter list
// (ncclDevComm + PutBatch) stays below 4 KB.
//
// Every function returns the result of cudaGetLastError() after the launch.
#pragma once

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

#include "paddle/fluid/distributed/collective/gin_overlap/kernels/comm_args.h"

namespace paddle {
namespace distributed {
namespace gin_overlap {

constexpr int kMaxOpsPerLaunch = 32;

// The operations of one launch: puts[0, count) / increments[0, count).
struct PutBatch {
  PutArgs puts[kMaxOpsPerLaunch];
  int count;
};

struct IncrementPeerCounterBatch {
  IncrementPeerCounterArgs increments[kMaxOpsPerLaunch];
  int count;
};

// GIN puts, <<<num_blocks, 256>>>. Block b uses GIN context
// b % num_gin_contexts and sends puts b, b + num_blocks, ... in turn.
// Every thread of the block takes part in each put (ncclCoopCta).
cudaError_t LaunchPut(const void* dev_comm,
                      const PutBatch& batch,
                      int num_gin_contexts,
                      int num_blocks,
                      cudaStream_t stream);

// Counter increments, <<<1, kMaxOpsPerLaunch>>>, one thread per increment.
// lsa_counter set: __threadfence_system() then atomicAdd_system on it.
// Otherwise: ncclGin::signal on
// GIN context (increment index % num_gin_contexts), one thread
// (ncclCoopThread).
cudaError_t LaunchIncrementPeerCounter(const void* dev_comm,
                                       const IncrementPeerCounterBatch& batch,
                                       int num_gin_contexts,
                                       cudaStream_t stream);

// Enqueues a flush of every GIN context on `stream` and returns without
// waiting for it. Once the flush kernel has completed on the stream, the
// source buffers of the puts issued on those contexts before it may be reused.
// <<<num_gin_contexts, 128>>>, one block per context.
cudaError_t LaunchFlush(const void* dev_comm,
                        int num_gin_contexts,
                        cudaStream_t stream);

}  // namespace gin_overlap
}  // namespace distributed
}  // namespace paddle
