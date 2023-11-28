/* Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */
#pragma once

#ifdef PADDLE_WITH_GCU
#include <memory>
#include <set>
#include <string>
#include <vector>

#include "paddle/fluid/platform/device/gcu/runtime/gcu_rt_interface.h"
#include "paddle/fluid/platform/device/gcu/runtime/rt_utils.h"

namespace paddle {
namespace platform {
using GcuStreamPtr = std::shared_ptr<paddle::platform::gcu::runtime::Stream>;
// Get the version of GCU runtime.
int GetGcuRuntimeVersion();

// Get the current GCU device id.
int GetGcuCurrentDeviceId();

// Set the GCU device id for next execution.
void SetGcuDeviceId(int device_id);

// Get the total number of GCU devices in system.
int GetGCUDeviceCount();

// Get the maximum allocation size of current GCU device.
size_t GcuMaxAllocSize();

// Get the initial allocation size of current GCU device.
size_t GcuInitAllocSize();

// Get the re-allocation size of current GCU device.
size_t GcuReallocSize();

// Get the minimum chunk size for GCU buddy allocator.
size_t GcuMinChunkSize();

// Get the maximum chunk size for GCU buddy allocator.
size_t GcuMaxChunkSize();

// Copy memory from address src to dst asynchronously.
void GcuMemcpyAsync(void *dst,
                    const void *src,
                    size_t count,
                    topsMemcpyKind kind,
                    topsStream_t stream);

// Copy memory from address src to dst synchronously.
void GcuMemcpySync(void *dst,
                   const void *src,
                   size_t count,
                   topsMemcpyKind kind);

// Set memory dst with value count size asynchronously
void GcuMemsetAsync(void *dst, int value, size_t count, topsStream_t stream);

// Blocks until stream has completed all operations.
void GcuStreamSync(topsStream_t stream);

// Blocks until device has completed all operations.
void GcuDeviceSync();

// Alloc device memory
topsError_t GcuAlloc(void **ptr, size_t size, int dev_id);

// Release device memory
void GcuFree(void *p, size_t size, int dev_id);

// Get tops stream
void *GcuGetTopsStream(const GcuStreamPtr &pstream);

// Adds a callback to be called on the host after all currently enqueued
// items in the stream have completed.
void GcuAddStreamCallback(topsStream_t stream,
                          topsStreamCallback_t callback,
                          void *user_data,
                          unsigned int flags = 0);

enum class GcuSupportListType { WHITE = 0, BLACK = 1, NONE = 2 };

void SetGcuEagerSupportOps(const GcuSupportListType type,
                           const std::set<std::string> support_ops);
void GetGcuEagerSupportOps(GcuSupportListType &type,             // NOLINT
                           std::set<std::string> &support_ops);  // NOLINT

}  // namespace platform
}  // namespace paddle
#endif
