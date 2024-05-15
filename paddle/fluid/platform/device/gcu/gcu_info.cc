/* Copyright (c) 2023 enflame Authors. All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/platform/device/gcu/gcu_info.h"

#include <algorithm>
#include <iostream>
#include <string>

#include "gflags/gflags.h"
#include "glog/logging.h"
#include "paddle/fluid/platform/device/gcu/gcu_device.h"
#include "paddle/phi/common/place.h"

DECLARE_string(selected_gcus);
namespace paddle {
namespace platform {
namespace {
inline std::vector<std::string> Split(std::string const &original,
                                      char separator) {
  std::vector<std::string> results;
  std::string token;
  std::istringstream is(original);
  while (std::getline(is, token, separator)) {
    if (!token.empty()) {
      results.push_back(token);
    }
  }
  return results;
}
}  // namespace

int GetGcuRuntimeVersion() {
  return 3000;  // use runtime 3.0
}

// Get the current GCU device id.
int GetGcuCurrentDeviceId() {
  return paddle::platform::gcu::runtime::GcuGetCurrentDevice();
}

// Set the GCU device id for next execution.
void SetGcuDeviceId(int device_id) {
  paddle::platform::gcu::runtime::GcuSetCurrentDevice(device_id);
}

// Get the total number of GCU devices in system.
int GetGCUDeviceCount() {
  return paddle::platform::gcu::runtime::GcuVisibleDeviceCount();
}

// Get the maximum allocation size of current GCU device.
size_t GcuMaxAllocSize() {
  return std::max(GcuInitAllocSize(), GcuReallocSize());
}

// Get the initial allocation size of current GCU device.
size_t GcuInitAllocSize() {
  // Initial memory size is tentatively 1GB, which can be adjusted or
  // controlled by environmental variables as needed
  size_t init_mb = 512;
  return init_mb << 20;
}

// Get the re-allocation size of current GCU device.
size_t GcuReallocSize() {
  // Realloc memory size is tentatively 1GB, which can be adjusted or
  // controlled by environmental variables as needed
  size_t re_alloc_mb = 512;
  return re_alloc_mb << 20;
}

// Get the minimum chunk size for GCU buddy allocator.
size_t GcuMinChunkSize() {
  // Allow to allocate the minimum chunk size is 256 bytes.
  return 1 << 8;
}

// Get the maximum chunk size for GCU buddy allocator.
size_t GcuMaxChunkSize() { return GcuMaxAllocSize(); }

// Copy memory from address src to dst asynchronously.
void GcuMemcpyAsync(void *dst,
                    const void *src,
                    size_t count,
                    topsMemcpyKind kind,
                    topsStream_t stream) {
  paddle::platform::gcu::runtime::GcuMemcpyAsync(dst, src, count, kind, stream);
}

// Copy memory from address src to dst synchronously.
void GcuMemcpySync(void *dst,
                   const void *src,
                   size_t count,
                   topsMemcpyKind kind) {
  paddle::platform::gcu::runtime::GcuMemcpySync(dst, src, count, kind);
}

// Set memory dst with value count size asynchronously
void GcuMemsetAsync(void *dst, int value, size_t count, topsStream_t stream) {
  paddle::platform::gcu::runtime::GcuMemsetAsync(dst, value, count, stream);
}

// Blocks until stream has completed all operations.
void GcuStreamSync(topsStream_t stream) {
  paddle::platform::gcu::runtime::GcuStreamSynchronize(stream);
}

// Blocks until device has completed all operations.
void GcuDeviceSync(int device_id) {
  paddle::platform::gcu::runtime::GcuSynchronizeDevice(device_id);
}

// Alloc device memory
topsError_t GcuAlloc(void **ptr, size_t size, int dev_id) {
  return paddle::platform::gcu::runtime::GcuNativeAlloc(ptr, size, dev_id);
}

// Release device memory
void GcuFree(void *p, size_t size, int dev_id) {
  paddle::platform::gcu::runtime::GcuNativeFree(p, size, dev_id);
}

// Get tops stream
void *GcuGetTopsStream(const GcuStreamPtr &pstream) {
  return paddle::platform::gcu::runtime::GcuStreamImpl(pstream);
}

// Adds a callback to be called on the host after all currently enqueued
// items in the stream have completed.
void GcuAddStreamCallback(topsStream_t stream,
                          topsStreamCallback_t callback,
                          void *user_data,
                          unsigned int flags) {
  paddle::platform::gcu::runtime::GcuAddStreamCallback(
      stream, callback, user_data, flags);
}

static GcuSupportListType type_ = GcuSupportListType::NONE;
static std::set<std::string> support_ops_;

void SetGcuEagerSupportOps(const GcuSupportListType type,
                           const std::set<std::string> support_ops) {
  type_ = type;
  support_ops_ = support_ops;
  std::ostringstream os;
  os << "gcu support ops ";
  if (type == platform::GcuSupportListType::NONE) {
    os << "type: none;";
  } else {
    os << "type: "
       << ((type == platform::GcuSupportListType::WHITE) ? "white" : "black")
       << "; ";
    os << "op_list:[ ";
    for (auto op_name : support_ops) {
      os << op_name << "; ";
    }
    os << "]\n";
  }
  VLOG(3) << os.str();
  return;
}
void GetGcuEagerSupportOps(GcuSupportListType &type,              // NOLINT
                           std::set<std::string> &support_ops) {  // NOLINT
  type = type_;
  support_ops = support_ops_;
  return;
}

}  // namespace platform
}  // namespace paddle
