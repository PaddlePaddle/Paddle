/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserve.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/platform/gpu_info.h"

#include "gflags/gflags.h"

#include "paddle/platform/enforce.h"

DEFINE_double(fraction_of_gpu_memory_to_use, 0.92,
              "Default use 92% of GPU memory for PaddlePaddle,"
              "reserve the rest for page tables, etc");

namespace paddle {
namespace platform {

int GetCUDADeviceCount() {
  int count;
  PADDLE_ENFORCE(
      hipGetDeviceCount(&count),
      "hipGetDeviceCount failed in paddle::platform::GetCUDADeviceCount");
  return count;
}

int GetCurrentDeviceId() {
  int device_id;
  PADDLE_ENFORCE(
      hipGetDevice(&device_id),
      "hipGetDevice failed in paddle::platform::GetCurrentDeviceId");
  return device_id;
}

void SetDeviceId(int id) {
  // TODO(qijun): find a better way to cache the cuda device count
  PADDLE_ENFORCE_LT(id, GetCUDADeviceCount(), "id must less than GPU count");
  PADDLE_ENFORCE(hipSetDevice(id),
                 "hipSetDevice failed in paddle::platform::SetDeviceId");
}

void GpuMemoryUsage(size_t &available, size_t &total) {
  PADDLE_ENFORCE(hipMemGetInfo(&available, &total),
                 "hipMemGetInfo failed in paddle::platform::GetMemoryUsage");
}

size_t GpuMaxAllocSize() {
  size_t total = 0;
  size_t available = 0;

  GpuMemoryUsage(available, total);

  // Reserve the rest for page tables, etc.
  return static_cast<size_t>(total * FLAGS_fraction_of_gpu_memory_to_use);
}

size_t GpuMinChunkSize() {
  // Allow to allocate the minimum chunk size is 256 bytes.
  return 1 << 8;
}

size_t GpuMaxChunkSize() {
  size_t total = 0;
  size_t available = 0;

  GpuMemoryUsage(available, total);

  // Reserving the rest memory for page tables, etc.
  size_t reserving = 0.05 * total;

  // If available less than minimum chunk size, no usable memory exists.
  available =
      std::max(std::max(available, GpuMinChunkSize()) - GpuMinChunkSize(),
               reserving) -
      reserving;

  size_t allocating = FLAGS_fraction_of_gpu_memory_to_use * total;

  PADDLE_ENFORCE_LT(allocating, available);

  return allocating;
}

void GpuMemcpyAsync(void *dst, const void *src, size_t count,
                    enum hipMemcpyKind kind, hipStream_t stream) {
  PADDLE_ENFORCE(hipMemcpyAsync(dst, src, count, kind, stream),
                 "hipMemcpyAsync failed in paddle::platform::GpuMemcpyAsync");
}

void GpuMemcpySync(void *dst, const void *src, size_t count,
                   enum hipMemcpyKind kind) {
  PADDLE_ENFORCE(hipMemcpy(dst, src, count, kind),
                 "hipMemcpy failed in paddle::platform::GpuMemcpySync");
  // note: hipMemcpy may actually be asynchronous with respect to the caller,
  //       block on stream 0 to make sure the copy has completed
  PADDLE_ENFORCE(
      hipStreamSynchronize(0),
      "hipStreamSynchronize failed in paddle::platform::GpuMemcpySync");
}

void GpuMemcpyPeer(void *dst, int dst_device, const void *src, int src_device,
                   size_t count, hipStream_t stream) {
  PADDLE_ENFORCE(
      hipMemcpyPeerAsync(dst, dst_device, src, src_device, count, stream),
      "hipMemcpyPeerAsync failed in paddle::platform::GpuMemcpyPeer");
}

void GpuMemsetAsync(void *dst, int value, size_t count, hipStream_t stream) {
  PADDLE_ENFORCE(hipMemsetAsync(dst, value, count, stream),
                 "hipMemsetAsync failed in paddle::platform::GpuMemsetAsync");
}
}  // namespace platform
}  // namespace paddle
