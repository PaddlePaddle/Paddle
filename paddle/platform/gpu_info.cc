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
      cudaGetDeviceCount(&count),
      "cudaGetDeviceCount failed in paddle::platform::GetCUDADeviceCount");
  return count;
}

int GetCurrentDeviceId() {
  int device_id;
  PADDLE_ENFORCE(
      cudaGetDevice(&device_id),
      "cudaGetDevice failed in paddle::platform::GetCurrentDeviceId");
  return device_id;
}

void SetDeviceId(int id) {
  // TODO(qijun): find a better way to cache the cuda device count
  PADDLE_ENFORCE_LT(id, GetCUDADeviceCount(), "id must less than GPU count");
  PADDLE_ENFORCE(cudaSetDevice(id),
                 "cudaSetDevice failed in paddle::platform::SetDeviceId");
}

void GpuMemoryUsage(size_t &available, size_t &total) {
  PADDLE_ENFORCE(cudaMemGetInfo(&available, &total),
                 "cudaMemGetInfo failed in paddle::platform::GetMemoryUsage");
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
  VLOG(10) << "GPU Usage " << available / 1024 / 1024 << "M/"
           << total / 1024 / 1024 << "M";
  size_t reserving = static_cast<size_t>(0.05 * total);
  // If available less than minimum chunk size, no usable memory exists.
  available =
      std::min(std::max(available, GpuMinChunkSize()) - GpuMinChunkSize(),
               total - reserving);

  // Reserving the rest memory for page tables, etc.

  size_t allocating = static_cast<size_t>(FLAGS_fraction_of_gpu_memory_to_use *
                                          (total - reserving));

  PADDLE_ENFORCE_LE(allocating, available);

  return allocating;
}

void GpuMemcpyAsync(void *dst, const void *src, size_t count,
                    enum cudaMemcpyKind kind, cudaStream_t stream) {
  PADDLE_ENFORCE(cudaMemcpyAsync(dst, src, count, kind, stream),
                 "cudaMemcpyAsync failed in paddle::platform::GpuMemcpyAsync");
}

void GpuMemcpyPeer(void *dst, int dst_device, const void *src, int src_device,
                   size_t count, cudaStream_t stream) {
  PADDLE_ENFORCE(
      cudaMemcpyPeerAsync(dst, dst_device, src, src_device, count, stream),
      "cudaMemcpyPeerAsync failed in paddle::platform::GpuMemcpyPeer");
}

void GpuMemsetAsync(void *dst, int value, size_t count, cudaStream_t stream) {
  PADDLE_ENFORCE(cudaMemsetAsync(dst, value, count, stream),
                 "cudaMemsetAsync failed in paddle::platform::GpuMemsetAsync");
}
}  // namespace platform
}  // namespace paddle
