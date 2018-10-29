/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/platform/gpu_info.h"

#include <algorithm>

#include "gflags/gflags.h"
#include "paddle/fluid/platform/enforce.h"

DEFINE_double(fraction_of_gpu_memory_to_use, 0.92,
              "Allocate a trunk of gpu memory that is this fraction of the "
              "total gpu memory size. Future memory usage will be allocated "
              "from the trunk. If the trunk doesn't have enough gpu memory, "
              "additional trunks of the same size will be requested from gpu "
              "until the gpu has no memory left for another trunk.");

namespace paddle {
namespace platform {

int GetCUDADeviceCount() {
  int count;
  cudaError_t err = cudaGetDeviceCount(&count);
  PADDLE_ENFORCE(
      err == cudaSuccess,
      "cudaGetDeviceCount failed in paddle::platform::GetCUDADeviceCount: %s",
      cudaGetErrorString(err));
  return count;
}

int GetCUDAComputeCapability(int id) {
  cudaDeviceProp device_prop;
  cudaError_t err = cudaGetDeviceProperties(&device_prop, id);
  PADDLE_ENFORCE(err == cudaSuccess,
                 "cudaGetDeviceProperties failed in "
                 "paddle::platform::GetCUDAComputeCapability: %s",
                 cudaGetErrorString(err));
  return device_prop.major * 10 + device_prop.minor;
}

int GetCUDARuntimeVersion(int id) {
  int runtime_version = 0;
  cudaError_t err = cudaRuntimeGetVersion(&runtime_version);
  PADDLE_ENFORCE(err == cudaSuccess,
                 "cudaRuntimeGetVersion failed in "
                 "paddle::platform::GetCUDARuntimeVersion: %s",
                 cudaGetErrorString(err));
  return runtime_version;
}

int GetCUDADriverVersion(int id) {
  int driver_version = 0;
  cudaError_t err = cudaDriverGetVersion(&driver_version);
  PADDLE_ENFORCE(err == cudaSuccess,
                 "cudaDriverGetVersion failed in "
                 "paddle::platform::GetCUDADriverVersion: %s",
                 cudaGetErrorString(err));
  return driver_version;
}

int GetCUDAMultiProcessors(int id) {
  int count;
  cudaError_t err =
      cudaDeviceGetAttribute(&count, cudaDevAttrMultiProcessorCount, id);
  PADDLE_ENFORCE(err == cudaSuccess,
                 "cudaDeviceGetAttribute failed in "
                 "paddle::platform::GetCUDAMultiProcessors:%s",
                 cudaGetErrorString(err));
  return count;
}

int GetCUDAMaxThreadsPerMultiProcessor(int id) {
  int count;
  cudaError_t err = cudaDeviceGetAttribute(
      &count, cudaDevAttrMaxThreadsPerMultiProcessor, id);
  PADDLE_ENFORCE(err == cudaSuccess,
                 "cudaDeviceGetAttribute failed in "
                 "paddle::platform::GetCUDAMaxThreadsPerMultiProcessor:%s",
                 cudaGetErrorString(err));
  return count;
}

int GetCurrentDeviceId() {
  int device_id;
  cudaError_t err = cudaGetDevice(&device_id);
  PADDLE_ENFORCE(err == cudaSuccess,
                 "cudaGetDevice failed in "
                 "paddle::platform::GetCurrentDeviceId:%s",
                 cudaGetErrorString(err));
  return device_id;
}

void SetDeviceId(int id) {
  cudaError_t err = cudaSetDevice(id);
  PADDLE_ENFORCE(err == cudaSuccess,
                 "cudaSetDevice failed in paddle::platform::SetDeviceId: %s",
                 cudaGetErrorString(err));
}

void GpuMemoryUsage(size_t *available, size_t *total) {
  PADDLE_ENFORCE(cudaMemGetInfo(available, total),
                 "cudaMemGetInfo failed in paddle::platform::GetMemoryUsage");
}

size_t GpuMaxAllocSize() {
  size_t total = 0;
  size_t available = 0;

  GpuMemoryUsage(&available, &total);

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

  GpuMemoryUsage(&available, &total);
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

  PADDLE_ENFORCE_LE(allocating, available,
                    "Insufficient GPU memory to allocation.");

  return allocating;
}

void GpuMemcpyAsync(void *dst, const void *src, size_t count,
                    enum cudaMemcpyKind kind, cudaStream_t stream) {
  PADDLE_ENFORCE(cudaMemcpyAsync(dst, src, count, kind, stream),
                 "cudaMemcpyAsync failed in paddle::platform::GpuMemcpyAsync");
}

void GpuMemcpySync(void *dst, const void *src, size_t count,
                   enum cudaMemcpyKind kind) {
  PADDLE_ENFORCE(cudaMemcpy(dst, src, count, kind),
                 "cudaMemcpy failed in paddle::platform::GpuMemcpySync");
}

void GpuMemcpyPeerAsync(void *dst, int dst_device, const void *src,
                        int src_device, size_t count, cudaStream_t stream) {
  PADDLE_ENFORCE(
      cudaMemcpyPeerAsync(dst, dst_device, src, src_device, count, stream),
      "cudaMemcpyPeerAsync failed in paddle::platform::GpuMemcpyPeerAsync");
}

void GpuMemcpyPeerSync(void *dst, int dst_device, const void *src,
                       int src_device, size_t count) {
  PADDLE_ENFORCE(
      cudaMemcpyPeer(dst, dst_device, src, src_device, count),
      "cudaMemcpyPeer failed in paddle::platform::GpuMemcpyPeerSync");
}

void GpuMemsetAsync(void *dst, int value, size_t count, cudaStream_t stream) {
  PADDLE_ENFORCE(cudaMemsetAsync(dst, value, count, stream),
                 "cudaMemsetAsync failed in paddle::platform::GpuMemsetAsync");
}
}  // namespace platform
}  // namespace paddle
