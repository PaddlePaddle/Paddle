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
#include <cstdlib>
#include <string>

#include "gflags/gflags.h"
#include "paddle/fluid/platform/enforce.h"
#include "paddle/fluid/string/split.h"

#ifndef _WIN32
constexpr static float fraction_of_gpu_memory_to_use = 0.92f;
#else
// fraction_of_gpu_memory_to_use cannot be too high on windows,
// since the win32 graphic sub-system can occupy some GPU memory
// which may lead to insufficient memory left for paddle
constexpr static float fraction_of_gpu_memory_to_use = 0.5f;
#endif

constexpr static float fraction_reserve_gpu_memory = 0.05f;

DEFINE_double(fraction_of_gpu_memory_to_use, fraction_of_gpu_memory_to_use,
              "Allocate a trunk of gpu memory that is this fraction of the "
              "total gpu memory size. Future memory usage will be allocated "
              "from the trunk. If the trunk doesn't have enough gpu memory, "
              "additional trunks of the same size will be requested from gpu "
              "until the gpu has no memory left for another trunk.");

DEFINE_uint64(
    initial_gpu_memory_in_mb, 0ul,
    "Allocate a trunk of gpu memory whose byte size is specified by "
    "the flag. Future memory usage will be allocated from the "
    "truck. If the trunk doesn't have enough gpu memory, additional "
    "trunks of the gpu memory will be requested from gpu with size "
    "specified by FLAGS_reallocate_gpu_memory_in_mb until the gpu has "
    "no memory left for the additional trunk. Note: if you set this "
    "flag, the memory size set by "
    "FLAGS_fraction_of_gpu_memory_to_use will be overrided by this "
    "flag. If you don't set this flag, PaddlePaddle will use "
    "FLAGS_fraction_of_gpu_memory_to_use to allocate gpu memory");

DEFINE_uint64(reallocate_gpu_memory_in_mb, 0ul,
              "If this flag is set, Paddle will reallocate the gpu memory with "
              "size specified by this flag. Else Paddle will reallocate by "
              "FLAGS_fraction_of_gpu_memory_to_use");

DEFINE_bool(
    enable_cublas_tensor_op_math, false,
    "The enable_cublas_tensor_op_math indicate whether to use Tensor Core, "
    "but it may loss precision. Currently, There are two CUDA libraries that"
    " use Tensor Cores, cuBLAS and cuDNN. cuBLAS uses Tensor Cores to speed up"
    " GEMM computations(the matrices must be either half precision or single "
    "precision); cuDNN uses Tensor Cores to speed up both convolutions(the "
    "input and output must be half precision) and recurrent neural networks "
    "(RNNs).");

DEFINE_string(selected_gpus, "",
              "A list of device ids separated by comma, like: 0,1,2,3. "
              "This option is useful when doing multi process training and "
              "each process have only one device (GPU). If you want to use "
              "all visible devices, set this to empty string. NOTE: the "
              "reason of doing this is that we want to use P2P communication"
              "between GPU devices, use CUDA_VISIBLE_DEVICES can only use"
              "share-memory only.");

namespace paddle {
namespace platform {

static int GetCUDADeviceCountImpl() {
  const auto *cuda_visible_devices = std::getenv("CUDA_VISIBLE_DEVICES");
  if (cuda_visible_devices != nullptr) {
    std::string cuda_visible_devices_str(cuda_visible_devices);
    if (std::all_of(cuda_visible_devices_str.begin(),
                    cuda_visible_devices_str.end(),
                    [](char ch) { return ch == ' '; })) {
      VLOG(2) << "CUDA_VISIBLE_DEVICES is set to be empty. No GPU detected.";
      return 0;
    }
  }

  int count;
  PADDLE_ENFORCE(
      cudaGetDeviceCount(&count),
      "cudaGetDeviceCount failed in paddle::platform::GetCUDADeviceCount");
  return count;
}

int GetCUDADeviceCount() {
  static auto dev_cnt = GetCUDADeviceCountImpl();
  return dev_cnt;
}

int GetCUDAComputeCapability(int id) {
  PADDLE_ENFORCE_LT(id, GetCUDADeviceCount(), "id must less than GPU count");
  cudaDeviceProp device_prop;
  PADDLE_ENFORCE(cudaGetDeviceProperties(&device_prop, id),
                 "cudaGetDeviceProperties failed in "
                 "paddle::platform::GetCUDAComputeCapability");
  return device_prop.major * 10 + device_prop.minor;
}

int GetCUDARuntimeVersion(int id) {
  PADDLE_ENFORCE_LT(id, GetCUDADeviceCount(), "id must less than GPU count");
  int runtime_version = 0;
  PADDLE_ENFORCE(cudaRuntimeGetVersion(&runtime_version),
                 "cudaRuntimeGetVersion failed in "
                 "paddle::platform::cudaRuntimeGetVersion");
  return runtime_version;
}

int GetCUDADriverVersion(int id) {
  PADDLE_ENFORCE_LT(id, GetCUDADeviceCount(), "id must less than GPU count");
  int driver_version = 0;
  PADDLE_ENFORCE(cudaDriverGetVersion(&driver_version),
                 "cudaDriverGetVersion failed in "
                 "paddle::platform::GetCUDADriverVersion");
  return driver_version;
}

bool TensorCoreAvailable() {
#if CUDA_VERSION >= 9000
  int device = GetCurrentDeviceId();
  int driver_version = GetCUDAComputeCapability(device);
  return driver_version >= 70;
#else
  return false;
#endif
}

int GetCUDAMultiProcessors(int id) {
  PADDLE_ENFORCE_LT(id, GetCUDADeviceCount(), "id must less than GPU count");
  int count;
  PADDLE_ENFORCE(
      cudaDeviceGetAttribute(&count, cudaDevAttrMultiProcessorCount, id),
      "cudaDeviceGetAttribute failed in "
      "paddle::platform::GetCUDAMultiProcessors");
  return count;
}

int GetCUDAMaxThreadsPerMultiProcessor(int id) {
  PADDLE_ENFORCE_LT(id, GetCUDADeviceCount(), "id must less than GPU count");
  int count;
  PADDLE_ENFORCE(cudaDeviceGetAttribute(
                     &count, cudaDevAttrMaxThreadsPerMultiProcessor, id),
                 "cudaDeviceGetAttribute failed in "
                 "paddle::platform::GetCUDAMaxThreadsPerMultiProcessor");
  return count;
}

int GetCurrentDeviceId() {
  int device_id;
  PADDLE_ENFORCE(
      cudaGetDevice(&device_id),
      "cudaGetDevice failed in paddle::platform::GetCurrentDeviceId");
  return device_id;
}

//! Get a list of device ids from environment variable or use all.
std::vector<int> GetSelectedDevices() {
  // use user specified GPUs in single-node multi-process mode.
  std::vector<int> devices;
  if (!FLAGS_selected_gpus.empty()) {
    auto devices_str = paddle::string::Split(FLAGS_selected_gpus, ',');
    for (auto id : devices_str) {
      devices.push_back(atoi(id.c_str()));
    }
  } else {
    int count = GetCUDADeviceCount();
    for (int i = 0; i < count; ++i) {
      devices.push_back(i);
    }
  }
  return devices;
}

void SetDeviceId(int id) {
  // TODO(qijun): find a better way to cache the cuda device count
  PADDLE_ENFORCE_LT(id, GetCUDADeviceCount(), "id must less than GPU count");
  PADDLE_ENFORCE(cudaSetDevice(id),
                 "cudaSetDevice failed in paddle::platform::SetDeviceId");
}

void GpuMemoryUsage(size_t *available, size_t *total) {
  PADDLE_ENFORCE(cudaMemGetInfo(available, total),
                 "cudaMemGetInfo failed in paddle::platform::GetMemoryUsage");
}

size_t GpuMaxAllocSize() {
  return std::max(GpuInitAllocSize(), GpuReallocSize());
}

size_t GpuInitAllocSize() {
  if (FLAGS_initial_gpu_memory_in_mb > 0ul) {
    // Initial memory will be allocated by FLAGS_initial_gpu_memory_in_mb
    return static_cast<size_t>(FLAGS_initial_gpu_memory_in_mb << 20);
  }

  // FLAGS_initial_gpu_memory_in_mb is 0, initial memory will be allocated by
  // fraction
  size_t total = 0;
  size_t available = 0;

  GpuMemoryUsage(&available, &total);
  size_t reserving = static_cast<size_t>(fraction_reserve_gpu_memory * total);

  return static_cast<size_t>((total - reserving) *
                             FLAGS_fraction_of_gpu_memory_to_use);
}

size_t GpuReallocSize() {
  if (FLAGS_reallocate_gpu_memory_in_mb > 0ul) {
    // Additional memory will be allocated by FLAGS_reallocate_gpu_memory_in_mb
    return static_cast<size_t>(FLAGS_reallocate_gpu_memory_in_mb << 20);
  }

  // FLAGS_reallocate_gpu_memory_in_mb is 0, additional memory will be allocated
  // by fraction
  size_t total = 0;
  size_t available = 0;

  GpuMemoryUsage(&available, &total);
  size_t reserving = static_cast<size_t>(fraction_reserve_gpu_memory * total);

  return static_cast<size_t>((total - reserving) *
                             FLAGS_fraction_of_gpu_memory_to_use);
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
  size_t reserving = static_cast<size_t>(fraction_reserve_gpu_memory * total);
  // If available less than minimum chunk size, no usable memory exists.
  available =
      std::min(std::max(available, GpuMinChunkSize()) - GpuMinChunkSize(),
               total - reserving);

  size_t allocating = GpuMaxAllocSize();

  PADDLE_ENFORCE_LE(allocating, available,
                    "Insufficient GPU memory to allocation.");

  return allocating;
}

void GpuMemcpyAsync(void *dst, const void *src, size_t count,
                    enum cudaMemcpyKind kind, cudaStream_t stream) {
  PADDLE_ENFORCE(cudaMemcpyAsync(dst, src, count, kind, stream),
                 "cudaMemcpyAsync failed in paddle::platform::GpuMemcpyAsync "
                 "(%p -> %p, length: %d)",
                 src, dst, static_cast<int>(count));
}

void GpuMemcpySync(void *dst, const void *src, size_t count,
                   enum cudaMemcpyKind kind) {
  PADDLE_ENFORCE(cudaMemcpy(dst, src, count, kind),
                 "cudaMemcpy failed in paddle::platform::GpuMemcpySync (%p -> "
                 "%p, length: %d)",
                 src, dst, static_cast<int>(count));
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
