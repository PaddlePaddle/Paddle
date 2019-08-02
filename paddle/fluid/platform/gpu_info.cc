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
    "trunk. If the trunk doesn't have enough gpu memory, additional "
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

inline std::string CudaErrorWebsite() {
  return "Please see detail in https://docs.nvidia.com/cuda/cuda-runtime-api"
         "/group__CUDART__TYPES.html#group__CUDART__TYPES_1g3f51e3575c217824"
         "6db0a94a430e0038";
}

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
  auto error_code = cudaGetDeviceCount(&count);
  PADDLE_ENFORCE(
      error_code,
      "cudaGetDeviceCount failed in "
      "paddle::platform::GetCUDADeviceCountImpl, error code : %d, %s",
      error_code, CudaErrorWebsite());
  return count;
}

int GetCUDADeviceCount() {
  static auto dev_cnt = GetCUDADeviceCountImpl();
  return dev_cnt;
}

int GetCUDAComputeCapability(int id) {
  PADDLE_ENFORCE_LT(id, GetCUDADeviceCount(), "id must less than GPU count");
  cudaDeviceProp device_prop;
  auto error_code = cudaGetDeviceProperties(&device_prop, id);
  PADDLE_ENFORCE(
      error_code,
      "cudaGetDeviceProperties failed in "
      "paddle::platform::GetCUDAComputeCapability, error code : %d, %s",
      error_code, CudaErrorWebsite());
  return device_prop.major * 10 + device_prop.minor;
}

int GetCUDARuntimeVersion(int id) {
  PADDLE_ENFORCE_LT(id, GetCUDADeviceCount(), "id must less than GPU count");
  int runtime_version = 0;
  auto error_code = cudaRuntimeGetVersion(&runtime_version);
  PADDLE_ENFORCE(error_code,
                 "cudaRuntimeGetVersion failed in "
                 "paddle::platform::GetCUDARuntimeVersion, error code : %d, %s",
                 error_code, CudaErrorWebsite());
  return runtime_version;
}

int GetCUDADriverVersion(int id) {
  PADDLE_ENFORCE_LT(id, GetCUDADeviceCount(), "id must less than GPU count");
  int driver_version = 0;
  auto error_code = cudaDriverGetVersion(&driver_version);
  PADDLE_ENFORCE(error_code,
                 "cudaDriverGetVersion failed in "
                 "paddle::platform::GetCUDADriverVersion, error code : %d, %s",
                 error_code, CudaErrorWebsite());
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
  auto error_code =
      cudaDeviceGetAttribute(&count, cudaDevAttrMultiProcessorCount, id);
  PADDLE_ENFORCE(error_code,
                 "cudaDeviceGetAttribute failed in "
                 "paddle::platform::GetCUDAMultiProcess, error code : %d, %s",
                 error_code, CudaErrorWebsite());
  return count;
}

int GetCUDAMaxThreadsPerMultiProcessor(int id) {
  PADDLE_ENFORCE_LT(id, GetCUDADeviceCount(), "id must less than GPU count");
  int count;
  auto error_code = cudaDeviceGetAttribute(
      &count, cudaDevAttrMaxThreadsPerMultiProcessor, id);
  PADDLE_ENFORCE(
      error_code,
      "cudaDeviceGetAttribute failed in paddle::"
      "platform::GetCUDAMaxThreadsPerMultiProcessor, error code : %d, %s",
      error_code, CudaErrorWebsite());
  return count;
}

int GetCurrentDeviceId() {
  int device_id;
  auto error_code = cudaGetDevice(&device_id);
  PADDLE_ENFORCE(error_code,
                 "cudaGetDevice failed in "
                 "paddle::platform::GetCurrentDeviceId, error code : %d, %s",
                 error_code, CudaErrorWebsite());
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
  auto error_code = cudaSetDevice(id);
  PADDLE_ENFORCE(error_code,
                 "cudaSetDevice failed in "
                 "paddle::platform::SetDeviced, error code : %d, %s",
                 error_code, CudaErrorWebsite());
}

void GpuMemoryUsage(size_t *available, size_t *total) {
  auto error_code = cudaMemGetInfo(available, total);
  PADDLE_ENFORCE(error_code,
                 "cudaMemGetInfo failed in "
                 "paddle::platform::GetMemoryUsage, error code : %d, %s",
                 error_code, CudaErrorWebsite());
}

size_t GpuAvailableMemToAlloc() {
  size_t total = 0;
  size_t available = 0;
  GpuMemoryUsage(&available, &total);
  size_t reserving =
      static_cast<size_t>(fraction_reserve_gpu_memory * available);
  // If available size is less than minimum chunk size, no usable memory exists
  size_t available_to_alloc = available - reserving;
  size_t min_chunk_size = GpuMinChunkSize();
  if (available_to_alloc < min_chunk_size) {
    available_to_alloc = 0;
  }
  VLOG(10) << "GPU usage " << (available >> 20) << "M/" << (total >> 20)
           << "M, " << (available_to_alloc >> 20) << "M available to allocate";
  return available_to_alloc;
}

size_t GpuMaxAllocSize() {
  return std::max(GpuInitAllocSize(), GpuReallocSize());
}

static size_t GpuAllocSize(bool realloc) {
  size_t available_to_alloc = GpuAvailableMemToAlloc();
  PADDLE_ENFORCE_GT(available_to_alloc, 0, "No enough available GPU memory");
  // If FLAGS_initial_gpu_memory_in_mb is 0, then initial memory will be
  // allocated by fraction
  size_t flag_mb = realloc ? FLAGS_reallocate_gpu_memory_in_mb
                           : FLAGS_initial_gpu_memory_in_mb;
  size_t alloc_bytes =
      (flag_mb > 0ul ? flag_mb << 20 : available_to_alloc *
                                           FLAGS_fraction_of_gpu_memory_to_use);
  PADDLE_ENFORCE_GE(available_to_alloc, alloc_bytes,
                    "No enough available GPU memory");
  VLOG(10) << "Alloc size is " << (alloc_bytes >> 20)
           << " MiB, is it Re-alloc: " << realloc;
  return alloc_bytes;
}

size_t GpuInitAllocSize() { return GpuAllocSize(/* realloc = */ false); }

size_t GpuReallocSize() { return GpuAllocSize(/* realloc = */ true); }

size_t GpuMinChunkSize() {
  // Allow to allocate the minimum chunk size is 256 bytes.
  return 1 << 8;
}

size_t GpuMaxChunkSize() {
  size_t max_chunk_size = GpuMaxAllocSize();
  VLOG(10) << "Max chunk size " << (max_chunk_size >> 20) << "M";
  return max_chunk_size;
}

void GpuMemcpyAsync(void *dst, const void *src, size_t count,
                    enum cudaMemcpyKind kind, cudaStream_t stream) {
  auto error_code = cudaMemcpyAsync(dst, src, count, kind, stream);
  PADDLE_ENFORCE(error_code,
                 "cudaMemcpyAsync failed in paddle::platform::GpuMemcpyAsync "
                 "(%p -> %p, length: %d) error code : %d, %s",
                 src, dst, static_cast<int>(count), error_code,
                 CudaErrorWebsite());
}

void GpuMemcpySync(void *dst, const void *src, size_t count,
                   enum cudaMemcpyKind kind) {
  auto error_code = cudaMemcpy(dst, src, count, kind);
  PADDLE_ENFORCE(error_code,
                 "cudaMemcpy failed in paddle::platform::GpuMemcpySync "
                 "(%p -> %p, length: %d) error code : %d, %s",
                 src, dst, static_cast<int>(count), error_code,
                 CudaErrorWebsite());
}

void GpuMemcpyPeerAsync(void *dst, int dst_device, const void *src,
                        int src_device, size_t count, cudaStream_t stream) {
  auto error_code =
      cudaMemcpyPeerAsync(dst, dst_device, src, src_device, count, stream);
  PADDLE_ENFORCE(
      error_code,
      "cudaMemcpyPeerAsync failed in paddle::platform::GpuMemcpyPeerAsync "
      "error code : %d, %s",
      error_code, CudaErrorWebsite());
}

void GpuMemcpyPeerSync(void *dst, int dst_device, const void *src,
                       int src_device, size_t count) {
  auto error_code = cudaMemcpyPeer(dst, dst_device, src, src_device, count);
  PADDLE_ENFORCE(error_code,
                 "cudaMemcpyPeer failed in paddle::platform::GpuMemcpyPeerSync "
                 "error code : %d, %s",
                 error_code, CudaErrorWebsite());
}

void GpuMemsetAsync(void *dst, int value, size_t count, cudaStream_t stream) {
  auto error_code = cudaMemsetAsync(dst, value, count, stream);
  PADDLE_ENFORCE(error_code,
                 "cudaMemsetAsync failed in paddle::platform::GpuMemsetAsync "
                 "error code : %d, %s",
                 error_code, CudaErrorWebsite());
}
}  // namespace platform
}  // namespace paddle
