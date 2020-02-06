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
#include <sstream>
#include <string>

#include "gflags/gflags.h"
#include "paddle/fluid/platform/cuda_error.pb.h"
#include "paddle/fluid/platform/enforce.h"
#include "paddle/fluid/string/split.h"

DECLARE_double(fraction_of_gpu_memory_to_use);
DECLARE_uint64(initial_gpu_memory_in_mb);
DECLARE_uint64(reallocate_gpu_memory_in_mb);
DECLARE_bool(enable_cublas_tensor_op_math);
DECLARE_string(selected_gpus);

constexpr static float fraction_reserve_gpu_memory = 0.05f;

namespace paddle {
namespace platform {

/* Here is a very simple CUDA “pro tip”: cudaDeviceGetAttribute() is a much
faster way to query device properties. You can see details in
https://devblogs.nvidia.com/cuda-pro-tip-the-fast-way-to-query-device-properties/
*/

inline std::string GetCudaErrorWebsite(int32_t cuda_version) {
  switch (cuda_version) {
    case 100:
      return "https://docs.nvidia.com/cuda/archive/10.0/cuda-runtime-api/"
             "group__CUDART__TYPES.html"
             "#group__CUDART__TYPES_1g3f51e3575c2178246db0a94a430e0038";
      break;
    case 90:
      return "https://docs.nvidia.com/cuda/archive/9.0/cuda-runtime-api/"
             "group__CUDART__TYPES.html#group__CUDART__TYPES_"
             "1g3f51e3575c2178246db0a94a430e0038";
      break;
    default:
      return "https://docs.nvidia.com/cuda/cuda-runtime-api/"
             "group__CUDART__TYPES.html"
             "#group__CUDART__TYPES_1g3f51e3575c2178246db0a94a430e0038";
      break;
  }
}

std::string GetCudaErrorMessage(int32_t errorcode) {
#if CUDA_VERSION == 10000
  int32_t cuda_version = 100;
#elif CUDA_VERSION >= 9000
  int32_t cuda_version = 90;
#else
  int32_t cuda_version = 0;
#endif
  platform::proto::cudaerrorDesc cudaerror;
  std::ifstream fin("../include/tools/errormessage/data.pb",
                    std::ios::in | std::ios::binary);
  PADDLE_ENFORCE_EQ(
      cudaerror.ParseFromIstream(&fin), true,
      platform::errors::Unavailable(
          "Failed to parse errormessage data from "
          "file[Paddle/tools/errormessage/data.pb], "
          "Please generate data.pb by Paddle/tools/start.sh. And re-compile "
          "Paddle!\n"
          "Usage: bash start.sh <cuda-version>(optional) <url>(optional)\n"));
  VLOG(10)
      << "Parse errormessage data from file[Paddle/tools/errormessage/data.pb]"
      << "cuda_version: " << cuda_version << "errorcode: " << errorcode;
  std::ostringstream sout;
  for (int i = 0; i < cudaerror.allmessages_size(); ++i) {
    if (cuda_version == cudaerror.allmessages(i).version()) {
      for (int j = 0; j < cudaerror.allmessages(i).messages_size(); ++j) {
        if (errorcode == cudaerror.allmessages(i).messages(j).errorcode()) {
          VLOG(10) << "Cuda_version: " << cuda_version << "ErrorCode: "
                   << cudaerror.allmessages(i).messages(j).errorcode() << " "
                   << " ErrorMessage: "
                   << cudaerror.allmessages(i).messages(j).errormessage()
                   << " CudaErrorWebsite: "
                   << GetCudaErrorWebsite(cuda_version);
          sout << cudaerror.allmessages(i).messages(j).errormessage()
               << "\nIn summary, CUDA runtime error (" << errorcode << ") ";
          return sout.str();
        }
      }
    }
  }
  sout << "Please check the error message on website: "
       << GetCudaErrorWebsite(cuda_version) << "."
       << "CUDA runtime error (" << errorcode << ") ";
  return sout.str();
}

static int GetCUDADeviceCountImpl() {
  int driverVersion = 0;
  cudaError_t status = cudaDriverGetVersion(&driverVersion);

  if (!(status == cudaSuccess && driverVersion != 0)) {
    // No GPU driver
    VLOG(2) << "GPU Driver Version can't be detected. No GPU driver!";
    return 0;
  }

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
  PADDLE_ENFORCE_CUDA_SUCCESS(
      error_code,
      platform::errors::External("cudaGetDeviceCount failed in "
                                 "paddle::platform::GetCUDADeviceCountImpl. %s",
                                 GetCudaErrorMessage(error_code)));
  return count;
}

int GetCUDADeviceCount() {
  static auto dev_cnt = GetCUDADeviceCountImpl();
  return dev_cnt;
}

int GetCUDAComputeCapability(int id) {
  PADDLE_ENFORCE_LT(id, GetCUDADeviceCount(),
                    platform::errors::InvalidArgument(
                        "Device id must be less than GPU count, "
                        "but received id is: %d. GPU count is: %d.",
                        id, GetCUDADeviceCount()));
  int major, minor;

  auto major_error_code =
      cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, id);
  auto minor_error_code =
      cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, id);
  PADDLE_ENFORCE_CUDA_SUCCESS(
      major_error_code, platform::errors::External(
                            "cudaDevAttrComputeCapabilityMajor failed in "
                            "paddle::platform::GetCUDAComputeCapability. %s",
                            GetCudaErrorMessage(major_error_code)));
  PADDLE_ENFORCE_CUDA_SUCCESS(
      minor_error_code, platform::errors::External(
                            "cudaDevAttrComputeCapabilityMinor failed in "
                            "paddle::platform::GetCUDAComputeCapability. %s",
                            GetCudaErrorMessage(minor_error_code)));
  return major * 10 + minor;
}

dim3 GetGpuMaxGridDimSize(int id) {
  PADDLE_ENFORCE_LT(id, GetCUDADeviceCount(),
                    platform::errors::InvalidArgument(
                        "Device id must be less than GPU count, "
                        "but received id is: %d. GPU count is: %d.",
                        id, GetCUDADeviceCount()));
  dim3 ret;
  int size;
  auto error_code_x = cudaDeviceGetAttribute(&size, cudaDevAttrMaxGridDimX, id);
  PADDLE_ENFORCE_CUDA_SUCCESS(
      error_code_x,
      platform::errors::External("cudaDevAttrMaxGridDimX failed in "
                                 "paddle::platform::GetGpuMaxGridDimSize. %s",
                                 GetCudaErrorMessage(error_code_x)));
  ret.x = size;

  auto error_code_y = cudaDeviceGetAttribute(&size, cudaDevAttrMaxGridDimY, id);
  PADDLE_ENFORCE_CUDA_SUCCESS(
      error_code_y,
      platform::errors::External("cudaDevAttrMaxGridDimY failed in "
                                 "paddle::platform::GetGpuMaxGridDimSize. %s",
                                 GetCudaErrorMessage(error_code_y)));
  ret.y = size;

  auto error_code_z = cudaDeviceGetAttribute(&size, cudaDevAttrMaxGridDimZ, id);
  PADDLE_ENFORCE_CUDA_SUCCESS(
      error_code_z,
      platform::errors::External("cudaDevAttrMaxGridDimZ failed in "
                                 "paddle::platform::GetGpuMaxGridDimSize. %s",
                                 GetCudaErrorMessage(error_code_z)));
  ret.z = size;
  return ret;
}

int GetCUDARuntimeVersion(int id) {
  PADDLE_ENFORCE_LT(id, GetCUDADeviceCount(),
                    platform::errors::InvalidArgument(
                        "Device id must be less than GPU count, "
                        "but received id is: %d. GPU count is: %d.",
                        id, GetCUDADeviceCount()));
  int runtime_version = 0;
  auto error_code = cudaRuntimeGetVersion(&runtime_version);
  PADDLE_ENFORCE_CUDA_SUCCESS(
      error_code,
      platform::errors::External("cudaRuntimeGetVersion failed in "
                                 "paddle::platform::GetCUDARuntimeVersion. %s",
                                 GetCudaErrorMessage(error_code)));
  return runtime_version;
}

int GetCUDADriverVersion(int id) {
  PADDLE_ENFORCE_LT(id, GetCUDADeviceCount(),
                    platform::errors::InvalidArgument(
                        "Device id must be less than GPU count, "
                        "but received id is: %d. GPU count is: %d.",
                        id, GetCUDADeviceCount()));
  int driver_version = 0;
  auto error_code = cudaDriverGetVersion(&driver_version);
  PADDLE_ENFORCE_CUDA_SUCCESS(
      error_code,
      platform::errors::External("cudaDriverGetVersion failed in "
                                 "paddle::platform::GetCUDADriverVersion. %s",
                                 GetCudaErrorMessage(error_code)));
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
  PADDLE_ENFORCE_LT(id, GetCUDADeviceCount(),
                    platform::errors::InvalidArgument(
                        "Device id must be less than GPU count, "
                        "but received id is: %d. GPU count is: %d.",
                        id, GetCUDADeviceCount()));
  int count;
  auto error_code =
      cudaDeviceGetAttribute(&count, cudaDevAttrMultiProcessorCount, id);
  PADDLE_ENFORCE_CUDA_SUCCESS(
      error_code,
      platform::errors::External("cudaDeviceGetAttribute failed in "
                                 "paddle::platform::GetCUDAMultiProcess. %s",
                                 GetCudaErrorMessage(error_code)));
  return count;
}

int GetCUDAMaxThreadsPerMultiProcessor(int id) {
  PADDLE_ENFORCE_LT(id, GetCUDADeviceCount(),
                    platform::errors::InvalidArgument(
                        "Device id must be less than GPU count, "
                        "but received id is: %d. GPU count is: %d.",
                        id, GetCUDADeviceCount()));
  int count;
  auto error_code = cudaDeviceGetAttribute(
      &count, cudaDevAttrMaxThreadsPerMultiProcessor, id);
  PADDLE_ENFORCE_CUDA_SUCCESS(
      error_code,
      platform::errors::External(
          "cudaDeviceGetAttribute failed in "
          "paddle::platform::GetCUDAMaxThreadsPerMultiProcessor. %s",
          GetCudaErrorMessage(error_code)));
  return count;
}

int GetCUDAMaxThreadsPerBlock(int id) {
  PADDLE_ENFORCE_LT(id, GetCUDADeviceCount(),
                    platform::errors::InvalidArgument(
                        "Device id must be less than GPU count, "
                        "but received id is: %d. GPU count is: %d.",
                        id, GetCUDADeviceCount()));
  int count;
  auto error_code =
      cudaDeviceGetAttribute(&count, cudaDevAttrMaxThreadsPerBlock, id);
  PADDLE_ENFORCE_CUDA_SUCCESS(
      error_code, platform::errors::External(
                      "cudaDeviceGetAttribute failed in "
                      "paddle::platform::GetCUDAMaxThreadsPerBlock. %s",
                      GetCudaErrorMessage(error_code)));
  return count;
}

int GetCurrentDeviceId() {
  int device_id;
  auto error_code = cudaGetDevice(&device_id);
  PADDLE_ENFORCE_CUDA_SUCCESS(
      error_code, platform::errors::External(
                      "cudaDeviceGetAttribute failed in "
                      "paddle::platform::GetCUDAMaxThreadsPerBlock. %s",
                      GetCudaErrorMessage(error_code)));
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
  PADDLE_ENFORCE_LT(id, GetCUDADeviceCount(),
                    platform::errors::InvalidArgument(
                        "Device id must be less than GPU count, "
                        "but received id is: %d. GPU count is: %d.",
                        id, GetCUDADeviceCount()));
  auto error_code = cudaSetDevice(id);
  PADDLE_ENFORCE_CUDA_SUCCESS(
      error_code,
      platform::errors::External("cudaSetDevice failed in "
                                 "paddle::platform::SetDeviceId. %s",
                                 GetCudaErrorMessage(error_code)));
}

void GpuMemoryUsage(size_t *available, size_t *total) {
  auto error_code = cudaMemGetInfo(available, total);
  PADDLE_ENFORCE_CUDA_SUCCESS(
      error_code,
      platform::errors::External("cudaMemGetInfo failed in "
                                 "paddle::platform::GetMemoryUsage. %s",
                                 GetCudaErrorMessage(error_code)));
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
  PADDLE_ENFORCE_CUDA_SUCCESS(
      error_code,
      platform::errors::External(
          "cudaMemcpyAsync failed in paddle::platform::GpuMemcpyAsync "
          "(%p -> %p, length: %d). %s",
          src, dst, static_cast<int>(count), GetCudaErrorMessage(error_code)));
}

void GpuMemcpySync(void *dst, const void *src, size_t count,
                   enum cudaMemcpyKind kind) {
  auto error_code = cudaMemcpy(dst, src, count, kind);
  PADDLE_ENFORCE_CUDA_SUCCESS(
      error_code,
      platform::errors::External(
          "cudaMemcpy failed in paddle::platform::GpuMemcpySync "
          "(%p -> %p, length: %d). %s",
          src, dst, static_cast<int>(count), GetCudaErrorMessage(error_code)));
}

void GpuMemcpyPeerAsync(void *dst, int dst_device, const void *src,
                        int src_device, size_t count, cudaStream_t stream) {
  auto error_code =
      cudaMemcpyPeerAsync(dst, dst_device, src, src_device, count, stream);
  PADDLE_ENFORCE_CUDA_SUCCESS(
      error_code,
      platform::errors::External("cudaMemcpyPeerAsync failed in "
                                 "paddle::platform::GpuMemcpyPeerAsync. %s",
                                 GetCudaErrorMessage(error_code)));
}

void GpuMemcpyPeerSync(void *dst, int dst_device, const void *src,
                       int src_device, size_t count) {
  auto error_code = cudaMemcpyPeer(dst, dst_device, src, src_device, count);
  PADDLE_ENFORCE_CUDA_SUCCESS(
      error_code,
      platform::errors::External("cudaMemcpyPeer failed in "
                                 "paddle::platform::GpuMemcpyPeerSync. %s",
                                 GetCudaErrorMessage(error_code)));
}

void GpuMemsetAsync(void *dst, int value, size_t count, cudaStream_t stream) {
  auto error_code = cudaMemsetAsync(dst, value, count, stream);
  PADDLE_ENFORCE_CUDA_SUCCESS(
      error_code,
      platform::errors::External("cudaMemsetAsync failed in "
                                 "paddle::platform::GpuMemsetAsync. %s",
                                 GetCudaErrorMessage(error_code)));
}

void GpuStreamSync(cudaStream_t stream) {
  auto error_code = cudaStreamSynchronize(stream);
  PADDLE_ENFORCE_CUDA_SUCCESS(
      error_code,
      platform::errors::External("cudaStreamSynchronize failed in "
                                 "paddle::platform::GpuStreamSync. %s.",
                                 GetCudaErrorMessage(error_code)));
}

void RaiseNonOutOfMemoryError(cudaError_t *status) {
  if (*status == cudaErrorMemoryAllocation) {
    *status = cudaSuccess;
  }
  PADDLE_ENFORCE_CUDA_SUCCESS(*status);

  *status = cudaGetLastError();
  if (*status == cudaErrorMemoryAllocation) {
    *status = cudaSuccess;
  }
  PADDLE_ENFORCE_CUDA_SUCCESS(*status);
}

}  // namespace platform
}  // namespace paddle
