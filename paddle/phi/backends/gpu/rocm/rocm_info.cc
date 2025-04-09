// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

#include <array>

#include "paddle/phi/backends/gpu/gpu_info.h"

#include "paddle/phi/core/enforce.h"

static std::once_flag g_device_props_size_init_flag;
static std::vector<std::unique_ptr<std::once_flag>> g_device_props_init_flags;
static std::vector<phi::gpuDeviceProp> g_device_props;

namespace phi {
namespace backends {
namespace gpu {

int DnnVersion() {
  if (!dynload::HasCUDNN()) return -1;
  size_t version_major, version_minor, version_patch;
  PADDLE_ENFORCE_GPU_SUCCESS(dynload::miopenGetVersion(
      &version_major, &version_minor, &version_patch));
  return version_major * 100 + version_minor * 10 + version_patch;
}

static int GetGPUDeviceCountImpl() {
  int driverVersion = 0;
  hipError_t status = hipDriverGetVersion(&driverVersion);

  if (!(status == gpuSuccess && driverVersion != 0)) {
    // No GPU driver
    VLOG(2) << "GPU Driver Version can't be detected. No GPU driver!";
    return 0;
  }

  const auto *cuda_visible_devices = std::getenv("HIP_VISIBLE_DEVICES");

  if (cuda_visible_devices != nullptr) {
    std::string cuda_visible_devices_str(cuda_visible_devices);
    if (!cuda_visible_devices_str.empty()) {
      cuda_visible_devices_str.erase(
          0, cuda_visible_devices_str.find_first_not_of('\''));
      cuda_visible_devices_str.erase(
          cuda_visible_devices_str.find_last_not_of('\'') + 1);
      cuda_visible_devices_str.erase(
          0, cuda_visible_devices_str.find_first_not_of('\"'));
      cuda_visible_devices_str.erase(
          cuda_visible_devices_str.find_last_not_of('\"') + 1);
    }
    if (std::all_of(cuda_visible_devices_str.begin(),
                    cuda_visible_devices_str.end(),
                    [](char ch) { return ch == ' '; })) {
      VLOG(2) << "HIP_VISIBLE_DEVICES is set to be "
                 "empty. No GPU detected.";
      return 0;
    }
  }
  int count;
  PADDLE_ENFORCE_GPU_SUCCESS(hipGetDeviceCount(&count));
  return count;
}

int GetGPUDeviceCount() {
  // cache the count
  static auto dev_cnt = GetGPUDeviceCountImpl();
  return dev_cnt;
}

int GetGPUComputeCapability(int id) {
  PADDLE_ENFORCE_LT(id,
                    GetGPUDeviceCount(),
                    common::errors::InvalidArgument(
                        "Device id must be less than GPU count, "
                        "but received id is: %d. GPU count is: %d.",
                        id,
                        GetGPUDeviceCount()));
  int major, minor;
  auto major_error_code = hipDeviceGetAttribute(
      &major, hipDeviceAttributeComputeCapabilityMajor, id);
  auto minor_error_code = hipDeviceGetAttribute(
      &minor, hipDeviceAttributeComputeCapabilityMinor, id);

  PADDLE_ENFORCE_GPU_SUCCESS(major_error_code);
  PADDLE_ENFORCE_GPU_SUCCESS(minor_error_code);
  return major * 100 + minor;
}

int GetGPURuntimeVersion(int id) {
  PADDLE_ENFORCE_LT(id,
                    GetGPUDeviceCount(),
                    common::errors::InvalidArgument(
                        "Device id must be less than GPU count, "
                        "but received id is: %d. GPU count is: %d.",
                        id,
                        GetGPUDeviceCount()));
  int runtime_version = 0;
  PADDLE_ENFORCE_GPU_SUCCESS(hipRuntimeGetVersion(&runtime_version));
  return runtime_version;
}

int GetGPUDriverVersion(int id) {
  PADDLE_ENFORCE_LT(id,
                    GetGPUDeviceCount(),
                    common::errors::InvalidArgument(
                        "Device id must be less than GPU count, "
                        "but received id is: %d. GPU count is: %d.",
                        id,
                        GetGPUDeviceCount()));
  int driver_version = 0;
  PADDLE_ENFORCE_GPU_SUCCESS(hipDriverGetVersion(&driver_version));
  return driver_version;
}

bool TensorCoreAvailable() { return false; }

int GetGPUMultiProcessors(int id) {
  PADDLE_ENFORCE_LT(id,
                    GetGPUDeviceCount(),
                    common::errors::InvalidArgument(
                        "Device id must be less than GPU count, "
                        "but received id is: %d. GPU count is: %d.",
                        id,
                        GetGPUDeviceCount()));
  int count;
  PADDLE_ENFORCE_GPU_SUCCESS(
      hipDeviceGetAttribute(&count, hipDeviceAttributeMultiprocessorCount, id));
  return count;
}

int GetGPUMaxThreadsPerMultiProcessor(int id) {
  PADDLE_ENFORCE_LT(id,
                    GetGPUDeviceCount(),
                    common::errors::InvalidArgument(
                        "Device id must be less than GPU count, "
                        "but received id is: %d. GPU count is: %d.",
                        id,
                        GetGPUDeviceCount()));
  int count;
  PADDLE_ENFORCE_GPU_SUCCESS(hipDeviceGetAttribute(
      &count, hipDeviceAttributeMaxThreadsPerMultiProcessor, id));

  return count;
}

int GetGPUMaxThreadsPerBlock(int id) {
  PADDLE_ENFORCE_LT(id,
                    GetGPUDeviceCount(),
                    common::errors::InvalidArgument(
                        "Device id must be less than GPU count, "
                        "but received id is: %d. GPU count is: %d.",
                        id,
                        GetGPUDeviceCount()));
  int count;
  PADDLE_ENFORCE_GPU_SUCCESS(
      hipDeviceGetAttribute(&count, hipDeviceAttributeMaxThreadsPerBlock, id));
  return count;
}

int GetCurrentDeviceId() {
  int device_id;
  PADDLE_ENFORCE_GPU_SUCCESS(hipGetDevice(&device_id));
  return device_id;
}

std::array<unsigned int, 3> GetGpuMaxGridDimSize(int id) {
  PADDLE_ENFORCE_LT(id,
                    GetGPUDeviceCount(),
                    common::errors::InvalidArgument(
                        "Device id must be less than GPU count, "
                        "but received id is: %d. GPU count is: %d.",
                        id,
                        GetGPUDeviceCount()));
  std::array<unsigned int, 3> ret;
  int size;
  auto error_code_x =
      hipDeviceGetAttribute(&size, hipDeviceAttributeMaxGridDimX, id);
  PADDLE_ENFORCE_GPU_SUCCESS(error_code_x);
  ret[0] = size;

  auto error_code_y =
      hipDeviceGetAttribute(&size, hipDeviceAttributeMaxGridDimY, id);
  PADDLE_ENFORCE_GPU_SUCCESS(error_code_y);
  ret[1] = size;

  auto error_code_z =
      hipDeviceGetAttribute(&size, hipDeviceAttributeMaxGridDimZ, id);
  PADDLE_ENFORCE_GPU_SUCCESS(error_code_z);
  ret[2] = size;
  return ret;
}

std::pair<int, int> GetGpuStreamPriorityRange() {
  int least_priority, greatest_priority;
  PADDLE_ENFORCE_GPU_SUCCESS(
      hipDeviceGetStreamPriorityRange(&least_priority, &greatest_priority));
  return std::make_pair(least_priority, greatest_priority);
}

const gpuDeviceProp &GetDeviceProperties(int id) {
  std::call_once(g_device_props_size_init_flag, [&] {
    int gpu_num = 0;
    gpu_num = GetGPUDeviceCount();
    g_device_props_init_flags.resize(gpu_num);
    g_device_props.resize(gpu_num);
    for (int i = 0; i < gpu_num; ++i) {
      g_device_props_init_flags[i] = std::make_unique<std::once_flag>();
    }
  });

  if (id == -1) {
    id = GetCurrentDeviceId();
  }

  if (id < 0 || id >= static_cast<int>(g_device_props.size())) {
    PADDLE_THROW(common::errors::OutOfRange(
        "The device id %d is out of range [0, %d), where %d is the number of "
        "devices on this machine. Because the device id should be greater than "
        "or equal to zero and smaller than the number of gpus. Please input "
        "appropriate device again!",
        id,
        static_cast<int>(g_device_props.size()),
        static_cast<int>(g_device_props.size())));
  }

  std::call_once(*(g_device_props_init_flags[id]), [&] {
    PADDLE_ENFORCE_GPU_SUCCESS(hipGetDeviceProperties(&g_device_props[id], id));
  });

  return g_device_props[id];
}

void SetDeviceId(int id) {
  static thread_local bool first_call = true;
  if (first_call) {
    PADDLE_ENFORCE_LT(id,
                      GetGPUDeviceCount(),
                      common::errors::InvalidArgument(
                          "Device id must be less than GPU count, "
                          "but received id is: %d. GPU count is: %d.",
                          id,
                          GetGPUDeviceCount()));

    PADDLE_RETRY_CUDA_SUCCESS(hipSetDevice(id));
    VLOG(4) << "SetDeviceId " << id;
    first_call = false;
    return;
  }

  int prev_id;
  PADDLE_ENFORCE_GPU_SUCCESS(hipGetDevice(&prev_id));
  if (prev_id != id) {
    PADDLE_ENFORCE_LT(id,
                      GetGPUDeviceCount(),
                      common::errors::InvalidArgument(
                          "Device id must be less than GPU count, "
                          "but received id is: %d. GPU count is: %d.",
                          id,
                          GetGPUDeviceCount()));

    PADDLE_RETRY_CUDA_SUCCESS(hipSetDevice(id));
    VLOG(4) << "SetDeviceId " << id;
  }
}

void GpuMemcpyAsync(void *dst,
                    const void *src,
                    size_t count,
                    gpuMemcpyKind kind,
                    gpuStream_t stream) {
  PADDLE_ENFORCE_GPU_SUCCESS(hipMemcpyAsync(dst, src, count, kind, stream));
}

void GpuMemcpySync(void *dst,
                   const void *src,
                   size_t count,
                   gpuMemcpyKind kind) {
  PADDLE_ENFORCE_GPU_SUCCESS(hipMemcpy(dst, src, count, kind));
}

void GpuMemcpyPeerAsync(void *dst,
                        int dst_device,
                        const void *src,
                        int src_device,
                        size_t count,
                        gpuStream_t stream) {
  PADDLE_ENFORCE_GPU_SUCCESS(
      hipMemcpyPeerAsync(dst, dst_device, src, src_device, count, stream));
}

void GpuMemcpyPeerSync(
    void *dst, int dst_device, const void *src, int src_device, size_t count) {
  PADDLE_ENFORCE_GPU_SUCCESS(
      hipMemcpyPeer(dst, dst_device, src, src_device, count));
}

void GpuMemsetAsync(void *dst, int value, size_t count, gpuStream_t stream) {
  PADDLE_ENFORCE_GPU_SUCCESS(hipMemsetAsync(dst, value, count, stream));
}

void GpuStreamSync(gpuStream_t stream) {
  PADDLE_ENFORCE_GPU_SUCCESS(hipStreamSynchronize(stream));
}

void GpuDestroyStream(gpuStream_t stream) {
  PADDLE_ENFORCE_GPU_SUCCESS(hipStreamDestroy(stream));
}

void GpuDeviceSync() { PADDLE_ENFORCE_GPU_SUCCESS(hipDeviceSynchronize()); }

gpuError_t GpuGetLastError() { return hipGetLastError(); }

bool IsGPUManagedMemorySupported(int dev_id) {
  PADDLE_ENFORCE_LT(dev_id,
                    GetGPUDeviceCount(),
                    common::errors::InvalidArgument(
                        "Device id must be less than GPU count, "
                        "but received id is: %d. GPU count is: %d.",
                        dev_id,
                        GetGPUDeviceCount()));
  // TODO(qili93): Hygon DTK (21.04 and 22.04) not support
  // hipDeviceAttributeManagedMemory, temporary disable by default, to be
  // verified in next DTK release
  return false;
}

bool IsGPUManagedMemoryOversubscriptionSupported(int dev_id) {
  PADDLE_ENFORCE_LT(dev_id,
                    GetGPUDeviceCount(),
                    common::errors::InvalidArgument(
                        "Device id must be less than GPU count, "
                        "but received id is: %d. GPU count is: %d.",
                        dev_id,
                        GetGPUDeviceCount()));
#ifdef __linux__
  return IsGPUManagedMemorySupported(dev_id) &&
         GetGPUComputeCapability(dev_id) >= 60;
#else
  return false;
#endif
}

}  // namespace gpu
}  // namespace backends
}  // namespace phi
