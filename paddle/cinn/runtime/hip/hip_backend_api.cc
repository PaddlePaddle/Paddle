// Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/cinn/runtime/hip/hip_backend_api.h"
#include <glog/logging.h>
#include <hip/hip_runtime.h>
#include "paddle/cinn/runtime/hip/hip_util.h"
#include "paddle/common/enforce.h"

namespace cinn {
namespace runtime {
namespace hip {

HIPBackendAPI* HIPBackendAPI::Global() {
  static auto* inst = new HIPBackendAPI();
  return inst;
}

void HIPBackendAPI::set_device(int device_id) {
  HIP_CALL(hipSetDevice(device_id));
}

int HIPBackendAPI::get_device() {
  int device_id = 0;
  HIP_CALL(hipGetDevice(&device_id));
  return device_id;
}

int HIPBackendAPI::get_device_property(DeviceProperty device_property,
                                       std::optional<int> device_id) {
  int dev_index = device_id.value_or(get_device());
  int rv = -1;
  switch (device_property) {
    case DeviceProperty::MaxBlockDimX: {
      HIP_CALL(hipDeviceGetAttribute(
          &rv,
          hipDeviceAttribute_t::hipDeviceAttributeMaxBlockDimX,
          dev_index));
      break;
    }
    case DeviceProperty::MaxBlockDimY: {
      HIP_CALL(hipDeviceGetAttribute(
          &rv,
          hipDeviceAttribute_t::hipDeviceAttributeMaxBlockDimY,
          dev_index));
      break;
    }
    case DeviceProperty::MaxBlockDimZ: {
      HIP_CALL(hipDeviceGetAttribute(
          &rv,
          hipDeviceAttribute_t::hipDeviceAttributeMaxBlockDimZ,
          dev_index));
      break;
    }
    case DeviceProperty::MaxGridDimX: {
      HIP_CALL(hipDeviceGetAttribute(
          &rv, hipDeviceAttribute_t::hipDeviceAttributeMaxGridDimX, dev_index));
      break;
    }
    case DeviceProperty::MaxGridDimY: {
      HIP_CALL(hipDeviceGetAttribute(
          &rv, hipDeviceAttribute_t::hipDeviceAttributeMaxGridDimY, dev_index));
      break;
    }
    case DeviceProperty::MaxGridDimZ: {
      HIP_CALL(hipDeviceGetAttribute(
          &rv, hipDeviceAttribute_t::hipDeviceAttributeMaxGridDimZ, dev_index));
      break;
    }
    case DeviceProperty::MaxSharedMemoryPerBlock: {
      HIP_CALL(hipDeviceGetAttribute(
          &rv,
          hipDeviceAttribute_t::hipDeviceAttributeMaxSharedMemoryPerBlock,
          dev_index));
      break;
    }
    case DeviceProperty::MaxThreadsPerBlock: {
      HIP_CALL(hipDeviceGetAttribute(
          &rv,
          hipDeviceAttribute_t::hipDeviceAttributeMaxThreadsPerBlock,
          dev_index));
      break;
    }
    case DeviceProperty::MaxThreadsPerSM: {
      HIP_CALL(hipDeviceGetAttribute(
          &rv,
          hipDeviceAttribute_t::hipDeviceAttributeMaxThreadsPerMultiProcessor,
          dev_index));
      break;
    }
    case DeviceProperty::MultiProcessorCount: {
      HIP_CALL(hipDeviceGetAttribute(
          &rv,
          hipDeviceAttribute_t::hipDeviceAttributeMultiprocessorCount,
          dev_index));
      break;
    }
    case DeviceProperty::MaxBlocksPerSM: {
      HIP_CALL(hipDeviceGetAttribute(
          &rv,
          hipDeviceAttribute_t::hipDeviceAttributeMaxThreadsPerMultiProcessor,
          dev_index));
      break;
    }
    case DeviceProperty::WarpSize: {
      HIP_CALL(hipDeviceGetAttribute(
          &rv, hipDeviceAttribute_t::hipDeviceAttributeWarpSize, dev_index));
      break;
    }
    default:
      PADDLE_THROW(
          phi::errors::InvalidArgument("Not supported device property!"));
  }
  return rv;
}

void* HIPBackendAPI::malloc(size_t numBytes) {
  void* dev_mem = nullptr;
  HIP_CALL(hipMalloc(&dev_mem, numBytes));
  return dev_mem;
}

void HIPBackendAPI::free(void* data) { HIP_CALL(hipFree(data)); }

void HIPBackendAPI::memset(void* data, int value, size_t numBytes) {
  HIP_CALL(hipMemset(data, value, numBytes));
}

void HIPBackendAPI::memcpy(void* dest,
                           const void* src,
                           size_t numBytes,
                           MemcpyType type) {
  hipMemcpyKind copy_kind;
  switch (type) {
    case MemcpyType::HostToHost:
      copy_kind = hipMemcpyHostToHost;
      break;
    case MemcpyType::HostToDevice:
      copy_kind = hipMemcpyHostToDevice;
      break;
    case MemcpyType::DeviceToHost:
      copy_kind = hipMemcpyDeviceToHost;
      break;
    case MemcpyType::DeviceToDevice:
      copy_kind = hipMemcpyDeviceToDevice;
      break;
  }
  HIP_CALL(hipMemcpy(dest, src, numBytes, copy_kind));
}

void HIPBackendAPI::device_sync() { HIP_CALL(hipDeviceSynchronize()); }

void HIPBackendAPI::stream_sync(void* stream) {
  HIP_CALL(hipStreamSynchronize(static_cast<hipStream_t>(stream)));
}

}  // namespace hip
}  // namespace runtime
}  // namespace cinn
