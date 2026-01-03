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

#include "paddle/cinn/runtime/custom_device/custom_device_backend_api.h"

#include "paddle/cinn/runtime/custom_device/custom_device_util.h"

namespace cinn {
namespace runtime {
namespace custom_device {

HIPBackendAPI* HIPBackendAPI::Global() {
  static auto* inst = new HIPBackendAPI();
  return inst;
}

void HIPBackendAPI::set_device(int device_id) {
  HIP_CHECK(customDeviceSetDevice(device_id));
}

int HIPBackendAPI::get_device() {
  int device_id = 0;
  HIP_CHECK(customDeviceGetDevice(&device_id));
  return device_id;
}

int HIPBackendAPI::get_device_property(DeviceProperty device_property,
                                       std::optional<int> device_id) {
  int dev_index = device_id.value_or(get_device());
  int rv = -1;
  switch (device_property) {
    case DeviceProperty::MaxBlockDimX: {
      HIP_CHECK(customDeviceGetAttribute(
          &rv,
          customDeviceAttribute_t::customDeviceAttributeMaxBlockDimX,
          dev_index));
      break;
    }
    case DeviceProperty::MaxBlockDimY: {
      HIP_CHECK(customDeviceGetAttribute(
          &rv,
          customDeviceAttribute_t::customDeviceAttributeMaxBlockDimY,
          dev_index));
      break;
    }
    case DeviceProperty::MaxBlockDimZ: {
      HIP_CHECK(customDeviceGetAttribute(
          &rv,
          customDeviceAttribute_t::customDeviceAttributeMaxBlockDimZ,
          dev_index));
      break;
    }
    case DeviceProperty::MaxGridDimX: {
      HIP_CHECK(customDeviceGetAttribute(
          &rv,
          customDeviceAttribute_t::customDeviceAttributeMaxGridDimX,
          dev_index));
      break;
    }
    case DeviceProperty::MaxGridDimY: {
      HIP_CHECK(customDeviceGetAttribute(
          &rv,
          customDeviceAttribute_t::customDeviceAttributeMaxGridDimY,
          dev_index));
      break;
    }
    case DeviceProperty::MaxGridDimZ: {
      HIP_CHECK(customDeviceGetAttribute(
          &rv,
          customDeviceAttribute_t::customDeviceAttributeMaxGridDimZ,
          dev_index));
      break;
    }
    case DeviceProperty::MaxSharedMemoryPerBlock: {
      HIP_CHECK(customDeviceGetAttribute(
          &rv,
          customDeviceAttribute_t::customDeviceAttributeMaxSharedMemoryPerBlock,
          dev_index));
      break;
    }
    case DeviceProperty::MaxThreadsPerBlock: {
      HIP_CHECK(customDeviceGetAttribute(
          &rv,
          customDeviceAttribute_t::customDeviceAttributeMaxThreadsPerBlock,
          dev_index));
      break;
    }
    case DeviceProperty::MaxThreadsPerSM: {
      HIP_CHECK(customDeviceGetAttribute(
          &rv,
          customDeviceAttribute_t::
              customDeviceAttributeMaxThreadsPerMultiProcessor,
          dev_index));
      break;
    }
    case DeviceProperty::MultiProcessorCount: {
      HIP_CHECK(customDeviceGetAttribute(
          &rv,
          customDeviceAttribute_t::customDeviceAttributeMultiprocessorCount,
          dev_index));
      break;
    }
    case DeviceProperty::MaxBlocksPerSM: {
      HIP_CHECK(customDeviceGetAttribute(
          &rv,
          customDeviceAttribute_t::
              customDeviceAttributeMaxThreadsPerMultiProcessor,
          dev_index));
      break;
    }
    case DeviceProperty::WarpSize: {
      HIP_CHECK(customDeviceGetAttribute(
          &rv,
          customDeviceAttribute_t::customDeviceAttributeWarpSize,
          dev_index));
      break;
    }
    default:
      PADDLE_THROW(
          ::common::errors::InvalidArgument("Not supported device property!"));
  }
  return rv;
}

void* HIPBackendAPI::malloc(size_t numBytes) {
  void* dev_mem = nullptr;
  HIP_CHECK(customDeviceMalloc(&dev_mem, numBytes));
  return dev_mem;
}

void HIPBackendAPI::free(void* data) { HIP_CHECK(customDeviceFree(data)); }

void HIPBackendAPI::memset(void* data, int value, size_t numBytes) {
  HIP_CHECK(customDeviceMemset(data, value, numBytes));
}

void HIPBackendAPI::memcpy(void* dest,
                           const void* src,
                           size_t numBytes,
                           MemcpyType type) {
  customDevicetomDeviceMemcpyKind copy_kind;
  switch (type) {
    case MemcpyType::HostToHost:
      copy_kind = customDeviceMemcpyHostToHost;
      break;
    case MemcpyType::HostToDevice:
      copy_kind = customDeviceMemcpyHostToDevice;
      break;
    case MemcpyType::DeviceToHost:
      copy_kind = customDeviceMemcpyDeviceToHost;
      break;
    case MemcpyType::DeviceToDevice:
      copy_kind = customDeviceMemcpyDeviceToDevice;
      break;
  }
  HIP_CHECK(customDeviceMemcpy(dest, src, numBytes, copy_kind));
}

void HIPBackendAPI::device_sync() {
  HIP_CHECK(customDeviceDeviceSynchronize());
}

void HIPBackendAPI::stream_sync(void* stream) {
  HIP_CHECK(
      customDeviceStreamSynchronize(static_cast<customDeviceStream_t>(stream)));
}

std::array<int, 3> HIPBackendAPI::get_max_grid_dims(
    std::optional<int> device_id) {
  std::array<int, 3> kMaxGridDims;
  kMaxGridDims[0] = get_device_property(DeviceProperty::MaxGridDimX, device_id);
  kMaxGridDims[1] = get_device_property(DeviceProperty::MaxGridDimY, device_id);
  kMaxGridDims[2] = get_device_property(DeviceProperty::MaxGridDimZ, device_id);
  return kMaxGridDims;
}

std::array<int, 3> HIPBackendAPI::get_max_block_dims(
    std::optional<int> device_id) {
  std::array<int, 3> kMaxBlockDims;
  kMaxBlockDims[0] =
      get_device_property(DeviceProperty::MaxBlockDimX, device_id);
  kMaxBlockDims[1] =
      get_device_property(DeviceProperty::MaxBlockDimY, device_id);
  kMaxBlockDims[2] =
      get_device_property(DeviceProperty::MaxBlockDimZ, device_id);
  return kMaxBlockDims;
}

}  // namespace custom_device
}  // namespace runtime
}  // namespace cinn
