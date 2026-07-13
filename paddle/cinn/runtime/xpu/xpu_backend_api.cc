// Copyright (c) 2024 CINN Authors. All Rights Reserved.
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

#include "paddle/cinn/runtime/xpu/xpu_backend_api.h"

#include "paddle/cinn/runtime/xpu/xpu_util.h"

namespace cinn {
namespace runtime {
namespace xpu {

XpuBackendAPI* XpuBackendAPI::Global() {
  static auto* inst = new XpuBackendAPI();
  return inst;
}

void XpuBackendAPI::set_device(int device_id) {
  XPU_CHECK(cudaSetDevice(device_id));
}

int XpuBackendAPI::get_device() {
  int device_id = 0;
  XPU_CHECK(cudaGetDevice(&device_id));
  return device_id;
}

int XpuBackendAPI::get_device_property(DeviceProperty device_property,
                                       std::optional<int> device_id) {
  int dev_index = device_id.value_or(get_device());
  int rv = -1;
  switch (device_property) {
    case DeviceProperty::MaxBlockDimX:
      XPU_CHECK(cudaDeviceGetAttribute(
          &rv, cudaDevAttrMaxBlockDimX, dev_index));
      break;
    case DeviceProperty::MaxBlockDimY:
      XPU_CHECK(cudaDeviceGetAttribute(
          &rv, cudaDevAttrMaxBlockDimY, dev_index));
      break;
    case DeviceProperty::MaxBlockDimZ:
      XPU_CHECK(cudaDeviceGetAttribute(
          &rv, cudaDevAttrMaxBlockDimZ, dev_index));
      break;
    case DeviceProperty::MaxGridDimX:
      XPU_CHECK(cudaDeviceGetAttribute(
          &rv, cudaDevAttrMaxGridDimX, dev_index));
      break;
    case DeviceProperty::MaxGridDimY:
      XPU_CHECK(cudaDeviceGetAttribute(
          &rv, cudaDevAttrMaxGridDimY, dev_index));
      break;
    case DeviceProperty::MaxGridDimZ:
      XPU_CHECK(cudaDeviceGetAttribute(
          &rv, cudaDevAttrMaxGridDimZ, dev_index));
      break;
    case DeviceProperty::MaxSharedMemoryPerBlock:
      XPU_CHECK(cudaDeviceGetAttribute(
          &rv, cudaDevAttrMaxSharedMemoryPerBlock, dev_index));
      break;
    case DeviceProperty::MaxThreadsPerBlock:
      XPU_CHECK(cudaDeviceGetAttribute(
          &rv, cudaDevAttrMaxThreadsPerBlock, dev_index));
      break;
    case DeviceProperty::MaxThreadsPerSM:
      XPU_CHECK(cudaDeviceGetAttribute(
          &rv, cudaDevAttrMaxThreadsPerMultiProcessor, dev_index));
      break;
    case DeviceProperty::MultiProcessorCount:
      XPU_CHECK(cudaDeviceGetAttribute(
          &rv, cudaDevAttrMultiProcessorCount, dev_index));
      break;
    case DeviceProperty::MaxBlocksPerSM:
      XPU_CHECK(cudaDeviceGetAttribute(
          &rv, cudaDevAttrMaxBlocksPerMultiprocessor, dev_index));
      break;
    case DeviceProperty::WarpSize:
      XPU_CHECK(cudaDeviceGetAttribute(&rv, cudaDevAttrWarpSize, dev_index));
      break;
    default:
      PADDLE_THROW(
          ::common::errors::InvalidArgument("Not supported device property!"));
  }
  return rv;
}

void* XpuBackendAPI::malloc(size_t numBytes) {
  void* dev_mem = nullptr;
  XPU_CHECK(cudaMalloc(&dev_mem, numBytes));
  return dev_mem;
}

void XpuBackendAPI::free(void* data) { XPU_CHECK(cudaFree(data)); }

void XpuBackendAPI::memset(void* data, int value, size_t numBytes) {
  XPU_CHECK(cudaMemset(data, value, numBytes));
}

void XpuBackendAPI::memcpy(void* dest,
                           const void* src,
                           size_t numBytes,
                           MemcpyType type) {
  cudaMemcpyKind copy_kind;
  switch (type) {
    case MemcpyType::HostToHost:
      copy_kind = cudaMemcpyHostToHost;
      break;
    case MemcpyType::HostToDevice:
      copy_kind = cudaMemcpyHostToDevice;
      break;
    case MemcpyType::DeviceToHost:
      copy_kind = cudaMemcpyDeviceToHost;
      break;
    case MemcpyType::DeviceToDevice:
      copy_kind = cudaMemcpyDeviceToDevice;
      break;
  }
  XPU_CHECK(cudaMemcpy(dest, src, numBytes, copy_kind));
}

void XpuBackendAPI::device_sync() { XPU_CHECK(cudaDeviceSynchronize()); }

void XpuBackendAPI::stream_sync(void* stream) {
  XPU_CHECK(cudaStreamSynchronize(static_cast<cudaStream_t>(stream)));
}

std::array<int, 3> XpuBackendAPI::get_max_grid_dims(
    std::optional<int> device_id) {
  return {get_device_property(DeviceProperty::MaxGridDimX, device_id),
          get_device_property(DeviceProperty::MaxGridDimY, device_id),
          get_device_property(DeviceProperty::MaxGridDimZ, device_id)};
}

std::array<int, 3> XpuBackendAPI::get_max_block_dims(
    std::optional<int> device_id) {
  return {get_device_property(DeviceProperty::MaxBlockDimX, device_id),
          get_device_property(DeviceProperty::MaxBlockDimY, device_id),
          get_device_property(DeviceProperty::MaxBlockDimZ, device_id)};
}

}  // namespace xpu
}  // namespace runtime
}  // namespace cinn
