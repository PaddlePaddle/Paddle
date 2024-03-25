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

#include "paddle/cinn/runtime/cuda/cuda_backend_api.h"
#include <glog/logging.h>

namespace cinn {
namespace runtime {
namespace cuda {

CUDABackendAPI* CUDABackendAPI::Global() {
  static auto* inst = new CUDABackendAPI();
  return inst;
}

void CUDABackendAPI::set_device(int device_id) {
  CUDA_CALL(cudaSetDevice(device_id));
}
int CUDABackendAPI::get_device() {
  int device_id = 0;
  CUDA_CALL(cudaGetDevice(&device_id));
  return device_id;
}

std::variant<int, std::array<int, 3>> CUDABackendAPI::get_device_property(
    DeviceProperty device_property, std::optional<int> device_id) {
  // get device index
  int dev_index = device_id.value_or(get_device());
  // get device property
  std::variant<int, std::array<int, 3>> rv_variant;
  int rv = -1;
  switch (device_property) {
    case DeviceProperty::MaxBlockDims: {
      cudaDeviceProp prop_;
      CUDA_CALL(cudaGetDeviceProperties(&prop_, dev_index));
      rv_variant = std::array<int, 3>{prop_.maxThreadsDim[0],
                                      prop_.maxThreadsDim[1],
                                      prop_.maxThreadsDim[2]};
      break;
    }
    case DeviceProperty::MaxGridDims: {
      cudaDeviceProp prop_;
      CUDA_CALL(cudaGetDeviceProperties(&prop_, dev_index));
      rv_variant = std::array<int, 3>{
          prop_.maxGridSize[0], prop_.maxGridSize[1], prop_.maxGridSize[2]};
      break;
    }
    case DeviceProperty::MaxSharedMemoryPerBlock: {
      CUDA_CALL(cudaDeviceGetAttribute(
          &rv, cudaDeviceAttr::cudaDevAttrMaxSharedMemoryPerBlock, dev_index));
      rv_variant = rv;
      break;
    }
    case DeviceProperty::MaxThreadsPerBlock: {
      CUDA_CALL(cudaDeviceGetAttribute(
          &rv, cudaDeviceAttr::cudaDevAttrMaxThreadsPerBlock, dev_index));
      rv_variant = rv;
      break;
    }
    case DeviceProperty::MaxThreadsPerSM: {
      CUDA_CALL(cudaDeviceGetAttribute(
          &rv,
          cudaDeviceAttr::cudaDevAttrMaxThreadsPerMultiProcessor,
          dev_index));
      rv_variant = rv;
      break;
    }
    case DeviceProperty::MultiProcessorCount: {
      CUDA_CALL(cudaDeviceGetAttribute(
          &rv, cudaDeviceAttr::cudaDevAttrMultiProcessorCount, dev_index));
      rv_variant = rv;
      break;
    }
    case DeviceProperty::MaxBlocksPerSM: {
      CUDA_CALL(cudaDeviceGetAttribute(
          &rv,
          cudaDeviceAttr::cudaDevAttrMaxBlocksPerMultiprocessor,
          dev_index));
      rv_variant = rv;
      break;
    }
    case DeviceProperty::WarpSize: {
      CUDA_CALL(cudaDeviceGetAttribute(
          &rv, cudaDeviceAttr::cudaDevAttrWarpSize, dev_index));
      rv_variant = rv;
      break;
    }
    default:
      PADDLE_THROW(::common::errors::Fatal("Not supported device property!"));
  }
  return rv_variant;
}

void* CUDABackendAPI::malloc(size_t numBytes) {
  void* dev_mem = nullptr;
  CUDA_CALL(cudaMalloc(&dev_mem, numBytes));
  return dev_mem;
}

void CUDABackendAPI::free(void* data) { CUDA_CALL(cudaFree(data)); }

void CUDABackendAPI::memset(void* data, int value, size_t numBytes) {
  CUDA_CALL(cudaMemset(data, value, numBytes));
}

void CUDABackendAPI::memcpy(void* dest,
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
  CUDA_CALL(cudaMemcpy(dest, src, numBytes, copy_kind));
}

void CUDABackendAPI::device_sync() { CUDA_CALL(cudaDeviceSynchronize()); }

void CUDABackendAPI::stream_sync(void* stream) {
  CUDA_CALL(cudaStreamSynchronize(static_cast<cudaStream_t>(stream)));
}

}  // namespace cuda
}  // namespace runtime
}  // namespace cinn
