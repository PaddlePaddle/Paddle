// Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

#pragma once

#include "glog/logging.h"
#include "paddle/phi/api/include/context_pool.h"
#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/backends/gpu/gpu_info.h"
#include "paddle/phi/core/cuda_stream.h"

namespace deep_ep::detail {

using DeviceIndex = int8_t;
using StreamId = int64_t;

class CUDAStream {
 public:
  CUDAStream() { LOG(FATAL) << "CUDAStream::CUDAStream() is not implemented"; }
  explicit CUDAStream(const cudaStream_t& stream) : raw_stream_(stream) {}
  StreamId id() const { return reinterpret_cast<StreamId>(raw_stream_); }

  operator cudaStream_t() const { return raw_stream_; }

  const cudaStream_t& raw_stream() const { return raw_stream_; }

 private:
  cudaStream_t raw_stream_;
};

/**
 * Get the current CUDA stream, for the passed CUDA device, or for the
 * current device if no device index is passed.  The current CUDA stream
 * will usually be the default CUDA stream for the device, but it may
 * be different if someone called 'setCurrentCUDAStream' or used 'StreamGuard'
 * or 'CUDAStreamGuard'.
 */
inline CUDAStream getCurrentCUDAStream(DeviceIndex device_index = -1) {
  if (device_index == -1) {
    device_index = phi::backends::gpu::GetCurrentDeviceId();
  }

  return CUDAStream(
      paddle::GetCurrentCUDAStream(phi::GPUPlace(device_index))->raw_stream());
  // LOG(FATAL) << "getCurrentCUDAStream is not implemented";
  // return *(CUDAStream*)nullptr;
}

inline void setCurrentCUDAStream(cudaStream_t stream) {
  LOG(FATAL) << "setCurrentCUDAStream is not implemented";
}

}  // namespace deep_ep::detail
