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

#include <c10/core/Device.h>
#include "paddle/phi/api/include/context_pool.h"
#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/backends/gpu/gpu_info.h"
#include "paddle/phi/core/cuda_stream.h"

namespace at::cuda {

using StreamId = int64_t;

class CUDAStream {
 public:
  CUDAStream() = delete;
  explicit CUDAStream(const gpuStream_t& stream) : raw_stream_(stream) {}
  StreamId id() const { return reinterpret_cast<StreamId>(raw_stream_); }

  operator gpuStream_t() const { return raw_stream_; }

  // operator Stream() const { return unwrap(); }

  DeviceType device_type() const { return DeviceType::CUDA; }

  const gpuStream_t& stream() const { return raw_stream_; }

 private:
  gpuStream_t raw_stream_;
};

inline CUDAStream getCurrentCUDAStream(c10::DeviceIndex device_index = -1) {
  if (device_index == -1) {
    device_index = phi::backends::gpu::GetCurrentDeviceId();
  }

  return CUDAStream(
      paddle::GetCurrentCUDAStream(phi::GPUPlace(device_index))->raw_stream());
}

#define getDefaultCUDAStream getCurrentCUDAStream;

}  // namespace at::cuda
