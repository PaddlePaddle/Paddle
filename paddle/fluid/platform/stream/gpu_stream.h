/* Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#pragma once

#include <cuda_runtime.h>
#include "paddle/fluid/platform/stream/paddle_stream.h"
#include "paddle/fluid/platform/stream/stream_internal.h"

namespace paddle {
namespace platform {
namespace stream {

class DeviceContext;

namespace internal {
class StreamInterface;
}

class GpuStream : public internal::StreamInterface {
 public:
  explicit GpuStream(DeviceContext* ctx = 0)  // should hold ctx?
      : gpu_stream_(nullptr),
        finish_event_(nullptr) {}
  virtual ~GpuStream() {}
  bool Init();

  cudaStream_t gpu_stream() const {
    return const_cast<cudaStream_t>(gpu_stream_);
  }
  const cudaEvent_t* finish_event() const { return &finish_event_; }
  bool IsIdle() const;
  void Destroy();

 private:
  cudaStream_t gpu_stream_ = nullptr;
  cudaEvent_t finish_event_ = nullptr;
};

GpuStream* GetGpuStream(BaseStream* stream);

cudaStream_t GetCUDAStream(BaseStream* stream);

}  // namespace stream
}  // namespace platform
}  // namespace paddle
