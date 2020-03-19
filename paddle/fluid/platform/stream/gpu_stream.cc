/* Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.

licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/platform/stream/gpu_stream.h"
#include "paddle/fluid/platform/enforce.h"

namespace paddle {
namespace platform {
namespace stream {

bool GpuStream::Init() {
  PADDLE_ENFORCE_CUDA_SUCCESS(cudaStreamCreate(&gpu_stream_));
  PADDLE_ENFORCE_CUDA_SUCCESS(cudaEventCreate(&finish_event_));
  VLOG(3) << "GpuStream Init stream:" << gpu_stream_
          << " finish_event:" << finish_event_;
  return true;
}

GpuStream* GetGpuStream(BaseStream* stream) {
  return static_cast<GpuStream*>(stream->implementation());
}

bool GpuStream::IsIdle() const {
  auto res = cudaStreamQuery(gpu_stream_);
  if (res == cudaSuccess) {
    return true;
  }
  if (res != cudaErrorNotReady) {
    LOG(ERROR) << " Stream is in bad state when querying status:" << res;
  }
  return false;
}

void GpuStream::Destroy() {
  if (finish_event_ != nullptr) {
    PADDLE_ENFORCE_CUDA_SUCCESS(cudaEventDestroy(finish_event_));
    finish_event_ = nullptr;
  }
  if (gpu_stream_ != nullptr) {
    PADDLE_ENFORCE_CUDA_SUCCESS(cudaStreamDestroy(gpu_stream_));
    gpu_stream_ = nullptr;
  }
}

cudaStream_t GetCUDAStream(BaseStream* stream) {
  return GetGpuStream(stream)->gpu_stream();
}

}  // namespace stream
}  // namespace platform
}  // namespace paddle
