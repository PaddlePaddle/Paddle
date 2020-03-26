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

#include "paddle/fluid/platform/stream/cuda_stream.h"
#include "paddle/fluid/platform/cuda_device_guard.h"
#include "paddle/fluid/platform/enforce.h"

namespace paddle {
namespace platform {
namespace stream {

constexpr int64_t kHighPriority = -1;
constexpr int64_t kNormalPriority = 0;
constexpr unsigned int kDefaultFlag = cudaStreamNonBlocking;

bool CUDAStream::Init(const Place& place, const enum Priority& priority) {
  std::call_once(once_flag_, [&]() {
    PADDLE_ENFORCE_EQ(is_gpu_place(place), true,
                      platform::errors::InvalidArgument(
                          "Cuda stream must be created using cuda place."));
    place_ = place;
    CUDADeviceGuard guard(boost::get<CUDAPlace>(place_).device);
    if (priority == Priority::HIGH) {
      PADDLE_ENFORCE_CUDA_SUCCESS(
          cudaStreamCreateWithPriority(&stream_, kDefaultFlag, kHighPriority));
    } else if (priority == Priority::NORMAL) {
      PADDLE_ENFORCE_CUDA_SUCCESS(cudaStreamCreateWithPriority(
          &stream_, kDefaultFlag, kNormalPriority));
    }
    VLOG(3) << "CUDAStream Init stream: " << stream_
            << ", priority: " << static_cast<int>(priority);
  });
  return true;
}

bool CUDAStream::IsIdle() const {
  auto res = cudaStreamQuery(stream_);
  if (res == cudaSuccess) {
    return true;
  }
  if (res != cudaErrorNotReady) {
    LOG(ERROR) << "Stream is in bad state when querying status: " << res;
  }
  return false;
}

void CUDAStream::Destroy() {
  if (stream_) {
    PADDLE_ENFORCE_CUDA_SUCCESS(cudaStreamDestroy(stream_));
  }
  stream_ = nullptr;
}

}  // namespace stream
}  // namespace platform
}  // namespace paddle
