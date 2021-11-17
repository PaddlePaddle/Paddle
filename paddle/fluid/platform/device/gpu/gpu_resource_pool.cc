// Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
#include "paddle/fluid/platform/device/gpu/gpu_resource_pool.h"
#include "paddle/fluid/platform/device/gpu/gpu_info.h"

namespace paddle {
namespace platform {

CudaStreamResourcePool::CudaStreamResourcePool() {
  int dev_cnt = platform::GetGPUDeviceCount();
  pool_.reserve(dev_cnt);
  for (int dev_idx = 0; dev_idx < dev_cnt; ++dev_idx) {
    auto creator = [dev_idx] {
      platform::SetDeviceId(dev_idx);
      gpuStream_t stream;
#ifdef PADDLE_WITH_HIP
      PADDLE_ENFORCE_GPU_SUCCESS(
          hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));
#else
      PADDLE_ENFORCE_GPU_SUCCESS(
          cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
#endif
      return stream;
    };

    auto deleter = [dev_idx](gpuStream_t stream) {
      platform::SetDeviceId(dev_idx);
#ifdef PADDLE_WITH_HIP
      PADDLE_ENFORCE_GPU_SUCCESS(hipStreamDestroy(stream));
#else
      PADDLE_ENFORCE_GPU_SUCCESS(cudaStreamDestroy(stream));
#endif
    };

    pool_.emplace_back(
        ResourcePool<CudaStreamObject>::Create(creator, deleter));
  }
}

CudaStreamResourcePool& CudaStreamResourcePool::Instance() {
  static CudaStreamResourcePool pool;
  return pool;
}

std::shared_ptr<CudaStreamObject> CudaStreamResourcePool::New(int dev_idx) {
  PADDLE_ENFORCE_GE(
      dev_idx, 0,
      platform::errors::InvalidArgument(
          "The dev_idx should be not less than 0, but got %d.", dev_idx));
  PADDLE_ENFORCE_LT(
      dev_idx, pool_.size(),
      platform::errors::OutOfRange(
          "The dev_idx should be less than device count %d, but got %d.",
          pool_.size(), dev_idx));
  return pool_[dev_idx]->New();
}

CudaEventResourcePool::CudaEventResourcePool() {
  int dev_cnt = platform::GetGPUDeviceCount();
  pool_.reserve(dev_cnt);
  for (int dev_idx = 0; dev_idx < dev_cnt; ++dev_idx) {
    auto creator = [dev_idx] {
      platform::SetDeviceId(dev_idx);
      gpuEvent_t event;
#ifdef PADDLE_WITH_HIP
      PADDLE_ENFORCE_GPU_SUCCESS(
          hipEventCreateWithFlags(&event, hipEventDisableTiming));
#else
      PADDLE_ENFORCE_GPU_SUCCESS(
          cudaEventCreateWithFlags(&event, cudaEventDisableTiming));
#endif
      return event;
    };

    auto deleter = [dev_idx](gpuEvent_t event) {
      platform::SetDeviceId(dev_idx);
#ifdef PADDLE_WITH_HIP
      PADDLE_ENFORCE_GPU_SUCCESS(hipEventDestroy(event));
#else
      PADDLE_ENFORCE_GPU_SUCCESS(cudaEventDestroy(event));
#endif
    };

    pool_.emplace_back(ResourcePool<CudaEventObject>::Create(creator, deleter));
  }
}

CudaEventResourcePool& CudaEventResourcePool::Instance() {
  static CudaEventResourcePool pool;
  return pool;
}

std::shared_ptr<CudaEventObject> CudaEventResourcePool::New(int dev_idx) {
  PADDLE_ENFORCE_GE(
      dev_idx, 0,
      platform::errors::InvalidArgument(
          "The dev_idx should be not less than 0, but got %d.", dev_idx));
  PADDLE_ENFORCE_LT(
      dev_idx, pool_.size(),
      platform::errors::OutOfRange(
          "The dev_idx should be less than device count %d, but got %d.",
          pool_.size(), dev_idx));
  return pool_[dev_idx]->New();
}

}  // namespace platform
}  // namespace paddle

#endif
