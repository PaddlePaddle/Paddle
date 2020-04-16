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

#ifdef PADDLE_WITH_CUDA
#include "paddle/fluid/platform/cuda_resource_pool.h"
#include "paddle/fluid/platform/gpu_info.h"

namespace paddle {
namespace platform {

CudaStreamResourcePool::CudaStreamResourcePool() {
  int dev_cnt = platform::GetCUDADeviceCount();
  pool_.reserve(dev_cnt);
  for (int dev_idx = 0; dev_idx < dev_cnt; ++dev_idx) {
    auto creator = [dev_idx] {
      platform::SetDeviceId(dev_idx);
      cudaStream_t stream;
      PADDLE_ENFORCE_CUDA_SUCCESS(
          cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
      return stream;
    };

    auto deleter = [dev_idx](cudaStream_t stream) {
      platform::SetDeviceId(dev_idx);
      PADDLE_ENFORCE_CUDA_SUCCESS(cudaStreamDestroy(stream));
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
          "dev_idx should be not less than 0, but got %d", dev_idx));
  PADDLE_ENFORCE_LT(
      dev_idx, pool_.size(),
      platform::errors::OutOfRange(
          "dev_idx should be less than device count %d, but got %d",
          pool_.size(), dev_idx));
  return pool_[dev_idx]->New();
}

CudaEventResourcePool::CudaEventResourcePool() {
  int dev_cnt = platform::GetCUDADeviceCount();
  pool_.reserve(dev_cnt);
  for (int dev_idx = 0; dev_idx < dev_cnt; ++dev_idx) {
    auto creator = [dev_idx] {
      platform::SetDeviceId(dev_idx);
      cudaEvent_t event;
      PADDLE_ENFORCE_CUDA_SUCCESS(
          cudaEventCreateWithFlags(&event, cudaEventDisableTiming));
      return event;
    };

    auto deleter = [dev_idx](cudaEvent_t event) {
      platform::SetDeviceId(dev_idx);
      PADDLE_ENFORCE_CUDA_SUCCESS(cudaEventDestroy(event));
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
          "dev_idx should be not less than 0, but got %d", dev_idx));
  PADDLE_ENFORCE_LT(
      dev_idx, pool_.size(),
      platform::errors::OutOfRange(
          "dev_idx should be less than device count %d, but got %d",
          pool_.size(), dev_idx));
  return pool_[dev_idx]->New();
}

}  // namespace platform
}  // namespace paddle

#endif
