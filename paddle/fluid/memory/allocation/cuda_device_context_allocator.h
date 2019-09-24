// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

#include <cuda_runtime.h>

#include "paddle/fluid/memory/allocation/allocator.h"
#include "paddle/fluid/platform/device_context.h"
#include "paddle/fluid/platform/place.h"

namespace paddle {
namespace memory {
namespace allocation {

/**
 * CUDADeviceContextAllocator will allocate a CUDADeviceContextAllocation
 * after waiting for a self-created event on the default stream. It does so to
 * let the non-default stream be able to allocate GPU memory which will be
 * released by stream callback
 */
class CUDADeviceContextAllocator : public Allocator {
 public:
  explicit CUDADeviceContextAllocator(platform::CUDAPlace place,
                                      cudaStream_t default_stream);
  ~CUDADeviceContextAllocator();

 protected:
  Allocation *AllocateImpl(size_t size) override;
  void FreeImpl(Allocation *allocation) override;

 private:
  platform::CUDAPlace place_;
  cudaEvent_t event_{nullptr};
  cudaStream_t default_stream_{nullptr};
};

}  // namespace allocation
}  // namespace memory
}  // namespace paddle
