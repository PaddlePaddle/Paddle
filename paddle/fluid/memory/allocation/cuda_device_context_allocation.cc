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

#include "paddle/fluid/memory/allocation/cuda_device_context_allocation.h"
#include <utility>
#include "paddle/fluid/platform/enforce.h"

namespace paddle {
namespace memory {
namespace allocation {

CUDADeviceContextAllocation::CUDADeviceContextAllocation(
    AllocationPtr allocation)
    : Allocation(allocation->ptr(), allocation->size(), allocation->place()),
      underlying_allocation_(std::move(allocation)) {}

CUDADeviceContextAllocation::~CUDADeviceContextAllocation() {
  PADDLE_ENFORCE_NOT_NULL(
      dev_ctx_, "Didn't set device context for CUDADeviceContextAllocation");
  auto *p_allocation = underlying_allocation_.release();
  VLOG(4) << "Adding callback to delete CUDADeviceContextAllocation at "
          << p_allocation;
  dev_ctx_->AddStreamCallback([p_allocation] {
    VLOG(4) << "Delete CUDADeviceContextAllocation at " << p_allocation;
    AllocationDeleter()(p_allocation);
  });
}

void CUDADeviceContextAllocation::SetCUDADeviceContext(
    const platform::CUDADeviceContext *dev_ctx) {
  dev_ctx_ = dev_ctx;
}

}  // namespace allocation
}  // namespace memory
}  // namespace paddle
