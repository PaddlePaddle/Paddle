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
#include "paddle/fluid/memory/allocation/allocator.h"
#include "paddle/fluid/platform/device_context.h"

namespace paddle {
namespace memory {
namespace allocation {

/**
 * CUDADeviceContextAllocation is a wrapper of the underbeneath allocation.
 * CUDADeviceContextAllocation adds a CUDA stream callback for the underbeneath
 * allocation so that CUDADeviceContextAllocation can be used in a CUDA stream
 * which deletes allocation in the callback.
 */
class CUDADeviceContextAllocation : public Allocation {
 public:
  explicit CUDADeviceContextAllocation(AllocationPtr allocation);
  ~CUDADeviceContextAllocation();
  void SetCUDADeviceContext(const platform::CUDADeviceContext *dev_ctx);

 private:
  AllocationPtr underlying_allocation_;
  const platform::CUDADeviceContext *dev_ctx_{nullptr};
};

}  // namespace allocation
}  // namespace memory
}  // namespace paddle
