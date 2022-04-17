// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

#ifndef _WIN32
#pragma once

#include <memory>
#include <mutex>  // NOLINT
#include <string>
#include <unordered_set>
#include <utility>

#include "paddle/fluid/memory/allocation/allocator.h"
#include "paddle/fluid/platform/cuda_device_guard.h"
#include "paddle/fluid/platform/device/gpu/gpu_info.h"
#include "paddle/fluid/platform/enforce.h"

namespace paddle {
namespace memory {
namespace allocation {

std::shared_ptr<void> GetIpcBasePtr(std::string handle);

class CudaIpcAllocation : public Allocation {
 public:
  explicit CudaIpcAllocation(void *ptr, size_t size, int device_id,
                             std::shared_ptr<void> shared_ptr)
      : Allocation(ptr, size, platform::CUDAPlace(device_id)),
        device_id_(std::move(device_id)),
        shared_ptr_(std::move(shared_ptr)) {}

  inline const int &device_id() const { return device_id_; }

  ~CudaIpcAllocation() override;

 private:
  int device_id_;
  std::shared_ptr<void> shared_ptr_;
};

}  // namespace allocation
}  // namespace memory
}  // namespace paddle

#endif
