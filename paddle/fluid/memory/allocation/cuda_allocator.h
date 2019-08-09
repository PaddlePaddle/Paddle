// Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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
#include "paddle/fluid/platform/place.h"

namespace paddle {
namespace memory {
namespace allocation {

// CUDA System allocator and allocation.
// Just a flag type.
class CUDAAllocation : public Allocation {
 public:
  using Allocation::Allocation;
};

class CUDAAllocator : public Allocator {
 public:
  explicit CUDAAllocator(const platform::CUDAPlace& place) : place_(place) {}
  explicit CUDAAllocator(const platform::Place& place)
      : place_(boost::get<platform::CUDAPlace>(place)) {}
  bool IsAllocThreadSafe() const override;

 protected:
  void Free(Allocation* allocation) override;
  Allocation* AllocateImpl(size_t size, Allocator::Attr attr) override;

 private:
  platform::CUDAPlace place_;
};

}  // namespace allocation
}  // namespace memory
}  // namespace paddle
