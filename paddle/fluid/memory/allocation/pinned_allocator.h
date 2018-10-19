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

namespace paddle {
namespace memory {
namespace allocation {

// Allocator uses `cudaMallocHost`
class CPUPinnedAllocation : public Allocation {
 public:
  CPUPinnedAllocation(void* ptr, size_t size)
      : Allocation(ptr, size, platform::CUDAPinnedPlace()) {}
};

class CPUPinnedAllocator : public UnmanagedAllocator {
 public:
  std::unique_ptr<Allocation> Allocate(size_t size, Attr attr) override;
  void Free(Allocation* allocation) override;
  bool IsAllocThreadSafe() const override;
};

}  // namespace allocation
}  // namespace memory
}  // namespace paddle
