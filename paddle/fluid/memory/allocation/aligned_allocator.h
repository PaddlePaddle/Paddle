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
#include <memory>
#include "paddle/fluid/memory/allocation/allocator.h"

namespace paddle {
namespace memory {
namespace allocation {

template <size_t kAlignment>
class AlignedAllocation : public Allocation {
 public:
  AlignedAllocation(std::unique_ptr<Allocation>&& underlying_allocation,
                    size_t size)
      : Allocation(AlignedPtr(underlying_allocation->ptr()), size,
                   underlying_allocation->place()),
        underlying_allocation_(std::move(underlying_allocation)) {}

 private:
  static void* AlignedPtr(void* ptr) {
    auto ptr_addr = reinterpret_cast<uintptr_t>(ptr);
    ptr_addr = (ptr_addr & ~(kAlignment - 1)) + kAlignment;
    return reinterpret_cast<void*>(ptr_addr);
  }

  std::unique_ptr<Allocation> underlying_allocation_;
};

class ThinAlignedAllocator : public ManagedAllocator {
 public:
  explicit ThinAlignedAllocator(
      std::shared_ptr<ManagedAllocator> underlyning_allocator);

 protected:
  std::shared_ptr<ManagedAllocator> underlying_allocator_;
};

template <size_t kAlignment>
class AlignedAllocator : public ThinAlignedAllocator {
 public:
  using ThinAlignedAllocator::ThinAlignedAllocator;
  std::unique_ptr<Allocation> Allocate(size_t size, Attr attr) override {
    auto raw_allocation =
        underlying_allocator_->Allocate(size + kAlignment, attr);
    return std::unique_ptr<Allocation>(
        new AlignedAllocation<kAlignment>(std::move(raw_allocation), size));
  }
  std::shared_ptr<Allocation> AllocateShared(size_t size, Attr attr) override {
    return std::shared_ptr<Allocation>(Allocate(size, attr).release());
  }
};

}  // namespace allocation
}  // namespace memory
}  // namespace paddle
