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

#pragma once

#include "paddle/fluid/memory/allocation/allocator.h"
#include "paddle/fluid/memory/stats.h"

namespace paddle {
namespace memory {
namespace allocation {

class StatAllocator : public Allocator {
 public:
  explicit StatAllocator(std::shared_ptr<Allocator> underlying_allocator)
      : underlying_allocator_(std::move(underlying_allocator)) {}

  bool IsAllocThreadSafe() const override { return true; }

 protected:
  void FreeImpl(phi::Allocation* allocation) override {
    MEMORY_STAT_UPDATE(Allocated, allocation->place().GetDeviceId(),
                       -allocation->size());
    underlying_allocator_->Free(allocation);
  }

  phi::Allocation* AllocateImpl(size_t size) override {
    phi::Allocator::AllocationPtr allocation =
        underlying_allocator_->Allocate(size);
    MEMORY_STAT_UPDATE(Allocated, allocation->place().GetDeviceId(),
                       allocation->size());
    return allocation.release();
  }

  uint64_t ReleaseImpl(const platform::Place& place) override {
    return underlying_allocator_->Release(place);
  }

 private:
  std::shared_ptr<Allocator> underlying_allocator_;
};

}  // namespace allocation
}  // namespace memory
}  // namespace paddle
