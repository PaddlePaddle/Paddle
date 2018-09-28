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

class NaiveManagedAllocator;
class NaiveManagedAllocation : public Allocation {
 public:
  NaiveManagedAllocation(std::unique_ptr<Allocation>&& underlying_allocation,
                         std::shared_ptr<NaiveManagedAllocator> allocator)
      : Allocation(underlying_allocation->ptr(), underlying_allocation->size(),
                   underlying_allocation->place()),
        underlying_allocation_(std::move(underlying_allocation)),
        allocator_(allocator) {}

  ~NaiveManagedAllocation() final;

 private:
  std::unique_ptr<Allocation> underlying_allocation_;
  std::weak_ptr<NaiveManagedAllocator> allocator_;
};

class NaiveManagedAllocator
    : public ManagedAllocator,
      public std::enable_shared_from_this<NaiveManagedAllocator> {
 public:
  template <typename... ARGS>
  static std::shared_ptr<ManagedAllocator> Create(ARGS... args) {
    return std::static_pointer_cast<ManagedAllocator>(
        std::shared_ptr<NaiveManagedAllocator>(
            new NaiveManagedAllocator(std::move(args)...)));
  }

  inline UnmanagedAllocator& UnderlyingAllocator() {
    return *underlying_allocator_;
  }

  bool IsAllocThreadSafe() const override;
  std::unique_ptr<Allocation> Allocate(size_t size,
                                       Attr attr = kDefault) override;
  std::shared_ptr<Allocation> AllocateShared(size_t size,
                                             Attr attr = kDefault) override;

 private:
  explicit NaiveManagedAllocator(std::unique_ptr<Allocator>&& allocator);
  explicit NaiveManagedAllocator(
      std::unique_ptr<UnmanagedAllocator>&& allocator);
  void Init(std::unique_ptr<UnmanagedAllocator>&& allocator);

  std::unique_ptr<UnmanagedAllocator> underlying_allocator_;
};
}  // namespace allocation
}  // namespace memory
}  // namespace paddle
