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

#include <cstdint>
#include <map>
#include <memory>
#include <vector>

#include "paddle/fluid/memory/allocation/allocator.h"
#include "paddle/fluid/platform/lock_guard_ptr.h"

namespace paddle {
namespace memory {
namespace allocation {

// NOTE(zjl): BufferedAllocator maintains a memory pool to accelerate
// memory allocation and reuse memory.
// BufferedAllocator provides the same thread-safety level as
// underlying_allocator_
class BufferedAllocator : public Allocator {
 public:
  explicit BufferedAllocator(std::shared_ptr<Allocator> allocator);

  ~BufferedAllocator();

  bool IsAllocThreadSafe() const override;

  // only used in unittest
  inline void ClearCache() { FreeCache(-1UL); }

 private:
  void FreeCache(size_t size);

 protected:
  void FreeImpl(phi::Allocation *allocation) override;
  phi::Allocation *AllocateImpl(size_t size) override;

 private:
  std::shared_ptr<Allocator> underlying_allocator_;
  std::multimap<size_t, AllocationPtr> allocations_;
  std::unique_ptr<std::mutex> mtx_;
};

}  // namespace allocation
}  // namespace memory
}  // namespace paddle
