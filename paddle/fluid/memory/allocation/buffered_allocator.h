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

namespace paddle {
namespace memory {
namespace allocation {

// NOTE(zjl): BufferedAllocator maintains a memory pool to accelerate
// memory allocation and reuse memory.
// BufferedAllocator provides the same thread-safety level as
// underlying_allocator_
class BufferedAllocator : public UnmanagedAllocator {
 public:
  explicit BufferedAllocator(std::unique_ptr<Allocator>&& allocator);

  BufferedAllocator(std::unique_ptr<Allocator>&& allocator,
                    const std::vector<size_t>& division_plan);

  ~BufferedAllocator();

  std::unique_ptr<Allocation> Allocate(
      size_t size, Allocator::Attr attr = Allocator::Attr::kDefault) override;

  void FreeUniquePtr(std::unique_ptr<Allocation> allocation) override;

  bool IsAllocThreadSafe() const override;

  const std::vector<size_t>& GetDivisionPlan() const;

  void Flush();

 private:
  void InitAndEnforceCheck(std::unique_ptr<Allocator>&& allocator,
                           const std::vector<size_t>& division_plan);

  void InsertAllocation(std::unique_ptr<Allocation>&& allocation);
  void InsertAllocationImpl(std::unique_ptr<Allocation>&& allocation);

  static bool Match(size_t actual_size, size_t requested_size);
  std::unique_ptr<Allocation> RemoveAllocation(size_t size);
  std::unique_ptr<Allocation> RemoveAllocationImpl(size_t size);

  void FreeAllocations(size_t size);
  void FreeAllocationsImpl(size_t size);

  void FlushImpl();

  size_t GetListIndex(size_t size);

  std::unique_ptr<UnmanagedAllocator> underlying_allocator_;
  std::vector<std::multimap<size_t, std::unique_ptr<Allocation>>> allocations_;
  std::vector<size_t> division_plan_;
  std::unique_ptr<std::mutex> mtx_;
};

}  // namespace allocation
}  // namespace memory
}  // namespace paddle
