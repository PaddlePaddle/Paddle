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

#include <map>
#include <memory>
#include <mutex>  // NOLINT
#include <string>
#include <vector>

#include "paddle/fluid/memory/allocation/allocator.h"

namespace paddle {
namespace memory {
namespace allocation {

std::vector<size_t> ReadBufferedAllocatorDivisionPlanFromFile(
    const std::string& filepath);

class MultiBinBufferedAllocator : public Allocator {
 public:
  explicit MultiBinBufferedAllocator(
      std::shared_ptr<Allocator> underlying_allocator);

  MultiBinBufferedAllocator(std::shared_ptr<Allocator> underlying_allocator,
                            const std::vector<size_t>& division_plan);

  bool IsAllocThreadSafe() const override { return mtx_.front() != nullptr; }

  size_t ClearCache();

  const std::vector<size_t>& DivisionPlan() const { return division_plan_; }

 protected:
  Allocation* AllocateImpl(size_t size, Attr attr) override;
  void FreeImpl(Allocation* allocation) override;

 private:
  std::shared_ptr<Allocator> underlying_allocator_;
  std::vector<std::multimap<size_t, AllocationPtr>> allocations_;
  std::vector<size_t> accumulated_cache_size_;
  std::vector<size_t> division_plan_;
  std::vector<std::unique_ptr<std::mutex>> mtx_;
};

extern void UseMultiBinBufferedAllocatorGFlags();

}  // namespace allocation
}  // namespace memory
}  // namespace paddle
