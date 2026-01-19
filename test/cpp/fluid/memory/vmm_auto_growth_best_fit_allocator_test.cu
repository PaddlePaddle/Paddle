// Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/phi/core/memory/allocation/cuda_virtual_mem_allocator.h"
// Expose internals for white-box testing.
#define private public
#include "paddle/phi/core/memory/allocation/virtual_memory_auto_growth_best_fit_allocator.h"
#undef private

#include "gtest/gtest.h"
#include "paddle/common/errors.h"
#include "paddle/phi/core/memory/memory.h"

namespace paddle {
namespace memory {
namespace allocation {

class TestCUDAVirtualMemAllocator : public CUDAVirtualMemAllocator {
 public:
  using CUDAVirtualMemAllocator::CUDAVirtualMemAllocator;
  using CUDAVirtualMemAllocator::FreeImpl;
};

TEST(test_vmm_allocator, test_mem_stats) {
  size_t alignment = 256;
  auto underlying_allocator =
      std::make_shared<TestCUDAVirtualMemAllocator>(phi::GPUPlace());
  auto allocation = underlying_allocator->Allocate(1024);
  EXPECT_GT(DeviceMemoryStatCurrentValue("Reserved", 0), 1024);
  allocation.reset();
  EXPECT_EQ(DeviceMemoryStatCurrentValue("Reserved", 0), 0);
}

class DummyAllocator : public Allocator {
 public:
  bool IsAllocThreadSafe() const override { return true; }

 protected:
  phi::Allocation* AllocateImpl(size_t) override {
    PADDLE_THROW(common::errors::Unavailable(
        "DummyAllocator::AllocateImpl should not be called."));
  }
  void FreeImpl(phi::Allocation*) override {}
};

// Expose FreeImpl for testing.
class ExposedVmmAllocator : public VirtualMemoryAutoGrowthBestFitAllocator {
 public:
  using VirtualMemoryAutoGrowthBestFitAllocator::FreeImpl;
  using VirtualMemoryAutoGrowthBestFitAllocator::
      VirtualMemoryAutoGrowthBestFitAllocator;
};

TEST(test_vmm_allocator, free_impl_handles_stale_iterator) {
  auto underlying = std::make_shared<DummyAllocator>();
  phi::GPUPlace place(0);
  ExposedVmmAllocator allocator(underlying, 256, place);

  // Manually construct blocks: [free-prev][used-target][free-next]
  allocator.all_blocks_.clear();
  auto prev = allocator.all_blocks_.emplace(
      allocator.all_blocks_.end(), reinterpret_cast<void*>(0x1000), 1024, true);
  auto target = allocator.all_blocks_.emplace(allocator.all_blocks_.end(),
                                              reinterpret_cast<void*>(0x1400),
                                              2048,
                                              false);
  auto next = allocator.all_blocks_.emplace(
      allocator.all_blocks_.end(), reinterpret_cast<void*>(0x1C00), 4096, true);

  allocator.free_blocks_.clear();
  allocator.free_blocks_.emplace(std::make_pair(prev->size_, prev->ptr_), prev);
  allocator.free_blocks_.emplace(std::make_pair(next->size_, next->ptr_), next);

  // Stale allocation keeps an iterator to the "target" block, which will be
  // erased before calling FreeImpl to simulate a dangling iterator.
  auto stale_allocation = new BlockAllocation(target, place);

  // Invalidate the iterator by erasing the target block (simulating previous
  // merge/erase). free_blocks_ deliberately not updated to mimic the
  // inconsistent state seen in the crash reports.
  allocator.all_blocks_.erase(target);

  // FreeImpl should not crash; it should detect the missing block and return.
  EXPECT_NO_THROW(allocator.FreeImpl(stale_allocation));
}

}  // namespace allocation
}  // namespace memory
}  // namespace paddle
