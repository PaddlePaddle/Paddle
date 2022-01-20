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

#include <cstdlib>

#include "paddle/fluid/memory/allocation/aligned_allocator.h"
#include "paddle/fluid/memory/allocation/auto_growth_best_fit_allocator.h"

#include "gtest/gtest.h"

DECLARE_bool(free_idle_chunk);
DECLARE_bool(free_when_no_cache_hit);

namespace paddle {
namespace memory {
namespace allocation {

class RecordedAllocator : public Allocator {
 protected:
  pten::Allocation *AllocateImpl(size_t size) override {
    allocated_size_ += size;
    return new Allocation(malloc(size), size, platform::CPUPlace());
  }

  void FreeImpl(pten::Allocation *allocation) {
    allocated_size_ -= allocation->size();
    free(allocation->ptr());
    delete allocation;
  }

 public:
  size_t AllocatedSize() const { return allocated_size_; }

 private:
  size_t allocated_size_{0};
};

static void TestFreeIdleChunk(bool free_idle_chunk,
                              bool free_when_no_cache_hit) {
  FLAGS_free_idle_chunk = free_idle_chunk;
  FLAGS_free_when_no_cache_hit = free_when_no_cache_hit;
  auto recorded_allocator = std::make_shared<RecordedAllocator>();

  size_t alignment = 4096;
  size_t memory_size = 8192;
  auto underlying_allocator =
      std::make_shared<AlignedAllocator>(recorded_allocator, alignment);
  auto ag_allocator = std::make_shared<AutoGrowthBestFitAllocator>(
      underlying_allocator, alignment);

  for (size_t i = 0; i < 10; ++i) {
    auto allocation = ag_allocator->Allocate(memory_size);
    ASSERT_EQ(recorded_allocator->AllocatedSize(), memory_size + alignment);
    allocation.reset();
    if (free_idle_chunk) {
      ASSERT_EQ(recorded_allocator->AllocatedSize(), 0UL);
    } else {
      ASSERT_EQ(recorded_allocator->AllocatedSize(), memory_size + alignment);
    }
    ag_allocator->Release(platform::CPUPlace());
  }
}

class LimitedResourceAllocator : public Allocator {
 public:
  explicit LimitedResourceAllocator(size_t capacity) : capacity_(capacity) {}

  size_t AllocatedSize() const { return allocated_size_; }

 protected:
  pten::Allocation *AllocateImpl(size_t size) override {
    if (allocated_size_ + size > capacity_) {
      throw BadAlloc("", __FILE__, __LINE__);
    }

    allocated_size_ += size;
    return new Allocation(malloc(size), size, platform::CPUPlace());
  }

  void FreeImpl(pten::Allocation *allocation) {
    allocated_size_ -= allocation->size();
    free(allocation->ptr());
    delete allocation;
  }

 private:
  size_t allocated_size_{0};
  const size_t capacity_;
};

static void TestFreeWhenNoCacheHit(bool free_when_no_cache_hit) {
  FLAGS_free_idle_chunk = false;
  FLAGS_free_when_no_cache_hit = free_when_no_cache_hit;
  size_t alignment = 256;
  size_t base_memory_size = 4096;

  /*
   * Suppose that we have 3 memory allocation request, that is:
   *  - allocate x1, and then free x1
   *  - allocate x2, and then free x2
   *  - allocate x3, and then free x3
   *
   * where:
   *  - x1 + alignment < x2
   *  - x2 + alignment < x3
   *  - x1 + x2 <= memory_capacity < x1 + x2 + x3
   *
   * In this unittest, we obtain memory_capacity by
   * ((x1 + x2) + (x1 + x2 + x3) / 2 = x1 + x2 + x3 / 2.
   *
   * In this case, when FLAGS_free_when_no_cache_hit is true,
   * the cached memory size when each allocation request ends
   * would be: x1 + alignment, x2 + alignment, x3 + alignment.
   *
   * When FLAGS_free_when_no_cache_hit is false, the cached
   * memory size when each allocation request ends would be:
   * x1 + alignment, x1 + x2 + 2 * alignment, x3 + alignment.
   */
  std::vector<size_t> allocate_size = {base_memory_size,
                                       base_memory_size + alignment * 2,
                                       base_memory_size + alignment * 4};
  size_t memory_capacity =
      allocate_size[0] + allocate_size[1] + allocate_size[2] / 2;

  auto underlying_allocator =
      std::make_shared<LimitedResourceAllocator>(memory_capacity);
  auto aligned_allocator =
      std::make_shared<AlignedAllocator>(underlying_allocator, alignment);
  auto ag_allocator = std::make_shared<AutoGrowthBestFitAllocator>(
      aligned_allocator, alignment);

  ag_allocator->Allocate(allocate_size[0]);
  ASSERT_EQ(underlying_allocator->AllocatedSize(),
            allocate_size[0] + alignment);

  ag_allocator->Allocate(allocate_size[1]);
  if (free_when_no_cache_hit) {
    ASSERT_EQ(underlying_allocator->AllocatedSize(),
              allocate_size[1] + alignment);
  } else {
    ASSERT_EQ(underlying_allocator->AllocatedSize(),
              allocate_size[0] + allocate_size[1] + 2 * alignment);
  }

  ag_allocator->Allocate(allocate_size[2]);
  ASSERT_EQ(underlying_allocator->AllocatedSize(),
            allocate_size[2] + alignment);
}

TEST(test_auto_growth_allocator, test_free_idle_chunk) {
  for (auto free_idle_chunk : {false, true}) {
    for (auto free_when_no_cache_hit : {false, true}) {
      TestFreeIdleChunk(free_idle_chunk, free_when_no_cache_hit);
    }
  }
}

TEST(test_auto_growth_allocator, test_free_when_no_cache_hit) {
  TestFreeWhenNoCacheHit(false);
  TestFreeWhenNoCacheHit(true);
}

}  // namespace allocation
}  // namespace memory
}  // namespace paddle
