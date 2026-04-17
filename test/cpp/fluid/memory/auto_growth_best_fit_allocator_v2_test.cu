// Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
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

// Tests for AutoGrowthBestFitAllocatorV2 — verifying that fragmentation
// counters (cache_hit, cache_miss, etc.) are correctly incremented.

#include "paddle/phi/core/memory/allocation/auto_growth_best_fit_allocator_v2.h"
#include "gtest/gtest.h"
#include "paddle/phi/core/memory/allocation/aligned_allocator.h"
#include "paddle/phi/core/memory/allocation/cuda_allocator.h"

namespace paddle {
namespace memory {
namespace allocation {

namespace {

static const phi::GPUPlace kPlace{0};
static const size_t kAlign = 512;
static const size_t kBlock = 64 * 1024 * 1024;  // 64 MB chunk

std::shared_ptr<AutoGrowthBestFitAllocatorV2> MakeAllocator() {
  auto cuda = std::make_shared<CUDAAllocator>(kPlace);
  auto aligned = std::make_shared<AlignedAllocator>(cuda, kAlign);
  return std::make_shared<AutoGrowthBestFitAllocatorV2>(
      aligned, kAlign, kPlace, kBlock);
}

// RAII guard: restore warmup state after each test.
struct WarmupGuard {
  explicit WarmupGuard(bool warmup) {
    AutoGrowthBestFitAllocatorV2State::GetInstance().SetWarmup(warmup);
  }
  ~WarmupGuard() {
    // Always leave warmup=false so other tests are unaffected.
    AutoGrowthBestFitAllocatorV2State::GetInstance().SetWarmup(false);
  }
};

}  // namespace

// ---------------------------------------------------------------------------
// Warmup: cache miss increments on first allocation (no free block available)
// ---------------------------------------------------------------------------
TEST(AutoGrowthBestFitAllocatorV2Warmup, CacheMissOnFirstAlloc) {
  WarmupGuard guard(true);
  auto allocator = MakeAllocator();

  auto stats_before = allocator->GetStats();
  auto a = allocator->Allocate(kAlign);
  ASSERT_NE(a, nullptr);

  auto stats_after = allocator->GetStats();
  EXPECT_GT(stats_after.cache_miss_count, stats_before.cache_miss_count);
  EXPECT_EQ(stats_after.cache_hit_count, stats_before.cache_hit_count);
}

// ---------------------------------------------------------------------------
// Regular (non-warmup): cache miss counter increments on first allocation.
// Note: V2's FreeImpl is inherited from V1 and populates V1's free-block maps,
// while V2's AllocateImpl searches its own free_blocks_, so a free+realloc
// cycle does NOT produce a cache hit in V2.
// ---------------------------------------------------------------------------
TEST(AutoGrowthBestFitAllocatorV2Regular, CacheMissOnFirstAlloc) {
  WarmupGuard guard(false);
  auto allocator = MakeAllocator();

  auto stats_before = allocator->GetStats();
  auto a = allocator->Allocate(kAlign);
  ASSERT_NE(a, nullptr);

  auto stats_after = allocator->GetStats();
  EXPECT_GT(stats_after.cache_miss_count, stats_before.cache_miss_count);
  EXPECT_EQ(stats_after.cache_hit_count, stats_before.cache_hit_count);
}

}  // namespace allocation
}  // namespace memory
}  // namespace paddle
