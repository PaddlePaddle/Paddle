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

// End-to-end coverage of the memory-history hooks migrated onto the VMM V2
// stack (VMMAutoGrowthBestFitMultiPoolAllocatorV2). Mirrors the V1 coverage in
// memory_history_hook_test.cc so both allocator versions record the same
// alloc / free-completed timeline consumed by the snapshot visualization.

#include <algorithm>
#include <memory>
#include <vector>

#include "gtest/gtest.h"
#include "paddle/phi/core/memory/allocation/cuda_virtual_mem_allocator_v2.h"
#include "paddle/phi/core/memory/allocation/memory_history_recorder.h"
#include "paddle/phi/core/memory/allocation/vmm_auto_growth_best_fit_allocator_v2.h"
#include "paddle/phi/core/memory/allocation/vmm_auto_growth_best_fit_multi_pool_allocator_v2.h"

namespace paddle {
namespace memory {
namespace allocation {

namespace {
constexpr size_t kAlignment = 256;
constexpr size_t kHandleSize = 2UL << 20;
constexpr size_t kSmallLargeThreshold = 2UL << 20;

std::shared_ptr<VMMAutoGrowthBestFitAllocatorV2> CreatePoolAllocator(
    PoolType pool_type) {
  auto underlying = std::make_shared<CUDAVirtualMemAllocatorV2>(
      phi::GPUPlace(0), kHandleSize, pool_type);
  return std::make_shared<VMMAutoGrowthBestFitAllocatorV2>(
      underlying, kAlignment, phi::GPUPlace(0), pool_type);
}
}  // namespace

class MemoryHistoryHookV2Test : public ::testing::Test {
 protected:
  void SetUp() override {
    allocator_ = std::make_shared<VMMAutoGrowthBestFitMultiPoolAllocatorV2>(
        CreatePoolAllocator(PoolType::kSmall),
        CreatePoolAllocator(PoolType::kLarge),
        kSmallLargeThreshold,
        phi::GPUPlace(0));
    MemoryHistoryRecorder::Instance().SetEnabled(false, 0);
  }

  void TearDown() override {
    MemoryHistoryRecorder::Instance().SetEnabled(false, 0);
  }

  // Events of one action, in chronological order.
  std::vector<MemHistoryTraceEntry> EventsOf(MemHistoryAction action) {
    std::vector<MemHistoryTraceEntry> out;
    for (auto& e : MemoryHistoryRecorder::Instance().GetTrace(0)) {
      if (e.action == action) out.push_back(e);
    }
    return out;
  }

  std::shared_ptr<VMMAutoGrowthBestFitMultiPoolAllocatorV2> allocator_;
};

TEST_F(MemoryHistoryHookV2Test, NoEventsWhileDisabled) {
  { auto allocation = allocator_->Allocate(1024); }
  EXPECT_TRUE(MemoryHistoryRecorder::Instance().GetTrace(0).empty());
}

TEST_F(MemoryHistoryHookV2Test, AllocAndFreeAreRecorded) {
  MemoryHistoryRecorder::Instance().SetEnabled(true, 1024);

  uintptr_t addr = 0;
  size_t reported_size = 0;
  {
    auto allocation = allocator_->Allocate(1024);
    addr = reinterpret_cast<uintptr_t>(allocation->ptr());
    reported_size = allocation->size();
    ASSERT_NE(addr, 0u);

    auto allocs = EventsOf(MemHistoryAction::kAlloc);
    ASSERT_EQ(allocs.size(), 1u);
    EXPECT_EQ(allocs[0].addr, addr);
    EXPECT_EQ(allocs[0].size, reported_size);
    EXPECT_EQ(allocs[0].device, 0);
    // Free is only recorded once the allocation goes out of scope.
    EXPECT_TRUE(EventsOf(MemHistoryAction::kFreeCompleted).empty());
  }

  auto frees = EventsOf(MemHistoryAction::kFreeCompleted);
  ASSERT_EQ(frees.size(), 1u);
  EXPECT_EQ(frees[0].addr, addr);
  EXPECT_EQ(frees[0].size, reported_size);
}

TEST_F(MemoryHistoryHookV2Test, AllocAndFreeSizesAgreeForTinyRequest) {
  // Regression guard: the alloc hook must report the actual block size, not the
  // caller's request. A 1-byte request is rounded up to the allocator alignment
  // (256B on GPU), and the free hook reports allocation->size(); recording the
  // raw request would make the pair disagree (alloc=1 vs free=256) and would
  // under-state occupancy in the address-space view.
  MemoryHistoryRecorder::Instance().SetEnabled(true, 1024);

  { auto allocation = allocator_->Allocate(1); }

  auto allocs = EventsOf(MemHistoryAction::kAlloc);
  auto frees = EventsOf(MemHistoryAction::kFreeCompleted);
  ASSERT_EQ(allocs.size(), 1u);
  ASSERT_EQ(frees.size(), 1u);
  EXPECT_EQ(allocs[0].size, frees[0].size);
  EXPECT_EQ(allocs[0].addr, frees[0].addr);
  EXPECT_GE(allocs[0].size, kAlignment);
  EXPECT_EQ(allocs[0].size % kAlignment, 0u);
}

TEST_F(MemoryHistoryHookV2Test, OpLabelIsAttachedToAllocation) {
  MemoryHistoryRecorder::Instance().SetEnabled(true, 1024);
  {
    MemLabelGuard label("matmul");
    auto allocation = allocator_->Allocate(4096);
  }
  auto allocs = EventsOf(MemHistoryAction::kAlloc);
  ASSERT_EQ(allocs.size(), 1u);
  EXPECT_EQ(allocs[0].op_name, "matmul");
}

TEST_F(MemoryHistoryHookV2Test, SmallAndLargeRequestsAreBothRecorded) {
  MemoryHistoryRecorder::Instance().SetEnabled(true, 1024);
  {
    // Routed to the small and the large pool respectively; both go through the
    // same hook in VMMAutoGrowthBestFitMultiPoolAllocatorV2.
    auto small = allocator_->Allocate(64);
    auto large = allocator_->Allocate(4 << 20);
    EXPECT_NE(small->ptr(), large->ptr());
  }
  EXPECT_EQ(EventsOf(MemHistoryAction::kAlloc).size(), 2u);
  EXPECT_EQ(EventsOf(MemHistoryAction::kFreeCompleted).size(), 2u);
}

TEST_F(MemoryHistoryHookV2Test, EventsAreOrderedAndTimestamped) {
  MemoryHistoryRecorder::Instance().SetEnabled(true, 1024);
  { auto allocation = allocator_->Allocate(2048); }

  auto trace = MemoryHistoryRecorder::Instance().GetTrace(0);
  ASSERT_GE(trace.size(), 2u);
  EXPECT_EQ(trace.front().action, MemHistoryAction::kAlloc);
  EXPECT_EQ(trace.back().action, MemHistoryAction::kFreeCompleted);
  EXPECT_GT(trace.front().time_us, 0u);
  EXPECT_TRUE(std::is_sorted(
      trace.begin(), trace.end(), [](const auto& a, const auto& b) {
        return a.time_us < b.time_us;
      }));
}

}  // namespace allocation
}  // namespace memory
}  // namespace paddle
