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

#include <cstdint>
#include <stdexcept>
#include <utility>
#include <vector>

#include "glog/logging.h"
#include "gtest/gtest.h"

#include "paddle/phi/core/enforce.h"
#define private public
#include "paddle/phi/core/memory/allocation/cuda_virtual_mem_allocator_v2.h"
#include "paddle/phi/core/memory/allocation/remap_transaction.h"
#undef private
#include "paddle/phi/core/memory/allocation/vmm_backing_map.h"

namespace paddle {
namespace memory {
namespace allocation {

namespace {

class ScopedVLogLevel {
 public:
  explicit ScopedVLogLevel(int level) : old_level_(FLAGS_v) { FLAGS_v = level; }
  ~ScopedVLogLevel() { FLAGS_v = old_level_; }

 private:
  int old_level_;
};

__global__ void VMMBackingMapBusyWaitKernel(uint64_t cycles) {
  const auto start = clock64();
  while (clock64() - start < cycles) {
  }
}

}  // namespace

TEST(VMMBackingMap, TracksMappedAndUnmappedRanges) {
  ScopedVLogLevel vlog_guard(6);
  VMMBackingMap map;
  const VMMDevicePtr base = 0x10000000;
  const size_t page_size = 2UL << 20;
  map.Configure(base, page_size * 4, page_size, 0);

  EXPECT_TRUE(map.IsRangeUnmapped(base, page_size * 4));
  EXPECT_FALSE(map.IsRangeMapped(base, page_size));
  EXPECT_EQ(map.total_mapped_bytes(), 0UL);

  const VMMAllocHandle first_handle = static_cast<VMMAllocHandle>(0x101);
  const VMMAllocHandle second_handle = static_cast<VMMAllocHandle>(0x102);
  map.MarkMapped(base, first_handle, page_size);
  map.MarkMapped(base + page_size, second_handle, page_size);
  map.MarkMapped(base, first_handle, page_size);
  EXPECT_EQ(map.total_mapped_bytes(), page_size * 2);
  EXPECT_TRUE(map.IsRangeReleasable(base, page_size * 4));
  EXPECT_FALSE(map.IsRangeReleasable(base - page_size, page_size));

  std::vector<std::pair<VMMDevicePtr, size_t>> free_ranges = {
      {base, page_size}, {base + page_size, page_size * 3}};
  auto mapped_pages = map.CollectMappedPages(free_ranges);
  ASSERT_EQ(mapped_pages.size(), 2UL);
  EXPECT_EQ(mapped_pages[0].va, base);
  EXPECT_EQ(mapped_pages[0].handle, first_handle);
  EXPECT_EQ(mapped_pages[1].va, base + page_size);
  EXPECT_EQ(mapped_pages[1].handle, second_handle);
  mapped_pages = map.CollectMappedPages(free_ranges, page_size + 1);
  ASSERT_EQ(mapped_pages.size(), 2UL);
  EXPECT_EQ(mapped_pages[0].va, base);
  EXPECT_EQ(mapped_pages[1].va, base + page_size);
  mapped_pages = map.CollectMappedPages(free_ranges, page_size);
  ASSERT_EQ(mapped_pages.size(), 1UL);
  EXPECT_EQ(mapped_pages[0].va, base);
  const auto two_page_snapshot = map.CollectMappedPages(free_ranges);
  ASSERT_EQ(two_page_snapshot.size(), 2UL);
  EXPECT_TRUE(map.ValidateMappedPages(two_page_snapshot, "unit_test"));

  EXPECT_TRUE(map.IsRangeMapped(base, page_size * 2));
  EXPECT_FALSE(map.IsRangeMapped(base, page_size * 3));
  EXPECT_FALSE(map.IsRangeUnmapped(base, page_size));
  EXPECT_TRUE(map.IsRangeUnmapped(base + page_size * 2, page_size * 2));
  EXPECT_EQ(map.total_mapped_bytes(), page_size * 2);

  map.MarkUnmapped(base, page_size);
  map.MarkUnmapped(base, page_size);
  EXPECT_FALSE(map.ValidateMappedPages(two_page_snapshot, "unit_test_stale"));
  std::vector<std::pair<VMMDevicePtr, size_t>> unmapped_base_range = {
      {base, page_size}};
  auto unmapped_base_pages =
      map.CollectUnmappedPagesFullyInRange(unmapped_base_range);
  ASSERT_EQ(unmapped_base_pages.size(), 1UL);
  EXPECT_TRUE(map.ValidateUnmappedPages(unmapped_base_pages,
                                        "unit_test_unmapped_clears_handle"));
  mapped_pages = map.CollectMappedPages(free_ranges);
  ASSERT_EQ(mapped_pages.size(), 1UL);
  EXPECT_EQ(mapped_pages[0].va, base + page_size);
  EXPECT_EQ(mapped_pages[0].handle, second_handle);
  mapped_pages = map.CollectMappedPages(free_ranges, page_size * 4);
  ASSERT_EQ(mapped_pages.size(), 1UL);
  EXPECT_EQ(mapped_pages[0].va, base + page_size);
  std::vector<std::pair<VMMDevicePtr, size_t>> unaligned_free_ranges = {
      {base + page_size / 2, page_size * 3}};
  mapped_pages = map.CollectMappedPagesFullyInRange(unaligned_free_ranges);
  ASSERT_EQ(mapped_pages.size(), 1UL);
  EXPECT_EQ(mapped_pages[0].va, base + page_size);
  EXPECT_EQ(mapped_pages[0].handle, second_handle);
  mapped_pages =
      map.CollectMappedPagesFullyInRange(unaligned_free_ranges, page_size);
  ASSERT_EQ(mapped_pages.size(), 1UL);
  EXPECT_EQ(mapped_pages[0].va, base + page_size);
  auto unmapped_pages =
      map.CollectUnmappedPagesFullyInRange(unaligned_free_ranges);
  ASSERT_EQ(unmapped_pages.size(), 1UL);
  EXPECT_EQ(unmapped_pages[0].va, base + page_size * 2);
  EXPECT_TRUE(map.ValidateUnmappedPages(unmapped_pages, "unit_test_unmapped"));
  unmapped_pages =
      map.CollectUnmappedPagesFullyInRange(unaligned_free_ranges, page_size);
  ASSERT_EQ(unmapped_pages.size(), 1UL);
  EXPECT_EQ(unmapped_pages[0].va, base + page_size * 2);
  map.MarkIPCExported(base + page_size, page_size);
  EXPECT_FALSE(map.HasIPCExportedPages(base, page_size));
  EXPECT_TRUE(map.HasIPCExportedPages(base + page_size, page_size));
  EXPECT_TRUE(map.HasIPCExportedPages(base, page_size * 2));
  EXPECT_FALSE(map.IsRangeReleasable(base, page_size * 2));
  mapped_pages = map.CollectMappedPagesFullyInRange(unaligned_free_ranges);
  EXPECT_TRUE(mapped_pages.empty());
  EXPECT_FALSE(map.IsRangeMapped(base, page_size * 2));
  EXPECT_TRUE(map.IsRangeUnmapped(base, page_size));
  EXPECT_TRUE(map.IsRangeMapped(base + page_size, page_size));
  EXPECT_EQ(map.total_mapped_bytes(), page_size);

  map.MarkReleased(base + page_size, second_handle, page_size);
  map.MarkReleased(base + page_size, second_handle, page_size);
  EXPECT_TRUE(map.IsRangeUnmapped(base, page_size * 4));
  EXPECT_TRUE(map.IsRangeReleasable(base, page_size * 4));
  EXPECT_EQ(map.total_mapped_bytes(), 0UL);
  std::vector<std::pair<VMMDevicePtr, size_t>> all_ranges = {
      {base, page_size * 4}};
  auto all_unmapped_pages =
      map.CollectUnmappedPagesFullyInRange(all_ranges, page_size * 2);
  ASSERT_EQ(all_unmapped_pages.size(), 2UL);
  EXPECT_TRUE(
      map.ValidateUnmappedPages(all_unmapped_pages, "unit_test_all_unmapped"));
  const VMMAllocHandle third_handle = static_cast<VMMAllocHandle>(0x103);
  map.MarkMapped(base, third_handle, page_size);
  EXPECT_FALSE(map.ValidateUnmappedPages(all_unmapped_pages,
                                         "unit_test_unmapped_stale"));
}

TEST(VMMBackingMap, RejectsMappedPageHandleOverwrite) {
  ScopedVLogLevel vlog_guard(6);
  VMMBackingMap map;
  const VMMDevicePtr base = 0x18000000;
  const size_t page_size = 2UL << 20;
  map.Configure(base, page_size, page_size, 0);

  const VMMAllocHandle first_handle = static_cast<VMMAllocHandle>(0x181);
  const VMMAllocHandle second_handle = static_cast<VMMAllocHandle>(0x182);
  map.MarkMapped(base, first_handle, page_size);
  EXPECT_THROW(map.MarkMapped(base, second_handle, page_size),
               common::enforce::EnforceNotMet);

  map.MarkUnmapped(base, page_size);
  auto meta = std::make_shared<VMMHandleMeta>(base, page_size, first_handle, 0);
  map.MarkMapped(base, meta, page_size);
  auto other_meta =
      std::make_shared<VMMHandleMeta>(base, page_size, second_handle, 0);
  EXPECT_THROW(map.MarkMapped(base, other_meta, page_size),
               common::enforce::EnforceNotMet);
}

TEST(VMMBackingMap, RejectsInvalidConfiguration) {
  ScopedVLogLevel vlog_guard(6);
  VMMBackingMap map;
  const VMMDevicePtr base = 0x1a000000;
  const size_t page_size = 2UL << 20;

  EXPECT_THROW(map.Configure(base, page_size, 0, 0),
               common::enforce::EnforceNotMet);
  EXPECT_THROW(map.Configure(base, page_size + 1, page_size, 0),
               common::enforce::EnforceNotMet);
}

TEST(VMMBackingMap, ReconfigureKeepsOriginalLayout) {
  ScopedVLogLevel vlog_guard(6);
  VMMBackingMap map;
  const VMMDevicePtr base = 0x1c000000;
  const size_t page_size = 2UL << 20;

  map.Configure(base, page_size * 2, page_size, 0);
  EXPECT_TRUE(map.IsConfigured());
  map.Configure(base, page_size * 2, page_size, 0);
  map.Configure(base + page_size, page_size * 2, page_size, 1);

  EXPECT_TRUE(map.IsRangeUnmapped(base, page_size * 2));
  EXPECT_FALSE(map.IsRangeUnmapped(base + page_size * 2, page_size));
}

TEST(VMMBackingMap, UnconfiguredAndInvalidRangesReturnFalse) {
  ScopedVLogLevel vlog_guard(6);
  VMMBackingMap map;
  const VMMDevicePtr base = 0x1e000000;
  const size_t page_size = 2UL << 20;
  auto meta = std::make_shared<VMMHandleMeta>(
      base, page_size, static_cast<VMMAllocHandle>(0x1e1), 0);
  HandleLayout layout{meta};

  EXPECT_FALSE(map.IsRangeMapped(base, page_size));
  EXPECT_FALSE(map.IsRangeUnmapped(base, page_size));
  EXPECT_FALSE(map.IsRangeReleasable(base, page_size));
  EXPECT_FALSE(map.ValidateMappedPages(
      std::vector<VMMBackingMap::MappedPage>{{base, meta->handle(), meta, 0}},
      "unconfigured"));
  EXPECT_EQ(map.ClearRemapDestinationOwnershipInRange(base, page_size), 0UL);
  map.MarkRemapDestinationMapped(base, meta, page_size);
  map.MarkIPCExported(base, page_size);
  map.MarkPendingEvent(base, page_size, nullptr, nullptr);
  EXPECT_TRUE(map.CollectMappedPagesFullyInRange({{base, page_size}}).empty());
  EXPECT_TRUE(
      map.CollectUnmappedPagesFullyInRange({{base, page_size}}).empty());
  map.MarkMapped(base, meta, page_size);
  map.MarkUnmapped(base, page_size);
  map.MarkReleased(base, meta->handle(), page_size);

  map.Configure(base, page_size * 2, page_size, 0);
  map.MarkIPCExported(base, page_size);
  map.MarkPendingEvent(base, page_size, nullptr, nullptr);
  EXPECT_TRUE(map.MarkPendingEventForRange(base, page_size, nullptr, nullptr));
  EXPECT_TRUE(map.MarkPendingEventForRange(base, page_size, nullptr, nullptr));
  EXPECT_FALSE(map.IsRangeMapped(base, page_size / 2));
  EXPECT_FALSE(map.IsRangeUnmapped(base - page_size, page_size));
  EXPECT_FALSE(map.IsRangeReleasable(base + page_size * 2, page_size));
}

TEST(VMMBackingMap, ValidateMappedPagesDetectsMissingAndMismatchedPages) {
  ScopedVLogLevel vlog_guard(6);
  VMMBackingMap map;
  const VMMDevicePtr base = 0x22000000;
  const size_t page_size = 2UL << 20;
  map.Configure(base, page_size * 2, page_size, 0);

  auto first = std::make_shared<VMMHandleMeta>(
      base, page_size, static_cast<VMMAllocHandle>(0x221), 0);
  auto second = std::make_shared<VMMHandleMeta>(
      base + page_size, page_size, static_cast<VMMAllocHandle>(0x222), 0);
  EXPECT_FALSE(map.ValidateMappedPages(
      std::vector<VMMBackingMap::MappedPage>{{base, first->handle(), first, 0}},
      "missing mapped page"));
  EXPECT_FALSE(map.ValidateLayout(HandleLayout{first}, "missing layout"));

  map.MarkMapped(base, first, page_size);
  map.MarkMapped(base + page_size, second, page_size);
  auto wrong = std::make_shared<VMMHandleMeta>(
      base + page_size, page_size, static_cast<VMMAllocHandle>(0x223), 0);
  EXPECT_FALSE(map.ValidateLayout(HandleLayout{wrong}, "layout mismatch"));
  EXPECT_FALSE(map.ValidateMappedPages(
      std::vector<VMMBackingMap::MappedPage>{
          {base, first->handle(), first, 0},
          {base + page_size, wrong->handle(), wrong, 0}},
      "handle mismatch"));
}

TEST(VMMBackingMap, MarkReleasedAllowsMismatchAndIPCBlocksReleasableRange) {
  ScopedVLogLevel vlog_guard(6);
  VMMBackingMap map;
  const VMMDevicePtr base = 0x24000000;
  const size_t page_size = 2UL << 20;
  map.Configure(base, page_size * 2, page_size, 0);

  map.MarkMapped(base, static_cast<VMMAllocHandle>(0), page_size);
  EXPECT_TRUE(map.IsRangeMapped(base, page_size));
  EXPECT_TRUE(map.IsRangeReleasable(base, page_size));
  map.MarkIPCExported(base, page_size);
  EXPECT_FALSE(map.IsRangeReleasable(base, page_size));

  auto meta = std::make_shared<VMMHandleMeta>(
      base + page_size, page_size, static_cast<VMMAllocHandle>(0x241), 0);
  map.MarkMapped(base + page_size, meta, page_size);
  map.MarkReleased(
      base + page_size, static_cast<VMMAllocHandle>(0x242), page_size);
  EXPECT_TRUE(map.IsRangeUnmapped(base + page_size, page_size));
}

TEST(VMMBackingMap, CanReleaseHandleChecksPageState) {
  ScopedVLogLevel vlog_guard(6);
  VMMBackingMap map;
  const VMMDevicePtr base = 0x24800000;
  const size_t page_size = 2UL << 20;
  map.Configure(base, page_size * 2, page_size, 0);

  auto meta = std::make_shared<VMMHandleMeta>(
      base, page_size, static_cast<VMMAllocHandle>(0x2481), 0);
  auto other_meta = std::make_shared<VMMHandleMeta>(
      base, page_size, static_cast<VMMAllocHandle>(0x2482), 0);

  map.MarkMapped(base, meta, page_size);
  EXPECT_TRUE(map.CanReleaseHandle(base, meta->handle(), meta, page_size));
  EXPECT_FALSE(
      map.CanReleaseHandle(base - page_size, meta->handle(), meta, page_size));
  EXPECT_FALSE(map.CanReleaseHandle(
      base, static_cast<VMMAllocHandle>(0x9999), meta, page_size));
  EXPECT_FALSE(
      map.CanReleaseHandle(base, meta->handle(), other_meta, page_size));

  map.MarkIPCExported(base, page_size);
  EXPECT_FALSE(map.CanReleaseHandle(base, meta->handle(), meta, page_size));

  map.MarkReleased(base, meta->handle(), page_size);
  map.MarkMapped(base, meta, page_size);

  gpuStream_t busy_stream;
  ASSERT_EQ(cudaStreamCreate(&busy_stream), cudaSuccess);
  VMMBackingMapBusyWaitKernel<<<1, 1, 0, busy_stream>>>(500000000ULL);
  ASSERT_EQ(cudaGetLastError(), cudaSuccess);
  ASSERT_TRUE(
      map.MarkPendingEventForRange(base, page_size, busy_stream, nullptr));
  EXPECT_FALSE(map.CanReleaseHandle(base, meta->handle(), meta, page_size));

  ASSERT_EQ(cudaStreamSynchronize(busy_stream), cudaSuccess);
  EXPECT_TRUE(map.CanReleaseHandle(base, meta->handle(), meta, page_size));
  ASSERT_EQ(cudaStreamDestroy(busy_stream), cudaSuccess);

  map.MarkUnmapped(base, page_size);
  map.MarkUnmapped(base, page_size);
  EXPECT_FALSE(map.CanReleaseHandle(base, meta->handle(), meta, page_size));

  map.MarkMapped(base + page_size, meta, page_size);
  map.MarkReleased(
      base + page_size, static_cast<VMMAllocHandle>(0x9999), page_size);
}

TEST(VMMBackingMap, IPCDescriptorsAndExportedBytesUsePageGranularity) {
  ScopedVLogLevel vlog_guard(6);
  VMMBackingMap map;
  const VMMDevicePtr base = 0x25000000;
  const size_t page_size = 2UL << 20;
  map.Configure(base, page_size * 4, page_size, 0);

  auto first = std::make_shared<VMMHandleMeta>(
      base, page_size, static_cast<VMMAllocHandle>(0x251), 0);
  auto second = std::make_shared<VMMHandleMeta>(
      base + page_size, page_size, static_cast<VMMAllocHandle>(0x252), 0);
  auto third = std::make_shared<VMMHandleMeta>(
      base + page_size * 2, page_size, static_cast<VMMAllocHandle>(0x253), 0);
  map.MarkMapped(base, first, page_size);
  map.MarkMapped(base + page_size, second, page_size);
  map.MarkMapped(base + page_size * 2, third, page_size);

  std::vector<IPCPartDescriptor> descriptors;
  ASSERT_TRUE(map.CollectIPCPartDescriptors(
      base + page_size / 2, page_size + page_size / 2, &descriptors));
  ASSERT_EQ(descriptors.size(), 2UL);
  EXPECT_EQ(descriptors[0].handle_base, first->base());
  EXPECT_EQ(descriptors[0].handle_size, first->size());
  EXPECT_EQ(descriptors[0].handle_rel_off, page_size / 2);
  EXPECT_EQ(descriptors[0].len, page_size / 2);
  EXPECT_EQ(descriptors[1].handle_base, second->base());
  EXPECT_EQ(descriptors[1].handle_rel_off, 0UL);
  EXPECT_EQ(descriptors[1].len, page_size);
  EXPECT_TRUE(map.CollectIPCPartDescriptors(
      base + page_size / 2, page_size + page_size / 2, nullptr));

  map.MarkIPCExported(base + page_size / 4, page_size + page_size / 2);
  EXPECT_EQ(map.CountIPCExportedBytes({{base + page_size / 2, page_size}}),
            page_size);
  EXPECT_EQ(map.CountIPCExportedBytes({{base - page_size, page_size}}), 0UL);

  map.MarkUnmapped(base + page_size * 2, page_size);
  EXPECT_FALSE(map.CollectIPCPartDescriptors(
      base + page_size, page_size * 2, &descriptors));
}

TEST(VMMBackingMap, RemapDestinationOwnershipClearsFullyCoveredPagesOnly) {
  ScopedVLogLevel vlog_guard(6);
  VMMBackingMap map;
  const VMMDevicePtr base = 0x27000000;
  const size_t page_size = 2UL << 20;
  map.Configure(base, page_size * 4, page_size, 0);

  std::vector<std::shared_ptr<VMMHandleMeta>> metas;
  for (size_t i = 0; i < 3; ++i) {
    auto meta =
        std::make_shared<VMMHandleMeta>(base + i * page_size,
                                        page_size,
                                        static_cast<VMMAllocHandle>(0x271 + i),
                                        0);
    meta->MarkOwnedByRemapDestination();
    metas.push_back(meta);
    map.MarkRemapDestinationMapped(base + i * page_size, meta, page_size);
  }

  EXPECT_EQ(
      map.ClearRemapDestinationOwnershipInRange(base - page_size, page_size),
      0UL);
  EXPECT_EQ(map.ClearRemapDestinationOwnershipInRange(base + page_size / 2,
                                                      page_size * 2),
            page_size);
  std::vector<std::pair<VMMDevicePtr, size_t>> ranges = {{base, page_size * 3}};
  auto pages = map.CollectRemapSourcePagesFullyInRange(ranges, 0);
  ASSERT_EQ(pages.size(), 3UL);
  EXPECT_EQ(pages[0].remap_source_state,
            VMMBackingMap::RemapSourceState::kRemapDestinationOwned);
  EXPECT_EQ(pages[1].remap_source_state,
            VMMBackingMap::RemapSourceState::kReady);
  EXPECT_EQ(pages[2].remap_source_state,
            VMMBackingMap::RemapSourceState::kRemapDestinationOwned);
  EXPECT_TRUE(metas[0]->IsOwnedByRemapDestination());
  EXPECT_FALSE(metas[1]->IsOwnedByRemapDestination());
  EXPECT_TRUE(metas[2]->IsOwnedByRemapDestination());

  EXPECT_TRUE(map.ClearRemapDestinationOwnership(base, page_size));
  EXPECT_FALSE(metas[0]->IsOwnedByRemapDestination());
  map.MarkUnmapped(base + page_size * 2, page_size);
  EXPECT_FALSE(
      map.ClearRemapDestinationOwnership(base + page_size * 2, page_size));
}

TEST(BlockV2, UnmappedSubBlockTrimAndMerge) {
  auto* base = reinterpret_cast<void*>(0x26000000);
  BlockV2 block =
      BlockV2::MakeUnmappedFreeBlock(base, 8UL << 20, PoolType::kSmall);

  BlockV2 sub_block = block.MakeUnmappedFreeSubBlock(2UL << 20, 4UL << 20);
  EXPECT_TRUE(sub_block.IsUnmappedFree());
  EXPECT_EQ(sub_block.ptr(), reinterpret_cast<uint8_t*>(base) + (2UL << 20));
  EXPECT_EQ(sub_block.size(), 4UL << 20);
  EXPECT_EQ(sub_block.pool_type_, PoolType::kSmall);

  sub_block.TrimToSuffix(1UL << 20, 3UL << 20);
  EXPECT_EQ(sub_block.ptr(), reinterpret_cast<uint8_t*>(base) + (3UL << 20));
  EXPECT_EQ(sub_block.size(), 3UL << 20);

  BlockV2 next = BlockV2::MakeUnmappedFreeBlock(
      reinterpret_cast<uint8_t*>(base) + (6UL << 20),
      2UL << 20,
      PoolType::kSmall);
  sub_block.MergeAdjacentUnmappedFreeBlock(next);
  EXPECT_EQ(sub_block.size(), 5UL << 20);

  std::vector<BlockV2> restore_segments;
  EXPECT_EQ(block.BuildRestoreMappedFreeSegments(
                reinterpret_cast<VMMDevicePtr>(base) - (1UL << 20),
                2UL << 20,
                &restore_segments),
            BlockRestoreMappedFreeResult::kOutside);
  EXPECT_EQ(block.BuildRestoreMappedFreeSegments(
                reinterpret_cast<VMMDevicePtr>(base) + (6UL << 20),
                4UL << 20,
                &restore_segments),
            BlockRestoreMappedFreeResult::kRangeExceedsBlock);
  EXPECT_EQ(block.BuildRestoreMappedFreeSegments(
                reinterpret_cast<VMMDevicePtr>(base) + (2UL << 20),
                2UL << 20,
                &restore_segments),
            BlockRestoreMappedFreeResult::kBuilt);
  ASSERT_EQ(restore_segments.size(), 3UL);
  EXPECT_TRUE(restore_segments[0].IsUnmappedFree());
  EXPECT_EQ(restore_segments[0].size(), 2UL << 20);
  EXPECT_TRUE(restore_segments[1].IsMappedFree());
  EXPECT_EQ(restore_segments[1].ptr(),
            reinterpret_cast<uint8_t*>(base) + (2UL << 20));
  EXPECT_EQ(restore_segments[1].size(), 2UL << 20);
  EXPECT_TRUE(restore_segments[2].IsUnmappedFree());
  EXPECT_EQ(restore_segments[2].size(), 4UL << 20);

  BlockV2 safety = BlockV2::MakeMappedBlock(
      BlockType::kFree, base, 8UL << 20, PoolType::kSmall);
  auto first_stream = reinterpret_cast<gpuStream_t>(0x1);
  auto second_stream = reinterpret_cast<gpuStream_t>(0x2);
  safety.AppendRemapSafety(first_stream, nullptr);
  EXPECT_EQ(safety.owning_stream_, first_stream);
  safety.AppendRemapSafety(second_stream, nullptr);
  ASSERT_EQ(safety.remap_pending_states_.size(), 1UL);
  safety.AppendRemapSafety(second_stream, nullptr);
  EXPECT_EQ(safety.remap_pending_states_.size(), 1UL);

  BlockV2 additional = BlockV2::MakeMappedBlock(
      BlockType::kFree, base, 8UL << 20, PoolType::kSmall);
  additional.AppendRemapSafety(reinterpret_cast<gpuStream_t>(0x3), nullptr);
  additional.AppendRemapSafety(reinterpret_cast<gpuStream_t>(0x4), nullptr);
  safety.AppendRemapSafetyFrom(additional);
  EXPECT_EQ(safety.remap_pending_states_.size(), 3UL);
}

TEST(VMMBackingMap, ReplacesPendingEventForSameStream) {
  ScopedVLogLevel vlog_guard(6);
  VMMBackingMap map;
  const VMMDevicePtr base = 0x20000000;
  const size_t page_size = 2UL << 20;
  map.Configure(base, page_size, page_size, 0);
  auto meta = std::make_shared<VMMHandleMeta>(
      base, page_size, static_cast<VMMAllocHandle>(0x201), 0);
  map.MarkMapped(base, meta, page_size);

  {
    gpuEvent_t scoped_event;
    ASSERT_EQ(cudaEventCreateWithFlags(&scoped_event, cudaEventDisableTiming),
              cudaSuccess);
    CUDAEventGuard scoped_guard(scoped_event);
  }

  gpuStream_t key_stream;
  gpuStream_t busy_stream;
  ASSERT_EQ(cudaStreamCreate(&key_stream), cudaSuccess);
  ASSERT_EQ(cudaStreamCreate(&busy_stream), cudaSuccess);

  VMMBackingMapBusyWaitKernel<<<1, 1, 0, busy_stream>>>(500000000ULL);
  ASSERT_EQ(cudaGetLastError(), cudaSuccess);
  gpuEvent_t pending_event;
  ASSERT_EQ(cudaEventCreateWithFlags(&pending_event, cudaEventDisableTiming),
            cudaSuccess);
  ASSERT_EQ(cudaEventRecord(pending_event, busy_stream), cudaSuccess);
  map.MarkPendingEvent(base,
                       page_size,
                       key_stream,
                       std::make_shared<CUDAEventGuard>(pending_event));

  gpuEvent_t ready_event;
  ASSERT_EQ(cudaEventCreateWithFlags(&ready_event, cudaEventDisableTiming),
            cudaSuccess);
  ASSERT_EQ(cudaEventRecord(ready_event, key_stream), cudaSuccess);
  ASSERT_EQ(cudaEventSynchronize(ready_event), cudaSuccess);
  map.MarkPendingEvent(base,
                       page_size,
                       key_stream,
                       std::make_shared<CUDAEventGuard>(ready_event));

  std::vector<std::pair<VMMDevicePtr, size_t>> ranges = {{base, page_size},
                                                         {base, page_size}};
  auto mapped_snapshot = map.CollectMappedPages(ranges);
  ASSERT_EQ(mapped_snapshot.size(), 2UL);
  auto pages = map.CollectRemapSourcePagesFullyInRange(ranges, page_size);
  ASSERT_EQ(pages.size(), 1UL);
  EXPECT_EQ(pages[0].remap_source_state,
            VMMBackingMap::RemapSourceState::kReady);
  EXPECT_TRUE(map.ValidateMappedPages(mapped_snapshot,
                                      "unit_test_ready_event_gc_keeps_epoch"));

  ASSERT_EQ(cudaStreamSynchronize(busy_stream), cudaSuccess);
  ASSERT_EQ(cudaStreamDestroy(key_stream), cudaSuccess);
  ASSERT_EQ(cudaStreamDestroy(busy_stream), cudaSuccess);
}

TEST(VMMBackingMap, MarksPendingEventForUnalignedRangeOnce) {
  ScopedVLogLevel vlog_guard(6);
  VMMBackingMap map;
  const VMMDevicePtr base = 0x24000000;
  const size_t page_size = 2UL << 20;
  map.Configure(base, page_size * 2, page_size, 0);
  auto first_meta = std::make_shared<VMMHandleMeta>(
      base, page_size, static_cast<VMMAllocHandle>(0x241), 0);
  auto second_meta = std::make_shared<VMMHandleMeta>(
      base + page_size, page_size, static_cast<VMMAllocHandle>(0x242), 0);
  map.MarkMapped(base, first_meta, page_size);
  map.MarkMapped(base + page_size, second_meta, page_size);

  gpuStream_t busy_stream;
  ASSERT_EQ(cudaStreamCreate(&busy_stream), cudaSuccess);
  VMMBackingMapBusyWaitKernel<<<1, 1, 0, busy_stream>>>(500000000ULL);
  ASSERT_EQ(cudaGetLastError(), cudaSuccess);

  gpuEvent_t pending_event;
  ASSERT_EQ(cudaEventCreateWithFlags(&pending_event, cudaEventDisableTiming),
            cudaSuccess);
  ASSERT_EQ(cudaEventRecord(pending_event, busy_stream), cudaSuccess);
  EXPECT_TRUE(map.MarkPendingEventForRange(
      base + 128UL,
      page_size,
      busy_stream,
      std::make_shared<CUDAEventGuard>(pending_event)));

  std::vector<std::pair<VMMDevicePtr, size_t>> ranges = {{base, page_size * 2}};
  auto pages = map.CollectRemapSourcePagesFullyInRange(ranges, page_size * 2);
  ASSERT_EQ(pages.size(), 2UL);
  EXPECT_EQ(pages[0].remap_source_state,
            VMMBackingMap::RemapSourceState::kPendingEvent);
  EXPECT_EQ(pages[1].remap_source_state,
            VMMBackingMap::RemapSourceState::kPendingEvent);

  ASSERT_EQ(cudaStreamSynchronize(busy_stream), cudaSuccess);
  pages = map.CollectRemapSourcePagesFullyInRange(ranges, page_size * 2);
  ASSERT_EQ(pages.size(), 2UL);
  EXPECT_EQ(pages[0].remap_source_state,
            VMMBackingMap::RemapSourceState::kReady);
  EXPECT_EQ(pages[1].remap_source_state,
            VMMBackingMap::RemapSourceState::kReady);
  ASSERT_EQ(cudaStreamDestroy(busy_stream), cudaSuccess);
}

TEST(CUDAVirtualMemAllocatorV2, AppendWithBlockReturnsMappedFreeBlock) {
  CUDAVirtualMemAllocatorV2 allocator(
      phi::GPUPlace(), 2UL << 20, PoolType::kLarge);
  EXPECT_FALSE(allocator.IsAllocThreadSafe());

  auto allocation_with_block =
      allocator.AppendWithBlock(allocator.handle_size() * 2);
  ASSERT_NE(allocation_with_block.allocation, nullptr);

  const auto& block = allocation_with_block.block;
  ASSERT_EQ(block.size_, allocation_with_block.allocation->size());
  EXPECT_TRUE(block.IsMappedFree());

  auto base =
      reinterpret_cast<VMMDevicePtr>(allocation_with_block.allocation->ptr());
  std::vector<std::pair<VMMDevicePtr, size_t>> ranges = {
      {base, allocation_with_block.allocation->size()}};
  auto pages = allocator.CollectMappedPages(
      ranges, allocation_with_block.allocation->size());
  ASSERT_EQ(pages.size(), 2UL);
  for (size_t i = 0; i < pages.size(); ++i) {
    EXPECT_EQ(pages[i].va, base + i * allocator.handle_size());
    ASSERT_NE(pages[i].meta, nullptr);
    EXPECT_EQ(pages[i].meta->base(), pages[i].va);
    EXPECT_EQ(pages[i].meta->size(), allocator.handle_size());
  }
}

TEST(CUDAVirtualMemAllocatorV2,
     SmallPoolAppendWithBlockReturnsMappedFreeBlock) {
  CUDAVirtualMemAllocatorV2 allocator(
      phi::GPUPlace(), 2UL << 20, PoolType::kSmall);

  auto allocation_with_block =
      allocator.AppendWithBlock(allocator.handle_size());
  ASSERT_NE(allocation_with_block.allocation, nullptr);

  const auto& block = allocation_with_block.block;
  EXPECT_EQ(allocator.pool_type(), PoolType::kSmall);
  EXPECT_EQ(block.ptr_, allocation_with_block.allocation->ptr());
  EXPECT_EQ(block.size_, allocator.handle_size());
  EXPECT_TRUE(block.IsMappedFree());
}

TEST(CUDAVirtualMemAllocatorV2, CollectAllocationHandleLayoutTracksLifecycle) {
  CUDAVirtualMemAllocatorV2 allocator(
      phi::GPUPlace(), 2UL << 20, PoolType::kLarge);

  HandleLayout layout;
  EXPECT_FALSE(allocator.CollectAllocationHandleLayout(
      reinterpret_cast<void*>(0x1234), &layout));

  auto allocation_with_block =
      allocator.AppendWithBlock(allocator.handle_size() * 2);
  ASSERT_NE(allocation_with_block.allocation, nullptr);
  void* ptr = allocation_with_block.allocation->ptr();

  ASSERT_TRUE(allocator.CollectAllocationHandleLayout(ptr, &layout));
  ASSERT_EQ(layout.size(), 2UL);
  EXPECT_EQ(layout.front()->base(), reinterpret_cast<VMMDevicePtr>(ptr));
  EXPECT_EQ(layout.front()->size(), allocator.handle_size());

  layout.clear();
  allocation_with_block.allocation.reset();
  EXPECT_FALSE(allocator.CollectAllocationHandleLayout(ptr, &layout));
}

TEST(CUDAVirtualMemAllocatorV2, AllocateImplReturnsTrackedAllocation) {
  CUDAVirtualMemAllocatorV2 allocator(
      phi::GPUPlace(), 2UL << 20, PoolType::kLarge);

  auto allocation = allocator.Allocate(allocator.handle_size());
  ASSERT_NE(allocation, nullptr);
  EXPECT_NE(allocation->ptr(), nullptr);
  EXPECT_EQ(allocation->size(), allocator.handle_size());

  HandleLayout layout;
  EXPECT_TRUE(
      allocator.CollectAllocationHandleLayout(allocation->ptr(), &layout));
  EXPECT_EQ(layout.size(), 1UL);
}

TEST(CUDAVirtualMemAllocatorV2, RollbackCreatedHandlesReleasesLayout) {
  CUDAVirtualMemAllocatorV2 allocator(
      phi::GPUPlace(), 2UL << 20, PoolType::kLarge);

  allocator.InitOnce();
  HandleLayout layout;
  layout.push_back(nullptr);
  allocator.RollbackCreatedHandles(layout);

  layout = allocator.CreateMappedHandleLayout(
      allocator.virtual_mem_base(), allocator.handle_size(), "test rollback");
  ASSERT_EQ(layout.size(), 1UL);
  allocator.RollbackCreatedHandles(layout);
}

TEST(CUDAVirtualMemAllocatorV2, RequireHandleLayoutRejectsUnknownAllocation) {
  CUDAVirtualMemAllocatorV2 allocator(
      phi::GPUPlace(), 2UL << 20, PoolType::kLarge);

  EXPECT_THROW(
      allocator.RequireHandleLayout(reinterpret_cast<Allocation*>(0x1234)),
      common::enforce::EnforceNotMet);
}

TEST(CUDAVirtualMemAllocatorV2, PlaceAtVARejectsUnalignedAddress) {
  CUDAVirtualMemAllocatorV2 allocator(
      phi::GPUPlace(), 2UL << 20, PoolType::kLarge);

  auto allocation_with_block =
      allocator.AppendWithBlock(allocator.handle_size());
  ASSERT_NE(allocation_with_block.allocation, nullptr);
  auto ptr =
      reinterpret_cast<VMMDevicePtr>(allocation_with_block.allocation->ptr());

  EXPECT_THROW(allocator.PlaceAtVAWithBlock(ptr + 1, allocator.handle_size()),
               common::enforce::EnforceNotMet);
}

TEST(CUDAVirtualMemAllocatorV2, PlaceAtVARejectsOutOfRangeAddress) {
  CUDAVirtualMemAllocatorV2 allocator(
      phi::GPUPlace(), 2UL << 20, PoolType::kLarge);

  allocator.InitOnce();
  const VMMDevicePtr base = allocator.virtual_mem_base();
  const size_t handle_size = allocator.handle_size();

  EXPECT_THROW(allocator.PlaceAtVAWithBlock(base - handle_size, handle_size),
               common::enforce::EnforceNotMet);
  EXPECT_THROW(allocator.PlaceAtVAWithBlock(base + allocator.virtual_mem_size(),
                                            handle_size),
               common::enforce::EnforceNotMet);
  EXPECT_THROW(
      allocator.PlaceAtVAWithBlock(
          base + allocator.virtual_mem_size() - handle_size, handle_size * 2),
      common::enforce::EnforceNotMet);
}

TEST(CUDAVirtualMemAllocatorV2, ClearRemapDestinationOwnershipWrapper) {
  ScopedVLogLevel vlog_guard(6);
  CUDAVirtualMemAllocatorV2 allocator(
      phi::GPUPlace(), 2UL << 20, PoolType::kLarge);

  allocator.InitOnce();
  const VMMDevicePtr base = allocator.virtual_mem_base();
  const size_t handle_size = allocator.handle_size();

  EXPECT_FALSE(allocator.ClearRemapDestinationOwnership(base - handle_size,
                                                        handle_size));
  EXPECT_EQ(allocator.ClearRemapDestinationOwnershipInRange(base - handle_size,
                                                            handle_size),
            0UL);

  auto layout =
      allocator.CreateMappedHandleLayout(base, handle_size * 2, "test clear");
  ASSERT_EQ(layout.size(), 2UL);
  allocator.MarkRemapDestinationLayoutMapped(layout);
  std::vector<std::pair<VMMDevicePtr, size_t>> ranges = {
      {base, handle_size * 2}};
  auto sources = allocator.CollectRemapSourcePages(ranges, handle_size * 2);
  ASSERT_EQ(sources.size(), 2UL);
  EXPECT_EQ(sources[0].remap_source_state,
            VMMBackingMap::RemapSourceState::kRemapDestinationOwned);
  EXPECT_EQ(sources[1].remap_source_state,
            VMMBackingMap::RemapSourceState::kRemapDestinationOwned);

  EXPECT_TRUE(allocator.ClearRemapDestinationOwnership(base, handle_size));
  sources = allocator.CollectRemapSourcePages(ranges, handle_size * 2);
  ASSERT_EQ(sources.size(), 2UL);
  EXPECT_EQ(sources[0].remap_source_state,
            VMMBackingMap::RemapSourceState::kReady);
  EXPECT_EQ(sources[1].remap_source_state,
            VMMBackingMap::RemapSourceState::kRemapDestinationOwned);

  EXPECT_EQ(allocator.ClearRemapDestinationOwnershipInRange(base + handle_size,
                                                            handle_size),
            handle_size);
  sources = allocator.CollectRemapSourcePages(ranges, handle_size * 2);
  ASSERT_EQ(sources.size(), 2UL);
  EXPECT_EQ(sources[0].remap_source_state,
            VMMBackingMap::RemapSourceState::kReady);
  EXPECT_EQ(sources[1].remap_source_state,
            VMMBackingMap::RemapSourceState::kReady);

  allocator.RollbackCreatedHandles(layout);
}

TEST(CUDAVirtualMemAllocatorV2, RollbackMappedHandleRangeMarksUnmapped) {
  ScopedVLogLevel vlog_guard(6);
  CUDAVirtualMemAllocatorV2 allocator(
      phi::GPUPlace(), 2UL << 20, PoolType::kLarge);

  allocator.InitOnce();
  const VMMDevicePtr base = allocator.virtual_mem_base();
  const size_t handle_size = allocator.handle_size();
  auto layout = allocator.CreateMappedHandleLayout(
      base, handle_size, "test rollback mapped range");
  ASSERT_EQ(layout.size(), 1UL);
  allocator.MarkLayoutMapped(layout);
  EXPECT_TRUE(allocator.IsRangeReleasable(base, handle_size));

  allocator.RollbackMappedHandleRange(base, 1);
  EXPECT_TRUE(allocator.IsRangeUnmapped(base, handle_size));
  allocator.RollbackMappedHandleRange(base, 1);
  allocator.RollbackCreatedHandles(layout);
}

TEST(CUDAVirtualMemAllocatorV2, MoveBackingPageAndAccessHelpers) {
  ScopedVLogLevel vlog_guard(6);
  CUDAVirtualMemAllocatorV2 allocator(
      phi::GPUPlace(), 2UL << 20, PoolType::kLarge);

  auto allocation_with_block =
      allocator.AppendWithBlock(allocator.handle_size());
  ASSERT_NE(allocation_with_block.allocation, nullptr);
  const auto source_va =
      reinterpret_cast<VMMDevicePtr>(allocation_with_block.allocation->ptr());
  const auto target_va = source_va + allocator.handle_size();

  std::vector<std::pair<VMMDevicePtr, size_t>> source_range = {
      {source_va, allocator.handle_size()}};
  std::vector<std::pair<VMMDevicePtr, size_t>> target_range = {
      {target_va, allocator.handle_size()}};
  auto source_pages =
      allocator.CollectMappedPages(source_range, allocator.handle_size());
  auto target_pages =
      allocator.CollectUnmappedPages(target_range, allocator.handle_size());
  ASSERT_EQ(source_pages.size(), 1UL);
  ASSERT_EQ(target_pages.size(), 1UL);

  CUDAVirtualMemAllocatorV2::MoveBackingPageStats stats;
  EXPECT_TRUE(
      allocator.MoveBackingPage(source_pages[0], target_pages[0], &stats));
  EXPECT_TRUE(allocator.IsRangeUnmapped(source_va, allocator.handle_size()));
  EXPECT_TRUE(allocator.IsRangeReleasable(target_va, allocator.handle_size()));
  EXPECT_TRUE(allocator.SetAccessForMappedRange(
      target_va, allocator.handle_size(), &stats));
  EXPECT_TRUE(allocator.SetAccessForMappedRange(target_va, 0, &stats));
  EXPECT_FALSE(allocator.SetAccessForMappedRange(target_va + 1,
                                                 allocator.handle_size()));
  EXPECT_TRUE(allocator.UnmapMappedRangeForRemap(target_va, 0, &stats));
  EXPECT_FALSE(allocator.UnmapMappedRangeForRemap(target_va + 1, 1, &stats));
  EXPECT_EQ(allocator.CollectUnmappedPages(target_range, 0).size(), 0UL);

  auto moved_target_pages =
      allocator.CollectMappedPages(target_range, allocator.handle_size());
  auto source_holes =
      allocator.CollectUnmappedPages(source_range, allocator.handle_size());
  ASSERT_EQ(moved_target_pages.size(), 1UL);
  ASSERT_EQ(source_holes.size(), 1UL);
  EXPECT_TRUE(allocator.MoveBackingPage(
      moved_target_pages[0], source_holes[0], &stats));
  EXPECT_TRUE(allocator.IsRangeReleasable(source_va, allocator.handle_size()));
}

TEST(CUDAVirtualMemAllocatorV2, RemapDriverFailureRecovery) {
  ScopedVLogLevel vlog_guard(6);
  CUDAVirtualMemAllocatorV2 allocator(
      phi::GPUPlace(), 2UL << 20, PoolType::kLarge);

  auto source_allocation = allocator.AppendWithBlock(allocator.handle_size());
  ASSERT_NE(source_allocation.allocation, nullptr);
  const VMMDevicePtr source_va =
      reinterpret_cast<VMMDevicePtr>(source_allocation.allocation->ptr());
  const VMMDevicePtr target_va = source_va + allocator.handle_size();
  const std::vector<std::pair<VMMDevicePtr, size_t>> source_range = {
      {source_va, allocator.handle_size()}};
  const std::vector<std::pair<VMMDevicePtr, size_t>> target_range = {
      {target_va, allocator.handle_size()}};

  // cuMemSetAccess on an unmapped reserved VA must fail both the whole-range
  // attempt and the chunked fallback without mutating allocator metadata.
  EXPECT_FALSE(allocator.SetAccessForMappedRange(
      target_va, allocator.handle_size(), nullptr));

  auto source_pages =
      allocator.CollectMappedPages(source_range, allocator.handle_size());
  auto target_pages =
      allocator.CollectUnmappedPages(target_range, allocator.handle_size());
  ASSERT_EQ(source_pages.size(), 1UL);
  ASSERT_EQ(target_pages.size(), 1UL);

  // Keep the target mapped in the driver but absent from BackingMap. The move
  // must restore its source after the target cuMemMap fails.
  auto occupied_target = allocator.CreateMappedHandleLayout(
      target_va, allocator.handle_size(), "occupy remap target");
  ASSERT_EQ(occupied_target.size(), 1UL);
  EXPECT_FALSE(
      allocator.MoveBackingPage(source_pages[0], target_pages[0], nullptr));
  EXPECT_TRUE(allocator.IsRangeReleasable(source_va, allocator.handle_size()));
  allocator.RollbackCreatedHandles(occupied_target);
}

TEST(CUDAVirtualMemAllocatorV2, RemapTargetAccessFailureRollsBackMappings) {
  ScopedVLogLevel vlog_guard(6);
  CUDAVirtualMemAllocatorV2 allocator(
      phi::GPUPlace(), 2UL << 20, PoolType::kLarge);

  auto source_allocation = allocator.AppendWithBlock(allocator.handle_size());
  ASSERT_NE(source_allocation.allocation, nullptr);
  const VMMDevicePtr source_va =
      reinterpret_cast<VMMDevicePtr>(source_allocation.allocation->ptr());
  const VMMDevicePtr target_va = source_va + allocator.handle_size();
  const std::vector<std::pair<VMMDevicePtr, size_t>> source_range = {
      {source_va, allocator.handle_size()}};
  const std::vector<std::pair<VMMDevicePtr, size_t>> target_range = {
      {target_va, allocator.handle_size()}};
  auto source_pages =
      allocator.CollectMappedPages(source_range, allocator.handle_size());
  auto target_pages =
      allocator.CollectUnmappedPages(target_range, allocator.handle_size());
  ASSERT_EQ(source_pages.size(), 1UL);
  ASSERT_EQ(target_pages.size(), 1UL);
  ASSERT_NE(source_pages[0].meta, nullptr);

  auto access_desc = allocator.access_desc_;
  allocator.access_desc_.clear();
  CUDAVirtualMemAllocatorV2::MoveBackingPageStats stats;
  EXPECT_FALSE(
      allocator.MoveBackingPage(source_pages[0], target_pages[0], &stats));
  EXPECT_TRUE(allocator.IsRangeUnmapped(source_va, allocator.handle_size()));
  EXPECT_TRUE(allocator.IsRangeUnmapped(target_va, allocator.handle_size()));

  // Restore the source mapping so normal allocation cleanup remains valid.
  allocator.access_desc_ = std::move(access_desc);
  ASSERT_EQ(
      phi::dynload::cuMemMap(
          source_va, allocator.handle_size(), 0, source_pages[0].handle, 0),
      CUDA_SUCCESS);
  ASSERT_TRUE(allocator.SetAccessForMappedRange(
      source_va, allocator.handle_size(), nullptr));
  allocator.backing_map_.MarkMapped(
      source_va, source_pages[0].meta, allocator.handle_size());
}

TEST(CUDAVirtualMemAllocatorV2, RemapHelperFallbacks) {
  ScopedVLogLevel vlog_guard(6);
  CUDAVirtualMemAllocatorV2 allocator(
      phi::GPUPlace(), 2UL << 20, PoolType::kLarge);

  allocator.InitOnce();
  const VMMDevicePtr base = allocator.virtual_mem_base();
  const size_t handle_size = allocator.handle_size();
  const std::vector<std::pair<VMMDevicePtr, size_t>> unmapped_range = {
      {base, handle_size}};
  EXPECT_EQ(allocator.CollectUnmappedPages(unmapped_range, 0).size(), 1UL);

  VMMBackingMap::MappedPage source;
  source.va = base;
  VMMBackingMap::UnmappedPage target;
  target.va = base + handle_size;
  EXPECT_FALSE(
      allocator.MoveBackingPageForRemap(source, target, nullptr, nullptr));
  EXPECT_EQ(allocator.RestoreRemapSourceMapping(
                VMMAllocHandle{}, nullptr, handle_size),
            CUDAVirtualMemAllocatorV2::RestoreRemapSourceResult::kSkipped);

  auto unowned_meta = std::make_shared<VMMHandleMeta>(
      VMMHandleMeta{base, handle_size, static_cast<VMMAllocHandle>(0x1), 0});
  EXPECT_EQ(allocator.RestoreRemapSourceMapping(
                VMMAllocHandle{}, unowned_meta, handle_size),
            CUDAVirtualMemAllocatorV2::RestoreRemapSourceResult::kSkipped);

  allocator.DestroyStagedDestinationAllocation(nullptr);
  auto invalid_meta = std::make_shared<VMMHandleMeta>(VMMHandleMeta{
      base - handle_size, handle_size, static_cast<VMMAllocHandle>(0x2), 0});
  auto* staged = allocator.CreateTrackedAllocation(
      base - handle_size, handle_size, HandleLayout{invalid_meta});
  ASSERT_NE(staged, nullptr);
  allocator.DestroyStagedDestinationAllocation(staged);
}

TEST(CUDAVirtualMemAllocatorV2, StagedDestinationCleansUpOnMetadataFailure) {
  ScopedVLogLevel vlog_guard(6);
  CUDAVirtualMemAllocatorV2 allocator(
      phi::GPUPlace(), 2UL << 20, PoolType::kLarge);

  allocator.InitOnce();
  const VMMDevicePtr base = allocator.virtual_mem_base();
  auto mapped_layout = allocator.CreateMappedHandleLayout(
      base, allocator.handle_size(), "staged destination conflict");
  ASSERT_EQ(mapped_layout.size(), 1UL);
  allocator.MarkLayoutMapped(mapped_layout);

  const VMMAllocHandle conflicting_handle = static_cast<VMMAllocHandle>(0x1234);
  std::vector<VMMBackingMap::MappedPage> conflicting_pages = {
      {base, conflicting_handle, nullptr, 0}};
  EXPECT_THROW(allocator.CreateStagedRemapDestination(
                   base, conflicting_pages, 0, 1, PoolType::kLarge),
               common::enforce::EnforceNotMet);
  allocator.RollbackCreatedHandles(mapped_layout);
}

TEST(CUDAVirtualMemAllocatorV2, RestoreRemapSourceMappingRestoresOwnedSource) {
  CUDAVirtualMemAllocatorV2 allocator(
      phi::GPUPlace(), 2UL << 20, PoolType::kLarge);

  auto allocation_with_block =
      allocator.AppendWithBlock(allocator.handle_size());
  ASSERT_NE(allocation_with_block.allocation, nullptr);
  const VMMDevicePtr source_va =
      reinterpret_cast<VMMDevicePtr>(allocation_with_block.allocation->ptr());
  const VMMDevicePtr target_va = source_va + allocator.handle_size();
  std::vector<std::pair<VMMDevicePtr, size_t>> source_range = {
      {source_va, allocator.handle_size()}};
  std::vector<std::pair<VMMDevicePtr, size_t>> target_range = {
      {target_va, allocator.handle_size()}};
  auto source_pages =
      allocator.CollectMappedPages(source_range, allocator.handle_size());
  auto target_pages =
      allocator.CollectUnmappedPages(target_range, allocator.handle_size());
  ASSERT_EQ(source_pages.size(), 1UL);
  ASSERT_EQ(target_pages.size(), 1UL);
  auto meta = source_pages[0].meta;
  ASSERT_NE(meta, nullptr);

  ASSERT_TRUE(allocator.MoveBackingPageForRemap(
      source_pages[0], target_pages[0], meta));
  ASSERT_TRUE(meta->IsOwnedByRemapDestination());
  EXPECT_EQ(allocator.RestoreRemapSourceMapping(
                source_pages[0].handle, meta, allocator.handle_size()),
            CUDAVirtualMemAllocatorV2::RestoreRemapSourceResult::kRestored);
  meta->RestoreOriginalOwnership();
  allocator.RollbackMappedHandleRange(target_va, 1);
}

TEST(CUDAVirtualMemAllocatorV2, ForceReleaseOwnedRemapSource) {
  CUDAVirtualMemAllocatorV2 allocator(
      phi::GPUPlace(), 2UL << 20, PoolType::kLarge);
  auto allocation_with_block =
      allocator.AppendWithBlock(allocator.handle_size());
  ASSERT_NE(allocation_with_block.allocation, nullptr);
  const VMMDevicePtr source_va =
      reinterpret_cast<VMMDevicePtr>(allocation_with_block.allocation->ptr());
  const std::vector<std::pair<VMMDevicePtr, size_t>> source_range = {
      {source_va, allocator.handle_size()}};
  auto source_pages =
      allocator.CollectMappedPages(source_range, allocator.handle_size());
  ASSERT_EQ(source_pages.size(), 1UL);
  ASSERT_NE(source_pages[0].meta, nullptr);
  source_pages[0].meta->MarkOwnedByRemapDestination();

  EXPECT_EQ(
      allocator.ForceReleaseRemapSource(source_pages[0].handle,
                                        source_pages[0].meta,
                                        allocator.handle_size(),
                                        "unit test",
                                        true),
      CUDAVirtualMemAllocatorV2::RestoreRemapSourceResult::kForceReleased);
  EXPECT_TRUE(allocator.IsRangeUnmapped(source_va, allocator.handle_size()));
}

TEST(CUDAVirtualMemAllocatorV2, RestoreRemapSourceFailureRecovery) {
  ScopedVLogLevel vlog_guard(6);

  {
    CUDAVirtualMemAllocatorV2 allocator(
        phi::GPUPlace(), 2UL << 20, PoolType::kLarge);
    allocator.InitOnce();
    const VMMDevicePtr base = allocator.virtual_mem_base();
    auto layout = allocator.CreateMappedHandleLayout(
        base, allocator.handle_size(), "occupied restore source");
    ASSERT_EQ(layout.size(), 1UL);
    allocator.MarkLayoutMapped(layout);
    layout[0]->MarkOwnedByRemapDestination();

    EXPECT_EQ(
        allocator.RestoreRemapSourceMapping(
            layout[0]->handle(), layout[0], allocator.handle_size()),
        CUDAVirtualMemAllocatorV2::RestoreRemapSourceResult::kForceReleased);
    // Releasing a mapped handle keeps its mapping alive until it is unmapped.
    EXPECT_EQ(phi::dynload::cuMemUnmap(base, allocator.handle_size()),
              CUDA_SUCCESS);
  }

  {
    CUDAVirtualMemAllocatorV2 allocator(
        phi::GPUPlace(), 2UL << 20, PoolType::kLarge);
    allocator.InitOnce();
    const VMMDevicePtr base = allocator.virtual_mem_base();
    auto layout = allocator.CreateMappedHandleLayout(
        base, allocator.handle_size(), "restore source access failure");
    ASSERT_EQ(layout.size(), 1UL);
    allocator.MarkLayoutMapped(layout);
    ASSERT_TRUE(allocator.UnmapMappedRangeForRemap(base, 1));
    layout[0]->MarkOwnedByRemapDestination();
    allocator.access_desc_.clear();

    EXPECT_EQ(
        allocator.RestoreRemapSourceMapping(
            layout[0]->handle(), layout[0], allocator.handle_size()),
        CUDAVirtualMemAllocatorV2::RestoreRemapSourceResult::kForceReleased);
    EXPECT_TRUE(allocator.IsRangeUnmapped(base, allocator.handle_size()));
  }

  {
    CUDAVirtualMemAllocatorV2 allocator(
        phi::GPUPlace(), 2UL << 20, PoolType::kLarge);
    allocator.InitOnce();
    const VMMDevicePtr base = allocator.virtual_mem_base();
    auto layout = allocator.CreateMappedHandleLayout(
        base, allocator.handle_size(), "force release unmapped source");
    ASSERT_EQ(layout.size(), 1UL);
    allocator.MarkLayoutMapped(layout);
    ASSERT_EQ(phi::dynload::cuMemUnmap(base, allocator.handle_size()),
              CUDA_SUCCESS);
    layout[0]->MarkOwnedByRemapDestination();

    EXPECT_EQ(
        allocator.ForceReleaseRemapSource(layout[0]->handle(),
                                          layout[0],
                                          allocator.handle_size(),
                                          nullptr,
                                          true),
        CUDAVirtualMemAllocatorV2::RestoreRemapSourceResult::kForceReleased);
  }
}

TEST(CUDAVirtualMemAllocatorV2, FreeRemovesHandleRegistration) {
  CUDAVirtualMemAllocatorV2 allocator(
      phi::GPUPlace(), 2UL << 20, PoolType::kLarge);

  auto allocation_with_block =
      allocator.AppendWithBlock(allocator.handle_size());
  ASSERT_NE(allocation_with_block.allocation, nullptr);
  void* ptr = allocation_with_block.allocation->ptr();

  allocation_with_block.allocation.reset();

  auto reused = allocator.PlaceAtVAWithBlock(
      reinterpret_cast<VMMDevicePtr>(ptr), allocator.handle_size());
  ASSERT_NE(reused.allocation, nullptr);
  EXPECT_EQ(reused.allocation->ptr(), ptr);
  std::vector<std::pair<VMMDevicePtr, size_t>> ranges = {
      {reinterpret_cast<VMMDevicePtr>(ptr), allocator.handle_size()}};
  EXPECT_EQ(
      allocator.CollectMappedPages(ranges, allocator.handle_size()).size(),
      1UL);
  EXPECT_TRUE(allocator.IsRangeReleasable(reinterpret_cast<VMMDevicePtr>(ptr),
                                          allocator.handle_size()));
}

TEST(CUDAVirtualMemAllocatorV2, StagedRemapDestinationBlocksSource) {
  CUDAVirtualMemAllocatorV2 allocator(
      phi::GPUPlace(), 2UL << 20, PoolType::kLarge);

  auto allocation_with_block =
      allocator.AppendWithBlock(allocator.handle_size());
  ASSERT_NE(allocation_with_block.allocation, nullptr);

  const VMMDevicePtr source_va =
      reinterpret_cast<VMMDevicePtr>(allocation_with_block.allocation->ptr());
  const VMMDevicePtr target_va = source_va + allocator.handle_size();
  std::vector<std::pair<VMMDevicePtr, size_t>> source_ranges = {
      {source_va, allocator.handle_size()}};
  std::vector<std::pair<VMMDevicePtr, size_t>> target_ranges = {
      {target_va, allocator.handle_size()}};
  auto source_pages =
      allocator.CollectMappedPages(source_ranges, allocator.handle_size());
  auto target_pages =
      allocator.CollectUnmappedPages(target_ranges, allocator.handle_size());
  ASSERT_EQ(source_pages.size(), 1UL);
  ASSERT_EQ(target_pages.size(), 1UL);

  auto meta = source_pages[0].meta;
  ASSERT_NE(meta, nullptr);
  ASSERT_TRUE(allocator.MoveBackingPageForRemap(
      source_pages[0], target_pages[0], meta));
  EXPECT_TRUE(meta->IsOwnedByRemapDestination());

  auto staged = allocator.CreateStagedRemapDestination(
      target_va, source_pages, 0, 1, PoolType::kLarge);
  ASSERT_NE(staged.allocation, nullptr);
  EXPECT_TRUE(allocator.IsRemapDestinationAllocation(staged.allocation->ptr()));

  auto remap_sources =
      allocator.CollectRemapSourcePages(target_ranges, allocator.handle_size());
  ASSERT_EQ(remap_sources.size(), 1UL);
  EXPECT_EQ(remap_sources[0].remap_source_state,
            VMMBackingMap::RemapSourceState::kRemapDestinationOwned);

  std::vector<BlockPart> ipc_parts;
  EXPECT_FALSE(allocator.CollectIPCParts(
      staged.block.begin_va(), staged.block.size(), &ipc_parts));
  EXPECT_TRUE(ipc_parts.empty());

  auto destination_pages =
      allocator.CollectMappedPages(target_ranges, allocator.handle_size());
  ASSERT_EQ(destination_pages.size(), 1UL);
  ASSERT_NE(destination_pages[0].meta, nullptr);
  destination_pages[0].meta->RestoreOriginalOwnership();

  // Finalizing the destination meta makes IPC exportable, while the page-level
  // destination marker still prevents recursive remap before stale cleanup.
  remap_sources =
      allocator.CollectRemapSourcePages(target_ranges, allocator.handle_size());
  ASSERT_EQ(remap_sources.size(), 1UL);
  EXPECT_EQ(remap_sources[0].remap_source_state,
            VMMBackingMap::RemapSourceState::kRemapDestinationOwned);
  EXPECT_TRUE(allocator.CollectIPCParts(
      staged.block.begin_va(), staged.block.size(), &ipc_parts));
  ASSERT_EQ(ipc_parts.size(), 1UL);
  ASSERT_NE(ipc_parts[0].chunk, nullptr);
  EXPECT_EQ(ipc_parts[0].chunk->base, target_va);
  EXPECT_EQ(ipc_parts[0].chunk_rel_off, 0UL);
  EXPECT_EQ(ipc_parts[0].len, allocator.handle_size());

  auto committed = allocator.AdoptRemapDestinationAllocation(staged.allocation);
  staged.allocation = nullptr;
}

TEST(CUDAVirtualMemAllocatorV2,
     RemapTransactionCommitExceptionDoesNotDoubleDestroyStagedDestination) {
  CUDAVirtualMemAllocatorV2 allocator(
      phi::GPUPlace(), 2UL << 20, PoolType::kLarge);

  auto allocation_with_block =
      allocator.AppendWithBlock(allocator.handle_size());
  ASSERT_NE(allocation_with_block.allocation, nullptr);

  const VMMDevicePtr source_va =
      reinterpret_cast<VMMDevicePtr>(allocation_with_block.allocation->ptr());
  const VMMDevicePtr target_va = source_va + allocator.handle_size();
  std::vector<std::pair<VMMDevicePtr, size_t>> source_ranges = {
      {source_va, allocator.handle_size()}};
  std::vector<std::pair<VMMDevicePtr, size_t>> target_ranges = {
      {target_va, allocator.handle_size()}};
  auto source_pages =
      allocator.CollectMappedPages(source_ranges, allocator.handle_size());
  auto target_pages =
      allocator.CollectUnmappedPages(target_ranges, allocator.handle_size());
  ASSERT_EQ(source_pages.size(), 1UL);
  ASSERT_EQ(target_pages.size(), 1UL);

  bool commit_called = false;
  RemapTransaction transaction(
      &allocator,
      allocator.handle_size(),
      [&](std::vector<DecoratedAllocationPtr>* allocations) {
        commit_called = true;
        ASSERT_NE(allocations, nullptr);
        ASSERT_EQ(allocations->size(), 1UL);
        ASSERT_NE(allocations->front(), nullptr);
        throw std::runtime_error("injected commit failure");
      });
  RemapTransaction::BlockList blocks;
  blocks.push_back(BlockV2::MakeMappedBlock(BlockType::kFree,
                                            reinterpret_cast<void*>(source_va),
                                            allocator.handle_size(),
                                            PoolType::kLarge));
  RemapTransaction::SourceMovePlan move_plan;
  move_plan.source_pages.push_back(source_pages[0]);
  ASSERT_TRUE(transaction.MovePlannedPagesToTargets(
      &blocks, &move_plan, target_pages, nullptr));

  auto materialized = transaction.MaterializeDestinationRange(
      target_va, source_pages, 0, 1, PoolType::kLarge);
  EXPECT_EQ(materialized.size(), allocator.handle_size());

  EXPECT_THROW(transaction.Commit(), std::runtime_error);
  EXPECT_TRUE(commit_called);
  EXPECT_NO_THROW(transaction.Rollback());
  auto restored_sources =
      allocator.CollectMappedPages(source_ranges, allocator.handle_size());
  auto restored_targets =
      allocator.CollectUnmappedPages(target_ranges, allocator.handle_size());
  ASSERT_EQ(restored_sources.size(), 1UL);
  EXPECT_EQ(restored_sources[0].handle, source_pages[0].handle);
  EXPECT_EQ(restored_targets.size(), 1UL);
}

TEST(CUDAVirtualMemAllocatorV2, RemapTransactionStateGuards) {
  ScopedVLogLevel vlog_guard(6);
  CUDAVirtualMemAllocatorV2 allocator(
      phi::GPUPlace(), 2UL << 20, PoolType::kLarge);
  allocator.InitOnce();
  const VMMDevicePtr base = allocator.virtual_mem_base();
  const size_t handle_size = allocator.handle_size();
  RemapTransaction::BlockList blocks;

  RemapTransaction transaction(&allocator, handle_size);
  std::vector<VMMBackingMap::UnmappedPage> target_pages;
  EXPECT_FALSE(transaction.CollectTargetPagesForRange(
      base - handle_size, 1, "invalid target", &target_pages));

  RemapTransaction rejected_destination(
      &allocator, handle_size, {}, {}, [](void*, size_t) { return false; });
  EXPECT_FALSE(rejected_destination.PrepareDestinationRange(
      base, handle_size, "rejected destination"));

  RemapTransaction::SourceMovePlan move_plan;
  move_plan.source_pages.push_back(
      {base, static_cast<VMMAllocHandle>(0x1), nullptr, 0});
  EXPECT_FALSE(
      transaction.MovePlannedPagesToTargets(&blocks, &move_plan, {}, nullptr));

  {
    RemapTransaction unfinished(&allocator, handle_size);
    auto* staged =
        allocator.CreateTrackedAllocation(base, handle_size, HandleLayout{});
    unfinished.StageDestinationAllocation(staged);
  }

  auto mapped_free = [&](VMMDevicePtr va, size_t size) {
    return BlockV2::MakeMappedBlock(
        BlockType::kFree, reinterpret_cast<void*>(va), size, PoolType::kLarge);
  };
  auto unmapped_free = [&](VMMDevicePtr va, size_t size) {
    return BlockV2::MakeUnmappedFreeBlock(
        reinterpret_cast<void*>(va), size, PoolType::kLarge);
  };

  RemapTransaction block_editor(&allocator, handle_size);
  RemapTransaction::BlockList tail_blocks;
  tail_blocks.push_back(mapped_free(base, handle_size));
  block_editor.InstallTailFreeBlock(
      &tail_blocks, mapped_free(base + handle_size, handle_size));
  ASSERT_EQ(tail_blocks.size(), 1UL);
  EXPECT_EQ(tail_blocks.front().size(), 2UL * handle_size);

  RemapTransaction::BlockList split_blocks;
  split_blocks.push_back(mapped_free(base, handle_size));
  split_blocks.push_back(unmapped_free(base + handle_size, 2UL * handle_size));
  auto hole = std::prev(split_blocks.end());
  split_blocks.push_back(mapped_free(base + 3UL * handle_size, handle_size));
  auto installed = block_editor.ReplaceUnmappedRangeWithMappedFree(
      &split_blocks,
      hole,
      mapped_free(base + handle_size, handle_size),
      PoolType::kLarge);
  EXPECT_EQ(installed->begin_va(), base);
  EXPECT_EQ(installed->size(), 2UL * handle_size);
  ASSERT_EQ(split_blocks.size(), 3UL);
  EXPECT_TRUE(std::next(installed)->IsUnmappedFree());

  RemapTransaction::BlockList merge_next_blocks;
  merge_next_blocks.push_back(unmapped_free(base + handle_size, handle_size));
  auto exact_hole = merge_next_blocks.begin();
  merge_next_blocks.push_back(
      mapped_free(base + 2UL * handle_size, handle_size));
  auto merged = block_editor.ReplaceUnmappedRangeWithMappedFree(
      &merge_next_blocks,
      exact_hole,
      mapped_free(base + handle_size, handle_size),
      PoolType::kLarge);
  ASSERT_EQ(merge_next_blocks.size(), 1UL);
  EXPECT_EQ(merged->size(), 2UL * handle_size);
}

TEST(CUDAVirtualMemAllocatorV2, CollectsMetadataAndExportsIPCBlockBacking) {
  CUDAVirtualMemAllocatorV2 allocator(
      phi::GPUPlace(), 2UL << 20, PoolType::kLarge);

  auto allocation_with_block =
      allocator.AppendWithBlock(allocator.handle_size() * 2);
  ASSERT_NE(allocation_with_block.allocation, nullptr);

  BlockV2 block = allocation_with_block.block.MakeMappedActiveSubBlock(
      128, allocator.handle_size() - 128 + 2048);
  const auto block_base =
      reinterpret_cast<VMMDevicePtr>(allocation_with_block.allocation->ptr());
  std::vector<std::pair<VMMDevicePtr, size_t>> ranges = {
      {block_base, allocation_with_block.allocation->size()}};
  auto pages = allocator.CollectMappedPages(
      ranges, allocation_with_block.allocation->size());
  ASSERT_EQ(pages.size(), 2UL);
  std::vector<BlockPart> ipc_parts;
  ASSERT_TRUE(
      allocator.CollectIPCParts(block.begin_va(), block.size(), &ipc_parts));
  ASSERT_EQ(ipc_parts.size(), 2UL);
  EXPECT_EQ(ipc_parts[0].chunk->base, pages[0].va);
  EXPECT_EQ(ipc_parts[0].chunk->size, allocator.handle_size());
  EXPECT_EQ(ipc_parts[0].chunk->handle, pages[0].handle);
  EXPECT_EQ(ipc_parts[0].chunk->shared_fd, -1);
  EXPECT_EQ(ipc_parts[0].chunk_rel_off, 128UL);
  EXPECT_EQ(ipc_parts[0].len, allocator.handle_size() - 128);
  EXPECT_EQ(ipc_parts[1].chunk->base, pages[1].va);
  EXPECT_EQ(ipc_parts[1].chunk->shared_fd, -1);
  EXPECT_EQ(ipc_parts[1].chunk_rel_off, 0UL);
  EXPECT_EQ(ipc_parts[1].len, 2048UL);
  EXPECT_TRUE(allocator.ipc_export_fds_.empty());

  EXPECT_TRUE(allocator.IsRangeReleasable(block.begin_va(), block.size()));
#if defined(__linux__)
  const size_t second_page_index =
      (pages[1].va - allocator.backing_map_.base_) /
      allocator.backing_map_.page_size_;
  auto second_page_meta = allocator.backing_map_.pages_[second_page_index].meta;
  allocator.backing_map_.pages_[second_page_index].meta =
      std::make_shared<VMMHandleMeta>(second_page_meta->base(),
                                      second_page_meta->size(),
                                      0,
                                      second_page_meta->device());
  EXPECT_ANY_THROW(
      allocator.ExportIPCParts(block.begin_va(), block.size(), &ipc_parts));
  EXPECT_FALSE(allocator.HasIPCExportedRange(block.begin_va(), block.size()));
  EXPECT_TRUE(allocator.IsRangeReleasable(block.begin_va(), block.size()));
  allocator.backing_map_.pages_[second_page_index].meta = second_page_meta;
#endif

  ASSERT_TRUE(
      allocator.ExportIPCParts(block.begin_va(), block.size(), &ipc_parts));
  EXPECT_TRUE(allocator.HasIPCExportedRange(block.begin_va(), block.size()));
  EXPECT_FALSE(allocator.IsRangeReleasable(block.begin_va(), block.size()));
  EXPECT_EQ(allocator.CountIPCExportedBytes({{block.begin_va(), block.size()}}),
            block.size());
#if defined(__linux__)
  ASSERT_EQ(allocator.ipc_export_fds_.size(), 2UL);
  ASSERT_GE(ipc_parts[0].chunk->shared_fd, 0);
  ASSERT_GE(ipc_parts[1].chunk->shared_fd, 0);
  const int first_fd = ipc_parts[0].chunk->shared_fd;
  const int second_fd = ipc_parts[1].chunk->shared_fd;
  ASSERT_TRUE(
      allocator.ExportIPCParts(block.begin_va(), block.size(), &ipc_parts));
  EXPECT_EQ(ipc_parts[0].chunk->shared_fd, first_fd);
  EXPECT_EQ(ipc_parts[1].chunk->shared_fd, second_fd);
#endif

  ASSERT_NE(pages[0].meta, nullptr);
  pages[0].meta->MarkOwnedByRemapDestination();
  EXPECT_FALSE(
      allocator.CollectIPCParts(block.begin_va(), block.size(), &ipc_parts));
  pages[0].meta->RestoreOriginalOwnership();

  BlockV2 invalid_block = BlockV2::MakeMappedBlock(
      BlockType::kActive,
      reinterpret_cast<void*>(allocator.virtual_mem_base() -
                              allocator.handle_size()),
      allocator.handle_size(),
      PoolType::kLarge);
  EXPECT_FALSE(allocator.CollectIPCParts(
      invalid_block.begin_va(), invalid_block.size(), &ipc_parts));
  EXPECT_FALSE(allocator.ExportIPCParts(
      invalid_block.begin_va(), invalid_block.size(), &ipc_parts));
  EXPECT_FALSE(allocator.HasIPCExportedRange(invalid_block.begin_va(),
                                             invalid_block.size()));

  allocation_with_block.allocation.reset();
  EXPECT_TRUE(allocator.ipc_export_fds_.empty());
}

TEST(CUDAVirtualMemAllocatorV2, LazyPendingStreamBlocksRemapAndRelease) {
  CUDAVirtualMemAllocatorV2 allocator(
      phi::GPUPlace(), 2UL << 20, PoolType::kLarge);

  auto allocation_with_block =
      allocator.AppendWithBlock(allocator.handle_size());
  ASSERT_NE(allocation_with_block.allocation, nullptr);

  BlockV2 block = allocation_with_block.block;
  const auto block_base =
      reinterpret_cast<VMMDevicePtr>(allocation_with_block.allocation->ptr());

  cudaStream_t stream = nullptr;
  ASSERT_EQ(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking),
            cudaSuccess);
  VMMBackingMapBusyWaitKernel<<<1, 1, 0, stream>>>(500000000ULL);
  ASSERT_EQ(cudaGetLastError(), cudaSuccess);

  ASSERT_TRUE(allocator.SetBlockRemapEvent(block, stream, nullptr));
  EXPECT_FALSE(
      allocator.IsRangeReleasable(block_base, allocator.handle_size()));
  std::vector<std::pair<VMMDevicePtr, size_t>> ranges = {
      {block_base, allocator.handle_size()}};
  EXPECT_EQ(
      allocator.CollectMappedPages(ranges, allocator.handle_size()).size(),
      1UL);
  auto remap_sources =
      allocator.CollectRemapSourcePages(ranges, allocator.handle_size());
  ASSERT_EQ(remap_sources.size(), 1UL);
  EXPECT_EQ(remap_sources[0].va, block_base);
  EXPECT_EQ(remap_sources[0].remap_source_state,
            VMMBackingMap::RemapSourceState::kPendingEvent);

  ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
  EXPECT_TRUE(allocator.IsRangeReleasable(block_base, allocator.handle_size()));
  remap_sources =
      allocator.CollectRemapSourcePages(ranges, allocator.handle_size());
  ASSERT_EQ(remap_sources.size(), 1UL);
  EXPECT_EQ(remap_sources[0].remap_source_state,
            VMMBackingMap::RemapSourceState::kReady);
  ASSERT_EQ(cudaStreamDestroy(stream), cudaSuccess);
}

}  // namespace allocation
}  // namespace memory
}  // namespace paddle
