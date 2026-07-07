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
#include <utility>
#include <vector>

#include "gtest/gtest.h"

#include "paddle/phi/core/enforce.h"
#define private public
#include "paddle/phi/core/memory/allocation/cuda_virtual_mem_allocator_v2.h"
#undef private
#include "paddle/phi/core/memory/allocation/vmm_backing_map.h"

namespace paddle {
namespace memory {
namespace allocation {

TEST(VMMBackingMap, TracksMappedAndUnmappedRanges) {
  VMMBackingMap map;
  const VMMDevicePtr base = 0x10000000;
  const size_t page_size = 2UL << 20;
  map.Configure(base, page_size * 4, page_size, 0);

  EXPECT_TRUE(map.IsRangeUnmapped(base, page_size * 4));
  EXPECT_FALSE(map.IsRangeMapped(base, page_size));
  EXPECT_EQ(map.total_mapped_bytes(), 0UL);

  const VMMAllocHandle first_handle = static_cast<VMMAllocHandle>(0x101);
  const VMMAllocHandle second_handle = static_cast<VMMAllocHandle>(0x102);
  auto first_meta =
      std::make_shared<VMMHandleMeta>(base, page_size, first_handle, 0);
  auto second_meta = std::make_shared<VMMHandleMeta>(
      base + page_size, page_size, second_handle, 0);
  map.MarkMapped(base, first_meta, page_size);
  map.MarkMapped(base + page_size, second_meta, page_size);
  map.MarkMapped(base, first_meta, page_size);
  EXPECT_EQ(map.total_mapped_bytes(), page_size * 2);
  EXPECT_FALSE(map.IsRangeReleasable(base, page_size * 4));
  EXPECT_FALSE(map.IsRangeReleasable(base - page_size, page_size));

  EXPECT_TRUE(map.IsRangeMapped(base, page_size * 2));
  EXPECT_FALSE(map.IsRangeMapped(base, page_size * 3));
  EXPECT_FALSE(map.IsRangeUnmapped(base, page_size));
  EXPECT_TRUE(map.IsRangeUnmapped(base + page_size * 2, page_size * 2));
  EXPECT_EQ(map.total_mapped_bytes(), page_size * 2);

  map.MarkUnmapped(base, page_size);
  map.MarkUnmapped(base, page_size);

  EXPECT_FALSE(map.IsRangeMapped(base, page_size * 2));
  EXPECT_TRUE(map.IsRangeUnmapped(base, page_size));
  EXPECT_TRUE(map.IsRangeMapped(base + page_size, page_size));
  EXPECT_EQ(map.total_mapped_bytes(), page_size);

  map.MarkReleased(base + page_size, second_handle, page_size);
  map.MarkReleased(base + page_size, second_handle, page_size);
  EXPECT_TRUE(map.IsRangeUnmapped(base, page_size * 4));
  EXPECT_EQ(map.total_mapped_bytes(), 0UL);
  const VMMAllocHandle third_handle = static_cast<VMMAllocHandle>(0x103);
  map.MarkMapped(base, third_handle, page_size);
  EXPECT_TRUE(map.IsRangeMapped(base, page_size));
}

TEST(VMMBackingMap, RejectsMappedPageHandleOverwrite) {
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
  VMMBackingMap map;
  const VMMDevicePtr base = 0x1a000000;
  const size_t page_size = 2UL << 20;

  EXPECT_THROW(map.Configure(base, page_size, 0, 0),
               common::enforce::EnforceNotMet);
  EXPECT_THROW(map.Configure(base, page_size + 1, page_size, 0),
               common::enforce::EnforceNotMet);
}

TEST(VMMBackingMap, ReconfigureKeepsOriginalLayout) {
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
  VMMBackingMap map;
  const VMMDevicePtr base = 0x1e000000;
  const size_t page_size = 2UL << 20;
  auto meta = std::make_shared<VMMHandleMeta>(
      base, page_size, static_cast<VMMAllocHandle>(0x1e1), 0);
  HandleLayout layout{meta};

  EXPECT_FALSE(map.IsRangeMapped(base, page_size));
  EXPECT_FALSE(map.IsRangeUnmapped(base, page_size));
  EXPECT_FALSE(map.IsRangeReleasable(base, page_size));
  EXPECT_FALSE(map.ValidateLayout(layout, "unconfigured"));
  map.MarkMapped(base, meta, page_size);
  map.MarkUnmapped(base, page_size);
  map.MarkReleased(base, meta->handle(), page_size);

  map.Configure(base, page_size * 2, page_size, 0);
  EXPECT_FALSE(map.IsRangeMapped(base, page_size / 2));
  EXPECT_FALSE(map.IsRangeUnmapped(base - page_size, page_size));
  EXPECT_FALSE(map.IsRangeReleasable(base + page_size * 2, page_size));
}

TEST(VMMBackingMap, ValidateLayoutDetectsMissingAndMismatchedPages) {
  VMMBackingMap map;
  const VMMDevicePtr base = 0x22000000;
  const size_t page_size = 2UL << 20;
  map.Configure(base, page_size * 2, page_size, 0);

  auto first = std::make_shared<VMMHandleMeta>(
      base, page_size, static_cast<VMMAllocHandle>(0x221), 0);
  auto second = std::make_shared<VMMHandleMeta>(
      base + page_size, page_size, static_cast<VMMAllocHandle>(0x222), 0);
  EXPECT_FALSE(map.ValidateLayout(HandleLayout{first}, "missing mapped page"));

  map.MarkMapped(base, first, page_size);
  map.MarkMapped(base + page_size, second, page_size);
  auto wrong = std::make_shared<VMMHandleMeta>(
      base + page_size, page_size, static_cast<VMMAllocHandle>(0x223), 0);
  EXPECT_FALSE(
      map.ValidateLayout(HandleLayout{first, wrong}, "handle mismatch"));
}

TEST(VMMBackingMap, MarkReleasedAllowsMismatchAndNullMetaIsNotReleasable) {
  VMMBackingMap map;
  const VMMDevicePtr base = 0x24000000;
  const size_t page_size = 2UL << 20;
  map.Configure(base, page_size * 2, page_size, 0);

  map.MarkMapped(base, static_cast<VMMAllocHandle>(0), page_size);
  EXPECT_TRUE(map.IsRangeMapped(base, page_size));
  EXPECT_FALSE(map.IsRangeReleasable(base, page_size));
  map.MarkUnmapped(base + page_size, page_size);

  auto meta = std::make_shared<VMMHandleMeta>(
      base + page_size, page_size, static_cast<VMMAllocHandle>(0x241), 0);
  map.MarkMapped(base + page_size, meta, page_size);
  map.MarkReleased(
      base + page_size, static_cast<VMMAllocHandle>(0x242), page_size);
  EXPECT_TRUE(map.IsRangeUnmapped(base + page_size, page_size));
}

TEST(VMMBackingMap, PageCanUseBackingRejectsInvalidPageStates) {
  VMMBackingMap map;
  VMMBackingMap::Page page;

  EXPECT_FALSE(map.PageCanUseBackingLocked(nullptr, "null page"));
  EXPECT_FALSE(map.PageCanUseBackingLocked(&page, "unmapped page"));
  page.mapped = true;
  EXPECT_FALSE(map.PageCanUseBackingLocked(&page, "missing meta"));
}

TEST(BlockV2, UnmappedSubBlockTrimAndMerge) {
  auto* base = reinterpret_cast<void*>(0x20000000);
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
}

TEST(CUDAVirtualMemAllocatorV2, AppendWithBlockReturnsMappedFreeBlock) {
  CUDAVirtualMemAllocatorV2 allocator(
      phi::GPUPlace(), 2UL << 20, PoolType::kLarge);

  auto allocation_with_block =
      allocator.AppendWithBlock(allocator.handle_size() * 2);
  ASSERT_NE(allocation_with_block.allocation, nullptr);

  const auto& block = allocation_with_block.block;
  ASSERT_EQ(block.size_, allocation_with_block.allocation->size());
  EXPECT_EQ(block.ptr_, allocation_with_block.allocation->ptr());
  EXPECT_TRUE(block.IsMappedFree());
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

TEST(CUDAVirtualMemAllocatorV2, RequireHandleLayoutRejectsUnknownPointer) {
  CUDAVirtualMemAllocatorV2 allocator(
      phi::GPUPlace(), 2UL << 20, PoolType::kLarge);

  EXPECT_THROW(allocator.RequireHandleLayout(reinterpret_cast<void*>(0x1234)),
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
  EXPECT_TRUE(allocator.IsRangeReleasable(reinterpret_cast<VMMDevicePtr>(ptr),
                                          allocator.handle_size()));
}

}  // namespace allocation
}  // namespace memory
}  // namespace paddle
