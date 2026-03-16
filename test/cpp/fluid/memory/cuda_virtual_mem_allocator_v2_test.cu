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

#include "gtest/gtest.h"

#include "paddle/phi/core/memory/allocation/cuda_virtual_mem_allocator_v2.h"

namespace paddle {
namespace memory {
namespace allocation {

TEST(CUDAVirtualMemAllocatorV2, HandleSizeAligned) {
  CUDAVirtualMemAllocatorV2 allocator(
      phi::GPUPlace(), 1, PoolType::kTransient);

  auto allocation = allocator.Allocate(1);
  ASSERT_NE(allocation, nullptr);
  ASSERT_NE(allocation->ptr(), nullptr);
  ASSERT_GT(allocator.handle_size(), 0UL);
  ASSERT_EQ(allocation->size() % allocator.handle_size(), 0UL);
}

TEST(CUDAVirtualMemAllocatorV2, CollectAllocationParts) {
  CUDAVirtualMemAllocatorV2 allocator(
      phi::GPUPlace(), 2UL << 20, PoolType::kTransient);

  auto allocation = allocator.Allocate(allocator.handle_size() * 2);
  ASSERT_NE(allocation, nullptr);

  std::vector<BlockPartV2> parts;
  ASSERT_TRUE(allocator.CollectAllocationParts(allocation->ptr(), &parts));
  ASSERT_EQ(parts.size(), 2UL);

  auto base = reinterpret_cast<VmmDevicePtr>(allocation->ptr());
  for (size_t i = 0; i < parts.size(); ++i) {
    EXPECT_EQ(parts[i].chunk_rel_off, 0UL);
    EXPECT_EQ(parts[i].len, allocator.handle_size());
    ASSERT_TRUE(parts[i].chunk);
    EXPECT_EQ(parts[i].chunk->base, base + i * allocator.handle_size());
    EXPECT_EQ(parts[i].chunk->size, allocator.handle_size());
  }
}

TEST(CUDAVirtualMemAllocatorV2, TailOffsetAdvancesWithAllocationSize) {
  CUDAVirtualMemAllocatorV2 allocator(
      phi::GPUPlace(), 2UL << 20, PoolType::kTransient);

  auto first = allocator.Allocate(1);
  ASSERT_NE(first, nullptr);
  EXPECT_EQ(allocator.tail_offset(), first->size());

  auto second = allocator.Allocate(allocator.handle_size() + 1);
  ASSERT_NE(second, nullptr);
  EXPECT_EQ(allocator.tail_offset(), first->size() + second->size());
}

TEST(CUDAVirtualMemAllocatorV2, FreeRemovesHandleRegistration) {
  CUDAVirtualMemAllocatorV2 allocator(
      phi::GPUPlace(), 2UL << 20, PoolType::kLongLived);

  auto allocation = allocator.Allocate(allocator.handle_size());
  ASSERT_NE(allocation, nullptr);
  void* base_ptr = allocation->ptr();

  std::vector<BlockPartV2> parts;
  ASSERT_TRUE(allocator.CollectAllocationParts(base_ptr, &parts));
  ASSERT_EQ(parts.size(), 1UL);

  allocation.reset();

  EXPECT_FALSE(allocator.CollectAllocationParts(base_ptr, &parts));
}

TEST(CUDAVirtualMemAllocatorV2, UnmapAndMapHandleBackToSameVA) {
  CUDAVirtualMemAllocatorV2 allocator(
      phi::GPUPlace(), 2UL << 20, PoolType::kTransient);

  auto allocation = allocator.Allocate(allocator.handle_size() * 2);
  ASSERT_NE(allocation, nullptr);

  std::vector<BlockPartV2> parts;
  ASSERT_TRUE(allocator.CollectAllocationParts(allocation->ptr(), &parts));
  ASSERT_EQ(parts.size(), 2UL);

  const auto remap_ptr = parts[0].chunk->base;
  const auto remap_handle = parts[0].chunk->handle;
  allocator.UnmapHandle(remap_ptr, allocator.handle_size());
  allocator.MapHandlesToVA(remap_ptr, {remap_handle});

  std::vector<BlockPartV2> parts_after_remap;
  EXPECT_TRUE(
      allocator.CollectAllocationParts(allocation->ptr(), &parts_after_remap));
  ASSERT_EQ(parts_after_remap.size(), parts.size());
  EXPECT_EQ(parts_after_remap[0].chunk->base, parts[0].chunk->base);
  EXPECT_EQ(parts_after_remap[0].chunk->handle, parts[0].chunk->handle);
}

}  // namespace allocation
}  // namespace memory
}  // namespace paddle
