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
  CUDAVirtualMemAllocatorV2 allocator(phi::GPUPlace(), 1, PoolType::kTransient);

  auto allocation = allocator.Allocate(1);
  ASSERT_NE(allocation, nullptr);
  ASSERT_NE(allocation->ptr(), nullptr);
  ASSERT_GT(allocator.handle_size(), 0UL);
  ASSERT_EQ(allocation->size() % allocator.handle_size(), 0UL);
}

TEST(CUDAVirtualMemAllocatorV2, CollectAllocationHandleLayout) {
  CUDAVirtualMemAllocatorV2 allocator(
      phi::GPUPlace(), 2UL << 20, PoolType::kTransient);

  auto allocation = allocator.Allocate(allocator.handle_size() * 2);
  ASSERT_NE(allocation, nullptr);

  HandleLayout layout;
  ASSERT_TRUE(
      allocator.CollectAllocationHandleLayout(allocation->ptr(), &layout));
  ASSERT_EQ(layout.size(), 2UL);

  auto base = reinterpret_cast<VmmDevicePtr>(allocation->ptr());
  for (size_t i = 0; i < layout.size(); ++i) {
    ASSERT_TRUE(layout[i]);
    EXPECT_EQ(layout[i]->base, base + i * allocator.handle_size());
    EXPECT_EQ(layout[i]->size, allocator.handle_size());
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
  void* ptr = allocation->ptr();

  HandleLayout layout;
  ASSERT_TRUE(allocator.CollectAllocationHandleLayout(ptr, &layout));
  ASSERT_EQ(layout.size(), 1UL);

  allocation.reset();

  EXPECT_FALSE(allocator.CollectAllocationHandleLayout(ptr, &layout));
}

TEST(CUDAVirtualMemAllocatorV2, UnmapAndMapHandleBackToSameVA) {
  CUDAVirtualMemAllocatorV2 allocator(
      phi::GPUPlace(), 2UL << 20, PoolType::kTransient);

  auto allocation = allocator.Allocate(allocator.handle_size() * 2);
  ASSERT_NE(allocation, nullptr);

  HandleLayout layout;
  ASSERT_TRUE(
      allocator.CollectAllocationHandleLayout(allocation->ptr(), &layout));
  ASSERT_EQ(layout.size(), 2UL);

  const auto remap_ptr = layout[0]->base;
  const auto remap_handle = layout[0]->handle;
  allocator.UnmapHandle(remap_ptr, allocator.handle_size());
  allocator.MapHandlesToVA(remap_ptr, {remap_handle});

  HandleLayout layout_after_remap;
  EXPECT_TRUE(allocator.CollectAllocationHandleLayout(allocation->ptr(),
                                                      &layout_after_remap));
  ASSERT_EQ(layout_after_remap.size(), layout.size());
  EXPECT_EQ(layout_after_remap[0]->base, layout[0]->base);
  EXPECT_EQ(layout_after_remap[0]->handle, layout[0]->handle);
}

}  // namespace allocation
}  // namespace memory
}  // namespace paddle
