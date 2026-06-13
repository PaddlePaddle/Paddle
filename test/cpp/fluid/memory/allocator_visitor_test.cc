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
#include "paddle/phi/core/memory/allocation/allocator.h"
#include "paddle/phi/core/memory/allocation/auto_growth_best_fit_allocator.h"
#include "paddle/phi/core/memory/allocation/cpu_allocator.h"
#include "paddle/phi/core/memory/allocation/cuda_virtual_mem_allocator.h"
#include "paddle/phi/core/memory/allocation/retry_allocator.h"
#include "paddle/phi/core/memory/allocation/virtual_memory_auto_growth_best_fit_allocator.h"
#include "paddle/phi/core/platform/device/gpu/gpu_info.h"
#ifdef PADDLE_WITH_CUDA
#include <cuda.h>
#include <cuda_runtime.h>
#endif
#include "glog/logging.h"
#include "gtest/gtest.h"
namespace paddle {
namespace memory {
namespace allocation {

TEST(VirtualMemoryAutoGrowthBestFitAllocator, TestAllocatorVisitor) {
  FLAGS_v = 1;
  auto vmm_cuda_allocator =
      std::make_shared<CUDAVirtualMemAllocator>(phi::GPUPlace());
  auto vma_allocator =
      std::make_shared<VirtualMemoryAutoGrowthBestFitAllocator>(
          vmm_cuda_allocator, platform::GpuMinChunkSize(), phi::GPUPlace());
  memory::AllocatorVisitor visitor;
  vma_allocator->Accept(&visitor);
}

TEST(VirtualMemoryAutoGrowthBestFitAllocator, TestAllBlocksInfoVisitor) {
  auto vmm_cuda_allocator =
      std::make_shared<CUDAVirtualMemAllocator>(phi::GPUPlace());
  auto vma_allocator =
      std::make_shared<VirtualMemoryAutoGrowthBestFitAllocator>(
          vmm_cuda_allocator, platform::GpuMinChunkSize(), phi::GPUPlace());

  auto allocation = vma_allocator->Allocate(platform::GpuMinChunkSize());
  AllBlocksInfoVisitor visitor;
  vma_allocator->Accept(&visitor);
  auto all_blocks_info = visitor.GetAllBlocksInfo();

  ASSERT_EQ(all_blocks_info.size(), 1UL);
  ASSERT_GE(all_blocks_info[0].size(), 1UL);
  uintptr_t expected_addr = std::get<1>(all_blocks_info[0][0]);
  size_t total_size = 0;
  bool found_allocated_block = false;
  for (const auto& block_info : all_blocks_info[0]) {
    const size_t block_size = std::get<0>(block_info);
    const uintptr_t block_addr = std::get<1>(block_info);
    EXPECT_GT(block_size, 0UL);
    EXPECT_EQ(block_addr, expected_addr);
    expected_addr += block_size;
    total_size += block_size;
    if (std::get<1>(block_info) ==
        reinterpret_cast<uintptr_t>(allocation->ptr())) {
      EXPECT_EQ(std::get<0>(block_info), platform::GpuMinChunkSize());
      EXPECT_FALSE(std::get<2>(block_info));
      found_allocated_block = true;
    }
  }
  EXPECT_TRUE(found_allocated_block);

  allocation.reset();
  AllBlocksInfoVisitor after_free_visitor;
  vma_allocator->Accept(&after_free_visitor);
  auto after_free_info = after_free_visitor.GetAllBlocksInfo();

  ASSERT_EQ(after_free_info.size(), 1UL);
  ASSERT_EQ(after_free_info[0].size(), 1UL);
  EXPECT_EQ(std::get<0>(after_free_info[0][0]), total_size);
  EXPECT_EQ(std::get<1>(after_free_info[0][0]),
            std::get<1>(all_blocks_info[0][0]));
  EXPECT_TRUE(std::get<2>(after_free_info[0][0]));
}

TEST(AutoGrowthBestFitAllocator, TestAllBlocksInfoVisitor) {
  constexpr size_t kAlignment = 256;
  constexpr size_t kChunkSize = 4096;
  auto cpu_allocator = std::make_shared<CPUAllocator>();
  auto ag_allocator = std::make_shared<AutoGrowthBestFitAllocator>(
      cpu_allocator, kAlignment, kChunkSize);

  auto first = ag_allocator->Allocate(1024);
  auto second = ag_allocator->Allocate(2048);
  AllBlocksInfoVisitor visitor;
  ag_allocator->Accept(&visitor);
  auto all_blocks_info = visitor.GetAllBlocksInfo();

  ASSERT_EQ(all_blocks_info.size(), 1UL);
  ASSERT_EQ(all_blocks_info[0].size(), 3UL);
  uintptr_t expected_addr = std::get<1>(all_blocks_info[0][0]);
  size_t total_size = 0;
  for (const auto& block_info : all_blocks_info[0]) {
    const size_t block_size = std::get<0>(block_info);
    const uintptr_t block_addr = std::get<1>(block_info);
    EXPECT_GT(block_size, 0UL);
    EXPECT_EQ(block_addr, expected_addr);
    expected_addr += block_size;
    total_size += block_size;
  }
  ASSERT_EQ(total_size, kChunkSize);

  EXPECT_EQ(std::get<0>(all_blocks_info[0][0]), 1024UL);
  EXPECT_TRUE(std::get<2>(all_blocks_info[0][0]));
  EXPECT_EQ(std::get<0>(all_blocks_info[0][1]), 2048UL);
  EXPECT_EQ(std::get<1>(all_blocks_info[0][1]),
            reinterpret_cast<uintptr_t>(second->ptr()));
  EXPECT_FALSE(std::get<2>(all_blocks_info[0][1]));
  EXPECT_EQ(std::get<0>(all_blocks_info[0][2]), 1024UL);
  EXPECT_EQ(std::get<1>(all_blocks_info[0][2]),
            reinterpret_cast<uintptr_t>(first->ptr()));
  EXPECT_FALSE(std::get<2>(all_blocks_info[0][2]));

  first.reset();
  second.reset();
  AllBlocksInfoVisitor after_free_visitor;
  ag_allocator->Accept(&after_free_visitor);
  auto after_free_info = after_free_visitor.GetAllBlocksInfo();

  ASSERT_EQ(after_free_info.size(), 1UL);
  ASSERT_EQ(after_free_info[0].size(), 1UL);
  EXPECT_EQ(std::get<0>(after_free_info[0][0]), kChunkSize);
  EXPECT_EQ(std::get<1>(after_free_info[0][0]),
            std::get<1>(all_blocks_info[0][0]));
  EXPECT_TRUE(std::get<2>(after_free_info[0][0]));
}

}  // namespace allocation
}  // namespace memory
}  // namespace paddle
