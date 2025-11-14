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
#include "paddle/phi/core/memory/allocation/cuda_virtual_mem_allocator.h"
#include "paddle/phi/core/memory/allocation/retry_allocator.h"
#include "paddle/phi/core/memory/allocation/virtual_memory_auto_growth_best_fit_allocator.h"
#include "paddle/phi/core/memory/memory.h"
#include "paddle/phi/core/platform/device/gpu/gpu_info.h"

#include "gtest/gtest.h"
namespace paddle {
namespace memory {
namespace allocation {

TEST(VirtualMemoryAutoGrowthBestFitAllocator, TestAllocatorVisitor) {
  auto vmm_cuda_allocator =
      std::make_shared<CUDAVirtualMemAllocator>(phi::GPUPlace());
  auto vma_allocator =
      std::make_shared<VirtualMemoryAutoGrowthBestFitAllocator>(
          vmm_cuda_allocator, platform::GpuMinChunkSize(), phi::GPUPlace());
  size_t mb = (1 << 20);
  auto allocation1 = vma_allocator->Allocate(10 * mb);
  auto allocation2 = vma_allocator->Allocate(20 * mb);
  auto allocation3 = vma_allocator->Allocate(30 * mb);
  auto allocation4 = vma_allocator->Allocate(40 * mb);
  allocation2.reset();
  allocation4.reset();
  auto allocation5 = vma_allocator->Allocate(50 * mb);
  EXPECT_GT(DeviceMemoryStatCurrentValue("Reserved", 0), 110 * mb);
}

}  // namespace allocation
}  // namespace memory
}  // namespace paddle
