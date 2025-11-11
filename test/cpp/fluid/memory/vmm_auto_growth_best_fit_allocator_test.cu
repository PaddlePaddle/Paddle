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
#include "paddle/phi/core/memory/allocation/virtual_memory_auto_growth_best_fit_allocator.h"

#include "gtest/gtest.h"
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

}  // namespace allocation
}  // namespace memory
}  // namespace paddle
