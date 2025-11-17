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
#ifdef PADDLE_WITH_CUDA
#include <cuda.h>
#include <cuda_runtime.h>
#endif
#include "glog/logging.h"
#include "gtest/gtest.h"
PD_DECLARE_uint64(vmm_small_pool_pre_alloc_in_mb);
PD_DECLARE_uint64(vmm_large_pool_pre_alloc_in_mb);
PD_DECLARE_uint64(vmm_pre_alloc_in_mb);
PD_DECLARE_uint64(vmm_small_pool_size_in_mb);

namespace paddle {
namespace memory {
namespace allocation {

// Test fixture
class VirtualMemoryAutoGrowthBestFitMultiScalePoolAllocatorTest
    : public ::testing::Test {
 protected:
  void SetUp() override {
    auto vmm_cuda_allocator_small =
        std::make_shared<CUDAVirtualMemAllocator>(phi::GPUPlace(0));
    auto vmm_cuda_allocator_large =
        std::make_shared<CUDAVirtualMemAllocator>(phi::GPUPlace(0));
    // Create mock underlying allocators
    auto underlying_small =
        std::make_shared<VirtualMemoryAutoGrowthBestFitAllocator>(
            vmm_cuda_allocator_small, platform::GpuMinChunkSize(), GPUPlace(0));
    auto underlying_large =
        std::make_shared<VirtualMemoryAutoGrowthBestFitAllocator>(
            vmm_cuda_allocator_large, platform::GpuMinChunkSize(), GPUPlace(0));

    // Create the multi-scale pool allocator
    multi_scale_allocator_ =
        std::make_shared<VirtualMemoryAutoGrowthBestFitMultiScalePoolAllocator>(
            underlying_small,
            underlying_large,
            platform::GpuMinChunkSize(),
            GPUPlace(0));

    small_allocator_ = underlying_small;
    large_allocator_ = underlying_large;
  }

  size_t mb = (1 << 20);
  std::shared_ptr<VirtualMemoryAutoGrowthBestFitAllocator> small_allocator_;
  std::shared_ptr<VirtualMemoryAutoGrowthBestFitAllocator> large_allocator_;
  std::shared_ptr<VirtualMemoryAutoGrowthBestFitMultiScalePoolAllocator>
      multi_scale_allocator_;
};

// Test case for small pool pre-allocation only
TEST_F(VirtualMemoryAutoGrowthBestFitMultiScalePoolAllocatorTest,
       PreAllocSmallPoolOnly) {
  // Set flags for small pool pre-allocation
  FLAGS_vmm_small_pool_pre_alloc_in_mb = 10;  // 10 MB
  FLAGS_vmm_large_pool_pre_alloc_in_mb = 0;   // No large pool pre-allocation

  multi_scale_allocator_->PreAlloc();
  EXPECT_EQ(DeviceMemoryStatCurrentValue("Reserved", 0), 12 * mb);
}

// Test case for large pool pre-allocation only
TEST_F(VirtualMemoryAutoGrowthBestFitMultiScalePoolAllocatorTest,
       PreAllocLargePoolOnly) {
  // Set flags for large pool pre-allocation
  FLAGS_vmm_small_pool_pre_alloc_in_mb = 0;   // No small pool pre-allocation
  FLAGS_vmm_large_pool_pre_alloc_in_mb = 20;  // 20 MB

  multi_scale_allocator_->PreAlloc();
  EXPECT_EQ(DeviceMemoryStatCurrentValue("Reserved", 0), 22 * mb);
}

// Test case for both pools pre-allocation
TEST_F(VirtualMemoryAutoGrowthBestFitMultiScalePoolAllocatorTest,
       PreAllocBothPools) {
  // Set flags for both pools pre-allocation
  FLAGS_v = 4;
  FLAGS_vmm_small_pool_pre_alloc_in_mb = 5;   // 5 MB
  FLAGS_vmm_large_pool_pre_alloc_in_mb = 15;  // 15 MB

  multi_scale_allocator_->PreAlloc();
  EXPECT_EQ(DeviceMemoryStatCurrentValue("Reserved", 0), 22 * mb);
}

// Test case for no pre-allocation
TEST_F(VirtualMemoryAutoGrowthBestFitMultiScalePoolAllocatorTest,
       PreAllocNone) {
  // Set flags for no pre-allocation
  FLAGS_vmm_small_pool_pre_alloc_in_mb = 0;  // No pre-allocation
  FLAGS_vmm_large_pool_pre_alloc_in_mb = 0;  // No pre-allocation

  multi_scale_allocator_->PreAlloc();
  EXPECT_EQ(DeviceMemoryStatCurrentValue("Reserved", 0), 0 * mb);
}

TEST_F(VirtualMemoryAutoGrowthBestFitMultiScalePoolAllocatorTest,
       PreAllocWithZeroSize) {
  FLAGS_v = 4;
  FLAGS_vmm_pre_alloc_in_mb = 0;
  small_allocator_->PreAlloc();
  EXPECT_EQ(DeviceMemoryStatCurrentValue("Reserved", 0), 0 * mb);
}

TEST_F(VirtualMemoryAutoGrowthBestFitMultiScalePoolAllocatorTest,
       PreAllocWithPositiveSize) {
  FLAGS_vmm_pre_alloc_in_mb = 10;  // 10MB
  large_allocator_->PreAlloc();
  EXPECT_EQ(DeviceMemoryStatCurrentValue("Reserved", 0), 12 * mb);
}

TEST_F(VirtualMemoryAutoGrowthBestFitMultiScalePoolAllocatorTest,
       MultiScaleAlloc) {
  FLAGS_vmm_small_pool_size_in_mb = 20;

  auto allocation_small = multi_scale_allocator_->Allocate(10 * mb);
  auto allocation_large = multi_scale_allocator_->Allocate(30 * mb);
  auto safe = multi_scale_allocator_->IsAllocThreadSafe();
  EXPECT_EQ(DeviceMemoryStatCurrentValue("Reserved", 0), 44 * mb);
  EXPECT_EQ(safe, true);
}

}  // namespace allocation
}  // namespace memory
}  // namespace paddle
