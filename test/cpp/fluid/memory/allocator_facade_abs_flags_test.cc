// Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

#include <gtest/gtest.h>

#include "paddle/common/flags.h"
#include "paddle/phi/core/memory/allocation/allocator_facade.h"

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
COMMON_DECLARE_double(fraction_of_gpu_memory_to_use);
COMMON_DECLARE_double(fraction_of_cuda_pinned_memory_to_use);
COMMON_DECLARE_uint64(initial_gpu_memory_in_mb);
COMMON_DECLARE_uint64(reallocate_gpu_memory_in_mb);
PD_DECLARE_int64(gpu_allocator_retry_time);
#endif
COMMON_DECLARE_string(allocator_strategy);

namespace paddle {
namespace memory {
namespace allocation {

//! Run allocate test cases for different places
void AllocateTestCases() {
  auto &instance = AllocatorFacade::Instance();
  phi::Place place;
  size_t size = 1024;

  {
    place = phi::CPUPlace();
    size = 1024;
    auto cpu_allocation = instance.Alloc(place, size);
    ASSERT_NE(cpu_allocation, nullptr);
    ASSERT_NE(cpu_allocation->ptr(), nullptr);
    ASSERT_EQ(cpu_allocation->place(), place);
    ASSERT_EQ(cpu_allocation->size(), size);
  }

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
  {
    place = phi::GPUPlace(0);
    size = 1024;
    auto gpu_allocation = instance.Alloc(place, size);
    ASSERT_NE(gpu_allocation, nullptr);
    ASSERT_NE(gpu_allocation->ptr(), nullptr);
    ASSERT_EQ(gpu_allocation->place(), place);
    ASSERT_GE(gpu_allocation->size(), size);
  }

  {
    // Allocate 2GB gpu memory
    place = phi::GPUPlace(0);
    size = 2 * static_cast<size_t>(1 << 30);
    auto gpu_allocation = instance.Alloc(place, size);
    ASSERT_NE(gpu_allocation, nullptr);
    ASSERT_NE(gpu_allocation->ptr(), nullptr);
    ASSERT_EQ(gpu_allocation->place(), place);
    ASSERT_GE(gpu_allocation->size(), size);
  }

  {
    place = phi::GPUPinnedPlace();
    size = (1 << 20);
    auto cuda_pinned_allocation =
        instance.Alloc(phi::GPUPinnedPlace(), 1 << 20);
    ASSERT_NE(cuda_pinned_allocation, nullptr);
    ASSERT_NE(cuda_pinned_allocation->ptr(), nullptr);
    ASSERT_EQ(cuda_pinned_allocation->place(), place);
    ASSERT_GE(cuda_pinned_allocation->size(), size);
  }
#endif
}

TEST(Allocator, SpecifyGpuMemory) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
  // Set to 0.0 to test FLAGS_initial_gpu_memory_in_mb and
  // FLAGS_reallocate_gpu_memory_in_mb
  FLAGS_fraction_of_gpu_memory_to_use = 0.0;
  // 512 MB
  FLAGS_initial_gpu_memory_in_mb = 512;
  // 4 MB
  FLAGS_reallocate_gpu_memory_in_mb = 4;
  FLAGS_gpu_allocator_retry_time = 500;
  FLAGS_fraction_of_cuda_pinned_memory_to_use = 0.5;
#endif

  FLAGS_allocator_strategy = "naive_best_fit";

  AllocateTestCases();
}

}  // namespace allocation
}  // namespace memory
}  // namespace paddle
