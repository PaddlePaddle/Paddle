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

#include "paddle/fluid/memory/allocation/allocator_facade.h"
#include <gflags/gflags.h>
#include <gtest/gtest.h>

DECLARE_double(fraction_of_gpu_memory_to_use);
DECLARE_int32(gpu_allocator_retry_time);

namespace paddle {
namespace memory {
namespace allocation {

TEST(allocator, allocator) {
  FLAGS_fraction_of_gpu_memory_to_use = 0.01;
  FLAGS_gpu_allocator_retry_time = 500;

  auto &instance = AllocatorFacade::Instance();

  {
    auto cpu_allocation = instance.Alloc(platform::CPUPlace(), 1024);
    ASSERT_NE(cpu_allocation, nullptr);
  }

  {
    auto gpu_allocation = instance.Alloc(platform::CUDAPlace(0), 1024);
    ASSERT_NE(gpu_allocation, nullptr);
  }

  {
    // Allocate 2GB gpu memory
    auto gpu_allocation = instance.Alloc(platform::CUDAPlace(0),
                                         2 * static_cast<size_t>(1 << 30));
    ASSERT_NE(gpu_allocation, nullptr);
  }

  {}
}

}  // namespace allocation
}  // namespace memory
}  // namespace paddle
