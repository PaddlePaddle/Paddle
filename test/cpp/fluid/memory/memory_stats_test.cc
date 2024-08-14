// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

#include <algorithm>
#include <vector>

#include "gtest/gtest.h"
#include "paddle/phi/core/memory/memory.h"

namespace paddle {
namespace memory {

TEST(stat_allocator_test, host_memory_stat_test) {
  std::vector<int64_t> alloc_sizes{
      5278, 9593, 8492, 5041, 3351, 4232, 3706, 5963, 5896, 5057, 7527,
      6235, 0,    7810, 940,  1239, 1945, 789,  2891, 7553, 8046, 2685,
      1332, 6547, 5238, 5345, 1133, 5475, 9137, 3111, 8478, 6350, 9395,
      4,    1185, 2186, 357,  9774, 6743, 6136, 7073, 7674, 5640, 3935,
      528,  6699, 9821, 8717, 2264, 4708, 9936, 3566, 1373, 6955, 3694,
      221,  309,  3617, 3793, 3334, 7281, 1302};

  int64_t max_alloc_size = 0;
  for (int64_t size : alloc_sizes) {
    AllocationPtr allocation = Alloc(phi::CPUPlace(), size);
    int64_t alloc_size = static_cast<int64_t>(allocation->size());
    max_alloc_size = std::max(max_alloc_size, alloc_size);
    EXPECT_EQ(HostMemoryStatCurrentValue("Allocated", 0), alloc_size);
  }
  EXPECT_EQ(HostMemoryStatPeakValue("Allocated", 0), max_alloc_size);
}

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
TEST(stat_allocator_test, device_memory_stat_test) {
  std::vector<int64_t> alloc_sizes{
      5278, 9593, 8492, 5041, 3351, 4232, 3706, 5963, 5896, 5057, 7527,
      6235, 0,    7810, 940,  1239, 1945, 789,  2891, 7553, 8046, 2685,
      1332, 6547, 5238, 5345, 1133, 5475, 9137, 3111, 8478, 6350, 9395,
      4,    1185, 2186, 357,  9774, 6743, 6136, 7073, 7674, 5640, 3935,
      528,  6699, 9821, 8717, 2264, 4708, 9936, 3566, 1373, 6955, 3694,
      221,  309,  3617, 3793, 3334, 7281, 1302};

  int64_t max_alloc_size = 0;
  for (int64_t size : alloc_sizes) {
    AllocationPtr allocation = Alloc(phi::GPUPlace(), size);
    int64_t alloc_size = static_cast<int64_t>(allocation->size());
    max_alloc_size = std::max(max_alloc_size, alloc_size);
    EXPECT_EQ(DeviceMemoryStatCurrentValue("Allocated", 0), alloc_size);
  }
  EXPECT_EQ(DeviceMemoryStatPeakValue("Allocated", 0), max_alloc_size);
}
#endif

}  // namespace memory
}  // namespace paddle
