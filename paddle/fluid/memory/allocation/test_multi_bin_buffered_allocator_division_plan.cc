// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include "gtest/gtest.h"
#include "paddle/fluid/memory/allocation/multi_bin_buffered_allocator.h"

DECLARE_string(buffered_allocator_division_plan_path);

namespace paddle {
namespace memory {
namespace allocation {

TEST(buffered_allocator, division_plan) {
  std::string path = "/tmp/buffered_allocator_divison_plan";
  FLAGS_buffered_allocator_division_plan_path = path;

  {
    std::vector<std::string> plan(
        {"100b", "300.7K", "500.3m", "1.02gB", "2g", "4G"});

    std::ofstream os(path);
    for (auto &p : plan) {
      os << p << std::endl;
    }
    os.close();
  }

  auto plan = ReadBufferedAllocatorDivisionPlanFromFile(
      FLAGS_buffered_allocator_division_plan_path);
  ASSERT_EQ(plan.size(), 6UL);
  ASSERT_EQ(plan[0], 100UL);
  ASSERT_EQ(plan[1], static_cast<size_t>(300.7 * 1024));
  ASSERT_EQ(plan[2], static_cast<size_t>(500.3 * 1024 * 1024));
  ASSERT_EQ(plan[3], static_cast<size_t>(1.02 * 1024 * 1024 * 1024));
  ASSERT_EQ(plan[4], static_cast<size_t>(2.0 * 1024 * 1024 * 1024));
  ASSERT_EQ(plan[5], static_cast<size_t>(4.0 * 1024 * 1024 * 1024));
}

}  // namespace allocation
}  // namespace memory
}  // namespace paddle
