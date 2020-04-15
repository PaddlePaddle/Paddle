// Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/memory/allocation/naive_best_fit_allocator.h"
#include <gtest/gtest.h>
#include <memory>

namespace paddle {
namespace memory {
namespace allocation {

TEST(NaiveBestFitAllocator, cross_scope_release) {
  auto test_allocate = [](size_t size) {
    AllocationPtr allocation;
    {
      std::shared_ptr<Allocator> allocator(
          new NaiveBestFitAllocator(platform::CPUPlace()));
      allocation = std::move(allocator->Allocate(size));
    }
  };
  ::testing::FLAGS_gtest_death_test_style = "threadsafe";
  ASSERT_EXIT((test_allocate(10), exit(0)), ::testing::ExitedWithCode(0), ".*");
}

}  // namespace allocation
}  // namespace memory
}  // namespace paddle
