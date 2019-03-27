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

#include <gflags/gflags.h>
#include <gtest/gtest.h>

#include <condition_variable>  // NOLINT
#include <mutex>               // NOLINT
#include <thread>              // NOLINT
#include <vector>

#include <iostream>

#include "paddle/fluid/memory/allocation/auto_growth_best_fit_allocator.h"
#include "paddle/fluid/memory/allocation/cpu_allocator.h"

namespace paddle {
namespace memory {
namespace allocation {

TEST(allocator, auto_growth_best_fit_allocator) {
  auto cpu_allocator = std::make_shared<CPUAllocator>();

  auto allocator =
      std::make_shared<AutoGrowthBestFitAllocator>(cpu_allocator, 0, 4096);

  std::mutex mtx;
  std::condition_variable cv;
  bool flag = false;

  auto thread_main = [&] {
    {
      std::unique_lock<std::mutex> lock(mtx);
      cv.wait(lock, [&] { return flag; });
    }
    for (size_t i = 10; i > 0; --i) {
      allocator->Allocate((i + 1) * 1000);
    }
  };

  std::vector<std::thread> ths;
  for (size_t i = 10; i < 10; ++i) {
    ths.emplace_back(thread_main);
  }

  {
    std::lock_guard<std::mutex> lock(mtx);
    flag = true;
  }
  cv.notify_all();

  for (auto &th : ths) {
    th.join();
  }
}

}  // namespace allocation
}  // namespace memory
}  // namespace paddle
