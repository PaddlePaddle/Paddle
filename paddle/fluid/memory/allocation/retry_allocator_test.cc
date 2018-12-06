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

#include "paddle/fluid/memory/allocation/retry_allocator.h"
#include <algorithm>
#include <chrono>              // NOLINT
#include <condition_variable>  // NOLINT
#include <mutex>               // NOLINT
#include <thread>              // NOLINT
#include <vector>
#include "gtest/gtest.h"
#include "paddle/fluid/memory/allocation/best_fit_allocator.h"
#include "paddle/fluid/memory/allocation/cpu_allocator.h"
#include "paddle/fluid/memory/allocation/locked_allocator.h"

namespace paddle {
namespace memory {
namespace allocation {

TEST(RetryAllocator, RetryAllocator) {
  CPUAllocator cpu_allocator;

  size_t size = (1 << 20);
  auto cpu_allocation = cpu_allocator.Allocate(size, cpu_allocator.kDefault);

  std::unique_ptr<BestFitAllocator> best_fit_allocator(
      new BestFitAllocator(cpu_allocation.get()));
  std::unique_ptr<LockedAllocator> locked_allocator(
      new LockedAllocator(std::move(best_fit_allocator)));

  size_t thread_num = 32;
  size_t sleep_time = 40;
  size_t extra_time = 10;

  // Reserve to perform more tests in the future
  std::vector<std::shared_ptr<Allocator>> allocators;
  {
    std::unique_ptr<BestFitAllocator> best_fit_allocator(
        new BestFitAllocator(cpu_allocation.get()));
    std::unique_ptr<LockedAllocator> locked_allocator(
        new LockedAllocator(std::move(best_fit_allocator)));
    allocators.push_back(std::make_shared<RetryAllocator>(
        std::move(locked_allocator),
        (thread_num - 1) * (sleep_time + extra_time)));
  }

  for (auto &allocator : allocators) {
    std::vector<std::thread> threads(thread_num);
    std::vector<void *> addresses(threads.size(), nullptr);

    std::mutex mutex;
    std::condition_variable cv;
    bool flag = false;

    for (size_t i = 0; i < threads.size(); ++i) {
      threads[i] = std::thread([&, i]() {
        {
          std::unique_lock<std::mutex> lock(mutex);
          cv.wait(lock, [&] { return flag; });
        }

        auto ret = allocator->Allocate(size - 1);
        addresses[i] = ret->ptr();
        std::this_thread::sleep_for(std::chrono::milliseconds(sleep_time));
      });
    }

    {
      std::lock_guard<std::mutex> lock(mutex);
      flag = true;
      cv.notify_all();
    }

    for (auto &th : threads) {
      th.join();
    }

    void *val = cpu_allocation->ptr();
    bool is_all_equal = std::all_of(addresses.begin(), addresses.end(),
                                    [val](void *p) { return p == val; });
    ASSERT_TRUE(is_all_equal);
  }
}

}  // namespace allocation
}  // namespace memory
}  // namespace paddle
