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

#include "paddle/phi/core/memory/allocation/thread_local_allocator.h"

#include <condition_variable>  // NOLINT
#include <thread>              // NOLINT

#include "gtest/gtest.h"
#include "paddle/common/flags.h"
#include "paddle/phi/core/memory/malloc.h"

COMMON_DECLARE_double(fraction_of_gpu_memory_to_use);
COMMON_DECLARE_string(allocator_strategy);

namespace paddle {
namespace memory {
namespace allocation {

TEST(ThreadLocalAllocator, cross_scope_release) {
  FLAGS_fraction_of_gpu_memory_to_use = 0.1;
  FLAGS_allocator_strategy = "thread_local";

  const size_t thread_num = 5;
  const std::vector<int> devices = platform::GetSelectedDevices();

  std::vector<std::vector<void *>> allocator_addresses(devices.size());
  std::vector<std::vector<AllocationPtr>> thread_allocations(devices.size());

  for (size_t i = 0; i < devices.size(); ++i) {
    allocator_addresses[i].resize(thread_num);
    thread_allocations[i].resize(thread_num);
  }

  std::vector<std::thread> threads(thread_num);
  std::mutex mutex;
  std::condition_variable cv;
  bool flag = false;

  for (size_t i = 0; i < threads.size(); ++i) {
    threads[i] = std::thread([&, i]() {
      {
        std::unique_lock<std::mutex> lock(mutex);
        cv.wait(lock, [&] { return flag; });
      }
      for (size_t j = 0; j < devices.size(); ++j) {
        thread_allocations[j][i] = memory::Alloc(phi::GPUPlace(devices[j]), 10);
        auto tl_allocator_impl =
            ThreadLocalCUDAAllocatorPool::Instance().Get(devices[j]);
        allocator_addresses[j][i] = tl_allocator_impl.get();
        memory::Release(phi::GPUPlace(devices[j]));
      }
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

  for (auto &addresses : allocator_addresses) {
    std::sort(addresses.begin(), addresses.end());
    ASSERT_EQ(std::adjacent_find(
                  addresses.begin(), addresses.end(), std::equal_to<>()),
              addresses.end());
  }

  ::testing::FLAGS_gtest_death_test_style = "threadsafe";
  ASSERT_EXIT(([&]() { thread_allocations.clear(); }(), exit(0)),
              ::testing::ExitedWithCode(0),
              ".*");
}

}  // namespace allocation
}  // namespace memory
}  // namespace paddle
