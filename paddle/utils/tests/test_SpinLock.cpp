/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserve.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include <vector>

#include <gflags/gflags.h>
#include <gtest/gtest.h>

#include "paddle/utils/Locks.h"
#include "paddle/utils/Logging.h"
#include "paddle/utils/Util.h"

DEFINE_int32(test_thread_num, 100, "testing thread number");

void testNormalImpl(
    size_t thread_num,
    const std::function<void(size_t, size_t&, paddle::SpinLock&)>& callback) {
  paddle::SpinLock mutex;
  std::vector<std::thread> threads;
  threads.reserve(thread_num);

  size_t count = 0;
  for (size_t i = 0; i < thread_num; ++i) {
    threads.emplace_back([&thread_num, &count, &mutex, &callback] {
      callback(thread_num, count, mutex);
    });
  }
  for (auto& thread : threads) {
    thread.join();
  }
  // Check whether all threads reach this point or not
  CHECK_EQ(count, thread_num);
}

TEST(ThreadSpinLock, normalTest) {
  for (auto& thread_num : {10, 30, 50, 100, 300, 1000}) {
    testNormalImpl(
        thread_num,
        [](size_t thread_num, size_t& count, paddle::SpinLock& mutex) {
          std::lock_guard<paddle::SpinLock> lock(mutex);
          ++count;
        });
  }
}
