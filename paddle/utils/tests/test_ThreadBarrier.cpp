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

#include <set>
#include <vector>

#include <gflags/gflags.h>
#include <gtest/gtest.h>

#include "paddle/utils/Locks.h"
#include "paddle/utils/Logging.h"
#include "paddle/utils/Util.h"

DEFINE_int32(test_thread_num, 100, "testing thread number");

void testNormalImpl(
    size_t thread_num,
    const std::function<void(size_t,
                             std::mutex&,
                             std::set<std::thread::id>&,
                             paddle::ThreadBarrier&)>& callback) {
  std::mutex mutex;
  std::set<std::thread::id> tids;
  paddle::ThreadBarrier barrier(thread_num);

  std::vector<std::thread> threads;
  threads.reserve(thread_num);
  for (size_t i = 0; i < thread_num; ++i) {
    threads.emplace_back([&thread_num, &mutex, &tids, &barrier, &callback] {
      callback(thread_num, mutex, tids, barrier);
    });
  }

  for (auto& thread : threads) {
    thread.join();
  }
}

TEST(ThreadBarrier, normalTest) {
  for (auto& thread_num : {10, 30, 50, 100, 300, 1000}) {
    testNormalImpl(thread_num,
                   [](size_t thread_num,
                      std::mutex& mutex,
                      std::set<std::thread::id>& tids,
                      paddle::ThreadBarrier& barrier) {
                     {
                       std::lock_guard<std::mutex> guard(mutex);
                       tids.insert(std::this_thread::get_id());
                     }
                     barrier.wait();
                     // Check whether all threads reach this point or not
                     CHECK_EQ(tids.size(), thread_num);
                   });
  }
}
