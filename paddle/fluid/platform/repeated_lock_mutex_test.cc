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

#include "paddle/fluid/platform/repeated_lock_mutex.h"
#include <gtest/gtest.h>
#include <condition_variable>  // NOLINT
#include <mutex>               // NOLINT
#include <thread>              // NOLINT
#include <vector>

namespace paddle {
namespace platform {

TEST(repeated_lock_mutex, multiple_lock_unlock) {
  RepeatedLockMutex<std::mutex> mtx;
  size_t lock_num = 10;
  size_t unlock_num = 5;
  ASSERT_FALSE(mtx.is_locked());
  for (size_t i = 0; i < lock_num; ++i) {
    if (i == 0) {
      ASSERT_FALSE(mtx.is_locked());
    } else {
      ASSERT_TRUE(mtx.is_locked());
    }
    mtx.lock();
    ASSERT_TRUE(mtx.is_locked());
  }

  for (size_t i = 0; i < unlock_num; ++i) {
    if (i == 0) {
      ASSERT_TRUE(mtx.is_locked());
    } else {
      ASSERT_FALSE(mtx.is_locked());
    }
    mtx.unlock();
    ASSERT_FALSE(mtx.is_locked());
  }
}

TEST(repeated_lock_mutex, thread_local_status) {
  RepeatedLockMutex<std::mutex> mtx;
  mtx.lock();
  bool is_locked = mtx.is_locked();
  ASSERT_TRUE(is_locked);
  std::thread th([&] { is_locked = mtx.is_locked(); });
  th.join();
  ASSERT_FALSE(is_locked);
}

TEST(repeated_lock_mutex, multi_threads) {
  RepeatedLockMutex<std::mutex> mtx;
  using MutexType = decltype(mtx);
  std::condition_variable_any cv;

  bool flag = false;

  size_t thread_num = 10;

  size_t turns = 100;
  size_t cur_val = 0;

  std::vector<std::thread> ths;
  std::vector<std::vector<size_t>> ret(thread_num);
  for (size_t i = 0; i < thread_num; ++i) {
    ths.emplace_back([&, i]() {
      {
        std::unique_lock<MutexType> lock(mtx);
        cv.wait(lock, [&] { return flag; });
      }

      for (size_t j = 0; j < turns; ++j) {
        {
          std::unique_lock<MutexType> lock(mtx);
          cv.wait(lock, [&] { return cur_val % thread_num == i; });
          ret[i].push_back(cur_val++);
        }
        cv.notify_all();
      }
    });
  }

  {
    std::lock_guard<MutexType> lock(mtx);
    flag = true;
  }
  cv.notify_all();

  for (auto &th : ths) {
    th.join();
  }

  for (size_t i = 0; i < thread_num; ++i) {
    ASSERT_EQ(ret[i].size(), turns);
    for (size_t j = 0; j < turns; ++j) {
      ASSERT_EQ(ret[i][j], i + j * thread_num);
    }
  }
}

}  // namespace platform
}  // namespace paddle
