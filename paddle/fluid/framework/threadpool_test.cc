/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include <gtest/gtest.h>
#include <atomic>

#include "paddle/fluid/framework/threadpool.h"

namespace framework = paddle::framework;

void do_sum(framework::ThreadPool* pool, std::atomic<int>* sum, int cnt) {
  std::vector<std::future<void>> fs;
  for (int i = 0; i < cnt; ++i) {
    fs.push_back(framework::Async([sum]() { sum->fetch_add(1); }));
  }
}

TEST(ThreadPool, ConcurrentInit) {
  framework::ThreadPool* pool;
  int n = 50;
  std::vector<std::thread> threads;
  for (int i = 0; i < n; ++i) {
    std::thread t([&pool]() { pool = framework::ThreadPool::GetInstance(); });
    threads.push_back(std::move(t));
  }
  for (auto& t : threads) {
    t.join();
  }
}

TEST(ThreadPool, ConcurrentRun) {
  framework::ThreadPool* pool = framework::ThreadPool::GetInstance();
  std::atomic<int> sum(0);
  std::vector<std::thread> threads;
  int n = 50;
  // sum = (n * (n + 1)) / 2
  for (int i = 1; i <= n; ++i) {
    std::thread t(do_sum, pool, &sum, i);
    threads.push_back(std::move(t));
  }
  for (auto& t : threads) {
    t.join();
  }
  pool->Wait();
  EXPECT_EQ(sum, ((n + 1) * n) / 2);
}
