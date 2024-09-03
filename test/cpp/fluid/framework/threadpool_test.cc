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

#include "paddle/fluid/framework/threadpool.h"

#include <gtest/gtest.h>

#include <atomic>

namespace framework = paddle::framework;

void do_sum(std::vector<std::future<void>>* fs,
            std::mutex* mu,
            std::atomic<int>* sum,
            int cnt) {
  for (int i = 0; i < cnt; ++i) {
    std::lock_guard<std::mutex> l(*mu);
    fs->push_back(phi::Async([sum]() { sum->fetch_add(1); }));
  }
}

TEST(ThreadPool, ConcurrentInit) {
  phi::ThreadPool* pool;
  int n = 50;
  std::vector<std::thread> threads;
  for (int i = 0; i < n; ++i) {
    std::thread t([&pool]() { pool = phi::ThreadPool::GetInstance(); });
    threads.push_back(std::move(t));
  }
  for (auto& t : threads) {
    t.join();
  }
}

TEST(ThreadPool, ConcurrentRun) {
  std::atomic<int> sum(0);
  std::vector<std::thread> threads;
  std::vector<std::future<void>> fs;
  std::mutex fs_mu;
  int n = 50;
  // sum = (n * (n + 1)) / 2
  for (int i = 1; i <= n; ++i) {
    std::thread t(do_sum, &fs, &fs_mu, &sum, i);
    threads.push_back(std::move(t));
  }
  for (auto& t : threads) {
    t.join();
  }
  for (auto& t : fs) {
    t.wait();
  }
  EXPECT_EQ(sum, ((n + 1) * n) / 2);
}
