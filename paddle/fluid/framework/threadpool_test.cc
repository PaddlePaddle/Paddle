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

void do_sum(std::vector<std::future<void>>* fs, std::mutex* mu,
            std::atomic<int>* sum, int cnt) {
  for (int i = 0; i < cnt; ++i) {
    std::lock_guard<std::mutex> l(*mu);
    fs->push_back(framework::Async([sum]() { sum->fetch_add(1); }));
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
static int64_t GetTS() {
  struct timeval tp;
  gettimeofday(&tp, NULL);
  return tp.tv_sec * 1000000 + tp.tv_usec;
}

void multi_call(std::function<void()> call) {
  for (int i = 0; i < 500; ++i) {
    call();
  }
}

TEST(ThreadPool, PERFORMANCE) {
  auto sum = [] {
    int a = 0;
    for (int i = 0; i < 1000; ++i) {
      a += i;
    }
  };
  // framework::ThreadPool *pool = new framework::ThreadPool(2);
  int64_t start = GetTS();
  for (int i = 0; i < 1000; ++i) {
    // int64_t s = GetTS();
    framework::Async(std::move(sum));
    // pool->Run(std::move(sum));
    // VLOG(5) << "push to pool spent : " << GetTS() - s << " (us).";
  }
  VLOG(5) << "pool spent: " << GetTS() - start << " (us).";
  start = GetTS();
  for (int i = 0; i < 1000; ++i) {
    sum();
  }
  VLOG(5) << "sequence call spent: " << GetTS() - start << " (us).";
  std::vector<std::thread> threads;
  start = GetTS();
  for (int i = 0; i < 2; ++i) {
    std::thread t(multi_call, std::ref(sum));
    threads.push_back(std::move(t));
  }
  for (auto& thread : threads) {
    thread.join();
  }
  VLOG(5) << "two threads spent: " << GetTS() - start << " (us).";
}
