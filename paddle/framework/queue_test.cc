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

#include "paddle/framework/queue.h"

#include <gtest/gtest.h>
#include <paddle/platform/variant.h>
#include <thread>

TEST(queue, PushPullSingleThread) {
  using namespace paddle::framework;

  Queue<int> q;
  q.Push(10);
  q.Push(12);

  ASSERT_EQ(10, q.Pull());
  ASSERT_EQ(12, q.Pull());
}

static constexpr size_t PRODUCER_THREAD_NUM = 30;
static constexpr size_t PRODUCE_TIMES = 1000;
static constexpr size_t CONSUMER_THREAD_NUM = 2;

TEST(queue, PushPullAsyncMultiThread) {
  using namespace paddle::framework;
  Queue<size_t> q;

  size_t counter[PRODUCER_THREAD_NUM] = {0};

  auto thread_main = [&q](size_t id) {
    for (size_t i = 0; i < PRODUCE_TIMES; ++i) {
      q.Push(id);
    }
  };

  for (size_t i = 0; i < PRODUCE_TIMES * PRODUCER_THREAD_NUM; ++i) {
    q.PullAsync([&counter](const size_t& val) { ++counter[val]; });
  }

  std::vector<std::thread> threads;
  for (size_t i = 0; i < PRODUCER_THREAD_NUM; ++i) {
    threads.emplace_back(std::bind(thread_main, i));
  }

  for (size_t i = 0; i < PRODUCER_THREAD_NUM; ++i) {
    threads[i].join();
  }

  for (size_t i = 0; i < PRODUCER_THREAD_NUM; ++i) {
    ASSERT_EQ(PRODUCE_TIMES, counter[i]);
  }
}

TEST(queue, PushPullSyncMultiThread) {
  using namespace paddle::framework;

  Queue<int> q;

  size_t counter[PRODUCER_THREAD_NUM] = {0};
  std::mutex counter_mu;

  auto consumer_main = [&q, &counter_mu, &counter] {
    while (true) {
      auto msg = q.Pull();
      // simulate pattern match. bool for quit.
      if (msg < 0) return;

      std::lock_guard<std::mutex> g(counter_mu);
      ++counter[msg];
    }
  };

  auto thread_main = [&q](size_t id) {
    for (size_t i = 0; i < PRODUCE_TIMES; ++i) {
      q.Push(id);
    }
  };

  std::vector<std::thread> consumers;
  std::vector<std::thread> producers;

  for (size_t i = 0; i < CONSUMER_THREAD_NUM; ++i) {
    consumers.emplace_back(consumer_main);
  }
  for (size_t i = 0; i < PRODUCER_THREAD_NUM; ++i) {
    producers.emplace_back(std::bind(thread_main, static_cast<int>(i)));
  }

  for (auto& t : producers) {
    t.join();
  }

  for (size_t i = 0; i < CONSUMER_THREAD_NUM; ++i) {
    q.Push(-1);
  }

  for (auto& t : consumers) {
    t.join();
  }

  for (auto& cnt : counter) {
    ASSERT_EQ(cnt, PRODUCE_TIMES);
  }
}