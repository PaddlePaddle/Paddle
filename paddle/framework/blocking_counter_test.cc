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

#include "blocking_counter.h"
#include <gtest/gtest.h>
#include <atomic>
#include <thread>
#include "threadpool.h"

namespace framework = paddle::framework;

TEST(BlockingCounter, SingleThread) {
  framework::BlockingCounter bc(2);
  bc.DecreaseCount();
  bc.DecreaseCount();
  bc.Wait();
}

TEST(BlockingCounter, MultipleThread) {
  framework::ThreadPool* pool = framework::ThreadPool::GetInstance();
  std::vector<int> cnt_per_task = {10, 2000, 30, 40, 50};
  int task_cnt = cnt_per_task.size();
  std::vector<framework::BlockingCounter*> bcs(task_cnt);
  std::vector<std::atomic<int>> sums(task_cnt);
  for (int i = 0; i < task_cnt; ++i) {
    int cnt = cnt_per_task[i];
    auto& sum = sums[i];
    sum = 0;
    auto& bc = bcs[i];
    bc = new framework::BlockingCounter(cnt);
    pool->Run([&sum, bc, cnt]() {
      for (int j = 0; j < cnt; ++j) {
        sum.fetch_add(1);
        bc->DecreaseCount();
      }
    });
    bcs.push_back(std::move(bc));
  }

  for (int i = 0; i < task_cnt; ++i) {
    bcs[i]->Wait();
  }
}