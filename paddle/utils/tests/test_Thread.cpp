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

#include <gtest/gtest.h>
#include <paddle/utils/Thread.h>
#include <atomic>

using paddle::AsyncThreadPool;  // NOLINT

TEST(AsyncThreadPool, addJob) {
  AsyncThreadPool pool(8);
  auto a = pool.addJob([] { return 1; });
  auto b = pool.addJob([] { return true; });
  auto c = pool.addJob([] { return false; });

  ASSERT_EQ(a.get(), 1);
  ASSERT_TRUE(b.get());
  ASSERT_FALSE(c.get());
}

TEST(AsyncThreadPool, addBatchJob) {
  AsyncThreadPool pool(8);
  std::atomic<int> counter{0};

  std::vector<AsyncThreadPool::JobFunc> jobs;

  for (int i = 0; i < 10000; i++) {
    jobs.emplace_back([&] { counter++; });
  }

  pool.addBatchJobs(jobs);

  ASSERT_EQ(counter, 10000);
}

TEST(AsyncThreadPool, multiThreadAddBatchJob) {
  AsyncThreadPool levelOnePool(200);
  AsyncThreadPool levelTwoPool(200);

  std::shared_ptr<std::mutex> mut = std::make_shared<std::mutex>();
  int counter = 0;
  const int numMonitors = 300;
  const int numSlaves = 300;
  std::vector<AsyncThreadPool::JobFunc> moniterJobs(numMonitors, [&] {
    std::vector<AsyncThreadPool::JobFunc> slaveJobs(numSlaves, [mut, &counter] {
      std::lock_guard<std::mutex> lk(*mut);
      counter++;
    });
    levelTwoPool.addBatchJobs(slaveJobs);
  });
  levelOnePool.addBatchJobs(moniterJobs);
  ASSERT_EQ(counter, numMonitors * numSlaves);
}

TEST(AsyncThreadPool, addBatchJobWithResults) {
  AsyncThreadPool pool(100);

  std::vector<std::function<int()>> jobs;
  const int numJobs = 100;
  for (int i = 0; i < numJobs; i++) {
    jobs.emplace_back([i] { return i; });
  }

  std::vector<int> res;
  pool.addBatchJobs(jobs, res);

  for (int i = 0; i < numJobs; i++) {
    ASSERT_EQ(res[i], i);
  }
}
