// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/framework/new_executor/workqueue.h"
#include <atomic>
#include "glog/logging.h"
#include "gtest/gtest.h"

TEST(WorkQueue, TestSingleThreadedWorkQueue) {
  VLOG(1) << "In Test";
  using paddle::framework::WorkQueue;
  using paddle::framework::CreateSingleThreadedWorkQueue;
  std::atomic<bool> finished{false};
  std::atomic<unsigned> counter{0};
  constexpr unsigned kLoopNum = 1000000;
  // CreateSingleThreadedWorkQueue
  std::unique_ptr<WorkQueue> work_queue = CreateSingleThreadedWorkQueue();
  // NumThreads
  EXPECT_EQ(work_queue->NumThreads(), 1u);
  // AddTask
  EXPECT_EQ(finished.load(), false);
  EXPECT_EQ(counter.load(), 0u);
  work_queue->AddTask([&counter, &finished, kLoopNum]() {
    for (unsigned i = 0; i < kLoopNum; ++i) {
      ++counter;
    }
    finished = true;
  });
  // WaitQueueEmpty
  EXPECT_EQ(finished.load(), false);
  work_queue->WaitQueueEmpty();
  EXPECT_EQ(finished.load(), true);
  EXPECT_EQ(counter.load(), kLoopNum);
}

TEST(WorkQueue, TestMultiThreadedWorkQueue) {
  VLOG(1) << "In Test";
  using paddle::framework::WorkQueue;
  using paddle::framework::CreateMultiThreadedWorkQueue;
  std::atomic<bool> finished{false};
  std::atomic<unsigned> counter{0};
  constexpr unsigned kExternalLoopNum = 100;
  constexpr unsigned kLoopNum = 1000000;
  // CreateSingleThreadedWorkQueue
  std::unique_ptr<WorkQueue> work_queue = CreateMultiThreadedWorkQueue(10);
  // NumThreads
  EXPECT_EQ(work_queue->NumThreads(), 10u);
  // AddTask
  EXPECT_EQ(finished.load(), false);
  EXPECT_EQ(counter.load(), 0u);
  for (unsigned i = 0; i < kExternalLoopNum; ++i) {
    work_queue->AddTask([&counter, &finished, kLoopNum]() {
      for (unsigned i = 0; i < kLoopNum; ++i) {
        ++counter;
      }
      finished = true;
    });
  }
  // WaitQueueEmpty
  EXPECT_EQ(finished.load(), false);
  work_queue->WaitQueueEmpty();
  EXPECT_EQ(finished.load(), true);
  EXPECT_EQ(counter.load(), kLoopNum * kExternalLoopNum);
}
