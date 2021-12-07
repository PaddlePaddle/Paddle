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
  using paddle::framework::WorkQueueOptions;
  using paddle::framework::WorkQueue;
  using paddle::framework::CreateSingleThreadedWorkQueue;
  std::atomic<bool> finished{false};
  std::atomic<unsigned> counter{0};
  constexpr unsigned kLoopNum = 1000000;
  // CreateSingleThreadedWorkQueue
  WorkQueueOptions options(/*num_threads*/ 1, /*allow_spinning*/ true,
                           /*track_task*/ true);
  auto work_queue = CreateSingleThreadedWorkQueue(options);
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
  using paddle::framework::WorkQueueOptions;
  using paddle::framework::WorkQueue;
  using paddle::framework::CreateMultiThreadedWorkQueue;
  std::atomic<bool> finished{false};
  std::atomic<unsigned> counter{0};
  constexpr unsigned kExternalLoopNum = 100;
  constexpr unsigned kLoopNum = 1000000;
  // CreateMultiThreadedWorkQueue
  WorkQueueOptions options(/*num_threads*/ 10, /*allow_spinning*/ true,
                           /*track_task*/ true);
  auto work_queue = CreateMultiThreadedWorkQueue(options);
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

TEST(WorkQueue, TestWorkQueueGroup) {
  using paddle::framework::WorkQueueOptions;
  using paddle::framework::WorkQueueGroup;
  using paddle::framework::CreateWorkQueueGroup;
  std::atomic<bool> finished{false};
  std::atomic<unsigned> counter{0};
  constexpr unsigned kExternalLoopNum = 100;
  constexpr unsigned kLoopNum = 1000000;
  // CreateMultiThreadedWorkQueue
  WorkQueueOptions sq_options(/*num_threads*/ 1, /*allow_spinning*/ true,
                              /*track_task*/ true);
  WorkQueueOptions mq_options(/*num_threads*/ 10, /*allow_spinning*/ true,
                              /*track_task*/ true);
  auto queue_group = CreateWorkQueueGroup({sq_options, mq_options});
  // NumThreads
  EXPECT_EQ(queue_group->QueueNumThreads(0), 1u);
  EXPECT_EQ(queue_group->QueueNumThreads(1), 10u);
  EXPECT_EQ(queue_group->QueueGroupNumThreads(), 11u);
  // AddTask
  EXPECT_EQ(counter.load(), 0u);
  for (unsigned i = 0; i < kExternalLoopNum; ++i) {
    queue_group->AddTask(1, [&counter, &finished, kLoopNum]() {
      for (unsigned i = 0; i < kLoopNum; ++i) {
        ++counter;
      }
    });
  }
  queue_group->AddTask(0, [&counter, &finished, kLoopNum]() {
    for (unsigned i = 0; i < kLoopNum; ++i) {
      ++counter;
    }
  });
  //  WaitQueueGroupEmpty()
  queue_group->WaitQueueGroupEmpty();
  EXPECT_EQ(counter.load(), kLoopNum * kExternalLoopNum + kLoopNum);
}
