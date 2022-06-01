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

#include "paddle/fluid/framework/new_executor/workqueue/workqueue.h"
#include <atomic>
#include <thread>
#include "glog/logging.h"
#include "gtest/gtest.h"
#include "paddle/fluid/framework/new_executor/workqueue/workqueue_utils.h"

TEST(WorkQueueUtils, TestEventsWaiter) {
  using paddle::framework::EventsWaiter;
  EventsWaiter events_waiter;
  auto notifier =
      events_waiter.RegisterEvent("test_register_lt", []() { return true; });
  EXPECT_EQ(events_waiter.WaitEvent(), "test_register_lt");
  EXPECT_EQ(notifier->GetEventName(), "test_register_lt");
  EXPECT_EQ(events_waiter.WaitEvent(), "test_register_lt");
  notifier.reset();
  notifier = events_waiter.RegisterEvent("test_register_et");
  notifier->NotifyEvent();
  EXPECT_EQ(events_waiter.WaitEvent(), "test_register_et");
  notifier->NotifyEvent();
  notifier->CancelEvent();
}

TEST(WorkQueue, TestSingleThreadedWorkQueue) {
  VLOG(1) << "In Test";
  using paddle::framework::WorkQueueOptions;
  using paddle::framework::WorkQueue;
  using paddle::framework::CreateSingleThreadedWorkQueue;
  using paddle::framework::EventsWaiter;
  std::atomic<bool> finished{false};
  std::atomic<unsigned> counter{0};
  constexpr unsigned kLoopNum = 1000000;
  // CreateSingleThreadedWorkQueue
  EventsWaiter events_waiter;
  WorkQueueOptions options(/*name*/ "SingleThreadedWorkQueueForTesting",
                           /*num_threads*/ 1, /*allow_spinning*/ true,
                           /*always_spinning*/ true,
                           /*track_task*/ true, /*detached*/ true,
                           &events_waiter);
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
  auto handle = work_queue->AddAwaitableTask([]() { return 1234; });
  // WaitQueueEmpty
  EXPECT_EQ(finished.load(), false);
  events_waiter.WaitEvent();
  EXPECT_EQ(finished.load(), true);
  EXPECT_EQ(counter.load(), kLoopNum);
  EXPECT_EQ(handle.get(), 1234);
  work_queue.reset();
  // Test default_options with no spinning
  WorkQueueOptions default_options("SingleThreadedWorkQueueForTesting",
                                   /*num_threads*/ 1,
                                   /*allow_spinning*/ false,
                                   /*track_task*/ false);
  work_queue = CreateSingleThreadedWorkQueue(default_options);
  handle = work_queue->AddAwaitableTask([]() { return 5678; });
  EXPECT_EQ(handle.get(), 5678);
}

TEST(WorkQueue, TestMultiThreadedWorkQueue) {
  VLOG(1) << "In Test";
  using paddle::framework::WorkQueueOptions;
  using paddle::framework::WorkQueue;
  using paddle::framework::CreateMultiThreadedWorkQueue;
  using paddle::framework::EventsWaiter;
  std::atomic<bool> finished{false};
  std::atomic<unsigned> counter{0};
  constexpr unsigned kExternalLoopNum = 100;
  constexpr unsigned kLoopNum = 1000000;
  // CreateMultiThreadedWorkQueue
  EventsWaiter events_waiter;
  WorkQueueOptions options(/*name*/ "MultiThreadedWorkQueueForTesting",
                           /*num_threads*/ 10, /*allow_spinning*/ true,
                           /*always_spinning*/ true,
                           /*track_task*/ true, /*detached*/ false,
                           &events_waiter);
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
  EXPECT_EQ(events_waiter.WaitEvent(), paddle::framework::kQueueEmptyEvent);
  EXPECT_EQ(finished.load(), true);
  EXPECT_EQ(counter.load(), kLoopNum * kExternalLoopNum);
  // Cancel
  work_queue->Cancel();
  // Wait kQueueDestructEvent
  std::thread waiter_thread([&events_waiter]() {
    EXPECT_EQ(events_waiter.WaitEvent(),
              paddle::framework::kQueueDestructEvent);
  });
  work_queue.reset();
  waiter_thread.join();
  // Forever spin unittest
  WorkQueueOptions default_options("MultiThreadedWorkQueueForTesting",
                                   /*num_threads*/ 10, /*allow_spinning*/ false,
                                   /*track_task*/ false);
  work_queue = CreateMultiThreadedWorkQueue(default_options);
  auto handle = work_queue->AddAwaitableTask([]() { return 5678; });
  EXPECT_EQ(handle.get(), 5678);
}

TEST(WorkQueue, TestWorkQueueGroup) {
  using paddle::framework::WorkQueueOptions;
  using paddle::framework::WorkQueueGroup;
  using paddle::framework::CreateWorkQueueGroup;
  using paddle::framework::EventsWaiter;
  std::atomic<bool> finished{false};
  std::atomic<unsigned> counter{0};
  constexpr unsigned kExternalLoopNum = 100;
  constexpr unsigned kLoopNum = 1000000;
  // ThreadedWorkQueueGroup
  EventsWaiter events_waiter;
  WorkQueueOptions sq_options(/*name*/ "SingleThreadedWorkQueueForTesting",
                              /*num_threads*/ 1, /*allow_spinning*/ true,
                              /*always_spinning*/ true,
                              /*track_task*/ true, /*detached*/ false,
                              &events_waiter);
  WorkQueueOptions mq_options(/*name*/ "MultiThreadedWorkQueueForTesting",
                              /*num_threads*/ 10, /*allow_spinning*/ true,
                              /*always_spinning*/ true,
                              /*track_task*/ true, /*detached*/ false,
                              &events_waiter);
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
  int random_num = 123456;
  auto handle =
      queue_group->AddAwaitableTask(1, [random_num]() { return random_num; });
  // WaitQueueGroupEmpty
  events_waiter.WaitEvent();
  EXPECT_EQ(counter.load(), kLoopNum * kExternalLoopNum + kLoopNum);
  EXPECT_EQ(handle.get(), random_num);
  // Cancel
  queue_group->Cancel();
  // Wait kQueueDestructEvent
  std::thread waiter_thread([&events_waiter]() {
    EXPECT_EQ(events_waiter.WaitEvent(),
              paddle::framework::kQueueDestructEvent);
    EXPECT_EQ(events_waiter.WaitEvent(), "NoEventNotifier");
  });
  queue_group.reset();
  waiter_thread.join();
}
