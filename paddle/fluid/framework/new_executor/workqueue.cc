// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include "paddle/fluid/framework/new_executor/workqueue.h"
#include "paddle/fluid/framework/new_executor/nonblocking_threadpool.h"

namespace paddle {
namespace framework {
namespace {

class SingleThreadedWorkQueue : public WorkQueue {
 public:
  explicit SingleThreadedWorkQueue(const WorkQueueOptions& options,
                                   TaskTracker* tracker = nullptr)
      : WorkQueue(options),
        queue_(1, options.allow_spinning, options.track_task, tracker) {
    assert(options.num_threads == 1);
    options_.num_threads = 1;
  }

  SingleThreadedWorkQueue(const SingleThreadedWorkQueue&) = delete;

  SingleThreadedWorkQueue& operator=(const SingleThreadedWorkQueue&) = delete;

  virtual ~SingleThreadedWorkQueue() = default;

  void AddTask(std::function<void()> fn) override {
    queue_.AddTask(std::move(fn));
  }

  void WaitQueueEmpty() override { queue_.WaitQueueEmpty(); }

  size_t NumThreads() override { return queue_.NumThreads(); }

 private:
  NonblockingThreadPool queue_;
};

class MultiThreadedWorkQueue : public WorkQueue {
 public:
  explicit MultiThreadedWorkQueue(const WorkQueueOptions& options,
                                  TaskTracker* tracker = nullptr)
      : WorkQueue(options),
        queue_(options.num_threads, options.allow_spinning, options.track_task,
               tracker) {
    assert(options.num_threads > 1);
  }

  MultiThreadedWorkQueue(const MultiThreadedWorkQueue&) = delete;

  MultiThreadedWorkQueue& operator=(const MultiThreadedWorkQueue&) = delete;

  virtual ~MultiThreadedWorkQueue() = default;

  void AddTask(std::function<void()> fn) override {
    queue_.AddTask(std::move(fn));
  }

  void WaitQueueEmpty() override { queue_.WaitQueueEmpty(); }

  size_t NumThreads() override { return queue_.NumThreads(); }

 private:
  NonblockingThreadPool queue_;
};

}  // namespace

std::unique_ptr<WorkQueue> CreateSingleThreadedWorkQueue(
    const WorkQueueOptions& options) {
  std::unique_ptr<WorkQueue> ptr(new SingleThreadedWorkQueue(options));
  return std::move(ptr);
}

std::unique_ptr<WorkQueue> CreateMultiThreadedWorkQueue(
    const WorkQueueOptions& options) {
  std::unique_ptr<WorkQueue> ptr(new MultiThreadedWorkQueue(options));
  return std::move(ptr);
}

}  // namespace framework
}  // namespace paddle
