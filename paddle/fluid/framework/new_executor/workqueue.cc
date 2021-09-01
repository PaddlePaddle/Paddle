// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include "paddle/fluid/framework/new_executor/workqueue.h"
#include "paddle/fluid/framework/new_executor/nonblocking_threadpool.h"

namespace paddle {
namespace framework {

class SingleThreadedWorkQueue : public WorkQueue {
 public:
  SingleThreadedWorkQueue() : queue_(1) {}

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

std::unique_ptr<WorkQueue> CreateSingleThreadedWorkQueue() {
  std::unique_ptr<WorkQueue> ptr(new SingleThreadedWorkQueue);
  return std::move(ptr);
}

class MultiThreadedWorkQueue : public WorkQueue {
 public:
  explicit MultiThreadedWorkQueue(int num_threads) : queue_(num_threads) {
    assert(num_threads > 1);
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

std::unique_ptr<WorkQueue> CreateMultiThreadedWorkQueue(int num_threads) {
  std::unique_ptr<WorkQueue> ptr(new MultiThreadedWorkQueue(num_threads));
  return std::move(ptr);
}

}  // namespace framework
}  // namespace paddle
