// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include "paddle/fluid/framework/new_executor/workqueue.h"
#include "paddle/fluid/framework/new_executor/nonblocking_threadpool.h"
#include "paddle/fluid/framework/new_executor/workqueue_utils.h"
#include "paddle/fluid/platform/enforce.h"

namespace paddle {
namespace framework {
namespace {

class WorkQueueImpl : public WorkQueue {
 public:
  explicit WorkQueueImpl(const WorkQueueOptions& options)
      : WorkQueue(options), queue_(nullptr), tracker_(nullptr) {
    if (options_.track_task) {
      void* storage = AlignedMalloc(sizeof(TaskTracker), alignof(TaskTracker));
      tracker_ = new (storage) TaskTracker;
    }
    queue_ = new NonblockingThreadPool(options_.num_threads,
                                       options_.allow_spinning);
  }

  virtual ~WorkQueueImpl() {
    if (tracker_ != nullptr) {
      tracker_->~TaskTracker();
      AlignedFree(tracker_);
    }
    delete queue_;
  }

  void AddTask(std::function<void()> fn) override {
    if (tracker_ != nullptr) {
      fn = [
        task = std::move(fn), raii = CounterGuard<TaskTracker>(tracker_)
      ]() mutable {
        task();
      };
    }
    queue_->AddTask(std::move(fn));
  }

  void WaitQueueEmpty() override {
    if (tracker_ == nullptr) {
      PADDLE_THROW(
          platform::errors::Unavailable("set WorkQueueOptions.track_task = "
                                        "true before call this interface."));
    }
    tracker_->WaitTaskNumToZero();
  }

  size_t NumThreads() const override { return queue_->NumThreads(); }

 private:
  NonblockingThreadPool* queue_;
  TaskTracker* tracker_;
};

class WorkQueueGroupImpl : public WorkQueueGroup {
 public:
  explicit WorkQueueGroupImpl(
      const std::vector<WorkQueueOptions>& queue_options);

  ~WorkQueueGroupImpl();

  void AddTask(size_t queue_idx, std::function<void()> fn) override;

  void WaitQueueGroupEmpty() override;

  size_t QueueNumThreads(size_t queue_idx) const override;

  size_t QueueGroupNumThreads() const override;

 private:
  std::vector<NonblockingThreadPool*> queues_;
  NonblockingThreadPool* queues_storage_;
  TaskTracker* tracker_;
};

WorkQueueGroupImpl::WorkQueueGroupImpl(
    const std::vector<WorkQueueOptions>& queues_options)
    : WorkQueueGroup(queues_options),
      queues_storage_(nullptr),
      tracker_(nullptr) {
  size_t num_queues = queues_options_.size();
  queues_.resize(num_queues);
  void* buffer = malloc(sizeof(NonblockingThreadPool) * num_queues);
  queues_storage_ = reinterpret_cast<NonblockingThreadPool*>(buffer);
  for (size_t idx = 0; idx < num_queues; ++idx) {
    const auto& options = queues_options_[idx];
    if (options.track_task && tracker_ == nullptr) {
      void* storage = AlignedMalloc(sizeof(TaskTracker), alignof(TaskTracker));
      tracker_ = new (storage) TaskTracker;
    }
    queues_[idx] = new (&queues_storage_[idx])
        NonblockingThreadPool(options.num_threads, options.allow_spinning);
  }
}

WorkQueueGroupImpl::~WorkQueueGroupImpl() {
  for (auto queue : queues_) {
    queue->~NonblockingThreadPool();
  }
  if (tracker_ != nullptr) {
    tracker_->~TaskTracker();
    AlignedFree(tracker_);
  }
  free(queues_storage_);
}

void WorkQueueGroupImpl::AddTask(size_t queue_idx, std::function<void()> fn) {
  assert(queue_idx < queues_.size());
  if (queues_options_.at(queue_idx).track_task) {
    fn = [
      task = std::move(fn), raii = CounterGuard<TaskTracker>(tracker_)
    ]() mutable {
      task();
    };
  }
  queues_[queue_idx]->AddTask(std::move(fn));
}

void WorkQueueGroupImpl::WaitQueueGroupEmpty() {
  if (nullptr == tracker_) {
    PADDLE_THROW(platform::errors::Unavailable(
        "set WorkQueueOptions.track_task = true for at least one of queues "
        "before call this interface."));
  }
  tracker_->WaitTaskNumToZero();
}

size_t WorkQueueGroupImpl::QueueNumThreads(size_t queue_idx) const {
  assert(queue_idx < queues_.size());
  return queues_.at(queue_idx)->NumThreads();
}

size_t WorkQueueGroupImpl::QueueGroupNumThreads() const {
  size_t total_num = 0;
  for (auto queue : queues_) {
    total_num += queue->NumThreads();
  }
  return total_num;
}

}  // namespace

std::unique_ptr<WorkQueue> CreateSingleThreadedWorkQueue(
    const WorkQueueOptions& options) {
  PADDLE_ENFORCE_EQ(options.num_threads, 1u,
                    platform::errors::InvalidArgument(
                        "For a SingleThreadedWorkQueue, "
                        "WorkQueueOptions.num_threads must equals to 1."));
  std::unique_ptr<WorkQueue> ptr(new WorkQueueImpl(options));
  return ptr;
}

std::unique_ptr<WorkQueue> CreateMultiThreadedWorkQueue(
    const WorkQueueOptions& options) {
  PADDLE_ENFORCE_GT(
      options.num_threads, 1u,
      platform::errors::InvalidArgument("For a MultiThreadedWorkQueue, "
                                        "WorkQueueOptions.num_threads must be "
                                        "greater than 1."));
  std::unique_ptr<WorkQueue> ptr(new WorkQueueImpl(options));
  return std::move(ptr);
}

std::unique_ptr<WorkQueueGroup> CreateWorkQueueGroup(
    const std::vector<WorkQueueOptions>& queues_options) {
  PADDLE_ENFORCE_GT(queues_options.size(), 1u,
                    platform::errors::InvalidArgument(
                        "For a WorkQueueGroup, the number of WorkQueueOptions "
                        "must be greater than 1."));
  std::unique_ptr<WorkQueueGroup> ptr(new WorkQueueGroupImpl(queues_options));
  return std::move(ptr);
}

}  // namespace framework
}  // namespace paddle
