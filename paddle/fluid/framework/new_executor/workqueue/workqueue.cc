// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include "paddle/fluid/framework/new_executor/workqueue/workqueue.h"

#include "paddle/fluid/framework/new_executor/workqueue/nonblocking_threadpool.h"
#include "paddle/fluid/framework/new_executor/workqueue/workqueue_utils.h"
#include "paddle/fluid/platform/enforce.h"
#include "paddle/fluid/platform/profiler/event_tracing.h"

namespace paddle {
namespace framework {

void WorkQueueOptions::Validate() const {
  PADDLE_ENFORCE_GT(name.size(),
                    0,
                    platform::errors::InvalidArgument(
                        "WorkQueueOptions.name must be nonempty"));
  PADDLE_ENFORCE_EQ(
      name.find('_'),
      std::string::npos,
      platform::errors::InvalidArgument(
          "WorkQueueOptions.name shouldn't contain an underline"));
  PADDLE_ENFORCE_EQ(
      allow_spinning == false && always_spinning == true,
      false,
      platform::errors::InvalidArgument("WorkQueueOptions.allow_spinning must "
                                        "be true when always_spinning is set"));
}

namespace {

using TaskTracker = TaskTracker<EventsWaiter::EventNotifier>;

class WorkQueueImpl : public WorkQueue {
 public:
  explicit WorkQueueImpl(const WorkQueueOptions& options) : WorkQueue(options) {
    if (options_.track_task && options.events_waiter != nullptr) {
      empty_notifier_ = options.events_waiter->RegisterEvent(kQueueEmptyEvent);
      void* storage = AlignedMalloc(sizeof(TaskTracker), alignof(TaskTracker));
      tracker_ = new (storage) TaskTracker(*empty_notifier_.get());
    }
    if (options_.detached == false && options.events_waiter != nullptr) {
      destruct_notifier_ =
          options.events_waiter->RegisterEvent(kQueueDestructEvent);
    }
    queue_ = new NonblockingThreadPool(options_.name,
                                       options_.num_threads,
                                       options_.allow_spinning,
                                       options_.always_spinning);
  }

  virtual ~WorkQueueImpl() {
    delete queue_;
    if (tracker_ != nullptr) {
      tracker_->~TaskTracker();
      AlignedFree(tracker_);
    }
    if (destruct_notifier_) {
      destruct_notifier_->NotifyEvent();
    }
  }

  void AddTask(std::function<void()> fn) override {
    platform::RecordEvent record("WorkQueue::AddTask",
                                 platform::TracerEventType::UserDefined,
                                 10 /*level*/);
    if (tracker_ != nullptr) {
      fn = [task = std::move(fn),
            raii = CounterGuard<TaskTracker>(tracker_)]() mutable { task(); };
    }
    queue_->AddTask(std::move(fn));
  }

  void Cancel() override {
    queue_->Cancel();
    queue_->WaitThreadsExit();
  }

  size_t NumThreads() const override { return queue_->NumThreads(); }

 private:
  NonblockingThreadPool* queue_{nullptr};
  TaskTracker* tracker_{nullptr};
  std::shared_ptr<EventsWaiter::EventNotifier> empty_notifier_;
  std::shared_ptr<EventsWaiter::EventNotifier> destruct_notifier_;
};

class WorkQueueGroupImpl : public WorkQueueGroup {
 public:
  explicit WorkQueueGroupImpl(
      const std::vector<WorkQueueOptions>& queue_options);

  ~WorkQueueGroupImpl();

  void AddTask(size_t queue_idx, std::function<void()> fn) override;

  size_t QueueNumThreads(size_t queue_idx) const override;

  size_t QueueGroupNumThreads() const override;

  void Cancel() override;

 private:
  std::vector<NonblockingThreadPool*> queues_;
  NonblockingThreadPool* queues_storage_;
  TaskTracker* tracker_;
  std::shared_ptr<EventsWaiter::EventNotifier> empty_notifier_;
  std::shared_ptr<EventsWaiter::EventNotifier> destruct_notifier_;
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
    if (options.num_threads == 0) {
      queues_[idx] = nullptr;
      continue;
    }
    if (options.track_task && tracker_ == nullptr &&
        options.events_waiter != nullptr) {
      empty_notifier_ = options.events_waiter->RegisterEvent(kQueueEmptyEvent);
      void* storage = AlignedMalloc(sizeof(TaskTracker), alignof(TaskTracker));
      tracker_ = new (storage) TaskTracker(*empty_notifier_.get());
    }
    if (options.detached == false && options.events_waiter != nullptr &&
        !destruct_notifier_) {
      destruct_notifier_ =
          options.events_waiter->RegisterEvent(kQueueDestructEvent);
    }
    queues_[idx] = new (&queues_storage_[idx])
        NonblockingThreadPool(options.name,
                              options.num_threads,
                              options.allow_spinning,
                              options.always_spinning);
  }
}

WorkQueueGroupImpl::~WorkQueueGroupImpl() {
  for (auto queue : queues_) {
    if (queue) {
      queue->~NonblockingThreadPool();
    }
  }
  if (tracker_ != nullptr) {
    tracker_->~TaskTracker();
    AlignedFree(tracker_);
  }
  free(queues_storage_);
  if (destruct_notifier_) {
    destruct_notifier_->NotifyEvent();
  }
}

void WorkQueueGroupImpl::AddTask(size_t queue_idx, std::function<void()> fn) {
  platform::RecordEvent record("WorkQueue::AddTask",
                               platform::TracerEventType::UserDefined,
                               10 /*level*/);
  assert(queue_idx < queues_.size());
  PADDLE_ENFORCE_NOT_NULL(
      queues_.at(queue_idx),
      platform::errors::NotFound("Workqueue of index %d is not initialized.",
                                 queue_idx));
  if (queues_options_.at(queue_idx).track_task) {
    fn = [task = std::move(fn),
          raii = CounterGuard<TaskTracker>(tracker_)]() mutable { task(); };
  }
  queues_[queue_idx]->AddTask(std::move(fn));
}

size_t WorkQueueGroupImpl::QueueNumThreads(size_t queue_idx) const {
  assert(queue_idx < queues_.size());
  if (!queues_.at(queue_idx)) {
    return 0;
  }
  return queues_.at(queue_idx)->NumThreads();
}

size_t WorkQueueGroupImpl::QueueGroupNumThreads() const {
  size_t total_num = 0;
  for (auto queue : queues_) {
    total_num += queue->NumThreads();
  }
  return total_num;
}

void WorkQueueGroupImpl::Cancel() {
  for (auto queue : queues_) {
    if (queue) {
      queue->Cancel();
    }
  }
  for (auto queue : queues_) {
    if (queue) {
      queue->WaitThreadsExit();
    }
  }
}

}  // namespace

std::unique_ptr<WorkQueue> CreateSingleThreadedWorkQueue(
    const WorkQueueOptions& options) {
  options.Validate();
  // extra check
  PADDLE_ENFORCE_EQ(options.num_threads,
                    1u,
                    platform::errors::InvalidArgument(
                        "For a SingleThreadedWorkQueue, "
                        "WorkQueueOptions.num_threads must equals to 1."));
  std::unique_ptr<WorkQueue> ptr(new WorkQueueImpl(options));
  return ptr;
}

std::unique_ptr<WorkQueue> CreateMultiThreadedWorkQueue(
    const WorkQueueOptions& options) {
  options.Validate();
  // extra check
  PADDLE_ENFORCE_GT(
      options.num_threads,
      1u,
      platform::errors::InvalidArgument("For a MultiThreadedWorkQueue, "
                                        "WorkQueueOptions.num_threads must be "
                                        "greater than 1."));
  std::unique_ptr<WorkQueue> ptr(new WorkQueueImpl(options));
  return ptr;
}

std::unique_ptr<WorkQueueGroup> CreateWorkQueueGroup(
    const std::vector<WorkQueueOptions>& queues_options) {
  PADDLE_ENFORCE_GT(queues_options.size(),
                    1u,
                    platform::errors::InvalidArgument(
                        "For a WorkQueueGroup, the number of WorkQueueOptions "
                        "must be greater than 1."));
  for (const auto& opts : queues_options) {
    opts.Validate();
  }
  std::unique_ptr<WorkQueueGroup> ptr(new WorkQueueGroupImpl(queues_options));
  return ptr;
}

}  // namespace framework
}  // namespace paddle
