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

#pragma once

#include <functional>
#include <memory>
#include <vector>

namespace paddle {
namespace framework {

constexpr const char* kQueueEmptyEvent = "QueueEmpty";
constexpr const char* kQueueDestructEvent = "QueueDestruct";

class EventsWaiter;

struct WorkQueueOptions {
  WorkQueueOptions(size_t num_threads, bool allow_spinning, bool track_task)
      : num_threads(num_threads),
        allow_spinning(allow_spinning),
        track_task(track_task) {}

  WorkQueueOptions(size_t num_threads, bool allow_spinning, bool track_task,
                   bool detached, EventsWaiter* waiter)
      : num_threads(num_threads),
        allow_spinning(allow_spinning),
        track_task(track_task),
        detached(detached),
        events_waiter(waiter) {}

  size_t num_threads;
  bool allow_spinning;
  // If you need to blocking the calling  thread to wait "queue empty", set
  // track_task = true and set events_waiter. EventsWaiter::WaitEvent will
  // block the calling thread until any of events (including "queue empty")
  // occured.
  bool track_task;
  // If you need to be noticed when a WorkQueue Destruct() , set detached =
  // false and set events_waiter.
  bool detached{true};
  EventsWaiter* events_waiter{nullptr};  // not owned
};

class WorkQueue {
 public:
  explicit WorkQueue(const WorkQueueOptions& options) : options_(options) {}

  WorkQueue(const WorkQueue&) = delete;

  WorkQueue& operator=(const WorkQueue&) = delete;

  virtual ~WorkQueue() = default;

  virtual void AddTask(std::function<void()> fn) = 0;

  // See WorkQueueOptions.track_task for details
  // virtual void WaitQueueEmpty() = 0;

  virtual size_t NumThreads() const = 0;

  virtual void Cancel() = 0;

 protected:
  WorkQueueOptions options_;
};

class WorkQueueGroup {
 public:
  explicit WorkQueueGroup(const std::vector<WorkQueueOptions>& queues_options)
      : queues_options_(queues_options) {}

  WorkQueueGroup(const WorkQueueGroup&) = delete;

  WorkQueueGroup& operator=(const WorkQueueGroup&) = delete;

  virtual ~WorkQueueGroup() = default;

  virtual void AddTask(size_t queue_idx, std::function<void()> fn) = 0;

  // See WorkQueueOptions.track_task for details
  // virtual void WaitQueueGroupEmpty() = 0;

  virtual size_t QueueNumThreads(size_t queue_idx) const = 0;

  virtual size_t QueueGroupNumThreads() const = 0;

  virtual void Cancel() = 0;

 protected:
  std::vector<WorkQueueOptions> queues_options_;
};

std::unique_ptr<WorkQueue> CreateSingleThreadedWorkQueue(
    const WorkQueueOptions& options);

std::unique_ptr<WorkQueue> CreateMultiThreadedWorkQueue(
    const WorkQueueOptions& options);

std::unique_ptr<WorkQueueGroup> CreateWorkQueueGroup(
    const std::vector<WorkQueueOptions>& queues_options);

}  // namespace framework
}  // namespace paddle
