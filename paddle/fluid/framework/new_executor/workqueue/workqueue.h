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
#include <future>
#include <memory>
#include <string>
#include <type_traits>
#include <vector>

#include "paddle/fluid/platform/enforce.h"

namespace paddle {
namespace framework {

constexpr const char* kQueueEmptyEvent = "QueueEmpty";
constexpr const char* kQueueDestructEvent = "QueueDestruct";

// For std::function
// https://stackoverflow.com/questions/25421346/how-to-create-an-stdfunction-from-a-move-capturing-lambda-expression
template <typename OnlyMovable>
class FakeCopyable {
 public:
  explicit FakeCopyable(OnlyMovable&& obj) : obj_(std::move(obj)) {
    static_assert(std::is_copy_constructible<OnlyMovable>::value == false,
                  "Need not to use FakeCopyable");
  }

  FakeCopyable(FakeCopyable&& other) : obj_(std::move(other.obj_)) {}

  FakeCopyable(const FakeCopyable& other) {
    PADDLE_THROW(common::errors::Unavailable(
        "Never use the copy constructor of FakeCopyable."));
  }

  OnlyMovable& Get() { return obj_; }

 private:
  OnlyMovable obj_;
};

class EventsWaiter;

struct WorkQueueOptions {
  WorkQueueOptions(const std::string& name,
                   size_t num_threads,
                   bool allow_spinning,
                   bool track_task)
      : name(name),
        num_threads(num_threads),
        allow_spinning(allow_spinning),
        track_task(track_task) {
    Validate();
  }

  WorkQueueOptions(const std::string& name,
                   size_t num_threads,
                   bool allow_spinning,
                   bool always_spinning,
                   bool track_task,
                   bool detached,
                   EventsWaiter* waiter)
      : name(name),
        num_threads(num_threads),
        allow_spinning(allow_spinning),
        always_spinning(always_spinning),
        track_task(track_task),
        detached(detached),
        events_waiter(waiter) {
    Validate();
  }

  // throw an exception if there is an invalid option
  void Validate() const;

  std::string name;
  size_t num_threads;
  // Worker threads will spin for a while if this flag is set.
  bool allow_spinning;
  // Worker threads will never sleep if this flag is set.
  // Better performance vs. higher CPU utilization.
  bool always_spinning{false};
  // If you need to blocking the calling  thread to wait "queue empty", set
  // track_task = true and set events_waiter. EventsWaiter::WaitEvent will
  // block the calling thread until any of events (including "queue empty")
  // occurred.
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

  // Higher cost than AddTask
  template <typename F, typename... Args>
  std::future<typename std::result_of<F(Args...)>::type> AddAwaitableTask(
      F&& f, Args&&... args) {
    using ReturnType = typename std::result_of<F(Args...)>::type;
    std::function<ReturnType()> task =
        std::bind(std::forward<F>(f), std::forward<Args>(args)...);
    std::promise<ReturnType> prom;
    std::future<ReturnType> res = prom.get_future();
    AddTask([t = std::move(task),
             p = FakeCopyable<std::promise<ReturnType>>(
                 std::move(prom))]() mutable { p.Get().set_value(t()); });
    return res;
  }

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

  // Higher cost than AddTask
  template <typename F, typename... Args>
  std::future<typename std::result_of<F(Args...)>::type> AddAwaitableTask(
      size_t queue_idx, F&& f, Args&&... args) {
    using ReturnType = typename std::result_of<F(Args...)>::type;
    std::function<ReturnType()> task =
        std::bind(std::forward<F>(f), std::forward<Args>(args)...);
    std::promise<ReturnType> prom;
    std::future<ReturnType> res = prom.get_future();
    AddTask(queue_idx,
            [t = std::move(task),
             p = FakeCopyable<std::promise<ReturnType>>(
                 std::move(prom))]() mutable { p.Get().set_value(t()); });
    return res;
  }

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
