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

struct WorkQueueOptions {
  size_t num_threads{0};
  bool allow_spinning{true};
  bool track_task{false};
};

class WorkQueue {
 public:
  explicit WorkQueue(const WorkQueueOptions& options) : options_(options) {}

  WorkQueue(const WorkQueue&) = delete;

  WorkQueue& operator=(const WorkQueue&) = delete;

  virtual ~WorkQueue() = default;

  virtual void AddTask(std::function<void()> fn) = 0;

  // set WorkQueueOptions.track_task = true before call this
  // interface, otherwise will abort()
  virtual void WaitQueueEmpty() = 0;

  virtual size_t NumThreads() = 0;

 protected:
  WorkQueueOptions options_;
};

class WorkQueueGroup {
 public:
  explicit WorkQueueGroup(const std::vector<WorkQueueOptions>& queue_options);

  void AddTask(size_t queue_idx, std::function<void()> fn) {
    queues_[queue_idx]->AddTask(std::move(fn));
  }

  void WaitQueueGroupEmpty();

  size_t QueueNumThreads(size_t queue_idx) {
    return queues_[queue_idx]->NumThreads();
  }

  size_t GroupNumThreads();

 private:
  std::vector<WorkQueue*> queues_;  // owned by group
};

std::unique_ptr<WorkQueue> CreateSingleThreadedWorkQueue(
    const WorkQueueOptions& options);

std::unique_ptr<WorkQueue> CreateMultiThreadedWorkQueue(
    const WorkQueueOptions& options);

}  // namespace framework
}  // namespace paddle
