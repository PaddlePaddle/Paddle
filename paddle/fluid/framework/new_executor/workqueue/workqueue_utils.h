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

#include <atomic>
#include <cassert>
#include <cstddef>
#include <cstdlib>
#include <memory>
#include <set>
#include <string>

#include "paddle/fluid/framework/new_executor/workqueue/events_waiter.h"
#include "paddle/fluid/platform/enforce.h"

namespace paddle {
namespace framework {

template <typename Holder>
class CounterGuard {
 public:
  explicit CounterGuard(Holder* holder) : counter_holder_(holder) {
    assert(holder != nullptr);
    counter_holder_->AddCounter();
  }

  ~CounterGuard() {
    if (counter_holder_ != nullptr) {
      counter_holder_->SubCounter();
    }
  }

  CounterGuard(CounterGuard&& other) : counter_holder_(other.counter_holder_) {
    other.counter_holder_ = nullptr;
  }

  CounterGuard& operator=(CounterGuard&& other) {
    counter_holder_ = other.counter_holder_;
    other.counter_holder_ = nullptr;
    return *this;
  }

  // copy constructor deleted, we define this for std::function
  // never use it directly
  CounterGuard(const CounterGuard& other) {
    PADDLE_THROW(common::errors::Unavailable(
        "Never use the copy constructor of CounterGuard."));
  }

  CounterGuard& operator=(const CounterGuard&) = delete;

 private:
  Holder* counter_holder_{nullptr};
};

void* AlignedMalloc(size_t size, size_t alignment);

void AlignedFree(void* memory_ptr);

template <typename Notifier>
class TaskTracker {
 public:
  TaskTracker() = default;

  explicit TaskTracker(Notifier& notifier) : notifier_(&notifier) {}

  TaskTracker(const TaskTracker&) = delete;

  TaskTracker& operator=(const TaskTracker&) = delete;

  ~TaskTracker() = default;

  void AddCounter() {
    if (0 == num_tasks_.fetch_add(1, std::memory_order_relaxed)) {
      if (notifier_ != nullptr) {
        notifier_->CancelEvent();
      }
    }
  }

  void SubCounter() {
    if (1 == num_tasks_.fetch_sub(1, std::memory_order_relaxed)) {
      if (notifier_ != nullptr) {
        notifier_->NotifyEvent();
      }
    }
  }

  uint64_t PendingTaskNum() { return num_tasks_.load(); }

 private:
  alignas(64) std::atomic<uint64_t> num_tasks_{0};
  Notifier* notifier_{nullptr};
};

}  // namespace framework
}  // namespace paddle
