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
#include <set>
#include <vector>
#include "paddle/fluid/framework/new_executor/event_count.h"
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
    PADDLE_THROW(platform::errors::Unavailable(
        "Never use the copy constructor of CounterGuard."));
  }

  CounterGuard& operator=(const CounterGuard&) = delete;

 private:
  Holder* counter_holder_{nullptr};
};

void* AlignedMalloc(size_t size, size_t alignment);

void AlignedFree(void* memory_ptr);

// A multiplexing waiter, be able to wait multi events simultaneously
// Blocking the calling thread to wait any of the registered events
class EventsWaiter {
 public:
  using EventId = int64_t;

  using EventChecker = std::function<bool()>;

  class Notifier {
   public:
    void NotifyEvent();

   private:
    friend EventsWaiter;
    Notifier(EventId id, EventsWaiter& waiter) : id_(id), waiter_(waiter) {}

    EventId id_;
    EventsWaiter& waiter_;
  };

  EventsWaiter();

  EventsWaiter(const EventsWaiter&) = delete;

  EventsWaiter& operator=(const EventsWaiter&) = delete;

  // RegisterEvent must be called before WaitEvent
  EventId RegisterEvent(EventChecker checker);

  std::reference_wrapper<Notifier> GetEventNotifier(const EventId& id);

  // Wait any of the registered events
  EventId WaitEvent();

  const std::set<EventId> WaitEvents();

 private:
  friend Notifier;
  void SetTriggerEvent(const EventId& id);

  std::vector<EventChecker> checkers_;
  std::vector<Notifier> notifiers_;
  std::atomic<EventId> trigger_event_;
  std::atomic<bool> waiting_;
  EventCount cv_;
};

}  // namespace framework
}  // namespace paddle
