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
#include <vector>
#include "paddle/fluid/framework/new_executor/event_count.h"
#include "paddle/fluid/memory/allocation/spin_lock.h"
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

// A multiplexing waiter, be able to wait multi events simultaneously.
// Blocking the calling thread to wait any of the registered events.
class EventsWaiter {
 public:
  using EventId = std::size_t;

  using EventChecker = std::function<bool()>;

  // Make sure EventsWaiter has a longer lifetime than EventNotifier.
  class EventNotifier {
   public:
    void NotifyEvent() { waiter_.TriggerEvent(id_); }

    void UnregisterEvent() { waiter_.UnregisterEvent(id_); }

    EventId GetEventId() { return id_; }

    // return "Unregistered" if the corresponding event was unregistered.
    std::string GetEventName() { return waiter_.GetEventName(id_); }

   private:
    friend EventsWaiter;
    EventNotifier(EventId id, EventsWaiter* waiter)
        : id_(id), waiter_(*waiter) {}
    EventNotifier(const EventNotifier&) = delete;
    void operator=(const EventNotifier&) = delete;

    EventId id_;
    EventsWaiter& waiter_;
  };

  EventsWaiter();
  EventsWaiter(const EventsWaiter&) = delete;
  EventsWaiter& operator=(const EventsWaiter&) = delete;

  // Register a level-triggered event. If the checker returns true or
  // EventNotifier::NotifyEvent is called, the corresponding event will be
  // distributed.
  std::shared_ptr<EventNotifier> RegisterEvent(const std::string& name,
                                               EventChecker checker);

  // Register an edge-triggered event. The corresponding event will be
  // distributed when EventNotifier::NotifyEvent is called.
  std::shared_ptr<EventNotifier> RegisterEvent(const std::string& name);

  void UnregisterEvent(const EventId& id);

  // Wait any of the registered events
  std::string WaitEvent();

 private:
  friend EventNotifier;

  enum class TriggerType { LevelTriggered, EdgeTriggered };

  struct EventInfo {
    EventId id;
    std::string name;
    TriggerType type;
    EventChecker checker;
  };

  void TriggerEvent(const EventId& id);

  std::string GetEventName(const EventId& id);

  std::unordered_map<EventId, EventInfo> events_;
  paddle::memory::SpinLock events_lock_;
  std::atomic<std::string*> trigger_event_;
  std::atomic<uint64_t> counter_;
  std::atomic<bool> waiting_;
  EventCount cv_;
};

template <typename Notifier>
class TaskTracker {
 public:
  TaskTracker() = default;

  explicit TaskTracker(Notifier& notifier) : notifier_(&notifier) {}

  TaskTracker(const TaskTracker&) = delete;

  TaskTracker& operator=(const TaskTracker&) = delete;

  ~TaskTracker() = default;

  void AddCounter() { num_tasks_.fetch_add(1, std::memory_order_relaxed); }

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
