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
#include <cstddef>
#include <functional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include "paddle/fluid/framework/new_executor/workqueue/event_count.h"
#include "paddle/fluid/memory/allocation/spin_lock.h"

namespace paddle {
namespace framework {

// A multiplexing waiter, be able to wait multiple kinds of events
// simultaneously.
// Muti-Producer single-consumer single-slot message-queue.
class EventsWaiter {
 public:
  using EventId = std::size_t;

  using EventChecker = std::function<bool()>;

  // Make sure EventsWaiter has a longer lifetime than EventNotifier.
  class EventNotifier {
   public:
    ~EventNotifier() { waiter_.UnregisterEvent(id_); }

    void NotifyEvent() { waiter_.TriggerEvent(id_); }

    void CancelEvent() { waiter_.CancelEvent(id_); }

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

  // Blocking the calling thread to wait any of the registered events.
  std::string WaitEvent();

  // Nonblocking.
  // Clear the slot, no matter whether there is an event.
  // Return value:
  //     -1 : another thread is waiting.
  //      0 : succ.
  int Clear();

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

  void CancelEvent(const EventId& id);

  std::string GetEventName(const EventId& id);

  std::unordered_map<EventId, EventInfo> events_;
  std::unordered_set<EventId> deleted_events_;
  paddle::memory::SpinLock events_lock_;
  std::atomic<EventId> trigger_event_;
  std::atomic<uint64_t> counter_;
  std::atomic<bool> eof_;
  std::atomic<bool> waiting_;
  EventCount cv_;
};

}  // namespace framework
}  // namespace paddle
