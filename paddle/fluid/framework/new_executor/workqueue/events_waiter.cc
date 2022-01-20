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

#include "paddle/fluid/framework/new_executor/workqueue/events_waiter.h"
#include <glog/logging.h>
#include "paddle/fluid/platform/enforce.h"

namespace paddle {
namespace framework {

EventsWaiter::EventsWaiter()
    : trigger_event_(nullptr), counter_(0), waiting_(false), cv_(1) {}

std::shared_ptr<EventsWaiter::EventNotifier> EventsWaiter::RegisterEvent(
    const std::string& name, EventChecker checker) {
  auto counter = counter_.fetch_add(1);
  auto id = std::hash<std::string>()(name + std::to_string(counter));
  VLOG(10) << "Register event id:" << id << " name:" << name;
  auto notifier = std::shared_ptr<EventNotifier>(new EventNotifier(id, this));
  EventInfo evt{id, name, TriggerType::LevelTriggered, std::move(checker)};
  std::lock_guard<paddle::memory::SpinLock> guard(events_lock_);
  events_[id] = std::move(evt);
  return notifier;
}

std::shared_ptr<EventsWaiter::EventNotifier> EventsWaiter::RegisterEvent(
    const std::string& name) {
  auto counter = counter_.fetch_add(1);
  auto id = std::hash<std::string>()(name + std::to_string(counter));
  VLOG(10) << "Register event id:" << id << " name:" << name;
  auto notifier = std::shared_ptr<EventNotifier>(new EventNotifier(id, this));
  EventInfo evt{id, name, TriggerType::EdgeTriggered, []() { return false; }};
  std::lock_guard<paddle::memory::SpinLock> guard(events_lock_);
  events_[id] = std::move(evt);
  return notifier;
}

void EventsWaiter::UnregisterEvent(const EventId& id) {
  VLOG(10) << "Unregister event id:" << id;
  std::lock_guard<paddle::memory::SpinLock> guard(events_lock_);
  events_.erase(id);
}

std::string EventsWaiter::WaitEvent() {
  // only one user can wait at any time
  bool waiting = false;
  if (!waiting_.compare_exchange_strong(waiting, true,
                                        std::memory_order_seq_cst,
                                        std::memory_order_relaxed)) {
    PADDLE_THROW(
        platform::errors::ResourceExhausted("Another thread is waiting."));
  }
  auto w = cv_.GetWaiter(0);
  cv_.Prewait();
  std::string* triggered = trigger_event_;
  if (triggered == nullptr) {
    // checkers
    {
      std::lock_guard<paddle::memory::SpinLock> guard(events_lock_);
      for (auto& kv : events_) {
        auto& evt = kv.second;
        if (TriggerType::LevelTriggered == evt.type && evt.checker()) {
          triggered = new std::string(evt.name);
          break;
        }
      }
    }
    if (triggered != nullptr) {
      std::string* prev = nullptr;
      if (!trigger_event_.compare_exchange_strong(prev, triggered,
                                                  std::memory_order_seq_cst,
                                                  std::memory_order_relaxed)) {
        delete triggered;
        triggered = prev;
      }
    }
  }
  if (triggered) {
    cv_.CancelWait();
  } else {
    cv_.CommitWait(w);
    triggered = trigger_event_;
  }
  trigger_event_.store(nullptr, std::memory_order_relaxed);
  waiting_.store(false);
  auto trigger_event = *triggered;
  delete triggered;
  return trigger_event;
}

int EventsWaiter::Clear() {
  bool waiting = false;
  if (!waiting_.compare_exchange_strong(waiting, true,
                                        std::memory_order_seq_cst,
                                        std::memory_order_relaxed)) {
    return -1;
  }
  trigger_event_.store(nullptr, std::memory_order_relaxed);
  waiting_.store(false);
  return 0;
}

void EventsWaiter::TriggerEvent(const EventId& id) {
  VLOG(10) << "Try to trigger event id:" << id;
  std::string* trigger_event = new std::string;
  {
    std::lock_guard<paddle::memory::SpinLock> guard(events_lock_);
    auto iter = events_.find(id);
    if (iter == events_.end()) {
      delete trigger_event;
      return;
    }
    *trigger_event = iter->second.name;
  }
  std::string* prev = nullptr;
  if (!trigger_event_.compare_exchange_strong(prev, trigger_event,
                                              std::memory_order_seq_cst,
                                              std::memory_order_relaxed)) {
    delete trigger_event;
    return;
  }
  VLOG(10) << "Triggered event id:" << id << " name:" << *trigger_event;
  cv_.Notify(true);
}

std::string EventsWaiter::GetEventName(const EventId& id) {
  std::lock_guard<paddle::memory::SpinLock> guard(events_lock_);
  auto iter = events_.find(id);
  if (iter == events_.end()) {
    return "Unregistered";
  }
  return iter->second.name;
}

}  // namespace framework
}  // namespace paddle
