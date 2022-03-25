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

constexpr EventsWaiter::EventId kEmptyEventId = 0;

EventsWaiter::EventsWaiter()
    : trigger_event_(kEmptyEventId),
      counter_(0),
      eof_(true),
      waiting_(false),
      cv_(1) {}

std::shared_ptr<EventsWaiter::EventNotifier> EventsWaiter::RegisterEvent(
    const std::string& name, EventChecker checker) {
  EventId id = kEmptyEventId;
  EventInfo* evt = nullptr;
  do {
    auto counter = counter_.fetch_add(1);
    id = std::hash<std::string>()(name + std::to_string(counter));
    if (id == kEmptyEventId) {
      continue;
    }
    std::lock_guard<paddle::memory::SpinLock> guard(events_lock_);
    if (events_.count(id) > 0) {
      continue;
    }
    evt = &(events_[id]);
  } while (evt == nullptr);
  evt->id = id;
  evt->name = name;
  evt->type = TriggerType::LevelTriggered;
  evt->checker = std::move(checker);
  eof_.store(false, std::memory_order_relaxed);
  VLOG(10) << "Register event id:" << id << " name:" << name;
  auto notifier = std::shared_ptr<EventNotifier>(new EventNotifier(id, this));
  return notifier;
}

std::shared_ptr<EventsWaiter::EventNotifier> EventsWaiter::RegisterEvent(
    const std::string& name) {
  EventId id = kEmptyEventId;
  EventInfo* evt = nullptr;
  do {
    auto counter = counter_.fetch_add(1);
    id = std::hash<std::string>()(name + std::to_string(counter));
    if (id == kEmptyEventId) {
      continue;
    }
    std::lock_guard<paddle::memory::SpinLock> guard(events_lock_);
    if (events_.count(id) > 0) {
      continue;
    }
    evt = &(events_[id]);
  } while (evt == nullptr);
  evt->id = id;
  evt->name = name;
  evt->type = TriggerType::EdgeTriggered;
  evt->checker = []() { return false; };
  eof_.store(false, std::memory_order_relaxed);
  VLOG(10) << "Register event id:" << id << " name:" << name;
  auto notifier = std::shared_ptr<EventNotifier>(new EventNotifier(id, this));
  return notifier;
}

void EventsWaiter::UnregisterEvent(const EventId& id) {
  VLOG(10) << "Unregister event id:" << id;
  {
    std::lock_guard<paddle::memory::SpinLock> guard(events_lock_);
    deleted_events_.insert(id);
    if (deleted_events_.size() == events_.size()) {
      eof_.store(true, std::memory_order_relaxed);
    }
  }
  if (eof_.load(std::memory_order_relaxed)) {
    cv_.Notify(true);
  }
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
  EventId triggered = trigger_event_;
  while (triggered == kEmptyEventId && !eof_) {
    cv_.Prewait();

    // double check
    triggered = trigger_event_;
    // checkers
    if (triggered == kEmptyEventId) {
      {
        std::lock_guard<paddle::memory::SpinLock> guard(events_lock_);
        for (auto& kv : events_) {
          auto& evt = kv.second;
          if (TriggerType::LevelTriggered == evt.type && evt.checker()) {
            triggered = evt.id;
            break;
          }
        }
      }
      if (triggered != kEmptyEventId) {
        EventId prev = kEmptyEventId;
        if (!trigger_event_.compare_exchange_strong(
                prev, triggered, std::memory_order_seq_cst,
                std::memory_order_relaxed)) {
          triggered = prev;
        }
      }
    }

    if (triggered != kEmptyEventId || eof_) {
      cv_.CancelWait();
    } else {
      cv_.CommitWait(w);
      triggered = trigger_event_;
    }
  }

  trigger_event_.store(kEmptyEventId, std::memory_order_relaxed);
  std::string evt_name =
      triggered == kEmptyEventId ? "NoEventNotifier" : GetEventName(triggered);
  VLOG(10) << "Consume event id:" << triggered << ", name:" << evt_name;
  // lazy deletion
  {
    triggered = trigger_event_;
    std::lock_guard<paddle::memory::SpinLock> guard(events_lock_);
    if (deleted_events_.size() > 0) {
      for (auto evt : deleted_events_) {
        if (evt == triggered) {
          continue;
        }
        events_.erase(evt);
      }
      deleted_events_.clear();
    }
  }
  waiting_.store(false, std::memory_order_relaxed);
  return evt_name;
}

int EventsWaiter::Clear() {
  bool waiting = false;
  if (!waiting_.compare_exchange_strong(waiting, true,
                                        std::memory_order_seq_cst,
                                        std::memory_order_relaxed)) {
    return -1;
  }
  trigger_event_.store(kEmptyEventId, std::memory_order_relaxed);
  waiting_.store(false);
  return 0;
}

void EventsWaiter::TriggerEvent(const EventId& id) {
  VLOG(10) << "Try to trigger event id:" << id;
  EventId prev = kEmptyEventId;
  if (!trigger_event_.compare_exchange_strong(
          prev, id, std::memory_order_seq_cst, std::memory_order_relaxed)) {
    VLOG(10) << "Event id:" << prev << " is pending";
    return;
  }
  VLOG(10) << "Triggered event id:" << id;
  cv_.Notify(true);
}

void EventsWaiter::CancelEvent(const EventId& id) {
  VLOG(10) << "Try to cancel event id:" << id;
  EventId prev = id;
  if (!trigger_event_.compare_exchange_strong(prev, kEmptyEventId,
                                              std::memory_order_seq_cst,
                                              std::memory_order_relaxed)) {
    VLOG(10) << "Event id:" << prev << " is pending";
    return;
  }
  VLOG(10) << "Cancelled event id:" << id;
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
