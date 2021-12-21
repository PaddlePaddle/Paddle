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

#include "paddle/fluid/framework/new_executor/workqueue_utils.h"
#include <cstdint>
#include <cstdlib>

namespace paddle {
namespace framework {

void* AlignedMalloc(size_t size, size_t alignment) {
  assert(alignment >= sizeof(void*) && (alignment & (alignment - 1)) == 0);
  size = (size + alignment - 1) / alignment * alignment;
#if defined(_POSIX_C_SOURCE) && _POSIX_C_SOURCE >= 200112L
  void* aligned_mem = nullptr;
  if (posix_memalign(&aligned_mem, alignment, size) != 0) {
    aligned_mem = nullptr;
  }
  return aligned_mem;
#elif defined(_WIN32)
  return _aligned_malloc(size, alignment);
#else
  void* mem = malloc(size + alignment);
  if (mem == nullptr) {
    return nullptr;
  }
  size_t adjust = alignment - reinterpret_cast<uint64_t>(mem) % alignment;
  void* aligned_mem = reinterpret_cast<char*>(mem) + adjust;
  *(reinterpret_cast<void**>(aligned_mem) - 1) = mem;
  assert(reinterpret_cast<uint64_t>(aligned_mem) % alignment == 0);
  return aligned_mem;
#endif
}

void AlignedFree(void* mem_ptr) {
#if defined(_POSIX_C_SOURCE) && _POSIX_C_SOURCE >= 200112L
  free(mem_ptr);
#elif defined(_WIN32)
  _aligned_free(mem_ptr);
#else
  if (mem_ptr) {
    free(*(reinterpret_cast<void**>(mem_ptr) - 1));
  }
#endif
}

EventsWaiter::EventsWaiter()
    : trigger_event_(nullptr), counter_(0), waiting_(false), cv_(1) {}

std::shared_ptr<EventsWaiter::EventNotifier> EventsWaiter::RegisterEvent(
    const std::string& name, EventChecker checker) {
  auto counter = counter_.fetch_add(1);
  auto id = std::hash<std::string>()(name + std::to_string(counter));
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
  auto notifier = std::shared_ptr<EventNotifier>(new EventNotifier(id, this));
  EventInfo evt{id, name, TriggerType::EdgeTriggered, []() { return false; }};
  std::lock_guard<paddle::memory::SpinLock> guard(events_lock_);
  events_[id] = std::move(evt);
  return notifier;
}

void EventsWaiter::UnregisterEvent(const EventId& id) {
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

void EventsWaiter::TriggerEvent(const EventId& id) {
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
