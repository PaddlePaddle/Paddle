/* Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#pragma once

#include <cstring>
#include <mutex>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <vector>
#include "paddle/fluid/platform/event.h"

namespace paddle {
namespace platform {

template <typename HeadType, typename... RestTypes>
struct ContainsStdString
    : std::conditional_t<
          std::is_same<std::string, std::remove_cv_t<std::remove_reference_t<
                                        HeadType>>>::value,
          std::true_type, ContainsStdString<RestTypes...>> {};

template <typename TailType>
struct ContainsStdString<TailType>
    : std::is_same<std::string,
                   std::remove_cv_t<std::remove_reference_t<TailType>>> {};

template <typename EventType>
class EventContainer {
 public:
  EventContainer() {
    event_blocks_ = cur_event_block_ = new EventBlock;
    str_blocks_ = cur_str_block_ = new StringBlock;
  }
  ~EventContainer() {
    Reduce();
    delete event_blocks_;
    for (auto cur = str_blocks_; cur != nullptr;) {
      auto next = cur->next;
      delete cur;
      cur = next;
    }
  }
  DISABLE_COPY_AND_ASSIGN(EventContainer);

 public:
  // Record an event
  template <typename... Args>
  void Record(Args &&... args) {
    DoRecord(ContainsStdString<Args...>(), std::forward<Args>(args)...);
  }

  // Get all events and clear the container
  std::vector<EventType> Reduce();

  // Return a buffer to store the string attribute of Event.
  // HostEventRecorder locates in the static data section.
  // So it's safe to use arena to avoid fragmented allocations.
  char *GetStrBufFromArena(size_t size) { return GetStringStorage(size); }

 private:
  struct EventBlock {
    union InitDeferedEvent {
      InitDeferedEvent() {}
      ~InitDeferedEvent() {}

      EventType event;
    };

    static constexpr size_t kBlockSize = 1 << 24;  // 16 MB
    static constexpr size_t kAvailSize =
        kBlockSize - sizeof(size_t) - sizeof(nullptr);
    static constexpr size_t kNumEvents = kAvailSize / sizeof(InitDeferedEvent);
    static constexpr size_t kPadSize =
        kAvailSize - kNumEvents * sizeof(InitDeferedEvent);
    static constexpr size_t kMinimumEventsPerBlock = 1024;
    static_assert(
        kNumEvents >= kMinimumEventsPerBlock,
        "EventType is too large for kBlockSize, make kBlockSize larger");

    size_t offset = 0;
    EventBlock *next = nullptr;
    InitDeferedEvent events[kNumEvents];
    char padding[kPadSize];
  };
  static_assert(sizeof(EventBlock) == EventBlock::kBlockSize,
                "sizeof EventBlock must equal to kBlockSize");

  struct StringBlock {
    static constexpr size_t kBlockSize = 1 << 22;  // 4 MB
    static constexpr size_t kAvailSize =
        kBlockSize - sizeof(size_t) - sizeof(nullptr);

    size_t offset = 0;
    StringBlock *next = nullptr;
    char storage[kAvailSize];
  };
  static_assert(sizeof(StringBlock) == StringBlock::kBlockSize,
                "sizeof StringBlock must equal to kBlockSize");

  // Record an event with string arguments
  template <typename... Args>
  void DoRecord(std::true_type, Args &&... args) {
    auto *storage = GetEventStorage();
    std::function<void *(size_t)> allocator = [this](size_t size) {
      return GetStrBufFromArena(size);
    };
    new (storage) EventType(allocator, std::forward<Args>(args)...);
  }

  // Record an event without any string argument
  template <typename... Args>
  void DoRecord(std::false_type, Args &&... args) {
    auto *storage = GetEventStorage();
    new (storage) EventType(std::forward<Args>(args)...);
  }

  EventType *GetEventStorage();

  char *GetStringStorage(size_t sz);

  EventBlock *event_blocks_ = nullptr;
  EventBlock *cur_event_block_ = nullptr;
  StringBlock *str_blocks_ = nullptr;
  StringBlock *cur_str_block_ = nullptr;
};

template <typename EventType>
std::vector<EventType> EventContainer<EventType>::Reduce() {
  std::vector<EventType> all_events;
  size_t event_cnt = 0;
  for (auto cur = event_blocks_; cur != nullptr; cur = cur->next) {
    event_cnt += cur->offset;
  }
  all_events.reserve(event_cnt);
  for (auto cur = event_blocks_; cur != nullptr;) {
    for (size_t i = 0; i < cur->offset; ++i) {
      all_events.emplace_back(cur->events[i].event);
    }
    auto next = cur->next;
    delete cur;
    cur = next;
  }
  event_blocks_ = cur_event_block_ = new EventBlock;
  return std::move(all_events);
}

template <typename EventType>
EventType *EventContainer<EventType>::GetEventStorage() {
  if (UNLIKELY(cur_event_block_->offset >=
               EventBlock::kNumEvents)) {  // another block
    cur_event_block_->next = new EventBlock;
    cur_event_block_ = cur_event_block_->next;
  }
  auto &obj = cur_event_block_->events[cur_event_block_->offset].event;
  ++cur_event_block_->offset;
  return &obj;
}

template <typename EventType>
char *EventContainer<EventType>::GetStringStorage(size_t sz) {
  if (UNLIKELY(cur_str_block_->offset + sz >
               StringBlock::kAvailSize)) {  // another block
    cur_str_block_->next = new StringBlock;
    cur_str_block_ = cur_str_block_->next;
  }
  char *storage = cur_str_block_->storage + cur_str_block_->offset;
  cur_str_block_->offset += sz;
  return storage;
}

struct ThreadEventSection {
  std::string thread_name;
  uint64_t thread_id;
  std::vector<CommonEvent> events;
};

class ThreadEventRecorder {
 public:
  ThreadEventRecorder();
  DISABLE_COPY_AND_ASSIGN(ThreadEventRecorder);

 public:
  // Forward call to EventContainer::Record
  template <typename... Args>
  void RecordEvent(Args &&... args) {
    base_evt_cntr_.Record(std::forward<Args>(args)...);
  }

  ThreadEventSection GatherEvents() {
    ThreadEventSection thr_sec;
    thr_sec.thread_name = thread_name_;
    thr_sec.thread_id = thread_id_;
    thr_sec.events = std::move(base_evt_cntr_.Reduce());
    return std::move(thr_sec);
  }

 private:
  uint64_t thread_id_;
  std::string thread_name_;
  EventContainer<CommonEvent> base_evt_cntr_;
};

struct HostEventSection {
  std::string process_name;
  uint64_t process_id;
  std::vector<ThreadEventSection> thr_sections;
};

class HostEventRecorder {
 public:
  // singleton
  static HostEventRecorder &GetInstance() {
    static HostEventRecorder instance;
    return instance;
  }

  // If your string argument has a longer lifetime than the Event,
  // use 'const char*'. e.g.: string literal, op name, etc.
  // Do your best to avoid using 'std::string' as the argument type.
  // It will cause deep-copy to harm performance.
  template <typename... Args>
  void RecordEvent(Args &&... args) {
    GetThreadLocalRecorder().RecordEvent(std::forward<Args>(args)...);
  }

  // Poor performance, call it at the ending
  HostEventSection GatherEvents();

  void RegisterThreadRecorder(uint64_t tid, ThreadEventRecorder *recorder) {
    const std::lock_guard<std::mutex> guard(thread_recorders_lock_);
    thread_recorders_[tid] = recorder;
  }

 private:
  HostEventRecorder() = default;
  DISABLE_COPY_AND_ASSIGN(HostEventRecorder);

  ThreadEventRecorder &GetThreadLocalRecorder() {
    static thread_local ThreadEventRecorder tls_recorder;
    return tls_recorder;
  }

  std::mutex thread_recorders_lock_;
  std::unordered_map<uint64_t, ThreadEventRecorder *> thread_recorders_;
};

}  // namespace platform
}  // namespace paddle
