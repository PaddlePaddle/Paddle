// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

#include <string>
#include <type_traits>
#include <vector>

#include "paddle/common/macros.h"
#include "paddle/phi/common/thread_data_registry.h"
#include "paddle/phi/core/os_info.h"

namespace phi {

template <typename HeadType, typename... RestTypes>
struct ContainsStdString
    : std::conditional_t<
          std::is_same<
              std::string,
              std::remove_cv_t<std::remove_reference_t<HeadType>>>::value,
          std::true_type,
          ContainsStdString<RestTypes...>> {};

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
  void Record(Args &&...args) {
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
    union InitDeferredEvent {
      InitDeferredEvent() {}
      ~InitDeferredEvent() {}

      EventType event;
    };

    static constexpr size_t kBlockSize = 1 << 24;  // 16 MB
    static constexpr size_t kAvailSize =
        kBlockSize - sizeof(size_t) - sizeof(nullptr);
    static constexpr size_t kNumEvents = kAvailSize / sizeof(InitDeferredEvent);
    static constexpr size_t kPadSize =
        kAvailSize - kNumEvents * sizeof(InitDeferredEvent);
    static constexpr size_t kMinimumEventsPerBlock = 1024;
    static_assert(
        kNumEvents >= kMinimumEventsPerBlock,
        "EventType is too large for kBlockSize, make kBlockSize larger");

    size_t offset = 0;
    EventBlock *next = nullptr;
    InitDeferredEvent events[kNumEvents];
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
  void DoRecord(std::true_type, Args &&...args) {
    auto *storage = GetEventStorage();
    std::function<void *(size_t)> allocator = [this](size_t size) {
      return GetStrBufFromArena(size);
    };
    new (storage) EventType(allocator, std::forward<Args>(args)...);
  }

  // Record an event without any string argument
  template <typename... Args>
  void DoRecord(std::false_type, Args &&...args) {
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
  return all_events;
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

template <typename EventType>
struct ThreadEventSection {
  std::string thread_name;
  uint64_t thread_id;
  std::vector<EventType> events;
};

template <typename EventType>
class ThreadEventRecorder {
 public:
  ThreadEventRecorder() {
    thread_id_ = GetCurrentThreadSysId();
    thread_name_ = GetCurrentThreadName();
  }

  DISABLE_COPY_AND_ASSIGN(ThreadEventRecorder);

 public:
  // Forward call to EventContainer::Record
  template <typename... Args>
  void RecordEvent(Args &&...args) {
    base_evt_cntr_.Record(std::forward<Args>(args)...);
  }

  ThreadEventSection<EventType> GatherEvents() {
    ThreadEventSection<EventType> thr_sec;
    thr_sec.thread_name = thread_name_;
    thr_sec.thread_id = thread_id_;
    thr_sec.events = std::move(base_evt_cntr_.Reduce());
    return thr_sec;
  }

 private:
  uint64_t thread_id_;
  std::string thread_name_;
  EventContainer<EventType> base_evt_cntr_;
};

template <typename EventType>
struct HostEventSection {
  std::string process_name;
  uint64_t process_id;
  std::vector<ThreadEventSection<EventType>> thr_sections;
};

template <typename EventType>
class HostEventRecorder {
 public:
  // singleton
  static HostEventRecorder &GetInstance() {
    static HostEventRecorder instance;
    return instance;
  }

  // thread-safe
  // If your string argument has a longer lifetime than the Event,
  // use 'const char*'. e.g.: string literal, op name, etc.
  // Do your best to avoid using 'std::string' as the argument type.
  // It will cause deep-copy to harm performance.
  template <typename... Args>
  void RecordEvent(Args &&...args) {
    // Get thread local ThreadEventRecorder
    // If not exists, we create a new one.
    // Both HostEventRecorder and thread-local variable in
    // ThreadEventRecorderRegistry keep the shared pointer. We add this to
    // prevent ThreadEventRecorder being destroyed by thread-local variable in
    // ThreadEventRecorderRegistry and lose data.
    if (GetThreadLocalRecorder()->get() == nullptr) {
      std::shared_ptr<ThreadEventRecorder<EventType>>
          thread_event_recorder_ptr =
              std::make_shared<ThreadEventRecorder<EventType>>();
      *(GetThreadLocalRecorder()) = thread_event_recorder_ptr;
      thr_recorders_.push_back(thread_event_recorder_ptr);
    }
    (*GetThreadLocalRecorder())->RecordEvent(std::forward<Args>(args)...);
  }

  // thread-unsafe, make sure make sure there is no running tracing.
  // Poor performance, call it at the ending
  HostEventSection<EventType> GatherEvents() {
    HostEventSection<EventType> host_sec;
    host_sec.process_id = GetProcessId();
    host_sec.thr_sections.reserve(thr_recorders_.size());
    for (auto &v : thr_recorders_) {
      host_sec.thr_sections.emplace_back(std::move(v->GatherEvents()));
    }
    return host_sec;
  }

 private:
  using ThreadEventRecorderRegistry =
      phi::ThreadDataRegistry<std::shared_ptr<ThreadEventRecorder<EventType>>>;

  HostEventRecorder() = default;
  DISABLE_COPY_AND_ASSIGN(HostEventRecorder);

  std::shared_ptr<ThreadEventRecorder<EventType>> *GetThreadLocalRecorder() {
    return ThreadEventRecorderRegistry::GetInstance()
        .GetMutableCurrentThreadData();
  }
  // Hold all thread-local ThreadEventRecorders
  // ThreadEventRecorderRegistry and HostEventRecorder both take care of this
  // shared pointer. We add this to prevent ThreadEventRecorder being destroyed
  // by thread-local variable in ThreadEventRecorderRegistry and lose data.
  std::vector<std::shared_ptr<ThreadEventRecorder<EventType>>> thr_recorders_;
};

}  // namespace phi
