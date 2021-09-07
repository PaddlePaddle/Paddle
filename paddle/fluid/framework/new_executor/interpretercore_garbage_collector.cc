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

#include "paddle/fluid/framework/new_executor/interpretercore_garbage_collector.h"
#include "paddle/fluid/framework/garbage_collector.h"

namespace paddle {
namespace framework {

InterpreterCoreGarbageCollector::InterpreterCoreGarbageCollector() {
  garbages_.reset(new GarbageQueue());
  max_memory_size_ = static_cast<size_t>(GetEagerDeletionThreshold());
  cur_memory_size_ = 0;
  queue_ = CreateSingleThreadedWorkQueue();
}

void InterpreterCoreGarbageCollector::Add(GarbageQueue& garbage,
                                          paddle::platform::DeviceEvent& event,
                                          const platform::DeviceContext* ctx) {
  if (max_memory_size_ <= 1) {
    auto* garbage_ptr = new GarbageQueue(std::move(garbage));
    DoFree(garbage_ptr, event, ctx);
  } else {
    GarbageQueue* garbage_ptr = nullptr;
    {
      std::lock_guard<paddle::memory::SpinLock> guard(spinlock_);
      for (auto& g : garbage) {
        if (!g) continue;
        cur_memory_size_ += g->size();
        garbages_->push_back(std::move(g));
      }
      if (cur_memory_size_ >= max_memory_size_) {
        cur_memory_size_ = 0;
        garbage_ptr = garbages_.release();
        garbages_.reset(new GarbageQueue());
      }
    }
    if (garbage_ptr) {
      DoFree(garbage_ptr, event, ctx);
    }
  }
}

void InterpreterCoreGarbageCollector::Add(
    std::shared_ptr<memory::Allocation> garbage,
    paddle::platform::DeviceEvent& event, const platform::DeviceContext* ctx) {
  if (max_memory_size_ <= 1) {
    DoFree(garbage, event, ctx);
  } else {
    if (!garbage) return;
    GarbageQueue* garbage_ptr = nullptr;
    {
      std::lock_guard<paddle::memory::SpinLock> guard(spinlock_);
      cur_memory_size_ += garbage->size();
      garbages_->push_back(std::move(garbage));

      if (cur_memory_size_ >= max_memory_size_) {
        cur_memory_size_ = 0;
        garbage_ptr = garbages_.release();
        garbages_.reset(new GarbageQueue());
      }
    }
    if (garbage_ptr) {
      DoFree(garbage_ptr, event, ctx);
    }
  }
}

void InterpreterCoreGarbageCollector::DoFree(
    GarbageQueue* garbage, paddle::platform::DeviceEvent& event,
    const platform::DeviceContext* ctx) {
  event.Record(ctx);
  event.SetFininshed();  // Only for CPU Event
  queue_->AddTask([ container = garbage, event = &event ]() {
    while (!event->Query()) {
#if defined(_WIN32)
      SleepEx(50, FALSE);
#else
      sched_yield();
#endif
      continue;
    }
    delete container;
  });
}

void InterpreterCoreGarbageCollector::DoFree(
    std::shared_ptr<memory::Allocation>& garbage,
    paddle::platform::DeviceEvent& event, const platform::DeviceContext* ctx) {
  event.Record(ctx);
  event.SetFininshed();  // Only for CPU Event
  queue_->AddTask([ container = garbage, event = &event ]() {
    while (!event->Query()) {
#if defined(_WIN32)
      SleepEx(50, FALSE);
#else
      sched_yield();
#endif
      continue;
    }
  });
}

}  // namespace framework
}  // namespace paddle
