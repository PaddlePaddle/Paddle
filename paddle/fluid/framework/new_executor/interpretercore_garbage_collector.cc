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

  WorkQueueOptions options(/*num_threads*/ 1, /*allow_spinning*/ true,
                           /*track_task*/ false);
  queue_ = CreateSingleThreadedWorkQueue(options);
}

void InterpreterCoreGarbageCollector::Add(
    std::shared_ptr<memory::Allocation> garbage,
    paddle::platform::DeviceEvent& event, const platform::DeviceContext* ctx) {
  if (max_memory_size_ <= 1) {
    Free(garbage, event, ctx);
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
      Free(garbage_ptr, event, ctx);
    }
  }
}

void InterpreterCoreGarbageCollector::Add(paddle::framework::Variable* var,
                                          paddle::platform::DeviceEvent& event,
                                          const platform::DeviceContext* ctx) {
  if (var->IsType<LoDTensor>()) {
    Add(var->GetMutable<LoDTensor>()->MoveMemoryHolder(), event, ctx);
  } else if (var->IsType<SelectedRows>()) {
    Add(var->GetMutable<SelectedRows>()->mutable_value()->MoveMemoryHolder(),
        event, ctx);
  } else if (var->IsType<LoDTensorArray>()) {
    auto* tensor_arr = var->GetMutable<LoDTensorArray>();
    for (auto& t : *tensor_arr) {
      Add(t.MoveMemoryHolder(), event, ctx);
    }
  } else {
    PADDLE_THROW(platform::errors::Unimplemented(
        "The variable(%s) is not supported in eager deletion.",
        framework::ToTypeName(var->Type())));
  }
}

void InterpreterCoreGarbageCollector::Free(GarbageQueue* garbages,
                                           paddle::platform::DeviceEvent& event,
                                           const platform::DeviceContext* ctx) {
  event.Record(ctx);
  event.SetFininshed();  // Only for CPU Event
  queue_->AddTask([ container = garbages, event = &event ]() {
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

void InterpreterCoreGarbageCollector::Free(
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
