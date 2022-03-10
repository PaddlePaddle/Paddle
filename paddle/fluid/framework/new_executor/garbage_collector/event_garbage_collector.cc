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

#include "paddle/fluid/framework/new_executor/garbage_collector/event_garbage_collector.h"

#if !defined(_WIN32)
#include <sched.h>
#else
#define NOMINMAX
#include <windows.h>
#endif  // !_WIN32

namespace paddle {
namespace framework {

InterpreterCoreEventGarbageCollector::InterpreterCoreEventGarbageCollector() {
  WorkQueueOptions options(/*name*/ "GarbageCollector", /*num_threads*/ 1,
                           /*allow_spinning*/ true,
                           /*track_task*/ false);
  queue_ = CreateSingleThreadedWorkQueue(options);
}

InterpreterCoreEventGarbageCollector::~InterpreterCoreEventGarbageCollector() {
  queue_.reset(nullptr);
}

void InterpreterCoreEventGarbageCollector::Add(
    Garbage garbage, platform::DeviceEvent* event,
    const platform::DeviceContext* ctx) {
  if (!garbage) {
    return;
  }

  if (max_memory_size_ <= 1) {
    Free(garbage, event, ctx);
  } else {
    std::unique_ptr<GarbageQueue> pending_delete_garbages;
    {  // lock guard
      std::lock_guard<memory::SpinLock> guard(spinlock_);
      cur_memory_size_ += garbage->size();
      garbages_->push_back(std::move(garbage));

      if (cur_memory_size_ >= max_memory_size_) {
        cur_memory_size_ = 0;
        pending_delete_garbages = std::move(garbages_);
        garbages_ = std::make_unique<GarbageQueue>();
      }
    }
  }
}

void InterpreterCoreEventGarbageCollector::Add(Variable* var) {
  PADDLE_THROW(platform::errors::Unimplemented(
      "Add(Variable* var) is not implemented for "
      "InterpreterCoreEventGarbageCollector."));
}

void InterpreterCoreEventGarbageCollector::Add(
    Variable* var, platform::DeviceEvent* event,
    const platform::DeviceContext* ctx) {
  if (UNLIKELY(max_memory_size_ < 0) || var == nullptr) {
    return;
  }

  if (var->IsType<LoDTensor>()) {
    Add(var->GetMutable<LoDTensor>()->MoveMemoryHolder(), event, ctx);
  } else if (var->IsType<
                 operators::reader::
                     OrderedMultiDeviceLoDTensorBlockingQueueHolder>()) {
    // TODO(xiongkun03) in old executor, this type of variable is not support
    // eager deletion. so we just leave it here ?
  } else if (var->IsType<LoDRankTable>()) {
    // TODO(xiongkun03) in old executor, this type of variable is not support
    // eager deletion. so we just leave it here ?
  } else if (var->IsType<phi::SelectedRows>()) {
    Add(var->GetMutable<phi::SelectedRows>()
            ->mutable_value()
            ->MoveMemoryHolder(),
        event, ctx);
    var->GetMutable<phi::SelectedRows>()->mutable_rows()->clear();
  } else if (var->IsType<LoDTensorArray>()) {
    auto* tensor_arr = var->GetMutable<LoDTensorArray>();
    for (auto& t : *tensor_arr) {
      Add(t.MoveMemoryHolder(), event, ctx);
    }
  } else if (var->IsType<std::vector<Scope*>>()) {
    // NOTE(@xiongkun03) conditional_op / while_op will create a STEP_SCOPE
    // refer to executor.cc to see what old garbage collector does.
    // do nothing, because the sub scope will be deleted by sub-executor.
  } else {
    PADDLE_THROW(platform::errors::Unimplemented(
        "The variable(%s) is not supported in eager deletion.",
        framework::ToTypeName(var->Type())));
  }
}

void InterpreterCoreEventGarbageCollector::Free(
    GarbageQueue* garbages, platform::DeviceEvent* event,
    const platform::DeviceContext* ctx) {
  event->Record(ctx);
  event->SetFininshed();  // Only for CPU Event
  queue_->AddTask([ container = garbages, event = event ]() {
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

void InterpreterCoreEventGarbageCollector::Free(
    const Garbage& garbage, platform::DeviceEvent* event,
    const platform::DeviceContext* ctx) {
  event->Record(ctx);
  event->SetFininshed();  // Only for CPU Event
  queue_->AddTask([ container = garbage, event = event ]() {
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
