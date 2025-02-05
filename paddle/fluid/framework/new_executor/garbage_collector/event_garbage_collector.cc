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

namespace paddle::framework {

InterpreterCoreEventGarbageCollector::InterpreterCoreEventGarbageCollector(
    const std::vector<Instruction>& vec_instruction)
    : queue_(nullptr), gc_event_(), events_() {
  WorkQueueOptions options(/*name*/ "GarbageCollector",
                           /*num_threads*/ 1,
                           /*allow_spinning*/ true,
                           /*track_task*/ false);
  queue_ = CreateSingleThreadedWorkQueue(options);
  for (auto& instruct : vec_instruction) {
    gc_event_.emplace_back(instruct.DeviceContext().GetPlace(),
                           platform::GenerateDeviceEventFlag());
  }
}

InterpreterCoreEventGarbageCollector::InterpreterCoreEventGarbageCollector(
    const std::vector<std::unique_ptr<InstructionBase>>& vec_instruction)
    : queue_(nullptr), gc_event_(), events_() {
  WorkQueueOptions options(/*name*/ "GarbageCollector",
                           /*num_threads*/ 1,
                           /*allow_spinning*/ true,
                           /*track_task*/ false);
  queue_ = CreateSingleThreadedWorkQueue(options);
  for (auto& instruct : vec_instruction) {
    gc_event_.emplace_back(instruct->DeviceContext().GetPlace(),
                           platform::GenerateDeviceEventFlag());
  }
}

InterpreterCoreEventGarbageCollector::
    ~InterpreterCoreEventGarbageCollector() {  // NOLINT
  queue_.reset(nullptr);
}

void InterpreterCoreEventGarbageCollector::Add(Variable* var,
                                               const Instruction& instr) {
  PADDLE_ENFORCE_LT(instr.Id(),
                    gc_event_.size(),
                    common::errors::OutOfRange(
                        "The index should be less than the size of gc event "
                        ", but got index is %d and size is %d",
                        instr.Id(),
                        gc_event_.size()));
  Add(var, &gc_event_.at(instr.Id()), &instr.DeviceContext());
}

void InterpreterCoreEventGarbageCollector::Add(Variable* var,
                                               const InstructionBase* instr) {
  PADDLE_ENFORCE_LT(instr->Id(),
                    gc_event_.size(),
                    common::errors::OutOfRange(
                        "The index should be less than the size of gc event "
                        ", but got index is %d and size is %d",
                        instr->Id(),
                        gc_event_.size()));
  Add(var, &gc_event_.at(instr->Id()), &instr->DeviceContext());
}

void InterpreterCoreEventGarbageCollector::Add(Variable* var,
                                               platform::DeviceEvent* event,
                                               const phi::DeviceContext* ctx) {
  if (UNLIKELY(max_memory_size_ < 0) || var == nullptr) {
    return;
  }

  if (var->IsType<phi::DenseTensor>()) {
    Add(var->GetMutable<phi::DenseTensor>()->MoveMemoryHolder(), event, ctx);
  } else if (
      var->IsType<
          operators::reader::
              OrderedMultiDeviceDenseTensorBlockingQueueHolder>()) {  // NOLINT
    // TODO(xiongkun03) in old executor, this type of variable is not support
    // eager deletion. so we just leave it here ?
  } else if (var->IsType<phi::SelectedRows>()) {
    Add(var->GetMutable<phi::SelectedRows>()
            ->mutable_value()
            ->MoveMemoryHolder(),
        event,
        ctx);
  } else if (var->IsType<phi::SparseCooTensor>()) {
    Add(var->GetMutable<phi::SparseCooTensor>()
            ->mutable_values()
            ->MoveMemoryHolder(),
        event,
        ctx);
    Add(var->GetMutable<phi::SparseCooTensor>()
            ->mutable_indices()
            ->MoveMemoryHolder(),
        event,
        ctx);
  } else if (var->IsType<phi::SparseCsrTensor>()) {
    Add(var->GetMutable<phi::SparseCsrTensor>()
            ->mutable_crows()
            ->MoveMemoryHolder(),
        event,
        ctx);
    Add(var->GetMutable<phi::SparseCsrTensor>()
            ->mutable_cols()
            ->MoveMemoryHolder(),
        event,
        ctx);
    Add(var->GetMutable<phi::SparseCsrTensor>()
            ->mutable_values()
            ->MoveMemoryHolder(),
        event,
        ctx);
  } else if (var->IsType<phi::TensorArray>()) {
    auto* tensor_arr = var->GetMutable<phi::TensorArray>();
    for (auto& t : *tensor_arr) {
      Add(t.MoveMemoryHolder(), event, ctx);
    }
  } else if (var->IsType<std::vector<Scope*>>()) {
    // NOTE(@xiongkun03) conditional_op / while_op will create a STEP_SCOPE
    // refer to executor.cc to see what old garbage collector does.
    // do nothing, because the sub scope will be deleted by sub-executor.
  } else {
    PADDLE_THROW(common::errors::Unimplemented(
        "The variable(%s) is not supported in eager deletion.",
        framework::ToTypeName(var->Type())));
  }
}

void InterpreterCoreEventGarbageCollector::Add(Garbage garbage,
                                               platform::DeviceEvent* event,
                                               const phi::DeviceContext* ctx) {
  if (!garbage) {
    return;
  }

  if (max_memory_size_ <= 1) {
    Free(garbage, event, ctx);
  } else {
    {  // lock guard
      std::lock_guard<memory::SpinLock> guard(spinlock_);
      cur_memory_size_ += static_cast<int64_t>(garbage->size());
      garbages_->push_back(std::move(garbage));
      events_[ctx] = event;

      if (cur_memory_size_ >= max_memory_size_) {
        FreeGarbages();
      }
    }
  }
}

void InterpreterCoreEventGarbageCollector::Free(const Garbage& garbage,
                                                platform::DeviceEvent* event,
                                                const phi::DeviceContext* ctx) {
  event->Record(ctx);
  event->SetFinished();  // Only for CPU Event
  queue_->AddTask([container = garbage, event = event]() {
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

void InterpreterCoreEventGarbageCollector::FreeGarbages() {
  for (auto& vals : events_) {
    vals.second->Record(vals.first);
    vals.second->SetFinished();  // Only for CPU Event
  }
  queue_->AddTask(
      [container = std::move(*garbages_), events = std::move(events_)]() {
        for (auto& vals : events) {
          while (!vals.second->Query()) {
#if defined(_WIN32)
            SleepEx(50, FALSE);
#else
            sched_yield();
#endif
            continue;
          }
        }
      });
  cur_memory_size_ = 0;
  garbages_->clear();
  events_.clear();
}

}  // namespace paddle::framework
