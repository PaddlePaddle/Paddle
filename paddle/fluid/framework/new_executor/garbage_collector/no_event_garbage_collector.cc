// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/framework/new_executor/garbage_collector/no_event_garbage_collector.h"

namespace paddle {
namespace framework {

InterpreterCoreNoEventGarbageCollector::
    InterpreterCoreNoEventGarbageCollector() {
  WorkQueueOptions options(/*name*/ "NoEventGarbageCollector",
                           /*num_threads*/ 1,
                           /*allow_spinning*/ true,
                           /*track_task*/ false);
  queue_ = CreateSingleThreadedWorkQueue(options);
}

InterpreterCoreNoEventGarbageCollector::
    ~InterpreterCoreNoEventGarbageCollector() {
  queue_.reset(nullptr);
}

void InterpreterCoreNoEventGarbageCollector::Add(Variable* var,
                                                 const Instruction& instr) {
  Add(var, &instr.DeviceContext());
}

void InterpreterCoreNoEventGarbageCollector::Add(
    Variable* var, const platform::DeviceContext* ctx) {
  if (UNLIKELY(max_memory_size_ < 0) || var == nullptr) {
    return;
  }

  if (var->IsType<LoDTensor>()) {
    Add(var->GetMutable<LoDTensor>()->MoveMemoryHolder(), ctx);
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
        ctx);
    var->GetMutable<phi::SelectedRows>()->mutable_rows()->clear();
  } else if (var->IsType<LoDTensorArray>()) {
    auto* tensor_arr = var->GetMutable<LoDTensorArray>();
    for (auto& t : *tensor_arr) {
      Add(t.MoveMemoryHolder(), ctx);
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

void InterpreterCoreNoEventGarbageCollector::Add(
    Garbage garbage, const platform::DeviceContext* ctx) {
  if (!garbage) {
    return;
  }
  if (max_memory_size_ <= 1) {
    queue_->AddTask([container = garbage, ctx = ctx]() { ctx->Wait(); });
  } else {
    // lock guard
    std::lock_guard<memory::SpinLock> guard(spinlock_);
    cur_memory_size_ += garbage->size();
    garbages_->emplace_back(std::move(garbage));
    ctxs_.insert(ctx);

    if (cur_memory_size_ >= max_memory_size_) {
      cur_memory_size_ = 0;
      queue_->AddTask(
          [container = std::move(*garbages_), dev_ctxs = std::move(ctxs_)]() {
            for (auto& ctx : dev_ctxs) {
              ctx->Wait();
            }
          });
      ctxs_.clear();
      garbages_->clear();
    }
  }
}

}  // namespace framework
}  // namespace paddle
