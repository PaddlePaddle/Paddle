// Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/framework/new_executor/garbage_collector/async_fast_garbage_collector.h"
#include "paddle/phi/api/profiler/event_tracing.h"
#include "paddle/phi/core/threadpool.h"

namespace paddle {
namespace framework {

SingleThreadLockFreeWorker::SingleThreadLockFreeWorker(int capacity)
    : capacity_(capacity), head_(0), tail_(0), running_(true) {
  tasks_queue_.resize(capacity);
  worker_ = std::thread([this]() { this->WorkerLoop(); });
}

void SingleThreadLockFreeWorker::AddTask(Task task) {
  tasks_queue_[tail_] = task;
  tail_++;
  if (tail_ >= tasks_queue_.size()) {
    tasks_queue_.resize(tasks_queue_.size() + capacity_);
  }
}

void SingleThreadLockFreeWorker::Wait() {
  running_ = false;
  if (worker_.joinable()) worker_.join();
}

void SingleThreadLockFreeWorker::WorkerLoop() {
  while (true) {
    if (head_ < tail_) {
      Task task = tasks_queue_[head_];
      task();
      head_++;
    } else if (head_ == tail_ && running_) {
      std::this_thread::yield();
    } else {
      break;
    }
  }
}

InterpreterCoreAsyncFastGarbageCollector::
    InterpreterCoreAsyncFastGarbageCollector(int num_instructions) {
  async_worker_ =
      std::make_unique<SingleThreadLockFreeWorker>(num_instructions);
}

void FreeVariable(Variable* var) {
  if (var == nullptr) {
    return;
  }

  if (var->IsType<phi::DenseTensor>()) {
    Garbage garbage = var->GetMutable<phi::DenseTensor>()->MoveMemoryHolder();
  } else if (
      var->IsType<
          operators::reader::
              OrderedMultiDeviceDenseTensorBlockingQueueHolder>()) {  // NOLINT
    // TODO(xiongkun03) in old executor, this type of variable is not support
    // eager deletion. so we just leave it here ?
  } else if (var->IsType<phi::SelectedRows>()) {
    Garbage garbage = var->GetMutable<phi::SelectedRows>()
                          ->mutable_value()
                          ->MoveMemoryHolder();
    var->GetMutable<phi::SelectedRows>()->mutable_rows()->clear();
  } else if (var->IsType<phi::TensorArray>()) {
    auto* tensor_arr = var->GetMutable<phi::TensorArray>();
    for (auto& t : *tensor_arr) {
      Garbage garbage = t.MoveMemoryHolder();
    }
    tensor_arr->clear();
  } else if (var->IsType<phi::SparseCooTensor>()) {
    Garbage indices = var->GetMutable<phi::SparseCooTensor>()
                          ->mutable_indices()
                          ->MoveMemoryHolder();
    Garbage values = var->GetMutable<phi::SparseCooTensor>()
                         ->mutable_values()
                         ->MoveMemoryHolder();
  } else if (var->IsType<phi::SparseCsrTensor>()) {
    Garbage cols = var->GetMutable<phi::SparseCsrTensor>()
                       ->mutable_cols()
                       ->MoveMemoryHolder();
    Garbage crows = var->GetMutable<phi::SparseCsrTensor>()
                        ->mutable_crows()
                        ->MoveMemoryHolder();
    Garbage values = var->GetMutable<phi::SparseCsrTensor>()
                         ->mutable_values()
                         ->MoveMemoryHolder();
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

void InterpreterCoreAsyncFastGarbageCollector::Add(
    const std::vector<Variable*>& vars) {
  async_worker_->AddTask([vars]() {
    for (const auto& var : vars) {
      FreeVariable(var);
    }
  });
}

}  // namespace framework
}  // namespace paddle
