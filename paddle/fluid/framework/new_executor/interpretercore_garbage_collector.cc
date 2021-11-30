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

DECLARE_bool(use_stream_safe_cuda_allocator);
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

InterpreterCoreGarbageCollector::~InterpreterCoreGarbageCollector() {
  queue_.reset(nullptr);
}

void InterpreterCoreGarbageCollector::Add(
    std::shared_ptr<memory::Allocation> garbage,
    paddle::platform::DeviceEvent& event, const platform::DeviceContext* ctx) {
  if (max_memory_size_ <= 1) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
    if (!FLAGS_use_stream_safe_cuda_allocator) {
      Free(garbage, event, ctx);
    }
#else
    Free(garbage, event, ctx);
#endif
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
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      if (FLAGS_use_stream_safe_cuda_allocator) {
        delete garbage_ptr;
      } else {
        Free(garbage_ptr, event, ctx);
      }
#else
      Free(garbage_ptr, event, ctx);
#endif
    }
  }
}

void InterpreterCoreGarbageCollector::Add(paddle::framework::Variable* var,
                                          paddle::platform::DeviceEvent& event,
                                          const platform::DeviceContext* ctx) {
  if (!var) {
    return;
  }

  if (var->IsType<LoDTensor>()) {
    Add(var->GetMutable<LoDTensor>()->MoveMemoryHolder(), event, ctx);
  } else if (var->IsType<
                 operators::reader::
                     OrderedMultiDeviceLoDTensorBlockingQueueHolder>()) {
    // var->Clear(); // TODO(xiongkun03) can we clear directly? Why we must use
    // Add interface?
  } else if (var->IsType<SelectedRows>()) {
    Add(var->GetMutable<SelectedRows>()->mutable_value()->MoveMemoryHolder(),
        event, ctx);
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

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
void InterpreterCoreGarbageCollector::StreamSynchronize(
    const Instruction& instr, const VariableScope& scope) {
  gpuStream_t stream = reinterpret_cast<const platform::CUDADeviceContext&>(
                           instr.DeviceContext())
                           .stream();
  auto TensorSynchronize = [&stream](Tensor& tensor) {
    const platform::Place& place = tensor.place();
    if (platform::is_gpu_place(place)) {
      tensor.RecordStream(stream);
    } else if (platform::is_cuda_pinned_place(place)) {
      // TODO(Ruibiao): Here should do something to make sure that the tensor is
      // not freed until the H2D copies done. However, simplely launch a CUDA
      // runtime callback to the H2D stream may lead a high performance
      // overhead. As all the cases we meet in H2D are copies from CPUPlace at
      // present, we just log a WARNING here. A better design is required.
      LOG(WARNING) << "Copy data from a CUDAPinned tensor in an asynchronous "
                      "manner may lead a data inconsistent";
    } else {
      // memory copies involve CPUPlace are always synchronous, so just do
      // nothing here
    }
  };

  /* NOTE(Ruibiao)：Cross-stream tensor synchronization is required only when
   * all the following conditions are satisfied:
   * 1. The tensor will be GC after running the instruction, i.e., in
   * instr.GCCheckVars.
   * 2. The stream which initializes this tensor is different from the stream
   * which the instruction run in.
   * 3. The tensor is the instruction's input, cause we assume that instruction
   * will initialize all output tensors with its running stream.
   * 4. In the OP function of this instruction, the tensor is an input of a
   * async CUDA kernel.
   *
   * Here we only process the first condition, because:
   * 1. Since the RecordStream function will directly return when the recored
   * stream is equal to the owning stream, recording a stream same as which
   * initialized this tensor has less time overhead. Conversely, it may take
   * more time if we try to extract those cross-stream input vars from
   * instr.GCCheckVars.
   * 2. Now the instruction has no idea of which vars involving async running in
   * OP function, and thus we can not recognize condition 4. It should be
   * supported later.
   */
  for (int var_id : instr.GCCheckVars()) {
    // persistable var will be ignore while GC
    if (scope.VarDesc(var_id) && scope.VarDesc(var_id)->Persistable()) {
      continue;
    }

    paddle::framework::Variable* var = scope.Var(var_id);
    if (var == nullptr) {
      continue;
    }

    if (var->IsType<LoDTensor>()) {
      TensorSynchronize(*(var->GetMutable<LoDTensor>()));
    } else if (var->IsType<
                   operators::reader::
                       OrderedMultiDeviceLoDTensorBlockingQueueHolder>()) {
      // do nothing
    } else if (var->IsType<SelectedRows>()) {
      TensorSynchronize(*(var->GetMutable<SelectedRows>()->mutable_value()));
    } else if (var->IsType<LoDTensorArray>()) {
      auto* tensor_arr = var->GetMutable<LoDTensorArray>();
      for (auto& tensor : *tensor_arr) {
        TensorSynchronize(tensor);
      }
    } else if (var->IsType<std::vector<Scope*>>()) {
      // do nothing
    } else {
      PADDLE_THROW(platform::errors::Unimplemented(
          "The variable(%s) is not supported in eager deletion.",
          framework::ToTypeName(var->Type())));
    }
  }
}
#endif

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
