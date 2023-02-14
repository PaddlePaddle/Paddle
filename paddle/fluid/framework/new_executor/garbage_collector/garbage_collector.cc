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

#include "paddle/fluid/framework/new_executor/garbage_collector/garbage_collector.h"
#include "paddle/fluid/framework/garbage_collector.h"
#include "paddle/fluid/framework/new_executor/garbage_collector/event_garbage_collector.h"
#include "paddle/fluid/framework/new_executor/garbage_collector/fast_garbage_collector.h"
#include "paddle/fluid/framework/new_executor/garbage_collector/no_event_garbage_collector.h"

DECLARE_bool(fast_eager_deletion_mode);
DECLARE_bool(new_executor_use_cuda_graph);

namespace paddle {
namespace framework {

bool IsInterpretercoreFastGCEnabled() {
  // When using cuda graph, fast GC must be used. Because
  // `EventQuery` method in event GC cannot be used in
  // cuda graph.
  PADDLE_ENFORCE_EQ(memory::allocation::AllocatorFacade::Instance()
                                .IsStreamSafeCUDAAllocatorUsed() == false &&
                        FLAGS_new_executor_use_cuda_graph,
                    false,
                    platform::errors::InvalidArgument(
                        "When FLAGS_new_executor_use_cuda_graph is true, "
                        "IsStreamSafeCUDAAllocatorUsed must be true, but "
                        "got false."));
  return (memory::allocation::AllocatorFacade::Instance()
              .IsStreamSafeCUDAAllocatorUsed() &&
          FLAGS_fast_eager_deletion_mode) ||
         FLAGS_new_executor_use_cuda_graph;
}

InterpreterCoreGarbageCollector::InterpreterCoreGarbageCollector() {
  garbages_ = std::make_unique<GarbageQueue>();
  max_memory_size_ = static_cast<int64_t>(GetEagerDeletionThreshold());
  cur_memory_size_ = 0;
}

std::unique_ptr<InterpreterCoreGarbageCollector>
CreateInterpreterCoreGarbageCollector(
    const platform::Place& place,
    const std::vector<Instruction>& vec_instruction) {
  if (platform::is_gpu_place(place)) {
    if (IsInterpretercoreFastGCEnabled()) {
      return std::unique_ptr<InterpreterCoreGarbageCollector>(
          new InterpreterCoreFastGarbageCollector());
    } else {
      return std::unique_ptr<InterpreterCoreGarbageCollector>(
          new InterpreterCoreEventGarbageCollector(vec_instruction));
    }
  } else if (platform::is_xpu_place(place)) {
    // Because there is no multi-stream on XPU device, fast GC can
    // be used.
    // Previously, XPU used no_event GC. But `Wait` in no_event GC
    // may cause GC delayed, causing no enough memory problem.
    // TODO(pangyoki): Multi-stream allocator and multi-stream GC
    // are needed to be adapted for XPU.
    return std::unique_ptr<InterpreterCoreGarbageCollector>(
        new InterpreterCoreFastGarbageCollector());
  } else if (platform::is_ipu_place(place)) {
    return std::unique_ptr<InterpreterCoreGarbageCollector>(
        new InterpreterCoreNoEventGarbageCollector());
  } else {
    return std::unique_ptr<InterpreterCoreGarbageCollector>(
        new InterpreterCoreEventGarbageCollector(vec_instruction));
  }
}

}  // namespace framework
}  // namespace paddle
