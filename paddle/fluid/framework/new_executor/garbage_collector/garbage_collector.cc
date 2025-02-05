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

namespace paddle::framework {

InterpreterCoreGarbageCollector::InterpreterCoreGarbageCollector()
    : garbages_(std::make_unique<GarbageQueue>()) {
  max_memory_size_ = static_cast<int64_t>(GetEagerDeletionThreshold());
  cur_memory_size_ = 0;
}

std::unique_ptr<InterpreterCoreGarbageCollector>
CreateInterpreterCoreGarbageCollector(
    const phi::Place& place,
    const std::vector<std::unique_ptr<InstructionBase>>& vec_instruction) {
  if (phi::is_gpu_place(place)) {
    if (IsInterpretercoreFastGCEnabled()) {  // NOLINT
      return std::unique_ptr<InterpreterCoreGarbageCollector>(
          new InterpreterCoreFastGarbageCollector());
    } else {
      return std::unique_ptr<InterpreterCoreGarbageCollector>(
          new InterpreterCoreEventGarbageCollector(vec_instruction));
    }
  } else if (phi::is_xpu_place(place)) {  // NOLINT
    // Because there is no multi-stream on XPU device, fast GC can
    // be used.
    // Previously, XPU used no_event GC. But `Wait` in no_event GC
    // may cause GC delayed, causing no enough memory problem.
    // TODO(pangyoki): Multi-stream allocator and multi-stream GC
    // are needed to be adapted for XPU.
    return std::unique_ptr<InterpreterCoreGarbageCollector>(
        new InterpreterCoreFastGarbageCollector());
  } else if (phi::is_ipu_place(place)) {
    return std::unique_ptr<InterpreterCoreGarbageCollector>(
        new InterpreterCoreNoEventGarbageCollector());
  } else {
    return std::unique_ptr<InterpreterCoreGarbageCollector>(
        new InterpreterCoreEventGarbageCollector(vec_instruction));
  }
}

std::unique_ptr<InterpreterCoreGarbageCollector>
CreateInterpreterCoreGarbageCollector(
    const phi::Place& place, const std::vector<Instruction>& vec_instruction) {
  if (phi::is_gpu_place(place)) {
    if (IsInterpretercoreFastGCEnabled()) {  // NOLINT
      return std::unique_ptr<InterpreterCoreGarbageCollector>(
          new InterpreterCoreFastGarbageCollector());
    } else {
      return std::unique_ptr<InterpreterCoreGarbageCollector>(
          new InterpreterCoreEventGarbageCollector(vec_instruction));
    }
  } else if (phi::is_xpu_place(place)) {  // NOLINT
    // Because there is no multi-stream on XPU device, fast GC can
    // be used.
    // Previously, XPU used no_event GC. But `Wait` in no_event GC
    // may cause GC delayed, causing no enough memory problem.
    // TODO(pangyoki): Multi-stream allocator and multi-stream GC
    // are needed to be adapted for XPU.
    return std::unique_ptr<InterpreterCoreGarbageCollector>(
        new InterpreterCoreFastGarbageCollector());
  } else if (phi::is_ipu_place(place)) {
    return std::unique_ptr<InterpreterCoreGarbageCollector>(
        new InterpreterCoreNoEventGarbageCollector());
  } else {
    return std::unique_ptr<InterpreterCoreGarbageCollector>(
        new InterpreterCoreEventGarbageCollector(vec_instruction));
  }
}

}  // namespace paddle::framework
