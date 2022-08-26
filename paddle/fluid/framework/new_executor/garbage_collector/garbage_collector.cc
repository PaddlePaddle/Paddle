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

namespace paddle {
namespace framework {

bool IsInterpretercoreFastGCEnabled() {
  return memory::allocation::AllocatorFacade::Instance()
             .IsStreamSafeCUDAAllocatorUsed() &&
         FLAGS_fast_eager_deletion_mode;
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
  } else if (platform::is_xpu_place(place) || platform::is_ipu_place(place)) {
    return std::unique_ptr<InterpreterCoreGarbageCollector>(
        new InterpreterCoreNoEventGarbageCollector());
  } else {
    return std::unique_ptr<InterpreterCoreGarbageCollector>(
        new InterpreterCoreEventGarbageCollector(vec_instruction));
  }
}

}  // namespace framework
}  // namespace paddle
