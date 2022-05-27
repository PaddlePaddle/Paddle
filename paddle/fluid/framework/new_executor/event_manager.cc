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

#include "paddle/fluid/framework/new_executor/event_manager.h"
#include "paddle/fluid/platform/profiler/event_tracing.h"

namespace paddle {
namespace framework {
namespace interpreter {
void WaitEvent(const Instruction& instruction, const platform::Place& place) {
  // If InterpreterCore in on CPUPlace, do nothing.
  if (platform::is_cpu_place(place)) return;

  VLOG(3) << "Deal StreamWaitEventOrSync for " << instruction.OpBase()->Type();

  for (auto& event_iter : instruction.InputEvents()) {
    platform::RecordEvent record("WaitStreamEvent",
                                 platform::TracerEventType::UserDefined, 10);
    VLOG(3) << "wait var_id: " << event_iter.var_id_
            << " 's event with waiter_type: " << event_iter.waiter_type_;
    event_iter.event_->Wait(event_iter.waiter_type_,
                            &instruction.DeviceContext());
  }
}

void RecordEvent(const Instruction& instruction, const platform::Place& place) {
  // If InterpreterCore in on CPUPlace, do nothing.
  if (platform::is_cpu_place(place)) return;

  for (auto& event : instruction.OutputEvents()) {
    platform::RecordEvent record("RecordStreamEvent",
                                 platform::TracerEventType::UserDefined, 10);
    VLOG(3) << "Record event in out_var_id: " << event.var_id_;
    event.event_->Record(&instruction.DeviceContext());
  }
}

void RecordEvent(const Instruction& instruction) {
  // If InterpreterCore in on CPUPlace, do nothing.
  if (platform::is_cpu_place(instruction.DeviceContext().GetPlace())) return;

  for (auto& event : instruction.OutputEvents()) {
    platform::RecordEvent record("RecordStreamEvent",
                                 platform::TracerEventType::UserDefined, 10);
    VLOG(3) << "Record event in out_var_id: " << event.var_id_;
    event.event_->Record(&instruction.DeviceContext());
  }
}

}  // namespace interpreter
}  // namespace framework
}  // namespace paddle
