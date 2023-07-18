// Copyright (c) 2023 CINN Authors. All Rights Reserved.
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

#include "paddle/cinn/auto_schedule/post_schedule_rule/cooperative_process.h"

#include "paddle/cinn/ir/ir.h"
#include "paddle/cinn/ir/schedule/ir_schedule.h"
#include "paddle/cinn/ir/schedule/schedule_desc.h"
#include "paddle/cinn/ir/utils/ir_printer.h"

namespace cinn {
namespace auto_schedule {

int ExtractNumThreads(const ir::IRSchedule& ir_schedule,
                      const std::string& bind_axis) {
  const ir::ScheduleDesc& trace = ir_schedule.GetTraceDesc();
  for (auto&& step : trace.Steps()) {
    if (step.type == "Bind" &&
        step.attrs.find("thread_axis") != step.attrs.end() &&
        absl::get<std::string>(step.attrs.at("thread_axis")) == bind_axis) {
      CHECK_EQ(step.inputs.at("loop").size(), 1);
      return step.inputs.at("loop")[0].As<ir::For>()->extent.as_int32();
    }
  }
  return 0;
}

std::vector<std::string> FindCandidates(const ir::ScheduleDesc& trace) {
  std::vector<std::string> candidate_block_names;
  for (auto&& step : trace.Steps()) {
    if (step.type == "AnnotateIntAttr" &&
        absl::get<std::string>(step.attrs.at("key")) ==
            ir::attr::cooperative_process) {
      candidate_block_names.push_back(
          step.inputs.at("block")[0]
              .As<ir::ScheduleBlockRealize>()
              ->schedule_block.As<ir::ScheduleBlock>()
              ->name);
    }
  }
  return candidate_block_names;
}

bool CooperativeProcess::Apply(ir::IRSchedule* schedule) {
  int num_threads = ExtractNumThreads(*schedule, "threadIdx.x");
  const ir::ScheduleDesc& trace = schedule->GetTraceDesc();
  std::vector<std::string> candidate_block_names = FindCandidates(trace);
  for (auto&& candidate : candidate_block_names) {
    auto loop = schedule->GetLoops(candidate).back();
    if (loop.As<ir::For>()->extent.as_int32() <= num_threads) {
      schedule->Bind(loop, "threadIdx.x");
      loop = schedule->GetLoops(candidate).back();
      schedule->SyncThreads(loop);
    } else {
      auto splited_buffer_loop = schedule->Split(loop, {-1, num_threads});
      schedule->Bind(splited_buffer_loop.back(), "threadIdx.x");
      schedule->SyncThreads(splited_buffer_loop[0]);
    }
    auto block = schedule->GetBlock(candidate);
    schedule->Unannotate(block, ir::attr::cooperative_process);
  }
  return true;
}

}  // namespace auto_schedule
}  // namespace cinn
