// Copyright (c) 2022 CINN Authors. All Rights Reserved.
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

#include "paddle/cinn/auto_schedule/task_scheduler/task_scheduler.h"

#include <algorithm>

#include "paddle/cinn/auto_schedule/task/tune_task.h"
#include "paddle/cinn/auto_schedule/task_scheduler/efficiency_priority.h"
#include "paddle/cinn/auto_schedule/task_scheduler/round_robin.h"
#include "paddle/common/enforce.h"
namespace cinn {
namespace auto_schedule {

std::unique_ptr<TaskScheduler> TaskScheduler::Make(
    const std::vector<TuneTask>& tasks,
    const Config& config,
    const std::string& strategy) {
  PADDLE_ENFORCE_GT(
      tasks.size(),
      0,
      phi::errors::InvalidArgument("The task's size should greater than 0."));
  if (strategy == "round_robin") {
    return std::make_unique<RoundRobin>(tasks, config);
  } else if (strategy == "efficiency_priority") {
    return std::make_unique<EfficiencyPriority>(tasks, config);
  }

  std::stringstream ss;
  ss << "Unimplemented strategy:" << strategy;
  PADDLE_THROW(phi::errors::Unimplemented(ss.str()));
  return nullptr;
}

TaskScheduler::TaskScheduler(const std::vector<TuneTask>& tasks,
                             const Config& config)
    : tasks_(&tasks), config_(config), cur_task_id_(0) {}

void TaskScheduler::Reset() { cur_task_id_ = 0; }

}  // namespace auto_schedule
}  // namespace cinn
