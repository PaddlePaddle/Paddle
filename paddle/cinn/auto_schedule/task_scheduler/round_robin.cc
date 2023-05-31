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

#include "cinn/auto_schedule/task_scheduler/round_robin.h"

namespace cinn {
namespace auto_schedule {

int RoundRobin::NextTaskId() {
  if (cur_task_id_ < tasks_->size()) {
    return cur_task_id_++;
  }
  return -1;
}

}  // namespace auto_schedule
}  // namespace cinn
