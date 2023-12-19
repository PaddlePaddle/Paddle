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

#pragma once
#include "paddle/cinn/ir/schedule/ir_schedule.h"

namespace cinn {
namespace auto_schedule {

/**
 * Base class for rules of post process,
 * used to process schedules that rely on mutate results.
 */
class PostScheduleRule {
 public:
  PostScheduleRule() = default;

  /**
   * @brief Apply the post schedule rule to the given SearchState.
   * @param state The given SearchState for post schedule.
   * @return True if apply successfully.
   */
  virtual bool Apply(ir::IRSchedule* schedule) = 0;
};

}  // namespace auto_schedule
}  // namespace cinn
