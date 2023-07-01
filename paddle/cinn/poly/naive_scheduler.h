// Copyright (c) 2021 CINN Authors. All Rights Reserved.
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

#include <map>
#include <string>
#include <vector>

#include "paddle/cinn/poly/schedule.h"

namespace cinn {
namespace poly {

class NaiveGroupScheduler : public SchedulerBase {
 public:
  //! Constructor, for naive scheduler, each group has just one node.
  explicit NaiveGroupScheduler(Stage *x) {
    AddStage(*x);
    FinishStageAdd();
  }
  //! Just one node, need no schedule.
  void Build() {}
};

/**
 * The NaiveScheduler just schedule each noninlined Tensor as a unique group.
 * Only the `compute_at` will merge two tensor in the same group. It is simple
 * and robust.
 */
class NaiveScheduler : public SchedulerBase {
 public:
  NaiveScheduler() = default;
  explicit NaiveScheduler(const std::vector<Stage *> &stages) {
    for (auto *x : stages) AddStage(*x);
    FinishStageAdd();
  }

  std::unique_ptr<Schedule> BuildSchedule();

 private:
  void PartitionGroups();

 private:
  std::vector<ScheduleGroup> groups_;
};

}  // namespace poly
}  // namespace cinn
