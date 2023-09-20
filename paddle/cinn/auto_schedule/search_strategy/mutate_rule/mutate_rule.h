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

#include "paddle/cinn/ir/schedule/schedule_desc.h"
#include "paddle/cinn/utils/random_engine.h"

namespace cinn {
namespace auto_schedule {

/**
 * Base class for rules of mutate,
 * is used for mutating the trace(ScheduleDesc) to explore the search space.
 */
class MutateRule {
 public:
  MutateRule() = default;

  /**
   * @brief Apply the mutate rule to the given trace.
   * @param trace The given trace for mutation.
   * @param rand_seed The random seed for mutation.
   * @return The mutated trace.
   */
  virtual ir::ScheduleDesc Apply(
      const ir::ScheduleDesc& trace,
      utils::LinearRandomEngine::StateType* rand_seed) = 0;

  /**
   * @brief Create a MutateRule with name.
   * @param name The name of mutate rule, consisting of lowercase letters and
   * underscores
   * @return The created MutateRule.
   */
  static std::unique_ptr<MutateRule> Make(const std::string& name);
};

}  // namespace auto_schedule
}  // namespace cinn
