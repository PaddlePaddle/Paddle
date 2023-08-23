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

#include "paddle/cinn/auto_schedule/search_space/auto_gen_rule/auto_gen_rule.h"

#include <glog/logging.h>

#include <cstdlib>

#include "paddle/cinn/common/target.h"
#include "paddle/cinn/ir/schedule/ir_schedule.h"

namespace cinn {
namespace auto_schedule {

AutoGenRule::AutoGenRule(const common::Target& target) : target_(&target) {}

int AutoGenRule::NumberApplicable() const {
  CHECK_GE(num_applicable_, 0)
      << "Call " << GetRuleName()
      << "::NumberApplicable() without initialization.";
  return num_applicable_;
}

void AutoGenRule::ApplyRandomly() {
  CHECK_GT(num_applicable_, 0)
      << "Call " << GetRuleName()
      << "::ApplyRandomly() with NumberApplicable() == 0";
  int index = rand() % num_applicable_;  // NOLINT
  return Apply(index);
}

}  // namespace auto_schedule
}  // namespace cinn
