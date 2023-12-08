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

#include <string>
#include <vector>

#include "paddle/cinn/auto_schedule/search_space/auto_gen_rule/auto_gen_rule.h"
#include "paddle/cinn/ir/ir.h"
#include "paddle/cinn/ir/schedule/ir_schedule.h"

namespace cinn {
namespace auto_schedule {

class ReductionFactoring : public AutoGenRule {
 public:
  explicit ReductionFactoring(const cinn::common::Target& target)
      : AutoGenRule(target) {}
  ~ReductionFactoring() = default;

  // In the future, we will no longer use this interface.
  RuleApplyType Init(ir::IRSchedule* init_schedule) override {
    return RuleApplyType::kCannotApply;
  }
  // In the future, we will no longer use this interface.
  void Apply(int index) override {
    LOG(FATAL) << "This is a deprecated interface, please do not use it.";
    return;
  }

  RuleApplyType AnalyseApplyType(SearchState state,
                                 const std::string& block_name) const override;

  std::string GetRuleName() const override { return "ReductionFactoring"; }

  std::vector<SearchState> ApplyOnBlock(SearchState state,
                                        const std::string& block_name) override;

  void Apply(const std::string& block_name, ir::IRSchedule* ir_schedule);

 private:
  bool CanApply(const std::string& block_name,
                ir::IRSchedule* ir_schedule) const;
};

}  // namespace auto_schedule
}  // namespace cinn
