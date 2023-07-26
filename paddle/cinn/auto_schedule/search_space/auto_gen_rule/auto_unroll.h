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

#pragma once

#include <string>
#include <vector>

#include "paddle/cinn/auto_schedule/search_space/auto_gen_rule/auto_gen_rule.h"
#include "paddle/cinn/ir/ir.h"
#include "paddle/cinn/ir/schedule/ir_schedule.h"

namespace cinn {
namespace auto_schedule {

// This rule can be applied in a ScheduleBlock has reduce axis or has loops with
// non-serial type. As a result, it will set a attribute with key named
// ir::attr::auto_unroll_max_step and value indicating max permitted unrolled
// step in the applied ScheduleBlock. Finally, UnrollLoop pass will do unroll
// based on actual situation.
class AutoUnroll : public AutoGenRule {
 public:
  explicit AutoUnroll(const common::Target& target) : AutoGenRule(target) {}
  ~AutoUnroll() = default;

  RuleApplyType Init(ir::IRSchedule* init_schedule) override;

  void Apply(int index) override;

  std::string GetRuleName() const override { return "AutoUnroll"; }

  RuleApplyType AnalyseApplyType(SearchState state,
                                 const std::string& block_name) const override;

  std::vector<SearchState> ApplyOnBlock(SearchState state,
                                        const std::string& block_name) override;

 private:
  bool MeetCondition(const ir::ScheduleBlock* schedule_block) const;

 private:
  std::vector<Expr> applicable_schedule_blocks_;
};

}  // namespace auto_schedule
}  // namespace cinn
