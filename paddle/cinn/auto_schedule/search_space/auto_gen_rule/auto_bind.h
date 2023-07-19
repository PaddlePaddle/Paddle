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

// Auto bind GPU index(BlockIdx, ThreadIdx) to the loops around the block
class AutoBind : public AutoGenRule {
 public:
  explicit AutoBind(const common::Target& target) : AutoGenRule(target) {}
  ~AutoBind() = default;

  RuleApplyType Init(ir::IRSchedule* init_schedule) override;

  void Apply(int index) override;

  std::string GetRuleName() const override { return "AutoBind"; }

  RuleApplyType AnalyseApplyType(SearchState state,
                                 const std::string& block_name) const override;

  std::vector<SearchState> ApplyOnBlock(SearchState state,
                                        const std::string& block_name) override;

 private:
  std::vector<Expr> applicable_schedule_blocks_;
};

}  // namespace auto_schedule
}  // namespace cinn
