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

#include "paddle/cinn/auto_schedule/search_space/rule_sampler.h"

#include <gtest/gtest.h>

#include "paddle/cinn/auto_schedule/search_space/auto_gen_rule/auto_unroll.h"
#include "paddle/cinn/auto_schedule/search_space/auto_gen_rule/skip_rule.h"

namespace cinn {
namespace auto_schedule {

#ifdef CINN_WITH_CUDA
Target target = common::DefaultNVGPUTarget();
#else
Target target = common::DefaultHostTarget();
#endif

std::vector<AutoGenRule*> GenerateTestRules() {
  return {new AutoUnroll(target), new SkipRule(target)};
}

TEST(RuleSampler, Make) {
  std::vector<AutoGenRule*> rules = GenerateTestRules();
  auto traversal_block_sampler = RuleSampler::Make(rules, true, "traversal");
  ASSERT_STREQ(traversal_block_sampler->Name(), "traversal");
  auto probabilistic_block_sampler =
      RuleSampler::Make(rules, true, "probabilistic");
  ASSERT_STREQ(probabilistic_block_sampler->Name(), "probabilistic");
}

TEST(TraversalRuleSampler, NextRule) {
  std::vector<AutoGenRule*> rules = GenerateTestRules();
  auto traversal_rule_sampler = RuleSampler::Make(rules, true, "traversal");
  AutoGenRule* rule = traversal_rule_sampler->NextRule();
  ASSERT_EQ("AutoUnroll", rule->GetRuleName());
  rule = traversal_rule_sampler->NextRule();
  ASSERT_EQ("SkipRule", rule->GetRuleName());
  traversal_rule_sampler->Reset();
  rule = traversal_rule_sampler->NextRule();
  ASSERT_EQ("AutoUnroll", rule->GetRuleName());

  traversal_rule_sampler = RuleSampler::Make(rules, false, "traversal");
  rule = traversal_rule_sampler->NextRule();
  ASSERT_EQ("AutoUnroll", rule->GetRuleName());
  rule = traversal_rule_sampler->NextRule();
  ASSERT_EQ("AutoUnroll", rule->GetRuleName());
}

TEST(ProbabilisticRuleSampler, NextRule) {
  std::vector<AutoGenRule*> rules = GenerateTestRules();
  auto probabilistic_rule_sampler =
      RuleSampler::Make(rules, false, "probabilistic", 0, {4, 1});
  AutoGenRule* rule;
  for (int i = 0; i < 20; ++i) {
    rule = probabilistic_rule_sampler->NextRule();
    VLOG(6) << "next rule name: " << rule->GetRuleName();
  }

  probabilistic_rule_sampler =
      RuleSampler::Make(rules, true, "probabilistic", 0, {4, 1});
  probabilistic_rule_sampler->NextRule();
  probabilistic_rule_sampler->NextRule();
  ASSERT_EQ(nullptr, probabilistic_rule_sampler->NextRule());
}

}  // namespace auto_schedule
}  // namespace cinn
