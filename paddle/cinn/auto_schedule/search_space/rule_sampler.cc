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

#include <algorithm>
#include <random>

namespace cinn {
namespace auto_schedule {

std::unique_ptr<RuleSampler> RuleSampler::Make(
    const std::vector<AutoGenRule*>& potential_rules,
    bool default_remove_policy,
    const std::string& strategy,
    utils::LinearRandomEngine::StateType rand_seed,
    const std::vector<int>& weights) {
  CHECK_GT(potential_rules.size(), 0) << "Empty rule list";
  if (strategy == "traversal") {
    return std::make_unique<TraversalRuleSampler>(potential_rules,
                                                  default_remove_policy);
  } else if (strategy == "probabilistic") {
    return std::make_unique<ProbabilisticRuleSampler>(
        potential_rules, default_remove_policy, rand_seed, weights);
  }

  LOG(FATAL) << "Unimplemented strategy:" << strategy;
  return nullptr;
}

AutoGenRule* TraversalRuleSampler::NextRule(bool remove) {
  if (cur_idx_ < potential_rules_->size()) {
    AutoGenRule* rule = potential_rules_->at(cur_idx_);
    if (remove) {
      ++cur_idx_;
    }
    return rule;
  }

  return nullptr;
}

ProbabilisticRuleSampler::ProbabilisticRuleSampler(
    const std::vector<AutoGenRule*>& potential_rules,
    bool default_remove_policy,
    utils::LinearRandomEngine::StateType rand_seed,
    const std::vector<int>& weights)
    : RuleSampler(potential_rules, default_remove_policy),
      weights_(weights),
      rand_seed_(utils::LinearRandomEngine::NormalizeState(rand_seed)) {
  if (weights.empty()) {
    weights_.resize(potential_rules.size(), 1);
  } else {
    CHECK_EQ(potential_rules.size(), weights_.size());
  }
  remains_ = potential_rules.size();
}

AutoGenRule* ProbabilisticRuleSampler::NextRule(bool remove) {
  if (remains_ == 0) {
    return nullptr;
  }
  int rule_idx =
      utils::SampleDiscreteFromDistribution<int>(weights_, &rand_seed_);
  if (remove) {
    weights_[rule_idx] = 0;
    --remains_;
  }

  return potential_rules_->at(rule_idx);
}

}  // namespace auto_schedule
}  // namespace cinn
