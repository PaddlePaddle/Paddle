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

#include <memory>
#include <random>
#include <vector>

#include "paddle/cinn/auto_schedule/search_space/auto_gen_rule/auto_gen_rule.h"
#include "paddle/cinn/utils/random_engine.h"

namespace cinn {
namespace auto_schedule {

class SearchState;

// Select the next potential rule for the SearchState during the search process.
class RuleSampler {
 public:
  /**
   * @brief Create a RuleSampler with the specific strategy name and necessary
   * construct parameters.
   * @param potential_rules All possible rules to be sampled.
   * @param default_remove_policy The default option to determine whether to
   * delete the next block after selecting it.
   * @param strategy The rule sampling strategy.
   *                 Currently, the available strategies are "traversal" and
   * "probabilistic", where "traversal" means to select rules one by one until
   * all rules are traversed, and "probabilistic" means randomly picking rules
   * according to the given distribution.
   * @param weights Used for the probabilistic policy, giving each candidate a
   * weight.
   */
  static std::unique_ptr<RuleSampler> Make(
      const std::vector<AutoGenRule*>& potential_rules,
      bool default_remove_policy = true,
      const std::string& strategy = "traversal",
      utils::LinearRandomEngine::StateType rand_seed = 0,
      const std::vector<int>& weights = {});
  // Return the name of sample strategy
  virtual const char* Name() const = 0;

  // Reset associated states to sample at the beginning
  virtual void Reset() = 0;

  // Select a rule with default remove policy.
  AutoGenRule* NextRule() { return NextRule(default_remove_policy_); }

 protected:
  // A RuleSampler object should be created with the static function Make()
  RuleSampler(const std::vector<AutoGenRule*>& potential_rules,
              bool default_remove_policy)
      : potential_rules_(&potential_rules),
        default_remove_policy_(default_remove_policy) {}

  // Select a rule to apply.
  // The param remove is used to determine whether to delete the next rule after
  // selecting it, If remove == true, it will not be sampled in the future.
  virtual AutoGenRule* NextRule(bool remove) = 0;

  // The pointer refers to all potential rules
  const std::vector<AutoGenRule*>* potential_rules_;

  // The default policy to determine whether to delete the next rule after
  // selecting it.
  bool default_remove_policy_;
};

// Sample rules with traversal strategy,
// witch means to select rules one by one until all rules are traversed.
class TraversalRuleSampler : public RuleSampler {
 public:
  TraversalRuleSampler(const std::vector<AutoGenRule*>& potential_rules,
                       bool default_remove_policy)
      : RuleSampler(potential_rules, default_remove_policy), cur_idx_(0) {}

  const char* Name() const override { return "traversal"; }

  void Reset() override { cur_idx_ = 0; }

 private:
  AutoGenRule* NextRule(bool remove) override;

 private:
  int cur_idx_;
};

// Sample rules with probabilistic strategy,
// which means randomly picking rules according to the given distribution.
class ProbabilisticRuleSampler : public RuleSampler {
 public:
  ProbabilisticRuleSampler(const std::vector<AutoGenRule*>& potential_rules,
                           bool default_remove_policy,
                           utils::LinearRandomEngine::StateType rand_seed = 0,
                           const std::vector<int>& weights = {});

  const char* Name() const override { return "probabilistic"; }

  void Reset() override {}

 private:
  AutoGenRule* NextRule(bool remove) override;

 private:
  std::vector<int> weights_;
  utils::LinearRandomEngine::StateType rand_seed_;
  int remains_;
};

}  // namespace auto_schedule
}  // namespace cinn
