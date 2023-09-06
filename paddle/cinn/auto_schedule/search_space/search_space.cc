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

#include "paddle/cinn/auto_schedule/search_space/search_space.h"

#include <glog/logging.h>

#include <cstdlib>
#include <utility>
#include <vector>

#include "paddle/cinn/auto_schedule/cost_model/expr_cost_model.h"
#include "paddle/cinn/auto_schedule/search_space/auto_gen_rule/auto_gen_rule.h"
#include "paddle/cinn/auto_schedule/search_space/auto_gen_rule/auto_inline.h"
#include "paddle/cinn/auto_schedule/search_space/auto_gen_rule/auto_unroll.h"
#include "paddle/cinn/auto_schedule/search_space/auto_gen_rule/multi_level_tiling.h"
#include "paddle/cinn/auto_schedule/search_space/auto_gen_rule/skip_rule.h"
#include "paddle/cinn/auto_schedule/search_space/block_sampler.h"
#include "paddle/cinn/auto_schedule/search_space/rule_sampler.h"
#include "paddle/cinn/auto_schedule/task/tune_task.h"
#include "paddle/cinn/ir/ir_base.h"
#include "paddle/cinn/ir/schedule/ir_schedule.h"
#include "paddle/cinn/ir/utils/ir_copy.h"
#include "paddle/cinn/runtime/flags.h"

PD_DECLARE_bool(auto_schedule_use_cost_model);

namespace cinn {
namespace auto_schedule {

SearchSpace::SearchSpace(const TuneTask& tune_task,
                         utils::LinearRandomEngine::StateType rand_seed)
    : tune_task_(tune_task),
      rand_seed_(utils::LinearRandomEngine::NormalizeState(rand_seed)) {
  const auto& target = tune_task_.target;
  // initialize a set of rules and they are commonly used by all states
  // TODO(zhhsplendid): pass correct output names to AutoInline
  // sketch_rules_.emplace_back(new AutoInline(target,
  // tune_task_.output_names));
  sketch_rules_.emplace_back(
      new MultiLevelTiling(target, MultiLevelTiling::kConfigs.at(target.arch)));
  sketch_rules_.emplace_back(new AutoUnroll(target));
  sketch_rules_.emplace_back(new SkipRule(target));
}

SearchState SearchSpace::GetScheduleMutate(const SearchState& state,
                                           const ExprCostModel& cost_model) {
  bool has_manual_schedule = false;
  if (has_manual_schedule) {
    SearchState ret = ManualScheduleMutate(state);
    return ret;
  }
  SearchState ret = RandomScheduleMutate(state);
  if (FLAGS_auto_schedule_use_cost_model) {
    ret->predicted_cost =
        cost_model.Predict(ret->ir_schedule.GetModule(), tune_task_.target);
  }
  VLOG(4) << JoinStatesDebugString(
      "SearchSpace::GetScheduleMutate", {state}, /*verbose=*/VLOG_IS_ON(5));
  return ret;
}

SearchState SearchSpace::ManualScheduleMutate(const SearchState& state) {
  // TODO(zhhsplendid): Add manual schedule mutate
  return state;
}

SearchState SearchSpace::RandomScheduleMutate(const SearchState& state) {
  // 1. Found the schedules which can apply on this Expr
  // 2. Make a distribution on those schedules
  std::map<int, int> weight_to_rule_index;
  int cur_weight = 0;
  SearchState ret(state);
  std::vector<RuleApplyType> apply_types(ret->applicable_rules.size());
  for (int idx = 0; idx != ret->applicable_rules.size(); ++idx) {
    AutoGenRule* rule = ret->applicable_rules.at(idx);
    RuleApplyType apply_type = rule->Init(&ret->ir_schedule);
    VLOG(6) << "Evaluate rule:" << rule->GetRuleName() << "="
            << static_cast<int>(apply_type);
    apply_types[idx] = apply_type;
    if (apply_type != RuleApplyType::kCannotApply) {
      weight_to_rule_index[cur_weight] = idx;
      cur_weight += rule->NumberApplicable();
    }
  }

  if (weight_to_rule_index.empty()) {
    // No applicable rule, return the input mod_expr
    VLOG(6) << "No applicable rule";
    return ret;
  }

  // 3. Sample a schedule on the distribution
  int sample_weighted_index =
      utils::SampleUniformInt(0, cur_weight, &rand_seed_);

  auto iter = weight_to_rule_index.upper_bound(sample_weighted_index);
  --iter;

  int sample_rule_index = iter->second;
  CHECK_LT(sample_rule_index, ret->applicable_rules.size());
  AutoGenRule* sample_rule = ret->applicable_rules.at(sample_rule_index);
  VLOG(7) << "Apply rule: " << sample_rule->GetRuleName()
          << " with index=" << sample_weighted_index - iter->first;
  // 4. Apply the schedule change
  sample_rule->Apply(sample_weighted_index - iter->first);

  // 5. Remove the rule after applying it
  if (apply_types.at(sample_rule_index) != RuleApplyType::kCannotApply) {
    ret->applicable_rules.erase(ret->applicable_rules.begin() +
                                sample_rule_index);
  }

  return ret;
}

std::vector<SearchState> SearchSpace::InitSketchWithRandomStrategy(int num) {
  VLOG(5) << "SearchSpace::GetRandomInitialSketch with num=" << num;
  ir::IRSchedule init_schedule(
      ir::ModuleExpr(tune_task_.GetLoweredFuncBodyExprs()),
      utils::ForkRandomState(&rand_seed_));
  std::vector<AutoGenRule*> init_rules;
  std::transform(sketch_rules_.begin(),
                 sketch_rules_.end(),
                 std::back_inserter(init_rules),
                 [](const auto& rule) { return rule.get(); });
  std::vector<SearchState> result;
  while (result.size() < num) {
    SearchState state(init_schedule, SearchState::NOT_INIT_COST, init_rules);
    for (int i = 0; i < init_sketch_random_depth_; ++i) {
      VLOG(6) << "Generating random sketch with RandomScheduleMutate at depth: "
              << i;
      state = RandomScheduleMutate(state);
      if (state->applicable_rules.empty()) {
        break;
      }
    }

    VLOG(5) << JoinStatesDebugString(
        "SearchSpace::GetRandomInitialSketch-New_Sketch",
        {state},
        /*verbose=*/VLOG_IS_ON(6));
    result.emplace_back(std::move(state));
  }
  return result;
}

std::vector<SearchState> SearchSpace::InitSketchWithRandomPrunedStrategy() {
  VLOG(5) << "SearchSpace::InitSketchWithRandomPrunedStrategy";
  ir::IRSchedule init_schedule(
      ir::ModuleExpr(tune_task_.GetLoweredFuncBodyExprs()),
      utils::ForkRandomState(&rand_seed_));
  auto all_blocks = init_schedule.GetAllBlocks();
  auto block_sampler = BlockSampler::Make(
      all_blocks, true, "probabilistic", utils::ForkRandomState(&rand_seed_));

  std::vector<AutoGenRule*> init_rules;
  std::transform(sketch_rules_.begin(),
                 sketch_rules_.end() - 1,
                 std::back_inserter(init_rules),
                 [](const auto& rule) { return rule.get(); });
  CHECK(init_rules.size() > 0) << "number of init rules cannot be 0";

  SearchState init_state(init_schedule, SearchState::NOT_INIT_COST, {});
  std::vector<SearchState> states_buf1{init_state}, states_buf2;
  std::vector<SearchState>* p_states_cur = &states_buf1;
  std::vector<SearchState>* p_states_next = &states_buf2;
  int total_steps = 0, steps;
  std::string block_name;
  while ("" != (block_name = block_sampler->NextBlock()) &&
         total_steps < init_sketch_random_depth_) {
    steps = utils::SampleUniformInt(1, init_rules.size() + 1, &rand_seed_);
    if (total_steps + steps > init_sketch_random_depth_) {
      steps = init_sketch_random_depth_ - total_steps;
    }
    total_steps += steps;
    p_states_next->clear();
    for (const auto& state : *p_states_cur) {
      auto rule_sampler =
          RuleSampler::Make(init_rules,
                            true,
                            "probabilistic",
                            utils::ForkRandomState(&rand_seed_));
      auto new_states = ApplySketchRule(
          state, block_name, rule_sampler.get(), steps, false, 1);
      p_states_next->insert(
          p_states_next->end(), new_states.begin(), new_states.end());
    }
    std::swap(p_states_cur, p_states_next);
  }
  VLOG(5) << JoinStatesDebugString(
      "SearchSpace::InitSketchWithRandomPrunedStrategy",
      *p_states_cur,
      /*verbose=*/VLOG_IS_ON(6));
  return *p_states_cur;
}

std::vector<SearchState> SearchSpace::InitSketchWithRulePrunedStrategy() {
  VLOG(5) << "SearchSpace::InitSketchWithRulePrunedStrategy";
  ir::IRSchedule init_schedule(
      ir::ModuleExpr(tune_task_.GetLoweredFuncBodyExprs()),
      utils::ForkRandomState(&rand_seed_));
  auto all_blocks = init_schedule.GetAllBlocks();
  std::reverse(all_blocks.begin(), all_blocks.end());
  auto block_sampler = BlockSampler::Make(all_blocks, true, "traversal");

  std::vector<AutoGenRule*> init_rules;
  std::transform(sketch_rules_.begin(),
                 sketch_rules_.end() - 1,
                 std::back_inserter(init_rules),
                 [](const auto& rule) { return rule.get(); });
  CHECK(init_rules.size() > 0) << "number of init rules cannot be 0";

  SearchState init_state(init_schedule, SearchState::NOT_INIT_COST, {});
  std::vector<SearchState> states_buf1{init_state}, states_buf2;
  std::vector<SearchState>* p_states_cur = &states_buf1;
  std::vector<SearchState>* p_states_next = &states_buf2;
  std::string block_name;
  while ("" != (block_name = block_sampler->NextBlock())) {
    p_states_next->clear();
    for (const auto& state : *p_states_cur) {
      auto rule_sampler = RuleSampler::Make(init_rules, true, "traversal");
      auto new_states =
          ApplySketchRule(state, block_name, rule_sampler.get(), 0, true);
      p_states_next->insert(
          p_states_next->end(), new_states.begin(), new_states.end());
    }
    std::swap(p_states_cur, p_states_next);
  }
  VLOG(5) << JoinStatesDebugString(
      "SearchSpace::InitSketchWithRulePrunedStrategy",
      *p_states_cur,
      /*verbose=*/VLOG_IS_ON(6));
  return *p_states_cur;
}

std::vector<SearchState> SearchSpace::GenerateSketches(
    int num, const std::string& strategy) {
  VLOG(4) << "SearchSpace::GenerateSketches with num = " << num;

  if (strategy == "random") {
    return InitSketchWithRandomStrategy(num);
  }

  std::vector<SearchState> result;
  while (result.size() < num) {
    std::vector<SearchState> sketchs;
    if (strategy == "rule_prune") {
      sketchs = InitSketchWithRulePrunedStrategy();
    } else if (strategy == "random_prune") {
      sketchs = InitSketchWithRandomPrunedStrategy();
    } else {
      LOG(FATAL) << "Unimplemented init sketch strategy";
    }

    // the more rules are applied, the greater the possibility of good results,
    // the more rules are applied, the more they are saved behind the queue,
    // so we give priority to the results in the rear
    for (auto iter = sketchs.rbegin(); iter != sketchs.rend(); ++iter) {
      result.push_back(*iter);
      if (result.size() == num) {
        break;
      }
    }
  }
  VLOG(4) << JoinStatesDebugString(
      "SearchSpace::GenerateSketches", result, /*verbose=*/VLOG_IS_ON(5));
  return result;
}

std::vector<SearchState> SearchSpace::ApplySketchRule(
    const SearchState& state,
    const std::string& block_name,
    RuleSampler* rule_sampler,
    int steps,
    bool prune_by_rule,
    double prune_probability) {
  std::list<SearchState> layer{state};
  int step = 0;
  AutoGenRule* rule;
  // After determining a SearchState and a block, each rule has two
  // possibilities: apply and not apply. In all transfer spaces, select a rule
  // at each step, and collect all possible new states arrived by apply and not
  // apply. This forms a tree, and we can use rule pruning or random pruning to
  // reduce the number of sketches.
  VLOG(6) << "Collect the states of all transfers within steps: " << steps;
  while ((step++ < steps || steps == 0) && (rule = rule_sampler->NextRule())) {
    VLOG(7) << "step = " << step << ", rule: " << rule->GetRuleName();
    std::list<SearchState> new_states;
    int id = 0;
    for (std::list<SearchState>::iterator iter = layer.begin();
         iter != layer.end();) {
      // Some rules will reduce the number of blocks, such as AutoInline,
      // so we need to check whether the SearchState still has the block.
      if (!(*iter)->ir_schedule.HasBlock(block_name)) {
        ++iter;
        continue;
      }
      auto type = rule->AnalyseApplyType(*iter, block_name);
      VLOG(7)
          << "At SearchState " << ++id << ", apply type = "
          << static_cast<typename std::underlying_type<RuleApplyType>::type>(
                 type);
      // if cannot apply the rule, skip it
      if (type == RuleApplyType::kCannotApply) {
        ++iter;
        continue;
      }
      // if can apply the rule, apply it and determine whether to prune the
      // branch that do not apply
      std::vector<SearchState> tmp_states =
          rule->ApplyOnBlock(*iter, block_name);
      new_states.insert(new_states.end(), tmp_states.begin(), tmp_states.end());
      bool need_prune = false;
      if (prune_by_rule) {
        need_prune = (type == RuleApplyType::kApplyAndPruneOtherRules);
      } else {
        need_prune =
            (utils::SampleUniformDouble(0, 1, &rand_seed_) < prune_probability);
      }
      if (need_prune) {
        iter = layer.erase(iter);
      } else {
        ++iter;
      }
    }
    VLOG(7) << "apply on block: " << block_name << ", generate "
            << new_states.size() << " new states at step " << step;
    layer.splice(layer.end(), std::move(new_states));
  }
  VLOG(6) << "apply on block: " << block_name << ", generate "
          << layer.size() - 1 << " more states at all";
  return std::vector<SearchState>(layer.begin(), layer.end());
}

}  // namespace auto_schedule
}  // namespace cinn
