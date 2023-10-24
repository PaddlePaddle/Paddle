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

#include "paddle/cinn/auto_schedule/search_strategy/evolutionary_search.h"

#include <glog/logging.h>

#include <algorithm>
#include <cstdlib>
#include <functional>
#include <limits>
#include <memory>
#include <utility>

#include "paddle/cinn/auto_schedule/database/database.h"
#include "paddle/cinn/auto_schedule/post_schedule_rule/cooperative_process.h"
#include "paddle/cinn/auto_schedule/search_space/search_space.h"
#include "paddle/cinn/auto_schedule/search_space/search_state.h"
#include "paddle/cinn/auto_schedule/search_strategy/mutate_rule/mutate_tile_size.h"
#include "paddle/cinn/auto_schedule/task/task_registry.h"
#include "paddle/cinn/auto_schedule/task/tune_task.h"
#include "paddle/cinn/auto_schedule/tuning.h"
#include "paddle/cinn/ir/utils/ir_copy.h"
#include "paddle/cinn/utils/multi_threading.h"
#include "paddle/cinn/utils/sized_multi_set.h"
#include "paddle/cinn/utils/string.h"

PD_DECLARE_bool(auto_schedule_use_cost_model);

namespace cinn {
namespace auto_schedule {

EvolutionarySearch::EvolutionarySearch(
    const TuneTask& tune_task,
    const ExprCostModel& cost_model,
    Database* database,
    utils::LinearRandomEngine::StateType rand_seed,
    const std::vector<std::tuple<std::string, double>>& mutate_rules)
    : tune_task_(tune_task),
      cost_model_(cost_model),
      database_(database),
      rand_seed_(utils::LinearRandomEngine::NormalizeState(rand_seed)),
      mutators_(mutate_rules) {
  search_space_ = std::make_unique<SearchSpace>(
      tune_task, utils::ForkRandomState(&rand_seed_));
  if (mutators_.empty()) {
    mutators_.push_back(std::make_tuple("mutate_tile_size", 1.0));
  }
  double accum_weight = 0.0;
  for (const auto& mutator : mutators_) {
    if (std::get<1>(mutator) > 0) {
      accum_weight += std::get<1>(mutator);
      weighted_mutators_.insert(
          std::make_pair(accum_weight, MutateRule::Make(std::get<0>(mutator))));
    }
  }

  post_schedule_rules_.emplace_back(new CooperativeProcess);
}

EvolutionarySearch::~EvolutionarySearch() {}

SearchState EvolutionarySearch::SearchModuleExpr(const TuningOptions& options) {
  return SearchModuleExprBests(options)[0];
}

std::vector<SearchState> EvolutionarySearch::SearchModuleExprBests(
    const TuningOptions& options) {
  VLOG(4) << "start SearchModuleExprBests with initial statistics: "
             "visited_candidates size="
          << visited_candidates_.size();
  std::vector<SearchState> init_population;
  std::vector<SearchState> topk_from_database =
      GetTopKCandidatesFromDatabase(options.evolution_pick_database_topk);
  VLOG(4) << JoinStatesDebugString(
      "EvolutionarySearch::GetTopKCandidatesFromDatabase",
      topk_from_database,
      /*verbose=*/VLOG_IS_ON(5));
  int init_num =
      options.evolution_init_population_num - topk_from_database.size();

  std::vector<SearchState> init_sketch = InitSketch(init_num, "rule_prune");
  VLOG(4) << JoinStatesDebugString(
      "EvolutionarySearch::InitSketch", init_sketch, /*verbose=*/VLOG_IS_ON(5));

  init_population.insert(init_population.end(),
                         topk_from_database.begin(),
                         topk_from_database.end());
  init_population.insert(
      init_population.end(), init_sketch.begin(), init_sketch.end());

  std::vector<SearchState> picked_bests =
      Evolve(init_population,
             options.evolution_cross_over_num,
             options.num_samples_per_iteration);
  VLOG(4) << JoinStatesDebugString(
      "EvolutionarySearch::Evolve", picked_bests, /*verbose=*/VLOG_IS_ON(5));
  return picked_bests;
}

std::vector<SearchState> EvolutionarySearch::SearchModuleExprEpsGreedy(
    const TuningOptions& options) {
  std::vector<SearchState> picked_bests = SearchModuleExprBests(options);
  int random_num = options.evolution_init_population_num -
                   options.evolution_pick_database_topk;
  auto results =
      PickNextGenerationEpsGreedy(picked_bests,
                                  InitSketch(random_num, "random_prune"),
                                  options.num_samples_per_iteration,
                                  options.evolution_eps_greedy);
  VLOG(4) << JoinStatesDebugString(
      "EvolutionarySearch::PickNextGenerationEpsGreedy",
      results,
      /*verbose=*/VLOG_IS_ON(5));
  return results;
}

std::vector<SearchState> EvolutionarySearch::GetTopKCandidatesFromDatabase(
    int topk) {
  std::vector<SearchState> results;
  const auto& task_key = tune_task_.serialized_key;
  auto records = database_->GetTopK(task_key, topk);
  InitialTaskRegistry* task_registry = InitialTaskRegistry::Global();
  for (auto&& record : records) {
    ir::IRSchedule ir_sch(
        ir::ir_utils::IRCopy(task_registry->Get(task_key)->module_expr),
        utils::ForkRandomState(&rand_seed_));
    ir::ScheduleDesc::ReplayWithProto(record.trace, &ir_sch);
    results.emplace_back(SearchState(std::move(ir_sch), record.predicted_cost));
  }
  return results;
}

void ApplyPostScheduleRules(
    ir::IRSchedule* schedule,
    const std::vector<std::unique_ptr<PostScheduleRule>>& post_schedule_rules) {
  schedule->TagPostSchedule();
  for (const auto& post_rule : post_schedule_rules) {
    post_rule->Apply(schedule);
  }
}

std::vector<SearchState> EvolutionarySearch::InitSketch(
    int num, const std::string& strategy) {
  VLOG(4) << "InitSketch with num:" << num << ", strategy: " << strategy;
  std::vector<SearchState> states =
      search_space_->GenerateSketches(num, strategy);
  auto post_schedule_fn = [this, &states](int index) {
    ApplyPostScheduleRules(&states[index]->ir_schedule, post_schedule_rules_);
  };
  utils::parallel_run(post_schedule_fn,
                      utils::SequenceDispatcher(0, states.size()),
                      states.size());

  return states;
}

SearchState EvolutionarySearch::CrossOver(const SearchState& state1,
                                          const SearchState& state2) {
  // TODO(CtfGo): tracing CrossOver with IRSchedule
  std::vector<ir::Expr> cross_over_exprs;
  std::vector<ir::Expr> father_exprs =
      state1->ir_schedule.GetModule().GetExprs();
  std::vector<ir::Expr> mother_exprs =
      state2->ir_schedule.GetModule().GetExprs();

  CHECK_EQ(father_exprs.size(), mother_exprs.size())
      << "CrossOver ModuleExpr in EvolutionarySearch must have same number of "
         "AST";

  for (size_t i = 0; i < father_exprs.size(); ++i) {
    if (utils::SampleUniformInt(0, 2, &rand_seed_) == 0) {
      cross_over_exprs.push_back(ir::ir_utils::IRCopy(father_exprs[i]));
    } else {
      cross_over_exprs.push_back(ir::ir_utils::IRCopy(mother_exprs[i]));
    }
  }
  auto res = SearchState(ir::IRSchedule(ir::ModuleExpr(cross_over_exprs),
                                        utils::ForkRandomState(&rand_seed_)));
  if (FLAGS_auto_schedule_use_cost_model) {
    res->predicted_cost =
        cost_model_.Predict(res->ir_schedule.GetModule(), tune_task_.target);
  }
  VLOG(5) << JoinStatesDebugString("EvolutionarySearch::CrossOver",
                                   {state1, state2, res},
                                   /*verbose=*/VLOG_IS_ON(6));
  return res;
}

SearchState EvolutionarySearch::Mutate(
    const SearchState& state, utils::LinearRandomEngine::StateType* rand_seed) {
  CHECK_GT(weighted_mutators_.size(), 0)
      << "There is no mutate rule can be applied.";
  double accu_weight = (weighted_mutators_.rbegin())->first;
  CHECK_GT(accu_weight, 0) << "The accumulate weight must be greater than 0.";
  // sample a mutate rule
  double sample_weight = utils::SampleUniformDouble(0, accu_weight, rand_seed);
  auto sampled_iter = weighted_mutators_.upper_bound(sample_weight);
  MutateRule* mutator = sampled_iter->second.get();
  CHECK(mutator) << "mutator not defined";
  // apply mutation on the trace of SearchState
  auto trace = state->ir_schedule.GetTraceDesc();
  auto new_trace = mutator->Apply(trace, rand_seed);
  // replay the mutated trace on original ModuleExpr to generate a new
  // ir_schedule
  const auto& task_key = tune_task_.serialized_key;
  InitialTaskRegistry* task_registry = InitialTaskRegistry::Global();
  ir::IRSchedule new_ir_sch(
      ir::ir_utils::IRCopy(task_registry->Get(task_key)->module_expr),
      utils::ForkRandomState(rand_seed));
  new_trace.Replay(&new_ir_sch, true);
  ApplyPostScheduleRules(&new_ir_sch, post_schedule_rules_);
  auto res = SearchState(std::move(new_ir_sch));

  VLOG(5) << JoinStatesDebugString(
      "EvolutionarySearch::Mutate", {state, res}, /*verbose=*/VLOG_IS_ON(6));
  return res;
}

std::vector<SearchState> EvolutionarySearch::Evolve(
    const std::vector<SearchState>& population,
    int cross_over_num,
    int ret_num) {
  VLOG(4) << utils::StringFormat(
      "Evolve with population size=%lu,cross_over_num:%lu,ret_num:%lu",
      population.size(),
      cross_over_num,
      ret_num);
  int generation_num = population.size();
  if (generation_num == 0) {
    return std::vector<SearchState>();
  }
  // init evolution
  std::vector<SearchState> evolution(population);
  for (SearchState& search_state : evolution) {
    if (search_state->predicted_cost == SearchState::NOT_INIT_COST &&
        FLAGS_auto_schedule_use_cost_model) {
      search_state->predicted_cost = cost_model_.Predict(
          search_state->ir_schedule.GetModule(), tune_task_.target);
    }
  }
  VLOG(4) << JoinStatesDebugString(
      "EvolutionarySearch::Evolve: Init evolution:",
      evolution,
      /*verbose=*/VLOG_IS_ON(5));
  // cross over
  for (int i = 0; i < cross_over_num; ++i) {
    int first_rand_idx =
        utils::SampleUniformInt(0, generation_num, &rand_seed_);
    int second_rand_idx =
        utils::SampleUniformInt(0, generation_num, &rand_seed_);
    while (first_rand_idx == second_rand_idx) {
      second_rand_idx = utils::SampleUniformInt(0, generation_num, &rand_seed_);
    }
    evolution.push_back(
        CrossOver(population[first_rand_idx], population[second_rand_idx]));
  }
  VLOG(4) << JoinStatesDebugString(
      "EvolutionarySearch::Evolve: after CrossOver evolution:",
      evolution,
      /*verbose=*/VLOG_IS_ON(5));
  // mutate
  std::vector<SearchState> mutated_individuals(evolution.size());
  std::vector<utils::LinearRandomEngine::StateType> rand_seeds(
      evolution.size());
  for (int i = 0; i < rand_seeds.size(); ++i) {
    rand_seeds[i] = utils::ForkRandomState(&rand_seed_);
  }
  auto mutate_fn = [this, &evolution, &mutated_individuals, &rand_seeds](
                       int index) {
    mutated_individuals[index] = Mutate(evolution[index], &rand_seeds[index]);
  };
  utils::parallel_run(mutate_fn,
                      utils::SequenceDispatcher(0, evolution.size()),
                      evolution.size());
  if (FLAGS_auto_schedule_use_cost_model) {
    for (size_t i = 0; i < mutated_individuals.size(); ++i) {
      mutated_individuals[i]->predicted_cost = cost_model_.Predict(
          mutated_individuals[i]->ir_schedule.GetModule(), tune_task_.target);
    }
  }
  VLOG(4) << JoinStatesDebugString(
      "EvolutionarySearch::Evolve: mutated individuals:",
      mutated_individuals,
      /*verbose=*/VLOG_IS_ON(5));
  // select top ret_num with predicted cost
  utils::SizedMultiSet<SearchState> evolution_with_cost(ret_num);
  for (size_t i = 0; i < evolution.size(); ++i) {
    evolution_with_cost.Push(evolution[i]);
  }
  for (size_t i = 0; i < mutated_individuals.size(); ++i) {
    evolution_with_cost.Push(mutated_individuals[i]);
  }
  auto selected_individuals =
      evolution_with_cost.ReturnAsContainer<std::vector<SearchState>>();
  VLOG(4) << JoinStatesDebugString(
      "EvolutionarySearch::Evolve: selected individuals:",
      selected_individuals,
      /*verbose=*/VLOG_IS_ON(5));

  return selected_individuals;
}

std::vector<SearchState> EvolutionarySearch::PickNextGenerationEpsGreedy(
    const std::vector<SearchState>& picked_bests,
    const std::vector<SearchState>& random_init,
    int num,
    float eps_greedy) {
  int num_rands = num * eps_greedy;
  int num_bests = num - num_rands;

  std::vector<SearchState> result;
  SearchState selected;
  int deduplicated_cnt = 0;
  int best_idx = 0;
  int rand_idx = 0;
  while (result.size() < num) {
    if (result.size() < num_bests && best_idx < picked_bests.size()) {
      selected = picked_bests[best_idx];
      ++best_idx;
    } else if (rand_idx < random_init.size()) {
      selected = random_init[rand_idx];
      ++rand_idx;
    } else if (best_idx < picked_bests.size()) {
      selected = picked_bests[best_idx];
      ++best_idx;
    } else {
      break;
    }

    if (!visited_candidates_.count(selected)) {  // deduplicate
      VLOG(4) << JoinStatesDebugString(
          "EvolutionarySearch::PickNextGenerationEpsGreedy-Selected",
          {selected},
          /*verbose=*/VLOG_IS_ON(5));
      visited_candidates_.insert(selected);
      result.push_back(selected);
    } else {
      ++deduplicated_cnt;
      VLOG(4) << JoinStatesDebugString(
          "EvolutionarySearch::PickNextGenerationEpsGreedy-Deduplicated",
          {selected},
          /*verbose=*/VLOG_IS_ON(5));
    }
  }

  VLOG(4) << utils::StringFormat(
      "PickNextGenerationEpsGreedy: picked_bests size=%lu,random_init "
      "size=%lu,num=%d,"
      "eps_greedy=%f,deduplicated_cnt=%d,result size=%lu",
      picked_bests.size(),
      random_init.size(),
      num,
      eps_greedy,
      deduplicated_cnt,
      result.size());
  return result;
}

}  // namespace auto_schedule
}  // namespace cinn
