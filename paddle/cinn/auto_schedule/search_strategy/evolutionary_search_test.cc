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

#include <gtest/gtest.h>

#include <memory>
#include <utility>

#include "paddle/cinn/auto_schedule/cost_model/expr_cost_model.h"
#include "paddle/cinn/auto_schedule/database/database.h"
#include "paddle/cinn/auto_schedule/search_space/search_space.h"
#include "paddle/cinn/auto_schedule/search_space/search_state.h"
#include "paddle/cinn/auto_schedule/task/task_creator.h"
#include "paddle/cinn/auto_schedule/task/task_registry.h"
#include "paddle/cinn/auto_schedule/task/tune_task.h"
#include "paddle/cinn/auto_schedule/tuning.h"
#include "paddle/cinn/hlir/framework/op_lowering.h"
#include "paddle/cinn/ir/ir_base.h"
#include "paddle/cinn/ir/schedule/ir_schedule.h"
#include "test/cpp/cinn/program_builder.h"

namespace cinn {
namespace auto_schedule {

std::vector<TuneTask> CreateTasks(const frontend::Program& program,
                                  const Target& target) {
  auto graph = std::make_shared<hlir::framework::Graph>(program, target);
  TaskCreator task_creator;
  auto tasks = task_creator.CreateTuneTaskOpLevel(graph.get());
  const auto& dtype_dict =
      graph->GetAttrs<absl::flat_hash_map<std::string, common::Type>>(
          "inferdtype");
  const auto& shape_dict = graph->GetAttrs<
      absl::flat_hash_map<std::string, hlir::framework::shape_t>>("infershape");
  auto op_lowerer =
      hlir::framework::CreateOpLowerer(dtype_dict, shape_dict, target);
  InitialTaskRegistry* task_registry = InitialTaskRegistry::Global();
  for (auto i = 0; i < tasks.size(); ++i) {
    tasks[i].Initialize(shape_dict, dtype_dict, &op_lowerer);
    task_registry->Regist(tasks[i].serialized_key,
                          ir::ModuleExpr(tasks[i].GetLoweredFuncBodyExprs()));
  }
  return tasks;
}

/**
 * A mock search space is only used for test. It creates integer ir::Expr from
 * 0, -1, -2, ... and set the cost value same as the integer value.
 *
 * So evolutionary search should be able to find the minimal ModuleExpr with
 * smallest ir::Expr. This file tests it.
 */
class MockSearchSpace : public SearchSpace {
 public:
  explicit MockSearchSpace(const TuneTask& tune_task)
      : SearchSpace(tune_task) {}

  int GetMinExprValue() const { return min_expr_value_; }

  int GetModuleExprSize() const { return module_expr_size_; }

  std::vector<SearchState> GenerateSketches(
      int num, const std::string& strategy) override {
    std::vector<SearchState> ret;
    for (int i = 0; i < num; ++i) {
      std::vector<ir::Expr> exprs;
      for (int j = 0; j < module_expr_size_; ++j) {
        exprs.push_back(ir::Expr(-i));
      }
      min_expr_value_ = -i;
      ret.push_back(SearchState(ir::IRSchedule(ir::ModuleExpr(exprs))));
    }
    return ret;
  }

 private:
  int module_expr_size_ = 10;
  int min_expr_value_ = 0;
};

class MockCostModel : public ExprCostModel {
  float Predict(const ir::ModuleExpr& sample,
                const common::Target& target) const override {
    float cost = 0.0f;
    std::vector<ir::Expr> exprs = sample.GetExprs();
    for (const ir::Expr& expr : exprs) {
      if (expr.as_int32()) {
        cost += static_cast<float>((expr.as_int32()));
      }
    }
    return cost;
  }
};

TEST(EvolutionarySearch, GetOneBest) {
  TuneTask mock_tune_task;
  mock_tune_task.serialized_key = "mock_task";
  mock_tune_task.target = common::DefaultTarget();
  InitialTaskRegistry* task_registry = InitialTaskRegistry::Global();
  task_registry->Regist(mock_tune_task.serialized_key,
                        ir::ModuleExpr({ir::Expr(0)}));
  MockCostModel cost_model;
  TuningOptions options;
  Database db(2);
  EvolutionarySearch evolutionary_search(mock_tune_task, cost_model, &db);

  MockSearchSpace* mock_search_space = new MockSearchSpace(mock_tune_task);
  // Ownership is transferred so don't delete mock_search_space
  evolutionary_search.SetSearchSpace(mock_search_space);
  SearchState best_state = evolutionary_search.SearchModuleExpr(options);
  std::vector<ir::Expr> exprs = best_state->ir_schedule.GetModule().GetExprs();
  EXPECT_GE(exprs.size(), 1UL);
  for (const ir::Expr& e : exprs) {
    EXPECT_EQ(e.as_int32(), mock_search_space->GetMinExprValue());
  }
}

TEST(EvolutionarySearch, GetEpsGreedy) {
  TuneTask mock_tune_task;
  mock_tune_task.serialized_key = "mock_task";
  mock_tune_task.target = common::DefaultTarget();
  InitialTaskRegistry* task_registry = InitialTaskRegistry::Global();
  task_registry->Regist(mock_tune_task.serialized_key,
                        ir::ModuleExpr({ir::Expr(0)}));
  ExprCostModel cost_model;
  TuningOptions options;
  Database db(2);
  EvolutionarySearch evolutionary_search(mock_tune_task, cost_model, &db);

  MockSearchSpace* mock_search_space = new MockSearchSpace(mock_tune_task);
  // Ownership is transferred so don't delete mock_search_space
  evolutionary_search.SetSearchSpace(mock_search_space);
  std::vector<SearchState> search_states =
      evolutionary_search.SearchModuleExprEpsGreedy(options);

  EXPECT_GE(search_states.size(), 1UL);
  size_t expr_size =
      static_cast<size_t>(mock_search_space->GetModuleExprSize());
  for (const SearchState& state : search_states) {
    EXPECT_EQ(state->ir_schedule.GetModule().GetExprs().size(), expr_size);
  }
}

TEST(EvolutionarySearch, Evolve) {
  auto target = common::DefaultNVGPUTarget();
  auto tasks = CreateTasks(
      tests::OpBuilder("matmul").Build({{"X", {32, 32}}, {"Y", {32, 32}}}),
      target);
  CHECK_EQ(tasks.size(), 1);
  ExprCostModel cost_model;
  std::vector<const ir::ModuleExpr*> cost_model_samples(1);
  std::vector<float> cost_model_labels(1);
  for (size_t i = 0; i < 2; ++i) {
    ir::ModuleExpr me({ir::Expr(tasks[0].lowered_funcs[0])});
    cost_model_samples[0] = &me;
    cost_model_labels[0] = i + 10;
    cost_model.Update(cost_model_samples, cost_model_labels, target);
  }

  Database db(2);
  TuningOptions options;
  options.evolution_pick_database_topk = 0;

  EvolutionarySearch evolutionary_search(tasks[0], cost_model, &db);

  int num_population = 10;
  std::vector<SearchState> init_sketch =
      evolutionary_search.TestInitSketch(num_population, "rule_prune");
  for (int i = 0; i < num_population; ++i) {
    ir::ModuleExpr me(init_sketch[i]->ir_schedule.GetModule());
    cost_model_samples[0] = &me;
    cost_model_labels[0] = i;
    cost_model.Update(cost_model_samples, cost_model_labels, target);
  }
  VLOG(6) << "init sketch costs:";
  for (auto s : init_sketch) {
    VLOG(6) << "cost = " << s->predicted_cost;
  }
  std::vector<SearchState>*population_pre_ptr = &init_sketch,
  *population_next_ptr;
  std::vector<SearchState> population;
  for (int i = 0; i < 10; ++i) {
    population = evolutionary_search.TestEvolve(
        *population_pre_ptr, /*cross_over_num*/ 0, /*ret_num*/ 10);
    population_next_ptr = &population;
    VLOG(6) << "population[" << i + 1 << "] costs:";
    double total_cost_pre = 0.0, total_cost_next = 0.0;
    for (auto s : *population_pre_ptr) {
      total_cost_pre += s->predicted_cost;
    }
    for (auto s : *population_next_ptr) {
      total_cost_next += s->predicted_cost;
      VLOG(6) << "cost = " << s->predicted_cost;
    }
    VLOG(6) << "total_cost_next = " << total_cost_next;
    CHECK_LE(total_cost_next, total_cost_pre);
    std::swap(population_pre_ptr, population_next_ptr);
  }
}

}  // namespace auto_schedule
}  // namespace cinn
