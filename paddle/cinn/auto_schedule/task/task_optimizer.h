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

#include "paddle/cinn/auto_schedule/cost_model/expr_cost_model.h"
#include "paddle/cinn/auto_schedule/database/database.h"
#include "paddle/cinn/auto_schedule/measure/schedule_measurer.h"
#include "paddle/cinn/auto_schedule/search_strategy/evolutionary_search.h"
#include "paddle/cinn/auto_schedule/task/tune_task.h"
#include "paddle/cinn/auto_schedule/tuning.h"
#include "paddle/cinn/ir/lowered_func.h"
#include "paddle/cinn/utils/random_engine.h"

namespace cinn {
namespace auto_schedule {

// This class is responsible for tuning a specific task,
// it will integrate necessary components to search the
// optimal schedule for the task.
class TaskOptimizer {
 public:
  TaskOptimizer(TuneTask* task,
                ScheduleMeasurer* schedule_measurer,
                Database* database,
                utils::LinearRandomEngine::StateType rand_seed = -1);

  FunctionGroup Optimize(const TuningOptions& options);

 private:
  struct Result {
    std::string from;
    double cost;
    FunctionGroup functions;
    explicit Result(const std::string& from_type)
        : from(from_type), cost(std::numeric_limits<double>::max()) {}
  };

  Result OptimizeByManual(bool need_measure);
  Result OptimizeByExternal(bool need_measure);
  Result OptimizeByEvolution(const TuningOptions& options);

  // call search candidates once by EvolutionarySearch and prune invalid ones
  std::vector<SearchState> SearchOneRound(
      const TuningOptions& options,
      std::vector<MeasureInput>* measure_candidates);

 private:
  // the max retry times if continuously get empty result
  static constexpr uint32_t kMaxRetryContinuousEmpty_ = 3;
  TuneTask* task_;
  ScheduleMeasurer* schedule_measurer_;
  std::unique_ptr<EvolutionarySearch> evolutionary_search_ = nullptr;
  ExprCostModel cost_model_;
  Database* database_;
  utils::LinearRandomEngine::StateType rand_seed_;
};

}  // namespace auto_schedule
}  // namespace cinn
