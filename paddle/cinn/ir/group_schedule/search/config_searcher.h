// Copyright (c) 2024 CINN Authors. All Rights Reserved.
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

#include <algorithm>
#include <functional>
#include <map>
#include <vector>

#include "paddle/cinn/ir/group_schedule/search/measurer.h"
#include "paddle/cinn/utils/random_engine.h"
#include "paddle/pir/include/core/program.h"

namespace cinn {
namespace ir {
namespace search {

using ScoreType = float;
using CandidateType = std::vector<int>;
using ConstraintFunc = std::function<bool(const CandidateType&)>;

class BaseObjectiveFunc {
 public:
  virtual ScoreType operator()(const CandidateType& candidate) = 0;
};

class WeightedSamplingTrailObjectiveFunc : public BaseObjectiveFunc {
 public:
  WeightedSamplingTrailObjectiveFunc(::pir::Program* program,
                                     const IterSpace& iter_space,
                                     double sampling_prob = 1.0,
                                     int max_sampling_times = 65536,
                                     int repeats = 10);

  ScoreType operator()(const CandidateType& candidate) override;

 private:
  ::pir::Program* program_;
  IterSpace iter_space_;
  Measurer measurer_;
  double sampling_prob_;
  int max_sampling_times_;
  int repeats_;
  int sampling_times_;

  utils::LinearRandomEngine::StateType rand_seed_ = 1;
};

class CandidateGenerator {
 public:
  CandidateGenerator(const std::vector<std::pair<int, int>>& candidate_range,
                     const std::vector<ConstraintFunc>& constraints);

  std::vector<CandidateType> Candidates() const;

  CandidateType Next(CandidateType candidate, int ndim, int step) const;

  bool IsValid(const CandidateType& candidate) const;

 private:
  std::vector<std::vector<int>> candidates_each_dim_;
  std::vector<ConstraintFunc> constraints_;
};

class ScheduleConfigSearcher {
 public:
  ScheduleConfigSearcher(
      std::unique_ptr<BaseObjectiveFunc> objective_func,
      const std::vector<std::pair<int, int>>& candidate_range,
      const std::vector<ConstraintFunc>& contraints = {});

  std::pair<ScoreType, CandidateType> Search(bool is_search_minimun = true);

 private:
  std::unique_ptr<BaseObjectiveFunc> objective_func_;
  std::vector<ConstraintFunc> contraints_;
  std::vector<std::pair<int, int>> candidate_range_;

  std::map<ScoreType, CandidateType> records_;
};

}  // namespace search
}  // namespace ir
}  // namespace cinn
