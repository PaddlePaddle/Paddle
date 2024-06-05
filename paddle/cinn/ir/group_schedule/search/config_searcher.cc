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

#include "paddle/cinn/ir/group_schedule/search/config_searcher.h"
#include "paddle/cinn/common/target.h"
#include "paddle/cinn/ir/group_schedule/config/database.h"
#include "paddle/cinn/ir/group_schedule/config/filedatabase.h"
#include "paddle/cinn/utils/string.h"

namespace cinn {
namespace ir {
namespace search {

WeightedSamplingTrailObjectiveFunc::WeightedSamplingTrailObjectiveFunc(
    ::pir::Program* program,
    const BucketInfo& bucket_info,
    double sampling_prob,
    int max_sampling_times,
    int repeats)
    : program_(program),
      bucket_info_(bucket_info),
      measurer_(program),
      sampling_prob_(sampling_prob),
      max_sampling_times_(max_sampling_times),
      repeats_(repeats) {
  double weighted_space_size = 1.0;
  for (const auto& dim : bucket_info_.space) {
    PADDLE_ENFORCE_EQ(dim.upper_bound - dim.lower_bound + 1,
                      dim.weights.size(),
                      ::common::errors::InvalidArgument(
                          "The number of weights does not match the difference "
                          "between the upper and lower bound"));

    double weights_sum =
        std::accumulate(dim.weights.begin(), dim.weights.end(), 0.0);
    weighted_space_size *= weights_sum;
  }
  sampling_times_ =
      std::min(static_cast<int>(weighted_space_size * sampling_prob),
               max_sampling_times);

  // Generate Sampling Inputs
  const auto Sample = [&]() -> std::vector<int64_t> {
    std::vector<int64_t> samples;
    for (BucketInfo::Dimension dim : bucket_info_.space) {
      int sampled = utils::SampleDiscreteFromDistribution<double>(dim.weights,
                                                                  &rand_seed_);
      samples.push_back(static_cast<int64_t>(sampled) + dim.lower_bound);
    }
    return samples;
  };

  // Currently, only one reduce input is supported
  const auto GenerateInputs =
      [&]() -> std::unordered_map<std::string, std::vector<int64_t>> {
    std::unordered_map<std::string, std::vector<int64_t>> inputs;
    inputs["x"] = Sample();
    return inputs;
  };

  for (int i = 0; i < sampling_times_; ++i) {
    std::unordered_map<std::string, std::vector<int64_t>>
        input_name_and_shapes = GenerateInputs();
    inputs_sampling_.push_back(input_name_and_shapes);
  }
}

ScoreType WeightedSamplingTrailObjectiveFunc::operator()(
    const CandidateType& candidate) {
  auto tile_config_database = std::make_shared<NaiveTileConfigDatabase>();
  IterSpaceType iter_space_type = [&] {
    std::vector<std::pair<std::string, std::string>> res;
    for (const auto& dim : bucket_info_.space) {
      res.emplace_back(dim.iter_type, (dim.is_dynamic ? "dynamic" : "static"));
    }
    return res;
  }();
  ScheduleConfig::TileConfig config{
      candidate[0], candidate[1], candidate[2], NoneReduceMethod()};
  tile_config_database->AddConfig(
      cinn::common::DefaultTarget(), bucket_info_, config);
  auto& schedule_config_manager = ScheduleConfigManager::Instance();
  schedule_config_manager.AddConfigDatabase("custom", tile_config_database);
  measurer_.Compile();

  for (auto& input_name_and_shapes : inputs_sampling_) {
    measurer_.Run(input_name_and_shapes, repeats_);
  }
  ScoreType score = measurer_.Result().avg_kernel_execute_time.count();
  return score;
}

CandidateGenerator::CandidateGenerator(
    const std::vector<std::pair<int, int>>& candidate_range,
    const std::vector<ConstraintFunc>& constraints)
    : constraints_(constraints) {
  const auto GenerateCandidatesEachDim = [](const std::pair<int, int>& range) {
    std::vector<int> res;
    for (int i = range.first; i <= range.second; ++i) {
      res.push_back(i);
    }
    return res;
  };
  for (int i = 0; i < candidate_range.size(); ++i) {
    candidates_each_dim_.push_back(
        GenerateCandidatesEachDim(candidate_range[i]));
  }
}

std::vector<CandidateType> CandidateGenerator::Candidates() const {
  std::vector<CandidateType> candidates;
  int ndim = candidates_each_dim_.size();
  std::vector<int> indices(ndim, 0);

  const auto IsLastIndexOnDim = [&](int i) {
    return indices[i] == candidates_each_dim_[i].size() - 1;
  };

  while (true) {
    CandidateType combination;
    for (int i = 0; i < ndim; ++i) {
      combination.push_back(candidates_each_dim_[i][indices[i]]);
    }
    if (IsValid(combination)) {
      candidates.push_back(combination);
    }

    int i = ndim - 1;
    while (i >= 0 && IsLastIndexOnDim(i)) {
      indices[i] = 0;
      --i;
    }
    if (i < 0) break;
    ++indices[i];
  }
  return candidates;
}

CandidateType CandidateGenerator::Next(CandidateType candidate,
                                       int ndim,
                                       int step) const {
  PADDLE_ENFORCE_EQ(candidate.size(),
                    candidates_each_dim_.size(),
                    ::common::errors::InvalidArgument(
                        "The length of the candidate vector is incorrect"));
  PADDLE_ENFORCE_EQ(ndim,
                    candidate.size(),
                    ::common::errors::InvalidArgument(
                        "The dimension to be modified needs to be less than "
                        "the total length of the vector"));

  auto iter = std::find(candidates_each_dim_[ndim].begin(),
                        candidates_each_dim_[ndim].end(),
                        candidate[ndim]);
  step = step % candidates_each_dim_[ndim].size();
  if (step >= (candidates_each_dim_[ndim].end() - iter)) {
    candidate[ndim] =
        candidates_each_dim_[ndim]
                            [step - (candidates_each_dim_[ndim].end() - iter)];
  } else {
    candidate[ndim] = *(iter + step);
  }
  return candidate;
}

bool CandidateGenerator::IsValid(const CandidateType& candidate) const {
  int i = 0;
  for (const auto& constraint : constraints_) {
    if (!constraint(candidate)) {
      return false;
    }
  }
  return true;
}

ScheduleConfigSearcher::ScheduleConfigSearcher(
    std::unique_ptr<BaseObjectiveFunc> objective_func,
    const std::vector<std::pair<int, int>>& candidate_range,
    const std::vector<ConstraintFunc>& contraints)
    : objective_func_(std::move(objective_func)),
      candidate_range_(candidate_range),
      contraints_(contraints) {}

std::pair<ScoreType, CandidateType> ScheduleConfigSearcher::Search(
    bool is_search_minimun) {
  VLOG(6) << "Start Search...";
  CandidateGenerator candidate_generator(candidate_range_, contraints_);
  std::vector<CandidateType> candidates = candidate_generator.Candidates();
  VLOG(6) << "Candidate num = " << candidates.size();
  for (const auto& candidate : candidates) {
    ScoreType score = (*objective_func_)(candidate);
    VLOG(6) << "Candidate: [" << utils::Join<int>(candidate, ", ") << "]";
    VLOG(6) << "Score = " << score;
    records_[score] = candidate;
  }
  return is_search_minimun ? *records_.begin() : *(records_.end()--);
}

}  // namespace search
}  // namespace ir
}  // namespace cinn
