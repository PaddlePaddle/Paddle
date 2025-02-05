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
#include "paddle/cinn/ir/group_schedule/config/schedule_config_manager.h"
#include "paddle/cinn/utils/string.h"

namespace cinn {
namespace ir {
namespace search {

WeightedSamplingTrailObjectiveFunc::WeightedSamplingTrailObjectiveFunc(
    ::pir::Program* program,
    const BucketInfo& bucket_info,
    double sampling_prob,
    int max_sampling_times,
    int repeats,
    std::vector<std::vector<double>> weights)
    : program_(program),
      bucket_info_(bucket_info),
      measurer_(program),
      sampling_prob_(sampling_prob),
      max_sampling_times_(max_sampling_times),
      repeats_(repeats) {
  double weighted_space_size = 1.0;
  if (weights.size() == 0) {
    for (int i = 0; i < bucket_info_.space.size(); i++) {
      auto weight =
          std::vector<double>(bucket_info_.space[i].upper_bound -
                                  bucket_info_.space[i].lower_bound + 1,
                              1.0);
      double weights_sum = std::accumulate(weight.begin(), weight.end(), 0.0);
      weighted_space_size *= weights_sum;
      weights_.push_back(weight);
    }
  } else {
    for (int i = 0; i < bucket_info_.space.size(); i++) {
      PADDLE_ENFORCE_EQ(
          bucket_info_.space[i].upper_bound -
              bucket_info_.space[i].lower_bound + 1,
          weights[i].size(),
          ::common::errors::InvalidArgument(
              "The number of weights does not match the difference "
              "between the upper and lower bound"));
      double weights_sum =
          std::accumulate(weights[i].begin(), weights[i].end(), 0.0);
      weighted_space_size *= weights_sum;
      weights_.push_back(weights[i]);
    }
  }
  sampling_times_ =
      std::min(static_cast<int>(weighted_space_size * sampling_prob),
               max_sampling_times);

  // Generate Sampling Inputs
  const auto Sample = [&]() -> std::vector<int64_t> {
    std::vector<int64_t> samples;
    for (int i = 0; i < bucket_info_.space.size(); i++) {
      BucketInfo::Dimension dim = bucket_info_.space[i];
      int sampled = utils::SampleDiscreteFromDistribution<double>(weights_[i],
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
  VLOG(3) << "Bucket_info_.space.size is " << bucket_info_.space.size();
  if (candidate.size() != 0) {
    ScheduleConfig::TileConfig config;
    config.warp_num = candidate[0];
    config.tree_reduce_num = candidate[1];
    config.spatial_inner_num = candidate[2];
    tile_config_database->AddConfig(
        cinn::common::DefaultTarget(), bucket_info_, config);
    auto& schedule_config_manager = ScheduleConfigManager::Instance();
    schedule_config_manager.AddConfigDatabase("search", tile_config_database);
  }
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
    std::vector<std::unique_ptr<BaseObjectiveFunc>> objective_funcs,
    const std::vector<std::pair<int, int>>& candidate_range,
    const std::vector<ConstraintFunc>& constraints)
    : objective_funcs_(std::move(objective_funcs)),
      candidate_range_(candidate_range),
      constraints_(constraints) {}

std::pair<ScoreType, CandidateType> ScheduleConfigSearcher::Search(
    bool is_search_minimum) {
  VLOG(6) << "Start Search...";
  CandidateGenerator candidate_generator(candidate_range_, constraints_);
  std::vector<CandidateType> candidates = candidate_generator.Candidates();
  VLOG(6) << "Candidate num = " << candidates.size();
  for (const auto& candidate : candidates) {
    ScoreType score = 0;
    for (auto& objective_func_ : objective_funcs_) {
      score += (*objective_func_)(candidate);
    }
    VLOG(6) << "Candidate: [" << utils::Join<int64_t>(candidate, ", ") << "]";
    VLOG(6) << "Score = " << score;
    records_[score] = candidate;
  }
  return is_search_minimum ? *records_.begin() : *(records_.end()--);
}

}  // namespace search
}  // namespace ir
}  // namespace cinn
