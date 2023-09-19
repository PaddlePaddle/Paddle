// Copyright (c) 2023 CINN Authors. All Rights Reserved.
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

#include "paddle/cinn/auto_schedule/search_strategy/mutate_rule/mutate_tile_size.h"

namespace cinn {
namespace auto_schedule {

using ::cinn::ir::ScheduleDesc;
using ::cinn::utils::LinearRandomEngine;

using SampledTile = std::tuple<ScheduleDesc::Step, std::vector<int>, int>;

static std::vector<int> Factorize(int n) {
  std::vector<int> res;
  for (int i = 1; i * i <= n; ++i) {
    if (n % i == 0) {
      res.push_back(i);
      if (i * i != n) {
        res.push_back(n / i);
      }
    }
  }
  std::sort(res.begin(), res.end());
  return res;
}

std::vector<SampledTile> FindSampledTiles(const ScheduleDesc& trace) {
  std::vector<SampledTile> tiles;
  int step_idx = 0;
  for (auto&& step : trace.Steps()) {
    if (step.type == "TagPostSchedule") {
      break;
    }
    if (step.type == "SamplePerfectTile") {
      std::vector<int> tile_factors =
          absl::get<std::vector<int>>(step.attrs.at("decision"));
      CHECK(tile_factors.size() >= 2)
          << "factors size must be greater equal than 2, which is "
          << tile_factors.size();
      tiles.push_back(std::make_tuple(step, tile_factors, step_idx));
    }
    ++step_idx;
  }

  return tiles;
}

ScheduleDesc DoMutateTileSize(const ScheduleDesc& trace,
                              const SampledTile& tile,
                              LinearRandomEngine::StateType* rand_seed) {
  ScheduleDesc::Step step = std::get<0>(tile);
  std::vector<int> tile_factors = std::get<1>(tile);
  int split_size = tile_factors.size();
  // Step 1. Choose 2 loops with index: 'loop_x' and 'loop_y'
  int loop_x, loop_y;

  bool all_one_factors = true;
  for (int t : tile_factors) {
    if (t != 1) {
      all_one_factors = false;
      break;
    }
  }
  if (all_one_factors) {
    VLOG(6) << "Factors are all 1, unable to mutate, return the original trace";
    return trace;
  }

  while (true) {
    VLOG(6) << "while (true) loop in DoMutateTileSize";
    loop_x = utils::SampleUniformInt(0, split_size, rand_seed);
    if (tile_factors.at(loop_x) <= 1) {
      continue;
    }
    loop_y = utils::SampleUniformInt(0, split_size - 1, rand_seed);
    if (loop_y >= loop_x) {
      ++loop_y;
    }
    std::vector<int> optional_factors = Factorize(tile_factors.at(loop_x));
    // Step 2. Choose the divisor for mutate.
    int divisor;
    if (loop_y == split_size - 1) {
      int max_innermost_factor =
          absl::get<int>(step.attrs.at("max_innermost_factor"));
      int max_optional_factor_idx = optional_factors.size() - 1;
      for (; max_optional_factor_idx > 0; --max_optional_factor_idx) {
        if (optional_factors.at(max_optional_factor_idx) *
                tile_factors.at(loop_y) <=
            max_innermost_factor) {
          break;
        }
      }
      if (max_optional_factor_idx == 0) {
        if (split_size <= 2) {
          VLOG(6) << "Unable to mutate, return the original trace";
          return trace;
        }
        continue;
      }
      divisor = optional_factors.at(
          utils::SampleUniformInt(1, max_optional_factor_idx + 1, rand_seed));
    } else {
      divisor = optional_factors.at(
          utils::SampleUniformInt(1, optional_factors.size(), rand_seed));
    }
    // Step 3. Determine the new tile value
    VLOG(6) << "DoMutateTileSize: divisor = " << divisor
            << ", before mutate: \n"
            << "factors[" << loop_x << "] = " << tile_factors[loop_x]
            << ", factors[" << loop_y << "] = " << tile_factors[loop_y];
    tile_factors[loop_x] /= divisor;
    tile_factors[loop_y] *= divisor;
    VLOG(6) << "after mutate: \n"
            << "factors[" << loop_x << "] = " << tile_factors[loop_x]
            << ", factors[" << loop_y << "] = " << tile_factors[loop_y];
    // Step 4. Create a new step with new tile values and return the new trace
    int step_idx = std::get<2>(tile);
    return trace.ForkAndUpdate(step_idx, tile_factors, true);
  }
}

ScheduleDesc MutateTileSize::Apply(const ScheduleDesc& trace,
                                   LinearRandomEngine::StateType* rand_seed) {
  VLOG(6) << "Start applying MutateTileSize, old trace: \n"
          << trace.DebugString();
  std::vector<ScheduleDesc::Step> sample_tile_steps;
  std::vector<std::vector<int>> sample_tile_data;

  auto sampled_tiles = FindSampledTiles(trace);
  if (sampled_tiles.size() == 0) {
    VLOG(6) << "MutateTileSize failed, try other mutate rules.";
    return trace;
  }
  int sample_step_idx =
      utils::SampleUniformInt(0, sampled_tiles.size(), rand_seed);
  auto new_trace =
      DoMutateTileSize(trace, sampled_tiles.at(sample_step_idx), rand_seed);
  VLOG(6) << "End applying MutateTileSize, new trace: \n"
          << new_trace.DebugString();
  return new_trace;
}

}  // namespace auto_schedule
}  // namespace cinn
