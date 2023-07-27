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

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "paddle/cinn/auto_schedule/search_space/auto_gen_rule/auto_gen_rule.h"
#include "paddle/cinn/common/target.h"
#include "paddle/cinn/ir/ir.h"
#include "paddle/cinn/ir/ir_base.h"
#include "paddle/cinn/ir/schedule/ir_schedule.h"

namespace cinn {
namespace auto_schedule {

class MultiLevelTiling : public AutoGenRule {
 public:
  struct Config {
    // Which thread axis each tiled loop is bound to
    std::vector<std::string> bind_axis;
    // Use char 'S' and 'R' to represent tile structure.
    // S means space tiling level and R means reduce tiling level
    //
    // For example, if tile_struct_ = "SSRSRS" and we are doing matrix
    // multiplication, i, j are the spatial indices and k is the reduce index,
    // the tiling result will be i_0, j0, i1, j1, k0, i2, j2, k1, i3, j3
    std::string tile_struct;
    // The storage type of read cache
    std::string read_cache_memory_type;
    // Which tiled levels are read cache block inserted at
    std::vector<int> read_cache_levels;
    // The storage type of write cache
    std::string write_cache_memory_type;
    // Which tiled levels are write cache block inserted at
    std::vector<int> write_cache_levels;
  };

  static const std::unordered_map<common::Target::Arch, Config> kConfigs;

  MultiLevelTiling(const common::Target& target, const Config& config);
  ~MultiLevelTiling() = default;

  // initialize the AutoGenRule, it must be called before further actions.
  // Returns false if the rule cannot be applied on the mod_expr, true otherwise
  RuleApplyType Init(ir::IRSchedule* init_schedule) override;

  // Applies rule on the ir::ModuleExpr for a schedule block specified by index
  // between 0 (inclusive) and NumberApplicable() (exclusive)
  void Apply(int index) override;

  // Returns the name of the rule, used for debug.
  std::string GetRuleName() const override;

  // Returns true if sche_block_realize is applicable by MultiLevelTiling
  bool MeetCondition(const ir::ScheduleBlockRealize& sche_block_realize) const;

  RuleApplyType AnalyseApplyType(SearchState state,
                                 const std::string& block_name) const override;

  std::vector<SearchState> ApplyOnBlock(SearchState state,
                                        const std::string& block_name) override;

  // Sample pair of integer type (a, b) such as a * b = extent
  template <typename T>
  std::vector<T> SampleSplitTwo(T extent) const {
    std::vector<std::vector<T>> candidates;
    for (T div = 1; div <= sqrt(extent); ++div) {
      if (extent % div == 0) {
        candidates.push_back({T(div), extent / div});
      }
    }
    if (candidates.size() == 0) {
      return {1, T(extent)};
    }
    int index = rand() % candidates.size();  // NOLINT
    std::vector<T> pick = candidates[index];
    if (rand() % 2 != 0) {  // NOLINT
      T tmp = pick[0];
      pick[0] = pick[1];
      pick[1] = tmp;
    }
    return pick;
  }

  // Sample num_split integers whose product equals extent
  template <typename T>
  std::vector<T> SampleTileSplit(T extent, int num_split) const {
    CHECK_GT(num_split, 0)
        << "num_split in SampleTileSplit must be greater than 0";
    if (num_split == 1) {
      return {extent};
    }
    std::vector<T> two_split = SampleSplitTwo<T>(extent);
    if (num_split == 2) {
      return two_split;
    }
    int half = num_split >> 1;
    std::vector<T> result = SampleTileSplit<T>(two_split[0], half);
    std::vector<T> remind = SampleTileSplit<T>(two_split[1], num_split - half);
    result.insert(result.end(), remind.begin(), remind.end());
    return result;
  }

 private:
  void ApplyTiling(ir::IRSchedule* ir_schedule,
                   ir::Expr& block_expr);  // NOLINT
  void ApplyCacheRead(ir::IRSchedule* ir_schedule,
                      ir::Expr& block_expr);  // NOLINT
  void ApplyCacheWrite(ir::IRSchedule* ir_schedule,
                       ir::Expr& block_expr);  // NOLINT

 private:
  std::vector<ir::Expr> all_block_realizes_;
  std::vector<int> applicable_indices_;

  Config config_;
  std::vector<int> s_indices_;
  std::vector<int> r_indices_;
  std::vector<std::vector<ir::Expr>> tile_loops_;

  // A factor to limit the split factor within max thread number per block
  int max_factor_ = 1024;
};

}  // namespace auto_schedule
}  // namespace cinn
