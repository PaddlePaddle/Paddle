// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

// The design and code is mainly from MLIR, thanks to the great project.

#pragma once

#include <functional>
#include <unordered_map>

#include "paddle/pir/include/core/op_info.h"
#include "paddle/pir/include/core/operation.h"
#include "paddle/pir/include/pattern_rewrite/pattern_match.h"

namespace pir {

class FrozenRewritePatternSet;
class RewritePattern;
class Pattern;

class PatternApplicator {
 public:
  using CostModel = std::function<PatternBenefit(const Pattern&)>;

  explicit PatternApplicator(const FrozenRewritePatternSet& frozen_patter_list);
  ~PatternApplicator() = default;

  bool MatchAndRewrite(Operation* op,
                       PatternRewriter& rewriter,  // NOLINT
                       std::function<bool(const Pattern&)> can_apply = {},
                       std::function<void(const Pattern&)> on_failure = {},
                       std::function<bool(const Pattern&)> on_success = {});

  void ApplyCostModel(const CostModel& model);

  void ApplyDefaultCostModel() {
    ApplyCostModel([](const Pattern& pattern) { return pattern.benefit(); });
  }

  void WalkAllPatterns(std::function<void(const Pattern&)> walk);

 private:
  const FrozenRewritePatternSet& frozen_pattern_list_;
  std::unordered_map<OpInfo, std::vector<const RewritePattern*>> patterns_;
  std::vector<const RewritePattern*> any_op_patterns_;
};

}  // namespace pir
