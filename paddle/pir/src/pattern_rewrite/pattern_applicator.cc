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

#include <algorithm>

#include "paddle/pir/include/pattern_rewrite/frozen_rewrite_pattern_set.h"
#include "paddle/pir/include/pattern_rewrite/pattern_applicator.h"
#include "paddle/pir/include/pattern_rewrite/pattern_match.h"

namespace pir {

PatternApplicator::PatternApplicator(
    const FrozenRewritePatternSet& frozen_pattern_list)
    : frozen_pattern_list_(frozen_pattern_list) {}

void PatternApplicator::ApplyCostModel(const CostModel& model) {
  // TODO(wilber): remove impossible patterns.
  patterns_.clear();
  for (const auto& it : frozen_pattern_list_.op_specific_native_patterns()) {
    for (const RewritePattern* pattern : it.second) {
      patterns_[it.first].push_back(pattern);
    }
  }

  any_op_patterns_.clear();
  for (auto& pattern : frozen_pattern_list_.match_any_op_native_patterns()) {
    any_op_patterns_.push_back(pattern.get());
  }

  // Sort by benefit based on the cost model.
  std::unordered_map<const Pattern*, PatternBenefit> benefits;
  auto cmp = [&benefits](const Pattern* lhs, const Pattern* rhs) {
    return benefits[lhs] > benefits[rhs];
  };
  auto ProcessPatternList = [&](std::vector<const RewritePattern*>& list) {
    if (list.size() == 1) return;

    benefits.clear();
    for (const Pattern* pat : list) benefits.emplace(pat, model(*pat));

    std::stable_sort(list.begin(), list.end(), cmp);
  };
  for (auto& it : patterns_) {
    ProcessPatternList(it.second);
  }
  ProcessPatternList(any_op_patterns_);
}

void PatternApplicator::WalkAllPatterns(
    std::function<void(const Pattern&)> walk) {
  for (const auto& it : frozen_pattern_list_.op_specific_native_patterns())
    for (auto* pattern : it.second) walk(*pattern);

  for (const auto& it : frozen_pattern_list_.match_any_op_native_patterns())
    walk(*it);
}

bool PatternApplicator::MatchAndRewrite(
    Operation* op,
    PatternRewriter& rewriter,
    std::function<bool(const Pattern&)> can_apply,
    std::function<void(const Pattern&)> on_failure,
    std::function<bool(const Pattern&)> on_success) {
  // whether there are patterns matching this operation type.
  std::vector<const RewritePattern*> op_patterns;
  auto pattern_it = patterns_.find(op->info());
  if (pattern_it != patterns_.end()) op_patterns = pattern_it->second;

  unsigned op_it = 0, op_e = op_patterns.size();
  unsigned any_it = 0, any_e = any_op_patterns_.size();
  bool result = false;
  do {
    // Find the next pattern with the highest benefit.
    const Pattern* best_pattern = nullptr;
    unsigned* best_pattern_it = &op_it;

    // For specific patterns
    if (op_it < op_e) best_pattern = op_patterns[op_it];
    // For op-agnostic patterns
    if (any_it < any_e &&
        (!best_pattern ||
         best_pattern->benefit() < any_op_patterns_[any_it]->benefit())) {
      best_pattern_it = &any_it;
      best_pattern = any_op_patterns_[any_it];
    }

    if (!best_pattern) break;

    // Update the pattern iterator, so that this pattern isn't attempted again.
    ++(*best_pattern_it);

    if (can_apply && !can_apply(*best_pattern)) continue;

    rewriter.set_insertion_point(op);

    const auto* pattern = static_cast<const RewritePattern*>(best_pattern);
    result = pattern->MatchAndRewrite(op, rewriter);

    if (result && on_success && !on_success(*best_pattern)) result = false;

    if (result) break;

    if (on_failure) on_failure(*best_pattern);
  } while (true);

  return result;
}

}  // namespace pir
