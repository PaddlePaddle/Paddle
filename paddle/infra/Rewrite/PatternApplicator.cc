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

#include "Rewrite/PatternApplicator.h"
#include <algorithm>
#include "IR/PatternMatch.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Support/LogicalResult.h"

namespace infra {

PatternApplicator::PatternApplicator(
    const FrozenRewritePatternSet& frozen_patter_list)
    : frozen_patter_list_(frozen_patter_list) {}

void PatternApplicator::ApplyCostModel(CostModel model) {
  // TODO(wilber): remove impossible patterns.
  patterns_.clear();
  for (const auto& it : frozen_patter_list_.GetOpSpecificNativePatterns()) {
    for (const RewritePattern* pattern : it.second) {
      patterns_[it.first].push_back(pattern);
    }
  }

  any_op_patterns_.clear();
  for (const RewritePattern& pattern :
       frozen_patter_list_.GetMatchAnyOpNativePatterns()) {
    any_op_patterns_.push_back(&pattern);
  }

  // Sort by benefit based on the cost model.
  llvm::DenseMap<const Pattern*, PatternBenefit> benefits;
  auto cmp = [&benefits](const Pattern* lhs, const Pattern* rhs) {
    return benefits[lhs] > benefits[rhs];
  };
  auto ProcessPatternList = [&](std::vector<const RewritePattern*>& list) {
    if (list.size() == 1) return;

    benefits.clear();
    for (const Pattern* pat : list) benefits.try_emplace(pat, model(*pat));

    std::stable_sort(list.begin(), list.end(), cmp);
  };
  for (auto& it : patterns_) {
    ProcessPatternList(it.second);
  }
  ProcessPatternList(any_op_patterns_);
}

void PatternApplicator::WalkAllPatterns(
    std::function<void(const Pattern&)> walk) {
  for (const auto& it : frozen_patter_list_.GetOpSpecificNativePatterns())
    for (auto* pattern : it.second) walk(*pattern);

  for (const Pattern& it : frozen_patter_list_.GetMatchAnyOpNativePatterns())
    walk(it);
}

mlir::LogicalResult PatternApplicator::MatchAndRewrite(
    mlir::Operation* op,
    PatternRewriter& rewriter,
    std::function<bool(const Pattern&)> can_apply,
    std::function<void(const Pattern&)> on_failure,
    std::function<mlir::LogicalResult(const Pattern&)> on_success) {
  // whether there are patterns matching this operation type.
  llvm::MutableArrayRef<const RewritePattern*> op_patterns;
  auto pattern_it = patterns_.find(op->getName());
  if (pattern_it != patterns_.end()) op_patterns = pattern_it->second;

  unsigned op_it = 0, op_e = op_patterns.size();
  unsigned any_it = 0, any_e = any_op_patterns_.size();
  mlir::LogicalResult result = mlir::failure();
  do {
    // Find the next pattern with the highest benefit.
    const Pattern* best_pattern = nullptr;
    unsigned* best_pattern_it = &op_it;

    // For specific patterns
    if (op_it < op_e) best_pattern = op_patterns[op_it];
    // For op-agnostic patterns
    if (any_it < any_e &&
        (!best_pattern ||
         best_pattern->GetBenefit() < any_op_patterns_[any_it]->GetBenefit())) {
      best_pattern_it = &any_it;
      best_pattern = any_op_patterns_[any_it];
    }

    if (!best_pattern) break;

    // Update the pattern iterator, so that this pattern isn't attempted again.
    ++(*best_pattern_it);

    if (can_apply && !can_apply(*best_pattern)) continue;

    rewriter.setInsertionPoint(op);

    // llvm::outs() << "Trying to match " << best_pattern->GetDebugName() <<
    // "\n";
    const auto* pattern = static_cast<const RewritePattern*>(best_pattern);
    result = pattern->MatchAndRewrite(op, rewriter);
    // llvm::outs() << best_pattern->GetDebugName() << " result " <<
    // mlir::succeeded(result) << "\n";

    if (mlir::succeeded(result) && on_success &&
        mlir::failed(on_success(*best_pattern)))
      result = mlir::failure();

    if (mlir::succeeded(result)) break;

    if (on_failure) on_failure(*best_pattern);
  } while (true);

  return result;
}

}  // namespace infra
