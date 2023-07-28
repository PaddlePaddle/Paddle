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

#include "paddle/ir/pattern_rewrite/pattern_rewrite_driver.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <numeric>
#include <unordered_map>
#include <unordered_set>

#include "paddle/ir/core/block.h"
#include "paddle/ir/core/ir_context.h"
#include "paddle/ir/core/operation.h"
#include "paddle/ir/core/region.h"
#include "paddle/ir/core/value.h"
#include "paddle/ir/pattern_rewrite/frozen_rewrite_pattern_set.h"
#include "paddle/ir/pattern_rewrite/pattern_applicator.h"
#include "paddle/ir/pattern_rewrite/pattern_match.h"

namespace {

class GreedyPatternRewriteDriver : public ir::PatternRewriter {
 public:
  explicit GreedyPatternRewriteDriver(
      ir::IrContext* ctx,
      const ir::FrozenRewritePatternSet& patterns,
      const ir::GreedyRewriteConfig& config)
      : ir::PatternRewriter(ctx),
        config_(config),
        region_(*config.region),
        matcher_(patterns) {
    worklist_.reserve(128);
    matcher_.ApplyDefaultCostModel();
    if (config.strict_mode != ir::GreedyRewriteStrictness::AnyOp) {
      for (auto it = region_.begin(); it != region_.end(); ++it) {
        for (auto op_it = (*it)->begin(); op_it != (*it)->end(); ++op_it) {
          strict_mode_filtered_ops_.insert(*op_it);
        }
      }
    }
  }

  bool Simplify() {
    bool changed = false;
    int64_t iteration = 0;
    do {
      // Check if the iteration limit was reached.
      if (iteration++ >= config_.max_iterations &&
          config_.max_iterations != ir::GreedyRewriteConfig::kNoLimit)
        break;
      VLOG(6) << "Iteration[" << iteration << "] for PatternRewrite";
      worklist_.clear();
      worklist_map_.clear();

      for (auto block_it = region_.begin(); block_it != region_.end();
           ++block_it) {
        for (auto op_it = (*block_it)->begin(); op_it != (*block_it)->end();
             ++op_it) {
          worklist_.push_back(*op_it);
        }
      }
      if (config_.use_top_down_traversal) {
        // Reverse the list so out pop-back loop process them in-order.
        std::reverse(worklist_.begin(), worklist_.end());
      }
      for (size_t i = 0; i < worklist_.size(); ++i) {
        worklist_map_[worklist_[i]] = i;
        VLOG(6) << "worklist[" << i << "] is " << worklist_[i]->name();
      }

      changed = ProcessWorklist();
    } while (changed);

    return !changed;
  }

 private:
  /// Process ops until the worklist is empty or `config.max_num_rewrites`
  /// is reached. Return `true` if any IR was changed.
  bool ProcessWorklist() {
    bool changed = false;
    int64_t num_rewrites = 0;

    while (!worklist_.empty() &&
           (num_rewrites < config_.max_num_rewrites ||
            config_.max_num_rewrites == ir::GreedyRewriteConfig::kNoLimit)) {
      auto* op = PopFromWorklist();
      if (op == nullptr) continue;
      VLOG(6) << "PopFromWorklist, get op: " << op->name();

      // TODO(wilber): ir is dead.
      // ...

      // TODO(wilber): fold logical.
      // ...

      bool match_result = matcher_.MatchAndRewrite(op, *this);
      if (match_result) {
        changed = true;
        ++num_rewrites;
      }
    }

    return changed;
  }

  // TODO(wilber): OpResult support GetUsers method.
  void NotifyRootReplaced(ir::Operation* op,
                          const std::vector<ir::Value>& replacement) override {
    //   for (uint32_t i = 0; i < op->num_results(); ++i) {
    //     auto res = op->GetResultByIndex(i);
    //   }
    // }
  }

  void FinalizeRootUpdate(ir::Operation* op) override { AddToWorklist(op); }

  void NotifyOperationRemoved(ir::Operation* op) override {
    for (uint32_t i = 0; i < op->num_operands(); ++i) {
      AddOperandToWorklist(op->operand_source(i));
    }
    for (uint32_t i = 0; i < op->num_regions(); ++i) {
      auto& region = op->region(i);
      for (auto it = region.begin(); it != region.end(); ++it) {
        for (auto op_it = (*it)->begin(); op_it != (*it)->end(); ++op_it) {
          RemoveFromWorklist(*op_it);
        }
      }
    }

    if (config_.strict_mode != ir::GreedyRewriteStrictness::AnyOp) {
      strict_mode_filtered_ops_.erase(op);
    }
  }

  void NotifyOperationInserted(ir::Operation* op) override {
    if (config_.strict_mode == ir::GreedyRewriteStrictness::ExistingAndNewOps)
      strict_mode_filtered_ops_.insert(op);
    AddToWorklist(op);
  }

  /// Add the given operation to the worklist.
  void AddToWorklist(ir::Operation* op) {
    if (config_.strict_mode == ir::GreedyRewriteStrictness::AnyOp ||
        strict_mode_filtered_ops_.count(op)) {
      if (worklist_map_.count(op)) return;

      worklist_map_[op] = worklist_.size();
      worklist_.push_back(op);
    }
  }

  void AddOperandToWorklist(ir::Value operand) {
    // If the use count of this operand is now < 2, we re-add the defining
    // operation to the worklist.
    // This is based on the fact that zero use operations may be deleted, and
    // that single use values often have more canonicalization opportunities.
    if (!operand || (!operand.use_empty() && !operand.HasOneUse())) return;

    if (auto* def_op = operand.GetDefiningOp()) AddToWorklist(def_op);
  }

  void AddOperandsToWorklist(const std::vector<ir::Value> operands) {
    for (auto& v : operands) {
      AddOperandToWorklist(v);
    }
  }

  /// Pop the next operation from the worklist
  ir::Operation* PopFromWorklist() {
    auto* op = worklist_.back();
    worklist_.pop_back();
    if (op) worklist_map_.erase(op);
    return op;
  }

  /// If the specified operation is in the worklist, remove it.
  void RemoveFromWorklist(ir::Operation* op) {
    auto it = worklist_map_.find(op);
    if (it != worklist_map_.end()) {
      worklist_[it->second] = nullptr;
      worklist_map_.erase(it);
    }
  }

 private:
  std::vector<ir::Operation*> worklist_;
  std::unordered_map<ir::Operation*, unsigned> worklist_map_;
  ir::GreedyRewriteConfig config_;
  std::unordered_set<ir::Operation*> strict_mode_filtered_ops_;
  ir::Region& region_;
  ir::PatternApplicator matcher_;
};

}  // namespace

namespace ir {

bool ApplyPatternsGreedily(Region& region,  // NOLINT
                           const FrozenRewritePatternSet& patterns,
                           GreedyRewriteConfig config) {
  if (!config.region) config.region = &region;

  GreedyPatternRewriteDriver driver(region.ir_context(), patterns, config);
  bool converged = driver.Simplify();
  if (!converged) {
    LOG(WARNING) << "The pattern rewrite did not converge after scaning "
                 << config.max_iterations << " times";
  }
  return converged;
}

}  // namespace ir
