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

#include "paddle/pir/include/pattern_rewrite/pattern_rewrite_driver.h"

#include <glog/logging.h>
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <numeric>
#include <unordered_map>
#include <unordered_set>

#include "paddle/pir/include/core/block.h"
#include "paddle/pir/include/core/ir_context.h"
#include "paddle/pir/include/core/operation.h"
#include "paddle/pir/include/core/region.h"
#include "paddle/pir/include/core/value.h"
#include "paddle/pir/include/pattern_rewrite/frozen_rewrite_pattern_set.h"
#include "paddle/pir/include/pattern_rewrite/pattern_applicator.h"
#include "paddle/pir/include/pattern_rewrite/pattern_match.h"

namespace {

class GreedyPatternRewriteDriver : public pir::PatternRewriter {
 public:
  explicit GreedyPatternRewriteDriver(
      pir::IrContext* ctx,
      const pir::FrozenRewritePatternSet& patterns,
      const pir::GreedyRewriteConfig& config)
      : pir::PatternRewriter(ctx),
        config_(config),
        region_(*config.region),
        matcher_(patterns) {
    worklist_.reserve(128);
    matcher_.ApplyDefaultCostModel();
    if (config.strict_mode != pir::GreedyRewriteStrictness::AnyOp) {
      for (auto& block : region_) {
        for (auto& op_item : block) {
          strict_mode_filtered_ops_.insert(&op_item);
        }
      }
    }
    if (config.value_replaced_hook) {
      value_replaced_hook_fn_ = config.value_replaced_hook;
    }
  }

  std::pair<bool, int64_t> Simplify() {
    int64_t sum_num_rewrites = 0;
    int64_t num_rewrites = 0;
    int64_t iteration = 0;
    do {
      // Check if the iteration limit was reached.
      if (iteration++ >= config_.max_iterations &&
          config_.max_iterations != pir::GreedyRewriteConfig::kNoLimit)
        break;
      VLOG(6) << "Iteration[" << iteration << "] for PatternRewrite";
      worklist_.clear();
      worklist_map_.clear();

      for (auto& block_item : region_) {
        for (auto& op_item : block_item) {
          worklist_.push_back(&op_item);
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

      num_rewrites = ProcessWorklist();
      sum_num_rewrites += num_rewrites;
    } while (num_rewrites != 0);
    bool converged = num_rewrites == 0;
    return std::make_pair(converged, sum_num_rewrites);
  }

 private:
  /// Process ops until the worklist is empty or `config.max_num_rewrites`
  /// is reached. Return `true` if any IR was changed.
  int64_t ProcessWorklist() {
    int64_t num_rewrites = 0;
    while (!worklist_.empty() &&
           (num_rewrites < config_.max_num_rewrites ||
            config_.max_num_rewrites == pir::GreedyRewriteConfig::kNoLimit)) {
      auto* op = PopFromWorklist();
      if (op == nullptr) continue;
      VLOG(6) << "PopFromWorklist, get op: " << op->name();

      // TODO(wilber): ir is dead.
      // ...

      // TODO(wilber): fold logical.
      // ...

      bool match_result = matcher_.MatchAndRewrite(op, *this);
      if (match_result) {
        ++num_rewrites;
      }
    }
    return num_rewrites;
  }

  void NotifyRootReplaced(pir::Operation* op,
                          const std::vector<pir::Value>& replacement) override {
    for (uint32_t i = 0; i < op->num_results(); ++i) {
      auto result = op->result(i);
      for (auto it = result.use_begin(); it != result.use_end(); ++it) {
        AddToWorklist(it->owner());
      }
    }
  }

  void FinalizeRootUpdate(pir::Operation* op) override { AddToWorklist(op); }

  void NotifyOperationRemoved(pir::Operation* op) override {
    for (uint32_t i = 0; i < op->num_operands(); ++i) {
      AddOperandToWorklist(op->operand_source(i));
    }

    if (op->num_regions() == 0) {
      RemoveFromWorklist(op);
    } else {
      for (uint32_t i = 0; i < op->num_regions(); ++i) {
        auto& region = op->region(i);
        for (auto& block : region) {
          for (auto& op_item : block) {
            RemoveFromWorklist(&op_item);
          }
        }
      }
    }

    if (config_.strict_mode != pir::GreedyRewriteStrictness::AnyOp) {
      strict_mode_filtered_ops_.erase(op);
    }
  }

  void NotifyOperationInserted(pir::Operation* op) override {
    if (config_.strict_mode == pir::GreedyRewriteStrictness::ExistingAndNewOps)
      strict_mode_filtered_ops_.insert(op);
    AddToWorklist(op);
  }

  void NotifyValueReplaced(pir::Value from, pir::Value to) override {
    if (value_replaced_hook_fn_) {
      value_replaced_hook_fn_(from, to);
    }
  }

  /// Add the given operation to the worklist.
  void AddToWorklist(pir::Operation* op) {
    if (config_.strict_mode == pir::GreedyRewriteStrictness::AnyOp ||
        strict_mode_filtered_ops_.count(op)) {
      if (worklist_map_.count(op)) return;

      worklist_map_[op] = worklist_.size();
      worklist_.push_back(op);
    }
  }

  void AddOperandToWorklist(pir::Value operand) {
    // If the use count of this operand is now < 2, we re-add the defining
    // operation to the worklist.
    // This is based on the fact that zero use operations may be deleted, and
    // that single use values often have more canonicalization opportunities.
    if (!operand || (!operand.use_empty() && !operand.HasOneUse())) return;

    if (auto* def_op = operand.defining_op()) AddToWorklist(def_op);
  }

  void AddOperandsToWorklist(const std::vector<pir::Value> operands) {
    for (auto& v : operands) {
      AddOperandToWorklist(v);
    }
  }

  /// Pop the next operation from the worklist
  pir::Operation* PopFromWorklist() {
    auto* op = worklist_.back();
    worklist_.pop_back();
    if (op) worklist_map_.erase(op);
    return op;
  }

  /// If the specified operation is in the worklist, remove it.
  void RemoveFromWorklist(pir::Operation* op) {
    auto it = worklist_map_.find(op);
    if (it != worklist_map_.end()) {
      worklist_[it->second] = nullptr;
      worklist_map_.erase(it);
    }
  }

 private:
  std::vector<pir::Operation*> worklist_;
  std::unordered_map<pir::Operation*, unsigned> worklist_map_;
  pir::GreedyRewriteConfig config_;
  std::unordered_set<pir::Operation*> strict_mode_filtered_ops_;
  pir::Region& region_;
  pir::PatternApplicator matcher_;
  pir::VALUE_REPLACED_HOOK_FUNC value_replaced_hook_fn_ = nullptr;
};

}  // namespace

namespace pir {

std::pair<bool, int64_t> ApplyPatternsGreedily(
    Region& region,  // NOLINT
    const FrozenRewritePatternSet& patterns,
    GreedyRewriteConfig config) {
  if (!config.region) config.region = &region;

  GreedyPatternRewriteDriver driver(region.ir_context(), patterns, config);
  auto [converged, num_rewrites] = driver.Simplify();
  if (!converged && config.max_iterations != 1) {
    LOG(WARNING) << "The pattern rewrite did not converge after scanning "
                 << config.max_iterations << " times";
  }
  return std::make_pair(converged, num_rewrites);
}

IR_API std::pair<bool, int64_t> ApplyPatternsGreedily(
    Operation* op,
    const FrozenRewritePatternSet& patterns,
    GreedyRewriteConfig config) {
  bool sum_converged = true;
  int64_t sum_num_rewrites = 0;
  for (uint32_t i = 0; i < op->num_regions(); ++i) {
    Region& region = op->region(i);
    auto [converged, num_rewrites] =
        ApplyPatternsGreedily(region, patterns, config);
    sum_converged &= converged;
    sum_num_rewrites += num_rewrites;
  }
  return std::make_pair(sum_converged, sum_num_rewrites);
}

}  // namespace pir
