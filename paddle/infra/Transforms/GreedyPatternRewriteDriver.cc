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

#include "Transforms/GreedyPatternRewriteDriver.h"
#include <algorithm>
#include <cassert>
#include <set>
#include "IR/PatternMatch.h"
#include "Rewrite/FrozenRewritePatternSet.h"
#include "Rewrite/PatternApplicator.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Region.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Support/LogicalResult.h"
namespace infra {

namespace {
class GreedyPatternRewriteDriver : public PatternRewriter {
 protected:
  explicit GreedyPatternRewriteDriver(mlir::MLIRContext* ctx,
                                      const FrozenRewritePatternSet& patterns,
                                      const GreedyRewriteConfig& config);

  /// Add the given operation to the worklist.
  void AddToWorklist(mlir::Operation* op);

  void FinalizeRootUpdate(mlir::Operation* op) override;

  void notifyOperationInserted(mlir::Operation* op) override;

  void notifyOperationRemoved(mlir::Operation* op) override;

  void notifyRootReplaced(mlir::Operation* op,
                          mlir::ValueRange replacement) override;

  /// Process ops until the worklist is empty or `config.max_num_rewrites`
  /// is reached. Return `true` if any IR was changed.
  bool ProcessWorklist();

  std::vector<mlir::Operation*> worklist_;
  llvm::DenseMap<mlir::Operation*, unsigned> worklist_map_;
  GreedyRewriteConfig config_;

  std::set<mlir::Operation*> strict_mode_filtered_ops_;

 private:
  /// Look over the provided operands for any defining operations that should
  /// be re-added to the worklist. This function should be called when an
  /// operation is modified or removed, as it may trigger futher
  /// simplifications.
  void AddOperandsToWorklist(mlir::ValueRange operands);

 public:
  /// Pop the next operation from the worklist
  mlir::Operation* PopFromWorklist();

  /// If the specified operation is in the worklist, remove it.
  void RemoveFromWorklist(mlir::Operation* op);

  PatternApplicator matcher_;
};

class RegionPatternRewriteDriver : public GreedyPatternRewriteDriver {
 public:
  explicit RegionPatternRewriteDriver(mlir::MLIRContext* ctx,
                                      const FrozenRewritePatternSet& patterns,
                                      const GreedyRewriteConfig& config,
                                      mlir::Region& region);  // NOLINT

  mlir::LogicalResult Simplify();

 private:
  mlir::Region& region_;
};

}  // namespace

GreedyPatternRewriteDriver::GreedyPatternRewriteDriver(
    mlir::MLIRContext* ctx,
    const FrozenRewritePatternSet& patterns,
    const GreedyRewriteConfig& config)
    : PatternRewriter(ctx), config_(config), matcher_(patterns) {
  assert(config.scope && "scope is not specified");
  worklist_.reserve(64);

  // Apply a simple cost model based sloely on pattern benefit.
  matcher_.ApplyDefaultCostModel();
}

bool GreedyPatternRewriteDriver::ProcessWorklist() {
  std::vector<mlir::Value> original_operands;

  bool changed = false;
  int64_t num_rewrites = 0;
  while (!worklist_.empty() &&
         (num_rewrites < config_.max_num_rewrites ||
          config_.max_num_rewrites == GreedyRewriteConfig::kNoLimit)) {
    auto* op = PopFromWorklist();
    if (op == nullptr) continue;

    // TODO(wilber): op dead. remove.
    if (mlir::isOpTriviallyDead(op)) {
      notifyOperationRemoved(op);
      op->erase();
      changed = true;
      continue;
    }

    // TODO(wilber): fold...

    // Try to match one of the patterns. The rewriter is automatically
    // notified of any necessary changes, so there is nothing else to do
    // here.
    auto can_apply = [&](const Pattern& pattern) { return true; };
    auto on_failure = [&](const Pattern& pattern) {};
    auto on_success = [&](const Pattern& pattern) { return mlir::success(); };

    mlir::LogicalResult match_result =
        matcher_.MatchAndRewrite(op, *this, can_apply, on_failure, on_success);

    if (mlir::succeeded(match_result)) {
      changed = true;
      ++num_rewrites;
    }
  }

  return changed;
}

void GreedyPatternRewriteDriver::AddToWorklist(mlir::Operation* op) {
  if (config_.strict_model == GreedyRewriteStrictness::AnyOp ||
      strict_mode_filtered_ops_.count(op)) {
    if (worklist_map_.count(op)) return;
    worklist_map_[op] = worklist_.size();
    worklist_.push_back(op);
  }
}

void GreedyPatternRewriteDriver::FinalizeRootUpdate(mlir::Operation* op) {
  AddToWorklist(op);
}

void GreedyPatternRewriteDriver::notifyOperationInserted(mlir::Operation* op) {
  if (config_.strict_model == GreedyRewriteStrictness::ExistingAndNewOps)
    strict_mode_filtered_ops_.insert(op);
  AddToWorklist(op);
}

void GreedyPatternRewriteDriver::AddOperandsToWorklist(
    mlir::ValueRange operands) {
  for (mlir::Value operand : operands) {
    // If the use count of this operand is now < 2, we re-add the defining
    // operation to the worklist.
    // TODO(MLIR): This is based on the fact that zero use operations
    // may be deleted, and that single use values often have more
    // canonicalization opportunities.
    if (!operand) continue;
    if (!(operand.use_empty() || operand.hasOneUse())) continue;
    if (auto* def_op = operand.getDefiningOp()) AddToWorklist(def_op);
  }
}

void GreedyPatternRewriteDriver::notifyOperationRemoved(mlir::Operation* op) {
  AddOperandsToWorklist(op->getOperands());
  op->walk([this](mlir::Operation* operation) {
    RemoveFromWorklist(operation);
    // TODO(wilber): folder.....
  });

  if (config_.strict_model != GreedyRewriteStrictness::AnyOp)
    strict_mode_filtered_ops_.erase(op);
}

void GreedyPatternRewriteDriver::notifyRootReplaced(
    mlir::Operation* op, mlir::ValueRange replacement) {
  for (auto result : op->getResults())
    for (auto* user : result.getUsers()) AddToWorklist(user);
}

mlir::Operation* GreedyPatternRewriteDriver::PopFromWorklist() {
  auto* op = worklist_.back();
  worklist_.pop_back();

  if (op) worklist_map_.erase(op);
  return op;
}

void GreedyPatternRewriteDriver::RemoveFromWorklist(mlir::Operation* op) {
  auto it = worklist_map_.find(op);
  if (it != worklist_map_.end()) {
    assert(worklist_[it->second] == op && "malformed worklist data structure");
    worklist_[it->second] = nullptr;
    worklist_map_.erase(it);
  }
}

RegionPatternRewriteDriver::RegionPatternRewriteDriver(
    mlir::MLIRContext* ctx,
    const FrozenRewritePatternSet& patterns,
    const GreedyRewriteConfig& config,
    mlir::Region& region)
    : GreedyPatternRewriteDriver(ctx, patterns, config), region_(region) {
  // Populate strict mode ops.
  if (config.strict_model != GreedyRewriteStrictness::AnyOp) {
    region.walk(
        [&](mlir::Operation* op) { strict_mode_filtered_ops_.insert(op); });
  }
}

mlir::LogicalResult RegionPatternRewriteDriver::Simplify() {
  auto InsertKnownConstant = [&](mlir::Operation*) {
    // TODO(wilber): support fold after ir ready.
    return false;
  };

  bool changed = false;
  int64_t iteration = 0;
  do {
    // Check if the iteration limit was reached.
    if (iteration++ >= config_.max_iterations &&
        config_.max_iterations != GreedyRewriteConfig::kNoLimit)
      break;

    worklist_.clear();
    worklist_map_.clear();

    for (auto& block : region_.getBlocks()) {
      for (auto& iop : block.getOperations()) {
        if (!InsertKnownConstant(&iop)) worklist_.push_back(&iop);
      }
    }
    if (config_.use_top_down_traversal) {
      // Reverse the list so our pop-back loop processes them in-order.
      std::reverse(worklist_.begin(), worklist_.end());
      for (size_t i = 0, e = worklist_.size(); i != e; ++i)
        worklist_map_[worklist_[i]] = i;
    }

    changed = ProcessWorklist();

    // TODO(wilber): SimplifyRegions....
    // if (config_.enable_region_simplification) {}
  } while (changed);

  return mlir::success(!changed);
}

mlir::LogicalResult ApplyPatternsGreedily(
    mlir::Region& region,  // NOLINT
    const FrozenRewritePatternSet& patterns,
    GreedyRewriteConfig config) {
  if (!config.scope) config.scope = &region;

  RegionPatternRewriteDriver driver(
      region.getContext(), patterns, config, region);
  mlir::LogicalResult coverged = driver.Simplify();
  if (mlir::failed(coverged)) {
    llvm::outs() << "The pattern rewrite did not converge after scanning "
                 << config.max_iterations << " times\n";
  }
  return coverged;
}

}  // namespace infra
