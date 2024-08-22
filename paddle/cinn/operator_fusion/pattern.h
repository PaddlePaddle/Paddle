// Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

#include <functional>
#include <unordered_set>
#include <variant>
#include <vector>
#include "glog/logging.h"
#include "paddle/cinn/operator_fusion/fusion_tracker/tracker.h"
#include "paddle/cinn/operator_fusion/pir_graph_analyzing/anchor_transform.h"
#include "paddle/cinn/operator_fusion/utils.h"
#include "paddle/pir/include/core/operation.h"

namespace cinn::fusion {

struct PatternContent {
  explicit PatternContent(pir::Operation* op) : op(op) {}
  pir::Operation* op;
  bool operator==(const PatternContent& other) const { return op == other.op; }
};

struct TrivialPattern {
  explicit TrivialPattern(const std::vector<pir::Operation*>& ops,
                          pir::Operation* sink_op,
                          const FusionTrackerPtr& tracker)
      : ops_(ops), sink_op_(sink_op), tracker_(tracker) {
    id_ = UniqueId();
  }
  std::vector<pir::Operation*> ops_;
  pir::Operation* sink_op_;
  std::vector<pir::Operation*> ops() const { return ops_; }
  pir::Operation* sink_op() const { return sink_op_; }

  static std::string name() { return "Trivial"; }

  static std::string UniqueId() {
    static std::atomic<int64_t> counter = 0;
    counter += 1;
    return name() + "_" + std::to_string(counter);
  }
  std::string id() const { return id_; }
  std::string id_;

  FusionTrackerPtr tracker_;
  void update_tracker() const {}
};

struct ReducePattern {
  explicit ReducePattern(const std::vector<pir::Operation*>& ops,
                         const FusionTrackerPtr& tracker)
      : ops_(ops), tracker_(tracker) {
    id_ = UniqueId();
  }
  std::vector<pir::Operation*> ops_;
  std::vector<pir::Operation*> ops() const { return ops_; }
  pir::Operation* GetReduceOp() const { return ops_.back(); }

  static std::string name() { return "Reduce"; }

  static std::string UniqueId() {
    static std::atomic<int64_t> counter = 0;
    counter += 1;
    return name() + "_" + std::to_string(counter);
  }
  std::string id() const { return id_; }
  std::string id_;

  FusionTrackerPtr tracker_;
  void update_tracker() const {}
};

struct ReduceTreePattern {
  explicit ReduceTreePattern(const std::vector<ReduceTreePattern>& childs,
                             const ReducePattern& root,
                             const FusionTrackerPtr& tracker)
      : childs_(childs), root_(root), tracker_(tracker) {
    id_ = UniqueId();
    cur_id_ = id_;
  }
  const ReducePattern& GetRootPattern() const { return root_; }
  std::vector<pir::Operation*> ops() const {
    std::vector<pir::Operation*> result{root_.ops()};
    for (const auto& child : childs_) {
      result = UniqueConcatVector(result, child.ops());
    }
    return result;
  }
  const std::vector<ReduceTreePattern>& childs() const { return childs_; }
  std::vector<ReduceTreePattern>& childs() { return childs_; }
  void InsertChild(const ReduceTreePattern& child) { childs_.push_back(child); }
  std::vector<ReducePattern> FlattenReducePattern() const {
    std::vector<ReducePattern> result{root_};
    for (const auto& child : childs_) {
      result = ConcatVector(result, child.FlattenReducePattern());
    }
    return result;
  }

  static std::string name() { return "ReduceTree"; }

  static std::string UniqueId() {
    static std::atomic<int64_t> counter = 0;
    counter += 1;
    return name() + "_" + std::to_string(counter);
  }
  std::string id() const { return id_; }
  std::string id_;

  mutable std::string cur_id_;
  std::string cur_id() const { return cur_id_; }
  void reset_cur_id(std::string id) const { cur_id_ = id; }

  FusionTrackerPtr tracker_;

  void update_tracker() const {
    const std::string& root_name = GetRootPattern().id();
    std::vector<std::string> names;
    UpdateTrackerImpl(
        root_name, *this, std::vector<size_t>(), this->tracker_, &names);
    tracker_->append(std::make_shared<CombineInstr>(names, cur_id()));
  }

  void UpdateTrackerImpl(const std::string root_name,
                         const ReduceTreePattern& root,
                         const std::vector<size_t>& fake_reduce_iter_idx,
                         FusionTrackerPtr tracker,
                         std::vector<std::string>* names) const {
    // Apply a brunch of tracker to get a output_name of ReduceTreePattern.
    // names and trackers collect all the needed fusion nodes.
    for (const auto& child : root.childs()) {
      auto origin_child_id = child.cur_id();
      auto new_child_id = GetNewTmpId(origin_child_id);
      child.reset_cur_id(new_child_id);
      tracker->append(
          std::make_shared<TmpTransformInstr>(origin_child_id,
                                              root_name,
                                              new_child_id,
                                              root.cur_id(),
                                              fake_reduce_iter_idx));
      UpdateTrackerImpl(
          new_child_id, child, fake_reduce_iter_idx, tracker, names);
    }
    names->push_back(root.cur_id());
  }

 private:
  std::vector<ReduceTreePattern> childs_;
  ReducePattern root_;
};

struct ReduceTreePlusTrivialPattern {
  explicit ReduceTreePlusTrivialPattern(const ReduceTreePattern& tree,
                                        const TrivialPattern& sink_trivial,
                                        const FusionTrackerPtr& tracker)
      : tree(tree), sink_trivial(sink_trivial), tracker_(tracker) {
    id_ = UniqueId();
  }
  ReduceTreePattern tree;
  TrivialPattern sink_trivial;
  std::vector<pir::Operation*> ops() const {
    return UniqueConcatVector(tree.ops(), sink_trivial.ops());
  }
  std::vector<size_t> fake_reduce_iter_idx;

  static std::string name() { return "ReduceTreePlusTrivial"; }

  static std::string UniqueId() {
    static std::atomic<int64_t> counter = 0;
    counter += 1;
    return name() + "_" + std::to_string(counter);
  }
  std::string id() const { return id_; }
  std::string id_;

  FusionTrackerPtr tracker_;

  void update_tracker() const {
    const std::string& root_name = id();
    const std::string& origin_tree_id = tree.cur_id();
    const std::string& new_tree_id = GetNewTmpId(origin_tree_id);
    tree.reset_cur_id(new_tree_id);
    std::vector<std::string> names;
    tracker_->append(std::make_shared<TmpTransformInstr>(origin_tree_id,
                                                         sink_trivial.id(),
                                                         new_tree_id,
                                                         root_name,
                                                         fake_reduce_iter_idx));
    tree.UpdateTrackerImpl(
        new_tree_id, tree, fake_reduce_iter_idx, this->tracker_, &names);
    names.push_back(root_name);
    // optimize the loop range of R + T for speed up.
    tracker_->append(std::make_shared<TrivialLoopAlignInstr>(
        new_tree_id, root_name, root_name, fake_reduce_iter_idx));
    // collect all the Expr and represent the root_name.
    tracker_->append(std::make_shared<CombineInstr>(names, root_name));
  }
};

struct AnchorPattern {
  explicit AnchorPattern(const std::vector<pir::Operation*>& ops,
                         const pir::Value& anchor,
                         const AnchorState& init_anchor_state,
                         const FusionTrackerPtr& tracker)
      : ops_(ops),
        anchor_(anchor),
        anchor_state(init_anchor_state),
        tracker_(tracker) {
    id_ = UniqueId();
  }
  AnchorState anchor_state;

  std::vector<pir::Operation*> ops() const { return ops_; }
  pir::Value anchor() const { return anchor_; }
  bool can_recompute() const {
    // Current Algorithm:
    // An AnchorPattern can be recomputed if:
    // 1. It didn't go through any pattern merging during prior fusions, which
    // means it only has one output_expr in anchor_state.
    // 2. It only contains trivial ops.

    if (anchor_state.promise.size() > 1) {
      return false;
    }

    for (const auto& op : ops_) {
      const auto& op_kind = GetOpPatternKind(op);
      if (op_kind >= hlir::framework::kReduction) {
        return false;
      }
    }

    return true;
  }

  static std::string name() { return "Anchor"; }

  static std::string UniqueId() {
    static std::atomic<int64_t> counter = 0;
    counter += 1;
    return name() + "_" + std::to_string(counter);
  }
  std::string id() const { return id_; }
  std::string id_;

  FusionTrackerPtr tracker_;
  void update_tracker() const {
    std::vector<std::string> tmp_names;
    for (int i = 0; i < anchor_state.promise.size(); i++) {
      auto promise = anchor_state.promise[i];
      std::string tmp_name = "tmp_" + std::to_string(i);
      tmp_names.emplace_back(tmp_name);
      tracker_->append(std::make_shared<AnchorTransformInstr>(
          promise.id_, tmp_name, promise.transform_route));
    }
    tracker_->append(std::make_shared<CombineInstr>(tmp_names, id()));
  }

 private:
  std::vector<pir::Operation*> ops_;
  pir::Value anchor_;  // Choose only one anchor
};

struct HorizontalFusionPattern {
  struct PaddingStmtPattern;
  explicit HorizontalFusionPattern(
      const std::vector<PaddingStmtPattern>& patterns,
      const FusionTrackerPtr& tracker)
      : padding_patterns_(patterns), tracker_(tracker) {
    id_ = UniqueId();
  }
  std::vector<PaddingStmtPattern> padding_patterns_;
  inline std::vector<pir::Operation*> ops() const;

  static std::string name() { return "Horizontal"; }

  static std::string UniqueId() {
    static std::atomic<int64_t> counter = 0;
    counter += 1;
    return name() + "_" + std::to_string(counter);
  }
  std::string id() const { return id_; }
  std::string id_;
  inline void update_tracker() const;
  FusionTrackerPtr tracker_;
};

struct UnsupportPattern {
  explicit UnsupportPattern(const std::vector<pir::Operation*>& ops,
                            const FusionTrackerPtr& tracker)
      : ops_(ops), tracker_(tracker) {
    id_ = UniqueId();
  }
  std::vector<pir::Operation*> ops_;
  std::vector<pir::Operation*> ops() const { return ops_; }

  static std::string name() { return "Unsupport"; }

  static std::string UniqueId() {
    static std::atomic<int64_t> counter = 0;
    counter += 1;
    return name() + "_" + std::to_string(counter);
  }
  std::string id() const { return id_; }
  std::string id_;

  FusionTrackerPtr tracker_;
  void update_tracker() const {}
};

using StmtPattern = std::variant<TrivialPattern,
                                 ReducePattern,
                                 ReduceTreePattern,
                                 ReduceTreePlusTrivialPattern,
                                 HorizontalFusionPattern,
                                 UnsupportPattern,
                                 AnchorPattern>;

static std::string GetPatternId(const StmtPattern& s);
static std::vector<pir::Operation*> GetOpsInPattern(const StmtPattern& pattern);

struct HorizontalFusionPattern::PaddingStmtPattern {
  StmtPattern pattern;
  std::vector<int> padding_pos;
  PaddingStmtPattern(const StmtPattern& pattern,
                     const std::vector<int>& padding_pos)
      : pattern(pattern), padding_pos(padding_pos) {}
};

inline void HorizontalFusionPattern::update_tracker() const {
  std::vector<std::string> tmp_names;
  for (int i = 0; i < padding_patterns_.size(); i++) {
    auto padding_pattern = padding_patterns_[i];
    std::string tmp_name = "tmp_" + std::to_string(i);
    tmp_names.emplace_back(tmp_name);
    tracker_->append(
        std::make_shared<PaddingInstr>(GetPatternId(padding_pattern.pattern),
                                       tmp_name,
                                       padding_pattern.padding_pos));
  }
  tracker_->append(std::make_shared<CombineInstr>(tmp_names, id()));
}

inline std::vector<pir::Operation*> HorizontalFusionPattern::ops() const {
  std::vector<pir::Operation*> result;
  for (const auto& pattern : padding_patterns_) {
    auto ops = GetOpsInPattern(pattern.pattern);
    ExtendVector(&result, ops);
  }
  return result;
}

static std::string StmtPatternDebugStr(const StmtPattern& stmt) {
  std::stringstream ss;
  auto all_ops = GetOpsInPattern(stmt);
  ss << "StmtPattern, size " << all_ops.size() << " :\n";
  ss << OpsDebugStr(all_ops);
  return ss.str();
}

static std::string GetPatternName(const StmtPattern& s) {
  return std::visit([](const auto& impl) { return impl.name(); }, s);
}

static std::string GetPatternId(const StmtPattern& s) {
  return std::visit([](const auto& impl) { return impl.id(); }, s);
}

static FusionTrackerPtr GetFusionTracker(const StmtPattern& s) {
  return std::visit([](const auto& impl) { return impl.tracker_; }, s);
}

static std::vector<pir::Operation*> GetOpsInPattern(
    const StmtPattern& pattern) {
  return std::visit([](const auto& impl) { return impl.ops(); }, pattern);
}

static std::unordered_set<pir::Value> GetPatternInputValuesIncludeInner(
    const StmtPattern& A) {
  std::unordered_set<pir::Value> result;
  for (const auto& op : GetOpsInPattern(A)) {
    for (const auto& value : op->operands()) {
      result.insert(value.source());
    }
  }
  return result;
}

static std::unordered_set<pir::Value> GetPatternOutputValuesIncludedInner(
    const StmtPattern& A) {
  std::unordered_set<pir::Value> result;
  for (const auto& op : GetOpsInPattern(A)) {
    for (const auto& value : op->results()) {
      result.insert(value);
    }
  }
  return result;
}

static std::unordered_set<pir::Value> GetPatternInputValues(
    const StmtPattern& A) {
  auto all_input_values = GetPatternInputValuesIncludeInner(A);
  for (const auto& value : GetPatternOutputValuesIncludedInner(A)) {
    all_input_values.erase(value);
  }
  VLOG(4) << "GetPatternInputValues: " << all_input_values.size();
  return all_input_values;
}

static void PatternUpdateTracker(const StmtPattern& pattern) {
  return std::visit([](const auto& impl) { impl.update_tracker(); }, pattern);
}
}  // namespace cinn::fusion
