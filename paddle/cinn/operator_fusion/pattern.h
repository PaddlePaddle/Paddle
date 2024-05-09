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

#include <unordered_set>
#include <variant>
#include <vector>
#include "glog/logging.h"
#include "paddle/cinn/operator_fusion/utils.h"
#include "paddle/pir/include/core/operation.h"

namespace cinn::fusion {

template <typename T>
struct PatternContent {};

template <typename T>
class TrivialPattern {};

template <typename T>
class ReducePattern {};

template <typename T>
struct ReduceTreePattern {
  explicit ReduceTreePattern(const std::vector<ReduceTreePattern<T>>& childs,
                             const ReducePattern<T>& root)
      : childs_(childs), root_(root) {}
  const ReducePattern<T>& GetRootPattern() const { return root_; }
  std::vector<pir::Operation*> ops() const {
    std::vector<pir::Operation*> result{root_.ops()};
    for (const auto& child : childs_) {
      result = UniqueConcatVector(result, child.ops());
    }
    return result;
  }
  static std::string name() { return "ReduceTree"; }
  const std::vector<ReduceTreePattern<T>>& childs() const { return childs_; }
  std::vector<ReduceTreePattern<T>>& childs() { return childs_; }
  void InsertChild(const ReduceTreePattern<T>& child) {
    childs_.push_back(child);
  }
  std::vector<ReducePattern<T>> FlattenReducePattern() const {
    std::vector<ReducePattern<T>> result;
    for (const auto& child : childs_) {
      result = ConcatVector(result, child.FlattenReducePattern());
    }
    return result;
  }

 private:
  std::vector<ReduceTreePattern<T>> childs_;
  ReducePattern<T> root_;
};

template <typename T>
struct ReduceTreePlusTrivialPattern {
  explicit ReduceTreePlusTrivialPattern(const ReduceTreePattern<T>& tree,
                                        const TrivialPattern<T>& sink_trivial)
      : tree(tree), sink_trivial(sink_trivial) {}
  ReduceTreePattern<T> tree;
  TrivialPattern<T> sink_trivial;
  std::vector<pir::Operation*> ops() const {
    return UniqueConcatVector(tree.ops(), sink_trivial.ops());
  }
  static std::string name() { return "ReduceTree+Trivial"; }
  std::vector<size_t> fake_reduce_iter_idx;
};

template <typename T>
struct StmtPattern;

template <typename T>
struct UnsupportPattern {};

template <typename T>
struct HorizontalFusionPattern {
  struct PaddingStmtPattern {
    StmtPattern<T> pattern;
    std::vector<int> padding_pos;
    PaddingStmtPattern(const StmtPattern<T>& pattern,
                       const std::vector<int>& padding_pos)
        : pattern(pattern), padding_pos(padding_pos) {}
  };
  explicit HorizontalFusionPattern(
      const std::vector<PaddingStmtPattern>& patterns)
      : padding_patterns_(patterns) {}
  std::vector<PaddingStmtPattern> padding_patterns_;
  std::vector<pir::Operation*> ops() const {
    std::vector<pir::Operation*> result;
    for (const auto& pattern : padding_patterns_) {
      auto ops = GetOpsInPattern(pattern.pattern);
      ExtendVector(&result, ops);
    }
    return result;
  }
  static std::string name() { return "HorizontalFusionPattern"; }
};

template <typename T>
using StmtPatternBase = std::variant<TrivialPattern<T>,
                                     ReducePattern<T>,
                                     ReduceTreePattern<T>,
                                     ReduceTreePlusTrivialPattern<T>,
                                     HorizontalFusionPattern<T>,
                                     UnsupportPattern<T>>;

template <typename T>
struct StmtPattern final : public StmtPatternBase<T> {
  using StmtPatternBase<T>::StmtPatternBase;
  const StmtPatternBase<T>& variant() const {
    return static_cast<const StmtPatternBase<T>&>(*this);
  }
};
}  // namespace cinn::fusion
