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
#include "paddle/pir/include/core/operation.h"

namespace cinn::frontend::group_cluster {

template <typename T>
struct PatternContent {};

template <typename T>
class TrivialPattern;
template <typename T>
class ReducePattern;
template <typename T>
class ReduceTreePattern;
template <typename T>
class ReduceTreePlusTrivialPattern;
template <typename T>
class UnsupportPattern;
template <typename T>
class HorizontalFusionPattern;

template <typename T>
using StmtPattern = std::variant<TrivialPattern<T>,
                                 ReducePattern<T>,
                                 ReduceTreePattern<T>,
                                 ReduceTreePlusTrivialPattern<T>,
                                 HorizontalFusionPattern<T>,
                                 UnsupportPattern<T>>;

template <typename T>
void ExtendVector(std::vector<T>* first, const std::vector<T>& second) {
  std::unordered_set<T> visited =
      std::unordered_set<T>(first->begin(), first->end());
  for (auto iter = second.begin(); iter != second.end(); iter++) {
    if (visited.find(*iter) == visited.end()) {
      visited.emplace(*iter);
      first->emplace_back(*iter);
    }
  }
}

template <typename T>
std::vector<T> MergeVector(const std::vector<T>& first,
                           const std::vector<T>& second) {
  std::vector<T> result = std::vector<T>(first);
  ExtendVector(&result, second);
  return result;
}

template <typename T>
struct TrivialPattern {
  explicit TrivialPattern(const std::vector<PatternContent<T>>& contents)
      : contents_(contents) {}
  std::vector<PatternContent<T>> contents_;
  static std::string name() { return "Trivial"; }
  std::vector<PatternContent<T>> contents() const { return contents_; }
};

template <typename T>
struct ReducePattern {
  explicit ReducePattern(const std::vector<PatternContent<T>>& contents)
      : contents_(contents) {}
  std::vector<PatternContent<T>> contents_;
  std::vector<PatternContent<T>> contents() const { return contents_; }
  pir::Operation* GetReduceOp() const { return contents_.back().op; }
  static std::string name() { return "Reduce"; }
};

template <typename T>
struct ReduceTreePattern {
  explicit ReduceTreePattern(const std::vector<ReducePattern<T>>& v,
                             const ReducePattern<T>& root)
      : reduce_patterns_(v), root_(root) {}
  const ReducePattern<T>& GetRootPattern() const { return root_; }
  std::vector<PatternContent<T>> contents() const {
    std::vector<PatternContent<T>> result;
    for (const auto& reduce_pattern : reduce_patterns_) {
      result = MergeVector(result, reduce_pattern.contents());
    }
    return result;
  }
  static std::string name() { return "ReduceTree"; }
  const std::vector<ReducePattern<T>>& reduce_patterns() const {
    return reduce_patterns_;
  }

 private:
  std::vector<ReducePattern<T>> reduce_patterns_;
  ReducePattern<T> root_;
};

template <typename T>
struct ReduceTreePlusTrivialPattern {
  explicit ReduceTreePlusTrivialPattern(const ReduceTreePattern<T>& tree,
                                        const TrivialPattern<T>& sink_trivial)
      : tree(tree), sink_trivial(sink_trivial) {}
  ReduceTreePattern<T> tree;
  TrivialPattern<T> sink_trivial;
  std::vector<PatternContent<T>> contents() const {
    return MergeVector(tree.contents(), sink_trivial.contents());
  }
  static std::string name() { return "ReduceTree+Trivial"; }
  std::vector<size_t> fake_reduce_iter_idx;
};

template <typename T>
struct UnsupportPattern {
  explicit UnsupportPattern(const std::vector<PatternContent<T>>& contents)
      : contents_(contents) {}
  std::vector<PatternContent<T>> contents_;
  std::vector<PatternContent<T>> contents() const { return contents_; }
  static std::string name() { return "Unsupport"; }
};

template <typename T>
struct HorizontalFusionPattern {
  explicit HorizontalFusionPattern(const std::vector<StmtPattern<T>>& patterns)
      : patterns_(patterns) {}
  std::vector<StmtPattern<T>> patterns_;
  std::vector<PatternContent<T>> contents() const;
  static std::string name() { return "HorizontalFusionPattern"; }
};

template <typename T>
std::vector<PatternContent<T>> HorizontalFusionPattern<T>::contents() const {
  std::vector<PatternContent<T>> result;
  for (const auto& pattern : patterns_) {
    auto contents = GetContentsInPattern(pattern);
    ExtendVector(&result, contents);
  }
  return result;
}

}  // namespace cinn::frontend::group_cluster
