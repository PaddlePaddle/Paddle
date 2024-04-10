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

class TrivialPattern;
class ReducePattern;
class ReduceTreePattern;
class ReduceTreePlusTrivialPattern;
class UnsupportPattern;
class HorizontalFusionPattern;

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

struct TrivialPattern {
  explicit TrivialPattern(const std::vector<pir::Operation*>& ops)
      : ops_(ops) {}
  std::vector<pir::Operation*> ops_;
  static std::string name() { return "Trivial"; }
  std::vector<pir::Operation*> ops() const { return ops_; }
};

struct ReducePattern {
  explicit ReducePattern(const std::vector<pir::Operation*>& ops) : ops_(ops) {}
  std::vector<pir::Operation*> ops_;
  std::vector<pir::Operation*> ops() const { return ops_; }
  pir::Operation* GetReduceOp() const { return ops_.back(); }
  static std::string name() { return "Reduce"; }
};

struct ReduceTreePattern {
  explicit ReduceTreePattern(const std::vector<ReducePattern>& v,
                             const ReducePattern& root)
      : reduce_patterns_(v), root_(root) {}
  std::vector<ReducePattern> reduce_patterns_;
  const ReducePattern& GetRootPattern() const { return root_; }
  std::vector<pir::Operation*> ops() const {
    std::vector<pir::Operation*> result;
    for (const auto& reduce_pattern : reduce_patterns_) {
      result = MergeVector(result, reduce_pattern.ops());
    }
    return result;
  }
  static std::string name() { return "ReduceTree"; }

 private:
  ReducePattern root_;
};

struct ReduceTreePlusTrivialPattern {
  explicit ReduceTreePlusTrivialPattern(const ReduceTreePattern& tree,
                                        const TrivialPattern& sink_trivial)
      : tree(tree), sink_trivial(sink_trivial) {}
  ReduceTreePattern tree;
  TrivialPattern sink_trivial;
  std::vector<pir::Operation*> ops() const {
    return MergeVector(tree.ops(), sink_trivial.ops());
  }
  static std::string name() { return "ReduceTree+Trivial"; }
  std::vector<size_t> fake_reduce_iter_idx;
};

struct UnsupportPattern {
  explicit UnsupportPattern(const std::vector<pir::Operation*>& ops)
      : ops_(ops) {}
  std::vector<pir::Operation*> ops_;
  std::vector<pir::Operation*> ops() const { return ops_; }
  static std::string name() { return "Unsupport"; }
};

struct HorizontalFusionPattern {
  explicit HorizontalFusionPattern(const std::vector<pir::Operation*>& ops)
      : ops_(ops) {}
  std::vector<pir::Operation*> ops_;
  std::vector<pir::Operation*> ops() const { return ops_; }
  static std::string name() { return "HorizontalFusionPattern"; }
};

using StmtPattern = std::variant<TrivialPattern,
                                 ReducePattern,
                                 ReduceTreePattern,
                                 ReduceTreePlusTrivialPattern,
                                 HorizontalFusionPattern,
                                 UnsupportPattern>;

}  // namespace cinn::frontend::group_cluster
