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

#include <variant>
#include <vector>
#include "paddle/pir/include/core/operation.h"

namespace cinn::frontend::group_cluster {

struct TrivialPattern {
  explicit TrivialPattern(const std::vector<pir::Operation*>& ops)
      : ops_(ops) {}
  std::vector<pir::Operation*> ops_;
  std::vector<pir::Operation*> ops() const { return ops_; }
};

struct ReducePattern {
  explicit ReducePattern(const std::vector<pir::Operation*>& ops)
      : ops_(ops) {}
  std::vector<pir::Operation*> ops_;
  std::vector<pir::Operation*> ops() const { return ops_; }
  pir::Operation* GetReduceOp() const { return ops_.back(); }
};

struct ReduceTreePattern {
  explicit ReduceTreePattern(const std::vector<ReducePattern>& v,
                             const ReducePattern& root)
      : reduce_patterns_(v), root_(root) {}
  std::vector<ReducePattern> reduce_patterns_;
  const ReducePattern& GetRootPattern() const { return root_; }
  std::vector<pir::Operation*> ops() const {
    std::vector<pir::Operation*> ops;
    for (const auto& reduce_pattern : reduce_patterns_) {
      for (const auto& op : reduce_pattern.ops()) {
        ops.push_back(op);
      }
    }
    return ops;
  }

 private:
  ReducePattern root_;
};

struct ReduceTreePlusTrivialPattern {
  explicit ReduceTreePlusTrivialPattern(const ReduceTreePattern& tree,
                                        const TrivialPattern& sink_trivial)
      : tree(tree), sink_trivial(sink_trivial) {}
  ReduceTreePattern tree;
  TrivialPattern sink_trivial;
  std::vector<pir::Operation*> ops() const { return {}; }
};

struct UnsupportPattern {
  explicit UnsupportPattern(const std::vector<pir::Operation*>& ops)
      : ops_(ops) {}
  std::vector<pir::Operation*> ops_;
  std::vector<pir::Operation*> ops() const { return ops_; }
};

// UnsupportedPattern can't fuse with any pattern
// Step 1: T x T|R => T|R                 TrivialPattern can always fuse with
// downstream Step 2: R x T|R => R                   Use Shardable Axes Policy
// to judge

// If we want add MatmulPattern =>
// StmtPattern = std::variant<TrivialPattern, ReducePattern, MatmulPattern,
// UnsupportPattern>; Fusion with different Pattern will have specialized logic
// to Judge, Update policy logic for MatmulPattern
using StmtPattern = std::variant<TrivialPattern,
                                 ReducePattern,
                                 ReduceTreePattern,
                                 ReduceTreePlusTrivialPattern,
                                 UnsupportPattern>;

}  // namespace cinn::frontend::group_cluster
