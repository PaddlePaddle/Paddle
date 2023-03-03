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

/// The design and code is mainly from MLIR, thanks to the greate project.

#pragma once

#include <cstdint>
#include "Rewrite/FrozenRewritePatternSet.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Region.h"
#include "mlir/Support/LogicalResult.h"
namespace infra {

/// This enum controls which ops are put on the worklist during a greedy
/// pattern rewrite.
enum class GreedyRewriteStrictness {
  /// No restrictions wrt. which ops are processed.
  AnyOp,
  /// Only pre-existing and newly created ops are processed.
  ExistingAndNewOps,
  /// Only pre-existing ops are processed.
  ExistingOps
};

/// Control over how the GreedyPatternRewriteDriver works.
class GreedyRewriteConfig {
 public:
  ///
  bool use_top_down_traversal = false;

  ///
  bool enable_region_simplification = true;

  int64_t max_iterations = 10;

  int64_t max_num_rewrites = kNoLimit;

  static constexpr int64_t kNoLimit = -1;

  mlir::Region* scope = nullptr;

  GreedyRewriteStrictness strict_model = GreedyRewriteStrictness::AnyOp;
};

mlir::LogicalResult ApplyPatternsGreedily(
    mlir::Region& region,  // NOLINT
    const FrozenRewritePatternSet& patterns,
    GreedyRewriteConfig config = GreedyRewriteConfig());

inline mlir::LogicalResult ApplyPatternsGreedily(
    mlir::Operation* op,
    const FrozenRewritePatternSet& patterns,
    GreedyRewriteConfig config = GreedyRewriteConfig()) {
  bool failed = false;
  for (mlir::Region& region : op->getRegions())
    failed |= ApplyPatternsGreedily(region, patterns, config).failed();
  return mlir::failure(failed);
}

}  // namespace infra
