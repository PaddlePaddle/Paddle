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

#pragma once

#include "paddle/pir/include/core/dll_decl.h"
#include "paddle/pir/include/core/region.h"

namespace pir {

using VALUE_REPLACED_HOOK_FUNC = std::function<void(pir::Value, pir::Value)>;

class FrozenRewritePatternSet;

/// This enum will control which ops will be added to the worklist during the
/// match rewrite process
enum class IR_API GreedyRewriteStrictness {
  /// No restrictions wrt. any ops are processed.
  AnyOp,
  /// Only pre-existing and newly created ops are processed.
  ExistingAndNewOps,
  /// Only pre-existing ops are processed.
  ExistingOps
};

/// Control over how the GreedyPatternRewriteDriver works.
class IR_API GreedyRewriteConfig {
 public:
  /// Control the way op is added to the worklist: bottom-up or top-down.
  bool use_top_down_traversal = false;

  /// Control the maximum number of iterations in the process of applying the
  /// pattern, use `kNolimit` to represent unlimited.
  int64_t max_iterations = 10;

  /// Control the upper limit of rewrite times during each iteration, use
  /// kNoLimit to represent unlimited.
  int64_t max_num_rewrites = kNoLimit;

  /// Only the op inside this region will be added to the worklist.
  Region* region{nullptr};

  /// Limit which ops will be added to the worklist during the Match and Rewrite
  /// process.
  /// - AnyOp: all ops will be added to the worklist.
  /// - ExistingAndNewOps: pre-existing ops and newly created ops are added to
  /// the worklist.
  /// - ExistingOps: only pre-existing ops are added to the worklist.
  GreedyRewriteStrictness strict_mode = GreedyRewriteStrictness::AnyOp;

  // Hook function for replacing the value.
  VALUE_REPLACED_HOOK_FUNC value_replaced_hook = nullptr;

  static constexpr int64_t kNoLimit = -1;
};

/// Perform the Match and Rewrite process in the specified region, greedily
/// apply the Pattern with the highest benefit, and repeat this process until
/// convergence or the upper limit of iterations.
///
/// Returns pair<bool,int64_t>
// the first is true if the iteration converges and no patterns can be applied.
// the second is the number of total match count.
std::pair<bool, int64_t> IR_API
ApplyPatternsGreedily(Region& region,  // NOLINT
                      const FrozenRewritePatternSet& patterns,
                      GreedyRewriteConfig config = GreedyRewriteConfig());

/// Perform a match and rewrite process for all regions of a given op.
IR_API std::pair<bool, int64_t> ApplyPatternsGreedily(
    Operation* op,
    const FrozenRewritePatternSet& patterns,
    GreedyRewriteConfig config = GreedyRewriteConfig());

}  // namespace pir
