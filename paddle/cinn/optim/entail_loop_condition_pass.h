// Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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
#include "paddle/cinn/pass/pass.h"

namespace cinn {
namespace optim {

/**
 * Remove the IfThenElse inside indivisible loops by entailing the if-condition
 * into the loop condition.
 *
 * Traditionally, when the element count is not divisible by the block size, an
 * IfThenElse inside the loop is needed to tell trailing blocks to break early.
 * However, the IfThenElse introduces an extra comparison in every iteration,
 * therefore heavily impacting the performance. This pass solves this problem by
 * observing that most IfThenElses are unnecessary, and can actually entail the
 * loop condition. By removing one comparison, we reduce the compute intensity
 * and bring up to 30% speedup.
 *
 * For example, in the following loop:
 *   for (k = 0; k < 8; k += 1) {
 *     if (((k * 32) + thread.x) < 234) {
 *       var_1[0] = var_1[0] + var[((block.x * 234) + (k * 32)) + thread.x];
 *     }
 *   }
 * The if-condition actually entails the loop condition, i.e. whenever
 * `((k * 32) + thread.x) < 234` is true, the `k < 8` is also true. This can
 * be proved by setting k = 8, then regardless of the value of thread.x, the
 * `(k * 32) + thread.x` will be at least 256, which violates the premise.
 *
 * Therefore, we can override the loop condition with the if-condition,
 * eliminating one comparison, resulting in a new loop:
 *   for (k = 0; ((k * 32) + thread.x) < 234; k += 1) {
 *     var_1[0] = var_1[0] + var[((block.x * 234) + (k * 32)) + thread.x];
 *   }
 *
 * Furthermore, after removing the loop condition, we can see that `k * 32`
 * becomes a common expression in the whole loop, so we can replace it with
 * `k_strided = k * 32`. The finally optimized loop is:
 *   for (k_stride = 0; (k_stride + thread.x) < 234; k_stride += 32) {
 *     var_1[0] = var_1[0] + var[((block.x * 234) + k_stride) + thread.x];
 *   }
 *
 *
 * Implementation Notes:
 *   To put the if-condition into the loop extent, we need the PolyFor node,
 *   which is currently not available in the new CINN IR. To quickly bring this
 *   pass online, we temporarily use a macro
 *     CINN_ENTAIL_LOOP_CONDITION(__loop_var, __cond, __stride)
 *   that plays some tricks at C level to change the loop structure. This is
 *   very non-standard and should be replaced by a safer method later.
 */
class EntailLoopConditionPass : public BlockPass {
 public:
  EntailLoopConditionPass() : BlockPass("entail_loop_condition") {}
  LogicalResult Run(ir::stmt::BlockRef func) override;
};

std::unique_ptr<BlockPass> CreateEntailLoopConditionPass();

}  // namespace optim
}  // namespace cinn
