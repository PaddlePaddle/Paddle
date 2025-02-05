// Copyright (c) 2024 CINN Authors. All Rights Reserved.
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
#include <vector>

#include "paddle/cinn/common/common.h"
#include "paddle/cinn/ir/ir.h"

namespace cinn {
namespace optim {

/**
 * This pass handles the cross-block reduction properly.
 *
 * Specific transformations:
 * 1. Reorders the schedule blocks to separate upstreams and downstreams
 *    of cross-block reduction.
 * 2. Replaces the cross-block reduction with an external call to the
 *    `grid_reduce` template function.
 * 3. Adds a condition check `is_last_block_done` to the grid reduce and all
 *    subsequent schedule blocks.
 * 4. Pushes global buffers (`rf` and `semaphore`) to the function’s argument
 *    list.
 *
 * Example:
 *
 * function reduce_sum (..., var_1)
 * {
 *   thread_bind[blockIdx.x] for (i, 0, 16):
 *     thread_bind[blockIdx.y] for (j, 0, 8): // reduce axis
 *       var_1[i] += var_1_rf[j, i]
 * }
 *
 * After pass:
 *
 * function reduce_sum (..., var_1, var_1_rf, semaphore)
 * {
 *   thread_bind[blockIdx.x] for (i, 0, 16):
 *     thread_bind[blockIdx.y] for (j, 0, 8): // reduce axis
 *       is_last_block_done = update_semaphore(semaphore)
 *       if (is_last_block_done):
 *         var_1[i] = grid_reduce_sum(var_1_rf)
 * }
 */
void ReplaceCrossBlockReduction(ir::LoweredFunc fn);

}  // namespace optim
}  // namespace cinn
