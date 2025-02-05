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

#include "paddle/cinn/ir/stmt.h"

namespace cinn {
namespace optim {

/**
 * Simplifies index expressions for local memory access in GPU kernels.
 *
 * This pass optimizes scenarios where GPU kernels use complex index expressions
 * to access local memory. It's particularly effective for compute-intensive
 * algorithms that heavily utilize shared memory.
 *
 * The pass analyzes and simplifies index expressions for local GPU memory
 * accesses. It extracts common factors (GCD, offset, or symbolic) and
 * transforms indices into a more efficient iterator-based form. This
 * optimization affects Load and Store operations targeting GPU local memory.
 *
 * Execution flow:
 * 1. Collect all index expressions for local GPU memory accesses.
 * 2. Apply three types of common factor elimination:
 *    a) GCD (Greatest Common Divisor) elimination.
 *       e.g., ([2i, 4i], [4i, 3i]) -> ([i, 4i], [2i, 3i])
 *    b) Offset elimination.
 *       e.g., ([i+2, i+3], [i+4, i+6]) -> ([i, i], [i+2, i+3])
 *    c) Symbolic common factor elimination.
 *       e.g., ([C, 2], [3C, 4]) -> ([1, 2], [3, 4])
 * 3. Update the IR, replacing original indices with simplified versions.
 * 4. Transform local buffer indices into iterator-based forms.
 *       e.g., [i, 0, 0] -> [0, 0, i]
 *
 * Key benefits:
 * 1. Reduces computational overhead in index calculations.
 * 2. Decreases the required size of local memory buffers.
 *
 * Safety and correctness:
 * This optimization is safe because local tensors are initialized within the
 * function, and the IR doesn't depend on their values before initialization.
 * The pass ensures that equivalent indices before simplification remain
 * equivalent after, and distinct indices remain distinct.
 *
 * Performance impact:
 * - Simplifies complex index calculations, reducing arithmetic operations.
 * - Can reduce the size of local memory buffers, e.g., transforming indices
 *   [2i] and [4i] to [i] and [2i], potentially halving the required buffer
 * size.
 * - Works well with subsequent optimizations like ResizeBufferToMaxVarRange for
 *   further memory savings.
 *
 * Example 1: GCD elimination
 * Input:
 *   for (int i = 0; i < 100; i++) {
 *     local_tensor[4*i, 0, 0] = global_tensor[i, 0, 0];
 *     local_tensor[2*i, 0, 0] = local_tensor[4*i, 0, 0];
 *   }
 * Output:
 *   for (int i = 0; i < 100; i++) {
 *     local_tensor[0, 0, 2*i] = global_tensor[i, 0, 0];
 *     local_tensor[0, 0, i] = local_tensor[0, 0, 2*i];
 *   }
 *
 * Example 2: Offset elimination
 * Input:
 *   for (int i = 0; i < 100; i++) {
 *     local_tensor[i+2, 0, 0] = global_tensor[i, 0, 0];
 *     local_tensor[i+4, 0, 0] = local_tensor[i+2, 0, 0];
 *   }
 * Output:
 *   for (int i = 0; i < 100; i++) {
 *     local_tensor[0, 0, i] = global_tensor[i, 0, 0];
 *     local_tensor[0, 0, i+2] = local_tensor[0, 0, i];
 *   }
 *
 * Example 3: Symbolic common factor elimination
 * Input:
 *     local_tensor[C, 0, 0] = global_tensor[i, 0, 0];
 *     // (C is a symbolic constant)
 * Output:
 *     local_tensor[0, 0, 0] = global_tensor[i, 0, 0];
 */
void EliminateCommonFactorOfLocalIndex(ir::stmt::BlockRef func_body);

}  // namespace optim
}  // namespace cinn
