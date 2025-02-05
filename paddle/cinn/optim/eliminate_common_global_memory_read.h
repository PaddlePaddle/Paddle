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

#include "paddle/cinn/ir/ir.h"

namespace cinn {
namespace optim {

/**
 *
 * This pass eliminates redundant global memory reads by substituting them with
 * local memory buffers.
 *
 * This pass is applicable in scenarios where multiple identical global memory
 * reads occur within the same scope of computation, such as loop nests or
 * blocks of code with shared memory access patterns. By identifying such
 * redundant reads, it can improve performance by reducing memory bandwidth
 * usage and improving cache locality.
 *
 * When applied, this pass performs the following modifications to the IR:
 * - Identifies global memory tensors with repeated access patterns and analyzes
 * the indices and extent of memory reads.
 * - Creates new local tensors (buffers) in the IR to store the data retrieved
 * from the global memory for reuse.
 * - Replaces the redundant global memory reads with corresponding reads from
 * the newly created local tensors.
 *
 * Performance impact:
 * - This pass reduces the reliance on global memory, which is typically slower
 * than local memory. It addresses performance bottlenecks such as high memory
 * bandwidth usage and low cache efficiency.
 * - It ensures memory operations are more localized, potentially leading to
 * significant speedups in memory-intensive computations.
 *
 */
void EliminateCommonGlobalMemoryRead(Expr* e);

}  // namespace optim
}  // namespace cinn
