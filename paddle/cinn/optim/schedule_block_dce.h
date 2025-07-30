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

/**
 * This file implements the strategy to remove the unnecessary schedule_block.
 */
#pragma once
#include <vector>

#include "paddle/cinn/common/common.h"
#include "paddle/cinn/ir/ir.h"

namespace cinn {
namespace optim {

/**
 *
 * This pass eliminates dead schedule blocks (those that do not affect the final
 * program output).
 *
 * This pass is applicable in scenarios where there are redundant computations
 * or store operations in the program. For example, certain schedule blocks may
 * not influence the final output or may be completely unused. This commonly
 * occurs in deeply nested loops where some sub-blocks are executed, but their
 * results are never used, leading to wasted computation. The goal of this pass
 * is to analyze the dependencies and data flow of schedule blocks to identify
 * and remove those that are unnecessary, simplifying the code and improving
 * execution efficiency.
 *
 * When applied, this pass will perform the following modifications to the IR:
 * 1. The IR will be scanned for schedule blocks that do not influence the final
 * output. These blocks will be marked as dead schedule blocks.
 * 2. Dead schedule blocks will be removed from the IR, eliminating unnecessary
 * computations. Specifically, in loops and conditional branches, any redundant
 * schedule blocks will be removed.
 * 3. While removing dead schedule blocks, other valid blocks in the IR will
 * remain unaffected, ensuring the correctness of the program.
 * 4. The final IR will be simplified, containing only the relevant computations
 * that contribute to the output, resulting in more efficient execution.
 *
 * Performance impact: This pass addresses performance issues by removing
 * unnecessary computations, potentially improving program execution efficiency,
 * especially when there are large amounts of unused computations. By
 * eliminating dead schedule blocks, execution time and memory consumption can
 * be reduced.
 *
 */
void EliminateDeadScheduleBlock(Expr* e,
                                const std::vector<std::string>& output_names);

}  // namespace optim
}  // namespace cinn
