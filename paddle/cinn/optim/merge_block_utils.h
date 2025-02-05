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

#include "paddle/cinn/ir/stmt.h"

namespace cinn {
namespace optim {

struct ForTreeNode {
  const ir::stmt::For val;
  std::vector<ForTreeNode> children;
};

using ForEqualFunc =
    std::function<bool(const ForTreeNode&, const ForTreeNode&)>;

/*
 * Determines if two blocks of code with nested for-loops have identical loop
 * extents and can be merged.
 *
 * This pass is applicable in scenarios where there are multiple code blocks
 * with nested for-loops, and we need to determine if these blocks can be
 * consolidated to simplify the code structure.
 *
 * When applied, this pass will not directly modify the IR but serves as a
 * prerequisite check to ensure that loop extents match. If they do, a separate
 * merging process can be safely conducted to combine the blocks into a single
 * block with shared loop structures.
 *
 * Performance impact: This pass itself does not directly impact performance but
 * enables further optimizations by identifying mergeable loop structures, which
 * can reduce code size and potentially improve cache efficiency by
 * consolidating similar data processing tasks.
 *
 * Examples:
 *
 * Simple identical loops:
 * Input IR:
 *   block(var_B)
 * for(i, 0, 10)
 *   for(j, 0, 10)
 *     B[i,j] = A[i,j]
 *   block(var_C)
 * for(i, 0, 10)
 *   for(j, 0, 10)
 *     C[i,j] = A[i,j]
 * Output IR:
 *   Can be merged since loop extents are identical.
 *
 * Different loop extents:
 * Input IR:
 *   block(var_B)
 * for(i, 0, 10)
 *   for(j, 0, 10)
 *     B[i,j] = A[i,j]
 *   block(var_C)
 * for(i, 0, 3)
 *   for(j, 0, 4)
 *     C[i,j] = A[i,j]
 * Output IR:
 *   Cannot be merged due to differing loop extents.
 */

bool CanMergeBlocks(const ir::stmt::For first,
                    const ir::stmt::For second,
                    const ForEqualFunc& IsEqual);

}  // namespace optim
}  // namespace cinn
