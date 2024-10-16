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

struct ForTreeNode {
  const ir::For* val;
  std::vector<ForTreeNode> children;
};

using ForEqualFunc =
    std::function<bool(const ForTreeNode&, const ForTreeNode&)>;

/**
 * Check if blocks can merge.
 * @param first for_loops vector
 * @param second Another for_loops vector.
 * @return Return if two block's for extents equal currently.
 */
/**
 * Example 1: CanMergeBlocks(var_B, var_C)
 * block(var_B)
 *   for(i, 0, 10)
 *     for(j, 0, 10)
 *        B[i,j] = A[i,j]
 *
 * block(var_C)
 *   for(i, 0, 10)
 *     for(j, 0, 10)
 *        C[i,j] = A[i,j]
 * =>
 * Return value:
 * true
 *
 * Example 2: CanMergeBlocks(var_B, var_C)
 * block(var_B)
 *   for(i, 0, 10)
 *     for(j, 0, 10)
 *        B[i,j] = A[i,j]
 *
 * block(var_C)
 *   for(i, 0, 3)
 *     for(j, 0, 4)
 *        C[i,j] = A[i,j]
 * =>
 * Return value:
 * false
 */
bool CanMergeBlocks(const ir::For* first,
                    const ir::For* second,
                    const ForEqualFunc& IsEqual);

}  // namespace optim
}  // namespace cinn
