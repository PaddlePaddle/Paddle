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
#include "paddle/cinn/pass/pass.h"

namespace cinn {
namespace optim {
class RemoveScheduleBlockPass : public BlockPass {
 public:
  RemoveScheduleBlockPass() : BlockPass("remove_schedule_block") {}
  LogicalResult Run(ir::stmt::BlockRef block) override;
};

/**
 * Removes ScheduleBlock nodes from the IR tree.
 *
 * This pass is applicable in scenarios where ScheduleBlock nodes are present in
 * the IR tree but are no longer needed for further optimization.
 *
 * When applied, this pass will traverse the IR tree and replace each
 * ScheduleBlockRealize node with its body. During this process, it will also
 * replace the iter_vars in the body with their corresponding iter_values. This
 * effectively removes the ScheduleBlock structure while preserving the
 * computational logic within it.
 *
 * Performance impact: This pass addresses the overhead of maintaining
 * ScheduleBlock structures in the IR. By removing these structures, it
 * simplifies the IR, which can lead to faster subsequent passes and potentially
 * more efficient code generation.
 *
 * Examples:
 * 1. Basic ScheduleBlock removal:
 *    Input IR:
 *      ScheduleBlock {
 *        iter_vars: [i, j]
 *        iter_values: [0, 1]
 *        body {
 *          body: A[i, j] = B[i, j] + C[i, j]
 *        }
 *      }
 *    Output IR:
 *      A[0, 1] = B[0, 1] + C[0, 1]
 */
std::unique_ptr<BlockPass> CreateRemoveScheduleBlockPass();

}  // namespace optim
}  // namespace cinn
