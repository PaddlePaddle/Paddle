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
class IfFusionPass : public BlockPass {
 public:
  IfFusionPass() : BlockPass("if_fusion") {}
  LogicalResult Run(ir::stmt::BlockRef block) override;
};

/**
 * Fuses consecutive if statements with identical conditions within a single
 * block.
 *
 * This pass is applicable in scenarios where multiple if statements with the
 * same condition appear consecutively within a block. Fusing them can
 * simplify the control flow and potentially improve performance.
 *
 * When applied, this pass will combine the bodies of consecutive if statements
 * with identical conditions into a single if statement. The resulting if
 * statement will contain all the operations from the original if statements in
 * their original order. Besides, it would recursively run on the inner blocks
 * of the fused if statement.
 *
 * Performance impact: This pass primarily addresses code size and readability.
 * By reducing the number of redundant condition checks, it may also slightly
 * improve branch prediction and reduce instruction cache pressure.
 *
 * Examples:
 * 1. Basic case:
 *    Input IR:
 *      if (S0 < 64) {
 *        a = a + 1;
 *      }
 *      if (S0 < 64) {
 *        b = b + 1;
 *      }
 *      if (S0 < 64) {
 *        c = c + 1;
 *      }
 *    Output IR:
 *      if (S0 < 64) {
 *        a = a + 1;
 *        b = b + 1;
 *        c = c + 1;
 *      }
 *
 * 2. Nested case:
 *    Input IR:
 *      if (S0 < 64) {
 *          if (S1 > 256) {
 *            a = a + 1;
 *          }
 *      }
 *      if (S0 < 64) {
 *          if (S1 > 256) {
 *            b = b + 1;
 *          }
 *      }
 *    Output IR:
 *      if (S0 < 64) {
 *        if (S1 > 256) {
 *          a = a + 1;
 *          b = b + 1;
 *        }
 *      }
 */
std::unique_ptr<BlockPass> CreateIfFusionPass();

}  // namespace optim
}  // namespace cinn
