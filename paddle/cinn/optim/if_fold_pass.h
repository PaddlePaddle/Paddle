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
class IfFoldPass : public StmtPass {
 public:
  IfFoldPass() : StmtPass("if_fold") {}
  LogicalResult Run(ir::stmt::StmtRef stmt) override;
};

/**
 * Simplify several consecutively nested `IfThenElse` with equal to 0 conditions
 * into one with simplified conditions.
 *
 * This pass is used when there are nested `IfThenElse` in a block, and their
 * conditions are all equal to zero, and these conditions can be mathematically
 * proven to be simplifiable.
 *
 * When applied, the continuously nested `IfThenElse` will be converted into an
 * equivalent `IfThenElse` in IR.
 *
 * Performance impact: This pass primarily addresses code size and readability.
 * By reducing the number of redundant condition checks, it may also slightly
 * improve branch prediction and reduce instruction cache pressure.
 *
 * Examples:
 * case1: All if can be simplified.
 * if ((((((256 * j) + ((1024 * i) + k)) / 56) / 56) == 0)) {
 *   if ((((((256 * j) + ((1024 * i) + k)) / 56) % 56) == 0)) {
 *     if (((((256 * j) + ((1024 * i) + k)) % 56) == 0)) {
 *       int32 a = 1
 *     }
 *   }
 * }
 * can be simplified to:
 * if (((((i * 1024ll) + k) + (j * 256ll)) == 0)) {
 *   int32 a = 1
 * }
 *
 * case2: All if can be simplified and the inner one has false branch.
 * if ((((((256 * j) + ((1024 * i) + k)) / 56) / 56) == 0)) {
 *   if ((((((256 * j) + ((1024 * i) + k)) / 56) % 56) == 0)) {
 *     if (((((256 * j) + ((1024 * i) + k)) % 56) == 0)) {
 *       int32 a = 1
 *       int32 b = 1
 *     } else {
 *       int32 c = 1
 *     }
 *   }
 * }
 * can be simplified to:
 * if (((((i * 1024ll) + k) + (j * 256ll)) == 0)) {
 *   int32 a = 1
 *   int32 b = 1
 * } else {
 *   int32 c = 1
 * }
 *
 * case3: The inner one can not be simplified.
 * if ((((((256 * j) + ((1024 * i) + k)) / 56) / 56) == 0)) {
 *   if ((((((256 * j) + ((1024 * i) + k)) / 56) % 56) == 0)) {
 *     if (((((256 * j) + ((1024 * i) + k)) % 56) == 0)) {
 *       if (l <= 0)) {
 *         int32 a = 1
 *       }
 *     }
 *   }
 * }
 * can be simplified to:
 * if (((((i * 1024ll) + k) + (j * 256ll)) == 0)) {
 *   if (l <= 0)) {
 *     int32 a = 1
 *   }
 * }
 */
std::unique_ptr<StmtPass> CreateIfFoldPass();

}  // namespace optim
}  // namespace cinn
