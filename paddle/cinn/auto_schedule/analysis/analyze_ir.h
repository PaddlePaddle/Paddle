// Copyright (c) 2022 CINN Authors. All Rights Reserved.
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

#include <string>
#include <unordered_set>

#include "paddle/cinn/ir/ir.h"
#include "paddle/cinn/ir/ir_base.h"
#include "paddle/cinn/ir/lowered_func.h"
#include "paddle/cinn/ir/schedule/ir_schedule.h"

namespace cinn {
namespace auto_schedule {

void AnalyzeScheduleBlockReadWriteBuffer(ir::ScheduleBlock* sche_block);

bool ContainsNodeType(ir::Expr expr,
                      const std::unordered_set<ir::IrNodeTy>& node_types);

/**
 * Collects all input lowered_funcs and return names of all output arguments
 */
std::unordered_set<std::string> GetOutputNamesFromLoweredFunc(
    const std::vector<ir::LoweredFunc>& lowered_funcs);

/**
 * Determine whether a schedule block needs multi-level tiling
 */
bool NeedsMultiLevelTiling(const ir::ScheduleBlockRealize& sche_block_realize);

/**
 * Get loop var names of reduce axis
 */
std::unordered_set<std::string> GetReduceLoopVarNames(const ir::Expr block);

/**
 * Get name of a ScheduleBlock
 */
std::string GetBlockName(const ir::Expr block);

}  // namespace auto_schedule
}  // namespace cinn
