// Copyright (c) 2021 CINN Authors. All Rights Reserved.
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
#include "paddle/cinn/ir/lowered_func.h"
#include "paddle/cinn/ir/module.h"

namespace cinn {
namespace optim {

/**
 * Optimize the expression but Module.
 * @param fn
 * @param runtime_debug_info
 * @return
 */
ir::LoweredFunc Optimize(ir::LoweredFunc fn,
                         Target target,
                         bool runtime_debug_info = false,
                         bool remove_gpu_for_loops = true);

/**
 * Optimize a Module.
 */
ir::Module Optimize(const ir::Module& module, const Target& target);

}  // namespace optim
}  // namespace cinn
