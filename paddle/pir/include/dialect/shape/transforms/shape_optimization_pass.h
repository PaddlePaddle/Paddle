// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

#include <memory>
#include "paddle/pir/include/core/builtin_op.h"
#include "paddle/pir/include/core/dll_decl.h"
#include "paddle/pir/include/pass/pass_manager.h"

namespace pir {

class Pass;

class InferSymbolicShapeContext;

IR_API std::unique_ptr<Pass> CreateShapeOptimizationPass();

void InferSymExprForBlock(const Block &block,
                          InferSymbolicShapeContext *infer_context);

void InferSymExprForAllValues(ModuleOp module_op);
}  // namespace pir

namespace pir::shape {
bool HasDynamicShape(const pir::Program &program);

void AddShapeOptimizationPass(
    std::shared_ptr<pir::PassManager> &pass_manager,  // NOLINT
    pir::Program &program);                           // NOLINT

}  // namespace pir::shape
