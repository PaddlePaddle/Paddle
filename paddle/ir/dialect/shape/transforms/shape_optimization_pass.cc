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

#include "paddle/ir/dialect/shape/transforms/shape_optimization_pass.h"

#include "paddle/ir/core/builtin_op.h"
#include "paddle/ir/core/program.h"
#include "paddle/ir/pass/pass.h"
#include "paddle/ir/pass/pass_registry.h"

namespace {

class ShapeOptimizationPass : public ir::Pass {
 public:
  ShapeOptimizationPass() : ir::Pass("shape_optimization", 0) {}

  void Run(ir::Operation *op) override {
    auto module_op = op->dyn_cast<ir::ModuleOp>();
    IR_ENFORCE(module_op, "ShapeOptimizationPass should run on module op.");
  }

  bool CanApplyOn(ir::Operation *op) const override {
    return op->name() == "builtin.module" && op->num_regions() > 0;
  }
};

}  // namespace

namespace ir {

std::unique_ptr<Pass> CreateShapeOptimizationPass() {
  return std::make_unique<ShapeOptimizationPass>();
}

}  // namespace ir

REGISTER_IR_PASS(shape_optimization, ShapeOptimizationPass);
