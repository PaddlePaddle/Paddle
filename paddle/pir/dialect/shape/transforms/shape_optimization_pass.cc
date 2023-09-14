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

#include "paddle/pir/dialect/shape/transforms/shape_optimization_pass.h"
#include "paddle/fluid/pir/dialect/operator/ir/op_type.h"
#include "paddle/pir/dialect/shape/ir/shape_op.h"

#include "paddle/pir/core/builtin_op.h"
#include "paddle/pir/core/program.h"
#include "paddle/pir/pass/pass.h"
#include "paddle/pir/pass/pass_registry.h"

namespace {

bool InsertTieShapeOnValue(pir::OpResult value,
                           pir::Builder& builder) {  // NOLINT
  auto ty = value.type().dyn_cast<paddle::dialect::DenseTensorType>();

  if (!ty || ty.dims().size() == 0) return true;
  std::vector<pir::OpResult> dimSizes;
  for (int64_t dim = 0, rank = ty.dims().size(); dim < rank; ++dim) {
    auto dimOp = builder.Build<pir::dialect::TensorDimOp>(value, dim);
    dimSizes.push_back(dimOp.out());
  }
  builder.Build<pir::dialect::TieShapeOp>(value, dimSizes);
  return true;
}

bool InsertTieShapeOnRegion(pir::Region* region);

bool InsertTieShapeOnOperation(pir::Operation* op,
                               pir::Builder& builder) {  // NOLINT
  if (op->isa<pir::dialect::TieShapeOp>()) return true;
  // TODO(liujinnan): skip the specialized Ops.

  for (size_t i = 0; i < op->num_regions(); ++i) {
    if (!InsertTieShapeOnRegion(&(op->region(i)))) return false;
  }
  builder.SetInsertionPointAfter(op);
  for (pir::OpResult v : op->results()) {
    if (!InsertTieShapeOnValue(v, builder)) return false;
  }

  return true;
}

bool insertTieShapeOnBlock(pir::Block* block) {
  pir::Builder builder =
      pir::Builder(pir::IrContext::Instance(), block, block->begin());
  // TODO(liujinnan): mapping block arguments

  std::vector<pir::Operation*> op_list;
  for (pir::Operation* op : *block) op_list.push_back(op);
  for (pir::Operation* op : op_list) {
    if (!InsertTieShapeOnOperation(op, builder)) return false;
  }
  return true;
}

bool InsertTieShapeOnRegion(pir::Region* region) {
  for (pir::Block* block : *region) {
    if (!insertTieShapeOnBlock(block)) return false;
  }
  return true;
}

bool MaterializeShapeComputation(pir::ModuleOp m) {
  if (!InsertTieShapeOnRegion(&(m->region(0)))) return false;
  // TODO(liujinnan): add rewitter pattern for reifyInferShape.
  return true;
}

class ShapeOptimizationPass : public pir::Pass {
 public:
  ShapeOptimizationPass() : pir::Pass("shape_optimization", 0) {}

  void Run(pir::Operation* op) override {
    auto module_op = op->dyn_cast<pir::ModuleOp>();
    IR_ENFORCE(module_op, "ShapeOptimizationPass should run on module op.");
    MaterializeShapeComputation(module_op);
  }

  bool CanApplyOn(pir::Operation* op) const override {
    return op->name() == "builtin.module" && op->num_regions() > 0;
  }
};

}  // namespace

namespace pir {

std::unique_ptr<Pass> CreateShapeOptimizationPass() {
  return std::make_unique<ShapeOptimizationPass>();
}

}  // namespace pir

REGISTER_IR_PASS(shape_optimization, ShapeOptimizationPass);
