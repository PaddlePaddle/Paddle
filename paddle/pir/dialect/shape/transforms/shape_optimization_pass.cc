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
#include "paddle/pir/dialect/shape/utils/shape_utils.h"
#include "paddle/pir/pass/pass.h"
#include "paddle/pir/pass/pass_manager.h"
#include "paddle/pir/pass/pass_registry.h"

namespace {
using PassPipelineRunner =
    std::function<bool(pir::PassManager&, pir::ModuleOp)>;

bool InsertTieShapeOnValue(pir::Value value,
                           pir::Builder& builder) {  // NOLINT
  auto ty = value.type().dyn_cast<paddle::dialect::DenseTensorType>();

  if (!ty || ty.dims().size() == 0) return true;
  std::vector<pir::Value> dimSizes;
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
  // TODO(zhangbo63): skip more specialized Ops.
  if (op->isa<pir::dialect::TieShapeOp>() || op->isa<pir::dialect::FuncOp>())
    return true;

  for (size_t i = 0; i < op->num_regions(); ++i) {
    if (!InsertTieShapeOnRegion(&(op->region(i)))) return false;
  }
  builder.SetInsertionPointAfter(op);
  for (pir::OpResult v : op->results()) {
    if (!InsertTieShapeOnValue(v, builder)) return false;
  }

  return true;
}

bool InsertTieShapeOnBlock(pir::Block* block) {
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
    if (!InsertTieShapeOnBlock(block)) return false;
  }
  return true;
}

bool MaterializeShapeComputation(pir::ModuleOp m) {
  if (!InsertTieShapeOnRegion(&(m->region(0)))) return false;
  // TODO(liujinnan): add rewitter pattern for reifyInferShape.
  return true;
}

bool OptimizeShapeComputation(pir::ModuleOp m, PassPipelineRunner runner) {
  // TODO(liujinnan): Do some Canonicalizer.
  pir::SymbolicDimMgr mgr(m);
  IR_ENFORCE(mgr.Load(),
             "SymbolicDimMgr Load failed in OptimizeShapeComputation.");
  pir::ShapeComputationIRAnalysis analysis(m, mgr);
  if (!analysis.Run()) {
    return false;
  }
  IR_ENFORCE(mgr.Save(),
             "SymbolicDimMgr save failed in OptimizeShapeComputation.");
  return true;
}

class ShapeOptimizationPass : public pir::Pass {
 public:
  ShapeOptimizationPass() : pir::Pass("shape_optimization", 0) {}

  void Run(pir::Operation* op) override {
    auto module_op = op->dyn_cast<pir::ModuleOp>();
    IR_ENFORCE(module_op, "ShapeOptimizationPass should run on module op.");
    MaterializeShapeComputation(module_op);
    // runner is for Canonicalizer.
    PassPipelineRunner runner = [this](pir::PassManager& pm, pir::ModuleOp m) {
      return pm.Run(m.program());
    };
    if (!OptimizeShapeComputation(module_op, runner)) {
      return;
    }
  }

  bool CanApplyOn(pir::Operation* op) const override {
    return op->isa<pir::ModuleOp>() && op->num_regions() > 0;
  }
};

}  // namespace

namespace pir {

std::unique_ptr<Pass> CreateShapeOptimizationPass() {
  return std::make_unique<ShapeOptimizationPass>();
}

}  // namespace pir

REGISTER_IR_PASS(shape_optimization, ShapeOptimizationPass);
