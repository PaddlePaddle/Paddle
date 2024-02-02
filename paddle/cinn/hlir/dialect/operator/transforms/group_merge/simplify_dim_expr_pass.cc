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

#include "paddle/cinn/hlir/dialect/operator/transforms/group_merge/simplify_dim_expr_pass.h"

#include "paddle/cinn/hlir/dialect/operator/ir/generate_shape_util.h"
#include "paddle/cinn/hlir/dialect/operator/ir/op_dialect.h"
#include "paddle/cinn/hlir/dialect/runtime/ir/runtime_dialect.h"
#include "paddle/fluid/pir/dialect/kernel/ir/kernel_dialect.h"
#include "paddle/pir/dialect/shape/utils/dim_expr.h"

namespace cinn {
namespace dialect {
namespace ir {

namespace {

template <typename DoEachT>
void VisitEachValue(const pir::Operation& op, const DoEachT& DoEach) {
  for (pir::Operation* op : fusion_op.GetOperators()) {
    for (std::size_t i = 0; i < op->num_operands(); ++i) {
      DoEach(op->operand_source(i));
    }
    for (std::size_t i = 0; i < op->num_results(); ++i) {
      DoEach(op->result(i));
    }
  }
}

void SimplifyDimExprForOneValue(
    pir::Value value, const pir::ShapeConstraintIRAnalysis& shape_analysis) {
  if (shape_analysis.HasShapeOrDataForValue(value)) {
    VLOG(4) << "SimplifyDimExpr shape_analysis.HasShapeOrDataForValue(value) "
               "return false";
    return;
  }
  const std::vector<symbol::DimExpr>& dim_expr_shapes =
      shape_analysis.GetShapeOrDataForValue(value).shape();
  VLOG(4) << "SimplifyDimExpr print :";
  for (const symbol::DimExpr& dim_expr : dim_expr_shapes) {
    auto dim_expr_base = dim_expr.variant();
    VLOG(4) << dim_expr_base.ToString() << " ";
  }
  VLOG(4) << "\n";
  for (const symbol::DimExpr& dim_expr : dim_expr_shapes) {
    auto dim_expr_base = dim_expr.variant();
    VLOG(4) << dim_expr_base.ToString() << " ";
  }
}

void SimplifyDimExpr(pir::ModuleOp module_op) {
  VLOG(4) << "SimplifyDimExpr start";
  const pir::ShapeConstraintIRAnalysis shape_analysis =
      pir::ShapeAnalysisManager::Instance().Get(module_op.program());
  for (uint32_t i = 0; i < module_op->num_regions(); i++) {
    for (const pir::Block& block : module_op->region(i)) {
      for (const pir::Operation& op : block) {
        VisitEachValue(op, [&](pir::Value value) {
          SimplifyDimExprForOneValue(value, shape_analysis);
        });
      }
    }

    VLOG(4) << "SimplifyDimExpr end";
  }

  class SimplifyDimExprPass : public pir::Pass {
   public:
    SimplifyDimExprPass()
        : pir::Pass("simplify_dim_expr_pass", 1) {}  // opt_level ?

    void Run(pir::Operation* op) override {
      pir::ModuleOp module_op = op->dyn_cast<pir::ModuleOp>();
      SimplifyDimExpr(module_op);
    }

    bool CanApplyOn(pir::Operation* op) const override {
      return op->isa<pir::ModuleOp>() && op->num_regions() > 0;
    }
  };
}
}  // namespace

std::unique_ptr<::pir::Pass> CreateSimplifyDimExprPass() {
  return std::make_unique<SimplifyDimExprPass>();
}

}  // namespace ir
}  // namespace dialect
}  // namespace cinn
