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
#include "paddle/pir/include/dialect/shape/ir/shape_attribute.h"
#include "paddle/pir/include/dialect/shape/utils/dim_expr_util.h"

namespace cinn {
namespace dialect {
namespace ir {

namespace {

template <typename DoEachT>
void VisitEachOp(pir::Operation* op, const DoEachT& DoEach) {
  for (uint32_t i = 0; i < op->num_regions(); i++) {
    for (pir::Block& block : op->region(i)) {
      for (pir::Operation& sub_op : block) {
        DoEach(sub_op);
        if (sub_op.num_regions() > 0) {
          VisitEachOp(&sub_op, DoEach);
        }
      }
    }
  }
}

template <typename DoEachT>
void VisitEachValue(const pir::Operation& op, const DoEachT& DoEach) {
  for (std::size_t i = 0; i < op.num_operands(); ++i) {
    DoEach(op.operand_source(i));
  }
  for (std::size_t i = 0; i < op.num_results(); ++i) {
    DoEach(op.result(i));
  }
}

std::vector<symbol::DimExpr> SimplifyDimExprVector(
    const std::vector<symbol::DimExpr>& original_dim_exprs) {
  std::vector<symbol::DimExpr> simplified_dim_exprs{};
  for (const symbol::DimExpr& dim_expr : original_dim_exprs) {
    simplified_dim_exprs.push_back(symbol::SimplifyDimExpr(dim_expr));
  }
  return simplified_dim_exprs;
}

symbol::TensorShapeOrDataDimExprs SimplifyTensorShapeOrData(
    const symbol::TensorShapeOrDataDimExprs& shape_or_data) {
  std::vector<symbol::DimExpr> simplified_shape =
      SimplifyDimExprVector(shape_or_data.shape());
  if (!shape_or_data.data().has_value()) {
    return symbol::ShapeOrData<symbol::DimExpr>(simplified_shape);
  }
  std::vector<symbol::DimExpr> simplified_data =
      SimplifyDimExprVector(shape_or_data.data().value());
  return symbol::ShapeOrData<symbol::DimExpr>(simplified_shape,
                                              simplified_data);
}

symbol::ShapeOrDataDimExprs SimplifyShapeOrData(
    const symbol::ShapeOrDataDimExprs& shape_or_data) {
  auto lambdas = ::common::Overloaded{
      [](const symbol::TensorShapeOrDataDimExprs& tensor_shape_or_data) {
        return symbol::ShapeOrDataDimExprs(
            SimplifyTensorShapeOrData(tensor_shape_or_data));
      },
      [](const symbol::TensorListShapeOrDataDimExprs& tensor_list) {
        symbol::TensorListShapeOrDataDimExprs simplified_tensor_list;
        for (symbol::TensorShapeOrDataDimExprs tensor_shape_or_data :
             tensor_list) {
          simplified_tensor_list.push_back(
              SimplifyTensorShapeOrData(tensor_shape_or_data));
        }
        return symbol::ShapeOrDataDimExprs(simplified_tensor_list);
      },
      [](const symbol::RankedTensorArrayShapeOrDataDimExprs& tensor_array) {
        return symbol::ShapeOrDataDimExprs(
            symbol::RankedTensorArrayShapeOrDataDimExprs(
                SimplifyDimExprVector(tensor_array.GetShapeHint())));
      },
      [](const symbol::NullShapeOrDataDimExpr& null_shape_or_data) {
        return symbol::ShapeOrDataDimExprs(null_shape_or_data);
      }};
  return std::visit(lambdas, shape_or_data.variant());
}

void SimplifyDimExpr(pir::Operation* module_op) {
  VLOG(4) << "SimplifyDimExpr start";
  pir::ShapeConstraintIRAnalysis* shape_analysis =
      &pir::ShapeAnalysisManager::Instance().Get(
          module_op->dyn_cast<pir::ModuleOp>().program());

  VisitEachOp(module_op, [&](pir::Operation& op) {
    VisitEachValue(op, [&](pir::Value value) {
      if (!value || !value.type()) {
        return;
      }
      const symbol::ShapeOrDataDimExprs& shape_or_data =
          shape_analysis->GetShapeOrDataForValue(value);
      VLOG(8) << op.name() << "     origin_shape_or_data: " << shape_or_data;
      symbol::ShapeOrDataDimExprs simplified_shape_or_data =
          SimplifyShapeOrData(shape_or_data);
      VLOG(8) << op.name()
              << " simplified_shape_or_data: " << simplified_shape_or_data;
      shape_analysis->SetShapeOrDataForValue(value, simplified_shape_or_data);
    });
    if (op.num_results() > 0) {
      pir::shape::SetShapeAttrForOp(
          &op, shape_analysis->GetShapeOrDataForValue(op.result(0)));
    } else {
      pir::shape::SetShapeAttrForOp(
          &op, shape_analysis->GetShapeOrDataForValue(op.operand_source(0)));
    }
    // TODO(JiaWenxuan): simplify the attribute "sym_shape_str" of the op
  });
  VLOG(4) << "SimplifyDimExpr end";
}

class SimplifyDimExprPass : public pir::Pass {
 public:
  SimplifyDimExprPass() : pir::Pass("simplify_dim_expr_pass", 1) {}

  void Run(pir::Operation* op) override { SimplifyDimExpr(op); }

  bool CanApplyOn(pir::Operation* op) const override {
    return op->isa<pir::ModuleOp>() && op->num_regions() > 0;
  }
};

}  // namespace

std::unique_ptr<::pir::Pass> CreateSimplifyDimExprPass() {
  return std::make_unique<SimplifyDimExprPass>();
}

}  // namespace ir
}  // namespace dialect
}  // namespace cinn
