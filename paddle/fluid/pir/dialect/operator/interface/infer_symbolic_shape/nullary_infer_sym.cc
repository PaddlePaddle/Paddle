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

#include "paddle/fluid/pir/dialect/operator/interface/infer_symbolic_shape/nullary_infer_sym.h"
#include "paddle/fluid/pir/dialect/operator/interface/infer_symbolic_shape/infer_sym_utils.h"

namespace paddle::dialect {

bool EmptyOpInferSymbolicShape(pir::Operation *op,
                               pir::ShapeConstraintIRAnalysis *shape_analysis) {
  const auto &shape_gen_op = op->operand_source(0).defining_op();
  if (shape_gen_op->isa<paddle::dialect::FullIntArrayOp>()) {
    std::vector<int64_t> shape = details::GetVectorAttr(
        shape_gen_op->dyn_cast<paddle::dialect::FullIntArrayOp>(), "value");
    std::vector<symbol::DimExpr> sym_dims;
    sym_dims.reserve(shape.size());
    for (const int64_t &dim : shape) {
      sym_dims.emplace_back(symbol::DimExpr(dim));
    }

    symbol::ShapeOrDataDimExprs shape_data{
        symbol::TensorShapeOrDataDimExprs(sym_dims)};
    shape_analysis->SetShapeOrDataForValue(op->result(0), shape_data);
    return true;

  } else {
    pir::Value operand_source = op->operand_source(0);
    const symbol::ShapeOrDataDimExprs &operand_shape_or_data =
        shape_analysis->GetShapeOrDataForValue(operand_source);

    shape_analysis->SetShapeOrDataForValue(op->result(0),
                                           operand_shape_or_data);
    return true;
  }
}

bool GaussianOpInferSymbolicShape(
    pir::Operation *op, pir::ShapeConstraintIRAnalysis *shape_analysis) {
  const auto &shape_gen_op = op->operand_source(0).defining_op();

  if (shape_gen_op->isa<paddle::dialect::FullIntArrayOp>()) {
    std::vector<int64_t> shape = details::GetVectorAttr(
        shape_gen_op->dyn_cast<paddle::dialect::FullIntArrayOp>(), "value");
    std::vector<symbol::DimExpr> sym_dims;
    sym_dims.reserve(shape.size());
    for (const int64_t &dim : shape) {
      sym_dims.emplace_back(symbol::DimExpr(dim));
    }

    symbol::ShapeOrDataDimExprs shape_data{
        symbol::TensorShapeOrDataDimExprs(sym_dims)};
    shape_analysis->SetShapeOrDataForValue(op->result(0), shape_data);
    return true;

  } else {
    PADDLE_THROW(phi::errors::Unimplemented(
        op->name() +
        " 's InferSymbolicShape interface is NOT implemented now."));
    return true;
  }
}

}  // namespace paddle::dialect
