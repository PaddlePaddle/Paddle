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

#include "paddle/fluid/pir/dialect/operator/utils/shape_analysis_utils.h"

namespace paddle {
namespace dialect {

const symbol::ShapeOrDataDimExprs& GetInputShape(
    const pir::InferSymbolicShapeContext* infer_context,
    pir::Operation* op,
    int index) {
  return infer_context->GetShapeOrDataForValue(op->operand_source(index));
}

const symbol::ShapeOrDataDimExprs& GetOutputShape(
    const pir::InferSymbolicShapeContext* infer_context,
    pir::Operation* op,
    int index) {
  return infer_context->GetShapeOrDataForValue(op->result(index));
}

symbol::ShapeOrDataDimExprs ClearDataInfo(
    const symbol::ShapeOrDataDimExprs& symbol_shape) {
  const auto& ClearDataOfSymbolShape = common::Overloaded{
      [](const symbol::TensorShapeOrDataDimExprs& shape_expr) {
        return symbol::ShapeOrDataDimExprs{
            symbol::TensorShapeOrDataDimExprs(shape_expr.shape())};
      },
      [](const symbol::TensorListShapeOrDataDimExprs& shape_exprs) {
        std::vector<symbol::TensorShapeOrDataDimExprs> new_shape_exprs;
        for (const auto& shape_expr : shape_exprs) {
          new_shape_exprs.emplace_back(shape_expr.shape());
        }
        return symbol::ShapeOrDataDimExprs{new_shape_exprs};
      },
      [](const symbol::RankedTensorArrayShapeOrDataDimExprs& shape_exprs) {
        return symbol::ShapeOrDataDimExprs{
            symbol::RankedTensorArrayShapeOrDataDimExprs{
                shape_exprs.GetShapeHint()}};
      },
      [](const symbol::NullShapeOrDataDimExpr& null_shape_or_data) {
        return symbol::ShapeOrDataDimExprs{null_shape_or_data};
      }};
  return std::visit(ClearDataOfSymbolShape, symbol_shape.variant());
}

symbol::ShapeOrDataDimExprs GetGradVarShapeFromOutput(
    const pir::InferSymbolicShapeContext* infer_context,
    pir::Operation* op,
    int index) {
  const auto& out_shape =
      infer_context->GetShapeOrDataForValue(op->result(index));
  return ClearDataInfo(out_shape);
}

symbol::ShapeOrDataDimExprs GetGradVarShapeFromInput(
    const pir::InferSymbolicShapeContext* infer_context,
    pir::Operation* op,
    int index) {
  const auto& out_shape =
      infer_context->GetShapeOrDataForValue(op->operand_source(index));
  return ClearDataInfo(out_shape);
}

}  // namespace dialect
}  // namespace paddle
