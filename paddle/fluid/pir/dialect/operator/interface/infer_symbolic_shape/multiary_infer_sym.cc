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

#include "paddle/fluid/pir/dialect/operator/interface/infer_symbolic_shape/multiary_infer_sym.h"
#include "paddle/common/ddim.h"
#include "paddle/fluid/pir/dialect/operator/interface/infer_symbolic_shape/infer_sym_slice_utils.h"
#include "paddle/fluid/pir/dialect/operator/interface/infer_symbolic_shape/infer_sym_utils.h"
#include "paddle/fluid/pir/dialect/operator/ir/op_attribute.h"

namespace paddle::dialect {

bool ConcatOpInferSymbolicShape(
    pir::Operation *op, pir::ShapeConstraintIRAnalysis *shape_analysis) {
  pir::Value operand_source = op->operand_source(0);
  const auto &shape_data_list =
      shape_analysis->GetShapeOrDataForValue(operand_source)
          .dyn_cast<symbol::TensorListShapeOrDataDimExprs>();

  CHECK(op->operand_source(1).defining_op()->isa<paddle::dialect::FullOp>());

  int64_t axis = op->operand_source(1)
                     .defining_op<paddle::dialect::FullOp>()
                     .attributes()
                     .at("value")
                     .dyn_cast<paddle::dialect::ScalarAttribute>()
                     .data()
                     .to<int64_t>();
  size_t rank = shape_data_list[0].shape().size();
  axis = axis >= 0 ? axis : std::max(int64_t(0), int64_t(axis + rank));

  if (shape_data_list[0].data().has_value()) {
    if (rank == 1) {
      ExprVec data = details::GetExprVecFromData(
          shape_analysis->GetShapeOrDataForValue(operand_source));
      const std::vector<symbol::DimExpr> shape{std::int64_t(data.size())};
      symbol::ShapeOrDataDimExprs shape_data{
          symbol::TensorShapeOrDataDimExprs(shape, data)};
      pir::Value res = op->result(0);
      shape_analysis->SetShapeOrDataForValue(res, shape_data);

      return true;
    } else {
      PADDLE_THROW(phi::errors::Unimplemented(
          op->name() +
          " 's InferSymbolicShape can NOT deal with rank > 1 now."));
    }
    std::vector<symbol::DimExpr> data;
    data.reserve(shape_data_list.size());
    for (auto &data_elem : shape_data_list) {
      data.push_back(data_elem.data().value()[0]);
    }
    const std::vector<symbol::DimExpr> shape{std::int64_t(data.size())};
    symbol::ShapeOrDataDimExprs shape_data{
        symbol::TensorShapeOrDataDimExprs(shape, data)};
    pir::Value res = op->result(0);
    shape_analysis->SetShapeOrDataForValue(res, shape_data);

    return true;
  }

  const std::vector<symbol::DimExpr> &out_dims = [&] {
    std::vector<symbol::DimExpr> out_dims = shape_data_list[0].shape();
    for (size_t i = 0; i < rank; ++i) {
      if (i != static_cast<size_t>(axis)) {
        details::BuildCstrEqForTensorListAlongAxis(
            shape_analysis, shape_data_list, i);
        continue;
      }
      for (size_t j = 1; j < shape_data_list.size(); ++j) {
        out_dims[axis] = out_dims[axis] + shape_data_list[j].shape()[axis];
      }
    }
    return out_dims;
  }();

  symbol::ShapeOrDataDimExprs shape_data{
      symbol::TensorShapeOrDataDimExprs(out_dims)};

  pir::Value res = op->result(0);
  shape_analysis->SetShapeOrDataForValue(res, shape_data);

  return true;
}

bool FullWithTensorOpInferSymbolicShape(
    pir::Operation *op, pir::ShapeConstraintIRAnalysis *shape_analysis) {
  pir::Value operand_source = op->operand_source(0);
  const symbol::ShapeOrDataDimExprs &operand_shape_or_data =
      shape_analysis->GetShapeOrDataForValue(operand_source);

  const auto &out_shape = operand_shape_or_data.data().has_value()
                              ? operand_shape_or_data.data().value()
                              : operand_shape_or_data.shape();

  shape_analysis->SetShapeOrDataForValue(
      op->result(0), symbol::TensorShapeOrDataDimExprs(out_shape));
  return true;
}

bool FlashAttnOpInferSymbolicShape(
    pir::Operation *op, pir::ShapeConstraintIRAnalysis *shape_analysis) {
  pir::Value operand_source = op->operand_source(0);
  const symbol::ShapeOrDataDimExprs &q =
      shape_analysis->GetShapeOrDataForValue(operand_source);

  const symbol::ShapeOrDataDimExprs &v =
      shape_analysis->GetShapeOrDataForValue(op->operand_source(2));

  std::vector<symbol::DimExpr> out_shape = q.shape();

  out_shape.back() = v.shape().back();

  shape_analysis->SetShapeOrDataForValue(
      op->result(0), symbol::TensorShapeOrDataDimExprs(out_shape));
  return true;
}

bool LinspaceOpInferSymbolicShape(
    pir::Operation *op, pir::ShapeConstraintIRAnalysis *shape_analysis) {
  const auto &num_shape_or_data =
      shape_analysis->GetShapeOrDataForValue(op->operand_source(2));
  const auto step = [&] {
    symbol::DimExpr expr;
    if (num_shape_or_data.data().has_value()) {
      expr = num_shape_or_data.data().value()[0];
    } else {
      expr = num_shape_or_data.shape()[0];
    }
    return expr;
  }();
  const symbol::ShapeOrDataDimExprs &shape_data = [&] {
    std::vector<symbol::DimExpr> out_dims{step};
    return symbol::ShapeOrDataDimExprs{
        symbol::TensorShapeOrDataDimExprs(out_dims)};
  }();
  shape_analysis->SetShapeOrDataForValue(op->result(0), shape_data);
  return true;
}
bool LogspaceOpInferSymbolicShape(
    pir::Operation *op, pir::ShapeConstraintIRAnalysis *shape_analysis) {
  return LinspaceOpInferSymbolicShape(op, shape_analysis);
}

bool StackOpInferSymbolicShape(pir::Operation *op,
                               pir::ShapeConstraintIRAnalysis *shape_analysis) {
  pir::Value operand_source = op->operand_source(0);

  const auto &attributes = op->attributes();
  int axis = attributes.at("axis").dyn_cast<pir::Int32Attribute>().data();

  const symbol::TensorListShapeOrDataDimExprs &shape_data_list =
      shape_analysis->GetShapeOrDataForValue(operand_source)
          .dyn_cast<symbol::TensorListShapeOrDataDimExprs>();

  int rank = shape_data_list[0].shape().size();
  if (axis < 0) axis += rank + 1;

  const symbol::ShapeOrDataDimExprs shape_data = [&] {
    std::vector<symbol::DimExpr> shape_dim_exprs;
    std::vector<symbol::DimExpr> data_dim_exprs;
    for (size_t i = 0; i < shape_data_list.size(); ++i) {
      if (shape_data_list[i].data().has_value() && axis == 0) {
        data_dim_exprs.emplace_back(shape_data_list[i].data().value()[0]);
      }
    }

    if (!data_dim_exprs.empty()) {
      shape_dim_exprs.emplace_back(
          static_cast<std::int64_t>(shape_data_list.size()));
    } else {
      for (int i = 0; i < rank; ++i) {
        details::BuildCstrEqForTensorListAlongAxis(
            shape_analysis, shape_data_list, i);
      }
      shape_dim_exprs.insert(shape_dim_exprs.begin() + axis,
                             static_cast<std::int64_t>(shape_data_list.size()));
    }

    return symbol::ShapeOrDataDimExprs(
        symbol::TensorShapeOrDataDimExprs(shape_dim_exprs, data_dim_exprs));
  }();

  pir::Value res = op->result(0);
  shape_analysis->SetShapeOrDataForValue(res, shape_data);
  return true;
}

bool WhereOpInferSymbolicShape(pir::Operation *op,
                               pir::ShapeConstraintIRAnalysis *shape_analysis) {
  shape_analysis->SetShapeOrDataForValue(
      op->result(0),
      shape_analysis->GetShapeOrDataForValue(op->operand_source(0)));

  const std::vector<pir::Value> &operands = {op->operand_source(0),
                                             op->operand_source(1)};

  size_t rank = shape_analysis->GetShapeOrDataForValue(op->operand_source(0))
                    .shape()
                    .size();

  for (size_t i = 0; i < rank; ++i) {
    paddle::dialect::details::BuildCstrEqForTensorListAlongAxis(
        shape_analysis, operands, i);
  }

  return true;
}

bool Where_OpInferSymbolicShape(
    pir::Operation *op, pir::ShapeConstraintIRAnalysis *shape_analysis) {
  return WhereOpInferSymbolicShape(op, shape_analysis);
}

}  // namespace paddle::dialect
