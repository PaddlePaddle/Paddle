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

#include "paddle/fluid/pir/dialect/operator/interface/infer_symbolic_shape/cinn_op_infer_sym.h"
#include "paddle/fluid/pir/dialect/operator/interface/infer_symbolic_shape/infer_sym_utils.h"

namespace cinn::dialect {

bool BroadcastOpInferSymbolicShape(
    pir::Operation *op, pir::ShapeConstraintIRAnalysis *shape_analysis) {
  const std::vector<int64_t> &shape =
      paddle::dialect::details::GetVectorAttr<int64_t>(op, "out_shape");

  const std::vector<symbol::DimExpr> &out_dims = [&] {
    std::vector<symbol::DimExpr> out_dims;
    for (int64_t dim : shape) {
      out_dims.emplace_back(dim);
    }
    return out_dims;
  }();

  symbol::ShapeOrDataDimExprs shape_data{
      symbol::TensorShapeOrDataDimExprs(out_dims)};
  shape_analysis->SetShapeOrDataForValue(op->result(0), shape_data);
  return true;
}

bool SliceOpInferSymbolicShape(pir::Operation *op,
                               pir::ShapeConstraintIRAnalysis *shape_analysis) {
  // TODO(zhangbopd): Not implemented yet, different from the one in paddle
  // dialect. And Currently only support start/end/axis with single value.
  pir::AttributeMap attributes = op->attributes();

  auto GetAttrInt64Value = [&](const std::string &name) -> int64_t {
    std::vector<pir::Attribute> attr =
        attributes[name].dyn_cast<pir::ArrayAttribute>().AsVector();
    PADDLE_ENFORCE_GT(
        attr.size(),
        0,
        phi::errors::PreconditionNotMet(
            "Only Support [%s] op len(%s) == 1 , but received %d.",
            op->name(),
            name,
            attr.size()));
    return attr[0].dyn_cast<pir::Int64Attribute>().data();
  };

  const int64_t start = GetAttrInt64Value("starts");
  const int64_t end = GetAttrInt64Value("ends");
  const int64_t axis = GetAttrInt64Value("axes");

  const pir::Value operand_source = op->operand_source(0);
  const auto &operand_shape_or_data =
      shape_analysis->GetShapeOrDataForValue(operand_source);

  const auto GetOutDimExprs = [&]() -> symbol::TensorShapeOrDataDimExprs {
    std::vector<symbol::DimExpr> out_sym_shape = operand_shape_or_data.shape();
    if (end == std::numeric_limits<int>::max()) {
      out_sym_shape[axis] = out_sym_shape[axis] - start;
    } else {
      out_sym_shape[axis] = end - start;
    }
    symbol::TensorShapeOrDataDimExprs shape_dim_expr(out_sym_shape);
    if (operand_shape_or_data.data().has_value()) {
      std::vector<symbol::DimExpr> out_data;
      for (int64_t i = start; i < end; i++) {
        out_data.push_back(operand_shape_or_data.data().value()[i]);
      }
      shape_dim_expr.SetData(out_data);
    }
    return shape_dim_expr;
  };
  symbol::ShapeOrDataDimExprs shape_data{GetOutDimExprs()};

  shape_analysis->SetShapeOrDataForValue(op->result(0), shape_data);
  return true;
}

bool ConcatOpInferSymbolicShape(
    pir::Operation *op, pir::ShapeConstraintIRAnalysis *shape_analysis) {
  const auto input_values = op->operands_source();
  const auto input_size = input_values.size();

  const int axis =
      op->attributes().at("axis").dyn_cast<pir::Int32Attribute>().data();

  // TODO(zhangbopd): Need support GetShapeOrDataForValue().data() case.
  const auto &GetOutDimExprs = [&]() -> std::vector<symbol::DimExpr> {
    std::vector<symbol::DimExpr> out_dims =
        shape_analysis->GetShapeOrDataForValue(input_values[0]).shape();
    for (size_t i = 1; i < input_size; ++i) {
      const auto &operand_shape_or_data =
          shape_analysis->GetShapeOrDataForValue(input_values[i]);
      out_dims[axis] = out_dims[axis] + operand_shape_or_data.shape()[axis];
    }
    return out_dims;
  };

  symbol::ShapeOrDataDimExprs shape_data{
      symbol::TensorShapeOrDataDimExprs(GetOutDimExprs())};

  shape_analysis->SetShapeOrDataForValue(op->result(0), shape_data);
  return true;
}

bool ReduceInferSymbolicShape(pir::Operation *op,
                              pir::ShapeConstraintIRAnalysis *shape_analysis) {
  const auto &attr_map = op->attributes();
  PADDLE_ENFORCE(
      attr_map.count("keep_dim"),
      phi::errors::PreconditionNotMet(
          "attr [keep_dim] MUST in attribute map for [%s] op", op->name()));
  bool keepdim = attr_map.at("keep_dim").dyn_cast<pir::BoolAttribute>().data();
  auto axis = paddle::dialect::details::GetVectorAttr(op, "dim");
  bool reduce_all = axis.size() == 0 ? true : false;
  return paddle::dialect::details::ReduceInferDim(
      op, shape_analysis, axis, keepdim, reduce_all);
}

bool ReduceMaxOpInferSymbolicShape(
    pir::Operation *op, pir::ShapeConstraintIRAnalysis *shape_analysis) {
  return ReduceInferSymbolicShape(op, shape_analysis);
}

bool ReduceMinOpInferSymbolicShape(
    pir::Operation *op, pir::ShapeConstraintIRAnalysis *shape_analysis) {
  return ReduceInferSymbolicShape(op, shape_analysis);
}

bool ReduceProdOpInferSymbolicShape(
    pir::Operation *op, pir::ShapeConstraintIRAnalysis *shape_analysis) {
  return ReduceInferSymbolicShape(op, shape_analysis);
}

bool ReduceSumOpInferSymbolicShape(
    pir::Operation *op, pir::ShapeConstraintIRAnalysis *shape_analysis) {
  return ReduceInferSymbolicShape(op, shape_analysis);
}

bool ReshapeOpInferSymbolicShape(
    pir::Operation *op, pir::ShapeConstraintIRAnalysis *shape_analysis) {
  std::vector<int> shape =
      paddle::dialect::details::GetVectorAttr<int>(op, "shape");

  std::vector<symbol::DimExpr> out_dims;
  for (int dim : shape) {
    out_dims.emplace_back(static_cast<std::int64_t>(dim));
  }
  symbol::ShapeOrDataDimExprs shape_data{
      symbol::TensorShapeOrDataDimExprs(out_dims)};
  shape_analysis->SetShapeOrDataForValue(op->result(0), shape_data);

  return true;
}

}  // namespace cinn::dialect
