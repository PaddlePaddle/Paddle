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

#include "paddle/fluid/pir/dialect/operator/interface/infer_symbolic_shape.h"
#include "paddle/pir/core/builtin_attribute.h"
#include "paddle/pir/core/builtin_type.h"
#include "paddle/pir/dialect/shape/ir/shape_op.h"

namespace paddle::dialect {

bool InferSymbolicShapeInterface::InferSymbolicShape(
    pir::ShapeConstraintIRAnalysis *shape_analysis) {
  return impl_->infer_symbolic_shapes(operation(), shape_analysis);
}
}  // namespace paddle::dialect

namespace paddle::dialect {

namespace {

bool InferSymbolicShapeAllEqualUnary(
    pir::Operation *op, pir::ShapeConstraintIRAnalysis *shape_analysis) {
  pir::Value operand_source = op->operand_source(0);
  std::string operand_source_id = pir::GetValueId(&operand_source);
  pir::OpResult res = op->result(0);
  std::string res_id = pir::GetValueId(&res);
  shape_analysis->value_id_to_shapeordata_[res_id] =
      shape_analysis->value_id_to_shapeordata_[operand_source_id];
  return true;
}

bool InferSymbolicShapeAllEqualBinary(
    pir::Operation *op, pir::ShapeConstraintIRAnalysis *shape_analysis) {
  pir::Value operand_source = op->operand_source(0);
  std::string operand_source_id = pir::GetValueId(&operand_source);
  pir::OpResult res = op->result(0);
  std::string res_id = pir::GetValueId(&res);
  shape_analysis->value_id_to_shapeordata_[res_id] =
      shape_analysis->value_id_to_shapeordata_[operand_source_id];
  return true;
}

}  // namespace

bool AbsOpInferSymbolicShape(pir::Operation *op,
                             pir::ShapeConstraintIRAnalysis *shape_analysis) {
  return InferSymbolicShapeAllEqualUnary(op, shape_analysis);
}

bool Abs_OpInferSymbolicShape(pir::Operation *op,
                              pir::ShapeConstraintIRAnalysis *shape_analysis) {
  return InferSymbolicShapeAllEqualUnary(op, shape_analysis);
}

bool CastOpInferSymbolicShape(pir::Operation *op,
                              pir::ShapeConstraintIRAnalysis *shape_analysis) {
  return InferSymbolicShapeAllEqualUnary(op, shape_analysis);
}

bool Cast_OpInferSymbolicShape(pir::Operation *op,
                               pir::ShapeConstraintIRAnalysis *shape_analysis) {
  return InferSymbolicShapeAllEqualUnary(op, shape_analysis);
}

bool ExpOpInferSymbolicShape(pir::Operation *op,
                             pir::ShapeConstraintIRAnalysis *shape_analysis) {
  return InferSymbolicShapeAllEqualUnary(op, shape_analysis);
}

bool Exp_OpInferSymbolicShape(pir::Operation *op,
                              pir::ShapeConstraintIRAnalysis *shape_analysis) {
  return InferSymbolicShapeAllEqualUnary(op, shape_analysis);
}

bool SubtractOpInferSymbolicShape(
    pir::Operation *op, pir::ShapeConstraintIRAnalysis *shape_analysis) {
  return InferSymbolicShapeAllEqualBinary(op, shape_analysis);
}

bool Subtract_OpInferSymbolicShape(
    pir::Operation *op, pir::ShapeConstraintIRAnalysis *shape_analysis) {
  return InferSymbolicShapeAllEqualBinary(op, shape_analysis);
}

bool ShapeOpInferSymbolicShape(pir::Operation *op,
                               pir::ShapeConstraintIRAnalysis *shape_analysis) {
  pir::Value operand_source = op->operand_source(0);
  std::string operand_source_id = pir::GetValueId(&operand_source);
  pir::OpResult res = op->result(0);
  std::string res_id = pir::GetValueId(&res);

  std::vector<int64_t> dims =
      common::vectorize(res.type().dyn_cast<pir::DenseTensorType>().dims());

  std::vector<symbol::DimExpr> shapes;
  for (int64_t dim : dims) {
    symbol::DimExpr dim_expr;
    if (dim == -1) {
      symbol::DimExpr res_dim_expr(shape_analysis->GetNextSymName());
      dim_expr = res_dim_expr;
    } else {
      symbol::DimExpr res_dim_expr(dim);
      dim_expr = res_dim_expr;
    }
    shapes.push_back(dim_expr);
  }

  symbol::ShapeOrDataDimExprs shape_data{shapes};
  shape_analysis->value_id_to_shapeordata_[res_id] = shape_data;
  return true;
}

bool ShapeSrOpInferSymbolicShape(
    pir::Operation *op, pir::ShapeConstraintIRAnalysis *shape_analysis) {
  return ShapeOpInferSymbolicShape(op, shape_analysis);
}

bool StackOpInferSymbolicShape(pir::Operation *op,
                               pir::ShapeConstraintIRAnalysis *shape_analysis) {
  pir::Value operand_source = op->operand_source(0);
  std::string operand_source_id = pir::GetValueId(&operand_source);
  pir::OpResult res = op->result(0);
  std::string res_id = pir::GetValueId(&res);

  symbol::ShapeOrDataDimExprs shape_data;
  shape_data = shape_analysis->value_id_to_shapeordata_[operand_source_id];
  shape_analysis->value_id_to_shapeordata_[res_id] = shape_data;
  return true;
}

bool ReshapeOpInferSymbolicShape(
    pir::Operation *op, pir::ShapeConstraintIRAnalysis *shape_analysis) {
  pir::Value operand_source_1 = op->operand_source(1);
  std::string operand_source_1_id = pir::GetValueId(&operand_source_1);
  pir::OpResult res = op->result(0);
  std::string res_id = pir::GetValueId(&res);

  symbol::ShapeOrDataDimExprs shape_data;

  shape_data = shape_analysis->value_id_to_shapeordata_[operand_source_1_id];
  shape_analysis->value_id_to_shapeordata_[res_id] = shape_data;
  return true;
}

bool Reshape_OpInferSymbolicShape(
    pir::Operation *op, pir::ShapeConstraintIRAnalysis *shape_analysis) {
  return ReshapeOpInferSymbolicShape(op, shape_analysis);
}

}  // namespace paddle::dialect
namespace cinn::dialect {

bool SliceOpInferSymbolicShape(pir::Operation *op,
                               pir::ShapeConstraintIRAnalysis *shape_analysis) {
  pir::Value operand_source = op->operand_source(0);
  std::string operand_source_id = pir::GetValueId(&operand_source);
  pir::OpResult res = op->result(0);
  std::string res_id = pir::GetValueId(&res);

  std::vector<int64_t> dims =
      common::vectorize(res.type().dyn_cast<pir::DenseTensorType>().dims());

  std::vector<symbol::DimExpr> shapes;
  for (int64_t dim : dims) {
    symbol::DimExpr dim_expr;
    if (dim == -1) {
      symbol::DimExpr res_dim_expr(shape_analysis->GetNextSymName());
      dim_expr = res_dim_expr;
    } else {
      symbol::DimExpr res_dim_expr(dim);
      dim_expr = res_dim_expr;
    }
    shapes.push_back(dim_expr);
  }

  // pir::AttributeMap attributes = op->attributes();

  // auto attr_starts =
  //     attributes["starts"].dyn_cast<pir::ArrayAttribute>().AsVector();
  // auto start = attr_starts[0].dyn_cast<pir::Int64Attribute>().data();

  // auto attr_ends =
  //     attributes["ends"].dyn_cast<pir::ArrayAttribute>().AsVector();
  // auto end = attr_ends[0].dyn_cast<pir::Int64Attribute>().data();

  symbol::ShapeOrDataDimExprs shape_data{shapes};
  shape_analysis->value_id_to_shapeordata_[res_id] = shape_data;
  return true;
}

}  // namespace cinn::dialect

IR_DEFINE_EXPLICIT_TYPE_ID(paddle::dialect::InferSymbolicShapeInterface)
