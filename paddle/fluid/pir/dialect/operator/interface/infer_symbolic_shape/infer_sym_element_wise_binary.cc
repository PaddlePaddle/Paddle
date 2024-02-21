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

#include "paddle/fluid/pir/dialect/operator/interface/infer_symbolic_shape/infer_sym_element_wise_binary.h"
#include "paddle/fluid/pir/dialect/operator/ir/pd_op.h"

bool InferSymbolicShapeElementWiseBinary(
    pir::Operation *op, pir::ShapeConstraintIRAnalysis *shape_analysis) {
  const auto &x_shapeordata =
      shape_analysis->GetShapeOrDataForValue(op->operand_source(0));
  std::vector<symbol::DimExpr> shape_0;
  // For ElementWiseBinary ops, if the input tensor is from full op, the value
  // of fullop is useless, only the shape need doing broadcast
  bool x_from_fullop =
      op->operand_source(0).defining_op()->isa<paddle::dialect::FullOp>();
  if (!x_from_fullop && x_shapeordata.data().has_value()) {
    shape_0 = x_shapeordata.data().value();
  } else {
    shape_0 = x_shapeordata.shape();
  }

  const auto &y_shapeordata =
      shape_analysis->GetShapeOrDataForValue(op->operand_source(1));
  std::vector<symbol::DimExpr> shape_1;
  bool y_from_fullop =
      op->operand_source(1).defining_op()->isa<paddle::dialect::FullOp>();
  if (!y_from_fullop && y_shapeordata.data().has_value()) {
    shape_1 = y_shapeordata.data().value();
  } else {
    shape_1 = y_shapeordata.shape();
  }

  int diff = shape_0.size() - shape_1.size();
  if (diff > 0) {
    for (int i = 0; i < diff; i++) {
      shape_1.emplace(shape_1.begin(), 1);
    }
  } else {
    for (int i = 0; i < -diff; i++) {
      shape_0.emplace(shape_0.begin(), 1);
    }
  }

  const std::vector<symbol::DimExpr> shapes = [&] {
    std::vector<symbol::DimExpr> shapes;
    symbol::DimExprBuilder builder{nullptr};
    for (size_t i = 0; i < shape_0.size(); i++) {
      if (shape_0[i] == shape_1[i]) {
        shapes.emplace_back(shape_0[i]);
      } else if (shape_0[i] == 1) {
        shapes.emplace_back(shape_1[i]);
      } else if (shape_1[i] == 1) {
        shapes.emplace_back(shape_0[i]);
      } else {
        shapes.emplace_back(builder.Broadcast(shape_0[i], shape_1[i]));
      }
    }
    return shapes;
  }();

  // TODO(lanxianghit): fill data when the operation is on shape computation
  // std::vector<symbol::DimExpr> data;
  symbol::ShapeOrDataDimExprs shape_data{
      symbol::TensorShapeOrDataDimExprs(shapes)};
  shape_analysis->SetShapeOrDataForValue(op->result(0), shape_data);

  return true;
}

namespace paddle::dialect {

bool AddOpInferSymbolicShape(pir::Operation *op,
                             pir::ShapeConstraintIRAnalysis *shape_analysis) {
  return InferSymbolicShapeElementWiseBinary(op, shape_analysis);
}

bool Add_OpInferSymbolicShape(pir::Operation *op,
                              pir::ShapeConstraintIRAnalysis *shape_analysis) {
  return InferSymbolicShapeElementWiseBinary(op, shape_analysis);
}

bool MultiplyOpInferSymbolicShape(
    pir::Operation *op, pir::ShapeConstraintIRAnalysis *shape_analysis) {
  return InferSymbolicShapeElementWiseBinary(op, shape_analysis);
}
bool MultiplySrOpInferSymbolicShape(
    pir::Operation *op, pir::ShapeConstraintIRAnalysis *shape_analysis) {
  return InferSymbolicShapeElementWiseBinary(op, shape_analysis);
}
bool Multiply_OpInferSymbolicShape(
    pir::Operation *op, pir::ShapeConstraintIRAnalysis *shape_analysis) {
  return InferSymbolicShapeElementWiseBinary(op, shape_analysis);
}
bool MultiplySr_OpInferSymbolicShape(
    pir::Operation *op, pir::ShapeConstraintIRAnalysis *shape_analysis) {
  return InferSymbolicShapeElementWiseBinary(op, shape_analysis);
}

bool DivideOpInferSymbolicShape(
    pir::Operation *op, pir::ShapeConstraintIRAnalysis *shape_analysis) {
  return InferSymbolicShapeElementWiseBinary(op, shape_analysis);
}
bool Divide_OpInferSymbolicShape(
    pir::Operation *op, pir::ShapeConstraintIRAnalysis *shape_analysis) {
  return InferSymbolicShapeElementWiseBinary(op, shape_analysis);
}

bool ElementwisePowOpInferSymbolicShape(
    pir::Operation *op, pir::ShapeConstraintIRAnalysis *shape_analysis) {
  return InferSymbolicShapeElementWiseBinary(op, shape_analysis);
}

}  // namespace paddle::dialect
