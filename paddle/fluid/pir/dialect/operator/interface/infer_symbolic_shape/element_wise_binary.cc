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

#include "paddle/fluid/pir/dialect/operator/interface/infer_symbolic_shape/element_wise_binary.h"
#include "paddle/fluid/pir/dialect/operator/ir/pd_op.h"

bool InferSymbolicShapeElementWiseBinary(
    pir::Operation *op,
    pir::InferSymbolicShapeContext *infer_context,
    const std::function<symbol::DimExpr(const symbol::DimExpr &,
                                        const symbol::DimExpr &)>
        &DataComputeFunc = nullptr) {
  const auto &x_shape =
      infer_context->GetShapeOrDataForValue(op->operand_source(0));
  std::vector<symbol::DimExpr> shape_0 = x_shape.shape();

  const auto &y_shape =
      infer_context->GetShapeOrDataForValue(op->operand_source(1));
  std::vector<symbol::DimExpr> shape_1 = y_shape.shape();

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
    symbol::DimExprBuilder builder;
    for (size_t i = 0; i < shape_0.size(); i++) {
      if (shape_0[i] == shape_1[i]) {
        shapes.emplace_back(shape_0[i]);
      } else if (shape_0[i] == 1) {
        shapes.emplace_back(shape_1[i]);
      } else if (shape_1[i] == 1) {
        shapes.emplace_back(shape_0[i]);
      } else {
        shapes.emplace_back(builder.Broadcast(shape_0[i], shape_1[i]));
        infer_context->AddBroadcastableCstr(shape_0[i], shape_1[i]);
      }
    }
    return shapes;
  }();

  if (x_shape.data() && y_shape.data() && DataComputeFunc) {
    PADDLE_ENFORCE_LE(
        x_shape.shape().size(),
        1,
        common::errors::InvalidArgument("When compute data, the rank of x "
                                        "should be 0 or 1, but now received %d",
                                        x_shape.shape().size()));
    PADDLE_ENFORCE_LE(
        y_shape.shape().size(),
        1,
        common::errors::InvalidArgument("When compute data, the rank of y "
                                        "should be 0 or 1, but now received %d",
                                        y_shape.shape().size()));
    PADDLE_ENFORCE_EQ(x_shape.data()->size(),
                      y_shape.data()->size(),
                      common::errors::InvalidArgument(
                          "When compute data, the size of x and y should be "
                          "equal, but now received %d and %d",
                          x_shape.data()->size(),
                          y_shape.data()->size()));
    std::vector<symbol::DimExpr> out_data;
    for (size_t i = 0; i < x_shape.data()->size(); ++i) {
      out_data.emplace_back(
          DataComputeFunc(x_shape.data()->at(i), y_shape.data()->at(i)));
    }
    symbol::ShapeOrDataDimExprs shape_data{
        symbol::TensorShapeOrDataDimExprs(shapes, out_data)};
    infer_context->SetShapeOrDataForValue(op->result(0), shape_data);
  } else {
    symbol::ShapeOrDataDimExprs shape_data{
        symbol::TensorShapeOrDataDimExprs(shapes)};
    infer_context->SetShapeOrDataForValue(op->result(0), shape_data);
  }
  return true;
}

#define OP_ELEMENT_WISE_BINARY(name)                                       \
  bool name##OpInferSymbolicShape(                                         \
      pir::Operation *op, pir::InferSymbolicShapeContext *infer_context) { \
    return InferSymbolicShapeElementWiseBinary(op, infer_context);         \
  }

namespace paddle::dialect {

bool AddOpInferSymbolicShape(pir::Operation *op,
                             pir::InferSymbolicShapeContext *infer_context) {
  return InferSymbolicShapeElementWiseBinary(
      op,
      infer_context,
      [](const symbol::DimExpr &x, const symbol::DimExpr &y) { return x + y; });
}

bool DivideOpInferSymbolicShape(pir::Operation *op,
                                pir::InferSymbolicShapeContext *infer_context) {
  return InferSymbolicShapeElementWiseBinary(
      op,
      infer_context,
      [](const symbol::DimExpr &x, const symbol::DimExpr &y) { return x / y; });
}

bool MultiplyOpInferSymbolicShape(
    pir::Operation *op, pir::InferSymbolicShapeContext *infer_context) {
  return InferSymbolicShapeElementWiseBinary(
      op,
      infer_context,
      [](const symbol::DimExpr &x, const symbol::DimExpr &y) { return x * y; });
}

bool SubtractOpInferSymbolicShape(
    pir::Operation *op, pir::InferSymbolicShapeContext *infer_context) {
  return InferSymbolicShapeElementWiseBinary(
      op,
      infer_context,
      [](const symbol::DimExpr &x, const symbol::DimExpr &y) { return x - y; });
}

bool FloorDivideOpInferSymbolicShape(
    pir::Operation *op, pir::InferSymbolicShapeContext *infer_context) {
  return InferSymbolicShapeElementWiseBinary(
      op,
      infer_context,
      [&](const symbol::DimExpr &x, const symbol::DimExpr &y) {
        // Note: The floor_divide operation in Paddle rounds the quotients
        // towards negative infinity. This is different from the standard
        // division in C++, so rounding errors may occur.
        return x / y;
      });
}

bool MinimumOpInferSymbolicShape(
    pir::Operation *op, pir::InferSymbolicShapeContext *infer_context) {
  return InferSymbolicShapeElementWiseBinary(
      op,
      infer_context,
      [](const symbol::DimExpr &x, const symbol::DimExpr &y) {
        symbol::DimExprBuilder builder;
        return builder.Min(x, y);
      });
}

OP_ELEMENT_WISE_BINARY(Add_)
OP_ELEMENT_WISE_BINARY(BitwiseAnd)
OP_ELEMENT_WISE_BINARY(BitwiseAnd_)
OP_ELEMENT_WISE_BINARY(BitwiseRightShift)
OP_ELEMENT_WISE_BINARY(BitwiseRightShift_)
OP_ELEMENT_WISE_BINARY(BitwiseLeftShift)
OP_ELEMENT_WISE_BINARY(BitwiseLeftShift_)
OP_ELEMENT_WISE_BINARY(BitwiseOr)
OP_ELEMENT_WISE_BINARY(BitwiseOr_)
OP_ELEMENT_WISE_BINARY(BitwiseXor)
OP_ELEMENT_WISE_BINARY(BitwiseXor_)
OP_ELEMENT_WISE_BINARY(Complex)
OP_ELEMENT_WISE_BINARY(Copysign)
OP_ELEMENT_WISE_BINARY(Copysign_)
OP_ELEMENT_WISE_BINARY(Divide_)
OP_ELEMENT_WISE_BINARY(ElementwisePow)
OP_ELEMENT_WISE_BINARY(Equal)
OP_ELEMENT_WISE_BINARY(Equal_)
OP_ELEMENT_WISE_BINARY(FloorDivide_)
OP_ELEMENT_WISE_BINARY(Fmax)
OP_ELEMENT_WISE_BINARY(Fmin)
OP_ELEMENT_WISE_BINARY(Gammaincc)
OP_ELEMENT_WISE_BINARY(Gammaincc_)
OP_ELEMENT_WISE_BINARY(GreaterEqual)
OP_ELEMENT_WISE_BINARY(GreaterEqual_)
OP_ELEMENT_WISE_BINARY(GreaterThan)
OP_ELEMENT_WISE_BINARY(GreaterThan_)
OP_ELEMENT_WISE_BINARY(Heaviside)
OP_ELEMENT_WISE_BINARY(LessEqual)
OP_ELEMENT_WISE_BINARY(LessEqual_)
OP_ELEMENT_WISE_BINARY(LessThan)
OP_ELEMENT_WISE_BINARY(LessThan_)
OP_ELEMENT_WISE_BINARY(LogicalAnd)
OP_ELEMENT_WISE_BINARY(LogicalAnd_)
OP_ELEMENT_WISE_BINARY(LogicalOr)
OP_ELEMENT_WISE_BINARY(LogicalOr_)
OP_ELEMENT_WISE_BINARY(LogicalXor)
OP_ELEMENT_WISE_BINARY(LogicalXor_)
OP_ELEMENT_WISE_BINARY(Maximum)
OP_ELEMENT_WISE_BINARY(MultiplySr)
OP_ELEMENT_WISE_BINARY(MultiplySr_)
OP_ELEMENT_WISE_BINARY(Multiply_)
OP_ELEMENT_WISE_BINARY(Nextafter)
OP_ELEMENT_WISE_BINARY(NotEqual)
OP_ELEMENT_WISE_BINARY(NotEqual_)
OP_ELEMENT_WISE_BINARY(Remainder)
OP_ELEMENT_WISE_BINARY(Remainder_)
OP_ELEMENT_WISE_BINARY(Subtract_)

}  // namespace paddle::dialect

#undef OP_ELEMENT_WISE_BINARY
