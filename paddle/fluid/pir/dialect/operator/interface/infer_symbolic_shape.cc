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
#include "paddle/fluid/pir/dialect/operator/ir/op_attribute.h"
#include "paddle/pir/core/builtin_attribute.h"
#include "paddle/pir/core/builtin_type.h"
#include "paddle/pir/core/builtin_type_interfaces.h"
#include "paddle/pir/dialect/shape/ir/shape_attribute.h"

namespace paddle::dialect {

bool InferSymbolicShapeInterface::InferSymbolicShape(
    pir::ShapeConstraintIRAnalysis *shape_analysis) {
  return impl_->infer_symbolic_shapes(operation(), shape_analysis);
}
}  // namespace paddle::dialect

namespace {

bool SameOperandsAndResultShape(
    pir::Operation *op, pir::ShapeConstraintIRAnalysis *shape_analysis) {
  pir::Value operand_source = op->operand_source(0);

  symbol::ShapeOrDataDimExprs operand_shape_or_data =
      shape_analysis->GetShapeOrDataForValue(operand_source);

  op->set_attribute("symbolic_shape",
                    pir::shape::SymbolAttribute::get(pir::IrContext::Instance(),
                                                     operand_shape_or_data));
  pir::OpResult res = op->result(0);
  shape_analysis->SetShapeOrDataForValue(res, operand_shape_or_data);
  return true;
}

bool InferSymbolicShapeElementWiseBinary(
    pir::Operation *op, pir::ShapeConstraintIRAnalysis *shape_analysis) {
  std::vector<symbol::DimExpr> shape_0{
      shape_analysis->GetShapeOrDataForValue(op->operand_source(0)).shape()};

  std::vector<symbol::DimExpr> shape_1{
      shape_analysis->GetShapeOrDataForValue(op->operand_source(1)).shape()};

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

  std::vector<symbol::DimExpr> shapes;
  symbol::DimExprBuilder builder{nullptr};
  for (size_t i = 0; i < shape_0.size(); i++) {
    if (shape_0[i] == shape_1[i]) {
      shapes.emplace_back(shape_0[i]);
    } else {
      shapes.emplace_back(builder.Broadcast(shape_0[i], shape_1[i]));
    }
  }

  // TODO(lanxianghit): fill data when the operation is on shape computation
  std::vector<symbol::DimExpr> data;

  pir::OpResult res = op->result(0);
  symbol::ShapeOrDataDimExprs shape_data{shapes, data};
  shape_analysis->SetShapeOrDataForValue(res, shape_data);
  op->set_attribute(
      "symbolic_shape",
      pir::shape::SymbolAttribute::get(pir::IrContext::Instance(), shape_data));
  return true;
}

}  // namespace

namespace paddle::dialect {
bool AbsOpInferSymbolicShape(pir::Operation *op,
                             pir::ShapeConstraintIRAnalysis *shape_analysis) {
  return SameOperandsAndResultShape(op, shape_analysis);
}

bool Abs_OpInferSymbolicShape(pir::Operation *op,
                              pir::ShapeConstraintIRAnalysis *shape_analysis) {
  return SameOperandsAndResultShape(op, shape_analysis);
}

bool DataOpInferSymbolicShape(pir::Operation *op,
                              pir::ShapeConstraintIRAnalysis *shape_analysis) {
  auto attributes = op->attributes();
  pir::Attribute attr = attributes["shape"];
  std::vector<int64_t> dims =
      attr.dyn_cast<paddle::dialect::IntArrayAttribute>().data().GetData();

  std::vector<symbol::DimExpr> sym_dims;
  for (auto dim : dims) {
    symbol::DimExpr dim_expr;
    if (dim == pir::ShapedTypeInterface::kDynamic) {
      symbol::DimExpr symbolic_dim_expr(shape_analysis->GetNextSymName());
      dim_expr = symbolic_dim_expr;
    } else {
      symbol::DimExpr numeric_dim_expr(dim);
      dim_expr = numeric_dim_expr;
    }
    sym_dims.push_back(dim_expr);
  }

  symbol::ShapeOrDataDimExprs shape_data{sym_dims};
  op->set_attribute(
      "symbolic_shape",
      pir::shape::SymbolAttribute::get(pir::IrContext::Instance(), shape_data));

  pir::OpResult res = op->result(0);
  shape_analysis->SetShapeOrDataForValue(res, shape_data);

  return true;
}

bool AddOpInferSymbolicShape(pir::Operation *op,
                             pir::ShapeConstraintIRAnalysis *shape_analysis) {
  return InferSymbolicShapeElementWiseBinary(op, shape_analysis);
}

bool Add_OpInferSymbolicShape(pir::Operation *op,
                              pir::ShapeConstraintIRAnalysis *shape_analysis) {
  return InferSymbolicShapeElementWiseBinary(op, shape_analysis);
}

bool CastOpInferSymbolicShape(pir::Operation *op,
                              pir::ShapeConstraintIRAnalysis *shape_analysis) {
  return SameOperandsAndResultShape(op, shape_analysis);
}

bool Cast_OpInferSymbolicShape(pir::Operation *op,
                               pir::ShapeConstraintIRAnalysis *shape_analysis) {
  return SameOperandsAndResultShape(op, shape_analysis);
}

bool ExpOpInferSymbolicShape(pir::Operation *op,
                             pir::ShapeConstraintIRAnalysis *shape_analysis) {
  return SameOperandsAndResultShape(op, shape_analysis);
}

bool Exp_OpInferSymbolicShape(pir::Operation *op,
                              pir::ShapeConstraintIRAnalysis *shape_analysis) {
  return SameOperandsAndResultShape(op, shape_analysis);
}

bool SubtractOpInferSymbolicShape(
    pir::Operation *op, pir::ShapeConstraintIRAnalysis *shape_analysis) {
  return SameOperandsAndResultShape(op, shape_analysis);
}

bool Subtract_OpInferSymbolicShape(
    pir::Operation *op, pir::ShapeConstraintIRAnalysis *shape_analysis) {
  return SameOperandsAndResultShape(op, shape_analysis);
}

bool ShapeOpInferSymbolicShape(pir::Operation *op,
                               pir::ShapeConstraintIRAnalysis *shape_analysis) {
  pir::Value operand_source = op->operand_source(0);
  pir::OpResult res = op->result(0);

  symbol::ShapeOrDataDimExprs operand_shape_or_data =
      shape_analysis->GetShapeOrDataForValue(operand_source);

  std::vector<int64_t> dims =
      common::vectorize(res.type().dyn_cast<pir::DenseTensorType>().dims());

  // TODO(zhangbopd): check whether it's right for other cases
  std::vector<symbol::DimExpr> sym_shape;
  for (auto dim : dims) {
    symbol::DimExpr dim_expr;
    if (dim == -1) {
      symbol::DimExpr res_dim_expr(shape_analysis->GetNextSymName());
      dim_expr = res_dim_expr;
    } else {
      symbol::DimExpr res_dim_expr(dim);
      dim_expr = res_dim_expr;
    }
    sym_shape.push_back(dim_expr);
  }

  symbol::ShapeOrDataDimExprs shape_or_data{sym_shape,
                                            operand_shape_or_data.shape()};

  shape_analysis->SetShapeOrDataForValue(res, shape_or_data);
  op->set_attribute("symbolic_shape",
                    pir::shape::SymbolAttribute::get(pir::IrContext::Instance(),
                                                     shape_or_data));
  return true;
}

bool ShapeSrOpInferSymbolicShape(
    pir::Operation *op, pir::ShapeConstraintIRAnalysis *shape_analysis) {
  return ShapeOpInferSymbolicShape(op, shape_analysis);
}

bool StackOpInferSymbolicShape(pir::Operation *op,
                               pir::ShapeConstraintIRAnalysis *shape_analysis) {
  pir::Value operand_source = op->operand_source(0);
  symbol::ShapeOrDataDimExprs operand_shape_or_data =
      shape_analysis->GetShapeOrDataForValue(operand_source);

  std::vector<symbol::DimExpr> out_dims;
  if (operand_shape_or_data.data().has_value()) {
    out_dims = operand_shape_or_data.data().value();
  }
  // else : pir::VectorType x =
  // operand_source.type().dyn_cast<pir::VectorType>();
  // TODO(zhangbopd): else branch is not implemented yet.

  symbol::ShapeOrDataDimExprs shape_data{out_dims};
  if (operand_shape_or_data.data().has_value()) {
    shape_data.SetData(operand_shape_or_data.shape());
  }

  op->set_attribute(
      "symbolic_shape",
      pir::shape::SymbolAttribute::get(pir::IrContext::Instance(), shape_data));
  pir::OpResult res = op->result(0);
  shape_analysis->SetShapeOrDataForValue(res, shape_data);
  return true;
}

bool SumOpInferSymbolicShape(pir::Operation *op,
                             pir::ShapeConstraintIRAnalysis *shape_analysis) {
  VLOG(1) << "SumOpInferSymbolicShape begin";

  auto attributes = op->attributes();
  pir::Attribute attr = attributes["keepdim"];
  bool keepdim = attr.dyn_cast<pir::BoolAttribute>().data();
  if (keepdim) {
    InferSymbolicShapeElementWiseBinary(op, shape_analysis);
    return true;
  }

  pir::OpResult res = op->result(0);

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
    VLOG(1) << "SumOpInferSymbolicShape dim_expr = " << dim_expr;
  }

  symbol::ShapeOrDataDimExprs shape_data{shapes};
  shape_analysis->SetShapeOrDataForValue(res, shape_data);
  return true;
}

bool ReshapeOpInferSymbolicShape(
    pir::Operation *op, pir::ShapeConstraintIRAnalysis *shape_analysis) {
  pir::Value operand_source_shape = op->operand_source(1);

  symbol::ShapeOrDataDimExprs operand_shape_or_data =
      shape_analysis->GetShapeOrDataForValue(operand_source_shape);

  std::vector<symbol::DimExpr> out_dims;
  if (operand_shape_or_data.data().has_value()) {
    out_dims = operand_shape_or_data.data().value();
  }

  symbol::ShapeOrDataDimExprs shape_data{out_dims};
  op->set_attribute(
      "symbolic_shape",
      pir::shape::SymbolAttribute::get(pir::IrContext::Instance(), shape_data));

  pir::OpResult res0 = op->result(0);
  pir::OpResult res1 = op->result(1);
  shape_analysis->SetShapeOrDataForValue(res0, shape_data);
  shape_analysis->SetShapeOrDataForValue(
      res1, shape_analysis->GetShapeOrDataForValue(operand_source_shape));
  return true;
}

bool Reshape_OpInferSymbolicShape(
    pir::Operation *op, pir::ShapeConstraintIRAnalysis *shape_analysis) {
  return ReshapeOpInferSymbolicShape(op, shape_analysis);
}

bool FullIntArrayOpInferSymbolicShape(
    pir::Operation *op, pir::ShapeConstraintIRAnalysis *shape_analysis) {
  auto attributes = op->attributes();
  pir::Attribute attr = attributes["value"];
  const auto &vec = attr.dyn_cast<pir::ArrayAttribute>().AsVector();

  std::vector<symbol::DimExpr> data;
  for (auto item : vec) {
    int64_t i = item.dyn_cast<pir::Int64Attribute>().data();
    data.push_back(symbol::DimExpr(i));
  }

  // TODO(zhangbopd): use op->result(0) to infer the shape
  std::vector<symbol::DimExpr> shape(std::int64_t(data.size()));

  symbol::ShapeOrDataDimExprs shape_data{shape, data};

  op->set_attribute(
      "symbolic_shape",
      pir::shape::SymbolAttribute::get(pir::IrContext::Instance(), shape_data));

  pir::OpResult res = op->result(0);
  shape_analysis->SetShapeOrDataForValue(res, shape_data);
  return true;
}

bool SliceOpInferSymbolicShape(pir::Operation *op,
                               pir::ShapeConstraintIRAnalysis *shape_analysis) {
  // TODO(zhangbopd): Not implemented yet.
  pir::Value operand_source = op->operand_source(0);
  pir::Value operand_starts = op->operand_source(1);
  // pir::Value operand_ends = op->operand_source(2);

  symbol::ShapeOrDataDimExprs operand_shape_or_data =
      shape_analysis->GetShapeOrDataForValue(operand_source);
  symbol::ShapeOrDataDimExprs starts_shape_data =
      shape_analysis->GetShapeOrDataForValue(operand_starts);

  int64_t start = 0;
  if (starts_shape_data.data().has_value()) {
    start = starts_shape_data.data()->at(0).Get<int64_t>();
  }

  std::vector<symbol::DimExpr> out_dims;
  if (operand_shape_or_data.data().has_value()) {
    out_dims.push_back(operand_shape_or_data.data().value()[start]);
  }

  symbol::ShapeOrDataDimExprs shape_data{out_dims};
  if (operand_shape_or_data.data().has_value()) {
    shape_data =
        symbol::ShapeOrDataDimExprs::MakeConsistentShapeOrData(shape_data);
  }
  op->set_attribute(
      "symbolic_shape",
      pir::shape::SymbolAttribute::get(pir::IrContext::Instance(), shape_data));

  pir::OpResult res = op->result(0);
  shape_analysis->SetShapeOrDataForValue(res, shape_data);
  return true;
}

bool FullOpInferSymbolicShape(pir::Operation *op,
                              pir::ShapeConstraintIRAnalysis *shape_analysis) {
  auto attributes = op->attributes();
  pir::Attribute attr_value = attributes["value"];
  auto value = attr_value.dyn_cast<pir::FloatAttribute>().data();

  pir::Attribute attr_shape = attributes["shape"];
  const auto &shape_vec =
      attr_shape.dyn_cast<paddle::dialect::IntArrayAttribute>()
          .data()
          .GetData();

  std::vector<symbol::DimExpr> sym_shape;
  for (auto dim : shape_vec) {
    symbol::DimExpr dim_expr;

    if (dim == -1) {
      symbol::DimExpr res_dim_expr(shape_analysis->GetNextSymName());
      dim_expr = res_dim_expr;
    } else {
      symbol::DimExpr res_dim_expr(dim);
      dim_expr = res_dim_expr;
    }
    sym_shape.push_back(dim_expr);
  }

  symbol::ShapeOrDataDimExprs shape_data{sym_shape};

  op->set_attribute(
      "symbolic_shape",
      pir::shape::SymbolAttribute::get(pir::IrContext::Instance(), shape_data));

  pir::OpResult res = op->result(0);
  shape_analysis->SetShapeOrDataForValue(res, shape_data);
  return true;
}

bool DivideOpInferSymbolicShape(
    pir::Operation *op, pir::ShapeConstraintIRAnalysis *shape_analysis) {
  return true;
}

bool Divide_OpInferSymbolicShape(
    pir::Operation *op, pir::ShapeConstraintIRAnalysis *shape_analysis) {
  return true;
}

bool ElementwisePowOpInferSymbolicShape(
    pir::Operation *op, pir::ShapeConstraintIRAnalysis *shape_analysis) {
  InferSymbolicShapeElementWiseBinary(op, shape_analysis);
  return true;
}

bool FullWithTensorOpInferSymbolicShape(
    pir::Operation *op, pir::ShapeConstraintIRAnalysis *shape_analysis) {
  return true;
}

bool MultiplyOpInferSymbolicShape(
    pir::Operation *op, pir::ShapeConstraintIRAnalysis *shape_analysis) {
  return true;
}

bool Multiply_OpInferSymbolicShape(
    pir::Operation *op, pir::ShapeConstraintIRAnalysis *shape_analysis) {
  return true;
}

bool MultiplySrOpInferSymbolicShape(
    pir::Operation *op, pir::ShapeConstraintIRAnalysis *shape_analysis) {
  return true;
}

bool MultiplySr_OpInferSymbolicShape(
    pir::Operation *op, pir::ShapeConstraintIRAnalysis *shape_analysis) {
  return true;
}

bool ScaleOpInferSymbolicShape(pir::Operation *op,
                               pir::ShapeConstraintIRAnalysis *shape_analysis) {
  return true;
}

bool Scale_OpInferSymbolicShape(
    pir::Operation *op, pir::ShapeConstraintIRAnalysis *shape_analysis) {
  return true;
}

bool ScaleSrOpInferSymbolicShape(
    pir::Operation *op, pir::ShapeConstraintIRAnalysis *shape_analysis) {
  return true;
}

bool ScaleSr_OpInferSymbolicShape(
    pir::Operation *op, pir::ShapeConstraintIRAnalysis *shape_analysis) {
  return true;
}

}  // namespace paddle::dialect
namespace cinn::dialect {

bool SliceOpInferSymbolicShape(pir::Operation *op,
                               pir::ShapeConstraintIRAnalysis *shape_analysis) {
  // TODO(zhangbopd): Not implemented yet, different from the one in paddle
  // dialect.
  pir::Value operand_source = op->operand_source(0);
  symbol::ShapeOrDataDimExprs operand_shape_or_data =
      shape_analysis->GetShapeOrDataForValue(operand_source);
  pir::AttributeMap attributes = op->attributes();

  std::vector<pir::Attribute> attr_starts =
      attributes["starts"].dyn_cast<pir::ArrayAttribute>().AsVector();

  int64_t start = attr_starts[0].dyn_cast<pir::Int64Attribute>().data();

  std::vector<symbol::DimExpr> out_dims;
  if (operand_shape_or_data.data().has_value()) {
    out_dims.push_back(operand_shape_or_data.data().value()[start]);
  }

  symbol::ShapeOrDataDimExprs shape_data{out_dims};
  if (operand_shape_or_data.data().has_value()) {
    shape_data.SetData(operand_shape_or_data.shape());
  }
  op->set_attribute(
      "symbolic_shape",
      pir::shape::SymbolAttribute::get(pir::IrContext::Instance(), shape_data));

  pir::OpResult res = op->result(0);
  shape_analysis->SetShapeOrDataForValue(res, shape_data);
  return true;
}

}  // namespace cinn::dialect

IR_DEFINE_EXPLICIT_TYPE_ID(paddle::dialect::InferSymbolicShapeInterface)
