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
#include "paddle/fluid/pir/dialect/operator/ir/pd_op.h"
#include "paddle/pir/core/builtin_attribute.h"
#include "paddle/pir/core/builtin_type.h"
#include "paddle/pir/core/builtin_type_interfaces.h"
#include "paddle/pir/dialect/shape/ir/shape_attribute.h"

template <typename T = int64_t>
std::vector<T> GetVectorAttr(const ::pir::Operation *op,
                             const std::string &name) {
  auto &attr_map = op->attributes();
  PADDLE_ENFORCE(
      attr_map.count(name),
      phi::errors::PreconditionNotMet(
          "attr [%s] MUST in attribute map for [%s] op", name, op->name()));
  auto &val = attr_map.at(name);

  PADDLE_ENFORCE(val.isa<::pir::ArrayAttribute>(),
                 phi::errors::PreconditionNotMet(
                     "axis Type MUST ArrayAttribute for [%s] op", op->name()));
  auto array_list = val.dyn_cast<::pir::ArrayAttribute>().AsVector();
  std::vector<T> vec_res;
  if (array_list.size() > 0) {
    PADDLE_ENFORCE_EQ(array_list[0].isa<::pir::Int64Attribute>(),
                      true,
                      phi::errors::Unimplemented(
                          "the 0th elementwise MUST be ir::Int64Attribute"));
    for (size_t i = 0; i < array_list.size(); ++i) {
      vec_res.push_back(array_list[i].dyn_cast<::pir::Int64Attribute>().data());
    }
  }
  return vec_res;
}

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
  pir::Value res = op->result(0);
  shape_analysis->SetShapeOrDataForValue(res, operand_shape_or_data);
  return true;
}

bool InferSymbolicShapeElementWiseBinary(
    pir::Operation *op, pir::ShapeConstraintIRAnalysis *shape_analysis) {
  auto x_shapeordata =
      shape_analysis->GetShapeOrDataForValue(op->operand_source(0));
  std::vector<symbol::DimExpr> shape_0;
  if (x_shapeordata.data().has_value()) {
    shape_0 = x_shapeordata.data().value();
  } else {
    shape_0 = x_shapeordata.shape();
  }

  auto y_shapeordata =
      shape_analysis->GetShapeOrDataForValue(op->operand_source(1));
  std::vector<symbol::DimExpr> shape_1;
  if (y_shapeordata.data().has_value()) {
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

  // TODO(lanxianghit): fill data when the operation is on shape computation
  // std::vector<symbol::DimExpr> data;
  pir::Value res = op->result(0);
  symbol::ShapeOrDataDimExprs shape_data{
      symbol::TensorShapeOrDataDimExprs(shapes)};
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

  symbol::ShapeOrDataDimExprs shape_data{
      symbol::TensorShapeOrDataDimExprs(sym_dims)};
  op->set_attribute(
      "symbolic_shape",
      pir::shape::SymbolAttribute::get(pir::IrContext::Instance(), shape_data));

  pir::Value res = op->result(0);
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
  pir::Value res = op->result(0);

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

  symbol::ShapeOrDataDimExprs shape_or_data{symbol::TensorShapeOrDataDimExprs(
      sym_shape, operand_shape_or_data.shape())};

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
  std::vector<symbol::DimExpr> out_dims_data;
  if (operand_shape_or_data.data().has_value()) {
    out_dims_data = operand_shape_or_data.data().value();
    out_dims.emplace_back(
        static_cast<std::int64_t>(operand_shape_or_data.shape().size()));
  }
  // else : pir::VectorType x =
  // operand_source.type().dyn_cast<pir::VectorType>();
  // TODO(zhangbopd): else branch is not implemented yet.
  symbol::ShapeOrDataDimExprs shape_data(
      symbol::TensorShapeOrDataDimExprs(out_dims, out_dims_data));

  op->set_attribute(
      "symbolic_shape",
      pir::shape::SymbolAttribute::get(pir::IrContext::Instance(), shape_data));
  pir::Value res = op->result(0);
  shape_analysis->SetShapeOrDataForValue(res, shape_data);
  return true;
}

bool ReduceInferDim(pir::Operation *op,
                    pir::ShapeConstraintIRAnalysis *shape_analysis,
                    const std::vector<int64_t> &axis,
                    bool keep_dim,
                    bool reduce_all) {
  auto x = op->operand_source(0);
  std::vector<int64_t> x_dims =
      common::vectorize(x.type().dyn_cast<pir::DenseTensorType>().dims());
  int x_rank = x_dims.size();

  std::vector<int64_t> formated_axis = axis;
  for (size_t i = 0; i < axis.size(); ++i) {
    // we always assume the value in axis are valid, since it has been checked
    // in Op's InferMeta
    if (axis[i] < 0) {
      formated_axis[i] = axis[i] + x_rank;
    }
  }

  bool full_dim = true;
  std::set<int64_t> dims_set(formated_axis.begin(), formated_axis.end());
  for (int64_t i = 0; i < x_rank; ++i) {
    if (dims_set.find(i) == dims_set.end()) {
      full_dim = false;
      break;
    }
  }
  bool empty_dim = axis.size() == 0;
  reduce_all = reduce_all || full_dim || empty_dim;

  symbol::ShapeOrDataDimExprs x_shape_or_data =
      shape_analysis->GetShapeOrDataForValue(x);
  std::vector<symbol::DimExpr> input_shapes;
  if (x_shape_or_data.data() == std::nullopt ||
      x_shape_or_data.data()->size() == 0) {
    input_shapes = x_shape_or_data.shape();
  } else {
    input_shapes = *x_shape_or_data.data();
  }

  std::vector<symbol::DimExpr> shapes;
  for (int i = 0; i < x_rank; ++i) {
    if (reduce_all || dims_set.find(i) != dims_set.end()) {
      if (keep_dim) {
        shapes.push_back(1);
      } else {
        continue;
      }
    } else {
      shapes.push_back(input_shapes.at(i));
    }
  }
  pir::Value res = op->result(0);
  symbol::ShapeOrDataDimExprs shape_data{
      symbol::TensorShapeOrDataDimExprs(shapes)};
  op->set_attribute(
      "symbolic_shape",
      pir::shape::SymbolAttribute::get(pir::IrContext::Instance(), shape_data));
  shape_analysis->SetShapeOrDataForValue(res, shape_data);
  return true;
}

bool SumOpInferSymbolicShape(pir::Operation *op,
                             pir::ShapeConstraintIRAnalysis *shape_analysis) {
  VLOG(1) << "SumOpInferSymbolicShape begin";

  auto attributes = op->attributes();
  bool keepdim = attributes["keepdim"].dyn_cast<pir::BoolAttribute>().data();

  bool reduce_all = false;

  auto axis_gen_op = op->operand_source(1).defining_op();
  if (axis_gen_op->isa<paddle::dialect::FullIntArrayOp>()) {
    std::vector<int64_t> axis = GetVectorAttr(
        axis_gen_op->dyn_cast<paddle::dialect::FullIntArrayOp>(), "value");
    if (axis.size() == 0) {
      reduce_all = true;
    }
    return ReduceInferDim(op, shape_analysis, axis, keepdim, reduce_all);
  } else {
    // TODO(lanxianghit): deal with other source: pir::VectorType,
    // paddle::dialect::DenseTensorType
    PADDLE_THROW(
        phi::errors::Unimplemented("SumOpInferSymbolicShape: 'axis' only "
                                   "support FullIntArrayOp's result now."));
  }

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

  symbol::ShapeOrDataDimExprs shape_data{
      symbol::TensorShapeOrDataDimExprs(out_dims)};
  op->set_attribute(
      "symbolic_shape",
      pir::shape::SymbolAttribute::get(pir::IrContext::Instance(), shape_data));

  pir::Value res0 = op->result(0);
  pir::Value res1 = op->result(1);
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
  pir::Attribute attr_value = attributes["value"];
  const auto &vec = attr_value.dyn_cast<pir::ArrayAttribute>().AsVector();

  std::vector<symbol::DimExpr> data;
  for (auto item : vec) {
    int64_t i = item.dyn_cast<pir::Int64Attribute>().data();
    data.push_back(symbol::DimExpr(i));
  }

  // TODO(zhangbopd): use op->result(0) to infer the shape
  std::vector<symbol::DimExpr> shape{std::int64_t(vec.size())};

  symbol::ShapeOrDataDimExprs shape_data{
      symbol::TensorShapeOrDataDimExprs(shape, data)};

  op->set_attribute(
      "symbolic_shape",
      pir::shape::SymbolAttribute::get(pir::IrContext::Instance(), shape_data));

  pir::Value res = op->result(0);
  shape_analysis->SetShapeOrDataForValue(res, shape_data);
  return true;
}

bool SliceOpInferSymbolicShape(pir::Operation *op,
                               pir::ShapeConstraintIRAnalysis *shape_analysis) {
  // TODO(zhangbopd): Not implemented yet.
  pir::Value operand_source = op->operand_source(0);
  pir::Value operand_starts = op->operand_source(1);
  pir::Value operand_ends = op->operand_source(2);
  pir::Value res = op->result(0);

  symbol::ShapeOrDataDimExprs operand_shape_or_data =
      shape_analysis->GetShapeOrDataForValue(operand_source);
  symbol::ShapeOrDataDimExprs starts_shape_data =
      shape_analysis->GetShapeOrDataForValue(operand_starts);
  symbol::ShapeOrDataDimExprs ends_shape_data =
      shape_analysis->GetShapeOrDataForValue(operand_ends);

  int64_t start = 0;
  if (starts_shape_data.data().has_value()) {
    start = starts_shape_data.data()->at(0).Get<int64_t>();
  }

  int64_t end = 0;
  if (ends_shape_data.data().has_value()) {
    end = ends_shape_data.data()->at(0).Get<int64_t>();
  }

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

  std::vector<symbol::DimExpr> out_data;
  if (operand_shape_or_data.data().has_value()) {
    for (int64_t i = start; i < end; i++) {
      out_data.push_back(operand_shape_or_data.data().value()[i]);
    }
  }

  symbol::ShapeOrDataDimExprs shape_data{
      symbol::TensorShapeOrDataDimExprs(sym_shape, out_data)};

  op->set_attribute(
      "symbolic_shape",
      pir::shape::SymbolAttribute::get(pir::IrContext::Instance(), shape_data));

  shape_analysis->SetShapeOrDataForValue(res, shape_data);
  return true;
}

bool FullOpInferSymbolicShape(pir::Operation *op,
                              pir::ShapeConstraintIRAnalysis *shape_analysis) {
  auto attributes = op->attributes();
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

  // DimExpr only keep shape info, which is always int type
  int64_t value = attributes.at("value")
                      .dyn_cast<paddle::dialect::ScalarAttribute>()
                      .data()
                      .to<int64_t>();
  std::vector<symbol::DimExpr> sym_data;
  sym_data.emplace_back(value);

  symbol::ShapeOrDataDimExprs shape_data{
      symbol::TensorShapeOrDataDimExprs(sym_shape)};

  op->set_attribute(
      "symbolic_shape",
      pir::shape::SymbolAttribute::get(pir::IrContext::Instance(), shape_data));

  pir::Value res = op->result(0);
  shape_analysis->SetShapeOrDataForValue(res, shape_data);
  return true;
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

bool ConcatOpInferSymbolicShape(
    pir::Operation *op, pir::ShapeConstraintIRAnalysis *shape_analysis) {
  return true;
}

bool GatherNdOpInferSymbolicShape(
    pir::Operation *op, pir::ShapeConstraintIRAnalysis *shape_analysis) {
  return true;
}

bool PowOpInferSymbolicShape(pir::Operation *op,
                             pir::ShapeConstraintIRAnalysis *shape_analysis) {
  return true;
}
bool Pow_OpInferSymbolicShape(pir::Operation *op,
                              pir::ShapeConstraintIRAnalysis *shape_analysis) {
  return PowOpInferSymbolicShape(op, shape_analysis);
}

bool RsqrtOpInferSymbolicShape(pir::Operation *op,
                               pir::ShapeConstraintIRAnalysis *shape_analysis) {
  return true;
}
bool Rsqrt_OpInferSymbolicShape(
    pir::Operation *op, pir::ShapeConstraintIRAnalysis *shape_analysis) {
  return RsqrtOpInferSymbolicShape(op, shape_analysis);
}

bool ScaleOpInferSymbolicShape(pir::Operation *op,
                               pir::ShapeConstraintIRAnalysis *shape_analysis) {
  return SameOperandsAndResultShape(op, shape_analysis);
}
bool Scale_OpInferSymbolicShape(
    pir::Operation *op, pir::ShapeConstraintIRAnalysis *shape_analysis) {
  return SameOperandsAndResultShape(op, shape_analysis);
}
bool ScaleSrOpInferSymbolicShape(
    pir::Operation *op, pir::ShapeConstraintIRAnalysis *shape_analysis) {
  return SameOperandsAndResultShape(op, shape_analysis);
}
bool ScaleSr_OpInferSymbolicShape(
    pir::Operation *op, pir::ShapeConstraintIRAnalysis *shape_analysis) {
  return SameOperandsAndResultShape(op, shape_analysis);
}

bool SqueezeOpInferSymbolicShape(
    pir::Operation *op, pir::ShapeConstraintIRAnalysis *shape_analysis) {
  return true;
}
bool Squeeze_OpInferSymbolicShape(
    pir::Operation *op, pir::ShapeConstraintIRAnalysis *shape_analysis) {
  return SqueezeOpInferSymbolicShape(op, shape_analysis);
}

bool UnsqueezeOpInferSymbolicShape(
    pir::Operation *op, pir::ShapeConstraintIRAnalysis *shape_analysis) {
  return true;
}
bool Unsqueeze_OpInferSymbolicShape(
    pir::Operation *op, pir::ShapeConstraintIRAnalysis *shape_analysis) {
  return UnsqueezeOpInferSymbolicShape(op, shape_analysis);
}

bool TileOpInferSymbolicShape(pir::Operation *op,
                              pir::ShapeConstraintIRAnalysis *shape_analysis) {
  pir::Value operand_x = op->operand_source(0);
  symbol::ShapeOrDataDimExprs x_shape_or_data =
      shape_analysis->GetShapeOrDataForValue(operand_x);
  pir::Value operand_repeat_times = op->operand_source(1);
  symbol::ShapeOrDataDimExprs repeat_times_shape_or_data =
      shape_analysis->GetShapeOrDataForValue(operand_repeat_times);

  std::vector<symbol::DimExpr> x_dimexpr;
  if (x_shape_or_data.data().has_value()) {
    x_dimexpr = x_shape_or_data.data().value();
  } else {
    x_dimexpr = x_shape_or_data.shape();
  }

  std::vector<symbol::DimExpr> repeat_times_dimexpr;
  if (repeat_times_shape_or_data.data().has_value()) {
    repeat_times_dimexpr = repeat_times_shape_or_data.data().value();
  } else {
    repeat_times_dimexpr = repeat_times_shape_or_data.shape();
  }
  if (repeat_times_dimexpr.empty()) {
    repeat_times_dimexpr = std::vector<symbol::DimExpr>(x_dimexpr.size(), 1);
  }

  auto out_rank = std::max(static_cast<size_t>(x_dimexpr.size()),
                           repeat_times_dimexpr.size());
  std::vector<symbol::DimExpr> out_shape(out_rank);
  if (x_dimexpr.size() > repeat_times_dimexpr.size()) {
    auto diff = x_dimexpr.size() - repeat_times_dimexpr.size();
    repeat_times_dimexpr.insert(repeat_times_dimexpr.begin(), diff, 1);
  } else {
    auto diff = repeat_times_dimexpr.size() - x_dimexpr.size();
    x_dimexpr.insert(x_dimexpr.begin(), diff, 1);
  }

  for (size_t i = 0; i < repeat_times_dimexpr.size(); ++i) {
    out_shape[i] = x_dimexpr[i] * repeat_times_dimexpr[i];
  }

  symbol::ShapeOrDataDimExprs shape_data{
      symbol::TensorShapeOrDataDimExprs(out_shape)};

  op->set_attribute(
      "symbolic_shape",
      pir::shape::SymbolAttribute::get(pir::IrContext::Instance(), shape_data));

  pir::Value res = op->result(0);
  shape_analysis->SetShapeOrDataForValue(res, shape_data);

  return true;
}

bool TransposeOpInferSymbolicShape(
    pir::Operation *op, pir::ShapeConstraintIRAnalysis *shape_analysis) {
  return true;
}
bool Transpose_OpInferSymbolicShape(
    pir::Operation *op, pir::ShapeConstraintIRAnalysis *shape_analysis) {
  return TransposeOpInferSymbolicShape(op, shape_analysis);
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

bool FullWithTensorOpInferSymbolicShape(
    pir::Operation *op, pir::ShapeConstraintIRAnalysis *shape_analysis) {
  return SameOperandsAndResultShape(op, shape_analysis);
}

bool ReluOpInferSymbolicShape(pir::Operation *op,
                              pir::ShapeConstraintIRAnalysis *shape_analysis) {
  return SameOperandsAndResultShape(op, shape_analysis);
}

bool Relu_OpInferSymbolicShape(pir::Operation *op,
                               pir::ShapeConstraintIRAnalysis *shape_analysis) {
  return SameOperandsAndResultShape(op, shape_analysis);
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

  symbol::ShapeOrDataDimExprs shape_data{
      symbol::TensorShapeOrDataDimExprs(out_dims)};
  if (operand_shape_or_data.data().has_value()) {
    shape_data.SetData(operand_shape_or_data.shape());
  }
  op->set_attribute(
      "symbolic_shape",
      pir::shape::SymbolAttribute::get(pir::IrContext::Instance(), shape_data));

  pir::Value res = op->result(0);
  shape_analysis->SetShapeOrDataForValue(res, shape_data);
  return true;
}

bool ScaleOpInferSymbolicShape(pir::Operation *op,
                               pir::ShapeConstraintIRAnalysis *shape_analysis) {
  return true;
}

}  // namespace cinn::dialect

IR_DEFINE_EXPLICIT_TYPE_ID(paddle::dialect::InferSymbolicShapeInterface)
