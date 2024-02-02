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
  const auto &attr_map = op->attributes();
  PADDLE_ENFORCE(
      attr_map.count(name),
      phi::errors::PreconditionNotMet(
          "attr [%s] MUST in attribute map for [%s] op", name, op->name()));
  const auto &val = attr_map.at(name);

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
  // For ElementWiseBinary ops, if the input tensor is from full op, the value
  // of fullop is useless, only the shape need doing broadcast
  bool x_from_fullop =
      op->operand_source(0).defining_op()->isa<paddle::dialect::FullOp>();
  if (!x_from_fullop && x_shapeordata.data().has_value()) {
    shape_0 = x_shapeordata.data().value();
  } else {
    shape_0 = x_shapeordata.shape();
  }

  auto y_shapeordata =
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
  const auto &attributes = op->attributes();
  pir::Attribute attr = attributes.at("shape");
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

  const auto &attributes = op->attributes();
  int axis = attributes.at("axis").dyn_cast<pir::Int32Attribute>().data();

  const symbol::TensorListShapeOrDataDimExprs &shape_data_list =
      shape_analysis->GetShapeOrDataForValue(operand_source)
          .dyn_cast<symbol::TensorListShapeOrDataDimExprs>();

  // TODO(zhangbopd): Only For 0-Dim tensor case, and else branch is not
  // implemented yet.
  const auto &GetDimTensorExprs = [&]() -> std::vector<symbol::DimExpr> {
    std::vector<symbol::DimExpr> dim_exprs;
    for (size_t i = 0; i < shape_data_list.size(); ++i) {
      if (shape_data_list[i].data().has_value() && axis == 0) {
        dim_exprs.emplace_back(shape_data_list[i].data().value()[0]);
      } else {
        PADDLE_THROW(phi::errors::Unimplemented(
            "StackOpInferSymbolicShape is now only support 0-Dim tensor."));
      }
    }
    return dim_exprs;
  };
  const std::vector<symbol::DimExpr> out_dims{
      static_cast<std::int64_t>(shape_data_list.size())};
  const std::vector<symbol::DimExpr> out_dims_data = GetDimTensorExprs();
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

  const symbol::ShapeOrDataDimExprs &x_shape_or_data =
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

  const auto &attributes = op->attributes();
  bool keepdim = attributes.at("keepdim").dyn_cast<pir::BoolAttribute>().data();

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

bool ProdOpInferSymbolicShape(pir::Operation *op,
                              pir::ShapeConstraintIRAnalysis *shape_analysis) {
  auto attributes = op->attributes();
  bool keepdim = attributes["keep_dim"].dyn_cast<pir::BoolAttribute>().data();

  bool reduce_all =
      attributes["reduce_all"].dyn_cast<pir::BoolAttribute>().data();

  auto axis_gen_op = op->operand_source(1).defining_op();
  if (axis_gen_op->isa<paddle::dialect::FullIntArrayOp>()) {
    std::vector<int64_t> axis = GetVectorAttr(
        axis_gen_op->dyn_cast<paddle::dialect::FullIntArrayOp>(), "value");
    return ReduceInferDim(op, shape_analysis, axis, keepdim, reduce_all);
  } else {
    // TODO(lanxianghit): deal with other source: pir::VectorType,
    // paddle::dialect::DenseTensorType
    PADDLE_THROW(
        phi::errors::Unimplemented("ProdOpInferSymbolicShape: 'axis' only "
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
  const auto &attributes = op->attributes();
  pir::Attribute attr_value = attributes.at("value");
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
  const auto &attributes = op->attributes();
  pir::Attribute attr_shape = attributes.at("shape");
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
  shape_data.SetData(sym_data);

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
  pir::Value operand_source = op->operand_source(0);
  // int axis = op->operand_source(1).type().dyn_cast<pir::Int32Type>();

  const auto &shape_data_list =
      shape_analysis->GetShapeOrDataForValue(operand_source)
          .dyn_cast<symbol::TensorListShapeOrDataDimExprs>();

  const std::vector<symbol::DimExpr> &out_dims = [&] {
    std::vector<symbol::DimExpr> out_dims = shape_data_list[0].shape();
    size_t axis = static_cast<size_t>(shape_data_list[0].shape().size() - 1);
    // TODO(zhangbopd): Dim size of non-axis dims may be infered out.
    for (size_t i = 0; i < shape_data_list.size(); ++i) {
      if (i != axis) continue;
      out_dims[axis] = out_dims[axis] + shape_data_list[i].shape()[axis];
    }
    return out_dims;
  }();

  symbol::ShapeOrDataDimExprs shape_data{
      symbol::TensorShapeOrDataDimExprs(out_dims)};

  op->set_attribute(
      "symbolic_shape",
      pir::shape::SymbolAttribute::get(pir::IrContext::Instance(), shape_data));

  pir::Value res = op->result(0);
  shape_analysis->SetShapeOrDataForValue(res, shape_data);

  return true;
}

bool GatherNdOpInferSymbolicShape(
    pir::Operation *op, pir::ShapeConstraintIRAnalysis *shape_analysis) {
  auto x_shape_or_data =
      shape_analysis->GetShapeOrDataForValue(op->operand_source(0));
  auto index_shape_or_data =
      shape_analysis->GetShapeOrDataForValue(op->operand_source(1));

  std::vector<symbol::DimExpr> x_sym_shape;
  if (x_shape_or_data.data().has_value()) {
    x_sym_shape = x_shape_or_data.data().value();
  } else {
    x_sym_shape = x_shape_or_data.shape();
  }
  int x_dims_size = x_sym_shape.size();

  std::vector<symbol::DimExpr> index_sym_shape;
  if (index_shape_or_data.data().has_value()) {
    index_sym_shape = index_shape_or_data.data().value();
  } else {
    index_sym_shape = index_shape_or_data.shape();
  }
  int index_dims_size = index_sym_shape.size();

  std::vector<symbol::DimExpr> result_sym_dims;
  // The result dims is
  //   Index.shape[:-1] + X.shape[Index.shape[-1]:]
  for (int i = 0; i < index_dims_size - 1; ++i) {
    result_sym_dims.emplace_back(index_sym_shape[i]);
  }

  PADDLE_ENFORCE_EQ(
      index_sym_shape[index_dims_size - 1].Has<std::int64_t>(),
      true,
      phi::errors::InvalidArgument(
          "in GatherNdOpInferSymbolicShape: index[-1] should be unknown"));

  for (int i = static_cast<int>(
           index_sym_shape[index_dims_size - 1].Get<std::int64_t>());
       i < x_dims_size;
       ++i) {
    result_sym_dims.emplace_back(x_sym_shape[i]);
  }

  symbol::ShapeOrDataDimExprs shape_data{
      symbol::TensorShapeOrDataDimExprs(result_sym_dims)};

  op->set_attribute(
      "symbolic_shape",
      pir::shape::SymbolAttribute::get(pir::IrContext::Instance(), shape_data));

  pir::Value res = op->result(0);
  shape_analysis->SetShapeOrDataForValue(res, shape_data);

  return true;
}

bool PowOpInferSymbolicShape(pir::Operation *op,
                             pir::ShapeConstraintIRAnalysis *shape_analysis) {
  return SameOperandsAndResultShape(op, shape_analysis);
}
bool Pow_OpInferSymbolicShape(pir::Operation *op,
                              pir::ShapeConstraintIRAnalysis *shape_analysis) {
  PADDLE_THROW(phi::errors::Unimplemented(
      op->name() + " DOES NOT have InferSymbolicShapeInterface!"));
  return PowOpInferSymbolicShape(op, shape_analysis);
}

bool RsqrtOpInferSymbolicShape(pir::Operation *op,
                               pir::ShapeConstraintIRAnalysis *shape_analysis) {
  return SameOperandsAndResultShape(op, shape_analysis);
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
  IR_ENFORCE(op->num_operands() == 2,
             "SqueezeOpInferSymbolicShape ONLY support num_operands() == 2 "
             "now, but got %d operands",
             op->num_operands());

  auto x_shape_or_data =
      shape_analysis->GetShapeOrDataForValue(op->operand_source(0));
  auto axes_shape_or_data =
      shape_analysis->GetShapeOrDataForValue(op->operand_source(1));

  std::vector<symbol::DimExpr> in_dims_sym;
  if (x_shape_or_data.data().has_value()) {
    in_dims_sym = x_shape_or_data.data().value();
  } else {
    in_dims_sym = x_shape_or_data.shape();
  }

  std::vector<symbol::DimExpr> squeeze_dims_sym;
  if (axes_shape_or_data.data().has_value()) {
    squeeze_dims_sym = axes_shape_or_data.data().value();
  } else {
    squeeze_dims_sym = axes_shape_or_data.shape();
  }

  std::vector<int> squeeze_dims;
  for (auto squeeze_dim : squeeze_dims_sym) {
    IR_ENFORCE(squeeze_dim.Has<std::int64_t>(),
               "in SqueezeOpInferSymbolicShape, axes must be known int type, "
               "but got: %s",
               symbol::ToString(squeeze_dim));
    squeeze_dims.emplace_back(
        static_cast<int>(squeeze_dim.Get<std::int64_t>()));
  }

  // GetOutputSqueezeShape
  size_t num_squeeze_dims = squeeze_dims.size();
  std::vector<bool> should_squeeze(in_dims_sym.size(), false);
  // Mark dimensions need to be squeezed.
  if (num_squeeze_dims == 0) {
    for (size_t i = 0; i < in_dims_sym.size(); ++i) {
      // TODO(lanxianghit): if symbol here, maybe we need the result of dim expr
      // simplification
      if (in_dims_sym[i] == 1) {
        should_squeeze[i] = true;
      }
    }
  } else {
    for (size_t i = 0; i < num_squeeze_dims; ++i) {
      if (in_dims_sym.size() == 0) {
        continue;
      }
      int current = squeeze_dims[i] < 0 ? squeeze_dims[i] + in_dims_sym.size()
                                        : squeeze_dims[i];

      if (!should_squeeze[current]) {
        // At compile time, dim of SYMBOL is allowed to squeeze?
        if (in_dims_sym[current] == 1) {
          should_squeeze[current] = true;
        } else if (!in_dims_sym[current].Has<std::int64_t>()) {
          PADDLE_THROW(
              phi::errors::Unimplemented("SqueezeOpInferSymbolicShape CAN NOT "
                                         "deal with symbol in axis now"));
        }
      }
    }
  }

  // Make output dimensions
  std::vector<symbol::DimExpr> output_shape_sym;
  for (size_t i = 0; i < in_dims_sym.size(); ++i) {
    if (!should_squeeze[i]) {
      output_shape_sym.emplace_back(in_dims_sym[i]);
    }
  }

  symbol::ShapeOrDataDimExprs shape_data{
      symbol::TensorShapeOrDataDimExprs(output_shape_sym)};

  op->set_attribute(
      "symbolic_shape",
      pir::shape::SymbolAttribute::get(pir::IrContext::Instance(), shape_data));

  pir::Value res = op->result(0);
  shape_analysis->SetShapeOrDataForValue(res, shape_data);

  return true;
}
bool Squeeze_OpInferSymbolicShape(
    pir::Operation *op, pir::ShapeConstraintIRAnalysis *shape_analysis) {
  return SqueezeOpInferSymbolicShape(op, shape_analysis);
}

bool UnsqueezeOpInferSymbolicShape(
    pir::Operation *op, pir::ShapeConstraintIRAnalysis *shape_analysis) {
  IR_ENFORCE(op->num_operands() == 2,
             "UnsqueezeOpInferSymbolicShape ONLY support num_operands() == 2 "
             "now, but got %d operands",
             op->num_operands());

  auto x_shape_or_data =
      shape_analysis->GetShapeOrDataForValue(op->operand_source(0));
  auto axes_shape_or_data =
      shape_analysis->GetShapeOrDataForValue(op->operand_source(1));

  std::vector<symbol::DimExpr> x_sym_shape;
  if (x_shape_or_data.data().has_value()) {
    x_sym_shape = x_shape_or_data.data().value();
  } else {
    x_sym_shape = x_shape_or_data.shape();
  }
  int x_dims_size = x_sym_shape.size();

  std::vector<symbol::DimExpr> axes_sym;
  if (axes_shape_or_data.data().has_value()) {
    axes_sym = axes_shape_or_data.data().value();
  } else {
    axes_sym = axes_shape_or_data.shape();
  }
  int axes_sym_size = axes_sym.size();

  // GetUnsqueezeShape
  int output_rank = x_dims_size + axes_sym_size;
  std::vector<symbol::DimExpr> result_sym_dims(output_rank, 0);

  int cur_output_rank = x_dims_size;
  for (auto axis_expr : axes_sym) {
    IR_ENFORCE(axis_expr.Has<std::int64_t>(),
               "in UnsqueezeOpInferSymbolicShape, axes must be known int type, "
               "but got: %s",
               symbol::ToString(axis_expr));
    int axis = static_cast<int>(axis_expr.Get<std::int64_t>());
    int cur = axis < 0 ? axis + cur_output_rank + 1 : axis;
    // No need to do vaildity check here, it's already done in InferMeta

    // Move old axis, and insert new axis
    for (int i = cur_output_rank; i >= cur; --i) {
      if (result_sym_dims[i] == 1) {
        // Move axis
        result_sym_dims[i + 1] = 1;
        result_sym_dims[i] = 0;
      }
    }
    result_sym_dims[cur] = 1;
    // Add the output size.
    cur_output_rank++;
  }

  // Make output shape
  for (int in_idx = 0, out_idx = 0; out_idx < output_rank; ++out_idx) {
    if (result_sym_dims[out_idx] == 0) {
      result_sym_dims[out_idx] = x_sym_shape[in_idx++];
    }
  }

  symbol::ShapeOrDataDimExprs shape_data{
      symbol::TensorShapeOrDataDimExprs(result_sym_dims)};

  op->set_attribute(
      "symbolic_shape",
      pir::shape::SymbolAttribute::get(pir::IrContext::Instance(), shape_data));

  pir::Value res = op->result(0);
  shape_analysis->SetShapeOrDataForValue(res, shape_data);

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
  PADDLE_THROW(phi::errors::Unimplemented(
      op->name() + " DOES NOT have InferSymbolicShapeInterface!"));
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

bool ArangeOpInferSymbolicShape(
    pir::Operation *op, pir::ShapeConstraintIRAnalysis *shape_analysis) {
  PADDLE_THROW(phi::errors::Unimplemented(
      op->name() + " DOES NOT have InferSymbolicShapeInterface!"));
  return true;
}

bool EmbeddingOpInferSymbolicShape(
    pir::Operation *op, pir::ShapeConstraintIRAnalysis *shape_analysis) {
  const auto x_shape_or_data =
      shape_analysis->GetShapeOrDataForValue(op->operand_source(0));
  const auto weight_shape_or_data =
      shape_analysis->GetShapeOrDataForValue(op->operand_source(1));
  const std::vector<symbol::DimExpr> &x_dims = [&] {
    std::vector<symbol::DimExpr> dims;
    if (x_shape_or_data.data().has_value()) {
      dims = x_shape_or_data.data().value();
    } else {
      dims = x_shape_or_data.shape();
    }
    return dims;
  }();

  const std::vector<symbol::DimExpr> &weight_dims = [&] {
    std::vector<symbol::DimExpr> dims;
    if (weight_shape_or_data.data().has_value()) {
      dims = weight_shape_or_data.data().value();
    } else {
      dims = weight_shape_or_data.shape();
    }
    return dims;
  }();

  const symbol::ShapeOrDataDimExprs &shape_data = [&] {
    std::vector<symbol::DimExpr> out_dims = x_dims;
    // no need to check validation of weight_dims index, since all checks have
    // been done at corresponding InferMeta
    out_dims.emplace_back(weight_dims[1]);
    return symbol::ShapeOrDataDimExprs{
        symbol::TensorShapeOrDataDimExprs(out_dims)};
  }();

  shape_analysis->SetShapeOrDataForValue(op->result(0), shape_data);

  return true;
}

bool SparseWeightEmbeddingOpInferSymbolicShape(
    pir::Operation *op, pir::ShapeConstraintIRAnalysis *shape_analysis) {
  PADDLE_THROW(phi::errors::Unimplemented(
      op->name() + " DOES NOT have InferSymbolicShapeInterface!"));
  return true;
}

bool ExpandOpInferSymbolicShape(
    pir::Operation *op, pir::ShapeConstraintIRAnalysis *shape_analysis) {
  PADDLE_THROW(phi::errors::Unimplemented(
      op->name() + " DOES NOT have InferSymbolicShapeInterface!"));
  return true;
}

bool MatmulOpInferSymbolicShape(
    pir::Operation *op, pir::ShapeConstraintIRAnalysis *shape_analysis) {
  PADDLE_THROW(phi::errors::Unimplemented(
      op->name() + " DOES NOT have InferSymbolicShapeInterface!"));
  return true;
}

bool MaxOpInferSymbolicShape(pir::Operation *op,
                             pir::ShapeConstraintIRAnalysis *shape_analysis) {
  PADDLE_THROW(phi::errors::Unimplemented(
      op->name() + " DOES NOT have InferSymbolicShapeInterface!"));
  return true;
}

bool TrilOpInferSymbolicShape(pir::Operation *op,
                              pir::ShapeConstraintIRAnalysis *shape_analysis) {
  PADDLE_THROW(phi::errors::Unimplemented(
      op->name() + " DOES NOT have InferSymbolicShapeInterface!"));
  return true;
}

bool Tril_OpInferSymbolicShape(pir::Operation *op,
                               pir::ShapeConstraintIRAnalysis *shape_analysis) {
  return TrilOpInferSymbolicShape(op, shape_analysis);
}

bool WhereOpInferSymbolicShape(pir::Operation *op,
                               pir::ShapeConstraintIRAnalysis *shape_analysis) {
  PADDLE_THROW(phi::errors::Unimplemented(
      op->name() + " DOES NOT have InferSymbolicShapeInterface!"));
  return true;
}

bool Where_OpInferSymbolicShape(
    pir::Operation *op, pir::ShapeConstraintIRAnalysis *shape_analysis) {
  return WhereOpInferSymbolicShape(op, shape_analysis);
}
}  // namespace paddle::dialect
namespace cinn::dialect {

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
    out_sym_shape[axis] = end - start;
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

  op->set_attribute(
      "symbolic_shape",
      pir::shape::SymbolAttribute::get(pir::IrContext::Instance(), shape_data));

  shape_analysis->SetShapeOrDataForValue(op->result(0), shape_data);
  return true;
}

bool ConcatOpInferSymbolicShape(
    pir::Operation *op, pir::ShapeConstraintIRAnalysis *shape_analysis) {
  const auto input_values = op->operands_source();
  const auto input_size = input_values.size();
  PADDLE_ENFORCE_GT(
      input_size,
      0,
      phi::errors::PreconditionNotMet(
          " [%s] op must have at least one input, but received %d.",
          op->name(),
          input_size));

  const int axis =
      op->attributes().at("axis").dyn_cast<pir::Int32Attribute>().data();
  PADDLE_ENFORCE_GT(axis,
                    0,
                    phi::errors::PreconditionNotMet(
                        "[%s] op shalle satisfy axis > 0 , but received %d.",
                        op->name(),
                        axis));
  PADDLE_ENFORCE_GT(
      input_size,
      0,
      phi::errors::PreconditionNotMet(
          " [%s] op must have at least one input, but received %d.",
          op->name(),
          input_size));
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

  op->set_attribute(
      "symbolic_shape",
      pir::shape::SymbolAttribute::get(pir::IrContext::Instance(), shape_data));

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
  auto axis = GetVectorAttr(op, "dim");
  bool reduce_all = axis.size() == 0 ? true : false;
  return paddle::dialect::ReduceInferDim(
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

}  // namespace cinn::dialect

IR_DEFINE_EXPLICIT_TYPE_ID(paddle::dialect::InferSymbolicShapeInterface)
