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

#include "paddle/fluid/pir/dialect/operator/interface/infer_symbolic_shape/unary_infer_sym.h"
#include "paddle/fluid/pir/dialect/operator/interface/infer_symbolic_shape/infer_sym_slice_utils.h"
#include "paddle/fluid/pir/dialect/operator/interface/infer_symbolic_shape/infer_sym_utils.h"

namespace paddle::dialect {

bool ArgmaxOpInferSymbolicShape(
    pir::Operation *op, pir::ShapeConstraintIRAnalysis *shape_analysis) {
  bool flatten = GetBoolAttr(op, "flatten");
  bool keepdims = GetBoolAttr(op, "keepdims");

  const auto &input_shape_or_data =
      shape_analysis->GetShapeOrDataForValue(op->operand_source(0));

  const auto &axis_shape_or_data =
      shape_analysis->GetShapeOrDataForValue(op->operand_source(1));
  int axis =
      static_cast<int>(axis_shape_or_data.data().value()[0].Get<int64_t>());

  const std::vector<symbol::DimExpr> &input_sym_shape =
      input_shape_or_data.data().has_value()
          ? input_shape_or_data.data().value()
          : input_shape_or_data.shape();

  int rank = input_sym_shape.size();
  if (axis < 0) axis += rank;

  const auto &out_sym_shape = [&] {
    std::vector<symbol::DimExpr> out_sym_shape;
    if (flatten) {
      if (keepdims) {
        out_sym_shape.emplace_back(std::int64_t(rank));
      } else {
        out_sym_shape.emplace_back(std::int64_t(0));
      }
    } else {
      for (int i = 0; i < axis; i++) {
        out_sym_shape.emplace_back(input_sym_shape[i]);
      }
      if (keepdims) {
        out_sym_shape.emplace_back(std::int64_t(1));
      }

      for (int i = axis + 1; i < rank; i++) {
        out_sym_shape.emplace_back(input_sym_shape[i]);
      }
    }
    return out_sym_shape;
  }();

  symbol::ShapeOrDataDimExprs shape_data{
      symbol::TensorShapeOrDataDimExprs(out_sym_shape)};

  shape_analysis->SetShapeOrDataForValue(op->result(0), shape_data);
  return true;
}

bool ArgminOpInferSymbolicShape(
    pir::Operation *op, pir::ShapeConstraintIRAnalysis *shape_analysis) {
  return ArgmaxOpInferSymbolicShape(op, shape_analysis);
}

bool AsComplexOpInferSymbolicShape(
    pir::Operation *op, pir::ShapeConstraintIRAnalysis *shape_analysis) {
  pir::Value operand_source = op->operand_source(0);
  const symbol::ShapeOrDataDimExprs &operand_shape_or_data =
      shape_analysis->GetShapeOrDataForValue(operand_source);

  const std::vector<symbol::DimExpr> out_dims = [&] {
    std::vector<symbol::DimExpr> out_dims = operand_shape_or_data.shape();
    out_dims.pop_back();
    return out_dims;
  }();

  symbol::ShapeOrDataDimExprs shape_data{
      symbol::TensorShapeOrDataDimExprs(out_dims)};

  shape_analysis->SetShapeOrDataForValue(op->result(0), shape_data);
  return true;
}
bool AsRealOpInferSymbolicShape(
    pir::Operation *op, pir::ShapeConstraintIRAnalysis *shape_analysis) {
  pir::Value operand_source = op->operand_source(0);
  const symbol::ShapeOrDataDimExprs &operand_shape_or_data =
      shape_analysis->GetShapeOrDataForValue(operand_source);

  const std::vector<symbol::DimExpr> out_dims = [&] {
    std::vector<symbol::DimExpr> out_dims = operand_shape_or_data.shape();
    out_dims.push_back(symbol::DimExpr(2));
    return out_dims;
  }();

  symbol::ShapeOrDataDimExprs shape_data{
      symbol::TensorShapeOrDataDimExprs(out_dims)};

  shape_analysis->SetShapeOrDataForValue(op->result(0), shape_data);
  return true;
}

bool CummaxOpInferSymbolicShape(
    pir::Operation *op, pir::ShapeConstraintIRAnalysis *shape_analysis) {
  pir::Value operand_source = op->operand_source(0);
  const symbol::ShapeOrDataDimExprs &operand_shape_or_data =
      shape_analysis->GetShapeOrDataForValue(operand_source);

  shape_analysis->SetShapeOrDataForValue(op->result(0), operand_shape_or_data);
  shape_analysis->SetShapeOrDataForValue(op->result(1), operand_shape_or_data);
  return true;
}
bool CumminOpInferSymbolicShape(
    pir::Operation *op, pir::ShapeConstraintIRAnalysis *shape_analysis) {
  return CummaxOpInferSymbolicShape(op, shape_analysis);
}
bool CumprodOpInferSymbolicShape(
    pir::Operation *op, pir::ShapeConstraintIRAnalysis *shape_analysis) {
  pir::Value operand_source = op->operand_source(0);
  const symbol::ShapeOrDataDimExprs &operand_shape_or_data =
      shape_analysis->GetShapeOrDataForValue(operand_source);
  shape_analysis->SetShapeOrDataForValue(op->result(0), operand_shape_or_data);
  return true;
}
bool Cumprod_OpInferSymbolicShape(
    pir::Operation *op, pir::ShapeConstraintIRAnalysis *shape_analysis) {
  return CumprodOpInferSymbolicShape(op, shape_analysis);
}
bool CumsumOpInferSymbolicShape(
    pir::Operation *op, pir::ShapeConstraintIRAnalysis *shape_analysis) {
  pir::Value operand_source = op->operand_source(0);

  const symbol::ShapeOrDataDimExprs &operand_shape_or_data =
      shape_analysis->GetShapeOrDataForValue(operand_source);

  bool flatten = GetBoolAttr(op, "flatten");
  if (flatten) {
    symbol::DimExpr product{1};
    const auto &dim_exprs = operand_shape_or_data.shape();
    for (const auto &dim_expr : dim_exprs) {
      product = product * dim_expr;
    }
    const std::vector<symbol::DimExpr> out_dims = {product};
    symbol::ShapeOrDataDimExprs shape_data{
        symbol::TensorShapeOrDataDimExprs(out_dims)};
    shape_analysis->SetShapeOrDataForValue(op->result(0), shape_data);

  } else {
    shape_analysis->SetShapeOrDataForValue(op->result(0),
                                           operand_shape_or_data);
  }
  return true;
}
bool Cumsum_OpInferSymbolicShape(
    pir::Operation *op, pir::ShapeConstraintIRAnalysis *shape_analysis) {
  return CumsumOpInferSymbolicShape(op, shape_analysis);
}

bool DiagEmbedOpInferSymbolicShape(
    pir::Operation *op, pir::ShapeConstraintIRAnalysis *shape_analysis) {
  pir::Value operand_source = op->operand_source(0);
  const symbol::ShapeOrDataDimExprs &operand_shape_or_data =
      shape_analysis->GetShapeOrDataForValue(operand_source);
  const auto &attributes = op->attributes();
  int dim1 = attributes.at("dim1").dyn_cast<pir::Int32Attribute>().data();
  int dim2 = attributes.at("dim2").dyn_cast<pir::Int32Attribute>().data();
  int offset = attributes.at("offset").dyn_cast<pir::Int32Attribute>().data();

  const auto &x_dims = operand_shape_or_data.shape();
  int dim1_ = dim1 < 0 ? x_dims.size() + dim1 + 1 : dim1;
  int dim2_ = dim2 < 0 ? x_dims.size() + dim2 + 1 : dim2;
  int64_t offset_ = static_cast<int64_t>(std::abs(offset));
  symbol::DimExpr new_dim_len =
      symbol::DimExpr(offset_) + x_dims[x_dims.size() - 1];

  const auto &out_dims = [&] {
    std::vector<symbol::DimExpr> out_dims = x_dims;
    out_dims.pop_back();
    out_dims.insert(out_dims.begin() + std::min(dim1_, dim2_), new_dim_len);
    out_dims.insert(out_dims.begin() + std::max(dim1_, dim2_), new_dim_len);
    return out_dims;
  }();
  symbol::ShapeOrDataDimExprs shape_data{
      symbol::TensorShapeOrDataDimExprs(out_dims)};
  shape_analysis->SetShapeOrDataForValue(op->result(0), shape_data);
  return true;
}
bool DiagonalOpInferSymbolicShape(
    pir::Operation *op, pir::ShapeConstraintIRAnalysis *shape_analysis) {
  pir::Value operand_source = op->operand_source(0);
  const symbol::ShapeOrDataDimExprs &operand_shape_or_data =
      shape_analysis->GetShapeOrDataForValue(operand_source);
  const auto &attributes = op->attributes();
  int axis1 = attributes.at("axis1").dyn_cast<pir::Int32Attribute>().data();
  int axis2 = attributes.at("axis2").dyn_cast<pir::Int32Attribute>().data();
  int offset = attributes.at("offset").dyn_cast<pir::Int32Attribute>().data();

  const auto &x_dims = operand_shape_or_data.shape();
  int axis1_ = axis1 < 0 ? x_dims.size() + axis1 : axis1;
  int axis2_ = axis2 < 0 ? x_dims.size() + axis2 : axis2;

  auto out_dims = x_dims;
  auto axis1_size = out_dims[axis1_];
  auto axis2_size = out_dims[axis2_];
  out_dims.erase(out_dims.begin() + std::max(axis1_, axis2_));
  out_dims.erase(out_dims.begin() + std::min(axis1_, axis2_));

  symbol::DimExprBuilder builder{nullptr};
  symbol::DimExpr zero{0};
  symbol::DimExpr res_shape;
  symbol::DimExpr offset_sym{offset};
  if (offset == 0) {
    res_shape = builder.Min(axis1_size, axis2_size);
  } else if (offset > 0) {
    if (axis2_size.isa<int64_t>()) {
      res_shape = (axis2_size.dyn_cast<int64_t>() - offset) > 0
                      ? builder.Min(axis1_size, axis2_size - offset_sym)
                      : zero;
    } else {
      res_shape = shape_analysis->GetNextSymName();
    }
  } else {
    if (axis1_size.isa<int64_t>()) {
      res_shape = (axis1_size.dyn_cast<int64_t>() + offset) > 0
                      ? builder.Min(axis1_size + offset_sym, axis2_size)
                      : zero;
    } else {
      res_shape = shape_analysis->GetNextSymName();
    }
  }
  out_dims.push_back(res_shape);

  symbol::ShapeOrDataDimExprs shape_data{
      symbol::TensorShapeOrDataDimExprs(out_dims)};
  shape_analysis->SetShapeOrDataForValue(op->result(0), shape_data);
  return true;
}

bool EinsumOpInferSymbolicShape(
    pir::Operation *op, pir::ShapeConstraintIRAnalysis *shape_analysis) {
  PADDLE_THROW(phi::errors::Unimplemented(
      op->name() + " 's InferSymbolicShape interface is NOT implemented now."));
  return true;
}

bool KthvalueOpInferSymbolicShape(
    pir::Operation *op, pir::ShapeConstraintIRAnalysis *shape_analysis) {
  pir::Value operand_source = op->operand_source(0);
  const symbol::ShapeOrDataDimExprs &operand_shape_or_data =
      shape_analysis->GetShapeOrDataForValue(operand_source);
  const auto &attributes = op->attributes();
  int axis = attributes.at("axis").dyn_cast<pir::Int32Attribute>().data();
  bool keepdim = GetBoolAttr(op, "keepdim");

  const auto &input_dims = operand_shape_or_data.shape();
  const int &dim_size = input_dims.size();
  if (axis < 0) axis += dim_size;
  std::vector<symbol::DimExpr> out_dims;
  for (int i = 0; i < axis; i++) {
    out_dims.emplace_back(input_dims[i]);
  }
  if (keepdim && dim_size > 0) {
    out_dims.emplace_back(symbol::DimExpr(1));
  }
  for (int i = axis + 1; i < dim_size; i++) {
    out_dims.emplace_back(input_dims[i]);
  }
  symbol::ShapeOrDataDimExprs shape_data{
      symbol::TensorShapeOrDataDimExprs(out_dims)};
  shape_analysis->SetShapeOrDataForValue(op->result(0), shape_data);
  shape_analysis->SetShapeOrDataForValue(op->result(1), shape_data);
  return true;
}

bool LogcumsumexpOpInferSymbolicShape(
    pir::Operation *op, pir::ShapeConstraintIRAnalysis *shape_analysis) {
  PADDLE_THROW(phi::errors::Unimplemented(
      op->name() + " 's InferSymbolicShape interface is NOT implemented now."));
  return true;
}

bool LogsumexpOpInferSymbolicShape(
    pir::Operation *op, pir::ShapeConstraintIRAnalysis *shape_analysis) {
  PADDLE_THROW(phi::errors::Unimplemented(
      op->name() + " 's InferSymbolicShape interface is NOT implemented now."));
  return true;
}

bool MaxOpInferSymbolicShape(pir::Operation *op,
                             pir::ShapeConstraintIRAnalysis *shape_analysis) {
  bool keepdim = GetBoolAttr(op, "keepdim");

  const std::vector<int64_t> axis = [&] {
    pir::Operation *axis_gen_op = op->operand_source(1).defining_op();
    std::vector<int64_t> axis_vec;
    if (axis_gen_op->isa<paddle::dialect::FullIntArrayOp>()) {
      axis_vec = details::GetVectorAttr(
          axis_gen_op->dyn_cast<paddle::dialect::FullIntArrayOp>(), "value");
    } else {
      // TODO(lanxianghit): there's other source: pir::VectorType,
      // paddle::dialect::DenseTensorType, but after PRIM, maybe always
      // FullIntArrayOp, to be confirmed
      PADDLE_THROW(
          phi::errors::Unimplemented("MaxOpInferSymbolicShape: 'axis' only "
                                     "support FullIntArrayOp's result now."));
    }
    return axis_vec;
  }();

  bool reduce_all = axis.size() == 0 ? true : false;

  return details::ReduceInferDim(op, shape_analysis, axis, keepdim, reduce_all);
}

bool MinOpInferSymbolicShape(pir::Operation *op,
                             pir::ShapeConstraintIRAnalysis *shape_analysis) {
  PADDLE_THROW(phi::errors::Unimplemented(
      op->name() + " 's InferSymbolicShape interface is NOT implemented now."));
  return true;
}

bool PadOpInferSymbolicShape(pir::Operation *op,
                             pir::ShapeConstraintIRAnalysis *shape_analysis) {
  PADDLE_THROW(phi::errors::Unimplemented(
      op->name() + " 's InferSymbolicShape interface is NOT implemented now."));
  return true;
}

bool PoissonOpInferSymbolicShape(
    pir::Operation *op, pir::ShapeConstraintIRAnalysis *shape_analysis) {
  PADDLE_THROW(phi::errors::Unimplemented(
      op->name() + " 's InferSymbolicShape interface is NOT implemented now."));
  return true;
}

bool ProdOpInferSymbolicShape(pir::Operation *op,
                              pir::ShapeConstraintIRAnalysis *shape_analysis) {
  bool keepdim = GetBoolAttr(op, "keep_dim");
  bool reduce_all = GetBoolAttr(op, "reduce_all");

  auto axis_gen_op = op->operand_source(1).defining_op();
  if (axis_gen_op->isa<paddle::dialect::FullIntArrayOp>()) {
    std::vector<int64_t> axis = details::GetVectorAttr(
        axis_gen_op->dyn_cast<paddle::dialect::FullIntArrayOp>(), "value");
    return details::ReduceInferDim(
        op, shape_analysis, axis, keepdim, reduce_all);
  } else {
    // TODO(lanxianghit): deal with other source: pir::VectorType,
    // paddle::dialect::DenseTensorType
    PADDLE_THROW(
        phi::errors::Unimplemented("ProdOpInferSymbolicShape: 'axis' only "
                                   "support FullIntArrayOp's result now."));
  }

  return true;
}

bool RepeatInterleaveOpInferSymbolicShape(
    pir::Operation *op, pir::ShapeConstraintIRAnalysis *shape_analysis) {
  PADDLE_THROW(phi::errors::Unimplemented(
      op->name() + " 's InferSymbolicShape interface is NOT implemented now."));
  return true;
}

symbol::ShapeOrDataDimExprs CreateShapeOrDataForXShape(
    const symbol::ShapeOrDataDimExprs &x_shape) {
  const std::vector<symbol::DimExpr> result = [&] {
    std::vector<symbol::DimExpr> new_x_dims;
    new_x_dims.reserve(x_shape.shape().size() + 1);
    new_x_dims.push_back(symbol::DimExpr{0});
    new_x_dims.insert(
        new_x_dims.end(), x_shape.shape().begin(), x_shape.shape().end());
    return new_x_dims;
  }();
  return symbol::ShapeOrDataDimExprs{symbol::TensorShapeOrDataDimExprs(result)};
}

bool ReshapeOpInferSymbolicShape(
    pir::Operation *op, pir::ShapeConstraintIRAnalysis *shape_analysis) {
  pir::Value operand_source = op->operand_source(0);
  if (shape_analysis->GetShapeOrDataForValue(operand_source)
          .data()
          .has_value()) {
    const symbol::ShapeOrDataDimExprs &operand_shape_or_data =
        shape_analysis->GetShapeOrDataForValue(operand_source);
    shape_analysis->SetShapeOrDataForValue(op->result(0),
                                           operand_shape_or_data);
    return true;
  }

  pir::Value operand_source_shape = op->operand_source(1);

  const symbol::ShapeOrDataDimExprs &operand_shape_or_data =
      shape_analysis->GetShapeOrDataForValue(operand_source_shape);

  const auto &GetProduct = [&](const auto &dim_exprs, const auto &Filter) {
    symbol::DimExpr product{1};
    for (const auto &dim_expr : dim_exprs) {
      if (Filter(dim_expr)) {
        product = product * dim_expr;
      }
    }
    return product;
  };

  const auto &IsNotMinusOne = [&](const symbol::DimExpr &dim_expr) {
    if (dim_expr.isa<int64_t>()) {
      return dim_expr.dyn_cast<int64_t>() != static_cast<int64_t>(-1);
    }
    return true;
  };

  const auto &IsZero = [&](const symbol::DimExpr &dim_expr) {
    if (dim_expr.isa<int64_t>()) {
      return dim_expr.dyn_cast<int64_t>() == static_cast<int64_t>(0);
    }
    return false;
  };

  const std::vector<symbol::DimExpr> out_dims = [&] {
    const auto &original_shape =
        shape_analysis->GetShapeOrDataForValue(op->operand_source(0)).shape();

    const auto &numel =
        GetProduct(original_shape, [](const auto &) { return true; });

    ExprVec target_shape = details::GetExprVecFromData(operand_shape_or_data);
    const auto &product_exclude_minus_one =
        GetProduct(target_shape, IsNotMinusOne);

    const auto &input_dims = target_shape;

    std::vector<symbol::DimExpr> out_dims;
    out_dims.reserve(input_dims.size());
    for (size_t i = 0; i < input_dims.size(); ++i) {
      auto out_dim_expr = IsNotMinusOne(input_dims[i])
                              ? input_dims[i]
                              : (numel / product_exclude_minus_one);
      out_dim_expr = IsZero(input_dims[i]) ? original_shape[i] : out_dim_expr;
      out_dims.emplace_back(out_dim_expr);
    }

    return out_dims;
  }();

  symbol::ShapeOrDataDimExprs shape_data{
      symbol::TensorShapeOrDataDimExprs(out_dims)};

  shape_analysis->SetShapeOrDataForValue(op->result(0), shape_data);

  const auto &x_shape = [&] {
    std::vector<symbol::DimExpr> x_shape{symbol::DimExpr(0)};
    const auto &original_shape =
        shape_analysis->GetShapeOrDataForValue(op->operand_source(0)).shape();
    for (const auto &dim : original_shape) {
      x_shape.push_back(dim);
    }
    return x_shape;
  }();
  shape_analysis->SetShapeOrDataForValue(
      op->result(1),
      CreateShapeOrDataForXShape(
          shape_analysis->GetShapeOrDataForValue(operand_source)));
  return true;
}

bool Reshape_OpInferSymbolicShape(
    pir::Operation *op, pir::ShapeConstraintIRAnalysis *shape_analysis) {
  return ReshapeOpInferSymbolicShape(op, shape_analysis);
}

bool ShapeOpInferSymbolicShape(pir::Operation *op,
                               pir::ShapeConstraintIRAnalysis *shape_analysis) {
  const symbol::ShapeOrDataDimExprs &operand_shape_or_data =
      shape_analysis->GetShapeOrDataForValue(op->operand_source(0));
  const auto &out_data = operand_shape_or_data.shape();
  const std::vector<symbol::DimExpr> shape{std::int64_t(out_data.size())};
  symbol::ShapeOrDataDimExprs shape_or_data{
      symbol::TensorShapeOrDataDimExprs(shape, out_data)};

  shape_analysis->SetShapeOrDataForValue(op->result(0), shape_or_data);
  return true;
}

bool ShapeSrOpInferSymbolicShape(
    pir::Operation *op, pir::ShapeConstraintIRAnalysis *shape_analysis) {
  return ShapeOpInferSymbolicShape(op, shape_analysis);
}

bool SliceOpInferSymbolicShape(pir::Operation *op,
                               pir::ShapeConstraintIRAnalysis *shape_analysis) {
  pir::Value operand_source = op->operand_source(0);
  pir::Value operand_starts = op->operand_source(1);
  pir::Value operand_ends = op->operand_source(2);
  pir::Value res = op->result(0);

  const symbol::ShapeOrDataDimExprs &operand_shape_or_data =
      shape_analysis->GetShapeOrDataForValue(operand_source);
  const symbol::ShapeOrDataDimExprs &starts_shape_data =
      shape_analysis->GetShapeOrDataForValue(operand_starts);
  const symbol::ShapeOrDataDimExprs &ends_shape_data =
      shape_analysis->GetShapeOrDataForValue(operand_ends);

  std::vector<int64_t> axes_vec = details::GetVectorAttr(op, "axes");

  // // Currently, we DO NOT support any element in `starts` is a Symbol.
  ExprVec starts = slice_utils::GetExprVecFromData(starts_shape_data);
  ExprVec ends = slice_utils::GetExprVecFromData(ends_shape_data);

  std::vector<int64_t> infer_flags = details::GetVectorAttr(op, "infer_flags");

  const std::vector<int64_t> decrease_axis =
      details::GetVectorAttr(op, "decrease_axis");

  shape_analysis->SetShapeOrDataForValue(
      res,
      slice_utils::SliceRawInferSymbolicShape(operand_shape_or_data,
                                              starts,
                                              ends,
                                              axes_vec,
                                              infer_flags,
                                              decrease_axis));

  return true;
}

bool SplitOpInferSymbolicShape(pir::Operation *op,
                               pir::ShapeConstraintIRAnalysis *shape_analysis) {
  // input
  const auto &x_shape_or_data =
      shape_analysis->GetShapeOrDataForValue(op->operand_source(0));
  PADDLE_ENFORCE_EQ(x_shape_or_data.data().has_value(),
                    false,
                    phi::errors::InvalidArgument(
                        "InferSymbolicShape of SplitOp only support input with "
                        "value now."));
  const auto &x_dims_sym = x_shape_or_data.shape();

  // axis
  CHECK(op->operand_source(2).defining_op()->isa<paddle::dialect::FullOp>());

  int64_t axis = op->operand_source(2)
                     .defining_op<paddle::dialect::FullOp>()
                     .attributes()
                     .at("value")
                     .dyn_cast<paddle::dialect::ScalarAttribute>()
                     .data()
                     .to<int64_t>();

  // sections
  const std::vector<symbol::DimExpr> &sections_sym = [&] {
    const auto &sections_shape_or_data =
        shape_analysis->GetShapeOrDataForValue(op->operand_source(1));
    std::vector<symbol::DimExpr> sections_sym;
    if (sections_shape_or_data.data().has_value()) {
      sections_sym = sections_shape_or_data.data().value();
    } else {
      sections_sym = sections_shape_or_data.shape();
    }
    return sections_sym;
  }();

  // output
  const symbol::TensorListShapeOrDataDimExprs &output_shape_data_list = [&] {
    const auto &GetSum = [&](const auto &dim_exprs, const auto &Filter) {
      symbol::DimExpr sum{0};
      for (const auto &dim_expr : dim_exprs) {
        if (Filter(dim_expr)) {
          sum = sum + dim_expr;
        }
      }
      return sum;
    };
    const auto &All = [&](const auto &dim_exprs, const auto &Cond) {
      for (const auto &dim_expr : dim_exprs) {
        if (!Cond(dim_expr)) {
          return false;
        }
      }
      return true;
    };
    const auto &IsNotMinusOne = [&](const symbol::DimExpr &dim_expr) {
      if (dim_expr.isa<int64_t>()) {
        return dim_expr.dyn_cast<int64_t>() != static_cast<int64_t>(-1);
      }
      return true;
    };
    const auto &sum_exclude_minus_one = GetSum(sections_sym, IsNotMinusOne);

    const bool &all_sections_sym_not_minus_one =
        All(sections_sym, IsNotMinusOne);
    if (all_sections_sym_not_minus_one) {
      shape_analysis->DimExprBuilder().CstrEq(x_dims_sym[axis],
                                              sum_exclude_minus_one);
    }

    symbol::TensorListShapeOrDataDimExprs shape_data_list;
    std::vector<symbol::DimExpr> output_dims_sym = x_dims_sym;
    if (!all_sections_sym_not_minus_one && sections_sym.size() == 1) {
      VLOG(3) << "[SplitOp]-1 is the only split section. The output shape is "
                 "identical to the input shape.";
      shape_data_list.push_back(
          symbol::TensorShapeOrDataDimExprs(output_dims_sym));
      return shape_data_list;
    }
    for (uint32_t idx = 0; idx < sections_sym.size(); idx++) {
      const auto &section_sym = sections_sym[idx];
      output_dims_sym[axis] = IsNotMinusOne(section_sym)
                                  ? section_sym
                                  : x_dims_sym[axis] - sum_exclude_minus_one;

      shape_data_list.push_back(
          symbol::TensorShapeOrDataDimExprs(output_dims_sym));
    }
    return shape_data_list;
  }();

  shape_analysis->SetShapeOrDataForValue(
      op->result(0), symbol::ShapeOrDataDimExprs{output_shape_data_list});

  return true;
}

bool SplitWithNumOpInferSymbolicShape(
    pir::Operation *op, pir::ShapeConstraintIRAnalysis *shape_analysis) {
  PADDLE_THROW(phi::errors::Unimplemented(
      op->name() + " 's InferSymbolicShape interface is NOT implemented now."));
  return true;
}

bool SumOpInferSymbolicShape(pir::Operation *op,
                             pir::ShapeConstraintIRAnalysis *shape_analysis) {
  bool keepdim = GetBoolAttr(op, "keepdim");
  bool reduce_all = false;

  auto axis_gen_op = op->operand_source(1).defining_op();
  if (axis_gen_op->isa<paddle::dialect::FullIntArrayOp>()) {
    std::vector<int64_t> axis = details::GetVectorAttr(
        axis_gen_op->dyn_cast<paddle::dialect::FullIntArrayOp>(), "value");
    if (axis.size() == 0) {
      reduce_all = true;
    }
    return details::ReduceInferDim(
        op, shape_analysis, axis, keepdim, reduce_all);
  } else {
    // TODO(lanxianghit): deal with other source: pir::VectorType,
    // paddle::dialect::DenseTensorType
    PADDLE_THROW(
        phi::errors::Unimplemented("SumOpInferSymbolicShape: 'axis' only "
                                   "support FullIntArrayOp's result now."));
  }

  return true;
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

  pir::Value res = op->result(0);
  shape_analysis->SetShapeOrDataForValue(res, shape_data);

  return true;
}

bool TopkOpInferSymbolicShape(pir::Operation *op,
                              pir::ShapeConstraintIRAnalysis *shape_analysis) {
  PADDLE_THROW(phi::errors::Unimplemented(
      op->name() + " 's InferSymbolicShape interface is NOT implemented now."));
  return true;
}

bool TransposeOpInferSymbolicShape(
    pir::Operation *op, pir::ShapeConstraintIRAnalysis *shape_analysis) {
  std::vector<pir::Attribute> perm =
      op->attributes().at("perm").dyn_cast<pir::ArrayAttribute>().AsVector();
  if (perm.size() == 1) {
    // perm must be [0], which means nothing to do with input, just copy the
    // info from input
    shape_analysis->SetShapeOrDataForValue(
        op->result(0),
        shape_analysis->GetShapeOrDataForValue(op->operand_source(0)));
    return true;
  }
  const std::vector<symbol::DimExpr> &x_dims = [&] {
    std::vector<symbol::DimExpr> dims;
    const auto &x_shape_or_data =
        shape_analysis->GetShapeOrDataForValue(op->operand_source(0));
    if (x_shape_or_data.data().has_value()) {
      dims = x_shape_or_data.data().value();
    } else {
      dims = x_shape_or_data.shape();
    }
    return dims;
  }();

  int x_rank = x_dims.size();

  const std::vector<int32_t> formatted_axis = [op, x_rank, &perm] {
    std::vector<int32_t> out(perm.size(), 0);
    std::transform(perm.begin(),
                   perm.end(),
                   out.begin(),
                   [](pir::Attribute &p) -> int32_t {
                     return p.dyn_cast<pir::Int32Attribute>().data();
                   });

    // format the negative axis
    std::for_each(out.begin(), out.end(), [x_rank](int32_t &v) {
      if (v < 0) {
        v += x_rank;
      }
    });
    return out;
  }();

  int axis_size = static_cast<int>(formatted_axis.size());

  std::vector<symbol::DimExpr> out_dims(x_dims);
  for (int i = 0; i < axis_size; ++i) {
    out_dims[i] = x_dims[formatted_axis[i]];
  }

  shape_analysis->SetShapeOrDataForValue(op->result(0),
                                         ShapeOrData{TensorExprs(out_dims)});

  return true;
}

bool Transpose_OpInferSymbolicShape(
    pir::Operation *op, pir::ShapeConstraintIRAnalysis *shape_analysis) {
  return TransposeOpInferSymbolicShape(op, shape_analysis);
}

bool TriuOpInferSymbolicShape(pir::Operation *op,
                              pir::ShapeConstraintIRAnalysis *shape_analysis) {
  PADDLE_THROW(phi::errors::Unimplemented(
      op->name() + " 's InferSymbolicShape interface is NOT implemented now."));
  return true;
}

bool Triu_OpInferSymbolicShape(pir::Operation *op,
                               pir::ShapeConstraintIRAnalysis *shape_analysis) {
  return TriuOpInferSymbolicShape(op, shape_analysis);
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
          should_squeeze[current] = true;
        } else {
          should_squeeze[current] = true;
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

  pir::Value res = op->result(0);
  shape_analysis->SetShapeOrDataForValue(res, shape_data);
  shape_analysis->SetShapeOrDataForValue(
      op->result(1), CreateShapeOrDataForXShape(x_shape_or_data));

  return true;
}
bool Squeeze_OpInferSymbolicShape(
    pir::Operation *op, pir::ShapeConstraintIRAnalysis *shape_analysis) {
  return SqueezeOpInferSymbolicShape(op, shape_analysis);
}

bool UnbindOpInferSymbolicShape(
    pir::Operation *op, pir::ShapeConstraintIRAnalysis *shape_analysis) {
  PADDLE_THROW(phi::errors::Unimplemented(
      op->name() + " 's InferSymbolicShape interface is NOT implemented now."));
  return true;
}

bool UniqueOpInferSymbolicShape(
    pir::Operation *op, pir::ShapeConstraintIRAnalysis *shape_analysis) {
  PADDLE_THROW(phi::errors::Unimplemented(
      op->name() + " 's InferSymbolicShape interface is NOT implemented now."));
  return true;
}

bool UniqueConsecutiveOpInferSymbolicShape(
    pir::Operation *op, pir::ShapeConstraintIRAnalysis *shape_analysis) {
  PADDLE_THROW(phi::errors::Unimplemented(
      op->name() + " 's InferSymbolicShape interface is NOT implemented now."));
  return true;
}

bool UnsqueezeOpInferSymbolicShape(
    pir::Operation *op, pir::ShapeConstraintIRAnalysis *shape_analysis) {
  IR_ENFORCE(op->num_operands() == 2,
             "UnsqueezeOp InferSymbolicShape ONLY support num_operands() == 2 "
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

  pir::Value res = op->result(0);
  shape_analysis->SetShapeOrDataForValue(res, shape_data);
  shape_analysis->SetShapeOrDataForValue(
      op->result(1), CreateShapeOrDataForXShape(x_shape_or_data));

  return true;
}
bool Unsqueeze_OpInferSymbolicShape(
    pir::Operation *op, pir::ShapeConstraintIRAnalysis *shape_analysis) {
  return UnsqueezeOpInferSymbolicShape(op, shape_analysis);
}

}  // namespace paddle::dialect
