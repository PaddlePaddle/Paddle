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

#include "paddle/fluid/pir/dialect/operator/interface/infer_symbolic_shape/paddle_op_infer_sym.h"
#include "paddle/fluid/pir/dialect/operator/interface/infer_symbolic_shape/infer_sym_utils.h"
#include "paddle/fluid/pir/dialect/operator/ir/op_attribute.h"

namespace paddle::dialect {

// To make codes shorter
using ShapeOrData = symbol::ShapeOrDataDimExprs;
using TensorExprs = symbol::TensorShapeOrDataDimExprs;
using TensorListExprs = symbol::TensorListShapeOrDataDimExprs;

bool DataOpInferSymbolicShape(pir::Operation *op,
                              pir::ShapeConstraintIRAnalysis *shape_analysis) {
  const auto &attributes = op->attributes();
  pir::Attribute attr = attributes.at("shape");

  const std::vector<symbol::DimExpr> sym_dims = [&] {
    std::vector<symbol::DimExpr> sym_dims;
    const std::vector<int64_t> &dims =
        attr.dyn_cast<paddle::dialect::IntArrayAttribute>().data().GetData();
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
    return sym_dims;
  }();

  symbol::ShapeOrDataDimExprs shape_data{
      symbol::TensorShapeOrDataDimExprs(sym_dims)};

  shape_analysis->SetShapeOrDataForValue(op->result(0), shape_data);

  return true;
}

bool ShapeOpInferSymbolicShape(pir::Operation *op,
                               pir::ShapeConstraintIRAnalysis *shape_analysis) {
  const symbol::ShapeOrDataDimExprs &operand_shape_or_data =
      shape_analysis->GetShapeOrDataForValue(op->operand_source(0));

  const std::vector<symbol::DimExpr> sym_shape = [&] {
    std::vector<symbol::DimExpr> sym_shape;
    symbol::DimExpr dim_expr(
        op->result(0).type().dyn_cast<pir::DenseTensorType>().dims()[0]);
    sym_shape.emplace_back(dim_expr);
    return sym_shape;
  }();

  symbol::ShapeOrDataDimExprs shape_or_data{symbol::TensorShapeOrDataDimExprs(
      sym_shape, operand_shape_or_data.shape())};

  shape_analysis->SetShapeOrDataForValue(op->result(0), shape_or_data);

  return true;
}

bool ShapeSrOpInferSymbolicShape(
    pir::Operation *op, pir::ShapeConstraintIRAnalysis *shape_analysis) {
  return ShapeOpInferSymbolicShape(op, shape_analysis);
}

void BuildCstrEqForTensorListAlongAxis(
    pir::ShapeConstraintIRAnalysis *shape_analysis,
    const symbol::TensorListShapeOrDataDimExprs &shape_data_list,
    int axis) {
  for (size_t i = 1; i < shape_data_list.size(); ++i) {
    shape_analysis->CreateDimExprBuilder().CstrEq(
        shape_data_list[0].shape()[axis], shape_data_list[i].shape()[axis]);
  }
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
        if (i == axis) continue;
        BuildCstrEqForTensorListAlongAxis(shape_analysis, shape_data_list, i);
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

bool SumOpInferSymbolicShape(pir::Operation *op,
                             pir::ShapeConstraintIRAnalysis *shape_analysis) {
  const auto &attributes = op->attributes();
  bool keepdim = attributes.at("keepdim").dyn_cast<pir::BoolAttribute>().data();

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

bool ProdOpInferSymbolicShape(pir::Operation *op,
                              pir::ShapeConstraintIRAnalysis *shape_analysis) {
  const auto &attributes = op->attributes();
  bool keepdim =
      attributes.at("keep_dim").dyn_cast<pir::BoolAttribute>().data();

  bool reduce_all =
      attributes.at("reduce_all").dyn_cast<pir::BoolAttribute>().data();

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

bool ReshapeOpInferSymbolicShape(
    pir::Operation *op, pir::ShapeConstraintIRAnalysis *shape_analysis) {
  pir::Value operand_source_shape = op->operand_source(1);

  const symbol::ShapeOrDataDimExprs &operand_shape_or_data =
      shape_analysis->GetShapeOrDataForValue(operand_source_shape);

  const std::vector<symbol::DimExpr> out_dims = [&] {
    std::vector<symbol::DimExpr> out_dims;
    out_dims = operand_shape_or_data.data().value();

    symbol::DimExpr product = symbol::DimExpr(1);
    symbol::DimExpr numel = symbol::DimExpr(1);

    const auto &original_shape =
        shape_analysis->GetShapeOrDataForValue(op->operand_source(0)).shape();
    for (auto &dim_expr : original_shape) {
      numel = numel * dim_expr;
    }

    for (size_t i = 0; i < out_dims.size(); i++) {
      if (out_dims[i].isa<int64_t>()) {
        if (out_dims[i].dyn_cast<int64_t>() != static_cast<int64_t>(-1)) {
          product = product * out_dims[i];
        } else if (i == out_dims.size() - 1) {
          out_dims[i] = numel / product;
        } else {
          // doing nothing
        }
      } else {
        product = product * out_dims[i];
      }
    }

    return out_dims;
  }();

  symbol::ShapeOrDataDimExprs shape_data{
      symbol::TensorShapeOrDataDimExprs(out_dims)};

  shape_analysis->SetShapeOrDataForValue(op->result(0), shape_data);
  shape_analysis->SetShapeOrDataForValue(
      op->result(1),
      shape_analysis->GetShapeOrDataForValue(operand_source_shape));
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

  const std::vector<symbol::DimExpr> data = [&] {
    std::vector<symbol::DimExpr> data;
    for (auto item : vec) {
      int64_t i = item.dyn_cast<pir::Int64Attribute>().data();
      data.push_back(symbol::DimExpr(i));
    }
    return data;
  }();

  const std::vector<symbol::DimExpr> shape{std::int64_t(vec.size())};

  symbol::ShapeOrDataDimExprs shape_data{
      symbol::TensorShapeOrDataDimExprs(shape, data)};

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

  const symbol::ShapeOrDataDimExprs &operand_shape_or_data =
      shape_analysis->GetShapeOrDataForValue(operand_source);
  const symbol::ShapeOrDataDimExprs &starts_shape_data =
      shape_analysis->GetShapeOrDataForValue(operand_starts);
  const symbol::ShapeOrDataDimExprs &ends_shape_data =
      shape_analysis->GetShapeOrDataForValue(operand_ends);

  // Currently, we DO NOT support the case that any element in `axes` `starts`
  // or `ends` is a Symbol.
  const std::vector<int64_t> axes = [&] {
    const auto &attributes = op->attributes();
    pir::Attribute attr_axes = attributes.at("axes");

    const auto &axes_vec = attr_axes.dyn_cast<pir::ArrayAttribute>().AsVector();
    std::vector<int64_t> axes;
    int64_t rank = int64_t(operand_shape_or_data.shape().size());
    for (auto item : axes_vec) {
      int64_t axis = item.dyn_cast<pir::Int64Attribute>().data();
      axes.emplace_back(axis >= 0 ? axis : std::max(int64_t(0), axis + rank));
    }
    return axes;
  }();

  const std::vector<int64_t> starts = [&] {
    std::vector<int64_t> starts;
    for (auto item : starts_shape_data.data().value()) {
      IR_ENFORCE(item.isa<int64_t>(),
                 "Currently, we DO NOT support the case that any element in "
                 "`starts` is a Symbol.");
      starts.push_back(item.Get<int64_t>());
    }
    return starts;
  }();

  const std::vector<int64_t> ends = [&] {
    std::vector<int64_t> ends;
    for (auto item : ends_shape_data.data().value()) {
      IR_ENFORCE(item.isa<int64_t>(),
                 "Currently, we DO NOT support the case that any element in "
                 "`ends` is a Symbol.");
      ends.push_back(item.Get<int64_t>());
    }
    return ends;
  }();

  // When `pd.slice` is operating on a tensor which is produced by a `pd.shape`
  // op, the reseult should be written into data.
  const auto &GetDataDimExprs = [&]() -> symbol::ShapeOrDataDimExprs {
    const std::vector<symbol::DimExpr> out_data = [&] {
      std::vector<symbol::DimExpr> out_data;
      for (int64_t i = starts[0]; i < ends[0]; i++) {
        out_data.push_back(operand_shape_or_data.data().value()[i]);
      }
      return out_data;
    }();
    const std::vector<symbol::DimExpr> shape{std::int64_t(out_data.size())};
    return symbol::ShapeOrDataDimExprs{
        symbol::TensorShapeOrDataDimExprs(shape, out_data)};
  };

  // Othewise, the reseult should be written into the shape.
  const auto &GetShapeDimExprs = [&]() -> symbol::ShapeOrDataDimExprs {
    std::vector<symbol::DimExpr> out_shape = operand_shape_or_data.shape();

    const std::vector<symbol::DimExpr> &dim_expr_starts =
        starts_shape_data.data().value();
    const std::vector<symbol::DimExpr> &dim_expr_ends =
        ends_shape_data.data().value();

    // For both start and end can be negtive or positive, we need to handle the
    // following different arrangements.
    auto IsMaxInt = [](const symbol::DimExpr &expr) {
      return expr.isa<int64_t>() &&
             expr.Get<int64_t>() ==
                 static_cast<int64_t>(std::numeric_limits<int>::max());
    };
    for (size_t i = 0; i < axes.size(); ++i) {
      int64_t axis = axes[i];
      auto end =
          IsMaxInt(dim_expr_ends[i]) ? out_shape[axis] : dim_expr_ends[i];
      if ((starts[i] >= 0 && ends[i] >= 0) ||
          (starts[i] <= 0 && ends[i] <= 0)) {  // both negtive or positive.
        out_shape[axis] = end - dim_expr_starts[i];
      } else if (starts[i] <= 0 &&
                 ends[i] >= 0) {  // negtive start, positive end
        out_shape[axis] = end - dim_expr_starts[i] - out_shape[axis];
      } else if (starts[i] >= 0 &&
                 ends[i] <= 0) {  // positive start, negtive end
        out_shape[axis] = out_shape[axis] - dim_expr_starts[i] + end;
      }
    }

    return symbol::ShapeOrDataDimExprs{
        symbol::TensorShapeOrDataDimExprs(out_shape)};
  };

  symbol::ShapeOrDataDimExprs shape_data =
      operand_shape_or_data.data().has_value() ? GetDataDimExprs()
                                               : GetShapeDimExprs();

  shape_analysis->SetShapeOrDataForValue(res, shape_data);
  return true;
}

bool FullOpInferSymbolicShape(pir::Operation *op,
                              pir::ShapeConstraintIRAnalysis *shape_analysis) {
  const auto &attributes = op->attributes();

  const std::vector<symbol::DimExpr> shape = [&] {
    std::vector<symbol::DimExpr> shape;
    pir::Attribute attr_shape = attributes.at("shape");
    const auto &shape_vec =
        attr_shape.dyn_cast<paddle::dialect::IntArrayAttribute>()
            .data()
            .GetData();

    for (auto &dim : shape_vec) {
      shape.push_back(symbol::DimExpr(dim));
    }
    return shape;
  }();

  // Keep shape info always with `int64_t` type.
  int64_t value = attributes.at("value")
                      .dyn_cast<paddle::dialect::ScalarAttribute>()
                      .data()
                      .to<int64_t>();
  std::vector<symbol::DimExpr> data{symbol::DimExpr(value)};

  symbol::ShapeOrDataDimExprs shape_data{
      symbol::TensorShapeOrDataDimExprs(shape, data)};

  shape_analysis->SetShapeOrDataForValue(op->result(0), shape_data);
  return true;
}

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

  const std::vector<symbol::DimExpr> &out_dims = [&] {
    std::vector<symbol::DimExpr> out_dims = shape_data_list[0].shape();
    for (size_t i = 1; i < shape_data_list.size(); ++i) {
      for (size_t j = 0; j < rank; ++j) {
        if (j != static_cast<size_t>(axis)) {
          // This func have bug
          BuildCstrEqForTensorListAlongAxis(shape_analysis, shape_data_list, i);
          continue;
        }
        out_dims[axis] = out_dims[axis] + shape_data_list[i].shape()[axis];
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

  pir::Value res = op->result(0);
  shape_analysis->SetShapeOrDataForValue(res, shape_data);

  return true;
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

  pir::Value res = op->result(0);
  shape_analysis->SetShapeOrDataForValue(res, shape_data);

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

  const std::vector<int32_t> formated_axis = [op, x_rank, &perm] {
    std::vector<int32_t> out(perm.size(), 0);
    std::transform(perm.begin(),
                   perm.end(),
                   out.begin(),
                   [](pir::Attribute &p) -> int32_t {
                     return p.dyn_cast<pir::Int32Attribute>().data();
                   });

    // format the negtive axis
    std::for_each(out.begin(), out.end(), [x_rank](int32_t &v) {
      if (v < 0) {
        v += x_rank;
      }
    });
    return out;
  }();

  int axis_size = static_cast<int>(formated_axis.size());

  std::vector<symbol::DimExpr> out_dims(x_dims);
  for (int i = 0; i < axis_size; ++i) {
    out_dims[i] = x_dims[formated_axis[i]];
  }

  shape_analysis->SetShapeOrDataForValue(op->result(0),
                                         ShapeOrData{TensorExprs(out_dims)});

  return true;
}
bool Transpose_OpInferSymbolicShape(
    pir::Operation *op, pir::ShapeConstraintIRAnalysis *shape_analysis) {
  return TransposeOpInferSymbolicShape(op, shape_analysis);
}

bool ArangeOpInferSymbolicShape(
    pir::Operation *op, pir::ShapeConstraintIRAnalysis *shape_analysis) {
  const auto &start_shape_or_data =
      shape_analysis->GetShapeOrDataForValue(op->operand_source(0));
  const auto &end_shape_or_data =
      shape_analysis->GetShapeOrDataForValue(op->operand_source(1));
  const auto &step_shape_or_data =
      shape_analysis->GetShapeOrDataForValue(op->operand_source(2));

  const auto start = [&] {
    symbol::DimExpr expr;
    if (start_shape_or_data.data().has_value()) {
      expr = start_shape_or_data.data().value()[0];
    } else {
      expr = start_shape_or_data.shape()[0];
    }
    return expr;
  }();

  const auto end = [&] {
    symbol::DimExpr expr;
    if (end_shape_or_data.data().has_value()) {
      expr = end_shape_or_data.data().value()[0];
    } else {
      expr = end_shape_or_data.shape()[0];
    }
    return expr;
  }();

  const auto step = [&] {
    symbol::DimExpr expr;
    if (step_shape_or_data.data().has_value()) {
      expr = step_shape_or_data.data().value()[0];
    } else {
      expr = step_shape_or_data.shape()[0];
    }
    return expr;
  }();

  const symbol::ShapeOrDataDimExprs &shape_data = [&] {
    std::vector<symbol::DimExpr> out_dims;
    // TODO(lanxianghit, jiahy0825): here should be ceil((end - start) / step),
    // but DimExpr doesn't support ceil and float now
    out_dims.emplace_back((end - start) / step);
    return symbol::ShapeOrDataDimExprs{
        symbol::TensorShapeOrDataDimExprs(out_dims)};
  }();

  shape_analysis->SetShapeOrDataForValue(op->result(0), shape_data);

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
      op->name() + " 's InferSymbolicShape interface is NOT implemented now."));
  return true;
}

bool ExpandOpInferSymbolicShape(
    pir::Operation *op, pir::ShapeConstraintIRAnalysis *shape_analysis) {
  PADDLE_THROW(phi::errors::Unimplemented(
      op->name() + " 's InferSymbolicShape interface is NOT implemented now."));
  return true;
}

bool MatmulOpInferSymbolicShape(
    pir::Operation *op, pir::ShapeConstraintIRAnalysis *shape_analysis) {
  // x_dims can't be const or ref here, in case to be broadcasted
  std::vector<symbol::DimExpr> x_dims = [&] {
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

  // y_dims can't be const or ref here, in case to be broadcasted
  std::vector<symbol::DimExpr> y_dims = [&] {
    std::vector<symbol::DimExpr> dims;
    const auto y_shape_or_data =
        shape_analysis->GetShapeOrDataForValue(op->operand_source(1));
    if (y_shape_or_data.data().has_value()) {
      dims = y_shape_or_data.data().value();
    } else {
      dims = y_shape_or_data.shape();
    }
    return dims;
  }();

  size_t ndims_x = x_dims.size();
  size_t ndims_y = y_dims.size();

  const bool x_broadcasted = [&] {
    bool broadcasted = false;
    if (ndims_x == 1) {
      x_dims.insert(x_dims.begin(), 1);
      ndims_x = 2;
      broadcasted = true;
    }
    return broadcasted;
  }();

  const bool y_broadcasted = [&] {
    bool broadcasted = false;
    if (ndims_y == 1) {
      y_dims.emplace_back(1);
      ndims_y = 2;
      broadcasted = true;
    }
    return broadcasted;
  }();

  std::vector<symbol::DimExpr> out_dims;
  if (ndims_x > ndims_y) {
    out_dims.assign(x_dims.begin(), x_dims.end() - 2);
  } else if (ndims_x < ndims_y) {
    out_dims.assign(y_dims.begin(), y_dims.end() - 2);
  } else {
    symbol::DimExprBuilder builder{nullptr};
    for (size_t i = 0; i < ndims_x - 2; ++i) {
      out_dims.emplace_back(builder.Broadcast(x_dims[i], y_dims[i]));
    }
  }

  symbol::DimExpr out_M =
      op->attributes().at("transpose_x").dyn_cast<pir::BoolAttribute>().data()
          ? x_dims[ndims_x - 1]
          : x_dims[ndims_x - 2];
  symbol::DimExpr out_N =
      op->attributes().at("transpose_y").dyn_cast<pir::BoolAttribute>().data()
          ? y_dims[ndims_y - 2]
          : y_dims[ndims_y - 1];
  if (!x_broadcasted) {
    out_dims.emplace_back(out_M);
  }
  if (!y_broadcasted) {
    out_dims.emplace_back(out_N);
  }

  shape_analysis->SetShapeOrDataForValue(op->result(0),
                                         ShapeOrData{TensorExprs(out_dims)});

  return true;
}

bool MaxOpInferSymbolicShape(pir::Operation *op,
                             pir::ShapeConstraintIRAnalysis *shape_analysis) {
  bool keepdim =
      op->attributes().at("keepdim").dyn_cast<pir::BoolAttribute>().data();

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

bool WhereOpInferSymbolicShape(pir::Operation *op,
                               pir::ShapeConstraintIRAnalysis *shape_analysis) {
  PADDLE_THROW(phi::errors::Unimplemented(
      op->name() + " 's InferSymbolicShape interface is NOT implemented now."));
  return true;
}

bool Where_OpInferSymbolicShape(
    pir::Operation *op, pir::ShapeConstraintIRAnalysis *shape_analysis) {
  return WhereOpInferSymbolicShape(op, shape_analysis);
}

bool FeedOpInferSymbolicShape(pir::Operation *op,
                              pir::ShapeConstraintIRAnalysis *shape_analysis) {
  // This Op has NO InferMeta in yaml, just return true
  return true;
}

bool TopPSamplingOpInferSymbolicShape(
    pir::Operation *op, pir::ShapeConstraintIRAnalysis *shape_analysis) {
  PADDLE_THROW(phi::errors::Unimplemented(
      op->name() + " 's InferSymbolicShape interface is NOT implemented now."));
  return true;
}

bool ExpandAsOpInferSymbolicShape(
    pir::Operation *op, pir::ShapeConstraintIRAnalysis *shape_analysis) {
  PADDLE_THROW(phi::errors::Unimplemented(
      op->name() + " 's InferSymbolicShape interface is NOT implemented now."));
  return true;
}

bool SplitOpInferSymbolicShape(pir::Operation *op,
                               pir::ShapeConstraintIRAnalysis *shape_analysis) {
  PADDLE_THROW(phi::errors::Unimplemented(
      op->name() + " 's InferSymbolicShape interface is NOT implemented now."));
  return true;
}

}  // namespace paddle::dialect
