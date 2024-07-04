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
#include "paddle/fluid/pir/dialect/operator/interface/infer_symbolic_shape/infer_sym_slice_utils.h"
#include "paddle/fluid/pir/dialect/operator/interface/infer_symbolic_shape/infer_sym_utils.h"

namespace cinn::dialect {

bool BroadcastOpInferSymbolicShape(
    pir::Operation *op, pir::InferSymbolicShapeContext *infer_context) {
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
  infer_context->SetShapeOrDataForValue(op->result(0), shape_data);
  return true;
}

bool ConcatOpInferSymbolicShape(pir::Operation *op,
                                pir::InferSymbolicShapeContext *infer_context) {
  const auto input_values = op->operands_source();
  const auto input_size = input_values.size();

  const auto IsAllDataValue = [&]() -> bool {
    for (const auto &value : input_values) {
      if (!infer_context->GetShapeOrDataForValue(value).data().has_value()) {
        return false;
      }
    }
    return true;
  };

  if (IsAllDataValue()) {
    std::vector<symbol::DimExpr> out_data;
    for (const auto &value : input_values) {
      const auto &shape_or_data = infer_context->GetShapeOrDataForValue(value);
      for (size_t i = 0; i < shape_or_data.data().value().size(); ++i) {
        out_data.emplace_back(shape_or_data.data().value()[i]);
      }
    }
    const std::vector<symbol::DimExpr> shape{std::int64_t(out_data.size())};
    symbol::ShapeOrDataDimExprs shape_data{
        symbol::TensorShapeOrDataDimExprs(shape, out_data)};

    pir::Value res = op->result(0);
    infer_context->SetShapeOrDataForValue(res, shape_data);
    return true;
  }

  int axis = op->attributes().at("axis").dyn_cast<pir::Int32Attribute>().data();

  const auto &GetOutDimExprs = [&]() -> std::vector<symbol::DimExpr> {
    std::vector<symbol::DimExpr> out_dims =
        infer_context->GetShapeOrDataForValue(input_values[0]).shape();

    size_t rank = out_dims.size();
    axis = axis >= 0 ? axis : std::max(int64_t(0), int64_t(axis + rank));

    for (size_t i = 1; i < input_size; ++i) {
      const auto &operand_shape_or_data =
          infer_context->GetShapeOrDataForValue(input_values[i]);
      out_dims[axis] = out_dims[axis] + operand_shape_or_data.shape()[axis];
    }

    for (size_t i = 0; i < rank; ++i) {
      if (i == static_cast<size_t>(axis)) continue;
      paddle::dialect::details::BuildCstrEqForTensorListAlongAxis(
          infer_context, input_values, i);
    }

    return out_dims;
  };

  symbol::ShapeOrDataDimExprs shape_data{
      symbol::TensorShapeOrDataDimExprs(GetOutDimExprs())};

  infer_context->SetShapeOrDataForValue(op->result(0), shape_data);
  return true;
}

bool ReduceInferSymbolicShape(pir::Operation *op,
                              pir::InferSymbolicShapeContext *infer_context) {
  bool keepdim = GetBoolAttr(op, "keepdim");
  auto axis = paddle::dialect::details::GetVectorAttr(op, "axis");
  bool reduce_all = axis.size() == 0 ? true : false;
  return paddle::dialect::details::ReduceInferDim(
      op, infer_context, axis, keepdim, reduce_all);
}

bool ReduceMaxOpInferSymbolicShape(
    pir::Operation *op, pir::InferSymbolicShapeContext *infer_context) {
  return ReduceInferSymbolicShape(op, infer_context);
}

bool ReduceMinOpInferSymbolicShape(
    pir::Operation *op, pir::InferSymbolicShapeContext *infer_context) {
  return ReduceInferSymbolicShape(op, infer_context);
}

bool ReduceProdOpInferSymbolicShape(
    pir::Operation *op, pir::InferSymbolicShapeContext *infer_context) {
  return ReduceInferSymbolicShape(op, infer_context);
}

bool ReduceSumOpInferSymbolicShape(
    pir::Operation *op, pir::InferSymbolicShapeContext *infer_context) {
  return ReduceInferSymbolicShape(op, infer_context);
}

bool ReshapeOpInferSymbolicShape(
    pir::Operation *op, pir::InferSymbolicShapeContext *infer_context) {
  std::vector<int> shape =
      paddle::dialect::details::GetVectorAttr<int>(op, "shape");

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

  const symbol::ShapeOrDataDimExprs &x_dim_expr =
      infer_context->GetShapeOrDataForValue(op->operand_source(0));

  const auto &original_shape = x_dim_expr.shape();

  const auto &out_dims = [&] {
    const auto &numel =
        GetProduct(original_shape, [](const auto &) { return true; });
    const auto &target_shape = [&] {
      std::vector<symbol::DimExpr> target_shape;
      for (size_t i = 0; i < shape.size(); ++i) {
        if (shape[i] == 0) {
          target_shape.emplace_back(original_shape[i]);
        } else {
          target_shape.emplace_back(static_cast<std::int64_t>(shape[i]));
        }
      }
      return target_shape;
    }();

    const auto &product_exclude_minus_one =
        GetProduct(target_shape, IsNotMinusOne);

    std::vector<symbol::DimExpr> out_dims;
    out_dims.reserve(target_shape.size());
    for (size_t i = 0; i < target_shape.size(); ++i) {
      auto out_dim_expr = IsNotMinusOne(target_shape[i])
                              ? target_shape[i]
                              : (numel / product_exclude_minus_one);
      out_dims.emplace_back(out_dim_expr);
    }

    return out_dims;
  }();

  symbol::ShapeOrDataDimExprs shape_data = [&] {
    if (x_dim_expr.data().has_value()) {
      return symbol::TensorShapeOrDataDimExprs(out_dims,
                                               x_dim_expr.data().value());
    }
    return symbol::TensorShapeOrDataDimExprs(out_dims);
  }();

  infer_context->SetShapeOrDataForValue(op->result(0), shape_data);

  return true;
}

bool SliceOpInferSymbolicShape(pir::Operation *op,
                               pir::InferSymbolicShapeContext *infer_context) {
  const std::vector<int64_t> starts_raw =
      paddle::dialect::details::GetVectorAttr(op, "starts");
  const std::vector<int64_t> ends_raw =
      paddle::dialect::details::GetVectorAttr(op, "ends");
  const std::vector<int64_t> axes_raw =
      paddle::dialect::details::GetVectorAttr(op, "axes");
  const std::vector<int64_t> infer_flags_raw =
      paddle::dialect::details::GetVectorAttr(op, "infer_flags");
  const std::vector<int64_t> decrease_axis_raw =
      paddle::dialect::details::GetVectorAttr(op, "decrease_axis");

  const ExprVec starts = paddle::dialect::details::VecInt642Expr(starts_raw);
  const ExprVec ends = paddle::dialect::details::VecInt642Expr(ends_raw);

  infer_context->SetShapeOrDataForValue(
      op->result(0),
      paddle::dialect::slice_utils::SliceRawInferSymbolicShape(
          infer_context->GetShapeOrDataForValue(op->operand_source(0)),
          starts,
          ends,
          axes_raw,
          infer_flags_raw,
          decrease_axis_raw));

  return true;
}

bool GatherOpInferSymbolicShape(pir::Operation *op,
                                pir::InferSymbolicShapeContext *infer_context) {
  const auto &input_shape_or_data =
      infer_context->GetShapeOrDataForValue(op->operand_source(0));
  const auto &index_shape_or_data =
      infer_context->GetShapeOrDataForValue(op->operand_source(1));

  const auto &numel = [&] {
    symbol::DimExpr numel{1};
    for (const auto &dim_expr : index_shape_or_data.shape()) {
      numel = numel * dim_expr;
    }
    return numel;
  }();

  const std::vector<symbol::DimExpr> &input_sym_shape =
      input_shape_or_data.data().has_value()
          ? input_shape_or_data.data().value()
          : input_shape_or_data.shape();

  const std::vector<symbol::DimExpr> &index_sym_shape =
      index_shape_or_data.data().has_value()
          ? index_shape_or_data.data().value()
          : index_shape_or_data.shape();

  int axis = op->attributes().at("axis").dyn_cast<pir::Int32Attribute>().data();
  if (axis < 0) axis += input_sym_shape.size();

  const auto &out_sym_shape = [&] {
    std::vector<symbol::DimExpr> out_sym_shape;

    if (index_sym_shape.size() == 0) {
      if (input_sym_shape.size() == 1) {
        out_sym_shape.push_back(symbol::DimExpr{0});
      } else {
        for (int i = 0; i < axis; ++i) {
          out_sym_shape.push_back(input_sym_shape[i]);
        }
        for (size_t i = axis + 1; i < input_sym_shape.size(); ++i) {
          out_sym_shape.push_back(input_sym_shape[i]);
        }
      }
    } else {
      for (int i = 0; i < axis; ++i) {
        out_sym_shape.push_back(input_sym_shape[i]);
      }
      out_sym_shape.push_back(numel);
      for (size_t i = axis + 1; i < input_sym_shape.size(); ++i) {
        out_sym_shape.push_back(input_sym_shape[i]);
      }
    }
    return out_sym_shape;
  }();

  symbol::ShapeOrDataDimExprs shape_data{
      symbol::TensorShapeOrDataDimExprs(out_sym_shape)};

  pir::Value res = op->result(0);
  infer_context->SetShapeOrDataForValue(res, shape_data);

  return true;
}

}  // namespace cinn::dialect
