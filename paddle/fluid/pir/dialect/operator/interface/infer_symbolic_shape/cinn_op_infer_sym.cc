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
#include "paddle/fluid/pir/dialect/operator/interface/infer_symbolic_shape/unary_infer_sym.cc"

namespace cinn::dialect {

bool BroadcastOpInferSymbolicShape(
    pir::Operation *op, pir::InferSymbolicShapeContext *infer_context) {
  const std::vector<symbol::DimExpr> &out_dims = [&] {
    const std::vector<int64_t> &shape =
        paddle::dialect::details::GetVectorAttr<int64_t>(op, "out_shape");
    std::vector<symbol::DimExpr> out_dims;
    for (int64_t dim : shape) {
      out_dims.emplace_back(dim);
    }
    return out_dims;
  }();

  const auto &DealWithMinusOneAndSetOutput =
      [&](const std::vector<symbol::DimExpr> &expand_shape) {
        std::vector<symbol::DimExpr> result_shape(expand_shape);
        const auto &x_shape_or_data =
            infer_context->GetShapeOrDataForValue(op->operand_source(0));
        const std::vector<symbol::DimExpr> &x_dims = x_shape_or_data.shape();
        for (size_t i = 0; i < expand_shape.size(); i++) {
          if (expand_shape[i] == symbol::DimExpr{-1}) {  // copy the dim from x
            // the shape is right aligned
            int index = i - (expand_shape.size() - x_dims.size());
            PADDLE_ENFORCE_GE(index,
                              0,
                              common::errors::InvalidArgument(
                                  "in ExpandOpInferSymbolicShape, "
                                  "the dim to copy must >= 0, "
                                  "but got %d",
                                  index));

            result_shape[i] = x_dims[index];
          }
        }

        infer_context->SetShapeOrDataForValue(
            op->result(0),
            symbol::ShapeOrDataDimExprs{
                symbol::TensorShapeOrDataDimExprs(result_shape)});
      };
  DealWithMinusOneAndSetOutput(out_dims);
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

  int axis = op->attributes().at("axis").dyn_cast<pir::Int32Attribute>().data();

  const auto &out_shape_dim_expr = [&]() -> std::vector<symbol::DimExpr> {
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
  }();

  if (IsAllDataValue() && out_shape_dim_expr.size() == 1) {
    std::vector<symbol::DimExpr> out_data;
    for (const auto &value : input_values) {
      const auto &shape_or_data = infer_context->GetShapeOrDataForValue(value);
      for (size_t i = 0; i < shape_or_data.data().value().size(); ++i) {
        out_data.emplace_back(shape_or_data.data().value()[i]);
      }
    }
    symbol::ShapeOrDataDimExprs shape_data{
        symbol::TensorShapeOrDataDimExprs(out_shape_dim_expr, out_data)};

    pir::Value res = op->result(0);
    infer_context->SetShapeOrDataForValue(res, shape_data);
    return true;
  }

  symbol::ShapeOrDataDimExprs shape_data{
      symbol::TensorShapeOrDataDimExprs(out_shape_dim_expr)};

  infer_context->SetShapeOrDataForValue(op->result(0), shape_data);
  return true;
}

bool SplitOpInferSymbolicShape(pir::Operation *op,
                               pir::InferSymbolicShapeContext *infer_context) {
  const auto &x_shape_or_data =
      infer_context->GetShapeOrDataForValue(op->operand_source(0));
  PADDLE_ENFORCE_EQ(x_shape_or_data.data().has_value(),
                    false,
                    common::errors::InvalidArgument(
                        "InferSymbolicShape of SplitOp only support input with "
                        "value now."));
  const auto &x_dims_sym = x_shape_or_data.shape();

  // axis
  int64_t axis = static_cast<int64_t>(
      op->attribute("axis").dyn_cast<pir::Int32Attribute>().data());
  size_t rank = x_dims_sym.size();
  axis = axis >= 0 ? axis : std::max(int64_t(0), int64_t(axis + rank));

  // sections
  auto sections_array = op->attribute("num_or_sections")
                            .dyn_cast<pir::ArrayAttribute>()
                            .AsVector();
  std::vector<symbol::DimExpr> sections_sym;
  if (sections_array.size() > 0) {
    PADDLE_ENFORCE_EQ(
        sections_array[0].isa<pir::Int32Attribute>(),
        true,
        common::errors::PreconditionNotMet(
            "Element in sections_array MUST be pir::Int64Attribute "));

    for (size_t i = 0; i < sections_array.size(); ++i) {
      sections_sym.push_back(
          sections_array[i].dyn_cast<pir::Int32Attribute>().data());
    }
  }

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
      infer_context->AddEqualCstr(x_dims_sym.at(axis), sum_exclude_minus_one);
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
      const auto &section_sym = sections_sym.at(idx);
      output_dims_sym.at(axis) =
          IsNotMinusOne(section_sym)
              ? section_sym
              : x_dims_sym.at(axis) - sum_exclude_minus_one;

      shape_data_list.push_back(
          symbol::TensorShapeOrDataDimExprs(output_dims_sym));
    }
    return shape_data_list;
  }();

  infer_context->SetShapeOrDataForValue(
      op->result(0), symbol::ShapeOrDataDimExprs{output_shape_data_list});

  return true;
}

bool Pool2dOpInferSymbolicShape(pir::Operation *op,
                                pir::InferSymbolicShapeContext *infer_context) {
  const auto &kernel_size_shape_or_data =
      infer_context->GetShapeOrDataForValue(op->operand_source(1));
  const auto &kernel_size =
      paddle::dialect::details::GetExprVecFromData(kernel_size_shape_or_data);
  infer_context->SetShapeOrDataForValue(
      op->result(0),
      Pool2dRawInferSymbolicShape(op, kernel_size, infer_context));
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
          op->operand_source(0),
          op->result(0),
          starts,
          ends,
          axes_raw,
          infer_flags_raw,
          decrease_axis_raw,
          infer_context));

  return true;
}

bool UniformRandomOpInferSymbolicShape(
    pir::Operation *op, pir::InferSymbolicShapeContext *infer_context) {
  const std::vector<int64_t> shape =
      paddle::dialect::details::GetVectorAttr<int64_t>(op, "shape");

  const std::vector<symbol::DimExpr> out_dims = [&] {
    std::vector<symbol::DimExpr> out_dims;
    for (int64_t dim : shape) {
      out_dims.emplace_back(symbol::DimExpr{dim});
    }
    return out_dims;
  }();

  infer_context->SetShapeOrDataForValue(
      op->result(0),
      symbol::ShapeOrDataDimExprs{symbol::TensorShapeOrDataDimExprs(out_dims)});

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
