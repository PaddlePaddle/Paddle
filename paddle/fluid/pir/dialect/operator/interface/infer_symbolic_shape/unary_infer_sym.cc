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
  // auto x_dims = x.dims();
  // int dim1_ = dim1 < 0 ? x_dims.size() + dim1 + 1 : dim1;
  // int dim2_ = dim2 < 0 ? x_dims.size() + dim2 + 1 : dim2;
  // int offset_ = std::abs(offset);
  // int new_dim_len = static_cast<int>(offset_ + x_dims[x_dims.size() - 1]);
  // auto sizes = common::vectorize(x_dims);
  // sizes.pop_back();
  // sizes.insert(sizes.begin() + std::min(dim1_, dim2_), new_dim_len);
  // sizes.insert(sizes.begin() + std::max(dim1_, dim2_), new_dim_len);
  // out->set_dims(common::make_ddim(sizes));
  return true;
}
bool DiagonalOpInferSymbolicShape(
    pir::Operation *op, pir::ShapeConstraintIRAnalysis *shape_analysis) {
  // auto x_dims = input.dims();
  // int offset_ = offset;
  // int axis1_ = axis1 < 0 ? x_dims.size() + axis1 : axis1;
  // int axis2_ = axis2 < 0 ? x_dims.size() + axis2 : axis2;

  // auto out_dims = common::vectorize(x_dims);
  // // from out_dims get the dim size of axis1_.
  // auto axis1_size = out_dims[axis1_];
  // auto axis2_size = out_dims[axis2_];
  // // delete two dims by attr axis1 and axis2 from out_dims.
  // /* example:
  //    out_dim = [2, 3, 4];
  //    axis1 = 0;
  //    axis2 = 1;
  //    according to the attr of axis1 and axis2, we get:
  //    out_dim = [4].
  // */
  // out_dims.erase(out_dims.begin() + std::max(axis1_, axis2_));
  // out_dims.erase(out_dims.begin() + std::min(axis1_, axis2_));

  // if (offset_ == 0) {
  //   out_dims.push_back(std::min(axis1_size, axis2_size));
  // } else if (offset_ > 0) {
  //   if ((axis2_size - offset_) > 0) {
  //     out_dims.push_back(std::min(axis1_size, axis2_size - offset_));
  //   } else {
  //     out_dims.push_back(0);
  //   }
  // } else {
  //   if ((axis1_size + offset_) > 0) {
  //     out_dims.push_back(std::min(axis1_size + offset_, axis2_size));
  //   } else {
  //     out_dims.push_back(0);
  //   }
  // }
  // out->set_dims(common::make_ddim(out_dims));
  return true;
}
bool DirichletOpInferSymbolicShape(
    pir::Operation *op, pir::ShapeConstraintIRAnalysis *shape_analysis) {
  // const auto alpha_dim = alpha.dims();
  // out->set_dims(alpha_dim);
  return true;
}
bool EinsumOpInferSymbolicShape(
    pir::Operation *op, pir::ShapeConstraintIRAnalysis *shape_analysis) {
  // need some xxx
  // out->set_dims(common::make_ddim(output_dims));

  // for (size_t i = 0; i < xshape.size(); ++i) {
  //   if (xshape[i] != nullptr) {
  //     xshape[i]->set_dims(inputs[i]->dims());
  //   }
  // }
  return true;
}

bool KthvalueOpInferSymbolicShape(
    pir::Operation *op, pir::ShapeConstraintIRAnalysis *shape_analysis) {
  // auto input_dims = x.dims();
  // const int &dim_size = input_dims.size();
  // if (axis < 0) axis += dim_size;
  // std::vector<int64_t> dimvec;
  // for (int i = 0; i < axis; i++) {
  //   dimvec.emplace_back(input_dims[i]);
  // }
  // if (keepdim && dim_size > 0) {
  //   dimvec.emplace_back(static_cast<int64_t>(1));
  // }
  // for (int i = axis + 1; i < dim_size; i++) {
  //   dimvec.emplace_back(input_dims[i]);
  // }
  // DDim dims = common::make_ddim(dimvec);
  // out->set_dims(dims);
  // out->share_lod(x);
  // out->set_dtype(x.dtype());
  // indices->set_dims(dims);
  // indices->share_lod(x);
  // indices->set_dtype(x.dtype());
  return true;
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

    const auto &product_exclude_minus_one =
        GetProduct(operand_shape_or_data.data().value(), IsNotMinusOne);

    const auto &input_dims = operand_shape_or_data.data().value();

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
  shape_analysis->SetShapeOrDataForValue(
      op->result(1),
      shape_analysis->GetShapeOrDataForValue(operand_source_shape));
  return true;
}

bool Reshape_OpInferSymbolicShape(
    pir::Operation *op, pir::ShapeConstraintIRAnalysis *shape_analysis) {
  return ReshapeOpInferSymbolicShape(op, shape_analysis);
}

}  // namespace paddle::dialect
