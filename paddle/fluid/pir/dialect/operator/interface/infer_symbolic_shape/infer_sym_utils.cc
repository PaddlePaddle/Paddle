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

#include "paddle/fluid/pir/dialect/operator/interface/infer_symbolic_shape/infer_sym_utils.h"

namespace paddle::dialect::details {

std::optional<std::vector<int64_t>> VecExpr2Int64(const ExprVec &expr_vec) {
  std::vector<int64_t> int64vec;
  for (auto item : expr_vec) {
    if (!item.isa<int64_t>()) {
      return std::nullopt;
    }
    int64vec.push_back(item.Get<int64_t>());
  }
  return int64vec;
}

ExprVec VecInt642Expr(const std::vector<int64_t> &int_vec) {
  ExprVec expr_vec(int_vec.size(), 0);
  std::transform(
      int_vec.begin(),
      int_vec.end(),
      expr_vec.begin(),
      [](int64_t val) -> symbol::DimExpr { return symbol::DimExpr(val); });
  return expr_vec;
}

bool ReduceInferDim(pir::Operation *op,
                    pir::InferSymbolicShapeContext *infer_context,
                    const std::vector<int64_t> &axis,
                    bool keep_dim,
                    bool reduce_all) {
  auto x = op->operand_source(0);
  int x_rank = x.type().dyn_cast<pir::DenseTensorType>().dims().size();

  const std::vector<int64_t> formatted_axis = [&] {
    std::vector<int64_t> formatted_axis = axis;
    for (size_t i = 0; i < axis.size(); ++i) {
      if (axis[i] < 0) {
        formatted_axis[i] = axis[i] + x_rank;
      }
    }
    return formatted_axis;
  }();

  bool full_dim = true;
  std::set<int64_t> dims_set(formatted_axis.begin(), formatted_axis.end());
  for (int64_t i = 0; i < x_rank; ++i) {
    if (dims_set.find(i) == dims_set.end()) {
      full_dim = false;
      break;
    }
  }
  bool empty_dim = axis.size() == 0;
  reduce_all = reduce_all || full_dim || empty_dim;

  const symbol::ShapeOrDataDimExprs &x_shape_or_data =
      infer_context->GetShapeOrDataForValue(x);
  std::vector<symbol::DimExpr> input_shapes;
  if (x_shape_or_data.data() == std::nullopt ||
      x_shape_or_data.data()->empty()) {
    input_shapes = x_shape_or_data.shape();
  } else {
    input_shapes = *x_shape_or_data.data();
  }

  const std::vector<symbol::DimExpr> shapes = [&] {
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
    return shapes;
  }();

  symbol::ShapeOrDataDimExprs shape_data{
      symbol::TensorShapeOrDataDimExprs(shapes)};

  infer_context->SetShapeOrDataForValue(op->result(0), shape_data);
  return true;
}

symbol::ShapeOrDataDimExprs CreateShapeOrDataForXShape(
    const symbol::ShapeOrDataDimExprs &x_dim_exprs) {
  const auto InsertZeros =
      [](const std::vector<symbol::DimExpr> &dims) -> decltype(auto) {
    auto out_dims = dims;
    out_dims.insert(out_dims.begin(), 0);
    return out_dims;
  };
  const auto &x_dims = x_dim_exprs.shape();
  return symbol::TensorShapeOrDataDimExprs(InsertZeros(x_dims));
}

void BuildCstrEqForTensorListAlongAxis(
    pir::InferSymbolicShapeContext *infer_context,
    const symbol::TensorListShapeOrDataDimExprs &shape_data_list,
    int axis) {
  for (size_t i = 1; i < shape_data_list.size(); ++i) {
    infer_context->AddEqualCstr(shape_data_list[0].shape()[axis],
                                shape_data_list[i].shape()[axis]);
  }
}

void BuildCstrEqForTensorListAlongAxis(
    pir::InferSymbolicShapeContext *infer_context,
    const std::vector<pir::Value> &values,
    int axis) {
  for (size_t i = 1; i < values.size(); ++i) {
    infer_context->AddEqualCstr(
        infer_context->GetShapeOrDataForValue(values[0]).shape()[axis],
        infer_context->GetShapeOrDataForValue(values[i]).shape()[axis]);
  }
}

}  // namespace paddle::dialect::details
