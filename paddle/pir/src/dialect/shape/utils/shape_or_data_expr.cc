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

#include "paddle/pir/include/dialect/shape/utils/shape_or_data_expr.h"

namespace symbol {
std::vector<DimExpr> SubstituteDimExprVector(
    const std::vector<DimExpr>& original_dim_expr,
    const std::unordered_map<DimExpr, DimExpr>& substitution_pattern) {
  std::vector<DimExpr> substituted_dim_expr{};
  for (const DimExpr& dim_expr : original_dim_expr) {
    const auto& tmp_dim_expr =
        SubstituteDimExpr(dim_expr, substitution_pattern);
    substituted_dim_expr.push_back(SimplifyDimExpr(tmp_dim_expr));
  }
  return substituted_dim_expr;
}

TensorShapeOrDataDimExprs SubstituteTensorShapeOrData(
    const TensorShapeOrDataDimExprs& shape_or_data,
    const std::unordered_map<DimExpr, DimExpr>& substitution_pattern) {
  std::vector<DimExpr> substituted_shape =
      SubstituteDimExprVector(shape_or_data.shape(), substitution_pattern);
  if (!shape_or_data.data().has_value()) {
    return ShapeOrData<DimExpr>(substituted_shape);
  } else {
    std::vector<DimExpr> substituted_data = SubstituteDimExprVector(
        shape_or_data.data().value(), substitution_pattern);
    return ShapeOrData<DimExpr>(substituted_shape, substituted_data);
  }
}

ShapeOrDataDimExprs SubstituteShapeOrData(
    const ShapeOrDataDimExprs& shape_or_data,
    const std::unordered_map<DimExpr, DimExpr>& substitution_pattern) {
  auto lambdas = common::Overloaded{
      [&](const TensorShapeOrDataDimExprs& tensor_shape_or_data) {
        return ShapeOrDataDimExprs(SubstituteTensorShapeOrData(
            tensor_shape_or_data, substitution_pattern));
      },
      [&](const TensorListShapeOrDataDimExprs& tensor_list) {
        TensorListShapeOrDataDimExprs substituted_tensor_list;
        for (const TensorShapeOrDataDimExprs& tensor_shape_or_data :
             tensor_list) {
          substituted_tensor_list.push_back(SubstituteTensorShapeOrData(
              tensor_shape_or_data, substitution_pattern));
        }
        return ShapeOrDataDimExprs(substituted_tensor_list);
      },
      [&](const RankedTensorArrayShapeOrDataDimExprs& tensor_array) {
        RankedTensorArrayShapeOrDataDimExprs substituted_tensor_array(
            SubstituteDimExprVector(tensor_array.GetShapeHint(),
                                    substitution_pattern));
        return ShapeOrDataDimExprs(substituted_tensor_array);
      },
      [&](const NullShapeOrDataDimExpr& null_shape_or_data) {
        return ShapeOrDataDimExprs(null_shape_or_data);
      }};
  return std::visit(lambdas, shape_or_data.variant());
}

std::ostream& operator<<(std::ostream& stream,
                         const ShapeOrDataDimExprs& shape_or_data) {
  auto lambdas = common::Overloaded{
      [&](const TensorShapeOrDataDimExprs& tensor_shape_data) {
        stream << "shape" << tensor_shape_data.shape();
        if (tensor_shape_data.data()) {
          stream << ", data" << tensor_shape_data.data().value();
        } else {
          stream << ", data[NULL]";
        }
      },
      [&](const TensorListShapeOrDataDimExprs& tensor_list_shape_data) {
        for (size_t i = 0; i < tensor_list_shape_data.size(); ++i) {
          stream << "shape" << tensor_list_shape_data[i].shape();
          if (tensor_list_shape_data[i].data()) {
            stream << ", data" << tensor_list_shape_data[i].data().value();
          } else {
            stream << ", data[NULL]";
          }
          if (i < tensor_list_shape_data.size() - 1) {
            stream << ", ";
          }
        }
      },
      [&](const RankedTensorArrayShapeOrDataDimExprs& tensor_array_shape_data) {
        stream << "TensorArray with shape hint: "
               << tensor_array_shape_data.GetShapeHint();
      },
      [&](const NullShapeOrDataDimExpr& null_shape_data) {
        stream << "shape[NULL], data[NULL]";
      }};

  std::visit(lambdas, shape_or_data.variant());

  return stream;
}

}  // namespace symbol
