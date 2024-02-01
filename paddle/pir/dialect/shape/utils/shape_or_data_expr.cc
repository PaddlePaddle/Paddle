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

#include "paddle/pir/dialect/shape/utils/shape_or_data_expr.h"

namespace symbol {

std::ostream& operator<<(std::ostream& stream,
                         const ShapeOrDataDimExprs& shape_or_data) {
  std::string result;
  auto lambdas = Overloaded{
      [&result](const TensorShapeOrDataDimExprs& tensor_shape_data) {
        result += "shape[";
        for (size_t i = 0; i < tensor_shape_data.shape().size(); ++i) {
          result += ToString(tensor_shape_data.shape()[i]);
          if (i < tensor_shape_data.shape().size() - 1) {
            result += ", ";
          }
        }
        result += "]";
        if (tensor_shape_data.data()) {
          result += ", data[";
          for (size_t i = 0; i < tensor_shape_data.data()->size(); ++i) {
            result += ToString(tensor_shape_data.data()->at(i));
            if (i < tensor_shape_data.data()->size() - 1) {
              result += ", ";
            }
          }
          result += "]";
        } else {
          result += ", data[NULL]";
        }
      },
      [&result](const TensorListShapeOrDataDimExprs& tensor_list_shape_data) {
        for (size_t i = 0; i < tensor_list_shape_data.size(); ++i) {
          result += "shape[";
          for (size_t i = 0; i < tensor_list_shape_data[i].shape().size();
               ++i) {
            result += ToString(tensor_list_shape_data[i].shape()[i]);
            if (i < tensor_list_shape_data[i].shape().size() - 1) {
              result += ", ";
            }
          }
          result += "]";
          if (tensor_list_shape_data[i].data()) {
            result += ", data[";
            for (size_t i = 0; i < tensor_list_shape_data[i].data()->size();
                 ++i) {
              result += ToString(tensor_list_shape_data[i].data()->at(i));
              if (i < tensor_list_shape_data[i].data()->size() - 1) {
                result += ", ";
              }
            }
            result += "]";
          } else {
            result += ", data[NULL]";
          }

          if (i < tensor_list_shape_data.size() - 1) {
            result += ", ";
          }
        }
      }};

  std::visit(lambdas, shape_or_data.variant());

  stream << result;

  return stream;
}

}  // namespace symbol
