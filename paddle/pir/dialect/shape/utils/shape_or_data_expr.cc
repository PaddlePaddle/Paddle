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
  auto lambdas = Overloaded{
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
      }};

  std::visit(lambdas, shape_or_data.variant());

  return stream;
}

}  // namespace symbol
