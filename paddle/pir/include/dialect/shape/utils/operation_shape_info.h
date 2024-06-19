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

#pragma once
#include <ostream>
#include "paddle/pir/include/core/attribute.h"
#include "paddle/pir/include/core/operation_utils.h"
#include "paddle/pir/include/dialect/shape/utils/shape_or_data_expr.h"

namespace pir {
/**
 * @class OperationShapeInfo
 * @brief This class represents information needed to determine the output shape
 * of an operator.
 *
 * The OperationShapeInfo class encapsulates the necessary details to identify
 * and manage the output shape of an operator in computational graphs or similar
 * structures. It includes the operator's name, input shapes, and attributes.
 *
 * Usage:
 * This class can be used in scenarios where the output shape of computational
 * operators needs to be determined. It is particularly useful when
 * InferSymbolicShapeInterface of an operator is not defined.
 */
class OperationShapeInfo {
 public:
  OperationShapeInfo(
      const Operation &op,
      const std::vector<symbol::ShapeOrDataDimExprs> &input_shape_or_datas);
  OperationShapeInfo(
      const std::string &op_name,
      const std::vector<symbol::ShapeOrDataDimExprs> &input_shape_or_datas,
      const AttributeMap &attributes);
  bool operator==(const OperationShapeInfo &other) const;
  std::size_t hash() const;

  friend std::ostream &operator<<(std::ostream &os,
                                  const OperationShapeInfo &info);

 private:
  std::string op_name_;
  std::vector<symbol::ShapeOrDataDimExprs> input_shape_or_datas_;
  std::vector<std::pair<std::string, ::pir::Attribute>> attributes_;
};

}  // namespace pir

namespace std {

template <>
struct hash<pir::OperationShapeInfo> {
  std::size_t operator()(const pir::OperationShapeInfo &obj) const {
    return obj.hash();
  }
};

}  // namespace std
