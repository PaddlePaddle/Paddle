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

#include <algorithm>
#include "paddle/pir/include/core/type_utils.h"

namespace pir {

Type GetElementTypeOrSelf(Type type) {
  if (auto sType = type.dyn_cast<ShapedTypeInterface>())
    return sType.GetElementType();
  return type;
}

bool VerifyCompatibleShape(const pir::DDim &lhs_shape,
                           const pir::DDim &rhs_shape) {
  if (lhs_shape.size() != rhs_shape.size()) return false;

  for (auto dim1 : common::vectorize(lhs_shape)) {
    for (auto dim2 : common::vectorize(rhs_shape)) {
      if (!ShapedTypeInterface::IsDynamic(dim1) &&
          !ShapedTypeInterface::IsDynamic(dim2) && dim1 != dim2)
        return false;
    }
  }
  return true;
}

bool VerifyCompatibleShape(Type lhs_type, Type rhs_type) {
  auto lhs_shaped_type = lhs_type.dyn_cast<ShapedTypeInterface>();
  auto rhs_shaped_type = rhs_type.dyn_cast<ShapedTypeInterface>();

  // Either both or neither type should be shaped.
  if (!lhs_shaped_type) return !rhs_shaped_type;
  if (!rhs_shaped_type) return false;

  if (!lhs_shaped_type.HasRank() || !rhs_shaped_type.HasRank()) return true;

  return VerifyCompatibleShape(lhs_shaped_type.GetShape(),
                               rhs_shaped_type.GetShape());
}

bool VerifyCompatibleDims(const std::vector<int64_t> &dims) {
  if (dims.empty()) return true;
  auto static_dim = std::accumulate(
      dims.begin(), dims.end(), dims.front(), [](auto &fold, auto &dim) {
        return ShapedTypeInterface::IsDynamic(dim) ? fold : dim;
      });
  return std::all_of(dims.begin(), dims.begin(), [&](auto dim) {
    return ShapedTypeInterface::IsDynamic(dim) || dim == static_dim;
  });
}

bool VerifyCompatibleShapes(const std::vector<Type> &lhs_types,
                            const std::vector<Type> &rhs_types) {
  if (lhs_types.size() != rhs_types.size()) return false;

  for (auto it1 : lhs_types) {
    for (auto it2 : rhs_types) {
      if (!VerifyCompatibleShape(it1, it2)) return false;
    }
  }
  return true;
}

bool VerifyCompatibleShapes(const std::vector<Type> &types) {
  std::vector<ShapedTypeInterface> shaped_type_interfaces;

  std::for_each(
      types.begin(), types.end(), [&shaped_type_interfaces](Type type) {
        shaped_type_interfaces.push_back(type.dyn_cast<ShapedTypeInterface>());
      });

  // Return false if some, but not all are not shaped. Return early if none
  // are shaped also.
  if (std::none_of(shaped_type_interfaces.begin(),
                   shaped_type_interfaces.end(),
                   [](auto t) { return t; }))
    return true;

  if (!std::all_of(shaped_type_interfaces.begin(),
                   shaped_type_interfaces.end(),
                   [](auto t) { return t; }))
    return false;

  // Remove all unranked shapes
  std::vector<ShapedTypeInterface> shapes;

  std::for_each(shaped_type_interfaces.begin(),
                shaped_type_interfaces.end(),
                [&shapes](ShapedTypeInterface type) {
                  if (type.HasRank())
                    shapes.push_back(type.dyn_cast<ShapedTypeInterface>());
                });
  if (shapes.empty()) return true;

  // All ranks should be equal
  int64_t firstRank = shapes.front().GetRank();

  if (std::any_of(shapes.begin(), shapes.end(), [&](auto shape) {
        return firstRank != shape.GetRank();
      }))
    return false;

  for (unsigned i = 0; i < firstRank; ++i) {
    // For all ranked dimensions
    std::vector<int64_t> dims;
    std::for_each(shapes.begin(), shapes.end(), [&](ShapedTypeInterface shape) {
      dims.push_back(shape.GetDimSize(i));
    });

    if (!VerifyCompatibleDims(dims)) return false;
  }

  return true;
}

}  // namespace pir
