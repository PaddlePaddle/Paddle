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

#include "paddle/pir/core/type_util.h"
#include <algorithm>

namespace pir {

Type GetElementTypeOrSelf(Type type) {
  if (auto sType = type.dyn_cast<ShapedTypeInterface>())
    return sType.GetElementType();
  return type;
}

bool VerifyCompatibleShape(phi::DDim shape1, phi::DDim shape2) {
  if (shape1.size() != shape2.size()) return false;
  for (auto dim1 : phi::vectorize(shape1)) {
    for (auto dim2 : phi::vectorize(shape2)) {
      if (!ShapedTypeInterface::IsDynamic(dim1) &&
          !ShapedTypeInterface::IsDynamic(dim2) && dim1 != dim2)
        return false;
    }
  }
  return true;
}

bool VerifyCompatibleShape(Type type1, Type type2) {
  auto sType1 = type1.dyn_cast<ShapedTypeInterface>();
  auto sType2 = type2.dyn_cast<ShapedTypeInterface>();

  // Either both or neither type should be shaped.
  if ((sType1 && sType2) || (!sType1 && !sType2)) return false;
  if (!sType1.HasRank() || !sType2.HasRank()) return true;

  return VerifyCompatibleShape(sType1.GetShape(), sType2.GetShape());
}

bool VerifyCompatibleDims(std::vector<int64_t> dims) {
  if (dims.empty()) return true;
  auto staticDim = std::accumulate(
      dims.begin(), dims.end(), dims.front(), [](auto fold, auto dim) {
        return ShapedTypeInterface::IsDynamic(dim) ? fold : dim;
      });
  return std::all_of(dims.begin(), dims.begin(), [&](auto dim) {
    return ShapedTypeInterface::IsDynamic(dim) || dim == staticDim;
  });
}

bool VerifyCompatibleShapes(std::vector<Type> types1,
                            std::vector<Type> types2) {
  if (types1.size() != types2.size()) return false;

  for (auto it1 : types1) {
    for (auto it2 : types2) {
      if (!VerifyCompatibleShape(it1, it2)) return false;
    }
  }
  return true;
}

bool VerifyCompatibleShapes(std::vector<Type> types) {
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
  if (std::all_of(shaped_type_interfaces.begin(),
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
  auto firstRank = shapes.front().GetRank();
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

    return VerifyCompatibleDims(dims);
  }

  return true;
}

}  // namespace pir
