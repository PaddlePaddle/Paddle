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

#pragma once

#include <algorithm>
#include <map>
#include <type_traits>

#include "paddle/common/enforce.h"
#include "paddle/pir/include/core/attribute.h"
#include "paddle/pir/include/core/attribute_base.h"
#include "paddle/pir/include/core/type.h"
#include "paddle/pir/include/core/utils.h"
#include "paddle/pir/include/dialect/shape/utils/shape_or_data_expr.h"

namespace pir::shape {

inline std::vector<symbol::DimExpr> GetExprVecFromData(
    const symbol::ShapeOrDataDimExprs &shapeordata) {
  if (shapeordata.isa<symbol::TensorListShapeOrDataDimExprs>()) {
    std::vector<symbol::DimExpr> result;
    symbol::TensorListShapeOrDataDimExprs list =
        shapeordata.dyn_cast<symbol::TensorListShapeOrDataDimExprs>();
    for (size_t i = 0; i < list.size(); i++) {
      if (list[i].data().has_value()) {
        for (auto expr : list[i].data().value()) {
          result.emplace_back(expr);
        }
      }
    }
    return result;
  } else {
    return shapeordata.data().has_value() ? shapeordata.data().value()
                                          : std::vector<symbol::DimExpr>{};
  }
}

inline std::vector<symbol::DimExpr> GetExprVecFromShape(
    const symbol::ShapeOrDataDimExprs &shapeordata) {
  const auto GetShapeExprsFromList = [&]() {
    std::vector<symbol::DimExpr> result;
    symbol::TensorListShapeOrDataDimExprs list =
        shapeordata.dyn_cast<symbol::TensorListShapeOrDataDimExprs>();
    for (size_t i = 0; i < list.size(); i++) {
      for (auto expr : list[i].shape()) {
        result.emplace_back(expr);
      }
    }
    return result;
  };
  if (shapeordata.isa<symbol::TensorListShapeOrDataDimExprs>()) {
    return GetShapeExprsFromList();
  } else {
    return shapeordata.shape();
  }
}

///
/// \brief Define Parametric AttributeStorage for SymbolAttribute.
///
struct SymbolAttributeStorage : public AttributeStorage {
  using ParamKey = symbol::ShapeOrDataDimExprs;

  explicit SymbolAttributeStorage(const ParamKey &key) : data_(key) {}

  static SymbolAttributeStorage *Construct(const ParamKey &key) {
    return new SymbolAttributeStorage(key);
  }

  static std::size_t HashValue(const ParamKey &key) {
    auto all_shapes = GetExprVecFromShape(key);
    std::size_t hash_value = 0;
    for (size_t i = 0; i < all_shapes.size(); ++i) {
      hash_value = detail::hash_combine(
          hash_value,
          std::hash<std::string>()(symbol::ToString(all_shapes[i])));
    }
    auto all_datas = GetExprVecFromData(key);
    for (size_t i = 0; i < all_datas.size(); ++i) {
      hash_value = detail::hash_combine(
          hash_value, std::hash<std::string>()(symbol::ToString(all_datas[i])));
    }

    return hash_value;
  }

  bool operator==(const ParamKey &key) const {
    return GetExprVecFromShape(data_) == GetExprVecFromShape(key) &&
           GetExprVecFromData(data_) == GetExprVecFromData(key);
  }

  ParamKey data() const { return data_; }

 private:
  ParamKey data_;
};

}  // namespace pir::shape
