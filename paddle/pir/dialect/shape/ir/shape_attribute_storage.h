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
#include "paddle/pir/core/attribute.h"
#include "paddle/pir/core/attribute_base.h"
#include "paddle/pir/core/type.h"
#include "paddle/pir/core/utils.h"
#include "paddle/pir/dialect/shape/utils/dim_expr.h"

namespace pir::shape {

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
    std::size_t hash_value = 0;
    for (size_t i = 0; i < key.shape().size(); ++i) {
      hash_value = hash_combine(
          hash_value,
          std::hash<std::string>()(symbol::ToString(key.shape()[i])));
    }
    if (key.data().has_value()) {
      for (size_t i = 0; i < key.data().value().size(); ++i) {
        hash_value = hash_combine(
            hash_value,
            std::hash<std::string>()(symbol::ToString(key.data().value()[i])));
      }
    }

    return hash_value;
  }

  bool operator==(const ParamKey &key) const {
    return data_.shape() == key.shape() && data_.data() == key.data();
  }

  ParamKey data() const { return data_; }

 private:
  ParamKey data_;
};

}  // namespace pir::shape
