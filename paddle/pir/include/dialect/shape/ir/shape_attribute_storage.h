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
    return std::hash<ParamKey>()(key);
  }

  bool operator==(const ParamKey &key) const { return data_ == key; }

  ParamKey data() const { return data_; }

 private:
  ParamKey data_;
};

}  // namespace pir::shape
