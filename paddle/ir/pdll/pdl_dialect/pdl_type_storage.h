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

#include "paddle/ir/core/type.h"
#include "paddle/ir/core/type_base.h"
#include "paddle/ir/core/utils.h"
namespace ir {
namespace pdl {
namespace detail {

struct RangeTypeStorage : public TypeStorage {
  ///
  /// \brief Declare ParamKey according to parameter type.
  ///
  using ParamKey = std::tuple<ir::Type>;

  explicit RangeTypeStorage(Type element_type) : element_type_(element_type) {}

  ///
  /// \brief Each derived TypeStorage must define a Construct method, which
  /// StorageManager uses to construct a derived TypeStorage.
  ///
  static RangeTypeStorage *Construct(ParamKey key) {
    auto element_type = std::get<0>(key);
    return new RangeTypeStorage(element_type);
  }

  ///
  /// \brief Each derived TypeStorage must provide a HashValue method.
  ///
  static std::size_t HashValue(const ParamKey &key) {
    std::size_t hash_value = 0;
    hash_value =
        hash_combine(hash_value, std::hash<ir::Type>()(std::get<0>(key)));
    return hash_value;
  }

  ///
  /// \brief Each derived TypeStorage needs to overload operator==.
  ///
  bool operator==(const ParamKey &key) const {
    return element_type_ == std::get<0>(key);
  }

  ParamKey GetAsKey() const { return ParamKey(element_type_); }

  Type element_type_;
};

}  // namespace detail
}  // namespace pdl
}  // namespace ir
