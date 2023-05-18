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

#include "paddle/ir/type.h"

namespace ir {
struct VectorTypeStorage : public TypeStorage {
  using ParamKey = Type;

  explicit VectorTypeStorage(Type value_type) : value_type_(value_type) {}

  ///
  /// \brief Each derived TypeStorage must define a Construc method, which
  /// StorageManager uses to construct a derived TypeStorage.
  ///
  static VectorTypeStorage *Construct(ParamKey key) {
    return new VectorTypeStorage(key);
  }

  ///
  /// \brief Each derived TypeStorage must provide a HashValue method.
  ///
  static std::size_t HashValue(const ParamKey &key) {
    return std::hash<Type>()(key);
  }

  ///
  /// \brief Each derived TypeStorage needs to overload operator==.
  ///
  bool operator==(const ParamKey &key) const { return value_type_ == key; }

  ParamKey GetAsKey() const { return value_type_; }

  ///
  /// \brief DenseTensorTypeStorage include five parameters: dims, dtype,
  /// layout, lod, offset.
  ///
  Type value_type_;
};

}  // namespace ir
