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

#include <type_traits>

#include "paddle/ir/attribute.h"

namespace ir {
///
/// \brief Define Parameteric AttributeStorage for StringAttribute.
///
/// NOTE(zhangbo9674): The derived AttributeStorage class needs to implement the
/// following methods: (1)declare ParamKey, (2)define Construction method,
/// (3)define HashValue method, (4)overload operator==.
///
struct StringAttributeStorage : public ir::AttributeStorage {
  ///
  /// \brief Declare ParamKey according to parameter type.
  ///
  using ParamKey = std::tuple<char *, uint32_t>;

  StringAttributeStorage(char *data, uint32_t size)
      : data_(data), size_(size) {}

  ~StringAttributeStorage() { free(data_); }

  ///
  /// \brief Each derived AttributeStorage must define a Construc method, which
  /// StorageManager uses to construct a derived AttributeStorage.
  ///
  static StringAttributeStorage *Construct(ParamKey key) {
    uint32_t size = std::get<1>(key);
    char *data = reinterpret_cast<char *>(malloc(size));
    memcpy(data, std::get<0>(key), size);
    return new StringAttributeStorage(data, size);
  }

  ///
  /// \brief Each derived AttributeStorage must provide a HashValue method.
  ///
  static std::size_t HashValue(const ParamKey &key) {
    std::size_t hash_value = 0;
    hash_value = hash_combine(hash_value,
                              std::hash<std::string>()(std::string(
                                  std::get<0>(key), std::get<1>(key))));
    return hash_value;
  }

  ///
  /// \brief Each derived TypeStorage needs to overload operator==.
  ///
  bool operator==(const ParamKey &key) const {
    return std::string(std::get<0>(key), std::get<1>(key)) ==
           std::string(data_, size_);
  }

  ParamKey GetAsKey() const { return ParamKey(data_, size_); }

  ///
  /// \brief StringAttributeStorage include two parameters: data, size,
  /// layout, lod, offset.
  ///
  char *data_;
  uint32_t size_;

 private:
  static std::size_t hash_combine(std::size_t lhs, std::size_t rhs) {
    return lhs ^= rhs + 0x9e3779b9 + (lhs << 6) + (lhs >> 2);
  }
};

}  // namespace ir
