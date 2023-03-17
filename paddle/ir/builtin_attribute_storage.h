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
#include <type_traits>

#include "paddle/ir/attribute.h"

namespace ir {
///
/// \brief Define Parameteric AttributeStorage for StrAttribute.
///
struct StrAttributeStorage : public ir::AttributeStorage {
  ///
  /// \brief Declare ParamKey according to parameter type.
  ///
  using ParamKey = std::tuple<char *, uint32_t>;

  StrAttributeStorage(char *data, uint32_t size) : data_(data), size_(size) {}

  ~StrAttributeStorage() { free(data_); }

  ///
  /// \brief Define a Construc method, which StorageManager uses to construct a
  /// derived AttributeStorage.
  ///
  static StrAttributeStorage *Construct(ParamKey key) {
    uint32_t size = std::get<1>(key);
    char *data = reinterpret_cast<char *>(malloc(size));
    memcpy(data, std::get<0>(key), size);
    return new StrAttributeStorage(data, size);
  }

  ///
  /// \brief Provide a HashValue method.
  ///
  static std::size_t HashValue(const ParamKey &key) {
    return hash_combine(0,
                        std::hash<std::string>()(
                            std::string(std::get<0>(key), std::get<1>(key))));
  }

  bool operator==(const ParamKey &key) const {
    return std::equal(data_, data_ + size_, std::get<0>(key));
  }

  ParamKey GetAsKey() const { return ParamKey(data_, size_); }

  ///
  /// \brief StrAttributeStorage include two parameters: data, size.
  ///
  char *data_;
  uint32_t size_;

 private:
  static std::size_t hash_combine(std::size_t lhs, std::size_t rhs) {
    return lhs ^= rhs + 0x9e3779b9 + (lhs << 6) + (lhs >> 2);
  }
};

///
/// \brief Define Parameteric AttributeStorage for DictionaryAttributeStorage.
///
struct DictionaryAttributeStorage : public AttributeStorage {
  using ParamKey = std::tuple<NamedAttribute *, uint32_t>;

  DictionaryAttributeStorage(NamedAttribute *data, uint32_t size)
      : data_(data), size_(size) {}

  ~DictionaryAttributeStorage() { free(data_); }

  static DictionaryAttributeStorage *Construct(ParamKey key) {
    auto value = std::get<0>(key);
    uint32_t size = std::get<1>(key);
    std::sort(value, value + size);
    NamedAttribute *data = reinterpret_cast<NamedAttribute *>(
        malloc(sizeof(NamedAttribute) * size));
    memcpy(data, value, sizeof(NamedAttribute) * size);
    return new DictionaryAttributeStorage(data, size);
  }

  static std::size_t HashValue(const ParamKey &key) {
    auto value = std::get<0>(key);
    uint32_t size = std::get<1>(key);
    std::sort(value, value + size);
    std::size_t hash_value = 0;
    hash_value = hash_combine(hash_value, std::hash<uint32_t>()(size));
    for (size_t i = 0; i < size; i++) {
      hash_value =
          hash_combine(hash_value, std::hash<NamedAttribute>()(value[i]));
    }
    return hash_value;
  }

  bool operator==(const ParamKey &key) const {
    uint32_t size = std::get<1>(key);
    if (size_ != size) return false;
    auto value = std::get<0>(key);
    std::sort(value, value + size);
    for (size_t i = 0; i < size_; i++) {
      if (data_[i] != value[i]) {
        return false;
      }
    }
    return true;
  }

  ParamKey GetAsKey() const { return ParamKey(data_, size_); }

  NamedAttribute *data_;

  uint32_t size_;

 private:
  static std::size_t hash_combine(std::size_t lhs, std::size_t rhs) {
    return lhs ^= rhs + 0x9e3779b9 + (lhs << 6) + (lhs >> 2);
  }
};
}  // namespace ir
