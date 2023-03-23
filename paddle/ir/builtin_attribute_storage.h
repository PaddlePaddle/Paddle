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

#include "paddle/ir/attribute.h"

namespace ir {
///
/// \brief Define Parameteric AttributeStorage for StrAttribute.
///
struct StrAttributeStorage : public ir::AttributeStorage {
  using ParamKey = std::string;

  explicit StrAttributeStorage(const ParamKey &key) {
    data_ = reinterpret_cast<char *>(malloc(key.size()));
    memcpy(data_, const_cast<char *>(key.c_str()), key.size());
    size_ = key.size();
  }

  ~StrAttributeStorage() { free(data_); }

  static StrAttributeStorage *Construct(ParamKey key) {
    return new StrAttributeStorage(key);
  }

  static std::size_t HashValue(const ParamKey &key) {
    return std::hash<std::string>()(key);
  }

  bool operator==(const ParamKey &key) const {
    return std::equal(data_, data_ + size_, const_cast<char *>(key.c_str()));
  }

  ParamKey GetAsKey() const { return ParamKey(data_, size_); }

 private:
  char *data_;
  uint32_t size_;
};

///
/// \brief Define Parameteric AttributeStorage for DictionaryAttributeStorage.
///
class StrAttribute;
class NamedAttribute;
struct DictionaryAttributeStorage : public AttributeStorage {
  using ParamKey = std::map<StrAttribute, Attribute>;

  explicit DictionaryAttributeStorage(const ParamKey &key);

  ~DictionaryAttributeStorage() { free(data_); }

  static DictionaryAttributeStorage *Construct(ParamKey key) {
    return new DictionaryAttributeStorage(key);
  }

  static std::size_t HashValue(const ParamKey &key);

  bool operator==(const ParamKey &key) const;

  ParamKey GetAsKey() const;

  Attribute GetValue(const StrAttribute &name) const;

  NamedAttribute *data() const { return data_; }

  uint32_t size() const { return size_; }

 private:
  NamedAttribute *data_;
  uint32_t size_;
};
}  // namespace ir
