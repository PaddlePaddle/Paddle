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

#include "paddle/ir/core/attribute.h"
#include "paddle/ir/core/utils.h"

namespace ir {

#define DECLARE_BASE_TYPE_ATTRIBUTE_STORAGE(concrete_storage, base_type) \
  struct concrete_storage : public ir::AttributeStorage {                \
    using ParamKey = base_type;                                          \
                                                                         \
    explicit concrete_storage(const ParamKey &key) { data_ = key; }      \
                                                                         \
    static concrete_storage *Construct(ParamKey key) {                   \
      return new concrete_storage(key);                                  \
    }                                                                    \
                                                                         \
    static std::size_t HashValue(const ParamKey &key) {                  \
      return std::hash<base_type>()(key);                                \
    }                                                                    \
                                                                         \
    bool operator==(const ParamKey &key) const { return data_ == key; }  \
                                                                         \
    ParamKey GetAsKey() const { return data_; }                          \
                                                                         \
   private:                                                              \
    ParamKey data_;                                                      \
  };

///
/// \brief Define Parametric AttributeStorage for StrAttribute.
///
struct StrAttributeStorage : public AttributeStorage {
  using ParamKey = std::string;

  explicit StrAttributeStorage(const ParamKey &key) {
    data_ = reinterpret_cast<char *>(malloc(key.size()));
    memcpy(data_, key.c_str(), key.size());
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
    return std::equal(data_, data_ + size_, key.c_str());
  }

  ParamKey GetAsKey() const { return ParamKey(data_, size_); }

 private:
  char *data_;
  uint32_t size_;
};

DECLARE_BASE_TYPE_ATTRIBUTE_STORAGE(BoolAttributeStorage, bool);
DECLARE_BASE_TYPE_ATTRIBUTE_STORAGE(FloatAttributeStorage, float);
DECLARE_BASE_TYPE_ATTRIBUTE_STORAGE(DoubleAttributeStorage, double);
DECLARE_BASE_TYPE_ATTRIBUTE_STORAGE(Int32_tAttributeStorage, int32_t);
DECLARE_BASE_TYPE_ATTRIBUTE_STORAGE(Int64_tAttributeStorage, int64_t);
DECLARE_BASE_TYPE_ATTRIBUTE_STORAGE(PointerAttributeStorage, void *);

struct ArrayAttributeStorage : public AttributeStorage {
  using ParamKey = std::vector<Attribute>;

  explicit ArrayAttributeStorage(const ParamKey &key) {
    data_ =
        reinterpret_cast<Attribute *>(malloc(sizeof(Attribute) * key.size()));
    memcpy(reinterpret_cast<void *>(data_),
           reinterpret_cast<const void *>(key.data()),
           sizeof(Attribute) * key.size());
    length_ = key.size();
  }

  ~ArrayAttributeStorage() { free(reinterpret_cast<void *>(data_)); }

  static ArrayAttributeStorage *Construct(ParamKey key) {
    return new ArrayAttributeStorage(key);
  }

  static std::size_t HashValue(const ParamKey &key) {
    std::size_t hash_value = 0;
    for (size_t i = 0; i < key.size(); ++i) {
      hash_value = hash_combine(hash_value, std::hash<Attribute>()(key[i]));
    }
    return hash_value;
  }

  bool operator==(const ParamKey &key) const {
    if (key.size() != length_) {
      return false;
    }
    for (size_t i = 0; i < length_; ++i) {
      if (data_[i] != key[i]) {
        return false;
      }
    }
    return true;
  }

  ParamKey GetAsKey() const { return ParamKey(data_, data_ + length_); }

 private:
  Attribute *data_ = nullptr;
  size_t length_ = 0;
};

}  // namespace ir
