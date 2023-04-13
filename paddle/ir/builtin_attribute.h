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

#include "paddle/ir/attribute.h"
#include "paddle/ir/builtin_attribute_storage.h"
#include "paddle/ir/utils.h"

namespace ir {
///
/// \brief All built-in attributes.
///
#define GET_BUILT_IN_ATTRIBUTE_LIST ir::StrAttribute, ir::DictionaryAttribute

class StrAttribute : public ir::Attribute {
 public:
  using Attribute::Attribute;

  DECLARE_ATTRIBUTE_UTILITY_FUNCTOR(StrAttribute, StrAttributeStorage);

  bool operator<(const StrAttribute &right) const {
    return storage() < right.storage();
  }

  std::string data() const;

  uint32_t size() const;
};

class NamedAttribute {
 public:
  NamedAttribute(StrAttribute name, Attribute value);

  StrAttribute name() const { return name_; }

  Attribute value() const { return value_; }

  void SetName(StrAttribute name) { name_ = name; }

  void SetValue(Attribute value) { value_ = value; }

  bool operator<(const NamedAttribute &right) const;

  bool operator==(const NamedAttribute &right) const;

  bool operator!=(const NamedAttribute &right) const;

  friend struct std::hash<NamedAttribute>;

  operator std::pair<const StrAttribute, Attribute>() const {
    return std::make_pair(name_, value_);
  }

 private:
  StrAttribute name_;
  Attribute value_;
};

class DictionaryAttribute : public ir::Attribute {
 public:
  using Attribute::Attribute;

  DECLARE_ATTRIBUTE_UTILITY_FUNCTOR(DictionaryAttribute,
                                    DictionaryAttributeStorage);

  Attribute GetValue(const StrAttribute &name);

  uint32_t size() const;
};

}  // namespace ir

namespace std {
template <>
struct hash<ir::NamedAttribute> {
  std::size_t operator()(const ir::NamedAttribute &obj) const {
    return ir::hash_combine(std::hash<ir::Attribute>()(obj.name_),
                            std::hash<ir::Attribute>()(obj.value_));
  }
};
}  // namespace std
