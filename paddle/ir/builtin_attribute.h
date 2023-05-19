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
#define GET_BUILT_IN_ATTRIBUTE_LIST ir::StrAttribute

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

}  // namespace ir

namespace std {
template <>
struct hash<ir::StrAttribute> {
  std::size_t operator()(const ir::StrAttribute &obj) const {
    return std::hash<const ir::StrAttribute::Storage *>()(obj.storage());
  }
};
}  // namespace std
