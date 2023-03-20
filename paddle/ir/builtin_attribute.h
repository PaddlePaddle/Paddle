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

namespace ir {
///
/// \brief All built-in attributes.
///
#define GET_BUILT_IN_ATTRIBUTE_LIST ir::StrAttribute, ir::DictionaryAttribute

class StrAttribute : public ir::Attribute {
 public:
  using Attribute::Attribute;

  DECLARE_ATTRIBUTE_UTILITY_FUNCTOR(StrAttribute, StrAttributeStorage);

  static StrAttribute get(ir::IrContext *ctx, const std::string &data) {
    return ir::AttributeManager::template get<StrAttribute>(
        ctx, const_cast<char *>(data.c_str()), data.size());
  }

  static int CompareMemory(const char *left, const char *right, size_t length) {
    if (length == 0) return 0;
    return memcmp(left, right, length);
  }

  int compare(const StrAttribute &right) const {
    if (*this == right) return 0;
    if (int compare =
            CompareMemory(storage()->data_,
                          right.storage()->data_,
                          std::min(storage()->size_, right.storage()->size_)))
      return compare < 0 ? -1 : 1;
    if (storage()->size_ == right.storage()->size_) return 0;
    return storage()->size_ < right.storage()->size_ ? -1 : 1;
  }

  bool operator<(const StrAttribute &right) const { return compare(right) < 0; }

  std::string data() const;

  const uint32_t &size() const;
};

class DictionaryAttribute : public ir::Attribute {
 public:
  using Attribute::Attribute;

  DECLARE_ATTRIBUTE_UTILITY_FUNCTOR(DictionaryAttribute,
                                    DictionaryAttributeStorage);

  Attribute GetValue(const StrAttribute &name);

  const uint32_t &size() const;
};

}  // namespace ir
