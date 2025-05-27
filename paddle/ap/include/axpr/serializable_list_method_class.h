// Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/ap/include/axpr/adt.h"
#include "paddle/ap/include/axpr/method_class.h"
#include "paddle/ap/include/axpr/serializable_list.h"
#include "paddle/ap/include/axpr/starred.h"

namespace ap::axpr {

template <typename ValueT>
struct MethodClassImpl<ValueT, adt::List<SerializableValue>> {
  using This = MethodClassImpl<ValueT, adt::List<SerializableValue>>;
  using Self = adt::List<SerializableValue>;

  adt::Result<ValueT> Length(const Self& self) {
    return static_cast<int64_t>(self->size());
  }

  adt::Result<ValueT> ToString(const Self& self) {
    ADT_LET_CONST_REF(str, SerializableValueHelper{}.ToString(self));
    return str;
  }

  adt::Result<ValueT> Hash(const Self& self) {
    ADT_LET_CONST_REF(hash_value, SerializableValueHelper{}.Hash(self));
    return hash_value;
  }

  adt::Result<ValueT> GetItem(const Self& self, const ValueT& idx) {
    return idx.Match(
        [&](int64_t index) -> Result<ValueT> {
          if (index < 0) {
            index += self->size();
          }
          if (index >= 0 && index < static_cast<int64_t>(self->size())) {
            return self->at(index).template CastTo<ValueT>();
          }
          return adt::errors::IndexError{"list index out of range"};
        },
        [&](const auto&) -> Result<ValueT> {
          return adt::errors::TypeError{std::string() +
                                        "list indices must be integers, not " +
                                        axpr::GetTypeName(idx)};
        });
  }

  adt::Result<ValueT> Starred(const Self& self) {
    return ap::axpr::Starred<ValueT>{self};
  }
};

template <typename ValueT>
struct MethodClassImpl<ValueT, TypeImpl<adt::List<SerializableValue>>> {
  using Self = TypeImpl<adt::List<SerializableValue>>;

  using This = MethodClassImpl<ValueT, Self>;
};

}  // namespace ap::axpr
