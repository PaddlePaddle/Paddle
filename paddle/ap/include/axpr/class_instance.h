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

#include <unordered_map>
#include "paddle/ap/include/axpr/adt.h"
#include "paddle/ap/include/axpr/class_attrs.h"
#include "paddle/ap/include/axpr/error.h"
#include "paddle/ap/include/axpr/instance_attrs.h"
#include "paddle/ap/include/axpr/serializable_value.h"
#include "paddle/ap/include/axpr/type.h"

namespace ap::axpr {

template <typename ValueT>
struct ClassInstance;

template <typename ValueT>
struct TypeImpl<ClassInstance<ValueT>> {
  explicit TypeImpl<ClassInstance<ValueT>>(
      const ClassAttrs<SerializableValue>& class_attr_val)
      : class_attrs(class_attr_val) {}

  ClassAttrs<SerializableValue> class_attrs;

  const std::string& Name() const { return class_attrs->Name(); }

  bool operator==(const TypeImpl<ClassInstance<ValueT>>& other) const {
    return this->class_attrs == other.class_attrs;
  }
};

template <typename ValueT>
struct ClassInstanceImpl {
  TypeImpl<ClassInstance<ValueT>> type;
  InstanceAttrs<ValueT> instance_attrs;

  bool operator==(const ClassInstanceImpl& other) const {
    return this == &other;
  }
};

template <typename ValueT>
ADT_DEFINE_RC(ClassInstance, ClassInstanceImpl<ValueT>);

}  // namespace ap::axpr
