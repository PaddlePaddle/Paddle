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

#include <sstream>
#include "paddle/ap/include/axpr/builtin_serializable_attr_map.h"
#include "paddle/ap/include/axpr/class_instance.h"
#include "paddle/ap/include/axpr/constants.h"
#include "paddle/ap/include/axpr/method_class.h"
#include "paddle/ap/include/axpr/packed_args.h"
#include "paddle/ap/include/axpr/serializable_value_helper.h"

namespace ap::axpr {

template <typename ValueT>
struct BuiltinSerializableAttrMapMethodClass {
  using This = BuiltinSerializableAttrMapMethodClass;
  using Self = AttrMap<SerializableValue>;

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

  adt::Result<ValueT> GetAttr(const Self& self, const ValueT& attr_name_val) {
    ADT_LET_CONST_REF(attr_name, attr_name_val.template TryGet<std::string>());
    ADT_LET_CONST_REF(val, self->Get(attr_name)) << adt::errors::AttributeError{
        std::string() + "'BuiltinSerializableAttrMap' has no attribute '" +
        attr_name + "'."};
    return val.template CastTo<ValueT>();
  }
};

template <typename ValueT>
struct TypeImplBuiltinSerializableAttrMapMethodClass {
  using This = TypeImplBuiltinSerializableAttrMapMethodClass;
  using Self = TypeImpl<AttrMap<SerializableValue>>;

  adt::Result<ValueT> Call(const Self&) { return &This::StaticConstruct; }

  static adt::Result<ValueT> StaticConstruct(const ValueT&,
                                             const std::vector<ValueT>& args) {
    return This{}.Construct(args);
  }

  adt::Result<ValueT> Construct(const std::vector<ValueT>& args) {
    const auto& packed_args = CastToPackedArgs(args);
    const auto& [pos_args, kwargs] = *packed_args;
    ADT_CHECK(pos_args->empty())
        << adt::errors::TypeError{std::string() +
                                  "the construct of BuiltinSerializableAttrMap "
                                  "takes no positional arguments."};
    ADT_LET_CONST_REF(serializable_val,
                      SerializableValueHelper{}.CastObjectFrom(kwargs));
    ADT_LET_CONST_REF(
        serializable_obj,
        serializable_val.template TryGet<AttrMap<SerializableValue>>());
    return serializable_obj;
  }
};

template <typename ValueT>
struct MethodClassImpl<ValueT, AttrMap<SerializableValue>>
    : public BuiltinSerializableAttrMapMethodClass<ValueT> {};

template <typename ValueT>
struct MethodClassImpl<ValueT, TypeImpl<AttrMap<SerializableValue>>>
    : public TypeImplBuiltinSerializableAttrMapMethodClass<ValueT> {};

}  // namespace ap::axpr
