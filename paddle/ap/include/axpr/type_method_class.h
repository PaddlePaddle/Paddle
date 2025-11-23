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

#include "paddle/ap/include/axpr/class_instance.h"
#include "paddle/ap/include/axpr/method_class.h"
#include "paddle/ap/include/axpr/serializable_value.h"
#include "paddle/ap/include/axpr/type.h"

namespace ap::axpr {

template <typename ValueT, typename... Ts>
struct MethodClassImpl<ValueT, Type<Ts...>> {};

template <typename ValueT, typename... Ts>
struct MethodClassImpl<ValueT, TypeImpl<Type<Ts...>>> {
  using Self = TypeImpl<Type<Ts...>>;
  using This = MethodClassImpl<ValueT, Self>;

  adt::Result<ValueT> Call(const Self& value) {
    return &This::StaticGetOrConstruct;
  }

  static adt::Result<ValueT> StaticGetOrConstruct(
      const ValueT& self_val, const std::vector<ValueT>& args) {
    if (args.size() == 1) {
      return GetType(args.at(0));
    }
    if (args.size() == 3) {
      return This{}.MakeClass(args.at(0), args.at(1), args.at(2));
    }
    return adt::errors::TypeError{std::string() +
                                  "type() takes 1 or 3 arguments, but " +
                                  std::to_string(args.size()) + " were given."};
  }

  adt::Result<ValueT> MakeClass(const ValueT& class_name_val,
                                const ValueT& superclasses_val,
                                const ValueT& attributes_object) {
    ADT_LET_CONST_REF(class_name, class_name_val.template TryGet<std::string>())
        << adt::errors::TypeError{
               std::string() + "the argument 1 of type() should be str not " +
               GetTypeName(class_name_val)};
    adt::List<std::shared_ptr<ClassAttrsImpl<SerializableValue>>> superclasses;
    {
      ADT_LET_CONST_REF(superclass_vals,
                        superclasses_val.template TryGet<adt::List<ValueT>>())
          << adt::errors::TypeError{
                 std::string() +
                 "the argument 2 of type() should be list not " +
                 GetTypeName(superclasses_val)};
      superclasses->reserve(superclass_vals->size());
      for (const auto& superclass_val : *superclass_vals) {
        ADT_LET_CONST_REF(
            type_impl,
            TryGetTypeImpl<TypeImpl<ClassInstance<ValueT>>>(superclass_val));
        superclasses->emplace_back(type_impl.class_attrs.shared_ptr());
      }
    }
    ADT_LET_CONST_REF(
        attrs,
        attributes_object.template TryGet<AttrMap<axpr::SerializableValue>>());
    ClassAttrs<SerializableValue> class_attrs{class_name, superclasses, attrs};
    return TypeImpl<ClassInstance<ValueT>>{class_attrs};
  }
};

}  // namespace ap::axpr
