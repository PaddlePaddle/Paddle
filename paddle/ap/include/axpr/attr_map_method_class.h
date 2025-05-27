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

#include "paddle/ap/include/axpr/attr_map.h"
#include "paddle/ap/include/axpr/constants.h"
#include "paddle/ap/include/axpr/method_class.h"
#include "paddle/ap/include/axpr/to_string.h"

namespace ap::axpr {

template <typename ValueT>
struct AttrMapMethodClass {
  using This = AttrMapMethodClass;
  using Self = AttrMap<ValueT>;

  adt::Result<ValueT> ToString(axpr::InterpreterBase<ValueT>* interpreter,
                               const Self& self) {
    std::ostringstream ss;
    ss << "AttrMap(";
    int i = 0;
    for (const auto& [k, v] : self->storage) {
      if (i++ > 0) {
        ss << ", ";
      }
      ADT_LET_CONST_REF(value_str, axpr::ToString(interpreter, v));
      ss << k << "=" << value_str;
    }
    ss << ")";
    return ss.str();
  }

  adt::Result<ValueT> GetAttr(const Self& self, const ValueT& attr_name_val) {
    ADT_LET_CONST_REF(attr_name, attr_name_val.template TryGet<std::string>());
    ADT_LET_CONST_REF(val, self->Get(attr_name)) << adt::errors::AttributeError{
        std::string() + "'object' has no attribute '" + attr_name + "'."};
    return val;
  }
};

template <typename ValueT>
struct TypeImplBuiltinAttrMapMethodClass {
  using This = TypeImplBuiltinAttrMapMethodClass;
  using Self = TypeImpl<AttrMap<ValueT>>;

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
                                  "the construct of AttrMap "
                                  "takes no positional arguments."};
    return kwargs;
  }
};

template <typename ValueT>
struct MethodClassImpl<ValueT, AttrMap<ValueT>>
    : public AttrMapMethodClass<ValueT> {};

template <typename ValueT>
struct MethodClassImpl<ValueT, TypeImpl<AttrMap<ValueT>>>
    : public TypeImplBuiltinAttrMapMethodClass<ValueT> {};

}  // namespace ap::axpr
