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
#include "paddle/ap/include/axpr/attr_map.h"
#include "paddle/ap/include/axpr/bool.h"
#include "paddle/ap/include/axpr/builtin_func_type.h"
#include "paddle/ap/include/axpr/builtin_high_order_func_type.h"
#include "paddle/ap/include/axpr/builtin_serializable_attr_map.h"
#include "paddle/ap/include/axpr/builtin_symbol.h"
#include "paddle/ap/include/axpr/class_instance.h"
#include "paddle/ap/include/axpr/closure.h"
#include "paddle/ap/include/axpr/continuation.h"
#include "paddle/ap/include/axpr/data_type.h"
#include "paddle/ap/include/axpr/data_value.h"
#include "paddle/ap/include/axpr/environment.h"
#include "paddle/ap/include/axpr/error.h"
#include "paddle/ap/include/axpr/float.h"
#include "paddle/ap/include/axpr/function.h"
#include "paddle/ap/include/axpr/int.h"
#include "paddle/ap/include/axpr/list.h"
#include "paddle/ap/include/axpr/method.h"
#include "paddle/ap/include/axpr/mutable_list.h"
#include "paddle/ap/include/axpr/mutable_ordered_dict.h"
#include "paddle/ap/include/axpr/nothing.h"
#include "paddle/ap/include/axpr/ordered_dict.h"
#include "paddle/ap/include/axpr/packed_args.h"
#include "paddle/ap/include/axpr/pointer_type.h"
#include "paddle/ap/include/axpr/pointer_value.h"
#include "paddle/ap/include/axpr/serializable_list.h"
#include "paddle/ap/include/axpr/serializable_value.h"
#include "paddle/ap/include/axpr/starred.h"
#include "paddle/ap/include/axpr/string.h"
#include "paddle/ap/include/axpr/type.h"
#include "paddle/ap/include/axpr/type_util.h"

namespace ap::axpr {

using adt::Nothing;

template <typename ValueT, typename... Ts>
using ValueBase = std::variant<Type<Nothing,
                                    bool,
                                    int64_t,
                                    double,
                                    std::string,
                                    DataType,
                                    DataValue,
                                    PointerType,
                                    PointerValue,
                                    adt::List<ValueT>,
                                    adt::List<SerializableValue>,
                                    MutableList<ValueT>,
                                    AttrMap<ValueT>,
                                    AttrMap<SerializableValue>,
                                    OrderedDict<ValueT>,
                                    MutableOrderedDict<ValueT>,
                                    BuiltinClassInstance<ValueT>,
                                    ClassInstance<ValueT>,
                                    PackedArgs<ValueT>,
                                    Starred<ValueT>,
                                    Function<SerializableValue>,
                                    Closure<ValueT>,
                                    Continuation<ValueT>,
                                    Method<ValueT>,
                                    builtin_symbol::Symbol,
                                    BuiltinFuncType<ValueT>,
                                    BuiltinHighOrderFuncType<ValueT>,
                                    Ts...>,
                               Nothing,
                               bool,
                               int64_t,
                               double,
                               std::string,
                               DataType,
                               DataValue,
                               PointerType,
                               PointerValue,
                               adt::List<ValueT>,
                               adt::List<SerializableValue>,
                               MutableList<ValueT>,
                               AttrMap<ValueT>,
                               AttrMap<SerializableValue>,
                               OrderedDict<ValueT>,
                               MutableOrderedDict<ValueT>,
                               BuiltinClassInstance<ValueT>,
                               ClassInstance<ValueT>,
                               PackedArgs<ValueT>,
                               Starred<ValueT>,
                               Function<SerializableValue>,
                               Closure<ValueT>,
                               Continuation<ValueT>,
                               Method<ValueT>,
                               builtin_symbol::Symbol,
                               BuiltinFuncType<ValueT>,
                               BuiltinHighOrderFuncType<ValueT>,
                               Ts...>;
template <typename ValueT>
ValueT GetType(const ValueT& value) {
  return value.Match(
      [](const BuiltinClassInstance<ValueT>& impl) -> ValueT {
        return impl.type;
      },
      [](const ClassInstance<ValueT>& impl) -> ValueT { return impl->type; },
      [](const auto& impl) -> ValueT {
        using T = std::decay_t<decltype(impl)>;
        return TypeImpl<T>{};
      });
}

template <typename ValueT>
adt::Result<typename TypeTrait<ValueT>::TypeT> CastToType(const ValueT& value) {
  ADT_LET_CONST_REF(type,
                    value.template TryGet<typename TypeTrait<ValueT>::TypeT>());
  return type;
}

template <typename TypeImplT, typename ValueT>
adt::Result<TypeImplT> TryGetTypeImpl(const ValueT& value) {
  ADT_LET_CONST_REF(type, CastToType(value));
  ADT_LET_CONST_REF(type_impl, type.template TryGet<TypeImplT>());
  return type_impl;
}

template <typename T, typename ValueT>
adt::Result<T> TryGetBuiltinClassInstance(const ValueT& val) {
  ADT_LET_CONST_REF(instance,
                    val.template TryGet<BuiltinClassInstance<ValueT>>());
  ADT_LET_CONST_REF(ret, instance.template TryGet<T>());
  return ret;
}

template <typename T, typename ValueT>
adt::Result<T> Get(const ValueT& val) {
  using TypeT = typename TypeTrait<ValueT>::TypeT;
  if constexpr (ValueT::template IsMyAlternative<T>()) {
    return val.template TryGet<T>();
  } else if constexpr (TypeT::template IsMyAlternative<T>()) {
    return TryGetTypeImpl<T>(val);
  } else {
    return TryGetBuiltinClassInstance<T>(val);
  }
}

template <typename T, typename ValueT>
adt::Result<bool> CastableTo(const ValueT& val) {
  using TypeT = typename TypeTrait<ValueT>::TypeT;
  if constexpr (ValueT::template IsMyAlternative<T>()) {
    return val.template Has<T>();
  } else if constexpr (TypeT::template IsMyAlternative<T>()) {
    ADT_LET_CONST_REF(type, CastToType(val));
    return type.template Has<T>();
  } else {
    ADT_LET_CONST_REF(instance,
                      val.template TryGet<BuiltinClassInstance<ValueT>>());
    return instance.template Has<T>();
  }
}

struct Value : public ValueBase<Value> {
  using ValueBase<Value>::ValueBase;
  ADT_DEFINE_VARIANT_METHODS(ValueBase<Value>);

  static axpr::AttrMap<Value> GetExportedTypes() {
    return axpr::GetObjectTypeName2Type<Value>();
  }

  template <typename T>
  adt::Result<T> CastTo() const {
    return axpr::Get<T>(*this);
  }

  template <typename T>
  bool CastableTo() const {
    const auto& ret = axpr::CastableTo<T>(*this);
    return ret.HasOkValue() ? ret.GetOkValue() : false;
  }
};

}  // namespace ap::axpr
