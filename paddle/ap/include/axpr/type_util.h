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
#include "paddle/ap/include/axpr/data_type.h"
#include "paddle/ap/include/axpr/data_value.h"
#include "paddle/ap/include/axpr/mutable_list.h"
#include "paddle/ap/include/axpr/mutable_ordered_dict.h"
#include "paddle/ap/include/axpr/ordered_dict.h"
#include "paddle/ap/include/axpr/packed_args.h"
#include "paddle/ap/include/axpr/pointer_type.h"
#include "paddle/ap/include/axpr/pointer_value.h"
#include "paddle/ap/include/axpr/type.h"

namespace ap::axpr {

namespace detail {

template <typename ValueT, typename... ValueImplTypes>
struct GetTypeName2TypeHelper;

template <typename ValueT>
struct GetTypeName2TypeHelper<ValueT> {
  static void Call(AttrMap<ValueT>*) {}
};

template <typename ValueT, typename ValueImplType0, typename... ValueImplTypes>
struct GetTypeName2TypeHelper<ValueT, ValueImplType0, ValueImplTypes...> {
  static void Call(AttrMap<ValueT>* ret) {
    TypeImpl<ValueImplType0> type_impl{};
    ValueT type{type_impl};
    (*ret)->Set(type_impl.Name(), type);
    GetTypeName2TypeHelper<ValueT, ValueImplTypes...>::Call(ret);
  }
};

template <typename ValueT, typename... ValueImplTypes>
struct GetNonPyStandardTypeName2TypeHelper;

template <typename ValueT>
struct GetNonPyStandardTypeName2TypeHelper<ValueT> {
  static void Call(AttrMap<ValueT>*) {}
};

template <typename ValueT, typename ValueImplType0, typename... ValueImplTypes>
struct GetNonPyStandardTypeName2TypeHelper<ValueT,
                                           ValueImplType0,
                                           ValueImplTypes...> {
  static void Call(AttrMap<ValueT>* ret) {
    TypeImpl<ValueImplType0> type_impl{};
    ValueT type{type_impl};
    (*ret)->Set(std::string() + "__builtin__" + type_impl.Name(), type);
    GetNonPyStandardTypeName2TypeHelper<ValueT, ValueImplTypes...>::Call(ret);
  }
};

}  // namespace detail

template <typename ValueT, typename... ValueImplTypes>
AttrMap<ValueT> GetObjectTypeName2Type() {
  AttrMap<ValueT> object;
  detail::GetTypeName2TypeHelper<ValueT,
                                 typename TypeTrait<ValueT>::TypeT,
                                 Nothing,
                                 bool,
                                 int64_t,
                                 double,
                                 std::string,
                                 DataType,
                                 DataValue,
                                 PointerType,
                                 PointerValue,
                                 MutableList<ValueT>,
                                 OrderedDict<ValueT>,
                                 MutableOrderedDict<ValueT>,
                                 PackedArgs<ValueT>,
                                 AttrMap<axpr::SerializableValue>,
                                 ValueImplTypes...>::Call(&object);
  detail::GetNonPyStandardTypeName2TypeHelper<ValueT,
                                              DataType,
                                              DataValue,
                                              PointerType,
                                              PointerValue,
                                              MutableList<ValueT>,
                                              OrderedDict<ValueT>,
                                              MutableOrderedDict<ValueT>,
                                              AttrMap<axpr::SerializableValue>,
                                              AttrMap<ValueT>,
                                              ValueImplTypes...>::Call(&object);
  return object;
}

}  // namespace ap::axpr
