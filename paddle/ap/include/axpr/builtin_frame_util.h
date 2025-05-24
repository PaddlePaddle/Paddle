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

#include "paddle/ap/include/adt/adt.h"
#include "paddle/ap/include/axpr/attr_map.h"
#include "paddle/ap/include/axpr/builtin_functions.h"
#include "paddle/ap/include/axpr/builtin_symbol.h"
#include "paddle/ap/include/axpr/exception_method_class.h"
#include "paddle/ap/include/axpr/immutable_registry.h"
#include "paddle/ap/include/axpr/module_mgr_helper.h"
#include "paddle/ap/include/fs/builtin_functions.h"
#include "paddle/ap/include/registry/builtin_frame_util.h"

namespace ap::axpr {

template <typename ValueT, typename YieldT>
void VisitEachBuiltinFrameAttr(const YieldT& Yield) {
  AttrMap<ValueT> base{ValueT::GetExportedTypes()};
  for (const auto& [k, v] : base->storage) {
    Yield(k, v);
  }
  Yield("import", &ModuleMgrHelper<ValueT>::ImportModule);
  Yield("__builtin__import", &ModuleMgrHelper<ValueT>::ImportModule);
  Yield("print", &Print);
  Yield("max", &Max);
  Yield("min", &Min);
  Yield("len", &Length);
  Yield("getattr", &GetAttr);
  Yield("setattr", &SetAttr);
  ForEachExceptionConstructor(Yield);
  Yield("raise", &Raise);
  Yield("__builtin__raise", &Raise);
  Yield("__builtin_not__", &BuiltinNot);

  Yield("__builtin__sorted", &Sorted);
  Yield("__builtin__foreach", &ForEach);
  Yield("__builtin__registry", &GetRegistry);
  Yield("__builtin__dirname", &fs::DirName);
  Yield("__builtin__basename", &fs::BaseName);

  Yield("__builtin__function_to_axpr_atomic", &FunctionToAtomicAxpr);
  Yield("__builtin__axpr_atomic_to_function", &AtomicAxprToFunction);
  Yield("__builtin__axpr_json_str_to_axpr", &AxprJsonStrToAxpr);
  Yield("__builtin__axpr_symbol", &AxprSymbol);
  Yield("__builtin__axpr_none", &AxprNone);
  Yield("__builtin__axpr_bool", &AxprBool);
  Yield("__builtin__axpr_int", &AxprInt);
  Yield("__builtin__axpr_float", &AxprFloat);
  Yield("__builtin__axpr_str", &AxprStr);
  Yield("__builtin__axpr_lambda", &AxprLambda);
  Yield("__builtin__axpr_atomic", &AxprAtomic);
  Yield("__builtin__axpr_call", &AxprCall);

  Yield("__builtin__quoted", &Quoted);

  Yield("__builtin__to_pure_function", &ToPureFunction);

  Yield("__builtin__auto_immutable_value_registry_key",
        &ApiAutoImmutableValueRegistryKey);

  Yield("__builtin__is_immutable_value_registered",
        &ApiIsImmutableValueRegistered);

  Yield("__builtin__get_registered_immutable_value",
        &ApiGetRegisteredImmutableValue);

  Yield("__builtin__register_immutable_value", &ApiRegisterImmutableValue);

  auto YieldTwice = [&](const auto& name, const auto& value) {
    Yield(name, value);
    Yield(std::string("__builtin__") + name, value);
  };
  YieldTwice("apply", &Apply);
  YieldTwice("replace_or_trim_left_comma", &ReplaceOrTrimLeftComma);
  YieldTwice("range", &MakeRange);
  YieldTwice("flat_map", &FlatMap);
  YieldTwice("map", &Map);
  YieldTwice("filter", &Filter);
  YieldTwice("reduce", &Reduce);
  YieldTwice("zip", &Zip);
  registry::VisitEachBuiltinFrameAttr<ValueT>(YieldTwice);
}

template <typename ValueT>
AttrMap<ValueT> MakeBuiltinFrameAttrMap() {
  AttrMap<ValueT> attr_map;
  VisitEachBuiltinFrameAttr<ValueT>(
      [&](const std::string& k, const ValueT& v) { attr_map->Set(k, v); });
  return attr_map;
}

}  // namespace ap::axpr
