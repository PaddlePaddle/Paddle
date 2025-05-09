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
#include "paddle/ap/include/axpr/module_mgr_helper.h"

namespace ap::axpr {

template <typename ValueT, typename YieldT>
void VisitEachBuiltinFrameAttr(const YieldT& Yield) {
  AttrMap<ValueT> base{ValueT::GetExportedTypes()};
  for (const auto& [k, v] : base->storage) {
    Yield(k, v);
  }
  Yield("import", &ModuleMgrHelper<ValueT>::ImportModule);
  Yield("print", &Print);
  Yield("max", &Max);
  Yield("min", &Min);
  Yield("len", &Length);
  Yield("getattr", &GetAttr);
  Yield("setattr", &SetAttr);
  ForEachExceptionConstructor(Yield);
  Yield("raise", &Raise);
  Yield("__builtin_not__", &BuiltinNot);

  Yield("__builtin__foreach", &ForEach);

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
}

template <typename ValueT>
AttrMap<ValueT> MakeBuiltinFrameAttrMap() {
  AttrMap<ValueT> attr_map;
  VisitEachBuiltinFrameAttr<ValueT>(
      [&](const std::string& k, const ValueT& v) { attr_map->Set(k, v); });
  return attr_map;
}

}  // namespace ap::axpr
