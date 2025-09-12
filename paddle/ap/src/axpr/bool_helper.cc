// Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/ap/include/axpr/bool_helper.h"
#include "paddle/ap/include/axpr/value.h"
#include "paddle/ap/include/axpr/value_method_class.h"

namespace ap::axpr {

adt::Result<bool> BoolHelper::ConvertToBool(const axpr::Value& cond) {
  using TypeT = typename TypeTrait<axpr::Value>::TypeT;
  return cond.Match(
      [](const TypeT&) -> Result<bool> { return true; },
      [](const bool c) -> Result<bool> { return c; },
      [](const int64_t c) -> Result<bool> { return c != 0; },
      [](const double c) -> Result<bool> { return c != 0; },
      [](const std::string& c) -> Result<bool> { return !c.empty(); },
      [](const Nothing&) -> Result<bool> { return false; },
      [](const adt::List<axpr::Value>& list) -> Result<bool> {
        return list->size() > 0;
      },
      [](const MutableList<axpr::Value>& list) -> Result<bool> {
        ADT_LET_CONST_REF(list_ptr, list.Get());
        return list_ptr->size() > 0;
      },
      [](const AttrMap<axpr::Value>& obj) -> Result<bool> {
        return obj->size() > 0;
      },
      [](const Lambda<CoreExpr>&) -> Result<bool> { return true; },
      [](const Closure<axpr::Value>&) -> Result<bool> { return true; },
      [](const Continuation<axpr::Value>&) -> Result<bool> { return true; },
      [](const Method<axpr::Value>&) -> Result<bool> { return true; },
      [](const builtin_symbol::Symbol&) -> Result<bool> { return true; },
      [](const BuiltinFuncType<axpr::Value>&) -> Result<bool> { return true; },
      [](const BuiltinHighOrderFuncType<axpr::Value>&) -> Result<bool> {
        return true;
      },
      [&](const auto&) -> Result<bool> {
        return TypeError{std::string() + "'" + axpr::GetTypeName(cond) +
                         "' could not be convert to bool"};
      });
}

}  // namespace ap::axpr
