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
#include "paddle/ap/include/axpr/builtin_class_instance.h"
#include "paddle/ap/include/axpr/naive_class_ops.h"
#include "paddle/ap/include/axpr/type.h"
#include "paddle/ap/include/axpr/value.h"
#include "paddle/ap/include/rt_module/function.h"
#include "paddle/ap/include/rt_module/function_helper.h"

namespace ap::rt_module {

struct FunctionMethodClass {
  using Self = Function;

  static adt::Result<axpr::Value> Call(const axpr::Value& self_val,
                                       const std::vector<axpr::Value>& args) {
    ADT_LET_CONST_REF(self, self_val.template CastTo<Self>());
    return FunctionHelper{}.Apply(self, args);
  }
};

inline axpr::TypeImpl<axpr::BuiltinClassInstance<axpr::Value>>
GetSoFunctionClass() {
  static auto cls(axpr::MakeBuiltinClass<axpr::Value>(
      "so_function", [&](const auto& DoEach) {
        DoEach("__call__", &FunctionMethodClass::Call);
      }));
  return axpr::MakeGlobalNaiveClassOps<FunctionMethodClass::Self>(cls);
}

}  // namespace ap::rt_module
