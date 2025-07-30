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

#include "paddle/ap/include/axpr/class_attrs_helper.h"
#include "paddle/ap/include/axpr/value.h"

namespace ap::axpr {

struct CallableHelper {
  bool IsCallable(const axpr::Value& value) const {
    return value.Match(
        [&](const BuiltinFuncType<axpr::Value>& func) -> bool { return true; },
        [&](const BuiltinHighOrderFuncType<axpr::Value>& func) -> bool {
          return true;
        },
        [&](const Method<axpr::Value>& method) -> bool { return true; },
        [&](const Closure<axpr::Value>& closure) -> bool { return true; },
        [&](const Continuation<axpr::Value>& continuation) -> bool {
          return true;
        },
        [&](const Function<SerializableValue>& function) -> bool {
          return true;
        },
        [&](const builtin_symbol::Symbol& symbol) -> bool { return true; },
        [&](const BuiltinClassInstance<axpr::Value>& builtin_class_instance)
            -> bool {
          const auto* class_attrs = builtin_class_instance.type.class_attrs();
          ClassAttrsHelper<axpr::Value, axpr::Value> helper{};
          return helper.OptGet(class_attrs, "__call__").has_value();
        },
        [&](const ClassInstance<axpr::Value>& class_instance) -> bool {
          const auto& class_attrs = class_instance->type.class_attrs;
          ClassAttrsHelper<axpr::Value, axpr::SerializableValue> helper{};
          return helper.OptGet(class_attrs, "__call__").has_value();
        },
        [&](const auto&) -> bool { return false; });
  }
};

}  // namespace ap::axpr
