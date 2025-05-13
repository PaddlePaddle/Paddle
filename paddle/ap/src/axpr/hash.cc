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

#include "paddle/ap/include/adt/adt.h"
#include "paddle/ap/include/axpr/interpreter_base.h"
#include "paddle/ap/include/axpr/value.h"
#include "paddle/ap/include/axpr/value_method_class.h"

namespace ap::axpr {

adt::Result<int64_t> HashImpl::operator()(
    InterpreterBase<axpr::Value>* interpreter, const axpr::Value& val) const {
  const auto& func = MethodClass<axpr::Value>::Hash(val);
  using RetT = adt::Result<int64_t>;
  return func.Match(
      [&](const adt::Nothing&) -> RetT {
        return adt::errors::TypeError{GetTypeName(val) +
                                      " class has no __hash__ function."};
      },
      [&](adt::Result<axpr::Value> (*unary_func)(const axpr::Value&)) -> RetT {
        ADT_LET_CONST_REF(hash_val, unary_func(val));
        ADT_LET_CONST_REF(hash, hash_val.template TryGet<int64_t>());
        return hash;
      },
      [&](adt::Result<axpr::Value> (*unary_func)(InterpreterBase<axpr::Value>*,
                                                 const axpr::Value&)) -> RetT {
        ADT_LET_CONST_REF(hash_val, unary_func(interpreter, val));
        ADT_LET_CONST_REF(hash, hash_val.template TryGet<int64_t>());
        return hash;
      });
}

}  // namespace ap::axpr
