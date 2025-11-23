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
#include "paddle/ap/include/axpr/error.h"
#include "paddle/ap/include/axpr/interpreter_base.h"
#include "paddle/ap/include/axpr/type.h"

namespace ap::axpr {

template <typename ValueT>
using BuiltinFuncType = Result<ValueT> (*)(const ValueT&,
                                           const std::vector<ValueT>& args);

template <typename This,
          Result<typename This::Val> (This::*func)(
              const typename This::Self&,
              const std::vector<typename This::Val>& args)>
Result<typename This::Val> WrapAsBuiltinFuncType(
    const typename This::Val& self_val,
    const std::vector<typename This::Val>& args) {
  ADT_LET_CONST_REF(self, self_val.template TryGet<typename This::Self>());
  return (This{}.*func)(self, args);
}

template <typename ValueT>
struct TypeImpl<BuiltinFuncType<ValueT>> : public std::monostate {
  using value_type = BuiltinFuncType<ValueT>;

  const char* Name() const { return "builtin_function"; }
};

template <typename ValueT>
using BuiltinHighOrderFuncType =
    Result<ValueT> (*)(InterpreterBase<ValueT>* interpreter,
                       const ValueT& obj,
                       const std::vector<ValueT>& args);

template <typename ValueT>
struct TypeImpl<BuiltinHighOrderFuncType<ValueT>> : public std::monostate {
  using value_type = BuiltinHighOrderFuncType<ValueT>;

  const char* Name() const { return "builtin_high_order_function"; }
};

template <typename ValueT>
using BuiltinFunctionImpl =
    std::variant<BuiltinFuncType<ValueT>, BuiltinHighOrderFuncType<ValueT>>;

template <typename ValueT>
struct BuiltinFunction : public BuiltinFunctionImpl<ValueT> {
  using BuiltinFunctionImpl<ValueT>::BuiltinFunctionImpl;
  ADT_DEFINE_VARIANT_METHODS(BuiltinFunctionImpl<ValueT>);

  template <typename T>
  T CastTo() const {
    return Match([](const auto& impl) -> T { return impl; });
  }
};

}  // namespace ap::axpr
