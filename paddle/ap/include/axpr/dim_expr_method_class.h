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

#include "paddle/ap/include/axpr/callable_helper.h"
#include "paddle/ap/include/axpr/interpreter_base.h"
#include "paddle/ap/include/axpr/method_class.h"
#include "paddle/ap/include/axpr/naive_class_ops.h"
#include "paddle/ap/include/axpr/packed_args.h"
#include "paddle/pir/include/dialect/shape/utils/dim_expr.h"

namespace ap::axpr {

template <typename ValueT>
struct DimExprMethodClass {
  using This = DimExprMethodClass;
  using Self = symbol::DimExpr;

  static adt::Result<ValueT> ToString(const ValueT& self_val,
                                      const std::vector<ValueT>& args) {
    ADT_LET_CONST_REF(self, self_val.template CastTo<Self>());
    return symbol::ToString(self);
  }

  static adt::Result<ValueT> Hash(const ValueT& self_val,
                                  const std::vector<ValueT>&) {
    ADT_LET_CONST_REF(self, self_val.template CastTo<Self>());
    int64_t hash_value = std::hash<Self>()(self);
    return hash_value;
  }

  static adt::Result<ValueT> Match(axpr::InterpreterBase<ValueT>* interpreter,
                                   const ValueT& self_val,
                                   const std::vector<ValueT>& packed_args_val) {
    ADT_LET_CONST_REF(self, self_val.template CastTo<Self>());
    const auto& packed_args = axpr::CastToPackedArgs<ValueT>(packed_args_val);
    const auto& [args, kwargs] = *packed_args;
    ADT_CHECK(args->size() == 0) << adt::errors::TypeError{
        std::string() +
        "DimExpr.match() supports keyword arguments only, but " +
        std::to_string(args->size()) + " positional arguments were given"};
    const std::string& type_name = This{}.GetTypeName(self);
    std::string key = type_name;
    if (!kwargs->Has(type_name)) {
      if (!kwargs->Has("_")) {
        return adt::errors::TypeError{std::string() +
                                      "DimExpr.match() failed. no keyword '" +
                                      type_name + "' or '_' provided"};
      }
      key = "_";
    }
    ADT_LET_CONST_REF(func, kwargs->Get(key));
    ADT_CHECK(axpr::CallableHelper{}.IsCallable(func))
        << adt::errors::TypeError{
               std::string() +
               "the arguments of DimExpr.match() should be callable"};
    if (key == "_") {
      return interpreter->InterpretCall(func, {});
    } else {
      const auto& make_args = self.Match(
          [&](int64_t c) -> adt::List<ValueT> { return adt::List<ValueT>{c}; },
          [&](const std::string& c) -> adt::List<ValueT> {
            return adt::List<ValueT>{c};
          },
          [&](const auto&) -> adt::List<ValueT> { return adt::List<Value>{}; });
      return interpreter->InterpretCall(func, make_args.vector());
    }
  }

  const char* GetTypeName(const symbol::DimExpr& dim_expr) const {
    return dim_expr.Match(
        [](int64_t) -> const char* { return "int64"; },
        [&](const std::string&) -> const char* { return "symbol"; },
        [&](const auto&) -> const char* { return "_"; });
  }
};

template <typename ValueT>
axpr::TypeImpl<axpr::BuiltinClassInstance<ValueT>> GetDimExprClass() {
  using Impl = DimExprMethodClass<ValueT>;
  static auto cls(
      axpr::MakeBuiltinClass<ValueT>("DimExpr", [&](const auto& Define) {
        Define("__str__", &Impl::ToString);
        Define("__hash__", &Impl::Hash);
        Define("match", &Impl::Match);
      }));
  return axpr::MakeGlobalNaiveClassOps<typename Impl::Self>(cls);
}

}  // namespace ap::axpr
