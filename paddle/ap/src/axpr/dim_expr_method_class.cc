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

#include "paddle/ap/include/axpr/dim_expr_method_class.h"
#include "paddle/ap/include/axpr/callable_helper.h"
#include "paddle/ap/include/axpr/interpreter_base.h"
#include "paddle/ap/include/axpr/method_class.h"
#include "paddle/ap/include/axpr/naive_class_ops.h"
#include "paddle/ap/include/axpr/packed_args.h"
#include "paddle/ap/include/axpr/value.h"
#include "paddle/pir/include/dialect/shape/utils/dim_expr.h"

namespace ap::axpr {

axpr::TypeImpl<axpr::BuiltinClassInstance<axpr::Value>> GetDimExprClass();

struct DimExprMethodClass {
  using This = DimExprMethodClass;
  using Self = symbol::DimExpr;

  static adt::Result<axpr::Value> ToString(
      const axpr::Value& self_val, const std::vector<axpr::Value>& args) {
    ADT_LET_CONST_REF(self, self_val.template CastTo<Self>());
    return symbol::ToString(self);
  }

  static adt::Result<axpr::Value> Hash(const axpr::Value& self_val,
                                       const std::vector<axpr::Value>&) {
    ADT_LET_CONST_REF(self, self_val.template CastTo<Self>());
    int64_t hash_value = std::hash<Self>()(self);
    return hash_value;
  }

  static adt::Result<axpr::Value> Add(const axpr::Value& self_val,
                                      const std::vector<axpr::Value>& args) {
    ADT_LET_CONST_REF(lhs, self_val.template CastTo<Self>());
    ADT_CHECK(args.size() == 1);
    ADT_LET_CONST_REF(rhs, args.at(0).template CastTo<Self>());
    return GetDimExprClass().New(lhs + rhs);
  }

  static adt::Result<axpr::Value> Sub(const axpr::Value& self_val,
                                      const std::vector<axpr::Value>& args) {
    ADT_LET_CONST_REF(lhs, self_val.template CastTo<Self>());
    ADT_CHECK(args.size() == 1);
    ADT_LET_CONST_REF(rhs, args.at(0).template CastTo<Self>());
    return GetDimExprClass().New(lhs - rhs);
  }

  static adt::Result<axpr::Value> Mul(const axpr::Value& self_val,
                                      const std::vector<axpr::Value>& args) {
    ADT_LET_CONST_REF(lhs, self_val.template CastTo<Self>());
    ADT_CHECK(args.size() == 1);
    ADT_LET_CONST_REF(rhs, args.at(0).template CastTo<Self>());
    return GetDimExprClass().New(lhs * rhs);
  }

  static adt::Result<axpr::Value> FloorDiv(
      const axpr::Value& self_val, const std::vector<axpr::Value>& args) {
    ADT_LET_CONST_REF(lhs, self_val.template CastTo<Self>());
    ADT_CHECK(args.size() == 1);
    ADT_LET_CONST_REF(rhs, args.at(0).template CastTo<Self>());
    return GetDimExprClass().New(lhs / rhs);
  }

  static adt::Result<axpr::Value> Match(
      axpr::InterpreterBase<axpr::Value>* interpreter,
      const axpr::Value& self_val,
      const std::vector<axpr::Value>& packed_args_val) {
    ADT_LET_CONST_REF(self, self_val.template CastTo<Self>());
    const auto& packed_args =
        axpr::CastToPackedArgs<axpr::Value>(packed_args_val);
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
          [&](int64_t c) -> adt::List<axpr::Value> {
            return adt::List<axpr::Value>{c};
          },
          [&](const std::string& c) -> adt::List<axpr::Value> {
            return adt::List<axpr::Value>{c};
          },
          [&](const auto&) -> adt::List<axpr::Value> {
            return adt::List<Value>{};
          });
      return interpreter->InterpretCall(func, make_args.vector());
    }
  }

  const char* GetTypeName(const symbol::DimExpr& dim_expr) const {
    return dim_expr.Match(
        [](int64_t) -> const char* { return "d_int64"; },
        [&](const std::string&) -> const char* { return "d_symbol"; },
        [&](const auto&) -> const char* { return "_"; });
  }
};

axpr::TypeImpl<axpr::BuiltinClassInstance<axpr::Value>> GetDimExprClass() {
  using Impl = DimExprMethodClass;
  static auto cls(
      axpr::MakeBuiltinClass<axpr::Value>("DimExpr", [&](const auto& Define) {
        Define("__str__", &Impl::ToString);
        Define("__add__", &Impl::Add);
        Define("__sub__", &Impl::Sub);
        Define("__mul__", &Impl::Mul);
        Define("__floordiv__", &Impl::FloorDiv);
        Define("__hash__", &Impl::Hash);
        Define("match", &Impl::Match);
      }));
  return axpr::MakeGlobalNaiveClassOps<typename Impl::Self>(cls);
}

adt::Result<axpr::Value> MakeInt64DimExpr(
    const axpr::Value&, const std::vector<axpr::Value>& args) {
  ADT_CHECK(args.size() == 1);
  ADT_LET_CONST_REF(int_val, args.at(0).template CastTo<int64_t>());
  return GetDimExprClass().New(symbol::DimExpr{int_val});
}

adt::Result<axpr::Value> MakeSymbolDimExpr(
    const axpr::Value&, const std::vector<axpr::Value>& args) {
  ADT_CHECK(args.size() == 1);
  ADT_LET_CONST_REF(symbol_name, args.at(0).template CastTo<std::string>());
  return GetDimExprClass().New(symbol::DimExpr{symbol_name});
}

}  // namespace ap::axpr
