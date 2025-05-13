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

#include "paddle/ap/include/axpr/atomic_axpr_method_class.h"
#include "paddle/ap/include/axpr/axpr_method_class.h"

namespace ap::axpr {

template <typename ValueT>
struct AtomicAxprMethodClass {
  using This = AtomicAxprMethodClass;
  using Self = axpr::Atomic<axpr::CoreExpr>;

  static adt::Result<ValueT> ToString(const ValueT& self_val,
                                      const std::vector<ValueT>& args) {
    ADT_LET_CONST_REF(self, self_val.template CastTo<Self>());
    std::ostringstream ss;
    ss << axpr::CoreExpr{self};
    return ss.str();
  }

  static adt::Result<ValueT> Hash(const ValueT& self_val,
                                  const std::vector<ValueT>&) {
    ADT_LET_CONST_REF(self, self_val.template CastTo<Self>());
    int64_t hash_value = std::hash<axpr::CoreExpr>()(axpr::CoreExpr{self});
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
        "AtomicAxpr.match() supports keyword arguments only, but " +
        std::to_string(args->size()) + " positional arguments were given"};
    const std::string& type_name = This{}.GetTypeNameImpl(self);
    std::string key = type_name;
    if (!kwargs->Has(type_name)) {
      if (!kwargs->Has("_")) {
        return adt::errors::TypeError{
            std::string() + "AtomicAxpr.match() failed. no keyword '" +
            type_name + "' or '_' provided"};
      }
      key = "_";
    }
    ADT_LET_CONST_REF(func, kwargs->Get(key));
    ADT_CHECK(axpr::CallableHelper{}.IsCallable(func))
        << adt::errors::TypeError{
               std::string() +
               "the arguments of AtomicAxpr.match() should be callable"};
    if (key == "_") {
      return interpreter->InterpretCall(func, {});
    } else {
      const auto& make_args = self.Match(
          [&](const axpr::Symbol& impl) -> adt::List<ValueT> {
            return adt::List<ValueT>{axpr::Value{impl.Name()}};
          },
          [&](const adt::Nothing&) -> adt::List<ValueT> {
            return adt::List<ValueT>{};
          },
          [&](const bool& c) -> adt::List<ValueT> {
            return adt::List<ValueT>{axpr::Value{c}};
          },
          [&](const int64_t& c) -> adt::List<ValueT> {
            return adt::List<ValueT>{axpr::Value{c}};
          },
          [&](const double& c) -> adt::List<ValueT> {
            return adt::List<ValueT>{axpr::Value{c}};
          },
          [&](const std::string& c) -> adt::List<ValueT> {
            return adt::List<ValueT>{axpr::Value{c}};
          },
          [&](const axpr::Lambda<axpr::CoreExpr>& lambda) -> adt::List<ValueT> {
            adt::List<axpr::Value> args{};
            args->reserve(lambda->args.size());
            for (const auto& arg : lambda->args) {
              args->emplace_back(arg.value());
            }
            const auto& body_expr = axpr::GetAxprClass().New(lambda->body);
            return adt::List<ValueT>{args, body_expr};
          });
      return interpreter->InterpretCall(func, make_args.vector());
    }
  }

  const char* GetTypeNameImpl(const Self& atomic_expr) const {
    return atomic_expr.Match(
        [&](const axpr::Symbol&) -> const char* { return "axpr_symbol"; },
        [&](const adt::Nothing&) -> const char* { return "axpr_none"; },
        [&](const bool&) -> const char* { return "axpr_bool"; },
        [&](const int64_t&) -> const char* { return "axpr_int"; },
        [&](const double&) -> const char* { return "axpr_float"; },
        [&](const std::string&) -> const char* { return "axpr_str"; },
        [&](const axpr::Lambda<axpr::CoreExpr>&) -> const char* {
          return "axpr_lambda";
        });
  }

  static adt::Result<ValueT> GetTypeName(const ValueT& self_val,
                                         const std::vector<ValueT>& args) {
    ADT_LET_CONST_REF(self, self_val.template CastTo<Self>());
    return std::string(This{}.GetTypeNameImpl(self));
  }
};

axpr::TypeImpl<axpr::BuiltinClassInstance<axpr::Value>> GetAtomicAxprClass() {
  using Impl = AtomicAxprMethodClass<axpr::Value>;
  static auto cls(axpr::MakeBuiltinClass<axpr::Value>(
      "AtomicAxpr", [&](const auto& Define) {
        Define("__str__", &Impl::ToString);
        Define("__hash__", &Impl::Hash);
        Define("match", &Impl::Match);
        Define("get_type_name", &Impl::GetTypeName);
      }));
  return axpr::MakeGlobalNaiveClassOps<typename Impl::Self>(cls);
}

}  // namespace ap::axpr
