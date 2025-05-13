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

#include "paddle/ap/include/axpr/axpr_method_class.h"
#include "paddle/ap/include/axpr/atomic_axpr_method_class.h"

namespace ap::axpr {

template <typename ValueT>
struct AxprMethodClass {
  using This = AxprMethodClass;
  using Self = axpr::CoreExpr;

  static adt::Result<ValueT> ToString(const ValueT& self_val,
                                      const std::vector<ValueT>& args) {
    ADT_LET_CONST_REF(self, self_val.template CastTo<Self>());
    std::ostringstream ss;
    ss << self;
    return ss.str();
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
        std::string() + "Axpr.match() supports keyword arguments only, but " +
        std::to_string(args->size()) + " positional arguments were given"};
    const std::string& type_name = This{}.GetTypeNameImpl(self);
    std::string key = type_name;
    if (!kwargs->Has(type_name)) {
      if (!kwargs->Has("_")) {
        return adt::errors::TypeError{std::string() +
                                      "Axpr.match() failed. no keyword '" +
                                      type_name + "' or '_' provided"};
      }
      key = "_";
    }
    ADT_LET_CONST_REF(func, kwargs->Get(key));
    ADT_CHECK(axpr::CallableHelper{}.IsCallable(func))
        << adt::errors::TypeError{
               std::string() +
               "the arguments of Axpr.match() should be callable"};
    if (key == "_") {
      return interpreter->InterpretCall(func, {});
    } else {
      const auto& make_args = self.Match(
          [&](const axpr::Atomic<axpr::CoreExpr>& atomic) -> adt::List<ValueT> {
            return adt::List<ValueT>{axpr::GetAtomicAxprClass().New(atomic)};
          },
          [&](const axpr::ComposedCallAtomic<axpr::CoreExpr>& call)
              -> adt::List<ValueT> {
            const auto& outer_func =
                axpr::GetAtomicAxprClass().New(call->outer_func);
            const auto& inner_func =
                axpr::GetAtomicAxprClass().New(call->inner_func);
            adt::List<axpr::Value> args{};
            args->reserve(call->args.size());
            for (const auto& arg : call->args) {
              args->emplace_back(axpr::GetAtomicAxprClass().New(arg));
            }
            return adt::List<ValueT>{outer_func, inner_func, args};
          });
      return interpreter->InterpretCall(func, make_args.vector());
    }
  }

  const char* GetTypeNameImpl(const Self& core_expr) const {
    return core_expr.Match(
        [](const axpr::Atomic<axpr::CoreExpr>&) -> const char* {
          return "axpr_atomic";
        },
        [&](const axpr::ComposedCallAtomic<axpr::CoreExpr>&) -> const char* {
          return "axpr_call";
        });
  }

  static adt::Result<ValueT> GetTypeName(const ValueT& self_val,
                                         const std::vector<ValueT>& args) {
    ADT_LET_CONST_REF(self, self_val.template CastTo<Self>());
    return std::string(This{}.GetTypeNameImpl(self));
  }
};

axpr::TypeImpl<axpr::BuiltinClassInstance<axpr::Value>> GetAxprClass() {
  using Impl = AxprMethodClass<axpr::Value>;
  static auto cls(
      axpr::MakeBuiltinClass<axpr::Value>("Axpr", [&](const auto& Define) {
        Define("__str__", &Impl::ToString);
        Define("__hash__", &Impl::Hash);
        Define("match", &Impl::Match);
        Define("get_type_name", &Impl::GetTypeName);
      }));
  return axpr::MakeGlobalNaiveClassOps<typename Impl::Self>(cls);
}

}  // namespace ap::axpr
