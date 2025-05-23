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

#include "paddle/ap/include/axpr/anf_expr_helper.h"
#include "paddle/ap/include/axpr/anf_expr_util.h"
#include "paddle/ap/include/axpr/function.h"
#include "paddle/ap/include/axpr/method_class.h"
#include "paddle/ap/include/axpr/serializable_value.h"

namespace ap::axpr {

template <typename ValueT>
struct MethodClassImpl<ValueT, Function<SerializableValue>>
    : public EmptyMethodClass<ValueT> {
  using This = MethodClassImpl<ValueT, Function<SerializableValue>>;
  using Self = Function<SerializableValue>;

  adt::Result<ValueT> ToString(const Self& function) {
    return ToStringImpl(function, "unnamed");
  }

  adt::Result<ValueT> ToStringImpl(const Self& function,
                                   const std::string& name) {
    const auto& lambda = function->lambda;
    const auto& anf_expr = ConvertCoreExprToAnfExpr(lambda);
    ADT_LET_CONST_REF(anf_atomic, anf_expr.template TryGet<Atomic<AnfExpr>>());
    ADT_LET_CONST_REF(anf_lambda,
                      anf_atomic.template TryGet<Lambda<AnfExpr>>());
    AnfExprHelper anf_expr_helper;
    ADT_LET_CONST_REF(anf_expr_str,
                      anf_expr_helper.FunctionToString(name, anf_lambda));
    return anf_expr_str;
  }

  static adt::Result<ValueT> FunctionToString(const ValueT& self_val,
                                              const std::vector<ValueT>& args) {
    ADT_LET_CONST_REF(self, self_val.template CastTo<Self>());
    std::string name = "unnamed";
    {
      ADT_CHECK(args.size() == 1 || args.size() == 0);
      if (args.size() == 1) {
        ADT_LET_CONST_REF(name_arg, args.at(0).template CastTo<std::string>());
        name = name_arg;
      }
    }
    return This{}.ToStringImpl(self, name);
  }

  adt::Result<ValueT> Hash(const Self& function) {
    ADT_LET_CONST_REF(hash_value, function->GetHashValue());
    return hash_value;
  }

  adt::Result<ValueT> GetAttr(const Self& self, const ValueT& attr_name_val) {
    ADT_LET_CONST_REF(attr_name, TryGetImpl<std::string>(attr_name_val));
    if (attr_name == "__function__") {
      return self;
    }
    if (attr_name == "to_string") {
      return axpr::Method<ValueT>{self, &This::FunctionToString};
    }
    return adt::errors::AttributeError{
        std::string() + "function has not attribute '" + attr_name + "'."};
  }
};

template <typename ValueT>
struct MethodClassImpl<ValueT, TypeImpl<Function<SerializableValue>>>
    : public EmptyMethodClass<ValueT> {};

}  // namespace ap::axpr
