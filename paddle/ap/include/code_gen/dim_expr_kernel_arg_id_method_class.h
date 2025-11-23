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

#include "paddle/ap/include/axpr/method_class.h"
#include "paddle/ap/include/axpr/naive_class_ops.h"
#include "paddle/ap/include/code_gen/code_gen_ctx.h"
#include "paddle/ap/include/code_gen/dim_expr_kernel_arg_id.h"
#include "paddle/ap/include/code_gen/kernel_arg_id_helper.h"

namespace ap::code_gen {

template <typename ValueT, typename BirNode /* background ir node */>
struct DimExprKernelArgIdMethodClass {
  using This = DimExprKernelArgIdMethodClass;
  using Self = DimExprKernelArgId<BirNode>;

  static adt::Result<ValueT> ToString(const ValueT& self_val,
                                      const std::vector<ValueT>& args) {
    ADT_LET_CONST_REF(self, self_val.template CastTo<Self>());
    std::ostringstream ss;
    ss << self->dim_expr;
    return ss.str();
  }

  static adt::Result<ValueT> Hash(const ValueT& self_val,
                                  const std::vector<ValueT>& args) {
    ADT_LET_CONST_REF(self, self_val.template CastTo<Self>());
    std::size_t hash_value = std::hash<symbol::DimExpr>()(self->dim_expr);
    return static_cast<int64_t>(hash_value);
  }

  static adt::Result<ValueT> GetAttr(const ValueT& self_val,
                                     const std::vector<ValueT>& args) {
    ADT_LET_CONST_REF(self, self_val.template CastTo<Self>());
    ADT_CHECK(args.size() == 1);
    const auto& attr_name_val = args.at(0);
    ADT_LET_CONST_REF(attr_name, attr_name_val.template TryGet<std::string>());
    if (attr_name == "value") {
      return self->template CastData<ValueT>();
    }
    if (attr_name == "type") {
      return This{}.GetArgType(self);
    }
    if (attr_name == "runtime_getter") {
      ADT_CHECK(self->runtime_getter.has_value())
          << adt::errors::ValueError{"no runtime getter initialized"};
      return self->runtime_getter.value();
    }
    return adt::errors::AttributeError{
        std::string() + "'DimExprKernelArgId' instance has no attribute '" +
        attr_name + "'."};
  }

  adt::Result<ValueT> GetArgType(const Self& self) {
    KernelArgIdHelper<BirNode> helper;
    ADT_LET_CONST_REF(arg_type, helper.GetArgType(self));
    return arg_type.template CastTo<ValueT>();
  }
};

template <typename ValueT, typename BirNode>
axpr::TypeImpl<axpr::BuiltinClassInstance<ValueT>>
GetDimExprKernelArgIdClass() {
  using ImplMethods = DimExprKernelArgIdMethodClass<ValueT, BirNode>;
  static auto cls(axpr::MakeBuiltinClass<ValueT>(
      "DimExprKernelArgId", [&](const auto& Define) {
        Define("__str__", &ImplMethods::ToString);
        Define("__hash__", &ImplMethods::Hash);
        Define("__getattr__", &ImplMethods::GetAttr);
      }));
  using Self = typename ImplMethods::Self;
  return axpr::MakeGlobalNaiveClassOps<Self>(cls);
}

}  // namespace ap::code_gen
