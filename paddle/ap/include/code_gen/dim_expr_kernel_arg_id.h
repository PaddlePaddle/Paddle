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
#include "paddle/ap/include/axpr/function.h"
#include "paddle/ap/include/axpr/serializable_value.h"
#include "paddle/ap/include/axpr/type.h"
#include "paddle/pir/include/dialect/shape/utils/dim_expr.h"

namespace ap::code_gen {

template <typename BirNode>
struct DimExprKernelArgIdImpl {
  symbol::DimExpr dim_expr;
  std::optional<axpr::Function<axpr::SerializableValue>> runtime_getter;

  bool operator==(const DimExprKernelArgIdImpl& other) const {
    return this->dim_expr == other.dim_expr;
  }

  template <typename ValueT>
  adt::Result<ValueT> CastData() const {
    axpr::BuiltinClassInstance<ValueT> instance{axpr::GetDimExprClass<ValueT>(),
                                                this->dim_expr};
    return ValueT{instance};
  }

  std::size_t GetHashValue() const {
    return std::hash<symbol::DimExpr>()(this->dim_expr);
  }
};

template <typename BirNode>
ADT_DEFINE_RC(DimExprKernelArgId, DimExprKernelArgIdImpl<BirNode>);

template <typename ValueT, typename BirNode>
axpr::TypeImpl<axpr::BuiltinClassInstance<ValueT>> GetDimExprKernelArgIdClass();

}  // namespace ap::code_gen
