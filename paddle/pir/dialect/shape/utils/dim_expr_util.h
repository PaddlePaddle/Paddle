// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

#include <optional>
#include "paddle/pir/core/builder.h"
#include "paddle/pir/dialect/shape/utils/dim_expr.h"

namespace symbol {

::pir::Attribute ConvertDimExprToAttribute(::pir::Builder* builder,
                                           const DimExpr& dim_expr);
std::optional<DimExpr> ConvertAttributeToDimExpr(::pir::Attribute attribute);

std::optional<DimExpr> SubstituteDimExpr(
    const DimExpr& dim_expr,
    const std::function<std::optional<DimExpr>(const std::string& symbol_name)>&
        DimExpr4SymbolName);

std::function<std::optional<DimExpr>(const std::string& symbol_name)>
MakeGetterDimExpr4SymbolName(
    const std::vector<std::tuple<std::string /*symbol_name*/,
                                 int /*in_tensor_idx*/,
                                 int /*in_tensor_dim_idx*/>>& symbol_bindings,
    const std::function<std::optional<DimExpr>(
        int in_tensor_idx, int in_tensor_dim_idx)>& DimExpr4InputDim);

}  // namespace symbol
