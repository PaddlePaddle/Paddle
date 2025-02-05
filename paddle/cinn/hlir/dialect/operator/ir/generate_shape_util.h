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

#include <functional>
#include <optional>
#include <vector>
#include "paddle/cinn/hlir/dialect/operator/ir/manual_op.h"
#include "paddle/cinn/hlir/dialect/operator/ir/symbol_bindings.h"
#include "paddle/pir/include/core/builder.h"
#include "paddle/pir/include/dialect/shape/utils/shape_or_data_expr.h"

namespace cinn::dialect {

::pir::Attribute ConvertDimExprToAttribute(pir::IrContext* ctx,
                                           const symbol::DimExpr& dim_expr);

std::optional<symbol::DimExpr> ConvertAttributeToDimExpr(
    ::pir::Attribute attribute);

std::optional<std::vector<symbol::DimExpr>> ConvertAttributeToDimExprs(
    ::pir::Attribute attribute);

symbol::DimExpr SubstituteDimExpr(
    const symbol::DimExpr& dim_expr,
    const std::function<std::optional<symbol::DimExpr>(
        const std::string& symbol_name)>& DimExpr4SymbolName);

std::function<std::optional<symbol::DimExpr>(const std::string& symbol_name)>
MakeGetterDimExpr4SymbolName(
    const SymbolBindings& symbol_bindings,
    const std::function<const symbol::ShapeOrDataDimExprs&(int in_tensor_idx)>&
        DimExpr4InputDim);

using ShapeOrDataDimExprs4ValueT =
    std::function<symbol::ShapeOrDataDimExprs(pir::Value)>;

// Returns true if success.
bool MakeGenerateShapeOpAttribute(
    pir::IrContext* ir_context,
    const ShapeOrDataDimExprs4ValueT& ShapeOrDataDimExprs4Value,
    const std::vector<symbol::DimExpr>& out_dim_exprs,
    const std::vector<pir::Value>& origin_inputs,
    std::vector<pir::Value>* minimal_inputs,
    std::vector<pir::Attribute>* output_dim_expr_attrs,
    SymbolBindings* symbol_bindings);

}  // namespace cinn::dialect
