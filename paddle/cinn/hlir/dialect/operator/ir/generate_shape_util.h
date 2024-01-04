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
#include "paddle/pir/core/builder.h"
#include "paddle/pir/dialect/shape/utils/dim_expr.h"

namespace cinn::dialect {

::pir::Attribute ConvertDimExprToAttribute(pir::IrContext* ctx,
                                           const symbol::DimExpr& dim_expr);

std::optional<symbol::DimExpr> ConvertAttributeToDimExpr(
    ::pir::Attribute attribute);

std::optional<symbol::DimExpr> SubstituteDimExpr(
    const symbol::DimExpr& dim_expr,
    const std::function<std::optional<symbol::DimExpr>(
        const std::string& symbol_name)>& DimExpr4SymbolName);

std::function<std::optional<symbol::DimExpr>(const std::string& symbol_name)>
MakeGetterDimExpr4SymbolName(
    const std::vector<std::tuple<std::string /*symbol_name*/,
                                 int /*in_tensor_idx*/,
                                 int /*in_tensor_dim_idx*/>>& symbol_bindings,
    const std::function<std::optional<symbol::DimExpr>(
        int in_tensor_idx, int in_tensor_dim_idx)>& DimExpr4InputDim);

std::function<std::optional<symbol::DimExpr>(const std::string& symbol_name)>
MakeGetterDimExpr4SymbolName(
    const GenerateShapeOp::SymbolBindings& symbol_bindings,
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
    std::vector<pir::Value>* minial_inputs,
    std::vector<pir::Attribute>* output_dim_expr_attrs,
    GenerateShapeOp::SymbolBindings* symbol_bindings);

}  // namespace cinn::dialect
