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

#include "gtest/gtest.h"

#include "paddle/cinn/hlir/dialect/operator/ir/generate_shape_util.h"
#include "paddle/fluid/pir/dialect/operator/ir/op_dialect.h"
#include "paddle/pir/include/dialect/shape/utils/dim_expr_builder.h"

#include "test/cpp/pir/tools/test_pir_utils.h"

namespace cinn::dialect {
using namespace symbol;  // NOLINT

namespace {
DimExpr CreateExampleDimExpr() {
  DimExprBuilder dim_expr_builder;
  DimExpr sym0 = DimExpr("S0");
  DimExpr sym1 = DimExpr("S1");
  DimExpr constant = DimExpr(2);
  DimExpr expr1 = (sym0 - sym1) * constant / sym0;
  DimExpr expr2 = dim_expr_builder.Max(expr1, sym0);
  DimExpr output = dim_expr_builder.Min(expr2, sym1);
  return output;
}
}  // namespace

TEST(DimExprUtil, Convert) {
  pir::IrContext* ctx = pir::IrContext::Instance();

  DimExpr dim_expr = CreateExampleDimExpr();
  ::pir::Attribute attr = ConvertDimExprToAttribute(ctx, dim_expr);
  std::optional<DimExpr> opt_expr = ConvertAttributeToDimExpr(attr);
  ASSERT_TRUE(opt_expr.has_value());
  ASSERT_EQ(opt_expr.value(), dim_expr);
}

TEST(DimExprUtil, Substitute) {
  DimExpr dim_expr = CreateExampleDimExpr();
  const auto& substituted_expr = SubstituteDimExpr(
      dim_expr, [](const std::string& str) -> std::optional<DimExpr> {
        if (str == "S0") {
          return DimExpr("symbol0");
        } else if (str == "S1") {
          return DimExpr("symbol1");
        } else {
          return std::nullopt;
        }
      });
  const auto& ret_expr = SubstituteDimExpr(
      substituted_expr, [](const std::string& str) -> std::optional<DimExpr> {
        if (str == "symbol0") {
          return DimExpr("S0");
        } else if (str == "symbol1") {
          return DimExpr("S1");
        } else {
          return std::nullopt;
        }
      });
  ASSERT_EQ(ret_expr, dim_expr);
}

TEST(DimExprUtil, MakeGetterDimExpr4SymbolName) {
  cinn::dialect::GenerateShapeOp::SymbolBindings symbol_bindings{};
  using ShapeSymbolBinding = cinn::dialect::GenerateShapeOp::ShapeSymbolBinding;
  symbol_bindings.emplace_back(ShapeSymbolBinding{"Symbol", 0, 0});
  const auto& dim_expr = CreateExampleDimExpr();

  const auto& shape_or_data_dim_exprs = symbol::ShapeOrDataDimExprs(
      symbol::TensorShapeOrDataDimExprs({dim_expr}));

  const auto& DimExpr4SymbolName = MakeGetterDimExpr4SymbolName(
      symbol_bindings,
      [&](int in_tensor_idx) -> const symbol::ShapeOrDataDimExprs& {
        return shape_or_data_dim_exprs;
      });
  const auto& opt_dim_expr = DimExpr4SymbolName("Symbol");
  ASSERT_TRUE(opt_dim_expr.has_value());
  ASSERT_EQ(opt_dim_expr.value(), dim_expr);
}

}  // namespace cinn::dialect
