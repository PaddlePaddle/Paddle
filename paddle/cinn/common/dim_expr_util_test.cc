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

#include "paddle/cinn/common/dim_expr_util.h"

#include "gtest/gtest.h"

namespace cinn::dialect {
using namespace symbol;  // NOLINT

namespace {
DimExpr CreateExampleDimExpr() {
  DimExprBuilder dim_expr_builder{nullptr};
  DimExpr sym0 = DimExpr("S0");
  DimExpr sym1 = DimExpr("S1");
  DimExpr constant = DimExpr(2);
  DimExpr expr1 = (sym0 - sym1) * constant / sym0;
  DimExpr expr2 = dim_expr_builder.Max(expr1, sym0);
  DimExpr output = dim_expr_builder.Min(expr2, sym1);
  return output;
}
}  // namespace

TEST(DimExprUtil, Substitute) {
  DimExpr dim_expr = CreateExampleDimExpr();
  const auto& opt_expr = SubstituteDimExpr(
      dim_expr, [](const DimExpr& expr) -> std::optional<DimExpr> {
        if (expr == DimExpr("S0")) {
          return DimExpr("symbol0");
        } else if (expr == DimExpr("S1")) {
          return DimExpr("symbol1");
        } else {
          return std::nullopt;
        }
      });
  ASSERT_TRUE(opt_expr.has_value());
  const auto& ret_expr = SubstituteDimExpr(
      opt_expr.value(), [](const DimExpr& expr) -> std::optional<DimExpr> {
        if (expr == DimExpr("symbol0")) {
          return DimExpr("S0");
        } else if (expr == DimExpr("symbol1")) {
          return DimExpr("S1");
        } else {
          return std::nullopt;
        }
      });
  ASSERT_TRUE(ret_expr.has_value());
  ASSERT_EQ(ret_expr.value(), dim_expr);
}

}  // namespace cinn::dialect
