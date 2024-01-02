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

namespace cinn::common {
using namespace symbol;  // NOLINT

namespace {
DimExpr CreateExampleDimExpr() {
  DimExpr sym0 = DimExpr("S0");
  DimExpr sym1 = DimExpr("S1");
  DimExpr constant = DimExpr(2);
  return (sym0 - sym1) * constant / sym0;
}
}  // namespace

TEST(DimExprUtil, Substitute) {
  DimExpr dim_expr = CreateExampleDimExpr();
  std::unordered_map<symbol::DimExpr, symbol::DimExpr> naive_to_full_name{
      {DimExpr("S0"), DimExpr("symbol0")}, {DimExpr("S1"), DimExpr("symbol1")}};
  std::unordered_map<symbol::DimExpr, symbol::DimExpr> full_name_to_naive{
      {DimExpr("symbol0"), DimExpr("S0")}, {DimExpr("symbol1"), DimExpr("S1")}};

  const auto& opt_expr = SubstituteDimExpr(dim_expr, naive_to_full_name);
  ASSERT_TRUE(opt_expr.has_value());
  const auto& ret_expr =
      SubstituteDimExpr(opt_expr.value(), full_name_to_naive);
  ASSERT_TRUE(ret_expr.has_value());
  ASSERT_EQ(ret_expr.value(), dim_expr);
}

}  // namespace cinn::common
