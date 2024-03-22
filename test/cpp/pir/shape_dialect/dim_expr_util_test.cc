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

#include "gtest/gtest.h"

#include "paddle/pir/include/dialect/shape/utils/dim_expr_builder.h"
#include "paddle/pir/include/dialect/shape/utils/dim_expr_util.h"

namespace symbol::test {

namespace {

// (S0 - S1) * 2 / S0
DimExpr CreateExampleDimExpr() {
  DimExpr sym0 = DimExpr("S0");
  DimExpr sym1 = DimExpr("S1");
  DimExpr constant = DimExpr(2);
  return (sym0 - sym1) * constant / sym0;
}
}  // namespace

TEST(DimExprUtil, SimplifyNeg) {
  DimExpr dim_expr = Negative<DimExpr>{-1};
  DimExpr ret = SimplifyDimExpr(dim_expr);
  ASSERT_TRUE(ret.Has<std::int64_t>());
  ASSERT_EQ(ret.Get<std::int64_t>(), 1);
  DimExpr double_neg_expr = Negative<DimExpr>{dim_expr};
  ret = SimplifyDimExpr(double_neg_expr);
  ASSERT_TRUE(ret.Has<std::int64_t>());
  ASSERT_EQ(ret.Get<std::int64_t>(), -1);
}

TEST(DimExprUtil, Substitute) {
  DimExpr dim_expr = CreateExampleDimExpr();
  std::unordered_map<symbol::DimExpr, symbol::DimExpr> naive_to_full_name{
      {DimExpr("S0"), DimExpr("symbol0")}, {DimExpr("S1"), DimExpr("symbol1")}};
  std::unordered_map<symbol::DimExpr, symbol::DimExpr> full_name_to_naive{
      {DimExpr("symbol0"), DimExpr("S0")}, {DimExpr("symbol1"), DimExpr("S1")}};

  const auto& mid_expr = SubstituteDimExpr(dim_expr, naive_to_full_name);
  const auto& ret_expr = SubstituteDimExpr(mid_expr, full_name_to_naive);
  ASSERT_EQ(ret_expr, dim_expr);
}

TEST(DimExprUtil, Calculate) {
  // (S0 - S1) * 2 / S0
  DimExpr dim_expr = CreateExampleDimExpr();
  // (4 - 2) * 2 / 4 => 1
  DimExpr substitute_expr = SubstituteDimExpr(dim_expr, {{"S0", 4}, {"S1", 2}});
  DimExpr ret = SimplifyDimExpr(substitute_expr);
  ASSERT_TRUE(ret.Has<std::int64_t>());
  ASSERT_EQ(ret.Get<std::int64_t>(), 1);
}

TEST(DimExpr, CollectDimExprSymbol) {
  DimExpr dim_expr = [&]() -> DimExpr {
    DimExprBuilder builder(nullptr);
    DimExpr max_expr = builder.Max(DimExpr("S2"), DimExpr("S3"));
    DimExpr min_expr = builder.Min(max_expr, DimExpr("S4"));
    DimExpr broadcast_expr = builder.Broadcast(min_expr, DimExpr("S5"));
    return CreateExampleDimExpr() + broadcast_expr;
  }();
  std::unordered_set<std::string> symbols = CollectDimExprSymbols(dim_expr);
  std::unordered_set<std::string> expected = {
      "S0", "S1", "S2", "S3", "S4", "S5"};
  ASSERT_EQ(symbols.size(), 6UL);
  for (const auto& symbol : symbols) {
    ASSERT_TRUE(expected.find(symbol) != expected.end());
  }
}

}  // namespace symbol::test
