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
#include "paddle/pir/dialect/shape/utils/dim_expr_builder.h"

#include "paddle/fluid/pir/dialect/operator/ir/op_dialect.h"
#include "paddle/fluid/pir/dialect/operator/ir/pd_op.h"
#include "paddle/pir/core/ir_context.h"

namespace symbol::test {

// Construct DimExpr by overloaded operator(+, - , *, /)
TEST(DimExpr, dim_expr_naive) {
  DimExpr sym0 = DimExpr("S0");
  DimExpr sym1 = DimExpr("S1");
  DimExpr constant1 = DimExpr(1);
  DimExpr output = (sym0 + sym1) * constant1;
}

// Construct DimExpr by DimExprBuilder
TEST(DimExpr, dim_expr_builder) {
  DimExprBuilder builder{nullptr};
  DimExpr sym0 = DimExpr("S0");
  DimExpr sym1 = DimExpr("S1");
  DimExpr constant1 = DimExpr(1);
  DimExpr add = builder.Add(sym0, sym1);
  DimExpr out = builder.Broadcast(add, constant1);
}

// Add constraints by DimExprBuilder
TEST(DimExpr, constraint) {
  std::vector<DimExprConstraint> constraints{};
  DimExprBuilder builder(&constraints);
  DimExpr sym0 = DimExpr("S0");
  DimExpr sym1 = DimExpr("S1");
  builder.CstrEq(sym0, sym1);
  ASSERT_EQ(static_cast<int>(constraints.size()), 1);
}

/*
  Simulate the ShapeOrDataDimExprs result of below codes:
  def (x, y):
    extend_x = x.shape
    out = pd.reshape(y, extend_x)
*/
TEST(DimExpr, data_shape_expr) {
  // Show ideal ShapeOrDataDimExprs of each pir::Value
  std::vector<DimExpr> x_shapes{DimExpr("S0"), DimExpr(2)};
  std::vector<DimExpr> y_shapes{DimExpr(1), DimExpr("S1"), DimExpr(2)};
  // x => {shape: [S0, 2], data: nullopt}
  ShapeOrDataDimExprs x_data_shape{x_shapes};
  // y => {shape: [1, S1, 2], data: nullopt}
  ShapeOrDataDimExprs y_data_shape{y_shapes};
  // extend_x => {shape: [2], data: [S0, 2]}
  ShapeOrDataDimExprs extend_x_data_shape =
      ShapeOrDataDimExprs::MakeConsistentShapeOrData(x_shapes);
  // out => {shape: [S0, 2], data: nullopt}
  ShapeOrDataDimExprs out_value_shape{x_shapes};
}

TEST(Simplify, NumberArithmetic) {
  DimExpr number = DimExpr(5);
  DimExpr add_minus = number + number - number;
  ASSERT_TRUE((add_minus.Has<std::int64_t>()));
  ASSERT_EQ((add_minus.Get<std::int64_t>()), 5);
  DimExpr mul_div = number * DimExpr(1) / number;
  ASSERT_TRUE((mul_div.Has<std::int64_t>()));
  ASSERT_EQ((mul_div.Get<std::int64_t>()), 1);
}

TEST(DimExpr, equal) {
  DimExprBuilder builder{nullptr};
  DimExpr sym0 = DimExpr("S0");
  DimExpr sym1 = DimExpr("S1");
  DimExpr constant1 = DimExpr(1);
  ASSERT_EQ(sym0 + sym1, sym0 + sym1);
  ASSERT_NE(sym0 + sym1, sym1 + sym0);
  ASSERT_EQ(sym0 + constant1, DimExpr("S0") + constant1);
  ASSERT_EQ(sym0 - sym1, sym0 - sym1);
  ASSERT_NE(sym0 - sym1, sym1 - sym0);
  ASSERT_EQ(sym0 - constant1, DimExpr("S0") - constant1);
  ASSERT_EQ(sym0 * sym1, sym0 * sym1);
  ASSERT_NE(sym0 * sym1, sym1 * sym0);
  ASSERT_EQ(sym0 * constant1, DimExpr("S0") * constant1);
  ASSERT_EQ(sym0 / sym1, sym0 / sym1);
  ASSERT_NE(sym0 / sym1, sym1 / sym0);
  ASSERT_EQ(sym0 / constant1, DimExpr("S0") / constant1);
  ASSERT_EQ(builder.Max(sym0, sym1), builder.Max(sym0, sym1));
  ASSERT_NE(builder.Max(sym0, sym1), builder.Max(sym1, sym0));
  ASSERT_EQ(builder.Max(sym0, constant1),
            builder.Max(DimExpr("S0"), constant1));
  ASSERT_EQ(builder.Min(sym0, sym1), builder.Min(sym0, sym1));
  ASSERT_NE(builder.Min(sym0, sym1), builder.Min(sym1, sym0));
  ASSERT_EQ(builder.Min(sym0, constant1),
            builder.Min(DimExpr("S0"), constant1));
  ASSERT_EQ(builder.Broadcast(sym0, sym1), builder.Broadcast(sym0, sym1));
  ASSERT_NE(builder.Broadcast(sym0, sym1), builder.Broadcast(sym1, sym0));
  ASSERT_EQ(builder.Broadcast(sym0, constant1),
            builder.Broadcast(DimExpr("S0"), constant1));
}

TEST(DimExpr, print) {
  DimExprBuilder builder{nullptr};
  DimExpr sym0 = DimExpr("S0");
  DimExpr sym1 = DimExpr("S1");
  ASSERT_EQ((ToString(sym0 + sym1)), "Add(S0, S1)");
  ASSERT_EQ((ToString(sym0 - sym1)), "Add(S0, -S1)");
  ASSERT_EQ((ToString(sym0 * sym1)), "Mul(S0, S1)");
  ASSERT_EQ((ToString(sym0 / sym1)), "Mul(S0, 1 / (S1))");
  ASSERT_EQ((ToString(builder.Max(sym0, sym1))), "Max(S0, S1)");
  ASSERT_EQ((ToString(builder.Min(sym0, sym1))), "Min(S0, S1)");
  ASSERT_EQ((ToString(builder.Broadcast(sym0, sym1))), "Broadcast(S0, S1)");
}

}  // namespace symbol::test
