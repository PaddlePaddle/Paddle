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
#include "paddle/pir/dialect/shape/ir/shape_dialect.h"
#include "test/cpp/pir/tools/test_pir_utils.h"

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

TEST(DimExpr, equal) {
  DimExpr sym0 = DimExpr("S0");
  DimExpr sym1 = DimExpr("S1");
  DimExpr constant1 = DimExpr(1);
  ASSERT_EQ(sym0 + sym1, sym0 + sym1);
  ASSERT_NE(sym0 + sym1, sym1 + sym0);
  ASSERT_EQ(sym0 + constant1, DimExpr("S0") + constant1);
}

}  // namespace symbol::test
