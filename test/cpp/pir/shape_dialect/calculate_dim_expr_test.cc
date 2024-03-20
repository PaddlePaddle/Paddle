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

#include "paddle/cinn/common/dim_expr_util.h"
#include "paddle/pir/include/dialect/shape/utils/dim_expr_simplify.h"

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

TEST(DimExprUtil, Calculate) {
  // (S0 - S1) * 2 / S0
  DimExpr dim_expr = CreateExampleDimExpr();
  // (4 - 2) * 2 / 4 => 1
  DimExpr substitute_expr =
      cinn::common::SubstituteDimExpr(dim_expr, {{"S0", 4}, {"S1", 2}});
  DimExpr ret = SimplifyDimExpr(substitute_expr);
  ASSERT_TRUE(ret.Has<std::int64_t>());
  ASSERT_EQ(ret.Get<std::int64_t>(), 1);
}

}  // namespace symbol::test
