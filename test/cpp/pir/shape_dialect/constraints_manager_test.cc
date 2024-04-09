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

#include "paddle/pir/include/dialect/shape/utils/constraints_manager.h"
#include "paddle/pir/include/dialect/shape/utils/dim_expr_builder.h"

namespace symbol::test {

TEST(ConstraintsManager, AddEqCstr) {
  ConstraintsManager cstr_mgr;
  DimExprBuilder builder(nullptr);

  // Eq(Mul(S0,S1),Mul(S2,S3))
  DimExpr sym_expr_0 = builder.Symbol("S0");
  DimExpr sym_expr_1 = builder.Symbol("S1");
  DimExpr sym_expr_2 = builder.Symbol("S2");
  DimExpr sym_expr_3 = builder.Symbol("S3");
  DimExpr mul_expr_0 = builder.Mul(sym_expr_0, sym_expr_1);
  DimExpr mul_expr_1 = builder.Mul(sym_expr_2, sym_expr_3);
  cstr_mgr.AddEqCstr(mul_expr_0, mul_expr_1);
  ASSERT_TRUE(cstr_mgr.IsEqual(mul_expr_0, mul_expr_1));

  // Eq(Add(S0,S1),Add(S0,1))
  DimExpr int_expr_1 = builder.ConstSize(1);
  DimExpr add_expr_0 = builder.Add(sym_expr_0, sym_expr_1);
  DimExpr add_expr_1 = builder.Add(sym_expr_0, int_expr_1);
  cstr_mgr.AddEqCstr(add_expr_0, add_expr_1);
  ASSERT_FALSE(cstr_mgr.IsEqual(add_expr_0, add_expr_1));
  DimExpr mul_expr_2 = builder.Mul(sym_expr_0, int_expr_1);
  ASSERT_TRUE(cstr_mgr.IsEqual(mul_expr_2, mul_expr_1));
}

}  // namespace symbol::test
