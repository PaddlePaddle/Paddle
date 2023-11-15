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

#include "paddle/cinn/adt/dim_expr.h"
#include "gtest/gtest.h"
#include "paddle/cinn/adt/dim_expr_simplifier.h"

namespace cinn::adt::test {

namespace {

DimExpr BD(const DimExpr& lhs, const DimExpr& rhs) {
  return MakeBroadcastedDim(lhs, rhs);
}

DimExpr MakeSymbolic() { return DimExpr{SymbolicDim{UniqueId::New()}}; }

DimExpr MakeConstant(std::int64_t value) { return DimExpr{value}; }

}  // namespace

TEST(DimExpr, flatten_bd) {
  DimExpr sym0 = MakeSymbolic();
  DimExpr sym1 = MakeSymbolic();
  DimExpr sym2 = MakeSymbolic();
  DimExpr origin = BD(BD(sym0, sym1), sym2);
  DimExpr expected = BroadcastedDim<DimExpr>{List<DimExpr>{sym0, sym1, sym2}};
  ASSERT_EQ(SimplifyDimExpr(origin), expected);
}

}  // namespace cinn::adt::test
