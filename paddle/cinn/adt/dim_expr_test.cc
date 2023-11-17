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

TEST(Simplify, NumberSum) {
  List<DimExpr> num_lists{DimExpr(5), Negative<DimExpr>(5)};
  DimExpr dim_expr{Sum<DimExpr>{num_lists}};

  DimExpr simplified_dim_expr = SimplifyDimExpr(dim_expr);
  ASSERT_TRUE((simplified_dim_expr.Has<std::int64_t>()));
  ASSERT_EQ((simplified_dim_expr.Get<std::int64_t>()), 0);
}

TEST(Simplify, UnitReciprocal) {
  DimExpr unit{Reciprocal<DimExpr>{DimExpr{1}}};

  DimExpr simplified_dim_expr = SimplifyDimExpr(unit);
  ASSERT_TRUE((simplified_dim_expr.Has<std::int64_t>()));
  ASSERT_EQ((simplified_dim_expr.Get<std::int64_t>()), 1);
}

TEST(Simplify, UnitNegative) {
  DimExpr unit{Negative<DimExpr>{DimExpr{0}}};

  DimExpr simplified_dim_expr = SimplifyDimExpr(unit);
  ASSERT_TRUE((simplified_dim_expr.Has<std::int64_t>()));
  ASSERT_EQ((simplified_dim_expr.Get<std::int64_t>()), 0);
}

TEST(Simplify, NumberNaiveProduct) {
  List<DimExpr> num_lists{DimExpr(5), DimExpr(5)};
  DimExpr dim_expr{Product<DimExpr>{num_lists}};

  DimExpr simplified_dim_expr = SimplifyDimExpr(dim_expr);
  ASSERT_TRUE((simplified_dim_expr.Has<std::int64_t>()));
  ASSERT_EQ((simplified_dim_expr.Get<std::int64_t>()), 25);
}

TEST(Simplify, NumberProduct) {
  List<DimExpr> num_lists{DimExpr(5), Reciprocal<DimExpr>(5)};
  DimExpr dim_expr{Product<DimExpr>{num_lists}};

  DimExpr simplified_dim_expr = SimplifyDimExpr(dim_expr);
  ASSERT_TRUE((simplified_dim_expr.Has<std::int64_t>()));
  ASSERT_EQ((simplified_dim_expr.Get<std::int64_t>()), 1);
}

TEST(Simplify, NestNumberSumProduct) {
  List<DimExpr> num_lists{DimExpr(5), Reciprocal<DimExpr>(5)};
  List<DimExpr> sum_lists{DimExpr(0), Product<DimExpr>{num_lists}};
  DimExpr dim_expr{Sum<DimExpr>{sum_lists}};

  DimExpr simplified_dim_expr = SimplifyDimExpr(dim_expr);
  ASSERT_TRUE((simplified_dim_expr.Has<std::int64_t>()));
  ASSERT_EQ((simplified_dim_expr.Get<std::int64_t>()), 1);
}

TEST(Simplify, NestNumberProductSum) {
  List<DimExpr> num_lists{DimExpr(5), Negative<DimExpr>(5)};
  List<DimExpr> product_lists{DimExpr(5), Sum<DimExpr>{num_lists}};
  DimExpr dim_expr{Product<DimExpr>{product_lists}};

  DimExpr simplified_dim_expr = SimplifyDimExpr(dim_expr);
  ASSERT_TRUE((simplified_dim_expr.Has<std::int64_t>()));
  ASSERT_EQ((simplified_dim_expr.Get<std::int64_t>()), 0);
}

TEST(Simplify, SymbolicProduct) {
  DimExpr sym = MakeSymbolic();
  List<DimExpr> num_lists{DimExpr(1), sym};
  DimExpr dim_expr{Product<DimExpr>{num_lists}};

  DimExpr simplified_dim_expr = SimplifyDimExpr(dim_expr);
  ASSERT_TRUE((simplified_dim_expr.Has<SymbolicDim>()));
  ASSERT_TRUE((simplified_dim_expr == sym));
}

TEST(Simplify, SymbolicProductUnit) {
  DimExpr sym = MakeSymbolic();
  List<DimExpr> num_lists{Reciprocal<DimExpr>(sym), sym};
  DimExpr dim_expr{Product<DimExpr>{num_lists}};

  DimExpr simplified_dim_expr = SimplifyDimExpr(dim_expr);
  ASSERT_TRUE((simplified_dim_expr.Has<std::int64_t>()));
  ASSERT_EQ((simplified_dim_expr.Get<std::int64_t>()), 1);
}

TEST(Simplify, NestSymbolicProductSumUnit) {
  DimExpr sym = MakeSymbolic();
  List<DimExpr> sum_lists{DimExpr(6), Negative<DimExpr>{DimExpr(5)}};
  List<DimExpr> product_lists = List<DimExpr>{Sum<DimExpr>{sum_lists}, sym};
  DimExpr dim_expr{Product<DimExpr>{product_lists}};

  DimExpr simplified_dim_expr = SimplifyDimExpr(dim_expr);
  ASSERT_TRUE((simplified_dim_expr.Has<SymbolicDim>()));
  ASSERT_TRUE((simplified_dim_expr == sym));
}

}  // namespace cinn::adt::test
