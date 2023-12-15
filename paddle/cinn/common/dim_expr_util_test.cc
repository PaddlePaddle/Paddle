#include "paddle/cinn/common/dim_expr_util.h"
#include "gtest/gtest.h"
#include <atomic>

namespace cinn::common::test {
using namespace symbol;

namespace {

DimExpr BD(const DimExpr& lhs, const DimExpr& rhs) {
  return Broadcast<DimExpr>{{lhs, rhs}};
}

DimExpr MakeSymbolic() {
  static std::atomic<int64_t> cnt(0);
  return DimExpr{std::to_string(cnt++)};
}

DimExpr MakeConstant(std::int64_t value) { return DimExpr{value}; }

}  // namespace

TEST(DimExpr, flatten_bd) {
  DimExpr sym0 = MakeSymbolic();
  DimExpr sym1 = MakeSymbolic();
  DimExpr sym2 = MakeSymbolic();
  DimExpr origin = BD(BD(sym0, sym1), sym2);
  DimExpr expected = Broadcast<DimExpr>{{sym0, sym1, sym2}};
  ASSERT_EQ(SimplifyDimExpr(origin), expected);
}

TEST(Simplify, NumberAdd) {
  List<DimExpr> num_lists{DimExpr(5), Negative<DimExpr>(5)};
  DimExpr dim_expr{Add<DimExpr>{num_lists}};

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

TEST(Simplify, NumberNaiveMul) {
  List<DimExpr> num_lists{DimExpr(5), DimExpr(5)};
  DimExpr dim_expr{Mul<DimExpr>{num_lists}};

  DimExpr simplified_dim_expr = SimplifyDimExpr(dim_expr);
  ASSERT_TRUE((simplified_dim_expr.Has<std::int64_t>()));
  ASSERT_EQ((simplified_dim_expr.Get<std::int64_t>()), 25);
}

TEST(Simplify, NumberMul) {
  List<DimExpr> num_lists{DimExpr(5), Reciprocal<DimExpr>(5)};
  DimExpr dim_expr{Mul<DimExpr>{num_lists}};

  DimExpr simplified_dim_expr = SimplifyDimExpr(dim_expr);
  ASSERT_TRUE((simplified_dim_expr.Has<std::int64_t>()));
  ASSERT_EQ((simplified_dim_expr.Get<std::int64_t>()), 1);
}

TEST(Simplify, NestNumberAddMul) {
  List<DimExpr> num_lists{DimExpr(5), Reciprocal<DimExpr>(5)};
  List<DimExpr> sum_lists{DimExpr(0), Mul<DimExpr>{num_lists}};
  DimExpr dim_expr{Add<DimExpr>{sum_lists}};

  DimExpr simplified_dim_expr = SimplifyDimExpr(dim_expr);
  ASSERT_TRUE((simplified_dim_expr.Has<std::int64_t>()));
  ASSERT_EQ((simplified_dim_expr.Get<std::int64_t>()), 1);
}

TEST(Simplify, NestNumberMulAdd) {
  List<DimExpr> num_lists{DimExpr(5), Negative<DimExpr>(5)};
  List<DimExpr> product_lists{DimExpr(5), Add<DimExpr>{num_lists}};
  DimExpr dim_expr{Mul<DimExpr>{product_lists}};

  DimExpr simplified_dim_expr = SimplifyDimExpr(dim_expr);
  ASSERT_TRUE((simplified_dim_expr.Has<std::int64_t>()));
  ASSERT_EQ((simplified_dim_expr.Get<std::int64_t>()), 0);
}

TEST(Simplify, SymbolicMul) {
  DimExpr sym = MakeSymbolic();
  List<DimExpr> num_lists{DimExpr(1), sym};
  DimExpr dim_expr{Mul<DimExpr>{num_lists}};

  DimExpr simplified_dim_expr = SimplifyDimExpr(dim_expr);
  ASSERT_TRUE((simplified_dim_expr.Has<std::string>()));
  ASSERT_TRUE((simplified_dim_expr == sym));
}

TEST(Simplify, SymbolicMulUnit) {
  DimExpr sym = MakeSymbolic();
  List<DimExpr> num_lists{Reciprocal<DimExpr>(sym), sym};
  DimExpr dim_expr{Mul<DimExpr>{num_lists}};

  DimExpr simplified_dim_expr = SimplifyDimExpr(dim_expr);
  ASSERT_TRUE((simplified_dim_expr.Has<std::int64_t>()));
  ASSERT_EQ((simplified_dim_expr.Get<std::int64_t>()), 1);
}

TEST(Simplify, NestSymbolicMulAddUnit) {
  DimExpr sym = MakeSymbolic();
  List<DimExpr> sum_lists{DimExpr(6), Negative<DimExpr>{DimExpr(5)}};
  List<DimExpr> product_lists = List<DimExpr>{Add<DimExpr>{sum_lists}, sym};
  DimExpr dim_expr{Mul<DimExpr>{product_lists}};

  DimExpr simplified_dim_expr = SimplifyDimExpr(dim_expr);
  ASSERT_TRUE((simplified_dim_expr.Has<std::string>()));
  ASSERT_TRUE((simplified_dim_expr == sym));
}
}