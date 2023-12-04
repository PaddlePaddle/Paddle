// Copyright (c) 2023 CINN Authors. All Rights Reserved.
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

#include "paddle/cinn/common/integer_set.h"

#include <glog/logging.h>
#include <gtest/gtest.h>
#include "paddle/cinn/ir/op/ir_operators.h"

namespace cinn {
namespace common {

class TestSymbolicExprAnalyzer : public ::testing::Test {
 public:
  void SetUp() override {
    i = ir::Var(ir::Expr(0), ir::Expr(7), "i");
    j = ir::Var(ir::Expr(0), ir::Expr(15), "j");
    var_intervals = {
        {"i", CasInterval(i->lower_bound, i->upper_bound)},
        {"j", CasInterval(j->lower_bound, j->upper_bound)},
    };
  }

  ir::Var i;
  ir::Var j;
  cas_intervals_t var_intervals;
  SymbolicExprAnalyzer analyzer = SymbolicExprAnalyzer(&var_intervals);
};

TEST_F(TestSymbolicExprAnalyzer, bound) {
  ir::Expr e1 = i + j;
  EXPECT_EQ(analyzer.LowerBound(e1), ir::Expr(0));
  EXPECT_EQ(analyzer.UpperBound(e1), ir::Expr(22));

  ir::Expr e2 = 16 * i + j;
  EXPECT_EQ(analyzer.LowerBound(e2), ir::Expr(0));
  EXPECT_EQ(analyzer.UpperBound(e2), ir::Expr(127));

  ir::Expr e3 = 16 * i + j + 1;
  EXPECT_EQ(analyzer.LowerBound(e3), ir::Expr(1));
  EXPECT_EQ(analyzer.UpperBound(e3), ir::Expr(128));

  ir::Expr e4 = (16 * i + j) / 16;
  EXPECT_EQ(analyzer.LowerBound(e4), ir::Expr(0));
  EXPECT_EQ(analyzer.UpperBound(e4), ir::Expr(7));

  ir::Expr e5 = (16 * i + j) % 16;
  EXPECT_EQ(analyzer.LowerBound(e5), ir::Expr(0));
  EXPECT_EQ(analyzer.UpperBound(e5), ir::Expr(15));

  ir::Expr e6 = i - j;
  EXPECT_EQ(analyzer.LowerBound(e6), ir::Expr(-15));
  EXPECT_EQ(analyzer.UpperBound(e6), ir::Expr(7));

  ir::Expr e7 = 0 - i - j;
  EXPECT_EQ(analyzer.LowerBound(e7), ir::Expr(-22));
  EXPECT_EQ(analyzer.UpperBound(e7), ir::Expr(0));

  ir::Expr e8 = -1 * i - j;
  EXPECT_EQ(analyzer.LowerBound(e8), ir::Expr(-22));
  EXPECT_EQ(analyzer.UpperBound(e8), ir::Expr(0));
}

TEST_F(TestSymbolicExprAnalyzer, compare) {
  // case 1
  ir::Expr e1 = 4 * i + 2 * j;
  ir::Expr e2 = 2 * i + j;

  EXPECT_TRUE(analyzer.CanProveEQ(e1, e1) &&
              analyzer.CanProve(ir::EQ::Make(e1, e1)));
  EXPECT_FALSE(analyzer.CanProveEQ(e1, e2) &&
               analyzer.CanProve(ir::EQ::Make(e1, e2)));
  EXPECT_FALSE(analyzer.CanProveNE(e1, e1) &&
               analyzer.CanProve(ir::NE::Make(e1, e1)));
  EXPECT_FALSE(analyzer.CanProveNE(e1, e2) &&
               analyzer.CanProve(ir::NE::Make(e1, e2)));

  EXPECT_TRUE(analyzer.CanProveGE(e1, e2) && analyzer.CanProve(e1 >= e2));
  EXPECT_FALSE(analyzer.CanProveGE(e2, e1) && analyzer.CanProve(e2 >= e1));
  EXPECT_TRUE(analyzer.CanProveLE(e2, e1) && analyzer.CanProve(e2 <= e1));
  EXPECT_FALSE(analyzer.CanProveLE(e1, e2) && analyzer.CanProve(e1 <= e2));

  EXPECT_FALSE(analyzer.CanProveGT(e1, e2) && analyzer.CanProve(e1 > e2));
  EXPECT_FALSE(analyzer.CanProveGT(e2, e1) && analyzer.CanProve(e2 > e1));
  EXPECT_FALSE(analyzer.CanProveLT(e2, e1) && analyzer.CanProve(e2 < e1));
  EXPECT_FALSE(analyzer.CanProveLT(e1, e2) && analyzer.CanProve(e1 < e2));

  // case 2
  ir::Expr e3 = i + j + 1;
  ir::Expr e4 = i + j;

  EXPECT_TRUE(analyzer.CanProveEQ(e3, e3) &&
              analyzer.CanProve(ir::EQ::Make(e3, e3)));
  EXPECT_FALSE(analyzer.CanProveEQ(e3, e4) &&
               analyzer.CanProve(ir::EQ::Make(e3, e4)));
  EXPECT_TRUE(analyzer.CanProveNE(e3, e4) &&
              analyzer.CanProve(ir::NE::Make(e3, e4)));
  EXPECT_FALSE(analyzer.CanProveNE(e4, e4) &&
               analyzer.CanProve(ir::NE::Make(e4, e4)));

  EXPECT_TRUE(analyzer.CanProveGE(e3, e4) && analyzer.CanProve(e3 >= e4));
  EXPECT_FALSE(analyzer.CanProveGE(e4, e3) && analyzer.CanProve(e4 >= e3));
  EXPECT_TRUE(analyzer.CanProveLE(e4, e3) && analyzer.CanProve(e4 <= e3));
  EXPECT_FALSE(analyzer.CanProveLE(e3, e4) && analyzer.CanProve(e3 <= e4));

  EXPECT_TRUE(analyzer.CanProveGT(e3, e4) && analyzer.CanProve(e3 > e4));
  EXPECT_FALSE(analyzer.CanProveGT(e4, e3) && analyzer.CanProve(e4 > e3));
  EXPECT_TRUE(analyzer.CanProveLT(e4, e3) && analyzer.CanProve(e4 < e3));
  EXPECT_FALSE(analyzer.CanProveLT(e3, e4) && analyzer.CanProve(e3 < e4));
}

TEST(SingleIntervalIntSet, constant) {
  SingleIntervalIntSet empty_set(ir::Expr(0), ir::Expr(-1));
  SingleIntervalIntSet all_set(SymbolicExprLimit::negative_inf,
                               SymbolicExprLimit::positive_inf);
  SingleIntervalIntSet single_point(ir::Expr(0), ir::Expr(0));
  SingleIntervalIntSet interval_0_2_set(ir::Expr(0), ir::Expr(2));
  SingleIntervalIntSet interval_0_4_set(ir::Expr(0), ir::Expr(4));
  SingleIntervalIntSet interval_2_6_set(ir::Expr(2), ir::Expr(6));
  SingleIntervalIntSet interval_8_9_set(ir::Expr(8), ir::Expr(9));

  EXPECT_TRUE(empty_set.IsEmpty());
  EXPECT_FALSE(empty_set.IsAll());
  EXPECT_FALSE(all_set.IsEmpty());
  EXPECT_TRUE(all_set.IsAll());
  EXPECT_TRUE(single_point.IsPoint());
  EXPECT_FALSE(interval_0_2_set.IsPoint());
  EXPECT_TRUE(interval_0_2_set.IsSubSet(interval_0_4_set));
  EXPECT_FALSE(interval_0_4_set.IsSubSet(interval_0_2_set));
  EXPECT_FALSE(interval_0_2_set.IsSuperSet(interval_0_4_set));
  EXPECT_TRUE(interval_0_4_set.IsSuperSet(interval_0_2_set));

  EXPECT_TRUE(interval_0_2_set == interval_0_2_set);
  EXPECT_FALSE(interval_0_2_set == interval_0_4_set);

  SingleIntervalIntSet union_0_6_set =
      Union(interval_0_2_set, interval_2_6_set);
  EXPECT_EQ(union_0_6_set.Min(), ir::Expr(0));
  EXPECT_EQ(union_0_6_set.Max(), ir::Expr(6));
  union_0_6_set = Union(interval_2_6_set, interval_0_2_set);
  EXPECT_EQ(union_0_6_set.Min(), ir::Expr(0));
  EXPECT_EQ(union_0_6_set.Max(), ir::Expr(6));
  SingleIntervalIntSet union_0_4_set =
      Union(interval_0_2_set, interval_0_4_set);
  EXPECT_EQ(union_0_4_set.Min(), ir::Expr(0));
  EXPECT_EQ(union_0_4_set.Max(), ir::Expr(4));
  union_0_4_set = Union(interval_0_4_set, interval_0_2_set);
  EXPECT_EQ(union_0_4_set.Min(), ir::Expr(0));
  EXPECT_EQ(union_0_4_set.Max(), ir::Expr(4));
  SingleIntervalIntSet union_0_9_set =
      Union(interval_0_4_set, interval_8_9_set);
  EXPECT_EQ(union_0_9_set.Min(), ir::Expr(0));
  EXPECT_EQ(union_0_9_set.Max(), ir::Expr(9));
  union_0_9_set = Union(interval_8_9_set, interval_0_4_set);
  EXPECT_EQ(union_0_9_set.Min(), ir::Expr(0));
  EXPECT_EQ(union_0_9_set.Max(), ir::Expr(9));

  SingleIntervalIntSet intersect_0_2_set =
      Intersect(interval_0_2_set, interval_0_4_set);
  EXPECT_EQ(intersect_0_2_set.Min(), ir::Expr(0));
  EXPECT_EQ(intersect_0_2_set.Max(), ir::Expr(2));
  intersect_0_2_set = Intersect(interval_0_4_set, interval_0_2_set);
  EXPECT_EQ(intersect_0_2_set.Min(), ir::Expr(0));
  EXPECT_EQ(intersect_0_2_set.Max(), ir::Expr(2));
  SingleIntervalIntSet intersect_2_2_set =
      Intersect(interval_0_2_set, interval_2_6_set);
  EXPECT_EQ(intersect_2_2_set.Min(), ir::Expr(2));
  EXPECT_EQ(intersect_2_2_set.Max(), ir::Expr(2));
  intersect_2_2_set = Intersect(interval_2_6_set, interval_0_2_set);
  EXPECT_EQ(intersect_2_2_set.Min(), ir::Expr(2));
  EXPECT_EQ(intersect_2_2_set.Max(), ir::Expr(2));
  SingleIntervalIntSet intersect_empty_set =
      Intersect(interval_0_4_set, interval_8_9_set);
  EXPECT_TRUE(intersect_empty_set.IsEmpty());
  intersect_empty_set = Intersect(interval_8_9_set, interval_0_4_set);
  EXPECT_TRUE(intersect_empty_set.IsEmpty());
}

TEST(SingleIntervalIntSet, case_0) {
  ir::Var S0 = ir::Var(ir::Expr(0), ir::Expr(7), "S0");
  ir::Expr e1 = S0 * 16;
  ir::Expr e2 = S0 * 16 + 7;
  ir::Expr e3 = S0 * 16 + 15;
  SingleIntervalIntSet empty_set(e2, e1);
  SingleIntervalIntSet single_point(e3, e3);
  SingleIntervalIntSet set_0(e1, e2);
  SingleIntervalIntSet set_1(e1, e3);

  EXPECT_TRUE(empty_set.IsEmpty());
  EXPECT_FALSE(empty_set.IsAll());
  EXPECT_TRUE(single_point.IsPoint());
  EXPECT_FALSE(set_0.IsPoint());
  EXPECT_TRUE(set_0 == set_0);
  EXPECT_FALSE(set_0 == set_1);

  EXPECT_TRUE(set_0.IsSubSet(set_1));
  EXPECT_FALSE(set_1.IsSubSet(set_0));
  EXPECT_FALSE(set_0.IsSuperSet(set_1));
  EXPECT_TRUE(set_1.IsSuperSet(set_0));

  EXPECT_EQ(Union(set_0, set_1), set_1);
  EXPECT_EQ(Intersect(set_0, set_1), set_0);
  EXPECT_EQ(Union(set_1, single_point), set_1);
  EXPECT_EQ(Intersect(set_1, single_point), single_point);
  EXPECT_EQ(Union(set_0, empty_set), set_0);
  EXPECT_EQ(Intersect(set_0, empty_set), empty_set);
  EXPECT_EQ(Union(set_0, single_point), set_1);
  EXPECT_TRUE(Intersect(set_0, single_point).IsEmpty());
}

TEST(SingleIntervalIntSet, case_1) {
  ir::Var S0 = ir::Var(ir::Expr(0), ir::Expr(7), "S0");
  ir::Var S1 = ir::Var(ir::Expr(0), ir::Expr(15), "S1");
  ir::Expr e1 = S0 * 16;
  ir::Expr e2 = S0 * 16 + S1;
  ir::Expr e3 = S0 * 16 + S1 * 2 + 1;
  SingleIntervalIntSet empty_set(e3, e1);
  SingleIntervalIntSet single_point(e3, e3);
  SingleIntervalIntSet set_0(e1, e2);
  SingleIntervalIntSet set_1(e1, e3);

  EXPECT_TRUE(empty_set.IsEmpty());
  EXPECT_FALSE(empty_set.IsAll());
  EXPECT_TRUE(single_point.IsPoint());
  EXPECT_FALSE(set_0.IsPoint());
  EXPECT_TRUE(set_0 == set_0);
  EXPECT_FALSE(set_0 == set_1);

  EXPECT_TRUE(set_0.IsSubSet(set_1));
  EXPECT_FALSE(set_1.IsSubSet(set_0));
  EXPECT_FALSE(set_0.IsSuperSet(set_1));
  EXPECT_TRUE(set_1.IsSuperSet(set_0));

  EXPECT_EQ(Union(set_0, set_1), set_1);
  EXPECT_EQ(Intersect(set_0, set_1), set_0);
  EXPECT_EQ(Union(set_1, single_point), set_1);
  EXPECT_EQ(Intersect(set_1, single_point), single_point);
  EXPECT_EQ(Union(set_0, empty_set), set_0);
  EXPECT_EQ(Intersect(set_0, empty_set), empty_set);
  EXPECT_EQ(Union(set_0, single_point), set_1);
  EXPECT_TRUE(Intersect(set_0, single_point).IsEmpty());
}

}  // namespace common
}  // namespace cinn
