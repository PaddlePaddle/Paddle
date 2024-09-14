// Copyright (c) 2024 CINN Authors. All Rights Reserved.
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

#include "paddle/cinn/common/iter_simplify.h"
#include <glog/logging.h>
#include <gtest/gtest.h>
#include "paddle/cinn/common/integer_set.h"
#include "paddle/cinn/ir/op/ir_operators.h"

namespace cinn {
namespace common {

#define ITER_MARK(var) ir::IterMark::Make(Expr(var.ptr()), var->upper_bound)
#define ITER_SPLIT(mark, ...) ir::IterSplit::Make(mark, ##__VA_ARGS__)
#define ITER_SUM(...) ir::IterSum::Make({__VA_ARGS__}, Expr(0))
#define ITER_SUM_WITH_BASE(base, ...) \
  ir::IterSum::Make({__VA_ARGS__}, Expr(base))

class TestIterSimplify : public ::testing::Test {
 public:
  void SetUp() override {
    i = ir::Var(ir::Expr(0), ir::Expr(2), "i");
    j = ir::Var(ir::Expr(0), ir::Expr(4), "j");
    k = ir::Var(ir::Expr(0), ir::Expr(8), "k");
    var_intervals = {{"i", CasInterval(i->lower_bound, i->upper_bound)},
                     {"j", CasInterval(j->lower_bound, j->upper_bound)},
                     {"k", CasInterval(k->lower_bound, k->upper_bound)}};
  }

  ir::Var i;
  ir::Var j;
  ir::Var k;
  cas_intervals_t var_intervals;
  SymbolicExprAnalyzer analyzer{var_intervals};
};

TEST_F(TestIterSimplify, IterExprMake) {
  // IterMark Make func.
  auto mark_expr = ITER_MARK(i);
  auto mark_expr_ = ITER_MARK(j);
  // IterSplit Make func.
  auto split_0_expr = ITER_SPLIT(mark_expr);
  auto split_1_expr = ITER_SPLIT(mark_expr, Expr(1));
  auto split_2_expr = ITER_SPLIT(mark_expr, Expr(1), Expr(2), Expr(1));
  auto split_3_expr = ITER_SPLIT(mark_expr, Expr(2), Expr(2), Expr(1));
  auto split_4_expr = ITER_SPLIT(mark_expr_, Expr(1), Expr(2), Expr(1));
  // IterSum Make func.
  auto sum_expr = ITER_SUM(split_0_expr, split_1_expr, split_2_expr);

  auto mark = mark_expr.As<ir::IterMark>();
  auto split_0 = split_0_expr.As<ir::IterSplit>();
  auto split_1 = split_1_expr.As<ir::IterSplit>();
  auto split_2 = split_2_expr.As<ir::IterSplit>();
  auto sum = sum_expr.As<ir::IterSum>();

  EXPECT_EQ(mark->source, Expr(i.ptr()));
  EXPECT_EQ(mark->extent, Expr(2));

  EXPECT_EQ(split_0->source, mark_expr);
  EXPECT_EQ(split_0->lower_factor, Expr(1));
  EXPECT_EQ(split_0->extent, Expr(2));
  EXPECT_EQ(split_0->scale, Expr(1));

  EXPECT_EQ(split_1->source, mark_expr);
  EXPECT_EQ(split_1->lower_factor, Expr(1));
  EXPECT_EQ(split_1->extent, Expr(2));
  EXPECT_EQ(split_1->scale, Expr(1));

  EXPECT_EQ(split_2->source, mark_expr);
  EXPECT_EQ(split_2->lower_factor, Expr(1));
  EXPECT_EQ(split_2->extent, Expr(2));
  EXPECT_EQ(split_2->scale, Expr(1));

  EXPECT_EQ(sum->args.size(), 3);
  EXPECT_EQ(sum->base, Expr(0));

  EXPECT_NE(mark_expr, mark_expr_);

  EXPECT_EQ(split_0_expr, split_1_expr);
  EXPECT_EQ(split_1_expr, split_2_expr);
  EXPECT_NE(split_2_expr, split_3_expr);
}

TEST_F(TestIterSimplify, conversion) {
  IterMapRewriter rewriter{{i}};
  IterMapToExprNormalizer normalizer{analyzer};
  ir::Expr e1 = i;
  auto gt = ITER_SUM(ITER_SPLIT(ITER_MARK(i)));
  rewriter.Rewrite(&e1);
  EXPECT_EQ(e1, gt);
  normalizer.Convert(&e1);
  EXPECT_EQ(e1, 0 + i);
}

TEST_F(TestIterSimplify, add) {
  IterMapRewriter rewriter{{i, j, k}};
  IterMapToExprNormalizer normalizer{analyzer};
  auto gt1 = ITER_SUM(ITER_SPLIT(ITER_MARK(i)), ITER_SPLIT(ITER_MARK(j)));
  auto gt2 = ITER_SUM_WITH_BASE(Expr(0) + Expr(5),
                                ITER_SPLIT(ITER_MARK(i)),
                                ITER_SPLIT(ITER_MARK(j)),
                                ITER_SPLIT(ITER_MARK(k)));
  auto gt3 = ITER_SUM(ITER_SPLIT(ITER_MARK(i), Expr(1) + Expr(1)));
  auto gt4 = ITER_SUM_WITH_BASE(Expr(12));

  ir::Expr e1 = i + j;
  ir::Expr e2 = i + j + k + 5;
  ir::Expr e3 = i + i;
  ir::Expr e4 = Expr(7) + Expr(5);

  rewriter.Rewrite(&e1);
  EXPECT_EQ(e1, gt1);
  normalizer.Convert(&e1);
  EXPECT_EQ(e1, 0 + i + j);
  rewriter.Rewrite(&e2);
  EXPECT_EQ(e2, gt2);
  normalizer.Convert(&e2);
  EXPECT_EQ(e2, 0 + i + j + k + (Expr(0) + Expr(5)));
  rewriter.Rewrite(&e3);
  EXPECT_EQ(e3, gt3);
  normalizer.Convert(&e3);
  EXPECT_EQ(e3, 0 + i * (Expr(1) + Expr(1)));
  rewriter.Rewrite(&e4);
  EXPECT_EQ(e4, gt4);
  normalizer.Convert(&e4);
  EXPECT_EQ(e4, Expr(0) + Expr(12));
}

TEST_F(TestIterSimplify, sub) {
  IterMapRewriter rewriter{{i, j, k}};
  IterMapToExprNormalizer normalizer{analyzer};
  auto gt1 = ITER_SUM(ITER_SPLIT(ITER_MARK(i)),
                      ITER_SPLIT(ITER_MARK(j), Expr(0) - Expr(1)));
  auto gt2 = ITER_SUM_WITH_BASE(Expr(0) + Expr(5),
                                ITER_SPLIT(ITER_MARK(i)),
                                ITER_SPLIT(ITER_MARK(j)),
                                ITER_SPLIT(ITER_MARK(k), Expr(0) - Expr(1)));
  auto gt3 = ITER_SUM(ITER_SPLIT(ITER_MARK(i), Expr(1) - Expr(1)));
  auto gt4 = ITER_SUM_WITH_BASE(Expr(2));

  ir::Expr e1 = i - j;
  ir::Expr e2 = i + j - k + 5;
  ir::Expr e3 = i - i;
  ir::Expr e4 = Expr(7) - Expr(5);

  rewriter.Rewrite(&e1);
  EXPECT_EQ(e1, gt1);
  normalizer.Convert(&e1);
  EXPECT_EQ(e1, 0 + i + j * (Expr(0) - Expr(1)));
  rewriter.Rewrite(&e2);
  EXPECT_EQ(e2, gt2);
  normalizer.Convert(&e2);
  EXPECT_EQ(e2, 0 + i + j + (k * (Expr(0) - Expr(1))) + (Expr(0) + Expr(5)));
  rewriter.Rewrite(&e3);
  EXPECT_EQ(e3, gt3);
  normalizer.Convert(&e3);
  EXPECT_EQ(e3, 0 + i * (Expr(1) - Expr(1)));
  rewriter.Rewrite(&e4);
  EXPECT_EQ(e4, gt4);
  normalizer.Convert(&e4);
  EXPECT_EQ(e4, Expr(0) + Expr(2));
}

TEST_F(TestIterSimplify, mul) {
  IterMapRewriter rewriter{{i, j, k}};
  IterMapToExprNormalizer normalizer{analyzer};
  auto gt1 = ITER_SUM(ITER_SPLIT(ITER_MARK(i), Expr(1) * Expr(2)),
                      ITER_SPLIT(ITER_MARK(j)));
  auto gt2 = ITER_SUM(ITER_SPLIT(ITER_MARK(i), Expr(1) * Expr(2)),
                      ITER_SPLIT(ITER_MARK(j), Expr(1) * Expr(2)),
                      ITER_SPLIT(ITER_MARK(k)));

  auto gt3 = ITER_SUM_WITH_BASE((Expr(0) + Expr(5)) * Expr(2),
                                ITER_SPLIT(ITER_MARK(i), Expr(1) * Expr(2)),
                                ITER_SPLIT(ITER_MARK(j), Expr(1) * Expr(2)),
                                ITER_SPLIT(ITER_MARK(k)));
  auto gt4 = ITER_SUM_WITH_BASE(Expr(35));

  ir::Expr e1 = i * 2 + j;
  ir::Expr e2 = (i + j) * 2 + k;
  ir::Expr e3 = (i + j + 5) * 2 + k;
  ir::Expr e4 = Expr(7) * Expr(5);

  rewriter.Rewrite(&e1);
  EXPECT_EQ(e1, gt1);
  normalizer.Convert(&e1);
  EXPECT_EQ(e1, 0 + i * (Expr(1) * Expr(2)) + j);
  rewriter.Rewrite(&e2);
  EXPECT_EQ(e2, gt2);
  normalizer.Convert(&e2);
  EXPECT_EQ(e2,
            ((0 + (i * (Expr(1) * Expr(2)))) + (j * (Expr(1) * Expr(2)))) + k);
  rewriter.Rewrite(&e3);
  EXPECT_EQ(e3, gt3);
  normalizer.Convert(&e3);
  EXPECT_EQ(
      e3,
      (((0 + (i * (Expr(1) * Expr(2)))) + (j * (Expr(1) * Expr(2)))) + k) +
          ((Expr(0) + Expr(5)) * Expr(2)));
  rewriter.Rewrite(&e4);
  EXPECT_EQ(e4, gt4);
  normalizer.Convert(&e4);
  EXPECT_EQ(e4, Expr(0) + Expr(35));
}

}  // namespace common
}  // namespace cinn
