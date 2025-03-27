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

#include <glog/logging.h>
#include <gtest/gtest.h>
#include "paddle/cinn/common/integer_set.h"
#include "paddle/cinn/common/simplify_special_pattern.h"
#include "paddle/cinn/ir/ir.h"
#include "paddle/cinn/ir/ir_base.h"
#include "paddle/cinn/ir/ir_mutator.h"
#include "paddle/cinn/ir/op/ir_operators.h"
#include "paddle/cinn/optim/simplify_util.h"

namespace cinn {
namespace common {

using optim::ChangeSeqOfDivMod;
using optim::CheckPattern;
using optim::ConstructIndexExprByNodeType;
using optim::MatchPattern;
using optim::ParseExpressionFromString;

class TestIndexExpr : public ::testing::Test {
 public:
  void SetUp() override {
    S4 = ir::Var(ir::Expr(static_cast<int64_t>(1)), ir::Expr(INT32_MAX), "S4")
             .set_index(true);
    S5 = ir::Var(ir::Expr(static_cast<int64_t>(1)), ir::Expr(INT32_MAX), "S5")
             .set_index(true);
    S6 = ir::Var(ir::Expr(static_cast<int64_t>(1)), ir::Expr(INT32_MAX), "S6")
             .set_index(true);
    S7 = ir::Var(ir::Expr(static_cast<int64_t>(1)), ir::Expr(INT32_MAX), "S7")
             .set_index(true);
    S8 = ir::Var(ir::Expr(static_cast<int64_t>(1)), ir::Expr(INT32_MAX), "S8")
             .set_index(true);
    S9 = ir::Var(ir::Expr(static_cast<int64_t>(1)), ir::Expr(INT32_MAX), "S9")
             .set_index(true);

    f = ir::Var(ir::Expr(static_cast<int64_t>(1)), ir::Expr(INT32_MAX), "f");
  };

  ir::Var S4, S5, S6, S7, S8, S9, f;
};
TEST_F(TestIndexExpr, IndexExpr_0) {
  ir::IndexExpr a(14);
  ir::IndexExpr b(7);
  Expr d(6);
  ir::Expr c0 = a + b;
  ir::Expr c1 = a - b;
  ir::Expr c2 = a * b;
  ir::Expr c3 = a / b;
  ir::Expr c4 = a % b;

  ir::Expr c5 = a / d.as_index();
  ir::Expr c6 = a % d.as_index();

  EXPECT_EQ(c0, Expr(21));
  EXPECT_EQ(c1, Expr(7));
  EXPECT_EQ(c2, Expr(98));
  EXPECT_EQ(c3, Expr(2));
  EXPECT_EQ(c4, Expr(0));
  EXPECT_EQ(c5, Expr(2));
  EXPECT_EQ(c6, Expr(2));
}

TEST_F(TestIndexExpr, IndexExpr_1) {
  auto test = S6 * S7;
  ir::IndexExpr e1 = (S5 * ((S4 * (S5 * (S6 * S7))) / S5));
  ir::IndexExpr e2 = (S4 * (S5 * (S6 * S7))) / S5;
  ir::IndexExpr e3 = (S4 * S5) / S5;

  ir::IndexExpr e4 = (S4 * (S5 * (S6 * S7)) + S5) / S5;
  ir::IndexExpr e5 = (S4 * (S5 * (S6 * S7)) + 2 * S5) / S5;

  ir::IndexExpr e6 = (S4 * (S5 * (S6 * S7)) + S5 / S6) / S5;
  ir::IndexExpr e7 = (S4 * (S5 * (S6 * S7)) + 2 * S5 / S6) / S5;

  EXPECT_EQ(e1.Normalize(), ir::IndexExpr((S6 * S7) * S4 * S5));
  EXPECT_EQ(e2.Normalize(), ir::IndexExpr((S6 * S7) * S4));
  EXPECT_EQ(e3.Normalize(), ir::IndexExpr(S4));
  EXPECT_EQ(e4.Normalize(), ir::IndexExpr(((S6 * S7) * S4) + 1));
  EXPECT_EQ(e5.Normalize(), ir::IndexExpr(((S6 * S7) * S4) + 2));
  EXPECT_EQ(e6.Normalize(), ir::IndexExpr(((S6 * S7) * S4) + (1 / S6)));
  EXPECT_EQ(e7.Normalize(), ir::IndexExpr(((S6 * S7) * S4) + (2 / S6)));
}

TEST_F(TestIndexExpr, IndexExpr_2) {
  ir::Expr q1 = S4;
  ir::Expr q2 = S4;

  ir::Expr q3 = S4 + S5;
  ir::Expr q4 = S5 + S4;

  ir::Expr q5 = S4 * 2 + S5 / 4;
  ir::Expr q6 = S5 / 4 + S4 * 2;

  ir::Expr q7 = S4 + S5 + S6;
  ir::Expr q8 = S5 + (S4 + S6);

  ir::Expr q9 = S4 + (S5 + S7 / 4 + S6 * 2);
  ir::Expr q10 = S5 + (S4 + S6 * 2 + S7 / 4);

  ir::Expr q11 = (S7 + S5) + (S4 + S6);
  ir::Expr q12 = (S4 + S5) + (S6 + S7);

  ir::Expr q13 = (S4 + S5) * 3 + (S6 / 2 + S7) * 2;
  ir::Expr q14 = (S6 / 2 + S7) * 2 + (S4 + S5) * 3;

  ir::Expr q15 = (S4 + S5 * 2) * 3 + (S6 / 2 + S7) * 2;
  ir::Expr q16 = (S6 / 2 + S7) * 2 + (S4 + S5 * 2) * 3;

  ir::Expr q17 = (S4 + S5 * 2) * 3 + (S6 / 2 + S7) * 2 + S4;
  ir::Expr q18 = (S6 / 2 + S7) * 2 + (S4 + S5 * 2) * 3 + S4;

  ir::Expr q19 = (S4 + S5 * 2) * 3 + (S6 / 2 + S7) * 2 + S4;
  ir::Expr q20 = (S6 / 2 + S7) * 2 + (S4 + S5 * 2) * 3 + S5;

  EXPECT_EQ(q1.as_index().Normalize(), q2.as_index().Normalize());
  EXPECT_EQ(q3.as_index().Normalize(), q4.as_index().Normalize());
  EXPECT_EQ(q5.as_index().Normalize(), q6.as_index().Normalize());
  EXPECT_EQ(q7.as_index().Normalize(), q8.as_index().Normalize());
  EXPECT_EQ(q9.as_index().Normalize(), q10.as_index().Normalize());
  EXPECT_EQ(q11.as_index().Normalize(), q12.as_index().Normalize());
  EXPECT_EQ(q13.as_index().Normalize(), q14.as_index().Normalize());
  EXPECT_EQ(q15.as_index().Normalize(), q16.as_index().Normalize());
  EXPECT_EQ(q17.as_index().Normalize(), q18.as_index().Normalize());
  EXPECT_NE(q19.as_index().Normalize(), q20.as_index().Normalize());
}

TEST_F(TestIndexExpr, IndexExpr_3) {
  // `Add` corner cases
  ir::Expr q1 = S4 / S5 * S5 + S4 % S5;
  ir::Expr q2 = (S4 + S5) / S6 * S6 + (S4 + S5) % S6;
  ir::Expr q3 = S4 / (S5 + S6) * (S5 + S6) + S4 % (S5 + S6);
  ir::Expr q4 = (S4 + S5) / (S6 + S7) * (S6 + S7) + (S4 + S5) % (S6 + S7);
  ir::Expr q5 = (S4 + S5) / 5 * 5 + (S4 + S5) * 11 % 5;
  ir::Expr q14 = (S4 + S5) / (S6 * S7) * S6 * S7 + (S4 + S5) % (S6 * S7);
  ir::Expr q15 =
      (S4 * 256 + S5 + S6 * 1024) % 25088 / 512 * 512 + (S4 * 256 + S5) % 512;
  ir::Expr q16 =
      ((S4 * 256 + S5) / S6 / S7 * S7 + (S4 * 256 + S5) / S6 % S7) * S6 +
      (S4 * 256 + S5) % S6;
  ir::Expr q17 = S4 / (S5 * S6) * S6 + S4 % (S5 * S6) / S5;
  ir::Expr q18 = (S4 * 1024 + S5 * 256 + S6) / 2097152 * 32 +
                 (S4 * 1024 + S5 * 256 + S6) % 2097152 / 65536;

  // `Div` corner cases
  ir::Expr q6 = (S4 % S5 - S4) / S5;
  ir::Expr q7 = (S4 - S4 % S5) / S5;
  ir::Expr q8 = ((S4 + S5) % S6 - S4 - S5) / S6;
  ir::Expr q9 = (S4 + S5 - (S4 + S5) % S6) / S6;

  // `Mod` corner cases
  ir::Expr q10 = (S4 % S5 - S4) % S5;
  ir::Expr q11 = (S4 - S4 % S5) % S5;
  ir::Expr q12 = ((S4 + S5) % S6 - S4 - S5) % S6;
  ir::Expr q13 = (S4 + S5 - (S4 + S5) % S6) % S6;

  EXPECT_EQ(q1.as_index().Normalize(), ir::IndexExpr(S4));
  EXPECT_EQ(q2.as_index().Normalize(), ir::IndexExpr(S4 + S5));
  EXPECT_EQ(q3.as_index().Normalize(), ir::IndexExpr(S4));
  EXPECT_EQ(q4.as_index().Normalize(), ir::IndexExpr(S4 + S5));
  EXPECT_EQ(q5.as_index().Normalize(), ir::IndexExpr(S4 + S5));
  EXPECT_EQ(q6.as_index().Normalize(), ir::IndexExpr((S4 / S5) * (-1)));
  EXPECT_EQ(q7.as_index().Normalize(), ir::IndexExpr(S4 / S5));
  EXPECT_EQ(q8.as_index().Normalize(), ir::IndexExpr(((S4 + S5) / S6) * (-1)));
  EXPECT_EQ(q9.as_index().Normalize(), ir::IndexExpr((S4 + S5) / S6));
  EXPECT_EQ(q10.as_index().Normalize(), ir::IndexExpr(0));
  EXPECT_EQ(q11.as_index().Normalize(), ir::IndexExpr(0));
  EXPECT_EQ(q12.as_index().Normalize(), ir::IndexExpr(0));
  EXPECT_EQ(q13.as_index().Normalize(), ir::IndexExpr(0));
  EXPECT_EQ(q14.as_index().Normalize(), ir::IndexExpr(S4 + S5));
  EXPECT_EQ(q15.as_index().Normalize(),
            ir::IndexExpr((S4 * 256 + S5 + S6 * 1024)) % 25088);
  EXPECT_EQ(q16.as_index().Normalize(ir::IndexExpr::OptLevel::Level2),
            ir::IndexExpr(S4 * 256 + S5));
  EXPECT_EQ(q17.as_index().Normalize(), ir::IndexExpr(S4 / S5));
  EXPECT_EQ(q18.as_index().Normalize(),
            ir::IndexExpr((S4 * 1024 + S5 * 256 + S6) / 65536));
}

TEST_F(TestIndexExpr, Change_Seq_Of_Div_Mod) {
  ir::Expr q1 = S4 / S5;
  ir::Expr q2 = S4 % S5;
  ir::Expr q3 = S4 / S5 % S6;
  ir::Expr q4 = S4 / S5 % S6;

  EXPECT_EQ(ChangeSeqOfDivMod(q1.as_index()), q1);
  EXPECT_EQ(ChangeSeqOfDivMod(q2.as_index()), q2);
  EXPECT_EQ(ChangeSeqOfDivMod(q3.as_index()), S4 % (S5 * S6) / S5);
}

TEST_F(TestIndexExpr, Test_ConstructIndexExprByNodeType) {
  ir::Expr result_add = ConstructIndexExprByNodeType(
      ir::IrNodeTy::Add, S4.as_index(), S5.as_index(), true);
  ir::Expr result_sub = ConstructIndexExprByNodeType(
      ir::IrNodeTy::Sub, S4.as_index(), S5.as_index(), false);
  ir::Expr result_mul = ConstructIndexExprByNodeType(
      ir::IrNodeTy::Mul, S4.as_index(), S5.as_index(), true);
  ir::Expr result_div = ConstructIndexExprByNodeType(
      ir::IrNodeTy::Div, S4.as_index(), S5.as_index(), true);
  ir::Expr result_mod = ConstructIndexExprByNodeType(
      ir::IrNodeTy::Mod, S4.as_index(), S5.as_index(), true);
  ir::Expr result_min = ConstructIndexExprByNodeType(
      ir::IrNodeTy::Min, S4.as_index(), S5.as_index(), false);
  ir::Expr result_max = ConstructIndexExprByNodeType(
      ir::IrNodeTy::Max, S4.as_index(), S5.as_index(), false);

  EXPECT_EQ(result_add, S4 + S5);
  EXPECT_EQ(result_sub, S4 - S5);
  EXPECT_EQ(result_mul, S4 * S5);
  EXPECT_EQ(result_div, S4 / S5);
  EXPECT_EQ(result_mod, S4 % S5);
  EXPECT_EQ(result_min, ir::Min::Make(S4, S5));
  EXPECT_EQ(result_max, ir::Max::Make(S4, S5));
}

TEST_F(TestIndexExpr, Test_dynamic) {
  ir::Expr q =
      ((((((((((((((((S7 * 1024) + S8) + (S9 * 4096)) / S6) % S5) * S6) +
                ((((S7 * 1024) + S8) + (S9 * 4096)) % S6)) +
               (((((((((S7 * 1024) + S8) + (S9 * 4096)) / S6) / S5) % 640) %
                  S4) *
                 S6) *
                S5)) +
              (((((((((S7 * 1024) + S8) + (S9 * 4096)) / S6) / S5) / 640) *
                 S5) *
                S6) *
               S4)) /
             ((S5 * S6) * S4)) *
            S4) +
           (((((((((((S7 * 1024) + S8) + (S9 * 4096)) / S6) % S5) * S6) +
                ((((S7 * 1024) + S8) + (S9 * 4096)) % S6)) +
               (((((((((S7 * 1024) + S8) + (S9 * 4096)) / S6) / S5) % 640) %
                  S4) *
                 S6) *
                S5)) +
              (((((((((S7 * 1024) + S8) + (S9 * 4096)) / S6) / S5) / 640) *
                 S5) *
                S6) *
               S4)) /
             (S5 * S6)) %
            S4)) *
          S5) +
         (((((((((((S7 * 1024) + S8) + (S9 * 4096)) / S6) % S5) * S6) +
              ((((S7 * 1024) + S8) + (S9 * 4096)) % S6)) +
             (((((((((S7 * 1024) + S8) + (S9 * 4096)) / S6) / S5) % 640) % S4) *
               S6) *
              S5)) +
            (((((((((S7 * 1024) + S8) + (S9 * 4096)) / S6) / S5) / 640) * S5) *
              S6) *
             S4)) /
           S6) %
          S5)) *
        S6) +
       ((((((((((S7 * 1024) + S8) + (S9 * 4096)) / S6) % S5) * S6) +
           ((((S7 * 1024) + S8) + (S9 * 4096)) % S6)) +
          (((((((((S7 * 1024) + S8) + (S9 * 4096)) / S6) / S5) % 640) % S4) *
            S6) *
           S5)) +
         (((((((((S7 * 1024) + S8) + (S9 * 4096)) / S6) / S5) / 640) * S5) *
           S6) *
          S4)) %
        S6));

  ir::Expr q1 =
      ((((((f % ((S5 * S6) * 640)) % ((S5 * S6) * S4)) / (S5 * S6)) * S6) *
        S5) +
       (f % (S5 * S6)));
  ir::Expr q2 = ((f % ((S5 * S6) * 640)) % ((S5 * S6) * S4)) % (S5 * S6);
  ir::Expr q3 = (S5 * S6) * S4 / (S5 * S6);
  ir::Expr q4 = (S5 * S6) * S4 % (S5 * S6);
  ir::Expr q5 =
      (((((((((((((f % ((S5 * S6) * 640)) % ((S5 * S6) * S4)) / (S5 * S6)) +
                ((f / ((S5 * S6) * 640)) * S4)) *
               S5) *
              S6) +
             (f % (S5 * S6))) %
            ((S5 * S6) * S4)) /
           (S5 * S6)) +
          (((((((((f % ((S5 * S6) * 640)) % ((S5 * S6) * S4)) / (S5 * S6)) +
                ((f / ((S5 * S6) * 640)) * S4)) *
               S5) *
              S6) +
             (f % (S5 * S6))) /
            ((S5 * S6) * S4)) *
           S4)) *
         S5) *
        S6) +
       (f % (S5 * S6)));

  EXPECT_EQ(
      q.as_index().Normalize(ir::IndexExpr::OptLevel::Level2),
      ((((((((S7 * 1024) + S8) + (S9 * 4096)) / ((S5 * S6) * 640)) * S5) * S6) *
        S4) +
       (((((S7 * 1024) + S8) + (S9 * 4096)) % ((S5 * S6) * 640)) %
        ((S5 * S6) * S4))));
  EXPECT_EQ(q1.as_index().Normalize(ir::IndexExpr::OptLevel::Level2),
            ((f % ((S5 * S6) * 640)) % ((S5 * S6) * S4)));
  EXPECT_EQ(q2.as_index().Normalize(ir::IndexExpr::OptLevel::Level2),
            f % (S5 * S6));
  EXPECT_EQ(q3.as_index().Normalize(ir::IndexExpr::OptLevel::Level2), Expr(S4));
  EXPECT_EQ(q4.as_index().Normalize(ir::IndexExpr::OptLevel::Level2), Expr(0));
  EXPECT_EQ(q5.as_index().Normalize(ir::IndexExpr::OptLevel::Level2),
            (((((f / ((S5 * S6) * 640)) * S4) * S5) * S6) +
             ((f % ((S5 * S6) * 640)) % ((S5 * S6) * S4))));
}

TEST_F(TestIndexExpr, CommonFactor) {
  ir::Var S0 = ir::Var("S0");
  ir::Var S1 = ir::Var("S1");
  ir::Var S2 = ir::Var("S2");
  ir::Var S3 = ir::Var("S3");
  ir::Var S4 = ir::Var("S4");
  ir::Var S5 = ir::Var("S5");
  ir::Var S6 = ir::Var("S6");
  ir::Var S7 = ir::Var("S7");
  ir::Var S8 = ir::Var("S8");
  ir::Var S9 = ir::Var("S9");
  ir::Var S13 = ir::Var("S13");
  ir::Var S17 = ir::Var("S17");
  ir::Var S21 = ir::Var("S21");
  ir::Var tx = ir::Var("tx");
  ir::Var bx = ir::Var("bx");

  ir::Expr q = ((((((((S1 + S13) + S17) + S21) + S5) + S9)) * S2) * S3);
  ir::Expr q1 = (((((((S3 * S5) * S2) + ((S3 * S9) * S2)) + ((S3 * S21) * S2)) +
                   ((S2 * S3) * S17)) +
                  ((S2 * S3) * S13)) +
                 ((S2 * S3) * S1));
  ir::Expr q2 =
      (((((((((f * 1024) + tx) + (bx * 4096)) %
            ((((((((((((((((((((((((((S3 * S5) * S2) * S0) +
                                   (((S3 * S9) * S2) * S0)) +
                                  (((S3 * S21) * S2) * S0)) +
                                 (((S2 * S3) * S17) * S0)) +
                                (((S2 * S3) * S13) * S0)) +
                               (((S2 * S3) * S1) * S0)) /
                              4096) *
                             4096) +
                            ((S3 * S5) * S2)) +
                           ((S3 * S9) * S2)) +
                          ((S3 * S21) * S2)) +
                         ((S2 * S3) * S17)) +
                        ((S2 * S3) * S13)) +
                       ((S2 * S3) * S1)) +
                      4095) /
                     (((((((S3 * S5) * S2) + ((S3 * S9) * S2)) +
                         ((S3 * S21) * S2)) +
                        ((S2 * S3) * S17)) +
                       ((S2 * S3) * S13)) +
                      ((S2 * S3) * S1))) *
                    S3) *
                   S5) *
                  S2) +
                 (((((((((((((((((((((S3 * S5) * S2) * S0) +
                                   (((S3 * S9) * S2) * S0)) +
                                  (((S3 * S21) * S2) * S0)) +
                                 (((S2 * S3) * S17) * S0)) +
                                (((S2 * S3) * S13) * S0)) +
                               (((S2 * S3) * S1) * S0)) /
                              4096) *
                             4096) +
                            ((S3 * S5) * S2)) +
                           ((S3 * S9) * S2)) +
                          ((S3 * S21) * S2)) +
                         ((S2 * S3) * S17)) +
                        ((S2 * S3) * S13)) +
                       ((S2 * S3) * S1)) +
                      4095) /
                     (((((((S3 * S5) * S2) + ((S3 * S9) * S2)) +
                         ((S3 * S21) * S2)) +
                        ((S2 * S3) * S17)) +
                       ((S2 * S3) * S13)) +
                      ((S2 * S3) * S1))) *
                    S3) *
                   S9) *
                  S2)) +
                (((((((((((((((((((((S3 * S5) * S2) * S0) +
                                  (((S3 * S9) * S2) * S0)) +
                                 (((S3 * S21) * S2) * S0)) +
                                (((S2 * S3) * S17) * S0)) +
                               (((S2 * S3) * S13) * S0)) +
                              (((S2 * S3) * S1) * S0)) /
                             4096) *
                            4096) +
                           ((S3 * S5) * S2)) +
                          ((S3 * S9) * S2)) +
                         ((S3 * S21) * S2)) +
                        ((S2 * S3) * S17)) +
                       ((S2 * S3) * S13)) +
                      ((S2 * S3) * S1)) +
                     4095) /
                    (((((((S3 * S5) * S2) + ((S3 * S9) * S2)) +
                        ((S3 * S21) * S2)) +
                       ((S2 * S3) * S17)) +
                      ((S2 * S3) * S13)) +
                     ((S2 * S3) * S1))) *
                   S3) *
                  S21) *
                 S2)) +
               (((((((((((((((((((((S3 * S5) * S2) * S0) +
                                 (((S3 * S9) * S2) * S0)) +
                                (((S3 * S21) * S2) * S0)) +
                               (((S2 * S3) * S17) * S0)) +
                              (((S2 * S3) * S13) * S0)) +
                             (((S2 * S3) * S1) * S0)) /
                            4096) *
                           4096) +
                          ((S3 * S5) * S2)) +
                         ((S3 * S9) * S2)) +
                        ((S3 * S21) * S2)) +
                       ((S2 * S3) * S17)) +
                      ((S2 * S3) * S13)) +
                     ((S2 * S3) * S1)) +
                    4095) /
                   (((((((S3 * S5) * S2) + ((S3 * S9) * S2)) +
                       ((S3 * S21) * S2)) +
                      ((S2 * S3) * S17)) +
                     ((S2 * S3) * S13)) +
                    ((S2 * S3) * S1))) *
                  S2) *
                 S3) *
                S17)) +
              (((((((((((((((((((((S3 * S5) * S2) * S0) +
                                (((S3 * S9) * S2) * S0)) +
                               (((S3 * S21) * S2) * S0)) +
                              (((S2 * S3) * S17) * S0)) +
                             (((S2 * S3) * S13) * S0)) +
                            (((S2 * S3) * S1) * S0)) /
                           4096) *
                          4096) +
                         ((S3 * S5) * S2)) +
                        ((S3 * S9) * S2)) +
                       ((S3 * S21) * S2)) +
                      ((S2 * S3) * S17)) +
                     ((S2 * S3) * S13)) +
                    ((S2 * S3) * S1)) +
                   4095) /
                  (((((((S3 * S5) * S2) + ((S3 * S9) * S2)) +
                      ((S3 * S21) * S2)) +
                     ((S2 * S3) * S17)) +
                    ((S2 * S3) * S13)) +
                   ((S2 * S3) * S1))) *
                 S2) *
                S3) *
               S13)) +
             (((((((((((((((((((((S3 * S5) * S2) * S0) +
                               (((S3 * S9) * S2) * S0)) +
                              (((S3 * S21) * S2) * S0)) +
                             (((S2 * S3) * S17) * S0)) +
                            (((S2 * S3) * S13) * S0)) +
                           (((S2 * S3) * S1) * S0)) /
                          4096) *
                         4096) +
                        ((S3 * S5) * S2)) +
                       ((S3 * S9) * S2)) +
                      ((S3 * S21) * S2)) +
                     ((S2 * S3) * S17)) +
                    ((S2 * S3) * S13)) +
                   ((S2 * S3) * S1)) +
                  4095) /
                 (((((((S3 * S5) * S2) + ((S3 * S9) * S2)) +
                     ((S3 * S21) * S2)) +
                    ((S2 * S3) * S17)) +
                   ((S2 * S3) * S13)) +
                  ((S2 * S3) * S1))) *
                S2) *
               S3) *
              S1))) /
           (((((((S3 * S5) * S2) + ((S3 * S9) * S2)) + ((S3 * S21) * S2)) +
              ((S2 * S3) * S17)) +
             ((S2 * S3) * S13)) +
            ((S2 * S3) * S1))) *
          (((((S1 + S13) + S17) + S21) + S5) + S9)) *
         S2) *
        S3) +
       ((((f * 1024) + tx) + (bx * 4096)) %
        (((((((S3 * S5) * S2) + ((S3 * S9) * S2)) + ((S3 * S21) * S2)) +
           ((S2 * S3) * S17)) +
          ((S2 * S3) * S13)) +
         ((S2 * S3) * S1))));

  EXPECT_EQ(q.as_index().Normalize(ir::IndexExpr::OptLevel::Level2),
            (((((((S1 + S13) + S17) + S21) + S5) + S9) * S2) * S3));
  EXPECT_EQ(q1.as_index().Normalize(ir::IndexExpr::OptLevel::Level2),
            (((((((S5 + S9) + S21) + S17) + S13) + S1) * S2) * S3));
  EXPECT_EQ(
      q2.as_index().Normalize(ir::IndexExpr::OptLevel::Level2),
      ((((f * 1024) + tx) + (bx * 4096)) %
       ((((((((((((((((S5 + S9) + S21) + S17) + S13) + S1) * S2) * S3) * S0) /
               4096) *
              4096) +
             (((((((S5 + S9) + S21) + S17) + S13) + S1) * S2) * S3)) +
            4095) /
           (((((((S5 + S9) + S21) + S17) + S13) + S1) * S2) * S3)) *
          S3) *
         S2) *
        (((((S5 + S9) + S21) + S17) + S13) + S1))));
}

TEST_F(TestIndexExpr, TestCheckPattern) {
  ir::Var a = ir::Var("a");
  ir::Var b = ir::Var("b");
  ir::Var f = ir::Var("f");

  ir::Var S0 = ir::Var("S0");
  ir::Var S1 = ir::Var("S1");
  ir::Var S2 = ir::Var("S2");
  ir::Var S3 = ir::Var("S3");
  ir::Var S4 = ir::Var("S4");
  ir::Var S5 = ir::Var("S5");
  ir::Var S6 = ir::Var("S6");
  ir::Var S7 = ir::Var("S7");
  ir::Var S8 = ir::Var("S8");
  ir::Var S9 = ir::Var("S9");

  ir::IndexExpr pattern = f / (a * b) * b + f % (a * b) / a;
  ir::IndexExpr pattern1 = f / (a * b) * a + f % (a * b) / b;
  ir::IndexExpr e = (S0 * (S1 + S2) + S1 * S2 + S2) / (S4 * S5) * S5 +
                    (S0 * (S1 + S2) + S1 * S2 + S2) % (S4 * S5) / S4;
  ir::IndexExpr e1 = (S0 * (S1 + S2) + S1 * S2 + S2) / (S4 * S5) * S4 +
                     (S0 * (S1 + S2) + S1 * S2 + S2) % (S4 * S5) / S5;
  std::unordered_map<std::string, ir::IndexExpr> map;
  EXPECT_TRUE(CheckPattern(e, pattern, &map));
  map.clear();
  EXPECT_FALSE(CheckPattern(e, pattern1, &map));
  map.clear();
  EXPECT_FALSE(CheckPattern(e1, pattern, &map));
  map.clear();
  EXPECT_TRUE(CheckPattern(e1, pattern1, &map));
}
TEST_F(TestIndexExpr, ParseExpression) {
  ir::Var a = ir::Var("a");
  ir::Var b = ir::Var("b");
  ir::Var a1 = ir::Var("a_1");
  ir::Var b2 = ir::Var("b2");

  ir::Expr e1 = a + b;
  ir::Expr e2 = a - b;
  ir::Expr e3 = a * b;
  ir::Expr e4 = a / b;
  ir::Expr e5 = a % b;
  ir::Expr e6 = a + ir::Expr(20);
  ir::Expr e7 = a - ir::Expr(10);
  ir::Expr e8 = ir::Expr(5) * b;
  ir::Expr e9 = ir::Expr(20) / b;
  ir::Expr e10 = a % ir::Expr(3) + b;
  ir::Expr e11 = (a + b) * (a - b);
  ir::Expr e12 = (a + (b * a)) - (b / a);
  ir::Expr e13 = (a + b) * (a - b) + (a / b) - (b % a);
  ir::Expr e14 = a1 + b2;
  ir::Expr e15 = a + b;

  EXPECT_EQ(e1, ParseExpressionFromString("a + b"));
  EXPECT_EQ(e2, ParseExpressionFromString("a - b"));
  EXPECT_EQ(e3, ParseExpressionFromString("a * b"));
  EXPECT_EQ(e4, ParseExpressionFromString("a / b"));
  EXPECT_EQ(e5, ParseExpressionFromString("a % b"));
  EXPECT_EQ(e6, ParseExpressionFromString("a + 20"));
  EXPECT_EQ(e7, ParseExpressionFromString("a - 10"));
  EXPECT_EQ(e8, ParseExpressionFromString("5 * b"));
  EXPECT_EQ(e9, ParseExpressionFromString("20 / b"));
  EXPECT_EQ(e10, ParseExpressionFromString("a % 3 + b"));
  EXPECT_EQ(e11, ParseExpressionFromString("(a + b) * (a - b)"));
  EXPECT_EQ(e12, ParseExpressionFromString("(a + (b * a)) - (b / a)"));
  EXPECT_EQ(e13,
            ParseExpressionFromString("(a + b) * (a - b) + (a / b) - (b % a)"));
  EXPECT_EQ(e14, ParseExpressionFromString("a_1 + b2"));
  EXPECT_EQ(e15, ParseExpressionFromString("  a   +   b  "));
  EXPECT_ANY_THROW(ParseExpressionFromString("a + #"));
  EXPECT_ANY_THROW(ParseExpressionFromString("(a + b"));
  EXPECT_ANY_THROW(ParseExpressionFromString(""));
}
TEST_F(TestIndexExpr, MatchPattern) {
  ir::Var a = ir::Var("a");
  ir::Var b = ir::Var("b");
  ir::Var x = ir::Var("x");
  ir::Var y = ir::Var("y");

  ir::IndexExpr expr1 = a + b;
  ir::IndexExpr expr2 = a * b;
  ir::IndexExpr expr3 = a + (b * 10);
  ir::IndexExpr expr4 = (a + b) * 10;
  ir::IndexExpr expr5 = x + y;
  ir::IndexExpr expr6 = x * y;

  auto result1 = MatchPattern(expr1, "a + b", nullptr);
  EXPECT_TRUE(result1.has_value());
  EXPECT_EQ(result1->at("a"), a);
  EXPECT_EQ(result1->at("b"), b);

  auto result2 = MatchPattern(expr3, "a + (b * 10)", nullptr);
  EXPECT_TRUE(result2.has_value());
  EXPECT_EQ(result2->at("a"), a);
  EXPECT_EQ(result2->at("b"), b);

  auto result3 = MatchPattern(expr1, "a * b", nullptr);
  EXPECT_FALSE(result3.has_value());

  auto result4 = MatchPattern(expr3, "a + (b * 20)", nullptr);
  EXPECT_FALSE(result4.has_value());

  auto condition =
      [](const std::unordered_map<std::string, ir::IndexExpr> &map) {
        return map.at("a") == Expr(ir::Var("a")) &&
               map.at("b") == Expr(ir::Var("b"));
      };
  auto result5 = MatchPattern(expr1, "a + b", condition);
  EXPECT_TRUE(result5.has_value());

  auto condition2 =
      [](const std::unordered_map<std::string, ir::IndexExpr> &map) {
        return map.at("a") == ir::Var("x") && map.at("b") == ir::Var("y");
      };
  auto result6 = MatchPattern(expr1, "a + b", condition2);
  EXPECT_FALSE(result6.has_value());

  auto result7 = MatchPattern(expr4, "(a + b) * 10", nullptr);
  EXPECT_TRUE(result7.has_value());
  EXPECT_EQ(result7->at("a"), a);
  EXPECT_EQ(result7->at("b"), b);

  auto result8 = MatchPattern(expr1, "x + y", nullptr);
  EXPECT_TRUE(result8.has_value());
  EXPECT_EQ(result8->at("x"), a);
  EXPECT_EQ(result8->at("y"), b);

  auto result9 = MatchPattern(expr6, "x * y", nullptr);
  EXPECT_TRUE(result9.has_value());
  EXPECT_EQ(result9->at("x"), x);
  EXPECT_EQ(result9->at("y"), y);
}
}  // namespace common
}  // namespace cinn
