// Copyright (c) 2021 CINN Authors. All Rights Reserved.
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

#include "paddle/cinn/common/arithmatic.h"

#include <ginac/ginac.h>
#include <glog/logging.h>
#include <gtest/gtest.h>

#include "paddle/cinn/common/ir_util.h"
#include "paddle/cinn/ir/ir.h"
#include "paddle/cinn/ir/op/ir_operators.h"
#include "paddle/cinn/ir/utils/ir_printer.h"
#include "paddle/cinn/utils/string.h"

namespace cinn {
namespace common {
using utils::GetStreamCnt;
using utils::Join;
using utils::Trim;
using namespace ir;  // NOLINT

TEST(GiNaC, simplify) {
  using namespace GiNaC;  // NOLINT
  symbol x("x");
  symbol y("y");

  ex e = x * 0 + 1 + 2 + 3 - 100 + 30 * y - y * 21 + 0 * x;
  LOG(INFO) << "e: " << e;
}

TEST(GiNaC, diff) {
  using namespace GiNaC;  // NOLINT
  symbol x("x"), y("y");
  ex e = (x + 1);
  ex e1 = (y + 1);

  e = diff(e, x);
  e1 = diff(e1, x);
  LOG(INFO) << "e: " << eval(e);
  LOG(INFO) << "e1: " << eval(e1);
}

TEST(GiNaC, solve) {
  using namespace GiNaC;  // NOLINT
  symbol x("x"), y("y");

  lst eqns{2 * x + 3 == 19};
  lst vars{x};

  LOG(INFO) << "solve: " << lsolve(eqns, vars);
  LOG(INFO) << diff(2 * x + 3, x);
}

TEST(Solve, basic) {
  Var i("i", Int(32));
  Expr lhs = Expr(i) * 2;
  Expr rhs = Expr(2) * Expr(200);
  Expr res;
  bool is_positive;
  std::tie(res, is_positive) = Solve(lhs, rhs, i);
  LOG(INFO) << "res: " << res;
  EXPECT_TRUE(is_positive);
  EXPECT_TRUE(res == Expr(200));
}

TEST(Solve, basic1) {
  Var i("i", Int(32));
  Expr lhs = Expr(i) * 2;
  Expr rhs = Expr(2) * Expr(200) + 3 * Expr(i);

  Expr res;
  bool is_positive;
  std::tie(res, is_positive) = Solve(lhs, rhs, i);
  LOG(INFO) << "res " << res;
  EXPECT_TRUE(res == Expr(-400));
  EXPECT_FALSE(is_positive);
}

}  // namespace common
}  // namespace cinn
