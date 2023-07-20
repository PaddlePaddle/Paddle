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

#include "paddle/cinn/optim/if_simplify.h"

#include <gtest/gtest.h>

#include <string>

#include "paddle/cinn/ir/utils/ir_printer.h"

namespace cinn::optim {

TEST(IfSimplify, if_true) {
  Var n("n");
  auto e = ir::IfThenElse::Make(
      Expr(1) /*true*/, ir::Let::Make(n, Expr(1)), ir::Let::Make(n, Expr(2)));

  LOG(INFO) << "\n" << e;

  IfSimplify(&e);

  LOG(INFO) << e;

  ASSERT_EQ(utils::GetStreamCnt(e), "int32 n = 1");
}

TEST(IfSimplify, if_false) {
  Var n("n");
  auto e = ir::IfThenElse::Make(
      Expr(0) /*false*/, ir::Let::Make(n, Expr(1)), ir::Let::Make(n, Expr(2)));

  LOG(INFO) << "\n" << e;

  IfSimplify(&e);

  LOG(INFO) << e;

  ASSERT_EQ(utils::GetStreamCnt(e), "int32 n = 2");
}

TEST(IfSimplify, if_else_empty) {
  Var n("n");
  auto e = ir::IfThenElse::Make(Expr(0) /*false*/, ir::Let::Make(n, Expr(1)));

  LOG(INFO) << "\n" << e;

  IfSimplify(&e);

  LOG(INFO) << e;

  std::string target = utils::Trim(R"ROC(
{

}
)ROC");

  ASSERT_EQ(utils::GetStreamCnt(e), target);
}

}  // namespace cinn::optim
