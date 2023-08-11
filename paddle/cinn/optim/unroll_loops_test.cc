// Copyright (c) 2022 CINN Authors. All Rights Reserved.
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

#include "paddle/cinn/optim/unroll_loops.h"

#include <gtest/gtest.h>

#include <vector>

#include "paddle/cinn/cinn.h"
#include "paddle/cinn/ir/schedule/ir_schedule.h"
#include "paddle/cinn/lang/lower.h"

namespace cinn {
namespace optim {

TEST(UnrollLoops, unrolled_tag) {
  using namespace ir;  // NOLINT

  Expr M(100);
  Expr N(4);

  Placeholder<float> A("A", {M, N});
  Placeholder<float> B("B", {M, N});

  Tensor C = Compute(
      {M, N}, [&](Var i, Var j) { return A(i, j) * B(i, j); }, "C");

  auto stages = CreateStages({C});

  Target target = common::DefaultHostTarget();
  auto func = cinn::lang::LowerVec(
      "test_unrolled_tag", stages, {A, B, C}, {}, {}, nullptr, target, true);
  auto ast_expr = func[0]->body;

  ir::ModuleExpr mod_expr({ast_expr});
  ir::IRSchedule ir_sch(mod_expr);
  auto loops = ir_sch.GetLoops("C");
  ASSERT_EQ(loops.size(), 2U);

  // extent of the loop exceed the max permitted value in the unroll_loops pass,
  // which currently set 50, so the loop can not be unrolled actually
  loops[1].As<ir::For>()->extent.As<ir::IntImm>()->value = 51;
  ir_sch.Unroll(loops[1]);
  UnrollLoop(&ast_expr);
  loops = ir_sch.GetLoops("C");
  ASSERT_EQ(loops.size(), 2U);

  // unrolled correctly
  loops[1].As<ir::For>()->extent.As<ir::IntImm>()->value = 4;
  UnrollLoop(&ast_expr);
  EXPECT_EQ(ir_sch.GetLoops("C").size(), 1);
}

TEST(UnrollLoops, auto_unroll) {
  using namespace ir;  // NOLINT

  Expr M(100);
  Expr N(4);
  Expr O(5);
  Expr const_value(2.11f);

  Placeholder<float> A("A", {M, N, O});

  // B = A + 2.11
  Tensor B = Compute(
      {M, N, O},
      [&](Var i, Var j, Var k) { return A(i, j, k) + const_value; },
      "B");

  auto stages = CreateStages({B});
  Target target = common::DefaultHostTarget();
  auto func = cinn::lang::LowerVec(
      "test_auto_unroll", stages, {A, B}, {}, {}, nullptr, target, true);
  auto ast_expr = func[0]->body;
  ir::ModuleExpr mod_expr({ast_expr});
  ir::IRSchedule ir_sch(mod_expr);
  ASSERT_EQ(ir_sch.GetLoops("B").size(), 3);
  UnrollLoop(&ast_expr);
  // check after the last UnrollLoop pass it will remain unchanged
  ASSERT_EQ(ir_sch.GetLoops("B").size(), 3);

  ASSERT_TRUE(
      ast_expr.As<ir::Block>()->stmts.front().As<ir::ScheduleBlockRealize>() !=
      nullptr);
  auto* block_realize =
      ast_expr.As<ir::Block>()->stmts.front().As<ir::ScheduleBlockRealize>();
  auto* schedule_block = block_realize->schedule_block.As<ir::ScheduleBlock>();
  // set the 'auto_unroll_max_step' attribute as value 25 that is bigger than
  // the product of extent of the inner 2 loops
  schedule_block->attrs.emplace(ir::attr::auto_unroll_max_step, 25);
  UnrollLoop(&ast_expr);
  EXPECT_EQ(ir_sch.GetLoops("B").size(), 1);
}

}  // namespace optim
}  // namespace cinn
