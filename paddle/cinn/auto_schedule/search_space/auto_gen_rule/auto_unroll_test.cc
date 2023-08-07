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

#include "paddle/cinn/auto_schedule/search_space/auto_gen_rule/auto_unroll.h"

#include <glog/logging.h>
#include <gtest/gtest.h>

#include "paddle/cinn/cinn.h"
#include "paddle/cinn/lang/lower.h"

namespace cinn {
namespace auto_schedule {

TEST(AutoUnroll, Init) {
  using namespace ir;  // NOLINT

  Expr M(100);
  Expr N(4);
  Placeholder<float> A("A", {M, N});
  Placeholder<float> B("B", {M, N});
  Tensor C = Compute(
      {M, N}, [&](Var i, Var j) { return A(i, j) * B(i, j); }, "C");

#ifdef CINN_WITH_CUDA
  Target target = common::DefaultNVGPUTarget();
#else
  Target target = common::DefaultHostTarget();
#endif
  auto stages = CreateStages({C});
  auto funcs = cinn::lang::LowerVec(
      "test_init", stages, {A, B, C}, {}, {}, nullptr, target, true);

  auto ast_expr = funcs[0]->body;
  ir::IRSchedule init_schedule(ir::ModuleExpr({ast_expr}));
  AutoUnroll test_rule(target);
  // not meet specific condition
  ASSERT_EQ(test_rule.Init(&init_schedule), RuleApplyType::kCannotApply);
}

TEST(AutoUnroll, UnrollableApply) {
  using namespace ir;  // NOLINT

  Expr M(100);
  Expr N(4);
  Expr K(32);
  Placeholder<float> A("A", {M, K});
  Placeholder<float> B("B", {K, N});
  Var k(K.as_int32(), "k0");
  Tensor C = Compute(
      {M, N},
      [&](Var i, Var j) { return ReduceSum(A(i, k) * B(k, j), {k}); },
      "C");

#ifdef CINN_WITH_CUDA
  Target target = common::DefaultNVGPUTarget();
#else
  Target target = common::DefaultHostTarget();
#endif
  auto stages = CreateStages({C});
  auto funcs = cinn::lang::LowerVec(
      "test_unrollable", stages, {A, B, C}, {}, {}, nullptr, target, true);

  auto ast_expr = funcs[0]->body;
  auto* init_block_realize =
      ast_expr.As<ir::Block>()->stmts.front().As<ir::ScheduleBlockRealize>();
  auto* init_schedule_block =
      init_block_realize->schedule_block.As<ir::ScheduleBlock>();
  ASSERT_NE(init_schedule_block, nullptr);
  ASSERT_TRUE(init_schedule_block->attrs.empty());
  VLOG(6) << "Before auto-unroll:\n" << ast_expr;

  AutoUnroll test_rule(target);
  ir::IRSchedule ir_schedule(ir::ModuleExpr({ast_expr}));
  SearchState state(ir_schedule, 0, {});
  ASSERT_EQ(test_rule.Init(&ir_schedule),
            RuleApplyType::kApplyAndPruneOtherRules);
  EXPECT_EQ(test_rule.NumberApplicable(), 1);
  test_rule.ApplyRandomly();

  // ApplyOnBlock
  EXPECT_EQ(test_rule.AnalyseApplyType(state, "C"),
            RuleApplyType::kApplyAndPruneOtherRules);
  std::vector<cinn::auto_schedule::SearchState> states =
      test_rule.ApplyOnBlock(state, "C");

  auto test_func = [](IRSchedule* ir_sch) {
    Expr applied_expr = ir_sch->GetModule().GetExprs().front();
    auto* applied_block_realize = applied_expr.As<ir::Block>()
                                      ->stmts.front()
                                      .As<ir::ScheduleBlockRealize>();
    auto* applied_schedule_block =
        applied_block_realize->schedule_block.As<ir::ScheduleBlock>();
    ASSERT_FALSE(applied_schedule_block->attrs.empty());
    EXPECT_EQ(
        applied_schedule_block->attrs.count(ir::attr::auto_unroll_max_step), 1);
    const auto& attr_value =
        applied_schedule_block->attrs.at(ir::attr::auto_unroll_max_step);
    const int* max_step = absl::get_if<int>(&attr_value);
    EXPECT_NE(max_step, nullptr);
    EXPECT_LE(*max_step, 128);
    VLOG(6) << "After auto-unroll:max_step=" << *max_step << ", Ast:\n"
            << ir_sch->GetModule().GetExprs().front();
  };

  test_func(&ir_schedule);
  test_func(&states[0]->ir_schedule);
}

}  // namespace auto_schedule
}  // namespace cinn
