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

#include "paddle/cinn/auto_schedule/search_space/auto_gen_rule/skip_rule.h"

#include <glog/logging.h>
#include <gtest/gtest.h>

#include <cstdlib>
#include <iostream>
#include <vector>

#include "paddle/cinn/auto_schedule/search_space/auto_gen_rule/auto_gen_rule.h"
#include "paddle/cinn/cinn.h"
#include "paddle/cinn/ir/ir.h"
#include "paddle/cinn/ir/ir_base.h"
#include "paddle/cinn/ir/schedule/ir_schedule.h"
#include "paddle/cinn/ir/tensor.h"
#include "paddle/cinn/lang/compute.h"
#include "paddle/cinn/lang/lower.h"
#include "paddle/cinn/poly/stage.h"

namespace cinn {
namespace auto_schedule {

TEST(SkipRule, Basic) {
  srand(0);
  Context::Global().ResetNameId();
#ifdef CINN_WITH_CUDA
  Target target = common::DefaultNVGPUTarget();
#else
  Target target = common::DefaultHostTarget();
#endif

  Expr M(32);
  Expr N(128);

  Placeholder<float> A("A", {M});
  Placeholder<float> B("B", {N});

  ir::Tensor C = Compute(
      {M, N}, [&](Var i, Var j) { return A(i) + B(j); }, "C");

  poly::StageMap stages = CreateStages({C});
  std::vector<ir::LoweredFunc> funcs = lang::LowerVec(
      "TestSkipRule_Basic", stages, {C}, {}, {}, nullptr, target, true);

  ir::Expr ast_expr = funcs[0]->body;
  VLOG(6) << "Expr before SkipRule: ";
  VLOG(6) << ast_expr;

  SkipRule skip_rule(target);
  ir::IRSchedule ir_schedule(ir::ModuleExpr({ast_expr}));
  SearchState state(ir_schedule, 0, {});

  EXPECT_EQ(skip_rule.Init(&ir_schedule), RuleApplyType::kApply);
  EXPECT_EQ(skip_rule.NumberApplicable(), 1);
  skip_rule.ApplyRandomly();

  // ApplyOnBlock
  EXPECT_EQ(skip_rule.AnalyseApplyType(state, "C"), RuleApplyType::kApply);
  std::vector<cinn::auto_schedule::SearchState> states =
      skip_rule.ApplyOnBlock(state, "C");

  auto test_func = [&ast_expr](ir::IRSchedule* ir_sch) {
    std::vector<ir::Expr> exprs = ir_sch->GetModule().GetExprs();
    EXPECT_EQ(exprs.size(), 1UL);
    EXPECT_EQ(ast_expr, exprs[0]);
  };

  test_func(&ir_schedule);
  test_func(&states[0]->ir_schedule);
}

TEST(SkipRule, ApplyOnSpecificBlock) {
  srand(0);
  Context::Global().ResetNameId();
#ifdef CINN_WITH_CUDA
  Target target = common::DefaultNVGPUTarget();
#else
  Target target = common::DefaultHostTarget();
#endif

  Expr M(32);
  Expr N(128);

  Placeholder<float> A("A", {M});
  Placeholder<float> B("B", {N});

  ir::Tensor C = Compute(
      {M, N}, [&](Var i, Var j) { return A(i) + B(j); }, "C");

  poly::StageMap stages = CreateStages({C});
  std::vector<ir::LoweredFunc> funcs = lang::LowerVec(
      "TestSkipRule_Basic", stages, {C}, {}, {}, nullptr, target, true);

  ir::Expr ast_expr = funcs[0]->body;
  VLOG(6) << "Expr before SkipRule: ";
  VLOG(6) << ast_expr;

  SkipRule skip_rule(target);
  ir::IRSchedule ir_schedule(ir::ModuleExpr({ast_expr}));
  SearchState state(ir_schedule, 0, {});

  EXPECT_EQ(skip_rule.AnalyseApplyType(state, "C"), RuleApplyType::kApply);
  std::vector<cinn::auto_schedule::SearchState> states =
      skip_rule.ApplyOnBlock(state, "C");

  std::vector<ir::Expr> exprs = states[0]->ir_schedule.GetModule().GetExprs();
  EXPECT_EQ(exprs.size(), 1UL);
  EXPECT_EQ(ast_expr, exprs[0]);
}

}  // namespace auto_schedule
}  // namespace cinn
