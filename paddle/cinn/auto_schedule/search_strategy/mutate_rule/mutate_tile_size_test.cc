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

#include "paddle/cinn/auto_schedule/search_strategy/mutate_rule/mutate_tile_size.h"

#include <glog/logging.h>
#include <gtest/gtest.h>

#include "paddle/cinn/cinn.h"
#include "paddle/cinn/ir/schedule/ir_schedule.h"

namespace cinn {
namespace auto_schedule {

TEST(MutateTileSize, Basic) {
  srand(0);
  Context::Global().ResetNameId();
#ifdef CINN_WITH_CUDA
  Target target = common::DefaultNVGPUTarget();
#else
  Target target = common::DefaultHostTarget();
#endif

  const int kSize = 32;
  Expr M(kSize);
  Expr N(kSize);
  Expr K(kSize);

  Placeholder<float> A("A", {M, K});
  Placeholder<float> B("B", {K, N});

  Var k(K.as_int32(), "reduce_axis_k");
  ir::Tensor C = Compute(
      {M, N},
      [&](Var i, Var j) { return ReduceSum(A(i, k) * B(k, j), {k}); },
      "C");

  poly::StageMap stages = CreateStages({A, B, C});
  std::vector<ir::LoweredFunc> funcs =
      lang::LowerVec("TestMutateTileSize_Basic",
                     stages,
                     {A, B, C},
                     {},
                     {},
                     nullptr,
                     target,
                     true);

  ir::Expr ast_expr = funcs[0]->body;
  VLOG(6) << "Original Expr: ";
  VLOG(6) << ast_expr;
  ir::ModuleExpr module_expr({ast_expr});
  // We need to fix the seed as a constant to ensure that the result can be
  // repeated.
  utils::LinearRandomEngine::StateType rand_seed = 123;
  ir::IRSchedule ir_schedule(module_expr, rand_seed);
  ir::IRSchedule new_ir_schedule(ir_schedule);

  // apply schedule
  auto loops = ir_schedule.GetLoops("C");
  auto factors = ir_schedule.SamplePerfectTile(loops[0], 2, kSize);
  auto splited = ir_schedule.Split(loops[0], factors);

  // apply mutate
  MutateTileSize mutator;
  ir::ScheduleDesc sch_desc =
      mutator.Apply(ir_schedule.GetTraceDesc(), &rand_seed);
  sch_desc.Replay(&new_ir_schedule, true);
  VLOG(6) << "Expr before mutate tile size: \n"
          << ir_schedule.GetModule().GetExprs()[0];
  VLOG(6) << "Expr after mutate tile size: \n"
          << new_ir_schedule.GetModule().GetExprs()[0];

  std::string target_new_ir = R"ROC({
  ScheduleBlock(root)
  {
    serial for (i_1, 0, 2)
    {
      serial for (i_2, 0, 16)
      {
        serial for (j, 0, 32)
        {
          ScheduleBlock(C__reduce_init)
          {
            i0, i1 = axis.bind(((16 * i_1) + i_2), j)
            C__reduce_init[i0, i1] = 0.00000000f
          }
          serial for (reduce_axis_k, 0, 32)
          {
            ScheduleBlock(C)
            {
              i0_0, i1_0, i2 = axis.bind(((16 * i_1) + i_2), j, reduce_axis_k)
              C[i0_0, i1_0] = (C[i0_0, i1_0] + (A[i0_0, i2] * B[i2, i1_0]))
            }
          }
        }
      }
    }
  }
})ROC";

  auto get_ir_str = [](const ir::IRSchedule* ir_sch) -> std::string {
    std::vector<ir::Expr> exprs = ir_sch->GetModule().GetExprs();
    EXPECT_EQ(exprs.size(), 1UL);
    std::stringstream ss;
    ss << exprs[0];
    return ss.str();
  };
  ASSERT_EQ(get_ir_str(&new_ir_schedule), target_new_ir);

  std::vector<int> last_tile_factors = {2, 16};
  for (int i = 0; i < 10; ++i) {
    sch_desc = mutator.Apply(sch_desc, &rand_seed);
    for (auto&& step : sch_desc.Steps()) {
      if (step.type == "SamplePerfectTile") {
        std::vector<int> tile_factors =
            absl::get<std::vector<int>>(step.attrs.at("decision"));
        ASSERT_EQ(tile_factors.size(), last_tile_factors.size());
        ASSERT_NE(tile_factors[0], last_tile_factors[0]);
        ASSERT_NE(tile_factors[1], last_tile_factors[1]);
        ASSERT_EQ(tile_factors[0] * tile_factors[1], kSize);
        last_tile_factors = tile_factors;
      }
    }
  }
}

}  // namespace auto_schedule
}  // namespace cinn
