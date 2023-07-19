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

#include "paddle/cinn/auto_schedule/cost_model/feature_extractor.h"

#include <gtest/gtest.h>
#include <pybind11/embed.h>

#include <cmath>
#include <unordered_set>
#include <vector>

#include "paddle/cinn/common/context.h"
#include "paddle/cinn/ir/ir.h"
#include "paddle/cinn/ir/ir_base.h"
#include "paddle/cinn/ir/schedule/ir_schedule.h"
#include "paddle/cinn/lang/builtin.h"
#include "paddle/cinn/lang/compute.h"
#include "paddle/cinn/lang/lower.h"
#include "paddle/cinn/lang/placeholder.h"
#include "paddle/cinn/poly/stage.h"

namespace cinn {
namespace auto_schedule {

TEST(FeatureExtractor, SimpleAssign) {
  Context::Global().ResetNameId();
#ifdef CINN_WITH_CUDA
  Target target = common::DefaultNVGPUTarget();
#else
  Target target = common::DefaultHostTarget();
#endif
  ir::Expr M(32);
  ir::Expr N(32);

  lang::Placeholder<float> A("A", {M, N});
  ir::Tensor B = lang::Compute(
      {M, N}, [&](Var i, Var j) { return A(i, j); }, "B");

  poly::StageMap stages = poly::CreateStages({A, B});
  std::vector<ir::LoweredFunc> funcs = lang::LowerVec(
      "SimpleAssign", stages, {A, B}, {}, {}, nullptr, target, true);
  ir::Expr ast_expr = funcs[0]->body;
  VLOG(6) << "Expr to test: " << ast_expr;

  std::vector<Expr> vec_ast{ast_expr};
  ir::ModuleExpr mod_expr(vec_ast);

  FeatureExtractor extractor;

  Feature feature = extractor.Extract(mod_expr, target);

  std::vector<float> to_check = feature.ToFixedSizeVector();

  ASSERT_EQ(to_check.size(),
            static_cast<size_t>(LoopBlockFeature::kTotalSize + 1));
  VLOG(6) << "Feature data before slog:";
  for (size_t i = 0; i < to_check.size(); ++i) {
    VLOG(6) << i << " " << (std::pow(2, to_check[i]) - 1);
    if (i != 0 && i != 17 && i != 18 && i != 29) {
      ASSERT_EQ(to_check[i], 0);
    }
  }
  // target
#ifdef CINN_WITH_CUDA
  ASSERT_EQ(to_check[0], 1);
#else
  ASSERT_EQ(to_check[0], 0);
#endif
  // mem_read
  ASSERT_EQ(to_check[17],
            slog(M.get_constant() * N.get_constant()));  // mem_read
  // mem_write
  ASSERT_EQ(to_check[18],
            slog(M.get_constant() * N.get_constant()));  // mem_write
  // non-opt loops, including root block
  ASSERT_EQ(to_check[29], slog(3));
}

TEST(FeatureExtractor, MatrixMultiply) {
  Context::Global().ResetNameId();
#ifdef CINN_WITH_CUDA
  Target target = common::DefaultNVGPUTarget();
#else
  Target target = common::DefaultHostTarget();
#endif

  ir::Expr M(2);
  ir::Expr N(2);
  ir::Expr K(4);

  lang::Placeholder<float> A("A", {M, K});
  lang::Placeholder<float> B("B", {K, N});

  ir::Var k(K.as_int32(), "reduce_axis_k");
  ir::Tensor C = lang::Compute(
      {M, N},
      [&](Var i, Var j) { return lang::ReduceSum(A(i, k) * B(k, j), {k}); },
      "C");

  poly::StageMap stages = poly::CreateStages({C});
  std::vector<ir::LoweredFunc> funcs = lang::LowerVec(
      "MatrixMultiply", stages, {C}, {}, {}, nullptr, target, true);

  std::vector<Expr> vec_ast{funcs[0]->body};
  ir::ModuleExpr mod_expr(vec_ast);
  ir::IRSchedule ir_sch(mod_expr);
  std::vector<ir::Expr> blocks = ir_sch.GetAllBlocks();
  std::vector<ir::Expr> loops = ir_sch.GetLoops(blocks[0]);
  ir_sch.Bind(loops.back(), "threadIdx.x");

  ir::Expr ast_expr = mod_expr.GetExprs()[0];
  VLOG(6) << "Expr to test: " << ast_expr;

  FeatureExtractor extractor;
  Feature feature = extractor.Extract(mod_expr, target);

  std::vector<float> to_check = feature.ToFixedSizeVector();

  ASSERT_EQ(to_check.size(),
            static_cast<size_t>(LoopBlockFeature::kTotalSize + 1));
  std::unordered_set<size_t> non_zero_indice = {0, 1, 2, 17, 18, 29, 30, 37};
  for (size_t i = 0; i < to_check.size(); ++i) {
    VLOG(6) << i << " " << (std::pow(2, to_check[i]) - 1);
    if (!non_zero_indice.count(i)) {
      ASSERT_EQ(to_check[i], 0);
    }
  }
  // target
#ifdef CINN_WITH_CUDA
  ASSERT_EQ(to_check[0], 1);
#else
  ASSERT_EQ(to_check[0], 0);
#endif
  float out_loop = M.get_constant() * N.get_constant();
  float total_loop = out_loop * K.get_constant();
  // float_mul
  ASSERT_EQ(to_check[1], slog(total_loop));
  // float_add_or_sub
  ASSERT_EQ(to_check[2], slog(total_loop));
  // mem_read
  ASSERT_EQ(to_check[17], slog(total_loop * 3));
  // mem_write
  ASSERT_EQ(to_check[18], slog(total_loop + out_loop));

  // non-opt loops, including root block
  ASSERT_EQ(to_check[29], slog(3));
  // GpuBind loop
  ASSERT_EQ(to_check[30], slog(1));
  // GpuBind loop
  ASSERT_EQ(to_check[37], slog(out_loop));
}

}  // namespace auto_schedule
}  // namespace cinn
