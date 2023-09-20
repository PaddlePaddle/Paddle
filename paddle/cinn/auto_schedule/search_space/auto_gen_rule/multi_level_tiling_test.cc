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

#include "paddle/cinn/auto_schedule/search_space/auto_gen_rule/multi_level_tiling.h"

#include <glog/logging.h>
#include <gtest/gtest.h>

#include <cstdlib>
#include <iostream>
#include <vector>

#include "paddle/cinn/auto_schedule/search_space/auto_gen_rule/auto_gen_rule.h"
#include "paddle/cinn/auto_schedule/search_space/auto_gen_rule/test_helper.h"
#include "paddle/cinn/cinn.h"
#include "paddle/cinn/frontend/syntax.h"
#include "paddle/cinn/ir/ir.h"
#include "paddle/cinn/ir/ir_base.h"
#include "paddle/cinn/ir/schedule/ir_schedule.h"
#include "paddle/cinn/ir/tensor.h"
#include "paddle/cinn/ir/utils/ir_printer.h"
#include "paddle/cinn/lang/compute.h"
#include "paddle/cinn/lang/lower.h"
#include "paddle/cinn/poly/stage.h"
#include "paddle/cinn/utils/string.h"
#include "test/cpp/cinn/program_builder.h"

namespace cinn {
namespace auto_schedule {

TEST(MultiLevelTile, SampleSplitTwo) {
  srand(0);
  Context::Global().ResetNameId();
#ifdef CINN_WITH_CUDA
  Target target = common::DefaultNVGPUTarget();
#else
  Target target = common::DefaultHostTarget();
#endif

  MultiLevelTiling multi_level_tiling(
      target, MultiLevelTiling::kConfigs.at(target.arch));

  for (int i = 0; i < 100; ++i) {
    size_t number_to_split =
        rand() % 65535 + 2;  // NOLINT, random number in [2, 2^16]
    std::vector<size_t> split =
        multi_level_tiling.SampleSplitTwo<size_t>(number_to_split);
    EXPECT_EQ(split.size(), 2UL);
    EXPECT_EQ(split[0] * split[1], number_to_split);
  }
}

TEST(MultiLevelTile, SampleTileSplit) {
  srand(0);
  Context::Global().ResetNameId();
#ifdef CINN_WITH_CUDA
  Target target = common::DefaultNVGPUTarget();
#else
  Target target = common::DefaultHostTarget();
#endif

  MultiLevelTiling multi_level_tiling(
      target, MultiLevelTiling::kConfigs.at(target.arch));

  for (int i = 0; i < 100; ++i) {
    int number_to_split =
        rand() % 65535 + 2;           // NOLINT, random number in [2, 2^16]
    int split_size = rand() % 5 + 1;  // NOLINT, random in [1, 5]
    std::vector<int> split =
        multi_level_tiling.SampleTileSplit<int>(number_to_split, split_size);
    EXPECT_EQ(split.size(), static_cast<size_t>(split_size));
    int product = 1;
    for (int num : split) {
      product *= num;
    }
    EXPECT_EQ(product, number_to_split);
  }
}

TEST(MultiLevelTile, SimpleLoops) {
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
  std::vector<ir::LoweredFunc> funcs =
      lang::LowerVec("TestMultiLevelTile_SimpleLoops",
                     stages,
                     {C},
                     {},
                     {},
                     nullptr,
                     target,
                     true);

  ir::Expr ast_expr = funcs[0]->body;
  VLOG(6) << "Expr before MultiLevelTiling: ";
  VLOG(6) << ast_expr;

  MultiLevelTiling multi_level_tiling(
      target, MultiLevelTiling::kConfigs.at(target.arch));
  ir::IRSchedule ir_schedule(ir::ModuleExpr({ast_expr}));
  SearchState state(ir_schedule, 0, {});
  EXPECT_EQ(multi_level_tiling.Init(&ir_schedule),
            RuleApplyType::kApplyAndPruneOtherRules);
  EXPECT_EQ(multi_level_tiling.NumberApplicable(), 1);
  multi_level_tiling.ApplyRandomly();

  // ApplyOnBlock
  EXPECT_EQ(multi_level_tiling.AnalyseApplyType(state, "C"),
            RuleApplyType::kApplyAndPruneOtherRules);
  auto new_states = multi_level_tiling.ApplyOnBlock(state, "C");

  auto test_func = [](ir::IRSchedule* ir_sch) {
    std::vector<ir::Expr> exprs = ir_sch->GetModule().GetExprs();
    EXPECT_EQ(exprs.size(), 1UL);
    std::stringstream ss;
    ss << exprs[0];
    std::string expr_str = ss.str();
    VLOG(6) << expr_str;
  };

  test_func(&ir_schedule);
  test_func(&new_states[0]->ir_schedule);
}

// TODO(SunNy820828449): fix in future
/*
TEST(MulitLevelTile, MatrixMultiply) {
  srand(0);
  Context::Global().ResetNameId();
#ifdef CINN_WITH_CUDA
  Target target = common::DefaultNVGPUTarget();
#else
  Target target = common::DefaultHostTarget();
#endif

  Expr M(32);
  Expr N(32);
  Expr K(32);

  Placeholder<float> A("A", {M, K});
  Placeholder<float> B("B", {K, N});

  Var k(K.as_int32(), "reduce_axis_k");
  ir::Tensor C = Compute(
      {M, N}, [&](Var i, Var j) { return ReduceSum(A(i, k) * B(k, j), {k}); },
"C");

  poly::StageMap stages = CreateStages({C});
  std::vector<ir::LoweredFunc> funcs =
      lang::LowerVec("TestMultiLevelTile_MatrixMultiply", stages, {C}, {}, {},
nullptr, target, true);

  ir::Expr ast_expr = funcs[0]->body;
  VLOG(6) << "Expr before MultiLevelTiling: ";
  VLOG(6) << ast_expr;

  MultiLevelTiling multi_level_tiling(target,
MultiLevelTiling::kConfigs.at(target.arch)); ir::IRSchedule
ir_schedule(ir::ModuleExpr({ast_expr})); SearchState state(ir_schedule, 0, {});
  EXPECT_EQ(multi_level_tiling.Init(&ir_schedule),
RuleApplyType::kApplyAndPruneOtherRules);
  EXPECT_EQ(multi_level_tiling.NumberApplicable(), 1);
  multi_level_tiling.ApplyRandomly();

  // ApplyOnBlock
  EXPECT_EQ(multi_level_tiling.AnalyseApplyType(state, "C"),
RuleApplyType::kApplyAndPruneOtherRules); auto new_states =
multi_level_tiling.ApplyOnBlock(state, "C");

  auto test_func = [](ir::IRSchedule* ir_sch) {
    std::vector<ir::Expr> exprs = ir_sch->GetModule().GetExprs();
    EXPECT_EQ(exprs.size(), 1UL);
    std::stringstream ss;
    ss << exprs[0];
    std::string expr_str = ss.str();
    VLOG(6) << expr_str;
  };

  test_func(&ir_schedule);
  test_func(&new_states[0]->ir_schedule);
}
*/
class TestMultiLevelTiling : public TestAutoGenRuleBase {
 public:
  int fixed_rand_seed = 1;
  std::vector<std::string> default_input_names;
  std::vector<std::string> default_output_names;
};

TEST_F(TestMultiLevelTiling, Matmul) {
  default_input_names = {"X", "Y"};
  default_output_names = {"temp_matmul_out"};
  std::vector<int32_t> X_shape = {32, 32};
  std::vector<int32_t> Y_shape = {32, 32};
  std::vector<int32_t> out_shape = {32, 32};

  Initialize(common::DefaultNVGPUTarget());
  frontend::Program matmul_op =
      tests::OpBuilder("matmul").Build({{"X", X_shape}, {"Y", Y_shape}});
  ir::IRSchedule ir_schedule = MakeIRSchedule(matmul_op, fixed_rand_seed);
  SearchState state(ir_schedule);
  VLOG(6) << "Original state:\n" << state->DebugString();

  // Apply MultiLevelTiling
  MultiLevelTiling multi_level_tiling(
      target_, MultiLevelTiling::kConfigs.at(target_.arch));
  EXPECT_EQ(multi_level_tiling.AnalyseApplyType(state, default_output_names[0]),
            RuleApplyType::kApplyAndPruneOtherRules);
  auto new_states =
      multi_level_tiling.ApplyOnBlock(state, default_output_names[0]);
  VLOG(6) << "After MultiLevelTiling, state:\n" << new_states[0]->DebugString();
  std::string ir = GetIR(new_states[0]->ir_schedule);
  std::string expected_ir = R"ROC(Expr 0 {
{
  ScheduleBlock(root)
  {
    {
      thread_bind[blockIdx.x] for (i_j_fused, 0, 4)
      {
        thread_bind[threadIdx.x] for (i_0_j_0_fused, 0, 1)
        {
          serial for (i_1, 0, 1)
          {
            serial for (j_1, 0, 1)
            {
              serial for (i_2, 0, 1)
              {
                serial for (j_2, 0, 1)
                {
                  serial for (i_3, 0, 8)
                  {
                    serial for (j_3, 0, 32)
                    {
                      ScheduleBlock(temp_matmul_out__reduce_init)
                      {
                        i0, i1 = axis.bind(((8 * i_0_j_0_fused) + ((8 * i_1) + ((8 * i_2) + ((8 * i_j_fused) + i_3)))), ((32 * j_1) + ((32 * j_2) + j_3)))
                        {
                          temp_matmul_out__reduce_init[((8 * i_0_j_0_fused) + ((8 * i_1) + ((8 * i_2) + ((8 * i_j_fused) + i_3)))), ((32 * j_1) + ((32 * j_2) + j_3))] = 0.00000000f
                        }
                      }
                    }
                  }
                }
              }
              {
                serial for (reduce_k_0, 0, 4)
                {
                  serial for (ax0_0_ax1_0_fused, 0, 256)
                  {
                    ScheduleBlock(Y_reshape_shared_temp_buffer)
                    {
                      v0, v1 = axis.bind(((ax0_0_ax1_0_fused / 32) + (8 * reduce_k_0)), ((ax0_0_ax1_0_fused % 32) + (32 * j_1)))
                      attrs(compute_at_extra_var:ax0_0,ax1_0, cooperative_process:0)
                      {
                        Y_reshape_shared_temp_buffer[v0, v1] = Y_reshape[v0, v1]
                      }
                    }
                  }
                  serial for (ax0_ax1_fused, 0, 64)
                  {
                    ScheduleBlock(X_reshape_shared_temp_buffer)
                    {
                      v0, v1 = axis.bind(((ax0_ax1_fused / 8) + ((8 * i_0_j_0_fused) + ((8 * i_1) + (8 * i_j_fused)))), ((ax0_ax1_fused % 8) + (8 * reduce_k_0)))
                      attrs(compute_at_extra_var:ax0,ax1, cooperative_process:0)
                      {
                        X_reshape_shared_temp_buffer[v0, v1] = X_reshape[v0, v1]
                      }
                    }
                  }
                  serial for (reduce_k_1, 0, 1)
                  {
                    serial for (i_2, 0, 1)
                    {
                      serial for (j_2, 0, 1)
                      {
                        serial for (reduce_k_2, 0, 8)
                        {
                          serial for (i_3, 0, 8)
                          {
                            serial for (j_3, 0, 32)
                            {
                              ScheduleBlock(temp_matmul_out_local_temp_buffer)
                              {
                                i0_0, i1_0, i2 = axis.bind(((8 * i_0_j_0_fused) + ((8 * i_1) + ((8 * i_2) + ((8 * i_j_fused) + i_3)))), ((32 * j_1) + ((32 * j_2) + j_3)), ((8 * reduce_k_0) + ((8 * reduce_k_1) + reduce_k_2)))
                                read_buffers(_temp_matmul_out[i(undefined:undefined), j(undefined:undefined)], _X[i(undefined:undefined), reduce_k(undefined:undefined)], _Y[reduce_k(undefined:undefined), j(undefined:undefined)])
                                write_buffers(_temp_matmul_out[i(undefined:undefined), j(undefined:undefined)])
                                {
                                  temp_matmul_out_local_temp_buffer[((8 * i_0_j_0_fused) + ((8 * i_1) + ((8 * i_2) + ((8 * i_j_fused) + i_3)))), ((32 * j_1) + ((32 * j_2) + j_3))] = (temp_matmul_out_local_temp_buffer[((8 * i_0_j_0_fused) + ((8 * i_1) + ((8 * i_2) + ((8 * i_j_fused) + i_3)))), ((32 * j_1) + ((32 * j_2) + j_3))] + (X_reshape_shared_temp_buffer[((8 * i_0_j_0_fused) + ((8 * i_1) + ((8 * i_2) + ((8 * i_j_fused) + i_3)))), ((8 * reduce_k_0) + ((8 * reduce_k_1) + reduce_k_2))] * Y_reshape_shared_temp_buffer[((8 * reduce_k_0) + ((8 * reduce_k_1) + reduce_k_2)), ((32 * j_1) + ((32 * j_2) + j_3))]))
                                }
                              }
                            }
                          }
                        }
                      }
                    }
                  }
                }
                serial for (ax0_1, 0, 8)
                {
                  serial for (ax1_1, 0, 32)
                  {
                    ScheduleBlock(temp_matmul_out)
                    {
                      v0, v1 = axis.bind((((8 * i_0_j_0_fused) + ((8 * i_1) + (8 * i_j_fused))) + ax0_1), ((32 * j_1) + ax1_1))
                      attrs(reverse_compute_at_extra_var:ax0_1,ax1_1)
                      {
                        temp_matmul_out[v0, v1] = temp_matmul_out_local_temp_buffer[v0, v1]
                      }
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
  }
}
}  // end Expr 0
)ROC";
  ASSERT_EQ(ir, expected_ir);

  // build ir::Module and debug source code
  auto ir_module = BuildIRModule(new_states[0]->ir_schedule);
  auto source_code = GenSourceCode(ir_module);
  VLOG(6) << "scheduled source code:\n" << source_code;

  // execute and check precision
  CheckResult(
      GenExecutableKernel(ir_module),
      GenExecutableKernel(BuildIRModule(MakeIRSchedule(
          matmul_op, fixed_rand_seed, /* apply_manual_schedule*/ true))),
      default_input_names,
      default_output_names,
      {X_shape, Y_shape},
      {out_shape},
      target_);
}

TEST_F(TestMultiLevelTiling, ReduceSum) {
  default_input_names = {"X"};
  default_output_names = {"var_0_tmp"};
  std::vector<int32_t> X_shape = {1, 16, 32};
  std::vector<int32_t> out_shape = {1, 16, 1};
  std::vector<int32_t> reduce_dim = {2};

  Initialize(common::DefaultNVGPUTarget());
  frontend::Program reduce_sum_op =
      tests::OpBuilder("reduce_sum")
          .Build({{"X", X_shape}}, {{"dim", reduce_dim}, {"keep_dim", false}});
  ir::IRSchedule ir_schedule = MakeIRSchedule(reduce_sum_op);
  SearchState state(ir_schedule);
  VLOG(6) << "Original state:\n" << state->DebugString();

  // Apply MultiLevelTiling
  MultiLevelTiling multi_level_tiling(
      target_, MultiLevelTiling::kConfigs.at(target_.arch));
  // EXPECT_EQ(multi_level_tiling.AnalyseApplyType(state,
  // default_output_names[0]), RuleApplyType::kCannotApply);
}

TEST_F(TestMultiLevelTiling, Pool2d) {
  default_input_names = {"input"};
  default_output_names = {"var_0", "pad_temp_0"};
  std::vector<std::vector<int32_t>> input_shapes{{2, 8, 16, 16}};
  std::vector<std::vector<int32_t>> output_shapes{{2, 8, 8, 8}, {2, 8, 18, 18}};
  std::string pooling_type = "max";
  std::vector<int> ksize{3, 3};
  std::vector<int> strides{2, 2};
  std::vector<int> paddings{1, 1, 1, 1};
  bool ceil_mode = false;
  bool exclusive = true;
  bool global_pooling = false;
  std::string data_format = "NCHW";
  bool adaptive = false;
  std::string padding_algorithm = "EXPLICIT";
  frontend::Program pool2d_program = tests::OpBuilder("pool2d").Build(
      {{"input", input_shapes[0]}},
      {{"pool_type", pooling_type},
       {"kernel_size", ksize},
       {"stride_size", strides},
       {"padding_size", paddings},
       {"ceil_mode", ceil_mode},
       {"exclusive", exclusive},
       {"global_pooling", global_pooling},
       {"data_format", data_format},
       {"adaptive", adaptive},
       {"padding_algorithm", padding_algorithm}});

  Initialize(common::DefaultNVGPUTarget());
  ir::IRSchedule ir_schedule = MakeIRSchedule(pool2d_program, fixed_rand_seed);
  SearchState state(ir_schedule);
  VLOG(6) << "Original state:\n" << state->DebugString();

  // Apply MultiLevelTiling
  MultiLevelTiling::Config mlt_config = {
      /*bind_axis*/ std::vector<std::string>{"blockIdx.x", "threadIdx.x"},
      /*tile_struct*/ std::string("SSRS"),
      /*read_cache_memory_type*/ std::string("shared"),
      /*read_cache_levels*/ std::vector<int>{3},
      /*write_cache_memory_type*/ std::string("local"),
      /*write_cache_levels*/ std::vector<int>{2},
  };
  MultiLevelTiling multi_level_tiling(target_, mlt_config);
  EXPECT_EQ(multi_level_tiling.AnalyseApplyType(state, default_output_names[0]),
            RuleApplyType::kApplyAndPruneOtherRules);
  auto new_states =
      multi_level_tiling.ApplyOnBlock(state, default_output_names[0]);
  VLOG(6) << "After MultiLevelTiling, state:\n" << new_states[0]->DebugString();

  std::string ir = GetIR(new_states[0]->ir_schedule);
  std::string expected_ir = R"ROC(Expr 0 {
{
  ScheduleBlock(root)
  {
    {
      serial for (i, 0, 2)
      {
        serial for (j, 0, 8)
        {
          serial for (k, 0, 18)
          {
            serial for (a, 0, 18)
            {
              ScheduleBlock(pad_temp_0)
              {
                i0, i1, i2, i3 = axis.bind(i, j, k, a)
                {
                  pad_temp_0[i, j, k, a] = select(((a < 17) and ((a >= 1) and ((k < 17) and (k >= 1)))), input[i, j, (-1 + k), (-1 + a)], -3.40282347e+38f)
                }
              }
            }
          }
        }
      }
      {
        thread_bind[blockIdx.x] for (i_j_k_a_fused, 0, 16)
        {
          thread_bind[threadIdx.x] for (i_0_j_0_k_0_a_0_fused, 0, 4)
          {
            serial for (i_1, 0, 1)
            {
              serial for (j_1, 0, 4)
              {
                serial for (k_1, 0, 1)
                {
                  serial for (a_1, 0, 4)
                  {
                    ScheduleBlock(var_0__reduce_init)
                    {
                      i0_0, i1_0, i2_0, i3_0 = axis.bind(((((i_j_k_a_fused / 2) / 2) / 2) + ((i_0_j_0_k_0_a_0_fused / 4) + i_1)), ((4 * (((i_j_k_a_fused / 2) / 2) % 2)) + j_1), ((i_0_j_0_k_0_a_0_fused % 4) + ((4 * ((i_j_k_a_fused / 2) % 2)) + k_1)), ((4 * (i_j_k_a_fused % 2)) + a_1))
                      {
                        var_0__reduce_init[((((i_j_k_a_fused / 2) / 2) / 2) + ((i_0_j_0_k_0_a_0_fused / 4) + i_1)), ((4 * (((i_j_k_a_fused / 2) / 2) % 2)) + j_1), ((4 * ((i_j_k_a_fused / 2) % 2)) + ((i_0_j_0_k_0_a_0_fused % 4) + k_1)), ((4 * (i_j_k_a_fused % 2)) + a_1)] = -3.40282347e+38f
                      }
                    }
                  }
                }
              }
            }
            {
              serial for (kernel_idx, 0, 3)
              {
                serial for (kernel_idx_0, 0, 3)
                {
                  serial for (ax0_ax1_ax2_ax3_fused, 0, 28)
                  {
                    ScheduleBlock(pad_temp_0_shared_temp_buffer)
                    {
                      v0, v1, v2, v3 = axis.bind(((((i_j_k_a_fused / 2) / 2) / 2) + ((i_0_j_0_k_0_a_0_fused / 4) + ((ax0_ax1_ax2_ax3_fused / 7) / 4))), (((ax0_ax1_ax2_ax3_fused / 7) % 4) + (4 * (((i_j_k_a_fused / 2) / 2) % 2))), ((8 * ((i_j_k_a_fused / 2) % 2)) + ((2 * (i_0_j_0_k_0_a_0_fused % 4)) + kernel_idx)), ((ax0_ax1_ax2_ax3_fused % 7) + ((8 * (i_j_k_a_fused % 2)) + kernel_idx_0)))
                      attrs(compute_at_extra_var:ax0,ax1,ax2,ax3, cooperative_process:0)
                      {
                        pad_temp_0_shared_temp_buffer[v0, v1, v2, v3] = pad_temp_0[v0, v1, v2, v3]
                      }
                    }
                  }
                  serial for (i_1, 0, 1)
                  {
                    serial for (j_1, 0, 4)
                    {
                      serial for (k_1, 0, 1)
                      {
                        serial for (a_1, 0, 4)
                        {
                          ScheduleBlock(var_0_local_temp_buffer)
                          {
                            i0_1, i1_1, i2_1, i3_1, i4, i5 = axis.bind(((((i_j_k_a_fused / 2) / 2) / 2) + ((i_0_j_0_k_0_a_0_fused / 4) + i_1)), ((4 * (((i_j_k_a_fused / 2) / 2) % 2)) + j_1), ((i_0_j_0_k_0_a_0_fused % 4) + ((4 * ((i_j_k_a_fused / 2) % 2)) + k_1)), ((4 * (i_j_k_a_fused % 2)) + a_1), kernel_idx, kernel_idx_0)
                            read_buffers(_var_0[i(undefined:undefined), j(undefined:undefined), k(undefined:undefined), a(undefined:undefined)], _pad_temp_0[i(undefined:undefined), j(undefined:undefined)])
                            write_buffers(_var_0[i(undefined:undefined), j(undefined:undefined), k(undefined:undefined), a(undefined:undefined)])
                            {
                              var_0_local_temp_buffer[((((i_j_k_a_fused / 2) / 2) / 2) + ((i_0_j_0_k_0_a_0_fused / 4) + i_1)), ((4 * (((i_j_k_a_fused / 2) / 2) % 2)) + j_1), ((4 * ((i_j_k_a_fused / 2) % 2)) + ((i_0_j_0_k_0_a_0_fused % 4) + k_1)), ((4 * (i_j_k_a_fused % 2)) + a_1)] = cinn_max(var_0_local_temp_buffer[((((i_j_k_a_fused / 2) / 2) / 2) + ((i_0_j_0_k_0_a_0_fused / 4) + i_1)), ((4 * (((i_j_k_a_fused / 2) / 2) % 2)) + j_1), ((i_0_j_0_k_0_a_0_fused % 4) + ((4 * ((i_j_k_a_fused / 2) % 2)) + k_1)), ((4 * (i_j_k_a_fused % 2)) + a_1)], pad_temp_0_shared_temp_buffer[((((i_j_k_a_fused / 2) / 2) / 2) + ((i_0_j_0_k_0_a_0_fused / 4) + i_1)), ((4 * (((i_j_k_a_fused / 2) / 2) % 2)) + j_1), ((8 * ((i_j_k_a_fused / 2) % 2)) + ((2 * (i_0_j_0_k_0_a_0_fused % 4)) + ((2 * k_1) + kernel_idx))), ((8 * (i_j_k_a_fused % 2)) + ((2 * a_1) + kernel_idx_0))])
                            }
                          }
                        }
                      }
                    }
                  }
                }
              }
              serial for (ax0_0, 0, 1)
              {
                serial for (ax1_0, 0, 4)
                {
                  serial for (ax2_0, 0, 1)
                  {
                    serial for (ax3_0, 0, 4)
                    {
                      ScheduleBlock(var_0)
                      {
                        v0, v1, v2, v3 = axis.bind((((((i_j_k_a_fused / 2) / 2) / 2) + (i_0_j_0_k_0_a_0_fused / 4)) + ax0_0), ((4 * (((i_j_k_a_fused / 2) / 2) % 2)) + ax1_0), (((4 * ((i_j_k_a_fused / 2) % 2)) + (i_0_j_0_k_0_a_0_fused % 4)) + ax2_0), ((4 * (i_j_k_a_fused % 2)) + ax3_0))
                        attrs(reverse_compute_at_extra_var:ax0_0,ax1_0,ax2_0,ax3_0)
                        {
                          var_0[v0, v1, v2, v3] = var_0_local_temp_buffer[v0, v1, v2, v3]
                        }
                      }
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
  }
}
}  // end Expr 0
)ROC";
  ASSERT_EQ(ir, expected_ir);

  // build ir::Module and debug source code
  auto ir_module = BuildIRModule(new_states[0]->ir_schedule);
  auto source_code = GenSourceCode(ir_module);
  VLOG(6) << "scheduled source code:\n" << source_code;

  // execute and check precision
  CheckResult(
      GenExecutableKernel(ir_module),
      GenExecutableKernel(BuildIRModule(MakeIRSchedule(
          pool2d_program, fixed_rand_seed, /* apply_manual_schedule*/ true))),
      default_input_names,
      default_output_names,
      input_shapes,
      output_shapes,
      target_);
}

}  // namespace auto_schedule
}  // namespace cinn
