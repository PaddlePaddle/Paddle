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

#include "paddle/cinn/auto_schedule/search_space/auto_gen_rule/auto_inline.h"

#include <glog/logging.h>
#include <gtest/gtest.h>

#include <cstdlib>
#include <iostream>
#include <vector>

#include "paddle/cinn/auto_schedule/search_space/auto_gen_rule/auto_gen_rule.h"
#include "paddle/cinn/auto_schedule/search_space/auto_gen_rule/test_helper.h"
#include "paddle/cinn/cinn.h"
#include "paddle/cinn/frontend/net_builder.h"
#include "paddle/cinn/hlir/framework/op_lowering.h"
#include "paddle/cinn/hlir/framework/pass.h"
#include "paddle/cinn/ir/function_base.h"
#include "paddle/cinn/ir/ir.h"
#include "paddle/cinn/ir/ir_base.h"
#include "paddle/cinn/ir/schedule/ir_schedule.h"
#include "paddle/cinn/ir/tensor.h"
#include "paddle/cinn/ir/utils/ir_printer.h"
#include "paddle/cinn/lang/compute.h"
#include "paddle/cinn/lang/lower.h"
#include "paddle/cinn/poly/stage.h"
#include "paddle/cinn/runtime/flags.h"
#include "paddle/cinn/utils/string.h"
#include "test/cpp/cinn/concrete_program_builder.h"

namespace cinn {
namespace auto_schedule {

using ::cinn::hlir::framework::Graph;
using ::cinn::hlir::framework::OpLowerer;

TEST(AutoInline, SingleLoopInline) {
  srand(0);
  Context::Global().ResetNameId();
  Target target = common::DefaultHostTarget();

  Expr M(32);

  Placeholder<float> A("A", {M});
  ir::Tensor B = Compute(
      {M}, [&](Var i) { return A(i) * ir::Expr(2.f); }, "B");
  ir::Tensor C = Compute(
      {M}, [&](Var i) { return B(i) + ir::Expr(1.f); }, "C");

  poly::StageMap stages = CreateStages({A, B, C});
  std::vector<ir::LoweredFunc> funcs =
      lang::LowerVec("TestAutoInline_SingleLoopInline",
                     stages,
                     {A, C},
                     {},
                     {},
                     nullptr,
                     target,
                     true);
  VLOG(6) << "Expr after lowering:";
  VLOG(6) << funcs[0]->body;

  /*
   * We have to use ComputeAt to put two Tensor loops together to create IR
   * test case for AutoInline.
   */
  ir::IRSchedule ir_sch(ir::ModuleExpr(std::vector<ir::Expr>{funcs[0]->body}));
  SearchState state(ir_sch, 0, {});
  ir::Expr block_b = ir_sch.GetBlock("B");
  std::vector<ir::Expr> loops = ir_sch.GetLoops("C");
  ir_sch.ComputeAt(block_b, loops[0]);

  ir::ModuleExpr mod_expr_before_inline = ir_sch.GetModule();
  VLOG(6) << "Expr after ComputeAt:";
  VLOG(6) << mod_expr_before_inline.GetExprs()[0];

  AutoInline auto_inline(target, {"C"});
  EXPECT_EQ(auto_inline.Init(&ir_sch), RuleApplyType::kApplyAndPruneOtherRules);
  EXPECT_EQ(auto_inline.NumberApplicable(), 1);
  auto_inline.ApplyRandomly();
  std::vector<ir::Expr> exprs = ir_sch.GetModule().GetExprs();
  EXPECT_EQ(exprs.size(), 1UL);

  // ApplyOnBlock
  EXPECT_EQ(auto_inline.AnalyseApplyType(state, "B"),
            RuleApplyType::kApplyAndPruneOtherRules);
  auto new_states = auto_inline.ApplyOnBlock(state, "B");

  auto test_func = [](ir::IRSchedule* ir_sch) {
    ir::ModuleExpr mod_expr_after_inline = ir_sch->GetModule();
    std::vector<ir::Expr> exprs = mod_expr_after_inline.GetExprs();
    EXPECT_EQ(exprs.size(), 1UL);

    std::stringstream ss;
    ss << exprs[0];

    std::string expr_str = ss.str();
    VLOG(6) << "After AutoInline:";
    VLOG(6) << expr_str;

    std::string target_str = R"ROC({
  ScheduleBlock(root)
  {
    {
      serial for (i, 0, 32)
      {
        ScheduleBlock(C)
        {
          i0 = axis.bind(i)
          read_buffers(_A[i0(0:32)])
          write_buffers(_C[i0(0:32)])
          C[i0] = ((A[i0] * 2.00000000f) + 1.00000000f)
        }
      }
    }
  }
})ROC";
    EXPECT_EQ(expr_str, target_str);
  };

  test_func(&ir_sch);
  test_func(&new_states[0]->ir_schedule);

  // Cannot inline above expr again
  EXPECT_EQ(auto_inline.Init(&ir_sch), RuleApplyType::kCannotApply);
  EXPECT_EQ(auto_inline.AnalyseApplyType(new_states[0], "C"),
            RuleApplyType::kCannotApply);
}

TEST(AutoInline, AddReluInline) {
  srand(0);
  Context::Global().ResetNameId();
  Target target = common::DefaultHostTarget();

  frontend::NetBuilder builder("test");

  auto a = builder.CreateInput(Float(32), {1, 64, 112, 112}, "A");
  auto b = builder.CreateInput(Float(32), {64}, "B");
  auto c = builder.Add(a, b, 1);
  auto d = builder.Relu(c);

  frontend::Program program = builder.Build();

  auto graph = std::make_shared<Graph>(program, target);
  hlir::framework::ApplyPass(graph.get(), "OpFusionPass");

  const auto& dtype_dict =
      graph->GetAttrs<absl::flat_hash_map<std::string, common::Type>>(
          "inferdtype");
  const auto& shape_dict = graph->GetAttrs<
      absl::flat_hash_map<std::string, hlir::framework::shape_t>>("infershape");
  auto op_lowerer =
      hlir::framework::CreateOpLowerer(dtype_dict, shape_dict, target);

  EXPECT_EQ(graph->fusion_groups.size(), 1UL);
  std::vector<ir::LoweredFunc> funcs =
      op_lowerer.Lower(graph->fusion_groups[0],
                       /*apply_op_schedule = */ false,
                       /*apply_group_schedule=*/false);

  VLOG(6) << "Expr before auto inline: " << funcs[0]->body;

  ir::ModuleExpr mod_expr_before_inline(std::vector<Expr>({funcs[0]->body}));
  ir::IRSchedule ir_sch(mod_expr_before_inline);
  SearchState state(ir_sch, 0, {});

  AutoInline auto_inline(target, {"var_2"});
  EXPECT_EQ(auto_inline.Init(&ir_sch), RuleApplyType::kApplyAndPruneOtherRules);
  EXPECT_EQ(auto_inline.NumberApplicable(), 2);

  auto_inline.Apply(1);
  ir::ModuleExpr mod_expr_after_inline = ir_sch.GetModule();
  std::vector<ir::Expr> exprs = mod_expr_after_inline.GetExprs();
  EXPECT_EQ(exprs.size(), 1UL);

  std::stringstream ss;
  ss << exprs[0];

  std::string expr_str = ss.str();
  VLOG(6) << "After AutoInline:";
  VLOG(6) << expr_str;

  // Auto Inline again
  EXPECT_EQ(auto_inline.Init(&ir_sch), RuleApplyType::kApplyAndPruneOtherRules);
  EXPECT_EQ(auto_inline.NumberApplicable(), 1);
  auto_inline.Apply(0);

  // ApplyOnBlock
  EXPECT_EQ(auto_inline.AnalyseApplyType(state, "var_1"),
            RuleApplyType::kApplyAndPruneOtherRules);
  auto new_states = auto_inline.ApplyOnBlock(state, "var_1");
  // Auto Inline again
  EXPECT_EQ(auto_inline.AnalyseApplyType(new_states[0], "var_3"),
            RuleApplyType::kApplyAndPruneOtherRules);
  new_states = auto_inline.ApplyOnBlock(new_states[0], "var_3");

  auto test_func = [](ir::IRSchedule* ir_sch) {
    ir::ModuleExpr final_mod_expr = ir_sch->GetModule();
    auto exprs = final_mod_expr.GetExprs();
    EXPECT_EQ(exprs.size(), 1UL);

    std::stringstream ss;
    ss << exprs[0];

    std::string expr_str = ss.str();
    VLOG(6) << "Final AutoInline:";
    VLOG(6) << expr_str;

    std::string target_str = R"ROC({
  ScheduleBlock(root)
  {
    {
      serial for (i, 0, 1)
      {
        serial for (j, 0, 64)
        {
          serial for (k, 0, 112)
          {
            serial for (a, 0, 112)
            {
              ScheduleBlock(var_2)
              {
                i0, i1, i2, i3 = axis.bind(0, j, k, a)
                read_buffers(_A[i0(0:1), i1(0:64), i2(0:112), i3(0:112)], _B[i1(0:64)])
                write_buffers(_var_2[i0(0:1), i1(0:64), i2(0:112), i3(0:112)])
                var_2[i0, i1, i2, i3] = cinn_max((A[i0, i1, i2, i3] + B[i1]), 0.00000000f)
              }
            }
          }
        }
      }
    }
  }
})ROC";
    EXPECT_EQ(expr_str, target_str);
  };

  test_func(&ir_sch);
  test_func(&new_states[0]->ir_schedule);

  // Cannot inline above expr again
  EXPECT_EQ(auto_inline.Init(&ir_sch), RuleApplyType::kCannotApply);
  EXPECT_EQ(auto_inline.AnalyseApplyType(new_states[0], "var_2"),
            RuleApplyType::kCannotApply);
}

#ifdef CINN_WITH_CUDA
class TestAutoInline : public TestAutoGenRuleBase {};

/* The single chain graph composed of multiple blocks can be inlined into one.
 *
 * Before AutoInline: The output of the previous block is the input of another
 * block. Loop1: x1 = Add() Loop2: x2 = Multiply(x1) Loop3: x3 = Add(x2) Loop4:
 *     x4 = Relu(x3)
 *
 * After AutoInline: All loops are inlined into a loop.
 *   Loop:
 *     Add(Multiply(Add(Relu())))
 */
TEST_F(TestAutoInline, SingleChain) {
  Target target = common::DefaultNVGPUTarget();
  Initialize(target);
  std::vector<std::string> input_names = {
      "bias", "conv_output", "bn_scale", "bn_offset"};
  std::vector<std::string> output_names = {
      "var_6", "var_5", "var_1", "var", "var_0", "var_4", "var_3"};
  std::vector<int32_t> conv_output_shape = {1, 512, 56, 56};
  int32_t channel = conv_output_shape[1];
  std::vector<tests::VariableInfo> inputs_varinfo(
      {{"conv_output", conv_output_shape},
       {"bias", {channel, 1, 1}},
       {"bn_scale", {channel, 1, 1}},
       {"bn_offset", {channel, 1, 1}}});

  // Construct the computation graph and convert it to ir::Expr
  Context::Global().ResetNameId();
  ir::IRSchedule ir_schedule =
      MakeIRSchedule(tests::BiasBnReLUBuilder().Build(inputs_varinfo));
  SearchState state(ir_schedule, 0, {});
  std::vector<ir::Expr> func_bodys = ir_schedule.GetModule().GetExprs();
  ASSERT_EQ(func_bodys.size(), 1UL);
  VLOG(6) << "Original Expr:\n" << func_bodys[0];

  // Apply AutoInline for every block that can be inline
  AutoInline auto_inline(target_, {output_names.front()});
  EXPECT_EQ(auto_inline.AnalyseApplyType(state, "var_3"),
            RuleApplyType::kApplyAndPruneOtherRules);
  auto new_states = auto_inline.ApplyOnBlock(state, "var_3");
  std::vector<std::string> inline_block_names(
      {"var_4", "var_5", "var_6", "var", "var_0", "var_1"});
  for (const auto& inline_block_name : inline_block_names) {
    new_states = auto_inline.ApplyOnBlock(new_states[0], inline_block_name);
  }
  std::vector<ir::Expr> exprs =
      new_states[0]->ir_schedule.GetModule().GetExprs();
  EXPECT_EQ(exprs.size(), 1UL);
  VLOG(6) << "Expr after AutoInline applied on block: " << exprs[0];

  // build ir::Module and debug source code
  auto build_module_auto = BuildIRModule(new_states[0]->ir_schedule);
  auto build_module_manually = BuildIRModule(MakeIRSchedule(
      tests::BiasBnReLUBuilder().Build(inputs_varinfo), -1, true));
  auto source_code_auto = GenSourceCode(build_module_auto);
  VLOG(6) << " auto-schedule source code:\n" << source_code_auto;
  auto source_code_manually = GenSourceCode(build_module_manually);
  VLOG(6) << " manually-schedule source code:\n" << source_code_manually;

  CheckResult(GenExecutableKernel(build_module_auto),
              GenExecutableKernel(build_module_manually),
              input_names,
              output_names,
              {{conv_output_shape[1], 1, 1},
               conv_output_shape,
               conv_output_shape,
               conv_output_shape},
              {conv_output_shape, {1}, {1}, {1}, {1}, {1}, {1}},
              target);
}

/* An op can be inlined into multiple consumers at the same time.
 *
 * Before AutoInline: The output of Exp is used by Add and Multiply.
 *   Loop1:
 *     x = Exp()
 *   Loop2:
 *     y = Add(x)
 *   Loop3:
 *     z = Multiply(x)
 *
 * After AutoInline: Exp is inlined into Add and Multiply.
 *   Loop:
 *     y = Add(Exp())
 *     z = Multiply(Exp())
 */
TEST_F(TestAutoInline, InlineToMultiConsumers) {
  Target target = common::DefaultNVGPUTarget();
  Initialize(target);
  std::vector<std::string> input_names = {"x"};
  std::vector<std::string> output_names = {"var_2", "var_1", "var_0"};
  std::vector<int32_t> input_shape{256, 256};
  std::vector<tests::VariableInfo> inputs_varinfo({{"x", input_shape}});

  // Construct the computation graph and convert it to ir::Expr
  Context::Global().ResetNameId();
  ir::IRSchedule ir_schedule =
      MakeIRSchedule(tests::ExpTwoConsumersOpBuilder().Build(inputs_varinfo));
  SearchState state(ir_schedule, 0, {});
  std::vector<ir::Expr> func_bodys = ir_schedule.GetModule().GetExprs();
  ASSERT_EQ(func_bodys.size(), 1UL);
  VLOG(6) << "Original Expr:\n" << func_bodys[0];

  // Apply AutoInline for every block that can be inline
  AutoInline auto_inline(target_, {output_names.front()});
  EXPECT_EQ(auto_inline.AnalyseApplyType(state, "var_0"),
            RuleApplyType::kApplyAndPruneOtherRules);
  auto new_states = auto_inline.ApplyOnBlock(state, "var_1");
  new_states = auto_inline.ApplyOnBlock(state, "var_0");
  std::vector<ir::Expr> exprs =
      new_states[0]->ir_schedule.GetModule().GetExprs();
  EXPECT_EQ(exprs.size(), 1UL);
  VLOG(6) << "Expr after AutoInline applied on block: " << exprs[0];

  // build ir::Module and debug source code
  auto build_module_auto = BuildIRModule(new_states[0]->ir_schedule);
  auto build_module_manually = BuildIRModule(MakeIRSchedule(
      tests::ExpTwoConsumersOpBuilder().Build(inputs_varinfo), -1, true));
  auto source_code_auto = GenSourceCode(build_module_auto);
  VLOG(6) << " auto-schedule source code:\n" << source_code_auto;
  auto source_code_manually = GenSourceCode(build_module_manually);
  VLOG(6) << " manually-schedule source code:\n" << source_code_manually;

  CheckResult(GenExecutableKernel(build_module_auto),
              GenExecutableKernel(build_module_manually),
              input_names,
              output_names,
              {input_shape},
              {input_shape, {1}, {1}},
              target);
}

/* Operators of type elementwise or injective can all be inlined.
 *
 * Before AutoInline: A graph of Gather, Add and Subtract
 *   Loop1:
 *     x1 = Gather()
 *   Loop2:
 *     x2 = Add(x1)
 *   Loop3:
 *     y1 = Gather()
 *   Loop4:
 *     z1 = Subtract(y1, x1)
 *
 * After AutoInline: All loops are inlined to one
 *     z1 = Subtract(Gather(), Add(Gather()))
 */
TEST_F(TestAutoInline, OnlySpatialOp) {
  Target target = common::DefaultNVGPUTarget();
  Initialize(target);
  std::vector<std::string> input_names = {"x", "y"};
  std::vector<std::string> output_names = {"var_6",
                                           "var_4",
                                           "constant_idx_last",
                                           "constant_idx_first",
                                           "var_2",
                                           "var_5"};
  std::vector<int32_t> input_shape{256, 256};
  std::vector<tests::VariableInfo> inputs_varinfo(
      {{"x", input_shape}, {"y", input_shape}});

  // Construct the computation graph and convert it to ir::Expr
  Context::Global().ResetNameId();
  ir::IRSchedule ir_schedule =
      MakeIRSchedule(tests::GatherAddSubBuilder().Build(inputs_varinfo));
  SearchState state(ir_schedule, 0, {});
  std::vector<ir::Expr> func_bodys = ir_schedule.GetModule().GetExprs();
  ASSERT_EQ(func_bodys.size(), 1UL);
  VLOG(6) << "Original Expr:\n" << func_bodys[0];

  // Apply AutoInline for every block that can be inline
  AutoInline auto_inline(target_, {output_names.front()});
  EXPECT_EQ(auto_inline.AnalyseApplyType(state, "constant_idx_first"),
            RuleApplyType::kApplyAndPruneOtherRules);
  auto new_states = auto_inline.ApplyOnBlock(state, "constant_idx_first");
  std::vector<std::string> inline_block_names(
      {"constant_idx_last", "var_2", "var_5", "var_4"});
  for (const auto& inline_block_name : inline_block_names) {
    new_states = auto_inline.ApplyOnBlock(new_states[0], inline_block_name);
  }
  std::vector<ir::Expr> exprs =
      new_states[0]->ir_schedule.GetModule().GetExprs();
  EXPECT_EQ(exprs.size(), 1UL);
  VLOG(6) << "Expr after AutoInline applied on block: " << exprs[0];

  // build ir::Module and debug source code
  auto build_module_auto = BuildIRModule(new_states[0]->ir_schedule);
  auto build_module_manually = BuildIRModule(MakeIRSchedule(
      tests::GatherAddSubBuilder().Build(inputs_varinfo), -1, true));
  auto source_code_auto = GenSourceCode(build_module_auto);
  VLOG(6) << " auto-schedule source code:\n" << source_code_auto;
  auto source_code_manually = GenSourceCode(build_module_manually);
  VLOG(6) << " manually-schedule source code:\n" << source_code_manually;

  CheckResult(GenExecutableKernel(build_module_auto),
              GenExecutableKernel(build_module_manually),
              input_names,
              output_names,
              {input_shape, input_shape},
              {input_shape, {1}, {1}, {1}, {1}, {1}},
              target);
}

/* An op that does not read data can be directly inlined.
 *
 * Before AutoInline: fill_constant op is in a separate loop.
 *   Loop1:
 *     x = fill_constant()
 *   Loop2:
 *     y = Add(x)
 *
 * After AutoInline: fill_constant op is inlined into other loop
 *   Loop:
 *     y = Add(fill_constant())
 */
TEST_F(TestAutoInline, NoReadBufferOp) {
  Target target = common::DefaultNVGPUTarget();
  Initialize(target);
  std::vector<std::string> input_names = {"x"};
  std::vector<std::string> output_names = {"var_0", "fill_constant"};
  std::vector<int32_t> input_shape{256, 256};
  std::vector<tests::VariableInfo> inputs_varinfo({{"x", input_shape}});

  // Construct the computation graph and convert it to ir::Expr
  ir::IRSchedule ir_schedule =
      MakeIRSchedule(tests::FillConstantAddBuilder().Build(inputs_varinfo));
  SearchState state(ir_schedule, 0, {});
  std::vector<ir::Expr> func_bodys = ir_schedule.GetModule().GetExprs();
  ASSERT_EQ(func_bodys.size(), 1UL);
  VLOG(6) << "Original Expr:\n" << func_bodys[0];

  // Apply AutoInline for every block that can be inline
  AutoInline auto_inline(target_, {output_names.front()});
  EXPECT_EQ(auto_inline.AnalyseApplyType(state, "fill_constant"),
            RuleApplyType::kApplyAndPruneOtherRules);
  auto new_states = auto_inline.ApplyOnBlock(state, "fill_constant");
  std::vector<ir::Expr> exprs =
      new_states[0]->ir_schedule.GetModule().GetExprs();
  EXPECT_EQ(exprs.size(), 1UL);
  VLOG(6) << "Expr after AutoInline applied on block: " << exprs[0];

  // build ir::Module and debug source code
  auto build_module_auto = BuildIRModule(new_states[0]->ir_schedule);
  auto build_module_manually = BuildIRModule(MakeIRSchedule(
      tests::FillConstantAddBuilder().Build(inputs_varinfo), -1, true));
  auto source_code_auto = GenSourceCode(build_module_auto);
  VLOG(6) << " auto-schedule source code:\n" << source_code_auto;
  auto source_code_manually = GenSourceCode(build_module_manually);
  VLOG(6) << " manually-schedule source code:\n" << source_code_manually;

  CheckResult(GenExecutableKernel(build_module_auto),
              GenExecutableKernel(build_module_manually),
              input_names,
              output_names,
              {input_shape},
              {input_shape, {1}},
              target);
}

/* An op can be inlined into multiple producers at the same time.
 */
// TEST_F(TestAutoInline, InlineToMultiProducers) {
// TODO(6clc): Complete the unit test, once ReverseComputeInline is ready.
// }
#endif
}  // namespace auto_schedule
}  // namespace cinn
