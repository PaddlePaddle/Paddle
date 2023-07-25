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

#include "paddle/cinn/auto_schedule/search_space/auto_gen_rule/auto_bind.h"

#include <glog/logging.h>
#include <gtest/gtest.h>

#include <cmath>
#include <functional>
#include <numeric>

#include "paddle/cinn/auto_schedule/search_space/auto_gen_rule/test_helper.h"
#include "paddle/cinn/ir/utils/ir_printer.h"
#include "test/cpp/cinn/program_builder.h"

namespace cinn {
namespace auto_schedule {

static constexpr uint32_t kMaxBlocks = 256;
static constexpr uint32_t kMaxThreadsPerBlock = 1024;

class TestAutoBind : public TestAutoGenRuleBase {
 public:
  std::vector<std::string> default_input_names = {"X", "Y"};
  std::vector<std::string> default_output_names = {"temp_matmul_out"};

  void TestApplyOnElementWiseAdd(const std::vector<int>& shape,
                                 const std::string& block_name) {
    Initialize(common::DefaultNVGPUTarget());
    auto test_program =
        tests::OpBuilder("elementwise_add").Build({{"X", shape}, {"Y", shape}});
    // construct input parameter
    ir::IRSchedule ir_schedule = MakeIRSchedule(test_program);
    SearchState state(ir_schedule, 0, {});
    std::vector<ir::Expr> func_bodys = ir_schedule.GetModule().GetExprs();
    ASSERT_EQ(func_bodys.size(), 1UL);
    VLOG(6) << "Original Expr:\n" << func_bodys[0];

    // apply
    AutoBind auto_bind(target_);
    ASSERT_EQ(auto_bind.AnalyseApplyType(state, block_name),
              RuleApplyType::kApplyAndPruneOtherRules);
    auto result = auto_bind.ApplyOnBlock(state, block_name)[0];
    std::vector<ir::Expr> exprs = result->ir_schedule.GetModule().GetExprs();
    EXPECT_EQ(exprs.size(), 1UL);
    VLOG(6) << "AutoBind applied Expr: " << exprs[0];

    // check bind result
    auto all_loops = result->ir_schedule.GetLoops(block_name);
    int total_num =
        std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>());
    if (total_num <= kMaxThreadsPerBlock) {
      ASSERT_EQ(all_loops.size(), 1);
      EXPECT_EQ(all_loops[0].As<ir::For>()->extent.as_int32(), total_num);
      EXPECT_TRUE(all_loops[0].As<ir::For>()->is_gpu_thread_binded());
    } else if (total_num <= kMaxBlocks * kMaxThreadsPerBlock) {
      ASSERT_EQ(all_loops.size(), 2);
      EXPECT_EQ(all_loops[0].As<ir::For>()->extent.as_int32(),
                static_cast<int32_t>(std::ceil(static_cast<double>(total_num) /
                                               kMaxThreadsPerBlock)));
      EXPECT_TRUE(all_loops[0].As<ir::For>()->is_gpu_block_binded());
      EXPECT_EQ(all_loops[1].As<ir::For>()->extent.as_int32(),
                kMaxThreadsPerBlock);
      EXPECT_TRUE(all_loops[1].As<ir::For>()->is_gpu_thread_binded());
    } else {
      ASSERT_EQ(all_loops.size(), 3);
      EXPECT_EQ(all_loops[0].As<ir::For>()->extent.as_int32(), kMaxBlocks);
      EXPECT_TRUE(all_loops[0].As<ir::For>()->is_gpu_block_binded());
      EXPECT_EQ(all_loops[1].As<ir::For>()->extent.as_int32(),
                kMaxThreadsPerBlock);
      EXPECT_TRUE(all_loops[1].As<ir::For>()->is_gpu_thread_binded());
      EXPECT_EQ(
          all_loops[2].As<ir::For>()->extent.as_int32(),
          static_cast<int32_t>(std::ceil(static_cast<double>(total_num) /
                                         (kMaxBlocks * kMaxThreadsPerBlock))));
      EXPECT_FALSE(all_loops[2].As<ir::For>()->is_binded());
    }

    // build and run
    auto ir_module = BuildIRModule(result->ir_schedule);
    auto source_code = GenSourceCode(ir_module);
    VLOG(6) << "Optimized source code:\n" << source_code;
    auto manual_ir_module = BuildIRModule(
        MakeIRSchedule(test_program, /* apply_manual_schedule*/ true));
    VLOG(6) << "Manual-schedule compiled source code:\n"
            << GenSourceCode(manual_ir_module);
    CheckResult(GenExecutableKernel(ir_module),
                GenExecutableKernel(manual_ir_module),
                default_input_names,
                {block_name},
                {shape, shape},
                {shape},
                target_);
  }
};

TEST_F(TestAutoBind, AnalyseApplyType) {
  Initialize(common::DefaultNVGPUTarget());
  ir::IRSchedule ir_schedule = MakeIRSchedule(
      tests::OpBuilder("matmul").Build({{"X", {32, 64}}, {"Y", {64, 32}}}));
  SearchState state(ir_schedule, 0, {});
  AutoBind auto_bind(target_);
  const std::string& applied_block_name = default_output_names.back();
  // outer two loops of initial Expr are spatial loops, so it can be applied
  EXPECT_EQ(auto_bind.AnalyseApplyType(state, applied_block_name),
            RuleApplyType::kApplyAndPruneOtherRules);
  state->ir_schedule.Fuse(applied_block_name, {0, 1});
  state->ir_schedule.Bind(state->ir_schedule.GetLoops(applied_block_name)[0],
                          "threadIdx.x");
  // after fuse and bind, there is no loops to be binded.
  EXPECT_EQ(auto_bind.AnalyseApplyType(state, applied_block_name),
            RuleApplyType::kCannotApply);
}

TEST_F(TestAutoBind, ApplyOnBlock) {
  TestApplyOnElementWiseAdd({64, 128}, "var_1");
  TestApplyOnElementWiseAdd({57, 133, 125}, "var_1");
}

}  // namespace auto_schedule
}  // namespace cinn
