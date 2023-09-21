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

#include "paddle/cinn/auto_schedule/search_space/auto_gen_rule/reduction_factoring.h"

#include <glog/logging.h>
#include <gtest/gtest.h>

#include <cmath>
#include <functional>
#include <numeric>

#include "paddle/cinn/auto_schedule/search_space/auto_gen_rule/test_helper.h"
#include "paddle/cinn/ir/utils/ir_printer.h"
#include "test/cpp/cinn/concrete_program_builder.h"

namespace cinn {
namespace auto_schedule {

class TestReductionFactoring : public TestAutoGenRuleBase {
 public:
  std::vector<std::string> default_input_names = {"X"};
  std::vector<std::string> default_output_names = {"out"};

  void TestApplyOnReduce(const std::vector<int>& shape,
                         const std::vector<int>& reduce_dim,
                         const std::string& block_name,
                         const std::string& expected_ir) {
    Initialize(common::DefaultHostTarget());
    auto test_program = tests::ReduceBuilder().Build(
        {{"X", shape}}, {{"reduce_dim", reduce_dim}});
    // construct input parameter
    ir::IRSchedule ir_schedule = MakeIRSchedule(test_program);
    SearchState state(ir_schedule, 0, {});
    std::vector<ir::Expr> func_bodys = ir_schedule.GetModule().GetExprs();
    ASSERT_EQ(func_bodys.size(), 1UL);
    VLOG(6) << "Original Expr:\n" << func_bodys[0];

    // apply
    ReductionFactoring reduction_factoring(target_);
    ASSERT_EQ(reduction_factoring.AnalyseApplyType(state, block_name),
              RuleApplyType::kApply);
    auto result = reduction_factoring.ApplyOnBlock(state, block_name)[0];
    std::vector<ir::Expr> exprs = result->ir_schedule.GetModule().GetExprs();
    EXPECT_EQ(exprs.size(), 1UL);
    std::stringstream ir;
    ir << exprs[0];
    VLOG(6) << "ReductionFactoring applied Expr: " << exprs[0];

    // check
    const std::vector<ir::Expr>& blocks = ir_schedule.GetAllBlocks();
    CHECK_EQ(blocks.size(), 2UL);
    CHECK_EQ(ir.str(), expected_ir);
  }
};

TEST_F(TestReductionFactoring, AnalyseApplyType) {
  Initialize(common::DefaultHostTarget());
  auto test_program =
      tests::OpBuilder("elementwise_add").Build({{"X", {4, 5}}, {"Y", {4, 5}}});
  ir::IRSchedule ir_schedule = MakeIRSchedule(test_program);
  VLOG(6) << "Original Expr:\n" << ir_schedule.GetModule().GetExprs()[0];
  SearchState state(ir_schedule, 0, {});
  ReductionFactoring reduction_factoring(target_);
  EXPECT_EQ(reduction_factoring.AnalyseApplyType(state, "var_1"),
            RuleApplyType::kCannotApply);
}

TEST_F(TestReductionFactoring, ApplyOnBlock) {
  std::string expected_ir = R"({
  ScheduleBlock(root)
  {
    {
      serial for (i, 0, 32)
      {
        serial for (rf_reduce_k_0, 0, 64)
        {
          ScheduleBlock(rf_var_0__reduce_init)
          {
            i0, i1_0 = axis.bind(i, rf_reduce_k_0)
            rf_var_0__reduce_init[i0, i1_0] = 0.00000000f
          }
          serial for (reduce_k_1, 0, 128)
          {
            ScheduleBlock(rf_var_0)
            {
              i0_0, i1, i2 = axis.bind(i, rf_reduce_k_0, reduce_k_1)
              read_buffers(_var_0[i0_0(0:32)], _X[i0_0(0:32), i1(0:64), i2(0:128)])
              write_buffers(_var_0[i0_0(0:32)])
              rf_var_0[i0_0, i1] = (rf_var_0[i0_0, i1] + X[i0_0, i1, i2])
            }
          }
        }
      }
      serial for (i, 0, 32)
      {
        ScheduleBlock(var_0__reduce_init)
        {
          i0 = axis.bind(i)
          var_0__reduce_init[i0] = 0.00000000f
        }
        serial for (reduce_k_0, 0, 64)
        {
          ScheduleBlock(var_0)
          {
            i0_0, i1 = axis.bind(i, reduce_k_0)
            read_buffers(_var_0[i0_0(0:32)], _X[i0_0(0:32), i1(0:64), i2(0:128)])
            write_buffers(_var_0[i0_0(0:32)])
            var_0[i0_0] = (var_0[i0_0] + rf_var_0[i0_0, i1])
          }
        }
      }
    }
  }
})";
  TestApplyOnReduce({32, 64, 128}, {1, 2}, "var_0", expected_ir);
}

}  // namespace auto_schedule
}  // namespace cinn
