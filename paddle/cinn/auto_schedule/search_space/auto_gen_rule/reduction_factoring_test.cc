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
#include "paddle/cinn/ir/ir_printer.h"
#include "test/cpp/cinn/concrete_program_builder.h"

PD_DECLARE_bool(cinn_new_group_scheduler);

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
    Initialize(cinn::common::DefaultNVGPUTarget());
    // In order to forcibly use the most basic Compute of reduction
    FLAGS_cinn_new_group_scheduler = 1;
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
  Context::Global().ResetNameId();
  Initialize(cinn::common::DefaultNVGPUTarget());
  auto test_program =
      tests::OpBuilder("elementwise_add").Build({{"X", {4, 5}}, {"Y", {4, 5}}});
  ir::IRSchedule ir_schedule = MakeIRSchedule(test_program);
  VLOG(6) << "Original Expr:\n" << ir_schedule.GetModule().GetExprs()[0];
  SearchState state(ir_schedule, 0, {});
  ReductionFactoring reduction_factoring(target_);
  EXPECT_EQ(reduction_factoring.AnalyseApplyType(state, "var_1"),
            RuleApplyType::kCannotApply);
}

#ifdef CINN_WITH_CUDA

TEST_F(TestReductionFactoring, ApplyOnBlock1ReduceDim) {
  Context::Global().ResetNameId();
  std::string expected_ir = R"({
  ScheduleBlock(root)
  {
    {
      serial for (i, 0, 32)
      {
        ScheduleBlock(var_0__reduce_init)
        {
          i0_0 = axis.bind(i)
          var_0__reduce_init[i0_0] = 0.00000000f
        }
        thread_bind[threadIdx.x] for (reduce_k_0_0, 0, 64)
        {
          ScheduleBlock(var_0_rf__reduce_init)
          {
            vreduce_k_0_0, i0_0 = axis.bind(reduce_k_0_0, i)
            var_0_rf__reduce_init[i0_0, vreduce_k_0_0] = 0.00000000f
          }
          {
            serial for (reduce_k_0_1, 0, 1)
            {
              ScheduleBlock(var_0_rf)
              {
                vreduce_k_0_0, i0_0, vreduce_k_0_1 = axis.bind(reduce_k_0_0, i, reduce_k_0_1)
                var_0_rf[i0_0, vreduce_k_0_0] = (var_0_rf[i0_0, vreduce_k_0_0] + X[i0_0, (vreduce_k_0_0 + vreduce_k_0_1)])
              }
            }
            {
              ScheduleBlock(var_0)
              {
                vreduce_k_0_0, i0_0 = axis.bind(reduce_k_0_0, i)
                var_0[i0_0] = (var_0[i0_0] + var_0_rf[i0_0, vreduce_k_0_0])
              }
            }
          }
        }
      }
    }
  }
})";
  TestApplyOnReduce({32, 64}, {1}, "var_0", expected_ir);
}

TEST_F(TestReductionFactoring, ApplyOnBlock2ReduceDim) {
  Context::Global().ResetNameId();
  std::string expected_ir = R"({
  ScheduleBlock(root)
  {
    {
      serial for (i, 0, 32)
      {
        ScheduleBlock(var_0__reduce_init)
        {
          i0_0 = axis.bind(i)
          var_0__reduce_init[i0_0] = 0.00000000f
        }
        thread_bind[threadIdx.x] for (reduce_k_0_reduce_k_1_fused, 0, 1024)
        {
          ScheduleBlock(var_0_rf__reduce_init)
          {
            vreduce_k_0_reduce_k_1_fused, i0_0 = axis.bind(reduce_k_0_reduce_k_1_fused, i)
            var_0_rf__reduce_init[i0_0, vreduce_k_0_reduce_k_1_fused] = 0.00000000f
          }
          {
            serial for (reduce_k_0_reduce_k_1_fused_0, 0, 8)
            {
              ScheduleBlock(var_0_rf)
              {
                vreduce_k_0_reduce_k_1_fused, i0_0, vreduce_k_0_reduce_k_1_fused_0 = axis.bind(reduce_k_0_reduce_k_1_fused, i, reduce_k_0_reduce_k_1_fused_0)
                var_0_rf[i0_0, vreduce_k_0_reduce_k_1_fused] = (var_0_rf[i0_0, vreduce_k_0_reduce_k_1_fused] + X[i0_0, (((8 * vreduce_k_0_reduce_k_1_fused) + vreduce_k_0_reduce_k_1_fused_0) / 128), (((8 * vreduce_k_0_reduce_k_1_fused) + vreduce_k_0_reduce_k_1_fused_0) % 128)])
              }
            }
            {
              ScheduleBlock(var_0)
              {
                vreduce_k_0_reduce_k_1_fused, i0_0 = axis.bind(reduce_k_0_reduce_k_1_fused, i)
                var_0[i0_0] = (var_0[i0_0] + var_0_rf[i0_0, vreduce_k_0_reduce_k_1_fused])
              }
            }
          }
        }
      }
    }
  }
})";
  TestApplyOnReduce({32, 64, 128}, {1, 2}, "var_0", expected_ir);
}

TEST_F(TestReductionFactoring, ApplyOnBlock3ReduceDim) {
  Context::Global().ResetNameId();
  std::string expected_ir = R"({
  ScheduleBlock(root)
  {
    {
      serial for (i, 0, 32)
      {
        ScheduleBlock(var_0__reduce_init)
        {
          i0_0 = axis.bind(i)
          var_0__reduce_init[i0_0] = 0.00000000f
        }
        thread_bind[threadIdx.x] for (reduce_k_0_reduce_k_1_reduce_k_2_fused, 0, 1024)
        {
          ScheduleBlock(var_0_rf__reduce_init)
          {
            vreduce_k_0_reduce_k_1_reduce_k_2_fused, i0_0 = axis.bind(reduce_k_0_reduce_k_1_reduce_k_2_fused, i)
            var_0_rf__reduce_init[i0_0, vreduce_k_0_reduce_k_1_reduce_k_2_fused] = 0.00000000f
          }
          {
            serial for (reduce_k_0_reduce_k_1_reduce_k_2_fused_0, 0, 256)
            {
              ScheduleBlock(var_0_rf)
              {
                vreduce_k_0_reduce_k_1_reduce_k_2_fused, i0_0, vreduce_k_0_reduce_k_1_reduce_k_2_fused_0 = axis.bind(reduce_k_0_reduce_k_1_reduce_k_2_fused, i, reduce_k_0_reduce_k_1_reduce_k_2_fused_0)
                var_0_rf[i0_0, vreduce_k_0_reduce_k_1_reduce_k_2_fused] = (var_0_rf[i0_0, vreduce_k_0_reduce_k_1_reduce_k_2_fused] + X[i0_0, ((((256 * vreduce_k_0_reduce_k_1_reduce_k_2_fused) + vreduce_k_0_reduce_k_1_reduce_k_2_fused_0) / 64) / 64), ((((256 * vreduce_k_0_reduce_k_1_reduce_k_2_fused) + vreduce_k_0_reduce_k_1_reduce_k_2_fused_0) / 64) % 64), (((256 * vreduce_k_0_reduce_k_1_reduce_k_2_fused) + vreduce_k_0_reduce_k_1_reduce_k_2_fused_0) % 64)])
              }
            }
            {
              ScheduleBlock(var_0)
              {
                vreduce_k_0_reduce_k_1_reduce_k_2_fused, i0_0 = axis.bind(reduce_k_0_reduce_k_1_reduce_k_2_fused, i)
                var_0[i0_0] = (var_0[i0_0] + var_0_rf[i0_0, vreduce_k_0_reduce_k_1_reduce_k_2_fused])
              }
            }
          }
        }
      }
    }
  }
})";
  TestApplyOnReduce({32, 64, 64, 64}, {1, 2, 3}, "var_0", expected_ir);
}
#endif

}  // namespace auto_schedule
}  // namespace cinn
