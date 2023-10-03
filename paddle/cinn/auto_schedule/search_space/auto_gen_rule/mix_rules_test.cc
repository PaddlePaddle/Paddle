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

#include <glog/logging.h>
#include <gtest/gtest.h>

#include <vector>

#include "paddle/cinn/auto_schedule/search_space/auto_gen_rule/auto_gen_rule.h"
#include "paddle/cinn/auto_schedule/search_space/auto_gen_rule/multi_level_tiling.h"
#include "paddle/cinn/auto_schedule/search_space/auto_gen_rule/test_helper.h"
#include "paddle/cinn/ir/ir_printer.h"
#include "paddle/cinn/ir/schedule/ir_schedule.h"
#include "test/cpp/cinn/program_builder.h"

namespace cinn {
namespace auto_schedule {

class TestMixRules : public TestAutoGenRuleBase {
 public:
  std::vector<std::string> default_input_names = {"X", "Y"};
  std::vector<std::string> default_output_names = {"temp_matmul_out"};
};

TEST_F(TestMixRules, 2DMatmulOnMultiTilingRelated) {
  frontend::Program matmul_op =
      tests::OpBuilder("matmul").Build({{"X", {32, 32}}, {"Y", {32, 32}}});
  Initialize(common::DefaultNVGPUTarget());
  ir::IRSchedule ir_schedule = MakeIRSchedule(matmul_op);
  std::vector<ir::Expr> func_bodys = ir_schedule.GetModule().GetExprs();
  ASSERT_EQ(func_bodys.size(), 1UL);
  VLOG(6) << "Original Expr:\n" << func_bodys[0];

  // Apply MultiLevelTiling
  MultiLevelTiling multi_level_tiling(
      target_, MultiLevelTiling::kConfigs.at(target_.arch));
  multi_level_tiling.Init(&ir_schedule);
  ASSERT_EQ(multi_level_tiling.NumberApplicable(), 1);
  multi_level_tiling.ApplyRandomly();
  VLOG(6) << "after MultiLevelTiling Expr:\n" << func_bodys[0];

  // build ir::Module and debug source code
  auto ir_module = BuildIRModule(ir_schedule);
  auto source_code = GenSourceCode(ir_module);
  VLOG(6) << "scheduled source code:\n" << source_code;
  // execute and check precision
  CheckResult(GenExecutableKernel(ir_module),
              GenExecutableKernel(BuildIRModule(
                  MakeIRSchedule(matmul_op, /* apply_manual_schedule */ true))),
              default_input_names,
              default_output_names,
              {{32, 32}, {32, 32}},
              {{32, 32}},
              target_);
}

}  // namespace auto_schedule
}  // namespace cinn
