// Copyright (c) 2024 CINN Authors. All Rights Reserved.
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

#include "paddle/cinn/optim/eliminate_common_factor_of_local_index.h"

#include <gtest/gtest.h>

#include "paddle/cinn/cinn.h"
#include "paddle/cinn/ir/ir.h"
#include "paddle/cinn/ir/ir_printer.h"
#include "paddle/cinn/ir/op/ir_operators.h"
#include "paddle/cinn/ir/schedule/ir_schedule.h"
#include "paddle/cinn/ir/utils/stmt_converter.h"
#include "paddle/cinn/utils/string.h"

namespace cinn {
namespace optim {

/*
thread_bind[threadIdx.x] for (threadIdx.x, 0, 32) {
  serial for (j_0, 0, 4) {
    Schedule (var_45) {
      var_45[(((blockIdx.x * 8ll) + threadIdx.y) / 16ll),
             ((((((blockIdx.x * 8ll) + threadIdx.y) % 16ll) * 128ll) + (j_0 *
32ll)) + threadIdx.x)] = cinn_min(cinn_max(var_18[0, ((j_0 * 32ll) /
128ll)], 9.99999975e-05f), 3.40282347e+38f)
    }
  }
}
*/
TEST(EliminateCommonFactorOfLocalIndex, Basic) {
  Context::Global().ResetNameId();

  // Create input IR
  ir::Expr loop_body = ir::Block::Make({ir::For::Make(
      ir::Var("threadIdx.x"),
      ir::Expr(0),
      ir::Expr(30),
      ir::ForType::Parallel,
      ir::DeviceAPI::GPU,
      ir::Block::Make({ir::For::Make(
          ir::Var("j_0"),
          ir::Expr(0),
          ir::Expr(4),
          ir::ForType::Serial,
          ir::DeviceAPI::None,
          ir::Block::Make({ir::ScheduleBlock::Make(
              {ir::Var("var_45")},
              {},
              "var_45",
              ir::Block::Make({ir::Store::Make(
                  "var_45",
                  ir::Call::Make(
                      ir::Float(32),
                      "cinn_min",
                      {ir::Call::Make(
                           ir::Float(32),
                           "cinn_max",
                           {ir::Load::Make(
                                ir::Float(32),
                                "var_18",
                                {ir::Expr(0),
                                 ir::Div::Make(ir::Mul::Make(ir::Var("j_0"),
                                                             ir::Expr(32)),
                                               ir::Expr(128))}),
                            ir::Expr(9.99999975e-05f)},
                           {},
                           ir::CallType::Extern),
                       ir::Expr(3.40282347e+38f)},
                      {},
                      ir::CallType::Extern),
                  {ir::Div::Make(
                       ir::Add::Make(
                           ir::Mul::Make(ir::Var("blockIdx.x"), ir::Expr(8)),
                           ir::Var("threadIdx.y")),
                       ir::Expr(16)),
                   ir::Add::Make(
                       ir::Add::Make(
                           ir::Mul::Make(
                               ir::Mod::Make(
                                   ir::Add::Make(
                                       ir::Mul::Make(ir::Var("blockIdx.x"),
                                                     ir::Expr(8)),
                                       ir::Var("threadIdx.y")),
                                   ir::Expr(16)),
                               ir::Expr(128)),
                           ir::Mul::Make(ir::Var("j_0"), ir::Expr(32))),
                       ir::Var("threadIdx.x"))})}))}))}))});

  ir::ModuleExpr mod_expr({loop_body});
  ir::IRSchedule ir_sch(mod_expr);

  VLOG(6) << "Before EliminateCommonFactorOfLocalIndex: "
          << ir_sch.GetModule().GetExprs()[0];
  EliminateCommonFactorOfLocalIndex(&ir_sch);
  VLOG(6) << "After EliminateCommonFactorOfLocalIndex: "
          << ir_sch.GetModule().GetExprs()[0];

  // Expected output verification
  std::string expected_ir = R"ROC({
  thread_bind[threadIdx.x] for (threadIdx.x, 0, 30)
  {
    serial for (j_0, 0, 4)
    {
      ScheduleBlock(var_45)
      {
        var_45[((blockIdx.x * 8 + threadIdx.y) / 16), ((((blockIdx.x * 8 + threadIdx.y) % 16) * 128 + j_0 * 32) + threadIdx.x)] = cinn_min(cinn_max(var_18[0, 0], 9.99999975e-05f), 3.40282347e+38f)
      }
    }
  }
})ROC";

  EXPECT_EQ(utils::GetStreamCnt(ir_sch.GetModule().GetExprs()[0]),
            utils::Trim(expected_ir));
}
}  // namespace optim
}  // namespace cinn
