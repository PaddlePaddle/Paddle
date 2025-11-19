// Copyright (c) 2025 CINN Authors. All Rights Reserved.
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

#include "paddle/cinn/optim/ir_simplify.h"

#include <gtest/gtest.h>

#include "paddle/cinn/cinn.h"
#include "paddle/cinn/ir/ir.h"
#include "paddle/cinn/ir/ir_base.h"
#include "paddle/cinn/ir/ir_printer.h"
#include "paddle/cinn/ir/op/ir_operators.h"
#include "paddle/cinn/ir/schedule/ir_schedule.h"
#include "paddle/cinn/ir/utils/ir_nodes_collector.h"
#include "paddle/cinn/ir/utils/stmt_converter.h"
#include "paddle/cinn/utils/string.h"

namespace cinn {
namespace optim {

/*
i_j_fused: [0ll, 524288ll)
j_0: [0, 128)
Before Normalize:
(j_0 % 128)
After Normalize:
j_0
*/
TEST(IRSimplifyBound, SimplifyMod) {
  Context::Global().ResetNameId();

  // Create input IR matching the specified pattern
  // Define loop variable
  ir::Var var_j_0 = ir::Var(ir::Expr(0), ir::Expr(128), "j_0");

  // Final expression
  ir::Expr expr = ir::Mod::Make(var_j_0, ir::Expr(128));

  VLOG(6) << "Before Simplify: " << expr;
  auto res = expr.as_index().ir::IndexExpr::Normalize(
      ir::IndexExpr::OptLevel::kLevel3);
  VLOG(6) << "After Simplify: " << res;

  // Expected output verification
  std::string expected_ir = R"ROC(j_0)ROC";

  EXPECT_EQ(utils::GetStreamCnt(res), utils::Trim(expected_ir));
}

/*
i_j_fused: [0ll, 524288ll)
j_0: [0, 128)
Before Normalize:
(j_0 / 128)
After Normalize:
0
*/
TEST(IRSimplifyBound, SimplifyDiv) {
  Context::Global().ResetNameId();

  // Create input IR matching the specified pattern
  // Define loop variable
  ir::Var var_j_0 = ir::Var(ir::Expr(0), ir::Expr(128), "j_0");

  // Final expression
  ir::Expr expr = ir::Div::Make(var_j_0, ir::Expr(128));

  VLOG(6) << "Before Normalize: " << expr;
  auto res = expr.as_index().ir::IndexExpr::Normalize(
      ir::IndexExpr::OptLevel::kLevel3);
  VLOG(6) << "After Normalize: " << res;

  // Expected output verification
  std::string expected_ir = R"ROC(0)ROC";

  EXPECT_EQ(utils::GetStreamCnt(res), utils::Trim(expected_ir));
}

/*
i_j_fused: [0ll, 524288ll)
j_0: [0, 128)
Before Normalize:
((((i_j_fused % 16) * 128) + j_0) / 128)
After Normalize:
(i_j_fused % 16)
*/
TEST(IRSimplifyBound, SimplifyLinearDiv) {
  Context::Global().ResetNameId();

  // Create input IR matching the specified pattern
  // Define loop variables
  ir::Var var_i_j_fused = ir::Var(ir::Expr(0), ir::Expr(524288), "i_j_fused");
  ir::Var var_j_0 = ir::Var(ir::Expr(0), ir::Expr(128), "j_0");

  // Final expression
  ir::Expr expr = ir::Div::Make(
      ir::Add::Make(ir::Mul::Make(ir::Mod::Make(var_i_j_fused, ir::Expr(16)),
                                  ir::Expr(128)),
                    var_j_0),
      ir::Expr(128));

  VLOG(6) << "Before Normalize: " << expr;
  auto res = expr.as_index().ir::IndexExpr::Normalize(
      ir::IndexExpr::OptLevel::kLevel3);
  VLOG(6) << "After Normalize: " << res;

  // Expected output verification
  std::string expected_ir = R"ROC((i_j_fused % 16))ROC";

  EXPECT_EQ(utils::GetStreamCnt(res), utils::Trim(expected_ir));
}

/*
i_j_fused: [0ll, 524288ll)
j_0: [0, 128)
Before Normalize:
((((i_j_fused % 16) * 128) + j_0) % 128)
After Normalize:
j_0
*/
TEST(IRSimplifyBound, SimplifyLinearMod) {
  Context::Global().ResetNameId();

  // Create input IR matching the specified pattern
  // Define loop variables
  ir::Var var_i_j_fused = ir::Var(ir::Expr(0), ir::Expr(524288), "i_j_fused");
  ir::Var var_j_0 = ir::Var(ir::Expr(0), ir::Expr(128), "j_0");

  // Final expression
  ir::Expr expr = ir::Mod::Make(
      ir::Add::Make(ir::Mul::Make(ir::Mod::Make(var_i_j_fused, ir::Expr(16)),
                                  ir::Expr(128)),
                    var_j_0),
      ir::Expr(128));

  VLOG(6) << "Before Normalize: " << expr;
  auto res = expr.as_index().ir::IndexExpr::Normalize(
      ir::IndexExpr::OptLevel::kLevel3);
  VLOG(6) << "After Normalize: " << res;

  // Expected output verification
  std::string expected_ir = R"ROC(j_0)ROC";

  EXPECT_EQ(utils::GetStreamCnt(res), utils::Trim(expected_ir));
}

/*
loop_var_2: [0, 32)
loop_var_3: [0, 4)
Before Normalize:
(((loop_var_3 * 32ll) + loop_var_2) / 128ll)
After Normalize:
0
*/
TEST(IRSimplifyBound, SimplifyLinearDiv2) {
  Context::Global().ResetNameId();

  // Create input IR matching the specified pattern
  // Define loop variables
  ir::Var loop_var_2 = ir::Var(ir::Expr(0), ir::Expr(32), "loop_var_2");
  ir::Var loop_var_3 = ir::Var(ir::Expr(0), ir::Expr(4), "loop_var_3");

  // Final expression
  ir::Expr expr = ir::Div::Make(
      ir::Add::Make(ir::Mul::Make(loop_var_3, ir::Expr(32)), loop_var_2),
      ir::Expr(128));

  VLOG(6) << "Before Normalize: " << expr;
  auto res = expr.as_index().ir::IndexExpr::Normalize(
      ir::IndexExpr::OptLevel::kLevel3);
  VLOG(6) << "After Normalize: " << res;

  // Expected output verification
  std::string expected_ir = R"ROC(0)ROC";

  EXPECT_EQ(utils::GetStreamCnt(res), utils::Trim(expected_ir));
}

}  // namespace optim
}  // namespace cinn
