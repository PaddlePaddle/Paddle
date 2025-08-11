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

#include <glog/logging.h>
#include <gtest/gtest.h>
#include "paddle/cinn/common/ir_util.h"
#include "paddle/cinn/ir/op/ir_operators.h"
#include "paddle/cinn/ir/schedule/schedule_base.h"
#include "paddle/cinn/ir/utils/stmt_converter.h"
#include "paddle/cinn/optim/if_fold_pass.h"
#include "paddle/cinn/pass/pass_manager.h"
#include "paddle/cinn/utils/string.h"
namespace cinn {
namespace common {
#define MAKE_FUNC(body)                                                        \
  std::vector<ir::Argument> args{                                              \
      ir::Argument(ir::Var("A"), ir::Argument::IO::kInput),                    \
      ir::Argument(ir::Var("B"), ir::Argument::IO::kOutput)};                  \
  auto new_func =                                                              \
      ir::_LoweredFunc_::Make("test_func", args, ir::Block::Make({body}), {}); \
  optim::StmtPassManager pass_manager;                                         \
  pass_manager.AddPass(optim::CreateIfFoldPass());                             \
  pass_manager.Run(new_func);

/*
 * serial for (i, 0, 2)
 * {
 *   serial for (j, 0, 4)
 *   {
 *     serial for (k, 0, 8)
 *     {
 *       if ((((((256 * j) + ((1024 * i) + k)) / 56) / 56) == 0)) {
 *         if ((((((256 * j) + ((1024 * i) + k)) / 56) % 56) == 0)) {
 *           if (((((256 * j) + ((1024 * i) + k)) % 56) == 0)) {
 *             int32 a = 1
 *           }
 *         }
 *       }
 *     }
 *   }
 * }
 */
TEST(IRSimplify, if_fold_correct_0) {
  std::vector<ir::Expr> shape = {Expr(2), Expr(4), Expr(8)};
  std::vector<Var> axis_vars = cinn::common::GenDefaultAxis(3);

  auto body = ir::IfThenElse::Make(
      ir::EQ::Make(
          ((((256 * axis_vars[1]) + ((1024 * axis_vars[0]) + axis_vars[2])) /
            56) /
           56),
          Expr(0)),
      ir::IfThenElse::Make(
          ir::EQ::Make(((((256 * axis_vars[1]) +
                          ((1024 * axis_vars[0]) + axis_vars[2])) /
                         56) %
                        56),
                       Expr(0)),
          ir::IfThenElse::Make(
              ir::EQ::Make(((256 * axis_vars[1]) +
                            ((1024 * axis_vars[0]) + axis_vars[2])) %
                               56,
                           Expr(0)),
              ir::Block::Make({ir::Let::Make(ir::Var("a"), Expr(1))}))));
  for (int i = shape.size() - 1; i >= 0; --i) {
    ir::Var loop_var = axis_vars[i];
    ir::Expr loop_extent = shape[i];
    body = ir::For::Make(loop_var,
                         Expr(0),
                         loop_extent,
                         ir::ForType::Serial,
                         ir::DeviceAPI::Host,
                         ir::Block::Make({body}));
  }

  MAKE_FUNC(body);
  EXPECT_EQ(utils::GetStreamCnt(new_func),
            utils::Trim(R"ROC(function test_func (A, B)
{
  serial for (i, 0, 2)
  {
    serial for (j, 0, 4)
    {
      serial for (k, 0, 8)
      {
        if (((((i * 1024) + k) + (j * 256)) == 0)) {
          int32 a = 1
        }
      }
    }
  }
}
)ROC"));
}

/*
 * serial for (i, 0, 2)
 * {
 *   serial for (j, 0, 4)
 *   {
 *     serial for (k, 0, 8)
 *     {
 *       if ((((((256 * j) + ((1024 * i) + k)) / 56) / 56) == 0)) {
 *         if ((((((256 * j) + ((1024 * i) + k)) / 56) % 56) == 0)) {
 *           if (((((256 * j) + ((1024 * i) + k)) % 56) == 0)) {
 *             int32 a = 1
 *             int32 b = 1
 *           }
 *         }
 *       }
 *     }
 *   }
 * }
 */
TEST(IRSimplify, if_fold_correct_1) {
  std::vector<ir::Expr> shape = {Expr(2), Expr(4), Expr(8)};
  std::vector<Var> axis_vars = cinn::common::GenDefaultAxis(3);

  auto body = ir::IfThenElse::Make(
      ir::EQ::Make(
          ((((256 * axis_vars[1]) + ((1024 * axis_vars[0]) + axis_vars[2])) /
            56) /
           56),
          Expr(0)),
      ir::IfThenElse::Make(
          ir::EQ::Make(((((256 * axis_vars[1]) +
                          ((1024 * axis_vars[0]) + axis_vars[2])) /
                         56) %
                        56),
                       Expr(0)),
          ir::IfThenElse::Make(
              ir::EQ::Make(((256 * axis_vars[1]) +
                            ((1024 * axis_vars[0]) + axis_vars[2])) %
                               56,
                           Expr(0)),
              ir::Block::Make({ir::Let::Make(ir::Var("a"), Expr(1)),
                               ir::Let::Make(ir::Var("b"), Expr(1))}))));
  for (int i = shape.size() - 1; i >= 0; --i) {
    ir::Var loop_var = axis_vars[i];
    ir::Expr loop_extent = shape[i];
    body = ir::For::Make(loop_var,
                         Expr(0),
                         loop_extent,
                         ir::ForType::Serial,
                         ir::DeviceAPI::Host,
                         ir::Block::Make({body}));
  }

  MAKE_FUNC(body);
  EXPECT_EQ(utils::GetStreamCnt(new_func),
            utils::Trim(R"ROC(function test_func (A, B)
{
  serial for (i, 0, 2)
  {
    serial for (j, 0, 4)
    {
      serial for (k, 0, 8)
      {
        if (((((i * 1024) + k) + (j * 256)) == 0)) {
          int32 a = 1
          int32 b = 1
        }
      }
    }
  }
}
)ROC"));
}

/*
 * serial for (i, 0, 2)
 * {
 *   serial for (j, 0, 4)
 *   {
 *     serial for (k, 0, 8)
 *     {
 *       if ((((((256 * j) + ((1024 * i) + k)) / 56) / 56) == 0)) {
 *         if ((((((256 * j) + ((1024 * i) + k)) / 56) % 56) == 0)) {
 *           if (((((256 * j) + ((1024 * i) + k)) % 56) == 0)) {
 *             int32 a = 1
 *             int32 b = 1
 *           } else {
 *             int32 c = 1
 *           }
 *         }
 *       }
 *     }
 *   }
 * }
 */
TEST(IRSimplify, if_fold_correct_2) {
  std::vector<ir::Expr> shape = {Expr(2), Expr(4), Expr(8)};
  std::vector<Var> axis_vars = cinn::common::GenDefaultAxis(3);

  auto body = ir::IfThenElse::Make(
      ir::EQ::Make(
          ((((256 * axis_vars[1]) + ((1024 * axis_vars[0]) + axis_vars[2])) /
            56) /
           56),
          Expr(0)),
      ir::IfThenElse::Make(
          ir::EQ::Make(((((256 * axis_vars[1]) +
                          ((1024 * axis_vars[0]) + axis_vars[2])) /
                         56) %
                        56),
                       Expr(0)),
          ir::IfThenElse::Make(
              ir::EQ::Make(((256 * axis_vars[1]) +
                            ((1024 * axis_vars[0]) + axis_vars[2])) %
                               56,
                           Expr(0)),
              ir::Block::Make({ir::Let::Make(ir::Var("a"), Expr(1)),
                               ir::Let::Make(ir::Var("b"), Expr(1))}),
              ir::Block::Make({ir::Let::Make(ir::Var("c"), Expr(1))}))));
  for (int i = shape.size() - 1; i >= 0; --i) {
    ir::Var loop_var = axis_vars[i];
    ir::Expr loop_extent = shape[i];
    body = ir::For::Make(loop_var,
                         Expr(0),
                         loop_extent,
                         ir::ForType::Serial,
                         ir::DeviceAPI::Host,
                         ir::Block::Make({body}));
  }

  MAKE_FUNC(body);
  EXPECT_EQ(utils::GetStreamCnt(new_func),
            utils::Trim(R"ROC(function test_func (A, B)
{
  serial for (i, 0, 2)
  {
    serial for (j, 0, 4)
    {
      serial for (k, 0, 8)
      {
        if (((((i * 1024) + k) + (j * 256)) == 0)) {
          int32 a = 1
          int32 b = 1
        } else {
          int32 c = 1
        }
      }
    }
  }
}
)ROC"));
}

/*
 * serial for (i, 0, 2)
 * {
 *   serial for (j, 0, 4)
 *   {
 *     serial for (k, 0, 8)
 *     {
 *       if ((((((256 * j) + ((1024 * i) + k)) / 56) / 56) == 0)) {
 *         if ((((((256 * j) + ((1024 * i) + k)) / 56) % 56) == 0)) {
 *           if (((((256 * j) + ((1024 * i) + k)) % 56) == 0)) {
 *             if (((((256 * j) + ((1024 * i) + k)) % 56) <= 0)) {
 *               int32 a = 1
 *             }
 *           }
 *         }
 *       }
 *     }
 *   }
 * }
 */
TEST(IRSimplify, if_fold_correct_3) {
  std::vector<ir::Expr> shape = {Expr(2), Expr(4), Expr(8)};
  std::vector<Var> axis_vars = cinn::common::GenDefaultAxis(3);

  auto body = ir::IfThenElse::Make(
      ir::EQ::Make(
          ((((256 * axis_vars[1]) + ((1024 * axis_vars[0]) + axis_vars[2])) /
            56) /
           56),
          Expr(0)),
      ir::IfThenElse::Make(
          ir::EQ::Make(((((256 * axis_vars[1]) +
                          ((1024 * axis_vars[0]) + axis_vars[2])) /
                         56) %
                        56),
                       Expr(0)),
          ir::IfThenElse::Make(
              ir::EQ::Make(((256 * axis_vars[1]) +
                            ((1024 * axis_vars[0]) + axis_vars[2])) %
                               56,
                           Expr(0)),
              ir::IfThenElse::Make(
                  ir::LE::Make(((256 * axis_vars[1]) +
                                ((1024 * axis_vars[0]) + axis_vars[2])) %
                                   56,
                               Expr(0)),
                  ir::Block::Make({ir::Let::Make(ir::Var("a"), Expr(1))})))));
  for (int i = shape.size() - 1; i >= 0; --i) {
    ir::Var loop_var = axis_vars[i];
    ir::Expr loop_extent = shape[i];
    body = ir::For::Make(loop_var,
                         Expr(0),
                         loop_extent,
                         ir::ForType::Serial,
                         ir::DeviceAPI::Host,
                         ir::Block::Make({body}));
  }

  MAKE_FUNC(body);
  EXPECT_EQ(utils::GetStreamCnt(new_func),
            utils::Trim(R"ROC(function test_func (A, B)
{
  serial for (i, 0, 2)
  {
    serial for (j, 0, 4)
    {
      serial for (k, 0, 8)
      {
        if (((((i * 1024) + k) + (j * 256)) == 0)) {
          if (((((256 * j) + ((1024 * i) + k)) % 56) <= 0)) {
            int32 a = 1
          }
        }
      }
    }
  }
}
)ROC"));
}

/*
 * if ((((((256 * j) + ((1024 * i) + k)) / 56) / 56) == 0)) {
 *   if ((((((256 * j) + ((1024 * i) + k)) / 56) % 56) == 0)) {
 *     if (((((256 * j) + ((1024 * i) + k)) % 56) == 0)) {
 *       int32 a = 1
 *     }
 *   } else {
 *     int32 b = 1
 *   }
 * }
 */
TEST(IRSimplify, if_fold_has_false_brh) {
  std::vector<Var> axis_vars = cinn::common::GenDefaultAxis(3);

  auto body = ir::IfThenElse::Make(
      ir::EQ::Make(
          ((((256 * axis_vars[1]) + ((1024 * axis_vars[0]) + axis_vars[2])) /
            56) /
           56),
          Expr(0)),
      ir::IfThenElse::Make(
          ir::EQ::Make(((((256 * axis_vars[1]) +
                          ((1024 * axis_vars[0]) + axis_vars[2])) /
                         56) %
                        56),
                       Expr(0)),
          ir::IfThenElse::Make(
              ir::EQ::Make(((256 * axis_vars[1]) +
                            ((1024 * axis_vars[0]) + axis_vars[2])) %
                               56,
                           Expr(0)),
              ir::Block::Make({ir::Let::Make(ir::Var("a"), Expr(1))})),
          ir::Block::Make({ir::Let::Make(ir::Var("b"), Expr(1))})));

  MAKE_FUNC(body);
  EXPECT_EQ(utils::GetStreamCnt(new_func),
            utils::Trim(R"ROC(function test_func (A, B)
{
  if ((((((256 * j) + ((1024 * i) + k)) / 56) / 56) == 0)) {
    if ((((((256 * j) + ((1024 * i) + k)) / 56) % 56) == 0)) {
      if (((((256 * j) + ((1024 * i) + k)) % 56) == 0)) {
        int32 a = 1
      }
    } else {
      int32 b = 1
    }
  }
}
)ROC"));
}

/*
 * if ((((((256 * j) + ((1024 * i) + k)) / 56) / 56) == 0)) {
 *   if ((((((256 * j) + ((1024 * i) + k)) / 56) % 56) <= 0)) {
 *     if (((((256 * j) + ((1024 * i) + k)) % 56) == 0)) {
 *       int32 a = 1
 *     }
 *   }
 * }
 */
TEST(IRSimplify, if_fold_LE) {
  std::vector<Var> axis_vars = cinn::common::GenDefaultAxis(3);

  auto body = ir::IfThenElse::Make(
      ir::EQ::Make(
          ((((256 * axis_vars[1]) + ((1024 * axis_vars[0]) + axis_vars[2])) /
            56) /
           56),
          Expr(0)),
      ir::IfThenElse::Make(
          ir::LE::Make(((((256 * axis_vars[1]) +
                          ((1024 * axis_vars[0]) + axis_vars[2])) /
                         56) %
                        56),
                       Expr(0)),
          ir::IfThenElse::Make(
              ir::EQ::Make(((256 * axis_vars[1]) +
                            ((1024 * axis_vars[0]) + axis_vars[2])) %
                               56,
                           Expr(0)),
              ir::Block::Make({ir::Let::Make(ir::Var("a"), Expr(1))}))));

  MAKE_FUNC(body);
  EXPECT_EQ(utils::GetStreamCnt(new_func),
            utils::Trim(R"ROC(function test_func (A, B)
{
  if ((((((256 * j) + ((1024 * i) + k)) / 56) / 56) == 0)) {
    if ((((((256 * j) + ((1024 * i) + k)) / 56) % 56) <= 0)) {
      if (((((256 * j) + ((1024 * i) + k)) % 56) == 0)) {
        int32 a = 1
      }
    }
  }
}
)ROC"));
}

/*
 * if ((((((256 * j) + ((1024 * i) + k)) / 56) / 56) == 0)) {
 *   if ((((((256 * j) + ((1024 * i) + k)) / 56) % 56) == 2)) {
 *     if (((((256 * j) + ((1024 * i) + k)) % 56) == 0)) {
 *       int32 a = 1
 *     }
 *   }
 * }
 */
TEST(IRSimplify, if_fold_EQ_2) {
  std::vector<Var> axis_vars = cinn::common::GenDefaultAxis(3);

  auto body = ir::IfThenElse::Make(
      ir::EQ::Make(
          ((((256 * axis_vars[1]) + ((1024 * axis_vars[0]) + axis_vars[2])) /
            56) /
           56),
          Expr(0)),
      ir::IfThenElse::Make(
          ir::EQ::Make(((((256 * axis_vars[1]) +
                          ((1024 * axis_vars[0]) + axis_vars[2])) /
                         56) %
                        56),
                       Expr(2)),
          ir::IfThenElse::Make(
              ir::EQ::Make(((256 * axis_vars[1]) +
                            ((1024 * axis_vars[0]) + axis_vars[2])) %
                               56,
                           Expr(0)),
              ir::Block::Make({ir::Let::Make(ir::Var("a"), Expr(1))}))));

  MAKE_FUNC(body);
  EXPECT_EQ(utils::GetStreamCnt(new_func),
            utils::Trim(R"ROC(function test_func (A, B)
{
  if ((((((256 * j) + ((1024 * i) + k)) / 56) / 56) == 0)) {
    if ((((((256 * j) + ((1024 * i) + k)) / 56) % 56) == 2)) {
      if (((((256 * j) + ((1024 * i) + k)) % 56) == 0)) {
        int32 a = 1
      }
    }
  }
}
)ROC"));
}

/*
serial for (i_j_fused, 0ll, 524288ll)
{
  serial for (j_0, 0, 128)
  {
    var_45[(i_j_fused / 16), (((i_j_fused % 16) * 128) + j_0)] =
      pow(2.0f, ceil(log2((0.00223214296f * var_31[0]))))
  }
 }
*/
TEST(IRSimplifyPowerCeilLog2BitOpLdexpf, Base) {
  Context::Global().ResetNameId();

  /// Create input IR matching the specified pattern
  const std::vector<ir::Expr> shape_2d = {ir::Expr(32768), ir::Expr(16)};
  const std::vector<ir::Expr> shape_3d = {ir::Expr(32768), ir::Expr(16)};

  ir::Tensor var_31 =
      ir::_Tensor_::Make("var_31", ir::Float(32), shape_2d, shape_2d);
  var_31->WithBuffer("global", "var_31_buffer");

  ir::Tensor var_45 =
      ir::_Tensor_::Make("var_45", ir::Float(32), shape_3d, shape_3d);
  var_45->WithBuffer("global", "var_45_buffer");

  // Define loop variables
  ir::Var var_i_j_fused = ir::Var(ir::Expr(0), ir::Expr(524288), "i_j_fused");
  ir::Var var_j_0 = ir::Var(ir::Expr(0), ir::Expr(128), "j_0");

  // Create innermost loop body
  ir::Expr body = ir::Store::Make(
      var_45,
      ir::Call::Make(
          ir::Float(32),  // Return type
          "pow",          // Intrinsic function name
          {ir::Expr(2.0f),
           ir::Call::Make(
               ir::Float(32),
               "ceil",
               {ir::Call::Make(
                   ir::Float(32),
                   "log2",
                   {ir::Mul::Make(ir::Expr(0.00223214296f),
                                  ir::Load::Make(var_31, {ir::Expr(0)}))},
                   {},
                   ir::CallType::Intrinsic)},
               {},
               ir::CallType::Intrinsic)},
          {},
          ir::CallType::Intrinsic),
      {ir::Div::Make(var_i_j_fused, ir::Expr(16)),
       ir::Add::Make(ir::Mul::Make(ir::Mod::Make(var_i_j_fused, ir::Expr(16)),
                                   ir::Expr(128)),
                     var_j_0)});

  // Create j_0 loop
  ir::Expr j_0_loop = ir::For::Make(var_j_0,
                                    ir::Expr(0),
                                    ir::Expr(128),
                                    ir::ForType::Serial,
                                    ir::DeviceAPI::Host,
                                    ir::Block::Make({body}));

  // Create i_j_fused loop
  ir::Expr i_j_fused_loop = ir::For::Make(var_i_j_fused,
                                          ir::Expr(0),
                                          ir::Expr(524288),
                                          ir::ForType::Serial,
                                          ir::DeviceAPI::Host,
                                          ir::Block::Make({j_0_loop}));

  // Final expression
  ir::Expr expr = ir::Block::Make({i_j_fused_loop});

  VLOG(6) << "Before Simplify: " << expr;
  cinn::optim::Simplify(&expr);
  VLOG(6) << "After Simplify: " << expr;

  // Expected output verification
  std::string expected_ir = R"ROC({
  serial for (i_j_fused, 0, 524288)
  {
    serial for (j_0, 0, 128)
    {
      var_45[(i_j_fused / 16), (((i_j_fused % 16) * 128) + j_0)] = ldexpf(1.00000000f, ((bitwise_and(right_shift(__float_as_uint((0.00223214296f * var_31[0])), 23), 255) - 127) + select((((bitwise_and(right_shift(__float_as_uint((0.00223214296f * var_31[0])), 23), 255) - 127) != -127) and (bitwise_and(__float_as_uint((0.00223214296f * var_31[0])), 8388607) != 0)), 1, 0)))
    }
  }
})ROC";

  EXPECT_EQ(utils::GetStreamCnt(expr), utils::Trim(expected_ir));
}

}  // namespace common
}  // namespace cinn
