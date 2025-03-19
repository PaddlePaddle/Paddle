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
}  // namespace common
}  // namespace cinn
