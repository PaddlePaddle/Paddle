// Copyright (c) 2021 CINN Authors. All Rights Reserved.
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

#include "paddle/cinn/optim/remove_schedule_block.h"

#include <gtest/gtest.h>

#include <string>
#include <vector>

#include "paddle/cinn/cinn.h"
#include "paddle/cinn/ir/ir.h"
#include "paddle/cinn/ir/op/ir_operators.h"
#include "paddle/cinn/ir/utils/ir_printer.h"
#include "paddle/cinn/utils/string.h"

namespace cinn {
namespace optim {

TEST(RemovescheduleBlock, basic) {
  using namespace ir;  // NOLINT
  Context::Global().ResetNameId();
  Placeholder<float> A("A", {Expr(100), Expr(20)});
  Placeholder<float> B("B", {Expr(20), Expr(50)});
  Target target = common::DefaultHostTarget();
  Module::Builder builder("matmul", target);
  // C = A * B
  Var k(20, "k0");
  Tensor C = Compute(
      {Expr(100), Expr(50)},
      [&](Var i, Var j) { return lang::ReduceSum(A(i, k) * B(k, j), {k}); },
      "C");
  auto stages = CreateStages({A, B, C});
  auto func = Lower("matmul", stages, {A, B, C}, {}, {}, nullptr, target, true);
  LOG(INFO) << "func\n" << func;

  std::string origin = utils::GetStreamCnt(func);
  EXPECT_EQ(origin, utils::Trim(R"ROC(
function matmul (_A, _B, _C)
{
  ScheduleBlock(root)
  {
    serial for (i, 0, 100)
    {
      serial for (j, 0, 50)
      {
        ScheduleBlock(C__reduce_init)
        {
          i0, i1 = axis.bind(i, j)
          C__reduce_init[i0, i1] = 0.00000000f
        }
        serial for (k0, 0, 20)
        {
          ScheduleBlock(C)
          {
            i0_0, i1_0, i2 = axis.bind(i, j, k0)
            C[i0_0, i1_0] = (C[i0_0, i1_0] + (A[i0_0, i2] * B[i2, i1_0]))
          }
        }
      }
    }
  }
}
)ROC"));

  RemoveScheduleBlock(&func->body);

  std::cout << "after RemovescheduleBlock:\n" << func << std::endl;

  EXPECT_EQ(utils::GetStreamCnt(func), utils::Trim(R"ROC(
function matmul (_A, _B, _C)
{
  serial for (i, 0, 100)
  {
    serial for (j, 0, 50)
    {
      C__reduce_init[i, j] = 0.00000000f
      serial for (k0, 0, 20)
      {
        C[i, j] = (C[i, j] + (A[i, k0] * B[k0, j]))
      }
    }
  }
}
)ROC"));
}

}  // namespace optim
}  // namespace cinn
