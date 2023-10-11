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

#include "paddle/cinn/ir/utils/ir_compare.h"

#include <glog/logging.h>
#include <gtest/gtest.h>

#include "paddle/cinn/cinn.h"
#include "paddle/cinn/common/context.h"
#include "paddle/cinn/utils/string.h"

namespace cinn {
namespace ir {
namespace ir_utils {
TEST(TestIrCompare, SingleFunction) {
  Target target = common::DefaultHostTarget();

  ir::Expr M(32);
  ir::Expr N(32);

  lang::Placeholder<float> A("A", {M, N});
  ir::Tensor B = lang::Compute(
      {M, N}, [&](Var i, Var j) { return A(i, j) + ir::Expr(2.f); }, "B");
  ir::Tensor C = lang::Compute(
      {M, N}, [&](Var i, Var j) { return A(i, j) + ir::Expr(2.f); }, "C");

  cinn::common::Context::Global().ResetNameId();
  auto funcs_1 = lang::LowerVec("add_const",
                                poly::CreateStages({A, B}),
                                {A, B},
                                {},
                                {},
                                nullptr,
                                target,
                                true);

  cinn::common::Context::Global().ResetNameId();
  auto funcs_2 = lang::LowerVec("add_const",
                                poly::CreateStages({A, B}),
                                {A, B},
                                {},
                                {},
                                nullptr,
                                target,
                                true);

  cinn::common::Context::Global().ResetNameId();
  auto funcs_3 = lang::LowerVec("add_const",
                                poly::CreateStages({A, C}),
                                {A, C},
                                {},
                                {},
                                nullptr,
                                target,
                                true);

  ASSERT_EQ(funcs_1.size(), 1);
  ASSERT_EQ(funcs_2.size(), 1);
  ASSERT_EQ(funcs_3.size(), 1);

  std::string func1_str = R"ROC(function add_const (_A, _B)
{
  ScheduleBlock(root)
  {
    serial for (i, 0, 32)
    {
      serial for (j, 0, 32)
      {
        ScheduleBlock(B)
        {
          i0, i1 = axis.bind(i, j)
          B[i0, i1] = (A[i0, i1] + 2.00000000f)
        }
      }
    }
  }
})ROC";

  std::string func2_str = R"ROC(function add_const (_A, _B)
{
  ScheduleBlock(root)
  {
    serial for (i, 0, 32)
    {
      serial for (j, 0, 32)
      {
        ScheduleBlock(B)
        {
          i0, i1 = axis.bind(i, j)
          B[i0, i1] = (A[i0, i1] + 2.00000000f)
        }
      }
    }
  }
})ROC";

  std::string func3_str = R"ROC(function add_const (_A, _C)
{
  ScheduleBlock(root)
  {
    serial for (i, 0, 32)
    {
      serial for (j, 0, 32)
      {
        ScheduleBlock(C)
        {
          i0, i1 = axis.bind(i, j)
          C[i0, i1] = (A[i0, i1] + 2.00000000f)
        }
      }
    }
  }
})ROC";

  ASSERT_EQ(func1_str, utils::GetStreamCnt(funcs_1.front()));
  ASSERT_EQ(func2_str, utils::GetStreamCnt(funcs_2.front()));
  ASSERT_EQ(func3_str, utils::GetStreamCnt(funcs_3.front()));

  // they are different at the name of root ScheduleBlock
  ASSERT_TRUE(IRCompare(funcs_1.front(), funcs_2.front()));
  // compare with itself
  ASSERT_TRUE(IRCompare(funcs_1.front(), funcs_1.front()));
  // they are euqal if allowing suffix of name different
  ASSERT_TRUE(IRCompare(funcs_1.front(), funcs_2.front(), true));

  ASSERT_FALSE(IRCompare(funcs_1.front(), funcs_3.front()));
  ASSERT_FALSE(IRCompare(funcs_1.front(), funcs_3.front(), true));
}
}  // namespace ir_utils
}  // namespace ir
}  // namespace cinn
