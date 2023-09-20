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

#include "paddle/cinn/auto_schedule/search_space/search_state.h"

#include <glog/logging.h>
#include <gtest/gtest.h>

#include "paddle/cinn/cinn.h"
#include "paddle/cinn/common/context.h"

namespace cinn {
namespace auto_schedule {

TEST(TestSearchState, SearchStateHash_Equal) {
  Target target = common::DefaultHostTarget();

  ir::Expr M(32);
  ir::Expr N(32);

  lang::Placeholder<float> A("A", {M, N});
  ir::Tensor B = lang::Compute(
      {M, N}, [&](Var i, Var j) { return A(i, j) + ir::Expr(2.f); }, "B");
  ir::Tensor C = lang::Compute(
      {M, N}, [&](Var i, Var j) { return A(i, j) + B(i, j); }, "C");

  cinn::common::Context::Global().ResetNameId();
  auto a_plus_const_funcs_1 = lang::LowerVec("A_plus_const",
                                             poly::CreateStages({A, B}),
                                             {A, B},
                                             {},
                                             {},
                                             nullptr,
                                             target,
                                             true);

  cinn::common::Context::Global().ResetNameId();
  auto a_plus_const_funcs_2 = lang::LowerVec("A_plus_const",
                                             poly::CreateStages({A, B}),
                                             {A, B},
                                             {},
                                             {},
                                             nullptr,
                                             target,
                                             true);

  cinn::common::Context::Global().ResetNameId();
  auto a_plus_b_funcs = lang::LowerVec("A_plus_B",
                                       poly::CreateStages({A, C}),
                                       {A, C},
                                       {},
                                       {},
                                       nullptr,
                                       target,
                                       true);

  std::string a_plus_const_funcs_1_str = R"ROC(function A_plus_const (_A, _B)
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

  std::string a_plus_const_funcs_2_str = R"ROC(function A_plus_const (_A, _B)
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

  std::string a_plus_b_funcs_str = R"ROC(function A_plus_B (_A, _C)
{
  ScheduleBlock(root)
  {
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
      serial for (i, 0, 32)
      {
        serial for (j, 0, 32)
        {
          ScheduleBlock(C)
          {
            i0_0, i1_0 = axis.bind(i, j)
            C[i0_0, i1_0] = (A[i0_0, i1_0] + B[i0_0, i1_0])
          }
        }
      }
    }
  }
})ROC";

  ASSERT_EQ(a_plus_const_funcs_1.size(), 1);
  EXPECT_EQ(a_plus_const_funcs_1_str,
            utils::GetStreamCnt(a_plus_const_funcs_1.front()));
  ASSERT_EQ(a_plus_const_funcs_2.size(), 1);
  EXPECT_EQ(a_plus_const_funcs_2_str,
            utils::GetStreamCnt(a_plus_const_funcs_2.front()));
  ASSERT_EQ(a_plus_b_funcs.size(), 1);
  EXPECT_EQ(a_plus_b_funcs_str, utils::GetStreamCnt(a_plus_b_funcs.front()));

  SearchState a_plus_const_state1(
      ir::IRSchedule(ir::ModuleExpr({a_plus_const_funcs_1.front()->body})));
  SearchState a_plus_const_state2(
      ir::IRSchedule(ir::ModuleExpr({a_plus_const_funcs_2.front()->body})));
  SearchState a_plus_b_state(
      ir::IRSchedule(ir::ModuleExpr({a_plus_b_funcs.front()->body})));

  SearchStateHash hash_functor;
  SearchStateEqual equal_functor;
  ASSERT_EQ(hash_functor(a_plus_const_state1),
            hash_functor(a_plus_const_state2));
  ASSERT_TRUE(equal_functor(a_plus_const_state1, a_plus_const_state2));
  ASSERT_NE(hash_functor(a_plus_const_state1), hash_functor(a_plus_b_state));
  ASSERT_FALSE(equal_functor(a_plus_const_state1, a_plus_b_state));
}

}  // namespace auto_schedule
}  // namespace cinn
