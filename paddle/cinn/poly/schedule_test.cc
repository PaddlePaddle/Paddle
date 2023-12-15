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

#include "paddle/cinn/poly/schedule.h"

#include <gtest/gtest.h>

#include "paddle/cinn/ir/op/ir_operators.h"
#include "paddle/cinn/ir/tensor.h"
#include "paddle/cinn/lang/compute.h"
#include "paddle/cinn/lang/lower.h"
#include "paddle/cinn/lang/placeholder.h"

namespace cinn {
namespace poly {

TEST(CreateStages, compute_at) {
  Expr N(100);
  lang::Placeholder<float> A("A", {N, N});

  auto B = lang::Compute(
      {N, N}, [&](Var i, Var j) { return A(i, j) + 1.f; }, "B");

  auto C = lang::Compute(
      {N, N, N}, [&](Var i, Var j, Var k) { return B(i, j) * B(j, k); }, "C");

  auto stages = CreateStages({C});
  stages[B]->ComputeAtSchedule(stages[C], 1);

  auto funcs = lang::Lower("func", stages, {B, C});

  std::cout << funcs->body << std::endl;

  auto target_out = R"ROC(
{
  serial for (i, 0, 100)
  {
    serial for (j, 0, 100)
    {
      B[i, j] = (1.00000000f + A[i, j])
      serial for (k, 0, 100)
      {
        C[i, j, k] = (B[i, j] * B[j, k])
      }
    }
  }
}
)ROC";

  EXPECT_EQ(utils::GetStreamCnt(funcs->body), utils::Trim(target_out));
}

TEST(CreateStages, buffer_bind_to_multiple_tensors_schedule) {
  Expr N(100);
  lang::Placeholder<float> A("A", {N, N});
  /*
   * We create three tensors all binded to the same buffer, but has no depend in
   * computation.
   */

  auto B = lang::Compute(
      {N, N}, [&](Var i, Var j) { return A(i, j) + 1.f; }, "B");
  lang::Buffer B_buf(B->type());
  B->Bind(B_buf);

  auto C = lang::Compute(
      {N, N}, [&](Var i, Var j) { return A(i, j) + 1.f; }, "C");
  C->Bind(B_buf);

  auto D = lang::Compute(
      {N, N}, [&](Var i, Var j) { return A(i, j) + 1.f; }, "D");
  D->Bind(B_buf);

  auto stages = CreateStages({B, C, D});

  stages[C]->ShareBufferWith(stages[B]);
  stages[D]->ShareBufferWith(stages[B]);
  stages[C]->CtrlDepend(B);
  stages[D]->CtrlDepend(C);

  auto funcs = lang::Lower("func", stages, {B, C, D});

  std::cout << funcs->body << std::endl;

  auto target_out = R"ROC(
{
  serial for (i, 0, 100)
  {
    serial for (j, 0, 100)
    {
      B[i, j] = (1.00000000f + A[i, j])
    }
  }
  serial for (i, 0, 100)
  {
    serial for (j, 0, 100)
    {
      C[i, j] = (1.00000000f + A[i, j])
    }
  }
  serial for (i, 0, 100)
  {
    serial for (j, 0, 100)
    {
      D[i, j] = (1.00000000f + A[i, j])
    }
  }
}
)ROC";

  EXPECT_EQ(utils::GetStreamCnt(funcs->body), utils::Trim(target_out));
}

}  // namespace poly
}  // namespace cinn
