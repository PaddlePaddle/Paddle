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

#include "paddle/cinn/optim/ir_simplify.h"

#include <gtest/gtest.h>

#include "paddle/cinn/cinn.h"
#include "paddle/cinn/ir/ir.h"

namespace cinn {
namespace optim {
using utils::GetStreamCnt;
using utils::Trim;

TEST(IrSimplify, basic) {
  auto A = Compute(
      {Expr(100), Expr(20)}, [&](Var i, Var j) { return Expr(1.f); }, "C");
  Buffer A_buf(A->type());
  A->Bind(A_buf);

  Var i("i"), j("j");
  i->set_type(Int(32));
  j->set_type(Int(32));

  {  // simple case
    auto B = A(i, Expr(0)) + 1.f * 0.f + 100.f + 24.5f;

    LOG(INFO) << "B " << B;
    // get (((C[(i * 20)] + 0) + 100) + 24.5)
    Simplify(&B);
    LOG(INFO) << "simplified: " << B;
    auto out = "(124.500000f + C[i, 0])";
    EXPECT_EQ(out, utils::GetStreamCnt(B));
  }

  {
    Placeholder<float> x("X", {100, 20});
    Placeholder<float> y("y", {100, 20});

    auto B = Compute(
        {Expr(100), Expr(20)},
        [&](Expr i, Expr j) {
          return x(i + 0, j + 0) + y(i, j * 0) * 1.f + 0.f * x(i, j) + 25.f +
                 100.f - 0.f + 9.f * 10000.f * 1.f * 1.f * 0.f;
        },
        "B");

    auto stages = CreateStages({B});
    auto func = Lower("func", stages, {B});
    auto body = func->body;

    LOG(INFO) << "original body:\n" << body;
    Simplify(&body);
    auto target_out = R"ROC(
{
  serial for (i, 0, 100)
  {
    serial for (j, 0, 20)
    {
      B[i, j] = (125.000000f + (X[i, j] + y[i, 0]))
    }
  }
}
)ROC";
    EXPECT_EQ(Trim(target_out), Trim(GetStreamCnt(body)));
  }

  {
    Placeholder<float> x("X", {100, 20});
    Placeholder<float> y("y", {100, 20});

    auto B = Compute(
        {Expr(100), Expr(20)},
        [&](Expr i, Expr j) {
          return x(100 * 10 * 1 * i + 0, j * 0) + y(i, j * 0) / (1.f + 2.f) +
                 0.f * x(i, j) + 25.f + 100.f - 0.f +
                 9.f * 10000.f * 1.f * 1.f * 0.f;
        },
        "B");

    auto stages = CreateStages({B});

    auto func = Lower("func", stages, {B});
    auto body = func->body;

    LOG(INFO) << "original body:\n" << body;
    Simplify(&body);

    auto target_out = R"ROC(
{
  serial for (i, 0, 100)
  {
    serial for (j, 0, 20)
    {
      B[i, j] = ((y[i, 0] / 3.00000000f) + (125.000000f + X[(1000 * i), 0]))
    }
  }
}
)ROC";
    EXPECT_EQ(Trim(target_out), Trim(GetStreamCnt(body)));
  }
}

TEST(reverse, prod) {
  Expr M(100), N(20);
  Placeholder<float> A("A", {M, N});
  auto C = Compute(
      {M, N}, [=](Var i, Var j) { return Expr(1.f) / A(i, j); }, "C");

  auto stages = CreateStages({A, C});
  auto fn = Lower("fn", stages, {A, C});
  LOG(INFO) << "fn:\n" << fn;
}

}  // namespace optim
}  // namespace cinn
