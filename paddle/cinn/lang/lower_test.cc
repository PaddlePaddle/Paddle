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

#include "cinn/lang/lower.h"

#include <gtest/gtest.h>

#include <set>

#include "cinn/cinn.h"
#include "cinn/lang/buffer.h"
#include "cinn/lang/compute.h"
#include "cinn/lang/placeholder.h"
#include "cinn/utils/string.h"

namespace cinn {
namespace lang {

TEST(lower, basic) {
  auto M = Expr(100);
  auto N = Expr(15);

  Placeholder<float> A("A", {Expr(M), Expr(N)});

  auto B = Compute(
      {M, N}, [=](Var i, Var j) -> Expr { return A(i, j) + 1.f; }, "B");

  auto stages = CreateStages({B});

  auto lower_funcs = Lower("cal_B", stages, {A, B});

  LOG(INFO) << "lower_size " << lower_funcs;

#define TEST_SOUTPUT(x, out)           \
  std::cout << "\n" << x << std::endl; \
  EXPECT_EQ(utils::GetStreamCnt(x), utils::Trim(out));

  auto out = R"ROC(
{
  serial for (i, 0, 100)
  {
    serial for (j, 0, 15)
    {
      B[i, j] = (1.00000000f + A[i, j])
    }
  }
}
)ROC";
  TEST_SOUTPUT(lower_funcs->body, out);
}

TEST(lower, more_complex) {
  Expr M(100);
  Expr N(15);
  Expr K(200);

  Placeholder<float> A("A", {Expr(M), Expr(N)});
  Placeholder<float> B("B", {Expr(N), Expr(K)});

  auto C = Compute(
      {M, N, K},
      [=](Var i, Var j, Var k) -> Expr { return A(i, j) * B(j, k); },
      "C");

  auto stages = CreateStages({C});

  auto lower_funcs = Lower("cal_C", stages, {A, B, C});

  std::cout << "func:\n" << Expr(lower_funcs->self()) << std::endl;
}

//! To support training, the dynamic shape support is vital. We test the
//! corresponding lower ability here.
TEST(lower, dynamic_shape) {
  Var B("B");  // B is like shape here.
  Expr N(15);
  Expr K(200);

  // Input is B * N, B is like batch.
  Placeholder<float> X("X", {Expr(B), Expr(N)});
  Placeholder<float> W("W", {Expr(N), Expr(K)});

  auto C = Compute(
      {B, N, K},
      [=](Var i, Var j, Var k) -> Expr { return X(i, j) * W(j, k); },
      "C");

  auto stages = CreateStages({C});
  auto lower_funcs = Lower("cal_C", stages, {X, W, C});

  std::cout << "func:\n" << Expr(lower_funcs->self()) << std::endl;
}

TEST(lower, lowered_call) {
  Var B("B");  // B is like shape here.
  Expr N(15);

  // Input is B * N, B is like batch.
  Placeholder<float> X("X", {Expr(B), Expr(N)});
  Placeholder<float> Y("Y", {Expr(B), Expr(N)});

  auto Z = Compute(
      {B, N}, [&](Var i, Var j) { return X(i, j) + Y(i, j); }, "Z");

  std::vector<ReturnType> return_types(
      {{Float(32), std::vector<Expr>{{B, N}}, "C"}});
  auto tensors = CallLowered("lowered_fun0", {X, Y, Z}, return_types);
  auto C = tensors[0];

  auto stages = CreateStages({X, Y, Z, C});

  LOG(INFO) << "call_op: " << C->operation->as<ir::CallOp>()->call_expr;

  auto lower_func = Lower("fn", stages, {X, Y, Z, C});
}

// test the temp_buffers are all collected.
TEST(lower, temp_buffer_collects) {
  Expr M(10);

  Placeholder<float> A("A", {M});

  auto B = Compute(
      {M}, [&](Expr i) -> Expr { return A(i); }, "B");  // temp
  auto C = Compute(
      {M}, [&](Expr i) -> Expr { return B(i); }, "C");  // temp
  auto D = Compute(
      {M}, [&](Expr i) -> Expr { return C(i); }, "D");  // temp
  auto output = Compute(
      {M}, [&](Expr i) -> Expr { return D(i); }, "output");

  ir::Module::Builder b("somemodule", common::DefaultHostTarget());

  auto stages = CreateStages({B, C, D, output});

  auto fn = Lower("fn", stages, {A, output}, {}, {}, &b);

  auto module = b.Build();

  ASSERT_EQ(module.buffers().size(), 3UL);

  std::set<std::string> detected_buffer_names({"_B", "_C", "_D"});

  for (auto& buffer : module.buffers()) {
    ASSERT_TRUE(detected_buffer_names.count(buffer->name));
  }
}

}  // namespace lang
}  // namespace cinn
