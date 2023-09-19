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

#include "paddle/cinn/lang/lower.h"

#include <gtest/gtest.h>

#include <set>

#include "paddle/cinn/ast_gen_ius/tensor_group.h"
#include "paddle/cinn/cinn.h"
#include "paddle/cinn/lang/buffer.h"
#include "paddle/cinn/lang/compute.h"
#include "paddle/cinn/lang/placeholder.h"
#include "paddle/cinn/utils/string.h"

namespace cinn {
namespace lang {

#define TEST_SOUTPUT(x, out)           \
  LOG(INFO) << "\n" << x << std::endl; \
  EXPECT_EQ(utils::GetStreamCnt(x), utils::Trim(out));

TEST(lower, basic) {
  auto M = Expr(100);
  auto N = Expr(15);

  Placeholder<float> A("A", {Expr(M), Expr(N)});

  auto B = Compute(
      {M, N}, [=](Var i, Var j) -> Expr { return A(i, j) + 1.f; }, "B");

  auto stages = CreateStages({B});

  auto lower_funcs = Lower("cal_B", stages, {A, B});

  LOG(INFO) << "lower_size " << lower_funcs;

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

  LOG(INFO) << "func:\n" << Expr(lower_funcs->self()) << std::endl;
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

TEST(lower_to_ast, basic) {
  Context::Global().ResetNameId();
  auto M = Expr(100);
  auto N = Expr(15);

  Placeholder<float> A("A", {Expr(M), Expr(N)});

  ir::Tensor B = Compute(
      {M, N}, [=](Var i, Var j) -> Expr { return A(i, j) + 1.f; }, "B");

  ast_gen_ius::TensorGroup tensor_group({B});

  ir::LoweredFunc lower_func = LowerToAst("cal_B", {A, B}, &tensor_group);

  LOG(INFO) << "lower_func " << lower_func;

  auto out = R"ROC(
function cal_B (_A, _B)
{
  serial for (i, 0, 100)
  {
    serial for (j, 0, 15)
    {
      B[i, j] = (A[i, j] + 1.00000000f)
    }
  }
}
)ROC";
  TEST_SOUTPUT(lower_func, out);
}

TEST(lower_to_ast, three_dim) {
  Context::Global().ResetNameId();
  Expr M(100);
  Expr N(15);
  Expr K(200);

  Placeholder<float> A("A", {Expr(M), Expr(N)});
  Placeholder<float> B("B", {Expr(N), Expr(K)});

  auto C = Compute(
      {M, N, K},
      [=](Var i, Var j, Var k) -> Expr { return A(i, j) * B(j, k); },
      "C");

  ast_gen_ius::TensorGroup tensor_group({C});

  ir::LoweredFunc lower_func = LowerToAst("cal_C", {A, B, C}, &tensor_group);

  LOG(INFO) << "func:\n" << lower_func << std::endl;

  auto out = R"ROC(
function cal_C (_A, _B, _C)
{
  serial for (i, 0, 100)
  {
    serial for (j, 0, 15)
    {
      serial for (k, 0, 200)
      {
        C[i, j, k] = (A[i, j] * B[j, k])
      }
    }
  }
}
)ROC";
  TEST_SOUTPUT(lower_func, out);
}

TEST(lower_to_ast, matmul_with_reduce_sum) {
  Context::Global().ResetNameId();
  Placeholder<float> A("A", {Expr(100), Expr(20)});
  Placeholder<float> B("B", {Expr(20), Expr(50)});

  Target target{};
  // C = A * B
  Var k(20, "k0");
  Tensor C = Compute(
      {Expr(100), Expr(50)},
      [&](Var i, Var j) { return lang::ReduceSum(A(i, k) * B(k, j), {k}); },
      "C");

  ast_gen_ius::TensorGroup tensor_group({C});
  ir::LoweredFunc lower_func = LowerToAst("matmul", {A, B, C}, &tensor_group);
  LOG(INFO) << "func:\n" << lower_func << std::endl;

  auto out = R"ROC(
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
)ROC";
  TEST_SOUTPUT(lower_func, out);
}

}  // namespace lang
}  // namespace cinn
