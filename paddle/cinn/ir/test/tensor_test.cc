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

#include "paddle/cinn/ir/tensor.h"

#include <gtest/gtest.h>

#include "paddle/cinn/backends/codegen_c.h"
#include "paddle/cinn/backends/llvm/execution_engine.h"
#include "paddle/cinn/cinn.h"
#include "paddle/cinn/common/test_helper.h"
#include "paddle/cinn/ir/op/ir_operators.h"
#include "paddle/cinn/ir/utils/ir_printer.h"
#include "paddle/cinn/lang/builtin.h"
#include "paddle/cinn/lang/compute.h"
#include "paddle/cinn/lang/lower.h"
#include "paddle/cinn/lang/packed_func.h"
#include "paddle/cinn/lang/placeholder.h"

namespace cinn {
namespace ir {
using utils::GetStreamCnt;
using utils::Trim;

TEST(Tensor, inlined) {
  Expr M(100), N(20);

  Placeholder<float> A("A", {M, N});
  Placeholder<float> B("B", {M, N});

  // C is inlined
  Tensor C = lang::Compute(
      {M, N}, [=](Var i, Var j) { return A(i, j) + B(i, j); }, "C");

  Tensor D = lang::Compute(
      {M, N}, [=](Var i, Var j) -> Expr { return C(i, j) * 2.f + 1.f; }, "D");

  auto stages = CreateStages({D});
  stages[C]->ComputeInline();

  auto func = lang::Lower("func_C", stages, {A, B, D});
  std::cout << "output: \n" << func << std::endl;
  auto out = GetStreamCnt(func);
  EXPECT_EQ(Trim(out), Trim(R"ROC(
function func_C (_A, _B, _D)
{
  serial for (i, 0, 100)
  {
    serial for (j, 0, 20)
    {
      D[i, j] = (1.00000000f + ((2.00000000f * A[i, j]) + (2.00000000f * B[i, j])))
    }
  }
}
)ROC"));
}

TEST(Tensor, IsDependOnStatement) {
  Expr N(100);

  Placeholder<float> X("X", {N});
  auto t = Compute(
      {N}, [&](Var i) -> Expr { return X(i); }, "t");

  ASSERT_TRUE(t->IsDependOnStatement("X"));
  ASSERT_FALSE(t->IsDependOnStatement("XXX"));
}

TEST(Tensor, Reshape) {
  Context::Global().ResetNameId();
  Expr M(100);
  Expr N(100);
  Placeholder<float> A("A", {M, N});

  auto stages = CreateStages({A});

  auto A1 = A->Reshape({Expr(10), Expr(10), Expr(100)}, stages);
  auto B = Compute(
      A1->shape,
      [=](Expr i, Expr j, Expr k) { return A1(i, j, k) * 2.f; },
      "B");

  stages->InsertLazily(B);

  auto func = lang::Lower("fn", stages, {A, B});

  ir::Module::Builder builder("some_modue", common::DefaultHostTarget());
  builder.AddFunction(func);

  backends::CodeGenC codegenc(common::DefaultHostTarget());
  codegenc.SetInlineBuiltinCodes(false);
  auto source = codegenc.Compile(builder.Build(), CodeGenC::OutputKind::CImpl);
  LOG(INFO) << "source:\n" << source;

  auto target_source = R"ROC(
#include <cinn_runtime.h>
#include <stdio.h>

void fn(void* _args, int32_t num_args)
{
  const cinn_buffer_t* _A = cinn_pod_value_to_buffer_p(&(((cinn_pod_value_t*)(_args))[0]));
  cinn_buffer_t* _B = cinn_pod_value_to_buffer_p(&(((cinn_pod_value_t*)(_args))[1]));
  cinn_buffer_malloc((void*)(0), _B);
  const float* A_reshape = ((const float*)(_A->memory));
  float* B = ((float*)(_B->memory));
  for (int32_t i = 0; i < 10; i += 1) {
    for (int32_t j = 0; j < 10; j += 1) {
      for (int32_t k = 0; k < 100; k += 1) {
        B[((1000 * i) + ((100 * j) + k))] = (2.00000000f * A_reshape[((1000 * i) + ((100 * j) + k))]);
      };
    };
  };
  cinn_buffer_free((void*)(0), _B);
}
)ROC";

  ASSERT_EQ(Trim(target_source), Trim(source));
}

TEST(Tensor, ReshapeCopied) {
  Context::Global().ResetNameId();
  Expr M(100);
  Expr N(100);
  Placeholder<float> A("A", {M, N});

  auto stages = CreateStages({A});

  auto A1 = A->ReshapeCopied({Expr(10), Expr(10), Expr(100)}, stages);
  auto B = Compute(
      A1->shape,
      [=](Expr i, Expr j, Expr k) { return A1(i, j, k) * 2.f; },
      "B");

  stages->InsertLazily(B);

  ir::Module::Builder builder("some_modue", common::DefaultHostTarget());
  auto func = lang::Lower("fn", stages, {A, B}, {}, {}, &builder);

  backends::CodeGenC codegenc(common::DefaultHostTarget());
  codegenc.SetInlineBuiltinCodes(false);
  auto source = codegenc.Compile(builder.Build(), CodeGenC::OutputKind::CImpl);
  LOG(INFO) << "source:\n" << source;

  auto target_source = R"ROC(
#include <cinn_runtime.h>
#include <stdio.h>

void fn(void* _args, int32_t num_args)
{
  const cinn_buffer_t* _A = cinn_pod_value_to_buffer_p(&(((cinn_pod_value_t*)(_args))[0]));
  cinn_buffer_t* _B = cinn_pod_value_to_buffer_p(&(((cinn_pod_value_t*)(_args))[1]));
  cinn_buffer_t* _A_copied_reshape = cinn_buffer_t::new_((cinn_device_kind_t)(0)/*target*/, cinn_float32_t(), { 10, 10, 100 }, 32/*align*/);
  cinn_buffer_malloc((void*)(0), _B);
  cinn_buffer_malloc((void*)(0), _A_copied_reshape);
  const float* A = ((const float*)(_A->memory));
  float* A_copied = ((float*)(_A_copied_reshape->memory));
  const float* A_copied_reshape = ((const float*)(_A_copied_reshape->memory));
  float* B = ((float*)(_B->memory));
  for (int32_t i = 0; i < 100; i += 1) {
    for (int32_t j = 0; j < 100; j += 1) {
      A_copied[((100 * i) + j)] = A[((100 * i) + j)];
    };
  };
  for (int32_t i = 0; i < 10; i += 1) {
    for (int32_t j = 0; j < 10; j += 1) {
      for (int32_t k = 0; k < 100; k += 1) {
        B[((1000 * i) + ((100 * j) + k))] = (2.00000000f * A_copied_reshape[((1000 * i) + ((100 * j) + k))]);
      };
    };
  };
  cinn_buffer_free((void*)(0), _A_copied_reshape);
  cinn_buffer_free((void*)(0), _B);
}
)ROC";

  ASSERT_EQ(Trim(target_source), Trim(source));
}

TEST(Tensor, reduce) {
  Placeholder<float> A("A", {Expr(10)});
  Var reduce_axis(Expr(10), "reduce_k");
  {
    auto C = Compute(
        A->shape,
        [=](const std::vector<Expr>& axis) {
          return lang::ReduceSum(A(reduce_axis) + 1.f, {reduce_axis});
        },
        "C");
    ASSERT_TRUE(C->has_expression());
    ASSERT_TRUE(C->is_reduce_sum());
    ASSERT_FALSE(C->is_reduce_mul());
  }

  {
    auto C = Compute(
        A->shape,
        [=](const std::vector<Expr>& axis) {
          return lang::ReduceMul(A(reduce_axis) + 1.f, {reduce_axis});
        },
        "C");
    ASSERT_TRUE(C->has_expression());
    ASSERT_TRUE(C->is_reduce_mul());
    ASSERT_FALSE(C->is_reduce_sum());
  }
}

}  // namespace ir
}  // namespace cinn
