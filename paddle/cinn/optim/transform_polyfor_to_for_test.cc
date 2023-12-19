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

#include "paddle/cinn/optim/transform_polyfor_to_for.h"

#include <gtest/gtest.h>

#include "paddle/cinn/cinn.h"

namespace cinn {
namespace optim {

TEST(Expr, basic) {
  using namespace ir;  // NOLINT

  Expr M(512);
  Expr K(200);
  Expr N(500);
  Placeholder<float> A("A", {M, K});
  Placeholder<float> B("B", {K, N});

  // C = A * B
  Var k(K.as_int32(), "k0");

  Tensor C = Compute(
      {M, N},
      [&](Var i, Var j) { return lang::ReduceSum(A(i, k) * B(k, j), {k}); },
      "C");

  auto stages = CreateStages({C});

  {
    stages[C]->Split("i", 8);
    stages[C]->Split("j", 8);
  }

  // Code gen
  auto func = Lower("matmul", stages, {A, B, C});

  Target target;
  target.arch = Target::Arch ::X86;
  target.bits = Target::Bit ::k32;
  target.os = Target::OS ::Linux;

  {
    ir::Module::Builder builder("module1", target);
    builder.AddFunction(func);

    CodeGenC codegen(target);
    codegen.SetInlineBuiltinCodes(false);
    auto out = codegen.Compile(builder.Build(), CodeGenC::OutputKind::CImpl);
    std::cout << "out:\n" << out;
  }

  optim::TransformPolyForToFor(&func->body);

  {
    ir::Module::Builder builder("module1", target);
    builder.AddFunction(func);

    CodeGenC codegen(target);
    codegen.SetInlineBuiltinCodes(false);
    auto out = codegen.Compile(builder.Build(), CodeGenC::OutputKind::CImpl);
    std::cout << "out:\n" << out;

    auto target_out = R"ROC(
#include <cinn_runtime.h>
#include <stdio.h>

void matmul(void* _args, int32_t num_args)
{
  const cinn_buffer_t* _A = cinn_pod_value_to_buffer_p(&(((cinn_pod_value_t*)(_args))[0]));
  const cinn_buffer_t* _B = cinn_pod_value_to_buffer_p(&(((cinn_pod_value_t*)(_args))[1]));
  cinn_buffer_t* _C = cinn_pod_value_to_buffer_p(&(((cinn_pod_value_t*)(_args))[2]));
  cinn_buffer_malloc((void*)(0), _C);
  const float* A = ((const float*)(_A->memory));
  const float* B = ((const float*)(_B->memory));
  float* C = ((float*)(_C->memory));
  float* C__reduce_init = ((float*)(_C->memory));
  for (int32_t i_outer = 0; i_outer < 64; i_outer += 1) {
    for (int32_t i_inner = 0; i_inner < 8; i_inner += 1) {
      for (int32_t j_outer = 0; j_outer < 63; j_outer += 1) {
        for (int32_t j_inner = 0; j_inner < cinn_min(8, (500 + (-8 * j_outer))); j_inner += 1) {
          C__reduce_init[((500 * i_inner) + ((4000 * i_outer) + ((8 * j_outer) + j_inner)))] = 0.00000000f;
          for (int32_t k0 = 0; k0 < 200; k0 += 1) {
            C[((500 * i_inner) + ((4000 * i_outer) + ((8 * j_outer) + j_inner)))] = fma(A[((200 * i_inner) + ((1600 * i_outer) + k0))], B[((8 * j_outer) + ((500 * k0) + j_inner))], C[((500 * i_inner) + ((4000 * i_outer) + ((8 * j_outer) + j_inner)))]);
          };
        };
      };
    };
  };
  cinn_buffer_free((void*)(0), _C);
}
)ROC";
    EXPECT_EQ(utils::Trim(target_out), utils::Trim(out));
  }
}

}  // namespace optim
}  // namespace cinn
