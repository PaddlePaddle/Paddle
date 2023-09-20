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

#include "paddle/cinn/optim/vectorize_loops.h"

#include <gtest/gtest.h>

#include <vector>

#include "paddle/cinn/cinn.h"
#include "paddle/cinn/common/common.h"
#include "paddle/cinn/common/ir_util.h"
#include "paddle/cinn/ir/op/ir_operators.h"
#include "paddle/cinn/optim/ir_simplify.h"
#include "paddle/cinn/optim/optimize.h"
#include "paddle/cinn/optim/transform_polyfor_to_for.h"
#include "paddle/cinn/utils/string.h"

namespace cinn {
namespace optim {
using namespace ir;  // NOLINT
using utils::GetStreamCnt;
using utils::Trim;

TEST(Vectorize, replace_var) {
  using namespace ir;  // NOLINT

  Expr M(100);
  Expr N(500);
  Placeholder<float> A("A", {M, N});
  Placeholder<float> B("B", {M, N});

  // C = A * B

  Tensor C = Compute(
      {M, N}, [&](Var i, Var j) { return A(i, j) * B(i, j); }, "C");

  auto stages = CreateStages({C});

  stages[C]->Vectorize(1, 16);

  auto funcs = Lower("matmul", stages, {A, B, C});

  Expr func = optim::Optimize(funcs, common::DefaultHostTarget());

  Target target;
  target.arch = Target::Arch ::X86;
  target.bits = Target::Bit ::k32;
  target.os = Target::OS ::Linux;

  ir::Module::Builder builder("module1", target);
  builder.AddFunction(ir::LoweredFunc(func.As<ir::_LoweredFunc_>()));

  CodeGenC codegen(target);
  codegen.SetInlineBuiltinCodes(false);
  auto out = codegen.Compile(builder.Build(), CodeGenC::OutputKind::CImpl);
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
  for (int32_t i = 0; i < 100; i += 1) {
    for (int32_t j = 0; j < 32; j += 1) {
      C[StackVec<16,int32_t>::Ramp(((500 * i) + (16 * j)), 1, 16)] = (StackedVec<float,16>::Load(A,((500 * i) + (16 * j))) * StackedVec<float,16>::Load(B,((500 * i) + (16 * j))));
    };
  };
  cinn_buffer_free((void*)(0), _C);
}
)ROC";
  EXPECT_EQ(Trim(target_out), Trim(out));
}

TEST(Vectorize, TestMarkVectorize) {
  // create two forloops, check only one forloop is marked Vectorize.
  Context::info_rgt().Clear();

  using namespace ir;  // NOLINT

  Expr M(100);
  Expr N(500);

  Target target;
  target.arch = Target::Arch ::X86;
  target.bits = Target::Bit ::k32;
  target.os = Target::OS ::Linux;

  Placeholder<float> A("A", {M, N});
  Placeholder<float> B("B", {M, N});

  // C = A * B

  Tensor C = Compute(
      {M, N}, [&](Var i, Var j) { return A(i, j) * B(i, j); }, "C");

  Tensor D = Compute(
      {M, N}, [&](Var i, Var j) { return A(i, j) * B(i, j); }, "D");

  auto stages = CreateStages({C, D});

  stages[D]->ShareBufferWith(stages[C]);

  // vectorize C, not D
  stages[C]->Vectorize(1, 16);

  auto func = Lower("matmul", stages, {A, B, C, D});

  std::cout << "before optim\n" << func->body << std::endl;

  optim::TransformPolyForToFor(&func->body);
  optim::VectorizeLoops(&func->body, target);
  optim::Simplify(&func->body);

  ir::Module::Builder builder("module1", target);
  builder.AddFunction(func);

  CodeGenC codegen(target);
  codegen.SetInlineBuiltinCodes(false);
  auto out = codegen.Compile(builder.Build(), CodeGenC::OutputKind::CImpl);
  std::cout << "out:\n" << out;

  auto target_out = R"ROC(
#include <cinn_runtime.h>
#include <stdio.h>

void matmul(const struct cinn_buffer_t *_A, const struct cinn_buffer_t *_B, struct cinn_buffer_t *_C)
{
  cinn_buffer_malloc((void*)(0), _C);
  const float* A = (const float*)(_A->memory);
  const float* B = (const float*)(_B->memory);
  float* C = (float*)(_C->memory);
  float* D = (float*)(_C->memory);
  for (int32_t i = 0; i < 100; i += 1) {
    for (int32_t j_outer = 0; j_outer < 31; j_outer += 1) {
      C[StackVec<16,int32_t>::Ramp(((500 * i) + (16 * j_outer)), 1, 16)] = (StackedVec<float,16>::Load(A,((500 * i) +
      (16 * j_outer))) * StackedVec<float,16>::Load(B,((500 * i) + (16 * j_outer))));
    };
    for (int32_t j_outer = 31; j_outer < 32; j_outer += 1) {
      for (int32_t j_inner = 0; j_inner < (500 + (-16 * j_outer)); j_inner += 1) {
        C[((500 * i) + ((16 * j_outer) + j_inner))] = (A[((500 * i) + ((16 * j_outer) + j_inner))] * B[((500 * i) +
        ((16 * j_outer) + j_inner))]);
      };
    };
  };
  for (int32_t i = 0; i < 100; i += 1) {
    for (int32_t j = 0; j < 500; j += 1) {
      D[((500 * i) + j)] = (A[((500 * i) + j)] * B[((500 * i) + j)]);
    };
  };
}
)ROC";

  EXPECT_EQ(Context::info_rgt().Get<int>("vectorized_forloop_count"), 1);
}

TEST(Vectorize, vectorize) {
  Var a("a");
  Var b("b");
  Var c("c");

  {
    Expr d = a * 10 + b;
    detail::Vectorize(a, 16, &d);
    Simplify(&d);
    EXPECT_EQ(GetStreamCnt(d), "Ramp(b,10,16)");
  }

  {
    Expr d = a * 10 + b;
    detail::Vectorize(b, 16, &d);
    Simplify(&d);
    EXPECT_EQ(GetStreamCnt(d), "Ramp((10 * a),1,16)");
  }

  {
    Placeholder<float> A("A", std::vector<int>{{10}});
    Placeholder<float> B("B", std::vector<int>{{10}});
    Placeholder<float> C("C", std::vector<int>{{10}});

    auto expr = Load::Make(ir::Tensor(A), {a * 2 + b * 2});
    expr = expr + 10.f * expr;
    detail::Vectorize(a, 16, &expr);
    EXPECT_EQ(GetStreamCnt(expr),
              "(A[Ramp(((b * 2) + (0 * 2)),(1 * 2),16)] + "
              "(Broadcast(10.0000000f,16) * A[Ramp(((b * 2) + (0 * 2)),(1 * "
              "2),16)]))");
  }
}

TEST(Vectorize, single_for) {
  Placeholder<float> A("A", std::vector<int>{{10}});
  Placeholder<float> B("B", std::vector<int>{{10}});
  Placeholder<float> C("C", std::vector<int>{{10}});

  Var loop_var("k0");

  Expr body = Store::Make(ir::Tensor(C),
                          ir::Add::Make(  //
                              ir::Load::Make(ir::Tensor(A), {Expr(loop_var)}),
                              ir::Load::Make(ir::Tensor(B), {Expr(loop_var)})),
                          {Expr(loop_var)});
  body = ir::Block::Make({body});

  VectorizeInfo vectorize_info(0, 16);
  auto forloop = ir::For::Make(loop_var,
                               common::make_const(0),
                               common::make_const(16),
                               ir::ForType::Vectorized,
                               ir::DeviceAPI::UNK,
                               body,
                               vectorize_info);

  forloop = optim::Optimize(forloop, common::DefaultHostTarget());

  LOG(INFO) << "Forloop\n" << forloop;
}

TEST(Vectorize, cuda_vectorize) {
  Expr M(100);
  Expr N(500);
  Placeholder<float> A("A", {M, N});
  Placeholder<float> B("B", {M, N});

  Tensor C = Compute(
      {M, N}, [&](Var i, Var j) { return A(i, j) * B(i, j); }, "C");

  auto stages = CreateStages({C});
  stages[C]->Vectorize(1, 4);
  Target target = common::DefaultNVGPUTarget();
  auto func = Lower("matmul", stages, {A, B, C}, {}, {}, nullptr, target);

  auto target_expr = R"ROC(
function matmul (_A, _B, _C)
{
  serial for (i, 0, 100)
  {
    serial for (j, 0, 125)
    {
      CudaVectorType::float4<4>* vectorized_C_ptr = CudaVectorType::float4<4>*(get_addr(C[i, ((j * 4) + 0)]))
      CudaVectorType::float4<4> vectorized_C = 0
      const CudaVectorType::float4<4> vectorized_A = const CudaVectorType::float4<4>*(get_addr(A[i, ((j * 4) + 0)]))[0]
      const CudaVectorType::float4<4> vectorized_B = const CudaVectorType::float4<4>*(get_addr(B[i, ((j * 4) + 0)]))[0]
      vectorized_C[0] = (vectorized_A[0] * vectorized_B[0])
      vectorized_C[1] = (vectorized_A[1] * vectorized_B[1])
      vectorized_C[2] = (vectorized_A[2] * vectorized_B[2])
      vectorized_C[3] = (vectorized_A[3] * vectorized_B[3])
      vectorized_C_ptr[0] = vectorized_C
    }
  }
}
)ROC";
  ASSERT_EQ(Trim(target_expr), Trim(GetStreamCnt(func)));
}

TEST(Vectorize, cuda_vectorize_with_constant) {
  Expr M(100);
  Expr N(500);
  Placeholder<float> A("A", {M, N});
  Expr const_value(2.11f);

  Tensor C = Compute(
      {M, N}, [&](Var i, Var j) { return const_value * A(i, j); }, "C");

  auto stages = CreateStages({C});
  stages[C]->Vectorize(1, 4);
  Target target = common::DefaultNVGPUTarget();
  auto func = Lower("mul_const", stages, {A, C}, {}, {}, nullptr, target);
}

}  // namespace optim
}  // namespace cinn
