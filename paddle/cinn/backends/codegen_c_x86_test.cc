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

#include "paddle/cinn/backends/codegen_c_x86.h"

#include <gtest/gtest.h>

#include "paddle/cinn/cinn.h"
#include "paddle/cinn/ir/module.h"
#include "paddle/cinn/lang/builtin.h"
#include "paddle/cinn/lang/compute.h"
#include "paddle/cinn/lang/lower.h"
#include "paddle/cinn/lang/placeholder.h"
#include "paddle/cinn/optim/ir_simplify.h"
#include "paddle/cinn/optim/transform_polyfor_to_for.h"
#include "paddle/cinn/optim/vectorize_loops.h"

namespace cinn {
namespace backends {

TEST(CodeGenCX86, basic) {
  // create two forloops, check only one forloop is marked Vectorize.
  Context::info_rgt().Clear();

  using namespace ir;  // NOLINT

  const int M = 100;
  const int K = 200;
  const int N = 500;
  const int bn = 32;

  Target target;
  target.arch = Target::Arch ::X86;
  target.bits = Target::Bit ::k32;
  target.os = Target::OS ::Linux;

  Placeholder<float> A("A", {M, N});
  Placeholder<float> B("B", {M, N});

  // C = A * B
  Tensor C = Compute(
      {Expr(M), Expr(N)}, [&](Var i, Var j) { return A(i, j) * B(i, j); }, "C");

  Tensor D = Compute(
      {Expr(M), Expr(N)}, [&](Var i, Var j) { return A(i, j) * B(i, j); }, "D");

  auto stages = CreateStages({C, D});
  // vectorize C, not D
  stages[C]->Vectorize(1, 16);
  stages[C]->Unroll(1);

  auto func = Lower("matmul", stages, {A, B, C, D});

  std::cout << "before optim\n" << func->body << std::endl;

  ir::Module::Builder builder("module1", target);
  builder.AddFunction(func);

  CodeGenCX86 codegen(target, CodeGenCX86::Feature::AVX512);
  codegen.SetInlineBuiltinCodes(false);
  auto out = codegen.Compile(builder.Build(), CodeGenC::OutputKind::CImpl);
  std::cout << "out:\n" << out;
}

}  // namespace backends
}  // namespace cinn
