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

#include <gtest/gtest.h>

#include "paddle/cinn/cinn.h"
#include "paddle/cinn/common/ir_util.h"
#include "paddle/cinn/hlir/pe/broadcast.h"
#include "paddle/cinn/optim/optimize.h"
namespace cinn {

TEST(test01_elementwise_add, basic) {
  Expr M(100), N(32);

  Placeholder<float> A("A", {M, N});
  Placeholder<float> B("B", {M, N});

  Buffer C_buf(Float(32));
  auto C = hlir::pe::Add(A.tensor(), B.tensor(), "C");
  C->Bind(C_buf);

  Target target;
  target.arch = Target::Arch ::X86;
  target.bits = Target::Bit ::k32;
  target.os = Target::OS ::Linux;
  Module::Builder builder("module1", target);

  auto stages = CreateStages({A, B, C});
  auto func = Lower("add1", stages, {A, B, C});

  builder.AddFunction(func);

  CodeGenC compiler(target);
  Outputs outputs;
  outputs = outputs.c_header("./test01_elementwise_add.h")
                .c_source("./test01_elementwise_add.cc");
  compiler.Compile(builder.Build(), outputs);
}

TEST(test01_elementwise_add, vectorize) {
  Expr M(100);
  Expr N(32);

  Placeholder<float> A("A", {M, N});
  Placeholder<float> B("B", {M, N});

  auto C = hlir::pe::Add(A.tensor(), B.tensor(), "C");

  auto stages = CreateStages({C});
  stages[C]->Vectorize(1, 8);

  Target target;
  target.arch = Target::Arch ::X86;
  target.bits = Target::Bit ::k32;
  target.os = Target::OS ::Linux;
  Module::Builder builder("module2", target);

  auto func = Lower("add1_vectorize", stages, {A, B, C});

  LOG(INFO) << "after optim:\n" << func;
  builder.AddFunction(ir::LoweredFunc(func.As<ir::_LoweredFunc_>()));
  // module.Append(C_buf);

  CodeGenCX86 compiler(target, CodeGenCX86::Feature::AVX256);
  Outputs outputs;
  outputs = outputs.c_header("./test01_elementwise_add_vectorize.h")
                .c_source("./test01_elementwise_add_vectorize.cc");
  compiler.Compile(builder.Build(), outputs);
}

auto BuildComputeAtExpr() {
  Expr M(100);
  Expr N(32);

  Placeholder<float> A("A", {M, N});
  Placeholder<float> B("B", {M, N});

  auto A_cache = Compute(
      {M, N},
      [=](Expr i, Expr j) {
        auto first = cinn::common::select(
            i > 0, A(i - 1, j), common::make_const(Float(32), 0.f));
        auto last = cinn::common::select(
            i < M - 1, A(i + 1, j), common::make_const(Float(32), 0.f));
        return first + A(i, j) + last;
      },
      "A_cache");
  auto C = Compute(
      {M, N}, [&](Var i, Var j) { return A_cache(i, j) + B(i, j); }, "C");

  return std::make_tuple(A, B, A_cache, C);
}

TEST(elementwise_add, compute_at) {
  auto _A_B_A_cache_C_ = BuildComputeAtExpr();
  auto &A = std::get<0>(_A_B_A_cache_C_);
  auto &B = std::get<1>(_A_B_A_cache_C_);
  auto &A_cache = std::get<2>(_A_B_A_cache_C_);
  auto &C = std::get<3>(_A_B_A_cache_C_);

  auto stages = CreateStages({A, B, A_cache, C});
  stages[A_cache]->ComputeAt2(stages[C], 0);
  stages[C]->Parallel(0);

  Module::Builder builder("module3", common::DefaultHostTarget());

  auto fn = Lower("fn_compute_at", stages, {A, B, C}, {}, {A_cache}, &builder);

  CodeGenCX86 compiler(common::DefaultHostTarget(),
                       CodeGenCX86::Feature::AVX256);
  Outputs outputs;
  outputs = outputs.c_header("./test01_elementwise_add_compute_at.h")
                .c_source("./test01_elementwise_add_compute_at.cc");
  compiler.Compile(builder.Build(), outputs);
}

TEST(elementwise_add, compute_at1) {
  auto _A_B_A_cache_C_ = BuildComputeAtExpr();
  auto &A = std::get<0>(_A_B_A_cache_C_);
  auto &B = std::get<1>(_A_B_A_cache_C_);
  auto &A_cache = std::get<2>(_A_B_A_cache_C_);
  auto &C = std::get<3>(_A_B_A_cache_C_);

  auto stages = CreateStages({A, B, A_cache, C});
  stages[A_cache]->ComputeAt2(stages[C], 1);
  stages[C]->Parallel(0);

  Module::Builder builder("module4", common::DefaultHostTarget());

  auto fn =
      Lower("fn_compute_at_level1", stages, {A, B, C}, {}, {A_cache}, &builder);

  CodeGenCX86 compiler(common::DefaultHostTarget(),
                       CodeGenCX86::Feature::AVX256);
  Outputs outputs;
  outputs = outputs.c_header("./test01_elementwise_add_compute_at_level1.h")
                .c_source("./test01_elementwise_add_compute_at_level1.cc");
  compiler.Compile(builder.Build(), outputs);
}

}  // namespace cinn
