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
#include "paddle/cinn/optim/optimize.h"
#include "test/cpp/cinn/test02_helper.h"

namespace cinn {
using poly::Iterator;

Expr M(1024);
Expr N(1024);
Expr K(1024);

TEST(test02_matmul, basic) {
  Placeholder<float> A("A", {M, K});
  Placeholder<float> B("B", {K, N});

  Var k(K.as_int32(), "k0");

  auto C = Compute(
      {M, N},
      [&](Var i, Var j) { return ReduceSum(A(i, k) * B(k, j), {k}); },
      "C");

  Target target = common::DefaultHostTarget();

  {
    auto stages = CreateStages({C});
    Module::Builder builder("module1", target);
    auto func = Lower("matmul", stages, {A, B, C});

    builder.AddFunction(func);

    CodeGenC compiler(target);
    Outputs outputs;
    outputs =
        outputs.c_header("./test02_matmul.h").c_source("./test02_matmul.cc");
    compiler.Compile(builder.Build(), outputs);
  }

  // Tile
  {
    auto stages = CreateStages({C});
    stages[C]->Tile(0, 1, 4, 4);

    Module::Builder builder("module2", target);
    auto func = Lower("matmul_tile", stages, {A, B, C});

    builder.AddFunction(func);

    CodeGenC compiler(target);
    Outputs outputs;
    outputs = outputs.c_header("./test02_matmul_tile.h")
                  .c_source("./test02_matmul_tile.cc");
    compiler.Compile(builder.Build(), outputs);
  }
}

TEST(matmul, Split) {
  Placeholder<float> A("A", {M, K});
  Placeholder<float> B("B", {K, N});

  Var k(K.as_int32(), "k0");

  auto C = Compute(
      {M, N},
      [&](Var i, Var j) { return ReduceSum(A(i, k) * B(k, j), {k}); },
      "C");

  auto stages = CreateStages({C});

  Target target = common::DefaultHostTarget();

  auto _i0_i1_ = stages[C]->Split(2, 16);
  auto &i0 = std::get<0>(_i0_i1_);
  auto &i1 = std::get<1>(_i0_i1_);
  std::vector<Iterator> iterators({stages[C]->ith_iterator(1),
                                   stages[C]->ith_iterator(0),
                                   stages[C]->ith_iterator(2),
                                   stages[C]->ith_iterator(3)});
  stages[C]->Reorder(iterators);

  Module::Builder builder("module3", target);
  auto func = Lower("matmul_split", stages, {A, B, C});

  builder.AddFunction(func);

  CodeGenCX86 compiler(target, CodeGenCX86::Feature::AVX512);
  Outputs outputs;
  outputs = outputs.c_header("./test02_matmul_split.h")
                .c_source("./test02_matmul_split.cc");
  compiler.Compile(builder.Build(), outputs);
}

TEST(matmul, Blocking) {
  Placeholder<float> A("A", {M, K});
  Placeholder<float> B("B", {K, N});

  Var k(K.as_int32(), "k0");

  int bn = 32;

  auto C = Compute(
      {M, N},
      [&](Var i, Var j) { return ReduceSum(A(i, k) * B(k, j), {k}); },
      "C");

  auto stages = CreateStages({C});

  Target target = common::DefaultHostTarget();

  // Blocking by loop tiling.
  {
    auto _i_outer_i_inner_j_outer_j_inner_ = stages[C]->Tile(0, 1, bn, bn);
    auto &i_outer = std::get<0>(_i_outer_i_inner_j_outer_j_inner_);
    auto &i_inner = std::get<1>(_i_outer_i_inner_j_outer_j_inner_);
    auto &j_outer = std::get<2>(_i_outer_i_inner_j_outer_j_inner_);
    auto &j_inner = std::get<3>(_i_outer_i_inner_j_outer_j_inner_);
    auto _k_outer_k_inner_ = stages[C]->Split("k0", 4);
    auto &k_outer = std::get<0>(_k_outer_k_inner_);
    auto &k_inner = std::get<1>(_k_outer_k_inner_);
    stages[C]->Reorder({i_outer, j_outer, k_outer, k_inner, i_inner, j_inner});
  }

  Module::Builder builder("module_block", target);
  auto func = Lower("matmul_block", stages, {A, B, C});

  builder.AddFunction(func);

  CodeGenCX86 compiler(target, CodeGenCX86::Feature::AVX512);
  Outputs outputs;
  outputs = outputs.c_header("./test02_matmul_block.h")
                .c_source("./test02_matmul_block.cc");
  compiler.Compile(builder.Build(), outputs);
}

TEST(matmul, Vectorization) {
  Placeholder<float> A("A", {M, K});
  Placeholder<float> B("B", {K, N});

  Var k(K.as_int32(), "k0");

  int bn = 32;

  auto C = Compute(
      {M, N},
      [&](Var i, Var j) { return ReduceSum(A(i, k) * B(k, j), {k}); },
      "C");

  auto stages = CreateStages({C});

  Target target = common::DefaultHostTarget();

  // Blocking by loop tiling.
  {
    auto _i_outer_i_inner_j_outer_j_inner_ = stages[C]->Tile(0, 1, bn, bn);
    auto &i_outer = std::get<0>(_i_outer_i_inner_j_outer_j_inner_);
    auto &i_inner = std::get<1>(_i_outer_i_inner_j_outer_j_inner_);
    auto &j_outer = std::get<2>(_i_outer_i_inner_j_outer_j_inner_);
    auto &j_inner = std::get<3>(_i_outer_i_inner_j_outer_j_inner_);
    auto _k_outer_k_inner_ = stages[C]->Split("k0", 4);
    auto &k_outer = std::get<0>(_k_outer_k_inner_);
    auto &k_inner = std::get<1>(_k_outer_k_inner_);
    stages[C]->Reorder({i_outer, j_outer, k_outer, k_inner, i_inner, j_inner});
    stages[C]->Vectorize(j_inner, 8);
  }

  Module::Builder builder("module_vectorize", target);
  auto func = Lower("matmul_vectorize", stages, {A, B, C});

  builder.AddFunction(func);

  CodeGenCX86 compiler(target, CodeGenCX86::Feature::AVX256);
  Outputs outputs;
  outputs = outputs.c_header("./test02_matmul_vectorize.h")
                .c_source("./test02_matmul_vectorize.cc");
  compiler.Compile(builder.Build(), outputs);
}

TEST(matmul, LoopPermutation) {
  auto module = tests::CreateMatmulLoopPermutation(
      common::DefaultHostTarget(), 1024, 1024, 1024);

  CodeGenCX86 compiler(common::DefaultHostTarget(),
                       CodeGenCX86::Feature::AVX256);
  Outputs outputs;
  outputs = outputs.c_header("./test02_matmul_loop_permutation.h")
                .c_source("./test02_matmul_loop_permutation.cc");
  compiler.Compile(module, outputs);
}

TEST(matmul, ArrayPacking) {
  auto target = common::DefaultHostTarget();

  auto module = tests::CreateMatmulArrayPacking(target, 1024, 1024, 1024);

  CodeGenCX86 compiler(target, CodeGenCX86::Feature::AVX256);
  Outputs outputs;
  outputs = outputs.c_header("./test02_matmul_array_packing.h")
                .c_source("./test02_matmul_array_packing.cc");
  compiler.Compile(module, outputs);
}

TEST(matmul, varient_shape) {
  Var M("M");  // M is a symbol.
  Placeholder<float> A("A", {M, K});
  Placeholder<float> B("B", {K, N});

  Var k(K.as_int32(), "k0");

  auto C = Compute(
      {M, N},
      [&](Var i, Var j) { return ReduceSum(A(i, k) * B(k, j), {k}); },
      "C");

  Target target = common::DefaultHostTarget();

  {
    auto stages = CreateStages({C});
    Module::Builder builder("matmul_dynamic_shape", target);
    auto func = Lower("matmul_dynamic_shape", stages, {A, B, C}, {M});

    builder.AddFunction(func);

    CodeGenC compiler(target);
    Outputs outputs;
    outputs = outputs.c_header("./test02_matmul_varient_shape.h")
                  .c_source("./test02_matmul_varient_shape.cc");
    compiler.Compile(builder.Build(), outputs);
  }

  {
    auto stages = CreateStages({C});
    int bn = 32;
    auto _i_outer_i_inner_j_outer_j_inner_ =
        stages[C]->Tile(0, 1, bn, bn);  // NOLINT
    auto &i_outer = std::get<0>(_i_outer_i_inner_j_outer_j_inner_);
    auto &i_inner = std::get<1>(_i_outer_i_inner_j_outer_j_inner_);
    auto &j_outer = std::get<2>(_i_outer_i_inner_j_outer_j_inner_);
    auto &j_inner = std::get<3>(_i_outer_i_inner_j_outer_j_inner_);

    Module::Builder builder("matmul_dynamic_shape_tile", target);
    auto func = Lower("matmul_dynamic_shape_tile",
                      stages,
                      {A, B, C} /*tensors*/,
                      {M} /*scalars*/);
    LOG(INFO) << "func " << Expr(func);

    builder.AddFunction(func);

    CodeGenC compiler(target);
    Outputs outputs;

    outputs = outputs.c_header("./test02_matmul_varient_shape_tile.h")
                  .c_source("./test02_matmul_varient_shape_tile.cc");
    compiler.Compile(builder.Build(), outputs);
  }
}

TEST(matmul, ArrayPacking_dynamic_shape) {
  Var M("M");
  Placeholder<float> A("A", {M, K});
  Placeholder<float> B("B", {K, N});

  Var k(K.as_int32(), "k0");

  Expr bn(32);

  auto packedB = Compute(
      {N / bn, K, bn},
      [&](Expr x, Expr y, Expr z) { return B(y, x * bn + z); },
      "packedB");

  auto C = Compute(
      {M, N},
      [&](Expr i, Expr j) {
        return ReduceSum(A(i, k) * packedB(j / bn, k, j % bn), {k});
      },
      "C");

  auto stages = CreateStages({C});

  stages[packedB]->Vectorize(2, 8);

  Target target;
  target.arch = Target::Arch::X86;
  target.bits = Target::Bit::k32;
  target.os = Target::OS::Linux;

  {
    auto _i_outer_i_inner_j_outer_j_inner_ =
        stages[C]->Tile(0, 1, bn.as_int32(), bn.as_int32());
    auto &i_outer = std::get<0>(_i_outer_i_inner_j_outer_j_inner_);
    auto &i_inner = std::get<1>(_i_outer_i_inner_j_outer_j_inner_);
    auto &j_outer = std::get<2>(_i_outer_i_inner_j_outer_j_inner_);
    auto &j_inner = std::get<3>(_i_outer_i_inner_j_outer_j_inner_);
    auto _k_outer_k_inner_ = stages[C]->Split("k0", 4);
    auto &k_outer = std::get<0>(_k_outer_k_inner_);
    auto &k_inner = std::get<1>(_k_outer_k_inner_);

    stages[C]->Reorder({i_outer, j_outer, k_outer, i_inner, k_inner, j_inner});
    stages[C]->Vectorize(j_inner, 8);
  }

  Module::Builder builder("module_array_packing_dynamic_shape", target);
  auto func = Lower("matmul_array_packing_dynamic_shape",
                    stages,
                    {A, B, C},
                    {M},
                    {packedB},
                    &builder);

  CodeGenCX86 compiler(target, CodeGenCX86::Feature::AVX256);
  Outputs outputs;
  outputs = outputs.c_header("./test02_matmul_array_packing_dynamic_shape.h")
                .c_source("./test02_matmul_array_packing_dynamic_shape.cc");
  compiler.Compile(builder.Build(), outputs);
}

TEST(matmul, call) {
  Placeholder<float> A("A", {M, K});
  Placeholder<float> B("B", {K, N});

  Var k(K.as_int32(), "k0");
  Buffer C_buf(Float(32));

  auto C = Compute(
      {M, N},
      [&](Var i, Var j) { return ReduceSum(A(i, k) * B(k, j), {k}); },
      "C");

  Target target = common::DefaultHostTarget();

  auto stages = CreateStages({C});

  Module::Builder builder("module_call", target);
  {
    auto func = Lower("matmul_kernel", stages, {A, B, C});

    builder.AddFunction(func);
  }

  {  // main
    std::vector<lang::ReturnType> returns(
        {lang::ReturnType{Float(32), C->shape, C->name}});
    auto tensors = lang::CallLowered("matmul_kernel", {A, B}, returns);
    auto C = tensors[0];

    LOG(INFO) << "stage domain: " << stages[C]->domain();
    auto fn = Lower("matmul_main", stages, {A, B, C}, {});
    builder.AddFunction(fn);
  }

  CodeGenC compiler(target);
  Outputs outputs;
  outputs = outputs.c_header("./test02_matmul_call.h")
                .c_source("./test02_matmul_call.cc");
  compiler.Compile(builder.Build(), outputs);
}

}  // namespace cinn
