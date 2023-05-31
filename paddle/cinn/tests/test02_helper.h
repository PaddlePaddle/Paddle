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

#pragma once

#include <absl/strings/string_view.h>
#include <gtest/gtest.h>

#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "paddle/cinn/backends/llvm/execution_engine.h"
#include "paddle/cinn/backends/llvm/simple_jit.h"
#include "paddle/cinn/cinn.h"
#include "paddle/cinn/optim/optimize.h"

namespace cinn {
namespace tests {

auto CreateMatmulBasicModule(Target target, int m, int n, int k) {
  auto _M_N_K_ = std::make_tuple(Expr(m), Expr(n), Expr(k));
  auto &M = std::get<0>(_M_N_K_);
  auto &N = std::get<1>(_M_N_K_);
  auto &K = std::get<2>(_M_N_K_);

  auto A = Placeholder<float>("A", {M, K});
  auto B = Placeholder<float>("B", {K, N});

  auto k1 = Var(K.as_int32(), "k1");
  auto C = Compute(
      {M, N},
      [&](Var i, Var j) { return ReduceSum(A(i, k1) * B(k1, j), {k1}); },
      "C");

  auto stages = CreateStages({C});

  Module::Builder builder("module_basic", target);

  auto func = Lower("matmul_basic", stages, {A, B, C});

  builder.AddFunction(func);
  return builder.Build();
}

auto CreateMatmulTileModule(Target target, int m, int n, int k) {
  auto _M_N_K_ = std::make_tuple(Expr(m), Expr(n), Expr(k));
  auto &M = std::get<0>(_M_N_K_);
  auto &N = std::get<1>(_M_N_K_);
  auto &K = std::get<2>(_M_N_K_);

  auto A = Placeholder<float>("A", {M, K});
  auto B = Placeholder<float>("B", {K, N});

  auto k1 = Var(K.as_int32(), "k1");
  auto C = Compute(
      {M, N},
      [&](Var i, Var j) { return ReduceSum(A(i, k1) * B(k1, j), {k1}); },
      "C");

  auto stages = CreateStages({C});

  stages[C]->Tile(0, 1, 4, 4);

  Module::Builder builder("module_tile", target);

  auto func = Lower("matmul_tile", stages, {A, B, C});

  builder.AddFunction(func);
  return builder.Build();
}

auto CreateMatmulSplitModule(Target target, int m, int n, int k) {
  auto _M_N_K_ = std::make_tuple(Expr(m), Expr(n), Expr(k));
  auto &M = std::get<0>(_M_N_K_);
  auto &N = std::get<1>(_M_N_K_);
  auto &K = std::get<2>(_M_N_K_);

  auto A = Placeholder<float>("A", {M, K});
  auto B = Placeholder<float>("B", {K, N});

  auto k1 = Var(K.as_int32(), "k1");
  auto C = Compute(
      {M, N},
      [&](Var i, Var j) { return ReduceSum(A(i, k1) * B(k1, j), {k1}); },
      "C");

  auto stages = CreateStages({C});

  stages[C]->Split(2, 16);

  std::vector<poly::Iterator> polyIters;
  for (auto idx : {1, 0, 2, 3}) {
    polyIters.push_back(stages[C]->ith_iterator(idx));
  }
  stages[C]->Reorder(polyIters);

  Module::Builder builder("module_split", target);

  auto func = Lower("matmul_split", stages, {A, B, C});

  builder.AddFunction(func);
  return builder.Build();
}

auto CreateMatmulBlockModule(Target target, int m, int n, int k) {
  auto _M_N_K_ = std::make_tuple(Expr(m), Expr(n), Expr(k));
  auto &M = std::get<0>(_M_N_K_);
  auto &N = std::get<1>(_M_N_K_);
  auto &K = std::get<2>(_M_N_K_);

  auto A = Placeholder<float>("A", {M, K});
  auto B = Placeholder<float>("B", {K, N});

  auto k1 = Var(K.as_int32(), "k1");
  auto C = Compute(
      {M, N},
      [&](Var i, Var j) { return ReduceSum(A(i, k1) * B(k1, j), {k1}); },
      "C");

  auto stages = CreateStages({C});

  constexpr int bn = 32;
  auto _i_outer_i_inner_j_outer_j_inner_ =
      stages[C]->Tile(0, 1, bn, bn);  // NOLINT
  auto &i_outer = std::get<0>(_i_outer_i_inner_j_outer_j_inner_);
  auto &i_inner = std::get<1>(_i_outer_i_inner_j_outer_j_inner_);
  auto &j_outer = std::get<2>(_i_outer_i_inner_j_outer_j_inner_);
  auto &j_inner = std::get<3>(_i_outer_i_inner_j_outer_j_inner_);
  auto _k_outer_k_inner_ = stages[C]->Split(k1->name, 4);  // NOLINT
  auto &k_outer = std::get<0>(_k_outer_k_inner_);
  auto &k_inner = std::get<1>(_k_outer_k_inner_);
  stages[C]->Reorder({i_outer, j_outer, k_outer, k_inner, i_inner, j_inner});

  Module::Builder builder("module_block", target);

  auto func = Lower("matmul_block", stages, {A, B, C});

  builder.AddFunction(func);
  return builder.Build();
}

auto CreateMatmulVectorizeModule(Target target, int m, int n, int k) {
  auto _M_N_K_ = std::make_tuple(Expr(m), Expr(n), Expr(k));
  auto &M = std::get<0>(_M_N_K_);
  auto &N = std::get<1>(_M_N_K_);
  auto &K = std::get<2>(_M_N_K_);

  auto A = Placeholder<float>("A", {M, K});
  auto B = Placeholder<float>("B", {K, N});

  Var k0(K.as_int32(), "k0");

  int bn = 32;

  auto C = Compute(
      {M, N},
      [&](Var i, Var j) { return ReduceSum(A(i, k0) * B(k0, j), {k0}); },
      "C");

  auto stages = CreateStages({C});

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

  return builder.Build();
}

ir::Module CreateMatmulLoopPermutation(Target target, int m, int n, int k_) {
  target.arch = Target::Arch::X86;
  target.bits = Target::Bit::k32;
  target.os = Target::OS::Linux;

  auto _M_N_K_ = std::make_tuple(Expr(m), Expr(n), Expr(k_));
  auto &M = std::get<0>(_M_N_K_);
  auto &N = std::get<1>(_M_N_K_);
  auto &K = std::get<2>(_M_N_K_);

  Placeholder<float> A("A", {M, K});
  Placeholder<float> B("B", {K, N});

  Var k(K.as_int32(), "k0");

  int bn = 32;

  auto C = Compute(
      {M, N},
      [&](Var i, Var j) { return ReduceSum(A(i, k) * B(k, j), {k}); },
      "C");

  auto stages = CreateStages({C});

  // Blocking by loop tiling.
  {
    auto _i_outer_i_inner_j_outer_j_inner_ =
        stages[C]->Tile(0, 1, bn, bn);  // NOLINT
    auto &i_outer = std::get<0>(_i_outer_i_inner_j_outer_j_inner_);
    auto &i_inner = std::get<1>(_i_outer_i_inner_j_outer_j_inner_);
    auto &j_outer = std::get<2>(_i_outer_i_inner_j_outer_j_inner_);
    auto &j_inner = std::get<3>(_i_outer_i_inner_j_outer_j_inner_);
    auto _k_outer_k_inner_ = stages[C]->Split("k0", 4);  // NOLINT
    auto &k_outer = std::get<0>(_k_outer_k_inner_);
    auto &k_inner = std::get<1>(_k_outer_k_inner_);

    stages[C]->Reorder({i_outer, j_outer, k_outer, i_inner, k_inner, j_inner});

    stages[C]->Vectorize(j_inner, 8);
    stages[C]->Unroll(5);
  }

  Module::Builder builder("module_loop_permutation", target);
  auto func = Lower("matmul_loop_permutation", stages, {A, B, C});

  builder.AddFunction(func);
  return builder.Build();
}

ir::Module CreateMatmulArrayPacking(Target target, int m, int n, int k_) {
  auto _M_N_K_ = std::make_tuple(Expr(m), Expr(n), Expr(k_));
  auto &M = std::get<0>(_M_N_K_);
  auto &N = std::get<1>(_M_N_K_);
  auto &K = std::get<2>(_M_N_K_);

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

  {
    auto _i_outer_i_inner_j_outer_j_inner_ =
        stages[C]->Tile(0, 1, bn.as_int32(), bn.as_int32());  // NOLINT
    auto &i_outer = std::get<0>(_i_outer_i_inner_j_outer_j_inner_);
    auto &i_inner = std::get<1>(_i_outer_i_inner_j_outer_j_inner_);
    auto &j_outer = std::get<2>(_i_outer_i_inner_j_outer_j_inner_);
    auto &j_inner = std::get<3>(_i_outer_i_inner_j_outer_j_inner_);
    auto _k_outer_k_inner_ = stages[C]->Split("k0", 4);  // NOLINT
    auto &k_outer = std::get<0>(_k_outer_k_inner_);
    auto &k_inner = std::get<1>(_k_outer_k_inner_);

    stages[C]->Reorder({i_outer, j_outer, k_outer, i_inner, k_inner, j_inner});
    stages[C]->Vectorize(j_inner, 8);
  }

  Module::Builder builder("module_array_packing", target);
  auto func = Lower("matmul_array_packing", stages, {A, B, C, packedB});

  builder.AddFunction(func);

  return builder.Build();
}

// TODO(Superjomn) To refactor this, strange to use if-else here.
auto CreateCinnMatmulModule(
    const std::string &name, Target target, int m, int n, int k) {
  if (name == "basic") {
    return CreateMatmulBasicModule(target, m, n, k);
  } else if (name == "tile") {
    return CreateMatmulTileModule(target, m, n, k);
  } else if (name == "split") {
    return CreateMatmulSplitModule(target, m, n, k);
  } else if (name == "block") {
    return CreateMatmulBlockModule(target, m, n, k);
  } else if (name == "vectorize") {
    return CreateMatmulVectorizeModule(target, m, n, k);
  } else if (name == "loop_permutation") {
    return CreateMatmulLoopPermutation(target, m, n, k);
  } else if (name == "array_packing") {
    return CreateMatmulArrayPacking(target, m, n, k);
  }
  { CINN_NOT_IMPLEMENTED }
}

auto CreateExecutionEngine(const cinn::ir::Module &module) {
  auto engine = cinn::backends::ExecutionEngine::Create({});
  engine->Link(module);
  return engine;
}

auto CreateSimpleJit(const cinn::ir::Module &module) {
  auto jit = cinn::backends::SimpleJIT::Create();
  jit->Link(module, true);

  return jit;
}
}  // namespace tests
}  // namespace cinn
