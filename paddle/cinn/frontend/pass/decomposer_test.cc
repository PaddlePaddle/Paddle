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

#include <random>

#include "paddle/cinn/frontend/decomposer/use_decomposer.h"
#include "paddle/cinn/frontend/decomposer_registry.h"
#include "paddle/cinn/frontend/net_builder.h"
#include "paddle/cinn/frontend/optimize.h"
#include "paddle/cinn/frontend/pass/use_program_pass.h"
#include "paddle/cinn/frontend/program_pass.h"
#include "paddle/cinn/hlir/framework/graph.h"
#include "paddle/cinn/hlir/framework/graph_compiler.h"
#include "paddle/cinn/hlir/framework/pass.h"
#include "paddle/cinn/hlir/framework/tensor.h"
#include "paddle/cinn/hlir/op/use_ops.h"
#include "paddle/cinn/hlir/pass/use_pass.h"
#include "paddle/cinn/utils/data_util.h"

namespace cinn::frontend {

Program CreateAddProgram() {
  constexpr int M = 32;
  constexpr int N = 24;

  NetBuilder builder("net_builder");
  auto a = builder.CreateInput(Float(32), {M, N});
  auto b = builder.CreateInput(Float(32), {M, N});
  auto c = builder.Relu(a);
  auto d = builder.Add(b, c);
  auto program = builder.Build();

  return program;
}

TEST(DecomposePassRegistry, basic) {
  ASSERT_NE(cinn::frontend::ProgramPassRegistry::Global()->Find("Decomposer"),
            nullptr);
  ASSERT_EQ(cinn::frontend::ProgramPassRegistry::Global()->Find("Test"),
            nullptr);
}

TEST(DecomposePass, basic) {
  auto prog = CreateAddProgram();
  for (int i = 0; i < prog.size(); i++) {
    LOG(INFO) << "instruction: " << prog[i];
  }

#ifdef CINN_WITH_CUDA
  Target target = common::DefaultNVGPUTarget();
#else
  Target target = common::DefaultHostTarget();
#endif

  ProgramPass::Apply(&prog, {}, target, {"Decomposer"});
  for (int i = 0; i < prog.size(); i++) {
    LOG(INFO) << "new instruction: " << prog[i];
  }

  auto graph = std::make_shared<hlir::framework::Graph>(prog, target);
  hlir::framework::ApplyPasses(graph.get(), DefaultOpFusionPasses());
  auto scope = BuildScope(target, graph);
  hlir::framework::CompilationContext context(graph, scope, target);
  hlir::framework::GraphCompiler gc(context);
  auto runtime_program = gc.Build();

  scope->Var<hlir::framework::Tensor>("A");
  scope->Var<hlir::framework::Tensor>("B");

  auto A = scope->GetTensor("A");
  auto B = scope->GetTensor("B");
  SetRandData<float>(A, target);
  SetRandData<float>(B, target);

  runtime_program->Execute();
}

}  // namespace cinn::frontend
