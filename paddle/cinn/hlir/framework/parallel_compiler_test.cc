// Copyright (c) 2022 CINN Authors. All Rights Reserved.
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

#include "paddle/cinn/hlir/framework/parallel_compiler.h"

#include <gtest/gtest.h>

#include "paddle/cinn/common/target.h"
#include "paddle/cinn/frontend/net_builder.h"
#include "paddle/cinn/frontend/optimize.h"
#include "paddle/cinn/hlir/framework/graph_compiler.h"
#include "paddle/cinn/hlir/framework/graph_compiler_util.h"

namespace cinn {
namespace hlir {
namespace framework {

TEST(ParallelCompilerTest, Add_TEST_0) {
  frontend::NetBuilder builder("Add_TEST_0");
  auto A = builder.CreateInput(Float(32), {128, 128}, "A");
  auto B = builder.CreateInput(Float(32), {128, 128}, "B");
  auto C = builder.Add(A, B);
  auto target = common::DefaultNVGPUTarget();
  auto program = builder.Build();
  auto graph = std::make_shared<Graph>(program, target);
  auto scope = BuildScope(target, graph);

  CompilationContext context(graph, scope, target);
  ParallelCompiler pc(&context);
  auto compilation_result = pc();
}

TEST(ParallelCompilerTest, Conv2d_Test_0) {
  frontend::NetBuilder builder("Conv2d_Test_0");
  auto A = builder.CreateInput(Float(32), {1, 64, 112, 112}, "A");
  auto B = builder.CreateInput(Float(32), {64, 64, 3, 3}, "B");
  auto C = builder.CreateInput(Float(32), {1, 64, 56, 56}, "C");
  auto D = builder.Conv2d(A, B, {2, 2}, {1, 1});
  auto E = builder.Add(C, D);

  auto target = common::DefaultNVGPUTarget();
  auto program = builder.Build();
  auto graph = frontend::Optimize(&program, {}, target);
  auto scope = BuildScope(target, graph);

  CompilationContext context(graph, scope, target);
  ParallelCompiler pc(&context);
  auto compilation_result = pc();
}

TEST(ParallelCompilerTest, Matmul_Test_0) {
  frontend::NetBuilder builder("Matmul_Test_0");
  auto A = builder.CreateInput(Float(32), {64, 128, 128}, "A");
  auto B = builder.CreateInput(Float(32), {64, 128, 128}, "B");
  auto C = builder.CreateInput(Float(32), {64, 128, 128}, "C");
  auto D = builder.Matmul(A, B);
  auto E = builder.Add(C, D);

  auto target = common::DefaultNVGPUTarget();
  auto program = builder.Build();
  auto graph = frontend::Optimize(&program, {}, target);
  auto scope = BuildScope(target, graph);

  CompilationContext context(graph, scope, target);
  ParallelCompiler pc(&context);
  auto compilation_result = pc();
}

}  // namespace framework
}  // namespace hlir
}  // namespace cinn
