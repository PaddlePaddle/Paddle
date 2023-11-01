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

#include "paddle/cinn/frontend/syntax.h"

#include <gtest/gtest.h>

#include <memory>
//
#include "paddle/cinn/cinn.h"
#include "paddle/cinn/frontend/net_builder.h"
#include "paddle/cinn/frontend/optimize.h"
#include "paddle/cinn/hlir/framework/graph.h"
#include "paddle/cinn/hlir/framework/graph_compiler.h"
#include "paddle/cinn/hlir/framework/graph_compiler_util.h"
#include "paddle/cinn/hlir/framework/pass.h"
#include "paddle/cinn/hlir/framework/scope.h"
#include "paddle/cinn/hlir/op/use_ops.h"
#include "paddle/cinn/hlir/pass/use_pass.h"
#include "paddle/cinn/utils/data_util.h"

PD_DEFINE_string(model_dir, "", "");

namespace cinn {
namespace frontend {

using ::cinn::hlir::framework::Graph;
using ::cinn::hlir::framework::Scope;

// using hlir::framework::Scope;
using utils::Join;

frontend::Program CreateAddProgram() {
  constexpr int M = 32;
  constexpr int N = 24;
  frontend::NetBuilder builder("test");

  auto a = builder.CreateInput(Float(32), {M, N}, "A");
  auto b = builder.CreateInput(Float(32), {M, N}, "B");
  auto c = builder.Add(a, b);
  auto d = builder.Add(a, c);
  return builder.Build();
}

TEST(syntax, basic) {
  auto program = CreateAddProgram();
  // output program
  for (int i = 0; i < program.size(); i++) {
    LOG(INFO) << "instruction: " << program[i];
  }
}

TEST(syntax, program_execute_multi_elementwise_add) {
  auto program = CreateAddProgram();
  Target target = common::DefaultTarget();
  std::unordered_set<std::string> fetch_ids;
  auto graph = Optimize(&program, fetch_ids, target);
  // auto graph    = std::make_shared<hlir::framework::Graph>(*program, target);
  LOG(INFO) << "graph:\n" << graph->Visualize();

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

TEST(syntax, program_execute_multi_elementwise_add2) {
  auto program = CreateAddProgram();
  Target target = common::DefaultTarget();
  std::unordered_set<std::string> fetch_ids;
  auto graph = Optimize(&program, fetch_ids, target);
  LOG(INFO) << "graph:\n" << graph->Visualize();

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

/*
// Load a simple Paddle model, execute it
TEST(load_paddle_model, fc_execute) {
  auto scope = std::make_shared<Scope>();

  std::unordered_map<std::string, std::vector<int>> input_shape_map = {{"A", {1,
30}}}; auto programTuple               = LoadPaddleProgram(FLAGS_model_dir,
scope.get(), input_shape_map, false); auto& program                   =
std::get<0>(programTuple); auto& var_map                   =
std::get<1>(programTuple); auto& var_map_paddle_to_program =
std::get<2>(programTuple);

  LOG(INFO) << "program:\n" << *program;

  Target target = common::DefaultHostTarget();
  std::unordered_set<std::string> fetch_ids;
  auto graph = cinn::frontend::Optimize(program.get(), fetch_ids, target);

  scope = BuildScope(target, graph, scope);
  hlir::framework::CompilationContext context(graph, scope,target);
  hlir::framework::GraphCompiler gc(context);
  auto runtime_program = gc.Build();

  auto at = scope->GetTensor("A");
  SetRandData<float>(at, target);
  LOG(INFO) << "Before Execute";

  runtime_program->Execute();

  LOG(INFO) << "scope.names: " << Join(scope->var_names(), ",");

  const std::string output_name = "fc_0.tmp_2";
  auto tensor = scope->GetTensor(var_map_paddle_to_program.at(output_name));
  LOG(INFO) << "tensor.shape: " << utils::Join(tensor->shape().data(), ",");
  auto data = GetTensorData<float>(tensor, target);
  for (int i = 0; i < 10; i++) {
    LOG(INFO) << "data: " << data[i];
  }
}
*/

}  // namespace frontend
}  // namespace cinn
