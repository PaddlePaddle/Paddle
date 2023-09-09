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

#include <memory>

#include "paddle/cinn/cinn.h"
#include "paddle/cinn/frontend/optimize.h"
#include "paddle/cinn/frontend/syntax.h"
#include "paddle/cinn/hlir/framework/graph.h"
#include "paddle/cinn/hlir/framework/graph_compiler.h"
#include "paddle/cinn/hlir/framework/pass.h"
#include "paddle/cinn/hlir/op/use_ops.h"
#include "paddle/cinn/hlir/pass/use_pass.h"
#include "paddle/cinn/utils/data_util.h"

PD_DEFINE_string(model_dir, "", "");

namespace cinn {
namespace frontend {

using hlir::framework::Scope;
using utils::Join;

// batch_norm primitives
TEST(batch_norm_meta, batch_norm_meta) {
  Placeholder A(Float(32), {1, 64, 112, 112}, "A");

  Placeholder Scale(Float(32), {64}, "Scale");
  Placeholder Bias(Float(32), {64}, "Bias");
  Placeholder Mean(Float(32), {64}, "Mean");
  Placeholder Variance(Float(32), {64}, "Variance");

  Program program;
  absl::flat_hash_map<std::string, Program::attr_t> attrs;
  attrs["epsilon"] = static_cast<float>(0.001);

  auto a = program.batchnorm(A, Scale, Bias, Mean, Variance, attrs);

  auto b =
      program.fused_batchnorm_inference(A, Scale, Bias, Mean, Variance, attrs);

  Target target = common::DefaultTarget();
  program.SetInputs({A});
  program.Validate();
  LOG(INFO) << "Program:\n" << program;
  auto graph = std::make_shared<hlir::framework::Graph>(program, target);

  hlir::framework::ApplyPass(graph.get(), "InferShape");
#ifndef CINN_WITH_CUDA
  hlir::framework::ApplyPass(graph.get(), "AlterLayout");
#endif
  hlir::framework::ApplyPasses(graph.get(), frontend::DefaultOpFusionPasses());
  auto scope = BuildScope(target, graph);
  LOG(INFO) << "graph:\n" << graph->Visualize();

  hlir::framework::CompilationContext context(graph, scope, target);
  hlir::framework::GraphCompiler gc(context);
  auto runtime_program = gc.Build();

  scope->Var<hlir::framework::Tensor>("A");

  auto A1 = scope->GetTensor("A");
  SetRandData<float>(A1, target);

  runtime_program->Execute();
}

TEST(reduction, reduce) {
  Placeholder A(Float(32), {1, 3, 224, 224}, "A");

  Program program;
  std::unordered_map<std::string, Program::attr_t> attrs;
  std::vector<int> axis = {1, 2};
  bool keep_dim = false;

  auto a = program.reduce_max(A, axis, keep_dim);
  auto b = program.reduce_min(A, axis, keep_dim);
  auto c = program.reduce_prod(A, axis, keep_dim);
  auto d = program.reduce_sum(A, {0, 1, 2, 3}, keep_dim);

  Target target = common::DefaultTarget();
  program.SetInputs({A});
  program.Validate();
  LOG(INFO) << "Program:\n" << program;
  auto graph = std::make_shared<hlir::framework::Graph>(program, target);

  hlir::framework::ApplyPass(graph.get(), "InferShape");
#ifndef CINN_WITH_CUDA
  hlir::framework::ApplyPass(graph.get(), "AlterLayout");
#endif
  hlir::framework::ApplyPasses(graph.get(), frontend::DefaultOpFusionPasses());
  auto scope = BuildScope(target, graph);
  LOG(INFO) << "graph:\n" << graph->Visualize();

  hlir::framework::CompilationContext context(graph, scope, target);
  hlir::framework::GraphCompiler gc(context);
  auto runtime_program = gc.Build();

  scope->Var<hlir::framework::Tensor>("A");

  auto A1 = scope->GetTensor("A");
  SetRandData<float>(A1, target);

  runtime_program->Execute();
}

TEST(Compare, Compare) {
  Placeholder A(Float(32), {1, 3, 224, 224}, "A");
  Placeholder B(Float(32), {1, 3, 224, 224}, "B");

  Program program;
  auto a = program.primitive_equal(A, B);

  Target target = common::DefaultTarget();
  program.SetInputs({A, B});
  program.Validate();
  LOG(INFO) << "Program:\n" << program;
  auto graph = std::make_shared<hlir::framework::Graph>(program, target);

  hlir::framework::ApplyPass(graph.get(), "InferShape");
#ifndef CINN_WITH_CUDA
  hlir::framework::ApplyPass(graph.get(), "AlterLayout");
#endif
  hlir::framework::ApplyPasses(graph.get(), frontend::DefaultOpFusionPasses());
  auto scope = BuildScope(target, graph);
  LOG(INFO) << "graph:\n" << graph->Visualize();

  hlir::framework::CompilationContext context(graph, scope, target);
  hlir::framework::GraphCompiler gc(context);
  auto runtime_program = gc.Build();

  scope->Var<hlir::framework::Tensor>("A");
  scope->Var<hlir::framework::Tensor>("B");

  auto A1 = scope->GetTensor("A");
  auto B1 = scope->GetTensor("B");
  SetRandData<float>(A1, target);
  SetRandData<float>(B1, target);

  runtime_program->Execute();
}

}  // namespace frontend
}  // namespace cinn
