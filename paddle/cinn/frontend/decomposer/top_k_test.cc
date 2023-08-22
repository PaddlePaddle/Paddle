// Copyright (c) 2023 CINN Authors. All Rights Reserved.
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

#include "paddle/cinn/frontend/decomposer/test_helper.h"

namespace cinn::frontend {

TEST(Decomposer, top_k_decomposer) {
  NetBuilder net_builder("top_k_decomposer");
  std::unordered_set<std::string> output_names;
  {
    auto x = net_builder.CreateInput(Float(32), {10, 5}, "x");
    auto y = net_builder.TopK(x, 1, -1, true);
    output_names.insert(y[0]->id);
    output_names.insert(y[1]->id);
  }
  auto program = net_builder.Build();

  auto target = common::DefaultTarget();
  RunDecomposer(&program, target);

  auto graph =
      std::make_shared<hlir::framework::Graph>(program, output_names, target);
  hlir::framework::ApplyPass(graph.get(), "OpFusionPass");
  hlir::framework::ApplyPass(graph.get(), "FusionMergePass");

  auto scope = BuildScope(target, graph);
  hlir::framework::CompilationContext context(graph, scope, target);
  hlir::framework::GraphCompiler gc(context);
  auto run_program = gc.Build();

  std::vector<float> x(10 * 5);
  InitRandomVector<float>(&x, 10 * 5, 0.0f, 1.0f, 1e-3);
  std::vector<std::pair<std::string, std::vector<float>>> inputs = {{"x", x}};
  for (auto& input : inputs) {
    scope->Var<hlir::framework::Tensor>(input.first);
    auto tensor = scope->GetTensor(input.first);
    auto* data = tensor->mutable_data<float>(target);
    CopyFromVector(input.second, tensor, target);
  }
  run_program->Execute();
}

}  // namespace cinn::frontend
