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

#include <gtest/gtest.h>

#include "paddle/cinn/frontend/decomposer/test_helper.h"

namespace cinn {
namespace frontend {

int GetSize(const std::vector<int>& shape) {
  return std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>());
}

void RunModelTest(Program& program,  // NOLINT
                  const std::vector<Variable>&& inputs,
                  const std::unordered_set<std::string>& fetch_ids) {
  // init input data.
  std::vector<std::vector<float>> inputs_data;
  for (auto input : inputs) {
    inputs_data.emplace_back(GetSize(input->shape));
    InitRandomVector<float>(
        &inputs_data.back(), inputs_data.back().size(), 0.0f, 1.0f, 1e-3);
  }

  auto target = common::DefaultTarget();
  std::unordered_map<std::string,
                     std::pair<std::vector<float>, std::vector<float>>>
      outputs;
  {
    auto graph =
        std::make_shared<hlir::framework::Graph>(program, fetch_ids, target);
    hlir::framework::ApplyPass(graph.get(), "TransToCustomCallPass");
    hlir::framework::ApplyPass(graph.get(), "OpFusionPass");
    hlir::framework::ApplyPass(graph.get(), "FusionMergePass");

    auto scope = BuildScope(target, graph);
    hlir::framework::CompilationContext context(graph, scope, target);
    hlir::framework::GraphCompiler gc(context);
    auto run_program = gc.Build();

    for (int idx = 0; idx < inputs.size(); ++idx) {
      scope->Var<hlir::framework::Tensor>(inputs[idx]->id);
      auto tensor = scope->GetTensor(inputs[idx]->id);
      auto* data = tensor->mutable_data<float>(target);
      CopyFromVector(inputs_data[idx], tensor, target);
    }
    run_program->Execute();
    for (auto id : fetch_ids) {
      auto tensor = scope->GetTensor(id);
      std::vector<float> data(tensor->shape().numel());
      CopyToVector(tensor, &data);
      outputs[id] = std::pair<std::vector<float>, std::vector<float>>(
          data, std::vector<float>());
    }
  }
  {
    auto graph =
        std::make_shared<hlir::framework::Graph>(program, fetch_ids, target);
    hlir::framework::ApplyPass(graph.get(), "DenseMergePass");
    hlir::framework::ApplyPass(graph.get(), "OpFusionPass");
    hlir::framework::ApplyPass(graph.get(), "FusionMergePass");

    auto scope = BuildScope(target, graph);
    hlir::framework::CompilationContext context(graph, scope, target);
    hlir::framework::GraphCompiler gc(context);
    auto run_program = gc.Build();

    for (int idx = 0; idx < inputs.size(); ++idx) {
      scope->Var<hlir::framework::Tensor>(inputs[idx]->id);
      auto tensor = scope->GetTensor(inputs[idx]->id);
      auto* data = tensor->mutable_data<float>(target);
      CopyFromVector(inputs_data[idx], tensor, target);
    }
    run_program->Execute();
    for (auto id : fetch_ids) {
      auto tensor = scope->GetTensor(id);
      std::vector<float> data(tensor->shape().numel());
      CopyToVector(tensor, &data);
      outputs[id].second = data;
    }
  }

  for (auto& output : outputs) {
    CheckOutput<float>(output.second.first, output.second.second, 1e-8, 1e-4);
  }
}

TEST(DenseMergePass, Test_Matmul_0) {
  int m = 128, k = 64, n = 128;
  NetBuilder net_builder("Test_Matmul_0");
  auto A = net_builder.CreateInput(Float(32), {m, k}, "A");
  auto B = net_builder.CreateInput(Float(32), {k, n}, "B");
  auto C = net_builder.CreateInput(Float(32), {k, n}, "C");
  auto D = net_builder.Matmul(A, B);
  auto E = net_builder.Matmul(A, C);

  auto fetch_ids = {D->id, E->id};
  auto program = net_builder.Build();
  RunModelTest(program, {A, B, C}, fetch_ids);
}

TEST(DenseMergePass, Test_Matmul_1) {
  NetBuilder net_builder("Test_Matmul_1");
  auto A = net_builder.CreateInput(Float(32), {128, 64}, "A");
  auto B = net_builder.CreateInput(Float(32), {128, 64}, "B");
  auto C = net_builder.CreateInput(Float(32), {64, 128}, "C");
  auto D = net_builder.Matmul(A, C);
  auto E = net_builder.Matmul(B, C);

  auto fetch_ids = {D->id, E->id};
  auto program = net_builder.Build();
  RunModelTest(program, {A, B, C}, fetch_ids);
}

TEST(DenseMergePass, Test_Matmul_2) {
  NetBuilder net_builder("Test_Matmul_2");
  auto A = net_builder.CreateInput(Float(32), {128, 64}, "A");
  auto B = net_builder.CreateInput(Float(32), {128, 64}, "B");
  auto C = net_builder.CreateInput(Float(32), {128, 64}, "C");
  auto D = net_builder.CreateInput(Float(32), {128, 64}, "D");
  auto E = net_builder.CreateInput(Float(32), {64, 128}, "E");
  auto F = net_builder.Matmul(A, E);
  auto G = net_builder.Matmul(B, E);
  auto H = net_builder.Matmul(C, E);
  auto I = net_builder.Matmul(D, E);

  auto fetch_ids = {F->id, G->id, H->id, I->id};
  auto program = net_builder.Build();
  RunModelTest(program, {A, B, C, D, E}, fetch_ids);
}

TEST(DenseMergePass, Test_Matmul_3) {
  NetBuilder net_builder("Test_Matmul_3");
  auto A = net_builder.CreateInput(Float(32), {128, 64}, "A");
  auto B = net_builder.CreateInput(Float(32), {128, 64}, "B");
  auto C = net_builder.CreateInput(Float(32), {64, 128}, "C");
  auto D = net_builder.CreateInput(Float(32), {128, 64}, "D");
  auto E = net_builder.CreateInput(Float(32), {128, 64}, "E");
  auto F = net_builder.Matmul(A, C);
  auto G = net_builder.Matmul(B, C);
  auto H = net_builder.Matmul(C, D);
  auto I = net_builder.Matmul(C, E);

  auto fetch_ids = {F->id, G->id, H->id, I->id};
  auto program = net_builder.Build();
  RunModelTest(program, {A, B, C, D, E}, fetch_ids);
}

TEST(DenseMergePass, Test_Matmul_4) {
  NetBuilder net_builder("Test_Matmul_4");
  auto A = net_builder.CreateInput(Float(32), {128, 64}, "A");
  auto B = net_builder.CreateInput(Float(32), {128, 64}, "B");
  auto C = net_builder.CreateInput(Float(32), {64, 128}, "C");
  auto D = net_builder.CreateInput(Float(32), {64, 128}, "D");
  auto F = net_builder.Matmul(A, C);
  auto G = net_builder.Matmul(B, C);
  auto H = net_builder.Matmul(A, D);
  auto I = net_builder.Matmul(B, D);

  auto fetch_ids = {F->id, G->id, H->id, I->id};
  auto program = net_builder.Build();
  RunModelTest(program, {A, B, C, D}, fetch_ids);
}

}  // namespace frontend
}  // namespace cinn
