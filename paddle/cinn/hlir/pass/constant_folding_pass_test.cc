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

namespace cinn {
namespace frontend {

int GetSize(const std::vector<int>& shape) {
  return std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>());
}

std::unordered_map<std::string, std::vector<float>> GetInputRandom(
    const std::vector<Variable>&& inputs) {
  std::unordered_map<std::string, std::vector<float>> input_data;
  for (auto input : inputs) {
    input_data[input->id] = std::vector<float>(GetSize(input->shape));
    InitRandomVector<float>(
        &input_data[input->id], input_data[input->id].size(), 0.0f, 1.0f, 1e-3);
  }

  return input_data;
}

std::unordered_map<std::string, std::vector<float>> RunModelTest(
    Program& program,  // NOLINT
    const std::vector<std::string>&& passes,
    const std::unordered_map<std::string, std::vector<float>>& input_data,
    const std::unordered_set<std::string>& fetch_ids) {
  auto target = common::DefaultTarget();
  auto graph =
      std::make_shared<hlir::framework::Graph>(program, fetch_ids, target);
  hlir::framework::ApplyPasses(graph.get(), passes);

  auto scope = BuildScope(target, graph);
  hlir::framework::CompilationContext context(graph, scope, target);
  hlir::framework::GraphCompiler gc(context);
  auto run_program = gc.Build();

  for (auto& data : input_data) {
    scope->Var<hlir::framework::Tensor>(data.first);
    auto tensor = scope->GetTensor(data.first);
    CopyFromVector(data.second, tensor, target);
  }
  run_program->Execute();

  std::unordered_map<std::string, std::vector<float>> outputs;
  for (auto id : fetch_ids) {
    auto tensor = scope->GetTensor(id);
    std::vector<float> data(tensor->shape().numel());
    CopyToVector(tensor, &data);
    outputs[id] = data;
  }

  return outputs;
}

TEST(Constant_Folding, fold_broadcast_to_const_scalar_1) {
  NetBuilder net_builder("fold_broadcast_to_const_scalar_1");
  // create model
  int h = 32, w = 32;
  auto A = net_builder.Constant<float>(1.0f, "A");
  auto B = net_builder.BroadcastTo(A, {h, w}, {1});
  auto C = net_builder.CreateInput(Float(32), {h, w}, "C");
  auto D = net_builder.Add(B, C);

  auto fetch_ids = {D->id};
  auto input_data = GetInputRandom({C});
  auto program = net_builder.Build();
  auto output0 = RunModelTest(
      program, {"OpFusionPass", "FusionMergePass"}, input_data, fetch_ids);
  auto output1 =
      RunModelTest(program,
                   {"ConstantFolding", "OpFusionPass", "FusionMergePass"},
                   input_data,
                   fetch_ids);

  for (auto& output : output0) {
    CHECK(output1.count(output.first));
    CheckOutput<float>(output.second, output1[output.first], 1e-8, 1e-4);
  }
}

TEST(Constant_Folding, fold_broadcast_to_const_scalar_2) {
  NetBuilder net_builder("fold_broadcast_to_const_scalar_2");
  // create model
  int h = 32, w = 32;
  auto A = net_builder.Constant<float>(1.0f, "A");
  auto B = net_builder.BroadcastTo(A, {h, w}, {1});
  auto C = net_builder.CreateInput(Float(32), {h, w}, "C");
  auto D = net_builder.CreateInput(Float(32), {1}, "D");
  auto E = net_builder.Add(B, C);
  auto F = net_builder.Add(A, D);

  auto fetch_ids = {E->id, F->id};
  auto input_data = GetInputRandom({C, D});
  auto program = net_builder.Build();
  auto output0 = RunModelTest(
      program, {"OpFusionPass", "FusionMergePass"}, input_data, fetch_ids);
  auto output1 =
      RunModelTest(program,
                   {"ConstantFolding", "OpFusionPass", "FusionMergePass"},
                   input_data,
                   fetch_ids);

  for (auto& output : output0) {
    CHECK(output1.count(output.first));
    CheckOutput<float>(output.second, output1[output.first], 1e-8, 1e-4);
  }
}

TEST(Constant_Folding, fold_broadcast_to_const_scalar_3) {
  NetBuilder net_builder("fold_broadcast_to_const_scalar_3");
  // create model
  int h = 32, w = 32;
  auto A = net_builder.Constant<float>(1.0f, "A");
  auto B = net_builder.BroadcastTo(A, {h, w}, {1});
  auto C = net_builder.CreateInput(Float(32), {h, w}, "C");
  auto D = net_builder.BroadcastTo(A, {h, w, w}, {2});
  auto E = net_builder.CreateInput(Float(32), {h, w, w}, "E");
  auto F = net_builder.Add(B, C);
  auto G = net_builder.Add(D, E);

  auto fetch_ids = {G->id, F->id};
  auto input_data = GetInputRandom({C, E});
  auto program = net_builder.Build();
  auto output0 = RunModelTest(
      program, {"OpFusionPass", "FusionMergePass"}, input_data, fetch_ids);
  auto output1 =
      RunModelTest(program,
                   {"ConstantFolding", "OpFusionPass", "FusionMergePass"},
                   input_data,
                   fetch_ids);

  for (auto& output : output0) {
    CHECK(output1.count(output.first));
    CheckOutput<float>(output.second, output1[output.first], 1e-8, 1e-4);
  }
}

TEST(Constant_Folding, fold_broadcast_to_fill_constant_1) {
  NetBuilder net_builder("fold_broadcast_to_fill_constant_1");
  // create model
  int h = 32, w = 32;
  auto A = net_builder.FillConstant<float>({w}, 1.0f, "A");
  auto B = net_builder.BroadcastTo(A, {h, w}, {1});
  auto C = net_builder.CreateInput(Float(32), {h, w}, "C");
  auto D = net_builder.Add(B, C);

  auto fetch_ids = {D->id};
  auto input_data = GetInputRandom({C});
  auto program = net_builder.Build();
  auto output0 = RunModelTest(
      program, {"OpFusionPass", "FusionMergePass"}, input_data, fetch_ids);
  auto output1 =
      RunModelTest(program,
                   {"ConstantFolding", "OpFusionPass", "FusionMergePass"},
                   input_data,
                   fetch_ids);

  for (auto& output : output0) {
    CHECK(output1.count(output.first));
    CheckOutput<float>(output.second, output1[output.first], 1e-8, 1e-4);
  }
}

TEST(Constant_Folding, fold_broadcast_to_fill_constant_2) {
  NetBuilder net_builder("fold_broadcast_to_fill_constant_2");
  // create model
  int h = 32, w = 32;
  auto A = net_builder.FillConstant<float>({w}, 1.0f, "A");
  auto B = net_builder.BroadcastTo(A, {h, w}, {1});
  auto C = net_builder.CreateInput(Float(32), {h, w}, "C");
  auto D = net_builder.CreateInput(Float(32), {w}, "D");
  auto E = net_builder.Add(B, C);
  auto F = net_builder.Add(A, D);

  auto fetch_ids = {E->id, F->id};
  auto input_data = GetInputRandom({C, D});
  auto program = net_builder.Build();
  auto output0 = RunModelTest(
      program, {"OpFusionPass", "FusionMergePass"}, input_data, fetch_ids);
  auto output1 =
      RunModelTest(program,
                   {"ConstantFolding", "OpFusionPass", "FusionMergePass"},
                   input_data,
                   fetch_ids);

  for (auto& output : output0) {
    CHECK(output1.count(output.first));
    CheckOutput<float>(output.second, output1[output.first], 1e-8, 1e-4);
  }
}

TEST(Constant_Folding, fold_reshape_fill_constant_1) {
  NetBuilder net_builder("fold_reshape_fill_constant_1");
  // create model
  int h = 32, w = 32;
  auto A = net_builder.FillConstant<float>({h * w}, 1.0f, "A");
  auto B = net_builder.Reshape(A, {h, w});
  auto C = net_builder.CreateInput(Float(32), {h, w}, "C");
  auto D = net_builder.Add(B, C);

  auto fetch_ids = {D->id};
  auto input_data = GetInputRandom({C});
  auto program = net_builder.Build();
  auto output0 = RunModelTest(
      program, {"OpFusionPass", "FusionMergePass"}, input_data, fetch_ids);
  auto output1 =
      RunModelTest(program,
                   {"ConstantFolding", "OpFusionPass", "FusionMergePass"},
                   input_data,
                   fetch_ids);

  for (auto& output : output0) {
    CHECK(output1.count(output.first));
    CheckOutput<float>(output.second, output1[output.first], 1e-8, 1e-4);
  }
}

TEST(Constant_Folding, fold_reshape_fill_constant_2) {
  NetBuilder net_builder("fold_reshape_fill_constant_2");
  // create model
  int h = 32, w = 32;
  auto A = net_builder.FillConstant<float>({h * w}, 1.0f, "A");
  auto B = net_builder.Reshape(A, {h, w});
  auto C = net_builder.CreateInput(Float(32), {h, w}, "C");
  auto D = net_builder.CreateInput(Float(32), {h * w}, "D");
  auto E = net_builder.Add(B, C);
  auto F = net_builder.Add(A, D);

  auto fetch_ids = {E->id, F->id};
  auto input_data = GetInputRandom({C, D});
  auto program = net_builder.Build();
  auto output0 = RunModelTest(
      program, {"OpFusionPass", "FusionMergePass"}, input_data, fetch_ids);
  auto output1 =
      RunModelTest(program,
                   {"ConstantFolding", "OpFusionPass", "FusionMergePass"},
                   input_data,
                   fetch_ids);

  for (auto& output : output0) {
    CHECK(output1.count(output.first));
    CheckOutput<float>(output.second, output1[output.first], 1e-8, 1e-4);
  }
}

TEST(Constant_Folding, fold_squeeze_fill_constant_1) {
  NetBuilder net_builder("fold_squeeze_fill_constant_1");
  // create model
  int h = 32, w = 32;
  auto A = net_builder.FillConstant<float>({h, 1, w, 1}, 1.0f, "A");
  auto B = net_builder.Squeeze(A, {1, 3});
  auto C = net_builder.CreateInput(Float(32), {h, w}, "C");
  auto D = net_builder.Add(B, C);

  auto fetch_ids = {D->id};
  auto input_data = GetInputRandom({C});
  auto program = net_builder.Build();
  auto output0 = RunModelTest(
      program, {"OpFusionPass", "FusionMergePass"}, input_data, fetch_ids);
  auto output1 =
      RunModelTest(program,
                   {"ConstantFolding", "OpFusionPass", "FusionMergePass"},
                   input_data,
                   fetch_ids);

  for (auto& output : output0) {
    CHECK(output1.count(output.first));
    CheckOutput<float>(output.second, output1[output.first], 1e-8, 1e-4);
  }
}

TEST(Constant_Folding, fold_squeeze_fill_constant_2) {
  NetBuilder net_builder("fold_squeeze_fill_constant_2");
  // create model
  int h = 32, w = 32;
  auto A = net_builder.FillConstant<float>({h, 1, w, 1}, 1.0f, "A");
  auto B = net_builder.Squeeze(A, {1, 3});
  auto C = net_builder.CreateInput(Float(32), {h, w}, "C");
  auto D = net_builder.CreateInput(Float(32), {h, 1, w, 1}, "D");
  auto E = net_builder.Add(B, C);
  auto F = net_builder.Add(A, D);

  auto fetch_ids = {E->id, F->id};
  auto input_data = GetInputRandom({C, D});
  auto program = net_builder.Build();
  auto output0 = RunModelTest(
      program, {"OpFusionPass", "FusionMergePass"}, input_data, fetch_ids);
  auto output1 =
      RunModelTest(program,
                   {"ConstantFolding", "OpFusionPass", "FusionMergePass"},
                   input_data,
                   fetch_ids);

  for (auto& output : output0) {
    CHECK(output1.count(output.first));
    CheckOutput<float>(output.second, output1[output.first], 1e-8, 1e-4);
  }
}

TEST(Constant_Folding, fold_expand_dims_to_fill_constant_1) {
  NetBuilder net_builder("fold_expand_dims_to_fill_constant_1");
  // create model
  int h = 32, w = 32;
  auto A = net_builder.FillConstant<float>({h, w}, 1.0f, "A");
  auto B = net_builder.ExpandDims(A, {1, 3});
  auto C = net_builder.CreateInput(Float(32), {h, 1, w, 1}, "C");
  auto D = net_builder.Add(B, C);

  auto fetch_ids = {D->id};
  auto input_data = GetInputRandom({C});
  auto program = net_builder.Build();
  auto output0 = RunModelTest(
      program, {"OpFusionPass", "FusionMergePass"}, input_data, fetch_ids);
  auto output1 =
      RunModelTest(program,
                   {"ConstantFolding", "OpFusionPass", "FusionMergePass"},
                   input_data,
                   fetch_ids);

  for (auto& output : output0) {
    CHECK(output1.count(output.first));
    CheckOutput<float>(output.second, output1[output.first], 1e-8, 1e-4);
  }
}

TEST(Constant_Folding, fold_expand_dims_to_fill_constant_2) {
  NetBuilder net_builder("fold_expand_dims_to_fill_constant_2");
  // create model
  int h = 32, w = 32;
  auto A = net_builder.FillConstant<float>({h, w}, 1.0f, "A");
  auto B = net_builder.ExpandDims(A, {1, 3});
  auto C = net_builder.CreateInput(Float(32), {h, 1, w, 1}, "C");
  auto D = net_builder.CreateInput(Float(32), {h, w}, "D");
  auto E = net_builder.Add(B, C);
  auto F = net_builder.Add(A, D);

  auto fetch_ids = {E->id, F->id};
  auto input_data = GetInputRandom({C, D});
  auto program = net_builder.Build();
  auto output0 = RunModelTest(
      program, {"OpFusionPass", "FusionMergePass"}, input_data, fetch_ids);
  auto output1 =
      RunModelTest(program,
                   {"ConstantFolding", "OpFusionPass", "FusionMergePass"},
                   input_data,
                   fetch_ids);

  for (auto& output : output0) {
    CHECK(output1.count(output.first));
    CheckOutput<float>(output.second, output1[output.first], 1e-8, 1e-4);
  }
}

TEST(Constant_Folding, fold_expand_dims_to_fill_constant_3) {
  NetBuilder net_builder("fold_expand_dims_to_fill_constant_3");
  // create model, ExpandDims axes have nagetive value
  int h = 32, w = 32;
  auto A = net_builder.FillConstant<float>({h, w}, 1.0f, "A");
  auto B = net_builder.ExpandDims(A, {1, -1});
  auto C = net_builder.CreateInput(Float(32), {h, 1, w, 1}, "C");
  auto D = net_builder.CreateInput(Float(32), {h, w}, "D");
  auto E = net_builder.Add(B, C);
  auto F = net_builder.Add(A, D);

  auto fetch_ids = {E->id, F->id};
  auto input_data = GetInputRandom({C, D});
  auto program = net_builder.Build();
  auto output0 = RunModelTest(
      program, {"OpFusionPass", "FusionMergePass"}, input_data, fetch_ids);
  auto output1 =
      RunModelTest(program,
                   {"ConstantFolding", "OpFusionPass", "FusionMergePass"},
                   input_data,
                   fetch_ids);

  for (auto& output : output0) {
    CHECK(output1.count(output.first));
    CheckOutput<float>(output.second, output1[output.first], 1e-8, 1e-4);
  }
}

}  // namespace frontend
}  // namespace cinn
