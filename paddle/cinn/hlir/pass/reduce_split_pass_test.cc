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

TEST(ReduceSplit, reduce_mean_nhwc) {
  NetBuilder net_builder("reduce_sum_nhwc");
  // create model
  int N = 64, H = 14, W = 14, C = 256;
  auto in = net_builder.CreateInput(Float(32), {N, H, W, C}, "in");
  auto out = net_builder.ReduceSum(in, {0, 1, 2});

  auto fetch_ids = {out->id};
  std::vector<float> input_data(N * H * W * C);
  InitRandomVector<float>(&input_data, input_data.size(), 0.0f, 1.0f, 1e-3);
  std::unordered_map<std::string, std::vector<float>> feeds = {
      {"in", input_data}};
  auto program = net_builder.Build();
  auto output = RunModelTest(program,
                             {"ReduceSplit", "OpFusionPass", "FusionMergePass"},
                             feeds,
                             fetch_ids);
  auto output_expect = RunModelTest(
      program, {"OpFusionPass", "FusionMergePass"}, feeds, fetch_ids);

  for (auto& out : output) {
    CheckOutput<float>(out.second, output_expect[out.first], 1e-8, 1e-4);
  }
}

TEST(ReduceSplit, reduce_mean_nhwc_small_size) {
  NetBuilder net_builder("reduce_sum_nhwc");
  // create model
  int N = 32, H = 2, W = 2, C = 256;
  auto in = net_builder.CreateInput(Float(32), {N, H, W, C}, "in");
  auto out = net_builder.ReduceSum(in, {0, 1, 2});

  auto fetch_ids = {out->id};
  std::vector<float> input_data(N * H * W * C);
  InitRandomVector<float>(&input_data, input_data.size(), 0.0f, 1.0f, 1e-3);
  std::unordered_map<std::string, std::vector<float>> feeds = {
      {"in", input_data}};
  auto program = net_builder.Build();
  auto output = RunModelTest(program,
                             {"ReduceSplit", "OpFusionPass", "FusionMergePass"},
                             feeds,
                             fetch_ids);
  auto output_expect = RunModelTest(
      program, {"OpFusionPass", "FusionMergePass"}, feeds, fetch_ids);

  for (auto& out : output) {
    // should be equal, since ReduceSplit is not affected
    CheckOutput<float>(out.second, output_expect[out.first], 0.0f, 0.0f);
  }
}

}  // namespace frontend
}  // namespace cinn
