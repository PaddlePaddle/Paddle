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

#include "paddle/cinn/frontend/decomposer/test_helper.h"

namespace cinn::frontend {

TEST(Decomposer, relu) {
  NetBuilder builder("relu");
  auto x = builder.CreateInput(Float(32), {20, 10}, "x");
  auto out = builder.Relu(x);

  auto relu_cpu = [](const std::vector<size_t>& lengths,
                     const std::vector<void*>& ptrs) {
    size_t n = lengths[0];
    float* x = static_cast<float*>(ptrs[0]);
    float* out = static_cast<float*>(ptrs[1]);
    for (size_t i = 0; i < n; ++i) {
      float tmp_0 = x[i];
      out[i] = tmp_0 > 0 ? tmp_0 : 0;
    }
  };

  std::vector<std::string> input_names = {x.id().data()};
  std::vector<std::string> output_names = {out->id};
  std::vector<std::vector<int>> output_shapes = {{20, 10}};
  RunAndCheck<float>(
      &builder, input_names, output_names, output_shapes, relu_cpu, -1, 1);
}

TEST(Decomposer, relu_grad) {
  NetBuilder builder("relu_grad");
  auto dout = builder.CreateInput(Float(32), {20, 10}, "dout");
  auto out = builder.CreateInput(Float(32), {20, 10}, "out");
  auto dx = builder.ReluGrad(dout, out);

  auto relu_grad_cpu = [](const std::vector<size_t>& lengths,
                          const std::vector<void*>& ptrs) {
    size_t n = lengths[0];
    float* dout = static_cast<float*>(ptrs[0]);
    float* out = static_cast<float*>(ptrs[1]);
    float* dx = static_cast<float*>(ptrs[2]);
    for (size_t i = 0; i < n; ++i) {
      dx[i] = out[i] > 0 ? dout[i] : 0;
    }
  };

  std::vector<std::string> input_names = {dout.id().data(), out.id().data()};
  std::vector<std::string> output_names = {dx->id};
  std::vector<std::vector<int>> output_shapes = {{20, 10}};
  RunAndCheck<float>(
      &builder, input_names, output_names, output_shapes, relu_grad_cpu, -1, 1);
}

TEST(Decomposer, softmax_decomposer) {
  int n = 16, c = 128, h = 14, w = 14;
  std::vector<int> axes = {1, 2, 3};
  NetBuilder net_builder("softmax_decomposer");
  std::unordered_set<std::string> output_names;
  {
    auto x = net_builder.CreateInput(Float(32), {n, c, h, w}, "x");
    auto y = net_builder.Softmax(x, axes);
    output_names.insert(y->id);
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

  std::vector<float> x(n * c * h * w);
  InitRandomVector<float>(&x, n * c * h * w, 0.0f, 1.0f, 1e-3);
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
