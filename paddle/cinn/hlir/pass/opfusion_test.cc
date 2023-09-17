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

/**
 *  complex case: diamond structure
 *         conv
 *        /     \
 *      add    relu
 *        \     /
 *          add
 */
TEST(complex2, complex2) {
  Placeholder A(Float(32), {1, 3, 224, 224}, "A");
  Placeholder B(Float(32), {3, 1, 7, 7}, "B");
  Placeholder C(Float(32), {1, 3, 112, 112}, "C");
  Placeholder D(Float(32), {1, 3, 1, 1}, "D");
  Placeholder E(Float(32), {1, 3, 1, 1}, "E");

  Placeholder Scale(Float(32), {64}, "Scale");
  Placeholder Bias(Float(32), {64}, "Bias");
  Placeholder Mean(Float(32), {64}, "Mean");
  Placeholder Variance(Float(32), {64}, "Variance");

  Program program;
  absl::flat_hash_map<std::string, Program::attr_t> attrs;
  attrs["stride"] = std::vector<int>({2, 2});
  attrs["dilation"] = std::vector<int>({1, 1});
  attrs["padding"] = std::vector<int>({3, 3});
  std::string src_layout = "NCHW";
  attrs["data_format"] = src_layout;

  absl::flat_hash_map<std::string, Program::attr_t> attrs1;
  attrs1["epsilon"] = static_cast<float>(0.001);

  auto c = program.depthwise_conv2d(A, B, attrs);
  auto d = program.elementwise_add(c, C);
  auto e = program.relu(c);
  auto f = program.elementwise_add(d, e);

  Target target = common::DefaultTarget();
  program.SetInputs({A, B, C, D, E});
  program.Validate();
  LOG(INFO) << "Program:\n" << program;
  auto graph = std::make_shared<hlir::framework::Graph>(program, target);

  hlir::framework::ApplyPass(graph.get(), "InferShape");
  hlir::framework::ApplyPass(graph.get(), "OpFusion");
  auto scope = BuildScope(target, graph);
  LOG(INFO) << "graph:\n" << graph->Visualize();

  hlir::framework::CompilationContext context(graph, scope, target);
  hlir::framework::GraphCompiler gc(context);
  auto runtime_program = gc.Build();

  scope->Var<hlir::framework::Tensor>("A");
  scope->Var<hlir::framework::Tensor>("B");
  scope->Var<hlir::framework::Tensor>("C");

  auto A1 = scope->GetTensor("A");
  auto B1 = scope->GetTensor("B");
  auto C1 = scope->GetTensor("C");
  SetRandData<float>(A1, target);
  SetRandData<float>(B1, target);
  SetRandData<float>(C1, target);

  runtime_program->Execute();
}
TEST(complex1, complex1) {
  Placeholder A(Float(32), {1, 3, 224, 224}, "A");
  Placeholder B(Float(32), {64, 3, 7, 7}, "B");
  Placeholder C(Float(32), {1, 64, 112, 112}, "C");
  Placeholder D(Float(32), {1, 64, 1, 1}, "D");
  Placeholder E(Float(32), {1, 64, 1, 1}, "E");

  Placeholder Scale(Float(32), {64}, "Scale");
  Placeholder Bias(Float(32), {64}, "Bias");
  Placeholder Mean(Float(32), {64}, "Mean");
  Placeholder Variance(Float(32), {64}, "Variance");

  Program program;
  absl::flat_hash_map<std::string, Program::attr_t> attrs;
  attrs["stride"] = std::vector<int>({2, 2});
  attrs["dilation"] = std::vector<int>({1, 1});
  attrs["padding"] = std::vector<int>({3, 3});
  std::string src_layout = "NCHW";
  attrs["data_format"] = src_layout;

  absl::flat_hash_map<std::string, Program::attr_t> attrs1;
  attrs1["epsilon"] = static_cast<float>(0.001);

  auto c = program.conv2d(A, B, attrs);
  auto d = program.elementwise_add(c, C);
  auto e = program.relu(c);
  auto f = program.elementwise_add(d, e);

  Target target = common::DefaultTarget();
  program.SetInputs({A, B, C, D, E});
  program.Validate();
  LOG(INFO) << "Program:\n" << program;
  auto graph = std::make_shared<hlir::framework::Graph>(program, target);

  hlir::framework::ApplyPass(graph.get(), "InferShape");
  hlir::framework::ApplyPass(graph.get(), "OpFusion");
  auto scope = BuildScope(target, graph);
  LOG(INFO) << "graph:\n" << graph->Visualize();

  hlir::framework::CompilationContext context(graph, scope, target);
  hlir::framework::GraphCompiler gc(context);
  auto runtime_program = gc.Build();

  scope->Var<hlir::framework::Tensor>("A");
  scope->Var<hlir::framework::Tensor>("B");
  scope->Var<hlir::framework::Tensor>("C");

  auto A1 = scope->GetTensor("A");
  auto B1 = scope->GetTensor("B");
  auto C1 = scope->GetTensor("C");
  SetRandData<float>(A1, target);
  SetRandData<float>(B1, target);
  SetRandData<float>(C1, target);

  runtime_program->Execute();
}

// add+relu
TEST(fuse_add_relu, fuse_add_relu) {
  Placeholder A(Float(32), {1, 64, 112, 112}, "A");
  Placeholder B(Float(32), {64}, "B");

  Program program;
  auto c = program.elementwise_add(A, B, 1);
  auto d = program.relu(c);

  Target target = common::DefaultTarget();
  program.SetInputs({A, B});
  program.Validate();
  LOG(INFO) << "Program:\n" << program;
  auto graph = std::make_shared<hlir::framework::Graph>(program, target);

  hlir::framework::ApplyPass(graph.get(), "InferShape");
  hlir::framework::ApplyPass(graph.get(), "OpFusion");
  auto scope = BuildScope(target, graph);
  LOG(INFO) << "graph:\n" << graph->Visualize();

  hlir::framework::CompilationContext context(graph, scope, target);
  hlir::framework::GraphCompiler gc(context);
  auto runtime_program = gc.Build();

  scope->Var<hlir::framework::Tensor>("A");
  scope->Var<hlir::framework::Tensor>("B");
  scope->Var<hlir::framework::Tensor>("C");

  auto A1 = scope->GetTensor("A");
  auto B1 = scope->GetTensor("B");
  auto C1 = scope->GetTensor("C");
  SetRandData<float>(A1, target);
  SetRandData<float>(B1, target);
  SetRandData<float>(C1, target);

  runtime_program->Execute();
}

// add+add
TEST(fuse_add, fuse_add) {
  Placeholder A(Float(32), {1, 64, 112, 112}, "A");
  Placeholder B(Float(32), {64}, "B");
  Placeholder C(Float(32), {64}, "C");

  Program program;
  auto c = program.elementwise_add(A, B, 1);
  auto d = program.elementwise_add(c, C, 1);

  Target target = common::DefaultTarget();
  program.SetInputs({A, B, C});
  program.Validate();
  LOG(INFO) << "Program:\n" << program;
  auto graph = std::make_shared<hlir::framework::Graph>(program, target);

  hlir::framework::ApplyPass(graph.get(), "InferShape");
  hlir::framework::ApplyPass(graph.get(), "OpFusion");
  auto scope = BuildScope(target, graph);
  LOG(INFO) << "graph:\n" << graph->Visualize();

  hlir::framework::CompilationContext context(graph, scope, target);
  hlir::framework::GraphCompiler gc(context);
  auto runtime_program = gc.Build();

  scope->Var<hlir::framework::Tensor>("A");
  scope->Var<hlir::framework::Tensor>("B");
  scope->Var<hlir::framework::Tensor>("C");

  auto A1 = scope->GetTensor("A");
  auto B1 = scope->GetTensor("B");
  auto C1 = scope->GetTensor("C");
  SetRandData<float>(A1, target);
  SetRandData<float>(B1, target);
  SetRandData<float>(C1, target);

  runtime_program->Execute();
}

// conv+bn+add+add+relu
TEST(conv_bn_conv, conv_bn_conv) {
  Placeholder A(Float(32), {1, 3, 224, 224}, "A");
  Placeholder B(Float(32), {64, 3, 7, 7}, "B");
  Placeholder C(Float(32), {1, 64, 112, 112}, "C");
  Placeholder D(Float(32), {1, 64, 1, 1}, "D");
  Placeholder E(Float(32), {1, 64, 1, 1}, "E");

  Placeholder Scale(Float(32), {64}, "Scale");
  Placeholder Bias(Float(32), {64}, "Bias");
  Placeholder Mean(Float(32), {64}, "Mean");
  Placeholder Variance(Float(32), {64}, "Variance");

  Program program;
  absl::flat_hash_map<std::string, Program::attr_t> attrs;
  attrs["stride"] = std::vector<int>({2, 2});
  attrs["dilation"] = std::vector<int>({1, 1});
  attrs["padding"] = std::vector<int>({3, 3});
  std::string src_layout = "NCHW";
  attrs["data_format"] = src_layout;

  absl::flat_hash_map<std::string, Program::attr_t> attrs1;
  attrs1["epsilon"] = static_cast<float>(0.001);

  auto c = program.conv2d(A, B, attrs);
  auto d = program.batchnorm(c, Scale, Bias, Mean, Variance, attrs1);
  auto e = program.elementwise_add(d, C);
  auto f = program.elementwise_mul(e, D);
  auto g = program.relu(f);

  Target target = common::DefaultTarget();
  program.SetInputs({A, B, C, D, E});
  program.Validate();
  LOG(INFO) << "Program:\n" << program;
  auto graph = std::make_shared<hlir::framework::Graph>(program, target);

  hlir::framework::ApplyPass(graph.get(), "InferShape");
  hlir::framework::ApplyPass(graph.get(), "OpFusion");
  auto scope = BuildScope(target, graph);
  LOG(INFO) << "graph:\n" << graph->Visualize();

  hlir::framework::CompilationContext context(graph, scope, target);
  hlir::framework::GraphCompiler gc(context);
  auto runtime_program = gc.Build();

  scope->Var<hlir::framework::Tensor>("A");
  scope->Var<hlir::framework::Tensor>("B");
  scope->Var<hlir::framework::Tensor>("C");
  scope->Var<hlir::framework::Tensor>("D");
  scope->Var<hlir::framework::Tensor>("E");

  auto A1 = scope->GetTensor("A");
  auto B1 = scope->GetTensor("B");
  auto C1 = scope->GetTensor("C");
  auto D1 = scope->GetTensor("D");
  auto E1 = scope->GetTensor("E");
  SetRandData<float>(A1, target);
  SetRandData<float>(B1, target);
  SetRandData<float>(C1, target);
  SetRandData<float>(D1, target);
  SetRandData<float>(E1, target);

  runtime_program->Execute();
}

// conv+add
TEST(fuse_conv_add, fuse_conv_add) {
  Placeholder A(Float(32), {1, 3, 224, 224}, "A");
  Placeholder B(Float(32), {64, 3, 7, 7}, "B");
  Placeholder C(Float(32), {64}, "C");

  Program program;
  absl::flat_hash_map<std::string, Program::attr_t> attrs;
  attrs["stride"] = std::vector<int>({2, 2});
  attrs["dilation"] = std::vector<int>({1, 1});
  attrs["padding"] = std::vector<int>({3, 3});
  std::string src_layout = "NCHW";
  attrs["data_format"] = src_layout;

  auto c = program.conv2d(A, B, attrs);
  auto d = program.elementwise_add(c, C, 1);

  Target target = common::DefaultTarget();
  program.SetInputs({A, B, C});
  program.Validate();
  LOG(INFO) << "Program:\n" << program;
  auto graph = std::make_shared<hlir::framework::Graph>(program, target);

  hlir::framework::ApplyPass(graph.get(), "InferShape");
  hlir::framework::ApplyPass(graph.get(), "OpFusion");
  auto scope = BuildScope(target, graph);
  LOG(INFO) << "graph:\n" << graph->Visualize();

  hlir::framework::CompilationContext context(graph, scope, target);
  hlir::framework::GraphCompiler gc(context);
  auto runtime_program = gc.Build();

  scope->Var<hlir::framework::Tensor>("A");
  scope->Var<hlir::framework::Tensor>("B");
  scope->Var<hlir::framework::Tensor>("C");
  scope->Var<hlir::framework::Tensor>("D");

  auto A1 = scope->GetTensor("A");
  auto B1 = scope->GetTensor("B");
  auto C1 = scope->GetTensor("C");
  auto D1 = scope->GetTensor("D");
  SetRandData<float>(A1, target);
  SetRandData<float>(B1, target);
  SetRandData<float>(C1, target);
  SetRandData<float>(D1, target);

  runtime_program->Execute();
}

// conv+add+mul
TEST(conv_add_mul, conv_add_mul) {
  Placeholder A(Float(32), {1, 3, 224, 224}, "A");
  Placeholder B(Float(32), {64, 3, 7, 7}, "B");
  Placeholder C(Float(32), {64}, "C");
  Placeholder D(Float(32), {64, 64, 7, 7}, "D");

  Placeholder Scale(Float(32), {1, 64, 1, 1}, "Scale");
  Placeholder Bias(Float(32), {64}, "Bias");
  Placeholder Mean(Float(32), {64}, "Mean");
  Placeholder Variance(Float(32), {64}, "Variance");

  Program program;
  absl::flat_hash_map<std::string, Program::attr_t> attrs;
  attrs["stride"] = std::vector<int>({2, 2});
  attrs["dilation"] = std::vector<int>({1, 1});
  attrs["padding"] = std::vector<int>({3, 3});
  std::string src_layout = "NCHW";
  attrs["data_format"] = src_layout;

  absl::flat_hash_map<std::string, Program::attr_t> attrs1;
  attrs1["epsilon"] = static_cast<float>(0.001);

  auto c = program.conv2d(A, B, attrs);
  auto d = program.elementwise_add(c, Scale);
  auto e = program.elementwise_mul(d, Bias, 1);

  Target target = common::DefaultTarget();
  program.SetInputs({A, B, D});
  program.Validate();
  LOG(INFO) << "Program:\n" << program;
  auto graph = std::make_shared<hlir::framework::Graph>(program, target);

  hlir::framework::ApplyPass(graph.get(), "InferShape");
  hlir::framework::ApplyPass(graph.get(), "OpFusion");
  auto scope = BuildScope(target, graph);
  LOG(INFO) << "graph:\n" << graph->Visualize();

  hlir::framework::CompilationContext context(graph, scope, target);
  hlir::framework::GraphCompiler gc(context);
  auto runtime_program = gc.Build();

  scope->Var<hlir::framework::Tensor>("A");
  scope->Var<hlir::framework::Tensor>("B");
  scope->Var<hlir::framework::Tensor>("C");
  scope->Var<hlir::framework::Tensor>("D");

  auto A1 = scope->GetTensor("A");
  auto B1 = scope->GetTensor("B");
  auto C1 = scope->GetTensor("C");
  auto D1 = scope->GetTensor("D");
  SetRandData<float>(A1, target);
  SetRandData<float>(B1, target);
  SetRandData<float>(C1, target);
  SetRandData<float>(D1, target);

  runtime_program->Execute();
}

// conv+add with different out shape
TEST(fuse_conv_add1, fuse_conv_add1) {
  Placeholder A(Float(32), {1, 8, 1, 1}, "A");
  Placeholder B(Float(32), {32, 8, 1, 1}, "B");
  Placeholder C(Float(32), {1, 32, 112, 112}, "C");

  Program program;
  absl::flat_hash_map<std::string, Program::attr_t> attrs;
  attrs["stride"] = std::vector<int>({1, 1});
  attrs["dilation"] = std::vector<int>({1, 1});
  attrs["padding"] = std::vector<int>({0, 0});
  std::string src_layout = "NCHW";
  attrs["data_format"] = src_layout;

  auto c = program.conv2d(A, B, attrs);
  auto d = program.elementwise_add(c, C);

  Target target = common::DefaultTarget();
  program.SetInputs({A, B, C});
  program.Validate();
  LOG(INFO) << "Program:\n" << program;
  auto graph = std::make_shared<hlir::framework::Graph>(program, target);

  hlir::framework::ApplyPass(graph.get(), "InferShape");
  hlir::framework::ApplyPass(graph.get(), "AlterLayout");
  hlir::framework::ApplyPass(graph.get(), "OpFusion");
  auto scope = BuildScope(target, graph);
  LOG(INFO) << "graph:\n" << graph->Visualize();

  hlir::framework::CompilationContext context(graph, scope, target);
  hlir::framework::GraphCompiler gc(context);
  auto runtime_program = gc.Build();

  scope->Var<hlir::framework::Tensor>("A");
  scope->Var<hlir::framework::Tensor>("B");
  scope->Var<hlir::framework::Tensor>("C");

  auto A1 = scope->GetTensor("A");
  auto B1 = scope->GetTensor("B");
  auto C1 = scope->GetTensor("C");
  SetRandData<float>(A1, target);
  SetRandData<float>(B1, target);
  SetRandData<float>(C1, target);

  runtime_program->Execute();
}

TEST(transpose_reshape_concat, transpose_reshape_concat) {
  Placeholder A(Float(32), {64, 2}, "A");
  Placeholder B(Float(32), {64, 2}, "B");

  Program program;
  auto a = program.transpose(A, {1, 0});
  auto b = program.transpose(B, {1, 0});
  auto c = program.reshape(a, {4, 32});
  auto d = program.reshape(b, {4, 32});
  auto e = program.concat({c, d});

  Target target = common::DefaultTarget();
  program.SetInputs({A, B});
  program.Validate();
  LOG(INFO) << "Program:\n" << program;
  auto graph = std::make_shared<hlir::framework::Graph>(program, target);

  hlir::framework::ApplyPass(graph.get(), "InferShape");
  hlir::framework::ApplyPass(graph.get(), "OpFusion");
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

// conv + fused_batch_norm
TEST(conv_bn, conv_bn) {
  Placeholder A(Float(32), {1, 3, 224, 224}, "A");
  Placeholder B(Float(32), {64, 3, 7, 7}, "B");
  Placeholder C(Float(32), {64}, "C");
  Placeholder D(Float(32), {64, 64, 7, 7}, "D");

  Placeholder Scale(Float(32), {64}, "Scale");
  Placeholder Bias(Float(32), {64}, "Bias");
  Placeholder Mean(Float(32), {64}, "Mean");
  Placeholder Variance(Float(32), {64}, "Variance");

  Program program;
  absl::flat_hash_map<std::string, Program::attr_t> attrs;
  attrs["stride"] = std::vector<int>({2, 2});
  attrs["dilation"] = std::vector<int>({1, 1});
  attrs["padding"] = std::vector<int>({3, 3});
  std::string src_layout = "NCHW";
  attrs["data_format"] = src_layout;

  absl::flat_hash_map<std::string, Program::attr_t> attrs1;
  attrs1["epsilon"] = static_cast<float>(0.001);

  auto c = program.conv2d(A, B, attrs);
  auto d =
      program.fused_batchnorm_inference(c, Scale, Bias, Mean, Variance, attrs1);

  Target target = common::DefaultTarget();
  program.SetInputs({A, B, Scale, Bias, Mean, Variance});
  program.Validate();
  LOG(INFO) << "Program:\n" << program;
  auto graph = std::make_shared<hlir::framework::Graph>(program, target);
  LOG(INFO) << "graph:\n" << graph->Visualize();

  hlir::framework::ApplyPass(graph.get(), "InferShape");
  hlir::framework::ApplyPass(graph.get(), "OpFusion");
  auto scope = BuildScope(target, graph);

  hlir::framework::CompilationContext context(graph, scope, target);
  hlir::framework::GraphCompiler gc(context);
  auto runtime_program = gc.Build();

  scope->Var<hlir::framework::Tensor>("A");
  scope->Var<hlir::framework::Tensor>("B");
  scope->Var<hlir::framework::Tensor>("C");
  scope->Var<hlir::framework::Tensor>("D");

  auto A1 = scope->GetTensor("A");
  auto B1 = scope->GetTensor("B");
  auto C1 = scope->GetTensor("C");
  auto D1 = scope->GetTensor("D");
  SetRandData<float>(A1, target);
  SetRandData<float>(B1, target);
  SetRandData<float>(C1, target);
  SetRandData<float>(D1, target);

  runtime_program->Execute();
}

}  // namespace frontend
}  // namespace cinn
