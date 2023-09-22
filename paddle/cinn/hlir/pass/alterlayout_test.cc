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

std::unique_ptr<Program> CreateAddProgram() {
  const int M = 32;
  const int N = 24;

  Placeholder a(Float(32), {M, N});
  Placeholder b(Float(32), {M, N});
  std::unique_ptr<Program> program(new Program);

  auto c = program->add(a, b);
  auto d = program->add(a, c);

  program->SetInputs({a, b});
  program->Validate();

  return program;
}

TEST(conv, conv) {
  Placeholder A(Float(32), {1, 3, 224, 224}, "A");
  Placeholder B(Float(32), {64, 3, 7, 7}, "B");
  Placeholder C(Float(32), {1, 64, 112, 112}, "C");

  Program program;
  absl::flat_hash_map<std::string, Program::attr_t> attrs;
  attrs["stride"] = std::vector<int>({2, 2});
  attrs["dilation"] = std::vector<int>({1, 1});
  attrs["padding"] = std::vector<int>({3, 3});
  std::string src_layout = "NCHW";
  attrs["data_format"] = src_layout;

  auto c = program.conv2d(A, B, attrs);

  Target target = common::DefaultHostTarget();
  program.SetInputs({A, B});
  program.Validate();
  LOG(INFO) << "Program:\n" << program;
  auto graph = std::make_shared<hlir::framework::Graph>(program, target);

  hlir::framework::ApplyPass(graph.get(), "InferShape");
  hlir::framework::ApplyPass(graph.get(), "AlterLayout");
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

TEST(conv_relu_conv, conv_relu_conv) {
  Placeholder A(Float(32), {1, 3, 224, 224}, "A");
  Placeholder B(Float(32), {64, 3, 7, 7}, "B");
  Placeholder C(Float(32), {1, 64, 112, 112}, "C");
  Placeholder D(Float(32), {64, 64, 7, 7}, "D");

  Program program;
  absl::flat_hash_map<std::string, Program::attr_t> attrs;
  attrs["stride"] = std::vector<int>({2, 2});
  attrs["dilation"] = std::vector<int>({1, 1});
  attrs["padding"] = std::vector<int>({3, 3});
  std::string src_layout = "NCHW";
  attrs["data_format"] = src_layout;

  auto c = program.conv2d(A, B, attrs);
  auto d = program.relu(c);
  auto e = program.conv2d(d, D, attrs);

  Target target = common::DefaultHostTarget();
  program.SetInputs({A, B, D});
  program.Validate();
  LOG(INFO) << "Program:\n" << program;
  auto graph = std::make_shared<hlir::framework::Graph>(program, target);

  hlir::framework::ApplyPass(graph.get(), "InferShape");
  hlir::framework::ApplyPass(graph.get(), "AlterLayout");
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

TEST(conv_add_conv, conv_add_conv) {
  Placeholder A(Float(32), {1, 3, 224, 224}, "A");
  Placeholder B(Float(32), {64, 3, 7, 7}, "B");
  Placeholder C(Float(32), {64}, "C");
  Placeholder D(Float(32), {64, 64, 7, 7}, "D");

  Program program;
  absl::flat_hash_map<std::string, Program::attr_t> attrs;
  attrs["stride"] = std::vector<int>({2, 2});
  attrs["dilation"] = std::vector<int>({1, 1});
  attrs["padding"] = std::vector<int>({3, 3});
  std::string src_layout = "NCHW";
  attrs["data_format"] = src_layout;

  auto c = program.conv2d(A, B, attrs);
  auto d = program.elementwise_add(c, C, 1);
  auto e = program.conv2d(d, D, attrs);

  Target target = common::DefaultHostTarget();
  program.SetInputs({A, B, D});
  program.Validate();
  LOG(INFO) << "Program:\n" << program;
  auto graph = std::make_shared<hlir::framework::Graph>(program, target);

  hlir::framework::ApplyPass(graph.get(), "InferShape");
  hlir::framework::ApplyPass(graph.get(), "AlterLayout");
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

TEST(conv_bn_conv, conv_bn_conv) {
  Placeholder A(Float(32), {1, 3, 224, 224}, "A");
  Placeholder B(Float(32), {64, 3, 7, 7}, "B");
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
  attrs1["epsilon"] = 0.001f;

  auto c = program.conv2d(A, B, attrs);
  auto d = program.batchnorm(c, Scale, Bias, Mean, Variance, attrs1);
  auto e = program.conv2d(d, D, attrs);

  Target target = common::DefaultHostTarget();
  program.SetInputs({A, B, D});
  program.Validate();
  LOG(INFO) << "Program:\n" << program;
  auto graph = std::make_shared<hlir::framework::Graph>(program, target);

  hlir::framework::ApplyPass(graph.get(), "InferShape");
  hlir::framework::ApplyPass(graph.get(), "AlterLayout");
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

TEST(conv_pool2d_conv, conv_pool2d_conv) {
  Placeholder A(Float(32), {1, 3, 224, 224}, "A");
  Placeholder B(Float(32), {64, 3, 7, 7}, "B");
  Placeholder C(Float(32), {1, 64, 112, 112}, "C");
  Placeholder D(Float(32), {64, 64, 7, 7}, "D");

  Program program;
  absl::flat_hash_map<std::string, Program::attr_t> attrs;
  attrs["stride"] = std::vector<int>({2, 2});
  attrs["dilation"] = std::vector<int>({1, 1});
  attrs["padding"] = std::vector<int>({3, 3});
  std::string src_layout = "NCHW";
  attrs["data_format"] = src_layout;

  absl::flat_hash_map<std::string, Program::attr_t> attrs2;
  attrs2["stride_size"] = std::vector<int>({2, 2});
  attrs2["padding_size"] = std::vector<int>({1, 1, 1, 1});
  attrs2["kernel_size"] = std::vector<int>({3, 3});
  std::string pool_type = "max";
  attrs2["pool_type"] = pool_type;

  auto c = program.conv2d(A, B, attrs);
  auto d = program.pool2d(c, attrs2);
  auto e = program.conv2d(d, D, attrs);

  Target target = common::DefaultHostTarget();
  program.SetInputs({A, B, D});
  program.Validate();
  LOG(INFO) << "Program:\n" << program;
  auto graph = std::make_shared<hlir::framework::Graph>(program, target);

  hlir::framework::ApplyPass(graph.get(), "InferShape");
  hlir::framework::ApplyPass(graph.get(), "AlterLayout");
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

TEST(conv_softmax_conv, conv_softmax_conv) {
  Placeholder A(Float(32), {1, 3, 224, 224}, "A");
  Placeholder B(Float(32), {64, 3, 7, 7}, "B");
  Placeholder D(Float(32), {64, 64, 7, 7}, "D");

  Program program;
  absl::flat_hash_map<std::string, Program::attr_t> attrs;
  attrs["stride"] = std::vector<int>({2, 2});
  attrs["dilation"] = std::vector<int>({1, 1});
  attrs["padding"] = std::vector<int>({3, 3});
  std::string src_layout = "NCHW";
  attrs["data_format"] = src_layout;

  absl::flat_hash_map<std::string, Program::attr_t> attrs1;
  attrs1["axis"] = static_cast<int>(-1);

  auto c = program.conv2d(A, B, attrs);
  auto d = program.softmax(c, attrs1);
  auto e = program.conv2d(d, D, attrs);

  Target target = common::DefaultHostTarget();
  program.SetInputs({A, B, D});
  program.Validate();
  LOG(INFO) << "Program:\n" << program;
  auto graph = std::make_shared<hlir::framework::Graph>(program, target);

  hlir::framework::ApplyPass(graph.get(), "InferShape");
  hlir::framework::ApplyPass(graph.get(), "AlterLayout");
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

TEST(conv_sigmoid_conv, conv_sigmoid_conv) {
  Placeholder A(Float(32), {1, 3, 224, 224}, "A");
  Placeholder B(Float(32), {64, 3, 7, 7}, "B");
  Placeholder D(Float(32), {64, 64, 7, 7}, "D");

  Program program;
  absl::flat_hash_map<std::string, Program::attr_t> attrs;
  attrs["stride"] = std::vector<int>({2, 2});
  attrs["dilation"] = std::vector<int>({1, 1});
  attrs["padding"] = std::vector<int>({3, 3});
  std::string src_layout = "NCHW";
  attrs["data_format"] = src_layout;

  auto c = program.conv2d(A, B, attrs);
  auto d = program.sigmoid(c);
  auto e = program.conv2d(d, D, attrs);

  Target target = common::DefaultHostTarget();
  program.SetInputs({A, B, D});
  program.Validate();
  LOG(INFO) << "Program:\n" << program;
  auto graph = std::make_shared<hlir::framework::Graph>(program, target);

  hlir::framework::ApplyPass(graph.get(), "InferShape");
  hlir::framework::ApplyPass(graph.get(), "AlterLayout");
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

TEST(conv_mul_conv, conv_mul_conv) {
  Placeholder A(Float(32), {3, 3, 224, 224}, "A");
  Placeholder B(Float(32), {64, 3, 7, 7}, "B");
  Placeholder C(Float(32), {1, 64, 112, 112}, "C");
  Placeholder D(Float(32), {64, 64, 7, 7}, "D");

  Program program;
  absl::flat_hash_map<std::string, Program::attr_t> attrs;
  attrs["stride"] = std::vector<int>({2, 2});
  attrs["dilation"] = std::vector<int>({1, 1});
  attrs["padding"] = std::vector<int>({3, 3});
  std::string src_layout = "NCHW";
  attrs["data_format"] = src_layout;

  absl::flat_hash_map<std::string, Program::attr_t> attrs1;
  attrs1["axis"] = static_cast<int>(-1);

  auto c = program.conv2d(A, B, attrs);
  auto d = program.mul(c, C, 1, 1);
  auto e = program.softmax(d, attrs1);

  Target target = common::DefaultHostTarget();
  program.SetInputs({A, B, D});
  program.Validate();
  LOG(INFO) << "Program:\n" << program;
  auto graph = std::make_shared<hlir::framework::Graph>(program, target);

  hlir::framework::ApplyPass(graph.get(), "InferShape");
  hlir::framework::ApplyPass(graph.get(), "AlterLayout");
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

}  // namespace frontend
}  // namespace cinn
