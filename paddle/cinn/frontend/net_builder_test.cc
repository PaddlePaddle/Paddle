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

#include "paddle/cinn/frontend/net_builder.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <memory>
#include <random>
#include <vector>

#include "paddle/cinn/common/target.h"
#include "paddle/cinn/frontend/optimize.h"
#include "paddle/cinn/frontend/syntax.h"
#include "paddle/cinn/hlir/framework/graph.h"
#include "paddle/cinn/hlir/framework/graph_compiler.h"
#include "paddle/cinn/hlir/framework/graph_compiler_util.h"
#include "paddle/cinn/hlir/framework/tensor.h"
#include "paddle/cinn/hlir/op/use_ops.h"
#include "paddle/cinn/utils/data_util.h"
#ifdef CINN_WITH_CUDA
#include <cuda_runtime.h>
#endif

namespace cinn {
namespace frontend {

using hlir::framework::OpRegistry;

namespace {
Program CreateAddProgram() {
  constexpr int M = 32;
  constexpr int N = 24;

  NetBuilder builder("net_builder");
  auto a = builder.CreateInput(Float(32), {M, N}, "A");
  auto b = builder.CreateInput(Float(32), {M, N}, "B");
  auto c = builder.Add(a, b);
  auto d = builder.Add(a, c);
  auto program = builder.Build();

  return program;
}

template <typename T, typename Alloc = std::allocator<T>>
std::ostream& operator<<(std::ostream& os, const std::vector<T, Alloc>& vec) {
  os << "{ ";
  for (auto e : vec) {
    os << e << " ";
  }
  os << "}\n";
  return os;
}
}  // namespace

TEST(net_build, basic) {
  LOG(INFO) << "The size of registered operators: "
            << OpRegistry::Global()->ListAllNames().size();
  LOG(INFO) << "Registered operators:\n"
            << OpRegistry::Global()->ListAllNames();
  auto program = CreateAddProgram();
  // output program
  for (int i = 0; i < program.size(); i++) {
    LOG(INFO) << "instruction: " << program[i];
  }
}

TEST(net_build, TestTransValidVarName) {
  std::string a_val_id = "A";
  std::string b_val_id = "B___";
  frontend::NetBuilder builder("net_builder");
  auto a = builder.CreateInput(Float(32), {1, 64, 112, 112}, "@A");
  auto b = builder.CreateInput(Float(32), {64}, "B/");
  EXPECT_EQ(a.id(), a_val_id);
  EXPECT_EQ(b.id(), b_val_id);
}

TEST(net_build, program_execute_multi_elementwise_add) {
  auto program = CreateAddProgram();
#ifdef CINN_WITH_CUDA
  Target target = common::DefaultNVGPUTarget();
#else
  Target target = common::DefaultHostTarget();
#endif

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
#ifdef CINN_WITH_CUDA
TEST(net_build, program_execute_fc) {
  constexpr int B = 10;  // batch size
  constexpr int M = 32;
  constexpr int K = 18;
  constexpr int N = 24;

  NetBuilder builder("net_builder");
  auto a = builder.CreateInput(Float(32), {B * M, K}, "A");
  auto w = builder.CreateInput(Float(32), {K, N}, "W");  // weight
  auto b = builder.CreateInput(Float(32), {N}, "B");     // bias

  auto mul_out = builder.Matmul(a, w);
  auto add_out = builder.Add(mul_out, b);
  auto program = builder.Build();

#ifdef CINN_WITH_CUDA
  Target target = common::DefaultNVGPUTarget();
#else
  Target target = common::DefaultHostTarget();
#endif

  std::unordered_set<std::string> fetch_ids;
  auto graph = Optimize(&program, fetch_ids, target);
  LOG(INFO) << "graph:\n" << graph->Visualize();

  auto scope = BuildScope(target, graph);
  hlir::framework::CompilationContext context(graph, scope, target);
  hlir::framework::GraphCompiler gc(context);
  auto runtime_program = gc.Build();

  scope->Var<hlir::framework::Tensor>(std::string(a.id()));
  scope->Var<hlir::framework::Tensor>(std::string(w.id()));
  scope->Var<hlir::framework::Tensor>(std::string(b.id()));
  scope->Var<hlir::framework::Tensor>(std::string(mul_out->id));

  auto a_ten = scope->GetTensor(std::string(a.id()));
  auto w_ten = scope->GetTensor(std::string(w.id()));
  auto b_ten = scope->GetTensor(std::string(b.id()));
  auto fake_out_ten = scope->GetTensor(std::string(mul_out->id));
  auto add_out_ten = scope->GetTensor(std::string(add_out->id));
  SetRandData<float>(a_ten, target);
  SetRandData<float>(w_ten, target);
  SetRandData<float>(b_ten, target);

  runtime_program->Execute();
}
#endif

#ifdef CINN_WITH_CUDA
TEST(net_build, program_execute_multi_elementwise_add_bf16) {
  constexpr int M = 32;
  constexpr int N = 24;

  NetBuilder builder("net_builder");
  auto a = builder.CreateInput(cinn::common::BFloat16(), {M, N}, "A");
  auto b = builder.CreateInput(cinn::common::BFloat16(), {M, N}, "B");
  auto c = builder.Add(a, b);
  auto d = builder.Add(a, c);
  auto program = builder.Build();

#ifdef CINN_WITH_CUDA
  Target target = common::DefaultNVGPUTarget();
#else
  Target target = common::DefaultHostTarget();
#endif

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

TEST(net_build, program_execute_fc_bf16) {
  constexpr int B = 10;  // batch size
  constexpr int M = 32;
  constexpr int K = 18;
  constexpr int N = 24;

  NetBuilder builder("net_builder");
  auto a = builder.CreateInput(cinn::common::BFloat16(), {B * M, K}, "A");
  auto w =
      builder.CreateInput(cinn::common::BFloat16(), {K, N}, "W");    // weight
  auto b = builder.CreateInput(cinn::common::BFloat16(), {N}, "B");  // bias

  auto mul_out = builder.Matmul(a, w);
  auto add_out = builder.Add(mul_out, b);
  auto program = builder.Build();

#ifdef CINN_WITH_CUDA
  Target target = common::DefaultNVGPUTarget();
#else
  Target target = common::DefaultHostTarget();
#endif

  std::unordered_set<std::string> fetch_ids;
  auto graph = Optimize(&program, fetch_ids, target);
  LOG(INFO) << "graph:\n" << graph->Visualize();

  auto scope = BuildScope(target, graph);
  hlir::framework::CompilationContext context(graph, scope, target);
  hlir::framework::GraphCompiler gc(context);
  auto runtime_program = gc.Build();

  scope->Var<hlir::framework::Tensor>(std::string(a.id()));
  scope->Var<hlir::framework::Tensor>(std::string(w.id()));
  scope->Var<hlir::framework::Tensor>(std::string(b.id()));
  scope->Var<hlir::framework::Tensor>(std::string(mul_out->id));

  auto a_ten = scope->GetTensor(std::string(a.id()));
  auto w_ten = scope->GetTensor(std::string(w.id()));
  auto b_ten = scope->GetTensor(std::string(b.id()));
  auto fake_out_ten = scope->GetTensor(std::string(mul_out->id));
  auto add_out_ten = scope->GetTensor(std::string(add_out->id));
  SetRandData<float>(a_ten, target);
  SetRandData<float>(w_ten, target);
  SetRandData<float>(b_ten, target);

  runtime_program->Execute();
}
#endif

TEST(net_build, program_execute_pool2d) {
  const int B = 16;
  const int C = 64;
  const int H = 112;
  const int W = 112;

  NetBuilder builder("net_builder");
  Placeholder input = builder.CreateInput(Float(32), {B, C, H, W}, "Img");
  std::string pooling_type = "max";
  std::vector<int> ksize{3, 3};
  std::vector<int> strides{2, 2};
  std::vector<int> paddings{1, 1, 1, 1};
  bool ceil_mode = false;
  bool exclusive = true;
  bool global_pooling = false;
  std::string data_format = "NCHW";
  bool adaptive = false;
  std::string padding_algorithm = "EXPLICIT";
  Variable pool_out = builder.Pool2d(input,
                                     pooling_type,
                                     ksize,
                                     strides,
                                     paddings,
                                     ceil_mode,
                                     exclusive,
                                     global_pooling,
                                     data_format,
                                     adaptive,
                                     padding_algorithm);
  auto program = builder.Build();

#ifdef CINN_WITH_CUDA
  Target target = common::DefaultNVGPUTarget();
#else
  Target target = common::DefaultHostTarget();
#endif

  std::unordered_set<std::string> fetch_ids;
  auto graph = Optimize(&program, fetch_ids, target);
  auto scope = BuildScope(target, graph);
  hlir::framework::CompilationContext context(graph, scope, target);
  hlir::framework::GraphCompiler gc(context);
  auto runtime_program = gc.Build();

  scope->Var<hlir::framework::Tensor>(std::string(input.id()));
  scope->Var<hlir::framework::Tensor>(std::string(pool_out->id));

  auto input_tensor = scope->GetTensor(std::string(input.id()));
  SetRandData<float>(input_tensor, target);
  runtime_program->Execute();
}

TEST(net_build, program_execute_reverse) {
  const int B = 16;
  const int C = 3;
  const int H = 224;
  const int W = 224;

  NetBuilder builder("net_builder");
  Placeholder input = builder.CreateInput(Float(32), {B, C, H, W}, "Img");
  Variable reverse_out = builder.Reverse(input, {2, 3});
  auto program = builder.Build();

#ifdef CINN_WITH_CUDA
  Target target = common::DefaultNVGPUTarget();
#else
  Target target = common::DefaultHostTarget();
#endif

  std::unordered_set<std::string> fetch_ids;
  auto graph = Optimize(&program, fetch_ids, target);
  LOG(INFO) << "graph:\n" << graph->Visualize();

  auto scope = BuildScope(target, graph);
  hlir::framework::CompilationContext context(graph, scope, target);
  hlir::framework::GraphCompiler gc(context);
  auto runtime_program = gc.Build();

  scope->Var<hlir::framework::Tensor>(std::string(input.id()));
  scope->Var<hlir::framework::Tensor>(std::string(reverse_out->id));

  auto input_tensor = scope->GetTensor(std::string(input.id()));
  SetRandData<float>(input_tensor, target);
  runtime_program->Execute();
}

TEST(net_build, program_execute_gather) {
  const int B = 4;
  const int H_IN1 = 18;
  const int H_IN2 = 14;

  NetBuilder builder("net_builder");
  Placeholder input1 = builder.CreateInput(Float(32), {B, H_IN1}, "In1");
  Placeholder input2 = builder.CreateInput(Int(32), {H_IN2}, "In2");
  Variable output = builder.Gather(input1, input2, 1);
  auto program = builder.Build();

#ifdef CINN_WITH_CUDA
  Target target = common::DefaultNVGPUTarget();
#else
  Target target = common::DefaultHostTarget();
#endif
  std::unordered_set<std::string> fetch_ids;
  auto graph = Optimize(&program, fetch_ids, target);

  auto scope = BuildScope(target, graph);
  hlir::framework::CompilationContext context(graph, scope, target);
  hlir::framework::GraphCompiler gc(context);
  auto runtime_program = gc.Build();

  scope->Var<hlir::framework::Tensor>(std::string(input1.id()));
  scope->Var<hlir::framework::Tensor>(std::string(input2.id()));
  scope->Var<hlir::framework::Tensor>(std::string(output->id));

  auto input1_tensor = scope->GetTensor(std::string(input1.id()));
  SetRandData<float>(input1_tensor, target);
  std::vector<float> input1_data = GetTensorData<float>(input1_tensor, target);

  auto input2_tensor = scope->GetTensor(std::string(input2.id()));
  SetRandInt(input2_tensor, target, -1, 0, H_IN1);
  std::vector<int> input2_data = GetTensorData<int>(input2_tensor, target);

  runtime_program->Execute();

  auto output_tensor = scope->GetTensor(std::string(output->id));
  const std::vector<int>& output_shape = output_tensor->shape().data();
  EXPECT_EQ(output_tensor->type(), Float(32));
  EXPECT_EQ(output_shape.size(), 2UL);
  EXPECT_EQ(output_shape[0], B);
  EXPECT_EQ(output_shape[1], H_IN2);

  std::vector<float> output_data = GetTensorData<float>(output_tensor, target);
  VLOG(6) << "Visualize output_data";
  for (int b = 0; b < B; ++b) {
    for (int h = 0; h < H_IN2; ++h) {
      std::string line;
      int index = h + H_IN2 * b;
      float in_data = input1_data[input2_data[h] + H_IN1 * b];
      float out_data = output_data[index];
      line += (std::to_string(out_data) + ", ");
      EXPECT_EQ(in_data, out_data);
      VLOG(6) << line;
    }
  }
}

TEST(net_build, program_execute_gather_nd) {
  const int B = 4;
  const int H_IN1 = 11;
  const int H_IN2 = 14;

  NetBuilder builder("net_builder");
  Placeholder input1 = builder.CreateInput(Float(32), {B, H_IN1}, "In1");
  Placeholder input2 = builder.CreateInput(Int(32), {B, H_IN2, 1}, "In2");
  Variable output = builder.GatherNd(input1, input2);
  auto program = builder.Build();

#ifdef CINN_WITH_CUDA
  Target target = common::DefaultNVGPUTarget();
#else
  Target target = common::DefaultHostTarget();
#endif

  std::unordered_set<std::string> fetch_ids;
  auto graph = Optimize(&program, fetch_ids, target);

  auto scope = BuildScope(target, graph);
  hlir::framework::CompilationContext context(graph, scope, target);
  hlir::framework::GraphCompiler gc(context);
  auto runtime_program = gc.Build();

  scope->Var<hlir::framework::Tensor>(std::string(input1.id()));
  scope->Var<hlir::framework::Tensor>(std::string(input2.id()));
  scope->Var<hlir::framework::Tensor>(std::string(output->id));

  auto input1_tensor = scope->GetTensor(std::string(input1.id()));
  SetRandData<float>(input1_tensor, target);
  std::vector<float> input1_data = GetTensorData<float>(input1_tensor, target);

  auto input2_tensor = scope->GetTensor(std::string(input2.id()));
  SetRandInt(input2_tensor, target, -1, 0, B);
  std::vector<int> input2_data = GetTensorData<int>(input2_tensor, target);

  runtime_program->Execute();

  auto output_tensor = scope->GetTensor(std::string(output->id));
  const std::vector<int>& output_shape = output_tensor->shape().data();
  EXPECT_EQ(output_tensor->type(), Float(32));
  EXPECT_EQ(output_shape.size(), 3UL);
  EXPECT_EQ(output_shape[0], B);
  EXPECT_EQ(output_shape[1], H_IN2);
  EXPECT_EQ(output_shape[2], H_IN1);

  std::vector<float> output_data = GetTensorData<float>(output_tensor, target);
  VLOG(6) << "Visualize output_data";
  for (int b = 0; b < B; ++b) {
    for (int h = 0; h < H_IN2; ++h) {
      std::string line;
      for (int c = 0; c < H_IN1; ++c) {
        float in_data = input1_data[input2_data[b * H_IN2 + h] * H_IN1 + c];
        int out_index = c + h * H_IN1 + H_IN1 * H_IN2 * b;
        float out_data = output_data[out_index];
        line += (std::to_string(out_data) + ", ");
        EXPECT_EQ(in_data, out_data);
      }
      VLOG(6) << line;
    }
  }
}

TEST(net_build, program_execute_cast) {
  const int B = 4;
  const int H = 7;

  NetBuilder builder("net_builder");
  Placeholder input = builder.CreateInput(Int(32), {B, H}, "In");
  Variable output = builder.Cast(input, "float");
  auto program = builder.Build();

#ifdef CINN_WITH_CUDA
  Target target = common::DefaultNVGPUTarget();
#else
  Target target = common::DefaultHostTarget();
#endif
  std::unordered_set<std::string> fetch_ids;
  auto graph = Optimize(&program, fetch_ids, target);

  auto scope = BuildScope(target, graph);
  hlir::framework::CompilationContext context(graph, scope, target);
  hlir::framework::GraphCompiler gc(context);
  auto runtime_program = gc.Build();

  scope->Var<hlir::framework::Tensor>(std::string(input.id()));
  scope->Var<hlir::framework::Tensor>(std::string(output->id));

  auto input_tensor = scope->GetTensor(std::string(input.id()));
  SetRandInt(input_tensor, target);
  std::vector<int> input_data = GetTensorData<int>(input_tensor, target);

  runtime_program->Execute();

  auto output_tensor = scope->GetTensor(std::string(output->id));
  const std::vector<int>& output_shape = output_tensor->shape().data();
  EXPECT_EQ(output_tensor->type(), Float(32));
  EXPECT_EQ(output_shape.size(), 2UL);
  EXPECT_EQ(output_shape[0], B);
  EXPECT_EQ(output_shape[1], H);

  std::vector<float> output_data = GetTensorData<float>(output_tensor, target);
  VLOG(6) << "Visualize output_data";
  for (int b = 0; b < B; ++b) {
    for (int h = 0; h < H; ++h) {
      std::string line;
      int index = h + H * b;
      float in_data = static_cast<float>(input_data[index]);
      float out_data = output_data[index];
      line += (std::to_string(out_data) + ", ");
      EXPECT_EQ(in_data, out_data);
      VLOG(6) << line;
    }
  }
}

TEST(net_build, program_execute_squeeze_case0) {
  const int B = 4;
  const int C = 1;
  const int H = 7;
  const int W = 1;

  NetBuilder builder("net_builder");
  Placeholder input = builder.CreateInput(Float(32), {B, C, H, W}, "In");
  Variable output = builder.Squeeze(input, {1});
  auto program = builder.Build();

#ifdef CINN_WITH_CUDA
  Target target = common::DefaultNVGPUTarget();
#else
  Target target = common::DefaultHostTarget();
#endif
  std::unordered_set<std::string> fetch_ids;
  auto graph = Optimize(&program, fetch_ids, target);

  auto scope = BuildScope(target, graph);
  hlir::framework::CompilationContext context(graph, scope, target);
  hlir::framework::GraphCompiler gc(context);
  auto runtime_program = gc.Build();

  scope->Var<hlir::framework::Tensor>(std::string(input.id()));
  scope->Var<hlir::framework::Tensor>(std::string(output->id));

  auto input_tensor = scope->GetTensor(std::string(input.id()));
  SetRandData<float>(input_tensor, target);
  std::vector<float> input_data = GetTensorData<float>(input_tensor, target);

  runtime_program->Execute();

  auto output_tensor = scope->GetTensor(std::string(output->id));
  const std::vector<int>& output_shape = output_tensor->shape().data();
  EXPECT_EQ(output_shape.size(), 3UL);
  EXPECT_EQ(output_shape[0], B);
  EXPECT_EQ(output_shape[1], H);
  EXPECT_EQ(output_shape[2], W);

  std::vector<float> output_data = GetTensorData<float>(output_tensor, target);
  VLOG(6) << "Visualize output_data";
  for (int b = 0; b < B; ++b) {
    for (int c = 0; c < C; ++c) {
      VLOG(6) << "b = " << b << ", c = " << c;
      for (int h = 0; h < H; ++h) {
        std::string line;
        for (int w = 0; w < W; ++w) {
          int index = w + W * (h + H * (c + C * b));
          float in_data = input_data[index];
          float out_data = output_data[index];
          line += (std::to_string(out_data) + ", ");
          EXPECT_EQ(in_data, out_data);
        }
        VLOG(6) << line;
      }
    }
  }
}

TEST(net_build, program_execute_squeeze_case1) {
  const int B = 4;
  const int C = 1;
  const int H = 7;
  const int W = 1;

  NetBuilder builder("net_builder");
  Placeholder input = builder.CreateInput(Float(32), {B, C, H, W}, "In");
  Variable output = builder.Squeeze(input, {-3});
  auto program = builder.Build();

#ifdef CINN_WITH_CUDA
  Target target = common::DefaultNVGPUTarget();
#else
  Target target = common::DefaultHostTarget();
#endif
  std::unordered_set<std::string> fetch_ids;
  auto graph = Optimize(&program, fetch_ids, target);

  auto scope = BuildScope(target, graph);
  hlir::framework::CompilationContext context(graph, scope, target);
  hlir::framework::GraphCompiler gc(context);
  auto runtime_program = gc.Build();

  scope->Var<hlir::framework::Tensor>(std::string(input.id()));
  scope->Var<hlir::framework::Tensor>(std::string(output->id));

  auto input_tensor = scope->GetTensor(std::string(input.id()));
  SetRandData<float>(input_tensor, target);
  std::vector<float> input_data = GetTensorData<float>(input_tensor, target);

  runtime_program->Execute();

  auto output_tensor = scope->GetTensor(std::string(output->id));
  const std::vector<int>& output_shape = output_tensor->shape().data();
  EXPECT_EQ(output_shape.size(), 3UL);
  EXPECT_EQ(output_shape[0], B);
  EXPECT_EQ(output_shape[1], H);
  EXPECT_EQ(output_shape[2], W);

  std::vector<float> output_data = GetTensorData<float>(output_tensor, target);
  VLOG(6) << "Visualize output_data";
  for (int b = 0; b < B; ++b) {
    for (int c = 0; c < C; ++c) {
      VLOG(6) << "b = " << b << ", c = " << c;
      for (int h = 0; h < H; ++h) {
        std::string line;
        for (int w = 0; w < W; ++w) {
          int index = w + W * (h + H * (c + C * b));
          float in_data = input_data[index];
          float out_data = output_data[index];
          line += (std::to_string(out_data) + ", ");
          EXPECT_EQ(in_data, out_data);
        }
        VLOG(6) << line;
      }
    }
  }
}

TEST(net_build, program_execute_squeeze_case2) {
  const int B = 4;
  const int C = 1;
  const int H = 7;
  const int W = 1;

  NetBuilder builder("net_builder");
  Placeholder input = builder.CreateInput(Float(32), {B, C, H, W}, "In");
  Variable output = builder.Squeeze(input, {1, 3});
  auto program = builder.Build();

#ifdef CINN_WITH_CUDA
  Target target = common::DefaultNVGPUTarget();
#else
  Target target = common::DefaultHostTarget();
#endif
  std::unordered_set<std::string> fetch_ids;
  auto graph = Optimize(&program, fetch_ids, target);

  auto scope = BuildScope(target, graph);
  hlir::framework::CompilationContext context(graph, scope, target);
  hlir::framework::GraphCompiler gc(context);
  auto runtime_program = gc.Build();

  scope->Var<hlir::framework::Tensor>(std::string(input.id()));
  scope->Var<hlir::framework::Tensor>(std::string(output->id));

  auto input_tensor = scope->GetTensor(std::string(input.id()));
  SetRandData<float>(input_tensor, target);
  std::vector<float> input_data = GetTensorData<float>(input_tensor, target);

  runtime_program->Execute();

  auto output_tensor = scope->GetTensor(std::string(output->id));
  const std::vector<int>& output_shape = output_tensor->shape().data();
  EXPECT_EQ(output_shape.size(), 2UL);
  EXPECT_EQ(output_shape[0], B);
  EXPECT_EQ(output_shape[1], H);

  std::vector<float> output_data = GetTensorData<float>(output_tensor, target);
  VLOG(6) << "Visualize output_data";
  for (int b = 0; b < B; ++b) {
    for (int c = 0; c < C; ++c) {
      VLOG(6) << "b = " << b << ", c = " << c;
      for (int h = 0; h < H; ++h) {
        std::string line;
        for (int w = 0; w < W; ++w) {
          int index = w + W * (h + H * (c + C * b));
          float in_data = input_data[index];
          float out_data = output_data[index];
          line += (std::to_string(out_data) + ", ");
          EXPECT_EQ(in_data, out_data);
        }
        VLOG(6) << line;
      }
    }
  }
}

TEST(net_build, program_execute_squeeze_case3) {
  const int B = 4;
  const int C = 1;
  const int H = 7;
  const int W = 1;

  NetBuilder builder("net_builder");
  Placeholder input = builder.CreateInput(Float(32), {B, C, H, W}, "In");
  Variable output = builder.Squeeze(input, {1, -1});
  auto program = builder.Build();

#ifdef CINN_WITH_CUDA
  Target target = common::DefaultNVGPUTarget();
#else
  Target target = common::DefaultHostTarget();
#endif
  std::unordered_set<std::string> fetch_ids;
  auto graph = Optimize(&program, fetch_ids, target);

  auto scope = BuildScope(target, graph);
  hlir::framework::CompilationContext context(graph, scope, target);
  hlir::framework::GraphCompiler gc(context);
  auto runtime_program = gc.Build();

  scope->Var<hlir::framework::Tensor>(std::string(input.id()));
  scope->Var<hlir::framework::Tensor>(std::string(output->id));

  auto input_tensor = scope->GetTensor(std::string(input.id()));
  SetRandData<float>(input_tensor, target);
  std::vector<float> input_data = GetTensorData<float>(input_tensor, target);

  runtime_program->Execute();

  auto output_tensor = scope->GetTensor(std::string(output->id));
  const std::vector<int>& output_shape = output_tensor->shape().data();
  EXPECT_EQ(output_shape.size(), 2UL);
  EXPECT_EQ(output_shape[0], B);
  EXPECT_EQ(output_shape[1], H);

  std::vector<float> output_data = GetTensorData<float>(output_tensor, target);
  VLOG(6) << "Visualize output_data";
  for (int b = 0; b < B; ++b) {
    for (int c = 0; c < C; ++c) {
      VLOG(6) << "b = " << b << ", c = " << c;
      for (int h = 0; h < H; ++h) {
        std::string line;
        for (int w = 0; w < W; ++w) {
          int index = w + W * (h + H * (c + C * b));
          float in_data = input_data[index];
          float out_data = output_data[index];
          line += (std::to_string(out_data) + ", ");
          EXPECT_EQ(in_data, out_data);
        }
        VLOG(6) << line;
      }
    }
  }
}

TEST(net_build, program_execute_squeeze_case4) {
  const int B = 4;
  const int C = 1;
  const int H = 7;
  const int W = 1;

  NetBuilder builder("net_builder");
  Placeholder input = builder.CreateInput(Float(32), {B, C, H, W}, "In");
  Variable output = builder.Squeeze(input, {});
  auto program = builder.Build();

#ifdef CINN_WITH_CUDA
  Target target = common::DefaultNVGPUTarget();
#else
  Target target = common::DefaultHostTarget();
#endif
  std::unordered_set<std::string> fetch_ids;
  auto graph = Optimize(&program, fetch_ids, target);

  auto scope = BuildScope(target, graph);
  hlir::framework::CompilationContext context(graph, scope, target);
  hlir::framework::GraphCompiler gc(context);
  auto runtime_program = gc.Build();

  scope->Var<hlir::framework::Tensor>(std::string(input.id()));
  scope->Var<hlir::framework::Tensor>(std::string(output->id));

  auto input_tensor = scope->GetTensor(std::string(input.id()));
  SetRandData<float>(input_tensor, target);
  std::vector<float> input_data = GetTensorData<float>(input_tensor, target);

  runtime_program->Execute();

  auto output_tensor = scope->GetTensor(std::string(output->id));
  const std::vector<int>& output_shape = output_tensor->shape().data();
  EXPECT_EQ(output_shape.size(), 2UL);
  EXPECT_EQ(output_shape[0], B);
  EXPECT_EQ(output_shape[1], H);

  std::vector<float> output_data = GetTensorData<float>(output_tensor, target);
  VLOG(6) << "Visualize output_data";
  for (int b = 0; b < B; ++b) {
    for (int c = 0; c < C; ++c) {
      VLOG(6) << "b = " << b << ", c = " << c;
      for (int h = 0; h < H; ++h) {
        std::string line;
        for (int w = 0; w < W; ++w) {
          int index = w + W * (h + H * (c + C * b));
          float in_data = input_data[index];
          float out_data = output_data[index];
          line += (std::to_string(out_data) + ", ");
          EXPECT_EQ(in_data, out_data);
        }
        VLOG(6) << line;
      }
    }
  }
}

TEST(net_build, program_execute_argsort) {
  const int B = 4;
  const int H = 7;

  NetBuilder builder("net_builder");
  Placeholder input = builder.CreateInput(Float(32), {B, H}, "In");
  Variable output = builder.ArgSort(input, 0, true).at(0);
  auto program = builder.Build();

#ifdef CINN_WITH_CUDA
  Target target = common::DefaultNVGPUTarget();
#else
  Target target = common::DefaultHostTarget();
#endif
  std::unordered_set<std::string> fetch_ids;
  auto graph = Optimize(&program, fetch_ids, target);

  auto scope = BuildScope(target, graph);
  hlir::framework::CompilationContext context(graph, scope, target);
  hlir::framework::GraphCompiler gc(context);
  auto runtime_program = gc.Build();

  scope->Var<hlir::framework::Tensor>(std::string(input.id()));
  scope->Var<hlir::framework::Tensor>(std::string(output->id));

  auto input_tensor = scope->GetTensor(std::string(input.id()));
  SetRandData<float>(input_tensor, target);
  std::vector<float> input_data = GetTensorData<float>(input_tensor, target);

  runtime_program->Execute();

  auto output_tensor = scope->GetTensor(std::string(output->id));
  const std::vector<int>& output_shape = output_tensor->shape().data();
  EXPECT_EQ(output_tensor->type(), Int(32));
  EXPECT_EQ(output_shape.size(), 2UL);
  EXPECT_EQ(output_shape[0], B);
  EXPECT_EQ(output_shape[1], H);

  std::vector<int> output_data = GetTensorData<int>(output_tensor, target);
  VLOG(6) << "Visualize output_data";
  for (int h = 0; h < H; ++h) {
    std::vector<float> sorted_data;
    std::vector<float> out_sorted_data(H);
    for (int b = 0; b < B; ++b) {
      int index = h + H * b;
      sorted_data.push_back(input_data[index]);
      out_sorted_data[b] = input_data[h + H * output_data[index]];
    }
    std::sort(sorted_data.begin(), sorted_data.begin() + B);

    for (int b = 0; b < B; ++b) {
      std::string line;
      int index = h + H * b;
      float true_data = sorted_data[b];
      float out_data = out_sorted_data[b];
      line += (std::to_string(out_data) + ", ");
      EXPECT_EQ(true_data, out_data);
      VLOG(6) << line;
    }
  }
}

TEST(net_build, program_execute_sort) {
  const int B = 4;
  const int H = 7;

  NetBuilder builder("net_builder");
  Placeholder input = builder.CreateInput(Float(32), {B, H}, "In");
  Variable output = builder.Sort(input, 0, true);
  auto program = builder.Build();

#ifdef CINN_WITH_CUDA
  Target target = common::DefaultNVGPUTarget();
#else
  Target target = common::DefaultHostTarget();
#endif
  std::unordered_set<std::string> fetch_ids;
  auto graph = Optimize(&program, fetch_ids, target);

  auto scope = BuildScope(target, graph);
  hlir::framework::CompilationContext context(graph, scope, target);
  hlir::framework::GraphCompiler gc(context);
  auto runtime_program = gc.Build();

  scope->Var<hlir::framework::Tensor>(std::string(input.id()));
  scope->Var<hlir::framework::Tensor>(std::string(output->id));

  auto input_tensor = scope->GetTensor(std::string(input.id()));
  SetRandData<float>(input_tensor, target);
  std::vector<float> input_data = GetTensorData<float>(input_tensor, target);

  runtime_program->Execute();

  auto output_tensor = scope->GetTensor(std::string(output->id));
  const std::vector<int>& output_shape = output_tensor->shape().data();
  EXPECT_EQ(output_tensor->type(), Float(32));
  EXPECT_EQ(output_shape.size(), 2UL);
  EXPECT_EQ(output_shape[0], B);
  EXPECT_EQ(output_shape[1], H);

  std::vector<float> output_data = GetTensorData<float>(output_tensor, target);
  VLOG(6) << "Visualize output_data";
  for (int h = 0; h < H; ++h) {
    std::vector<float> sorted_data;
    for (int b = 0; b < B; ++b) {
      int index = h + H * b;
      sorted_data.push_back(input_data[index]);
    }
    std::sort(sorted_data.begin(), sorted_data.begin() + B);

    for (int b = 0; b < B; ++b) {
      std::string line;
      int index = h + H * b;
      float true_data = sorted_data[b];
      float out_data = output_data[index];
      line += (std::to_string(out_data) + ", ");
      EXPECT_EQ(true_data, out_data);
      VLOG(6) << line;
    }
  }
}

TEST(net_build, program_execute_arange_float) {
  const float start = 1.5F;
  const float stop = 31.5F;
  const float step = 2.0F;
  const std::string dtype = "float32";

  NetBuilder builder("net_builder");
  Variable out = builder.Arange(start, stop, step, dtype);
  auto program = builder.Build();

#ifdef CINN_WITH_CUDA
  Target target = common::DefaultNVGPUTarget();
#else
  Target target = common::DefaultHostTarget();
#endif
  std::unordered_set<std::string> fetch_ids;
  auto graph = Optimize(&program, fetch_ids, target);

  auto scope = BuildScope(target, graph);
  hlir::framework::CompilationContext context(graph, scope, target);
  hlir::framework::GraphCompiler gc(context);
  auto runtime_program = gc.Build();

  scope->Var<hlir::framework::Tensor>(std::string(out->id));

  runtime_program->Execute();

  auto out_tensor = scope->GetTensor(std::string(out->id));
  const std::vector<int>& out_tensor_shape = out_tensor->shape().data();
  EXPECT_EQ(out_tensor->type(), Float(32));
  EXPECT_EQ(out_tensor_shape.size(), 1UL);

  int num_elem = static_cast<int>(std::ceil((stop - start) / step));
  EXPECT_EQ(out_tensor_shape[0], num_elem);

  std::vector<float> out_data = GetTensorData<float>(out_tensor, target);
  for (int i = 0; i < out_tensor_shape[0]; ++i) {
    EXPECT_NEAR(out_data[i], start + step * i, 1e-5);
    VLOG(6) << out_data[i];
  }
}

TEST(net_build, program_execute_arange_int) {
  const float start = 1.5F;
  const float stop = 31.5F;
  const float step = 1.6F;
  const std::string dtype = "int32";

  NetBuilder builder("net_builder");
  Variable out = builder.Arange(start, stop, step, dtype);
  auto program = builder.Build();

#ifdef CINN_WITH_CUDA
  Target target = common::DefaultNVGPUTarget();
#else
  Target target = common::DefaultHostTarget();
#endif
  std::unordered_set<std::string> fetch_ids;
  auto graph = Optimize(&program, fetch_ids, target);

  auto scope = BuildScope(target, graph);
  hlir::framework::CompilationContext context(graph, scope, target);
  hlir::framework::GraphCompiler gc(context);
  auto runtime_program = gc.Build();

  scope->Var<hlir::framework::Tensor>(std::string(out->id));

  runtime_program->Execute();

  auto out_tensor = scope->GetTensor(std::string(out->id));
  const std::vector<int>& out_tensor_shape = out_tensor->shape().data();
  EXPECT_EQ(out_tensor->type(), Int(32));
  EXPECT_EQ(out_tensor_shape.size(), 1UL);

  int num_elem = static_cast<int>(std::ceil((stop - start) / step));
  EXPECT_EQ(out_tensor_shape[0], num_elem);

  std::vector<int> out_data = GetTensorData<int>(out_tensor, target);
  for (int i = 0; i < out_tensor_shape[0]; ++i) {
    EXPECT_EQ(out_data[i], static_cast<int32_t>(start + step * i));
    VLOG(6) << out_data[i];
  }
}

TEST(net_build, program_argmax_case1) {
  const int N = 4;
  const int IN_C = 3;
  const int OUT_C = 1;
  const int H = 7;
  const int W = 7;

  NetBuilder builder("net_builder");
  Placeholder input = builder.CreateInput(Float(32), {N, IN_C, H, W}, "In");
  Variable output = builder.Argmax(input, 1, true);
  auto program = builder.Build();

#ifdef CINN_WITH_CUDA
  Target target = common::DefaultNVGPUTarget();
#else
  Target target = common::DefaultHostTarget();
#endif
  std::unordered_set<std::string> fetch_ids;
  auto graph = Optimize(&program, fetch_ids, target);

  auto scope = BuildScope(target, graph);
  hlir::framework::CompilationContext context(graph, scope, target);
  hlir::framework::GraphCompiler gc(context);
  auto runtime_program = gc.Build();

  scope->Var<hlir::framework::Tensor>(std::string(input.id()));
  scope->Var<hlir::framework::Tensor>(std::string(output->id));

  auto input_tensor = scope->GetTensor(std::string(input.id()));
  SetRandData<float>(input_tensor, target);
  std::vector<float> input_data = GetTensorData<float>(input_tensor, target);
  VLOG(6) << "Visualize input_data";
  for (int n = 0; n < N; ++n) {
    for (int c = 0; c < IN_C; ++c) {
      VLOG(6) << "n = " << n << ", c = " << c;
      for (int h = 0; h < H; ++h) {
        std::string line;
        for (int w = 0; w < W; ++w) {
          int index = w + W * (h + H * (c + IN_C * n));
          line += (std::to_string(input_data[index]) + ", ");
        }
        VLOG(6) << line;
      }
    }
  }
  runtime_program->Execute();

  auto output_tensor = scope->GetTensor(std::string(output->id));
  const std::vector<int>& output_shape = output_tensor->shape().data();
  EXPECT_EQ(output_shape.size(), 4UL);
  EXPECT_EQ(output_shape[0], N);
  EXPECT_EQ(output_shape[1], OUT_C);
  EXPECT_EQ(output_shape[2], H);
  EXPECT_EQ(output_shape[3], W);

  std::vector<int> output_data = GetTensorData<int>(output_tensor, target);
  VLOG(6) << "Visualize output_data";
  for (int n = 0; n < N; ++n) {
    for (int c = 0; c < IN_C; ++c) {
      VLOG(6) << "n = " << n << ", c = " << c;
      for (int h = 0; h < H; ++h) {
        std::string line;
        for (int w = 0; w < W; ++w) {
          int index = w + W * (h + H * (c + IN_C * n));
          int out_index = w + W * (h + H * n);
          float in_data = input_data[index];
          int out_data = output_data[out_index];
          EXPECT_LE(0, out_data);
          EXPECT_LT(out_data, IN_C);
          int max_index = w + W * (h + H * (out_data + IN_C * n));
          float max_value = input_data[max_index];
          line += (std::to_string(out_data) + ", ");
          EXPECT_LE(in_data, max_value);
        }
        VLOG(6) << line;
      }
    }
  }
}

TEST(net_build, program_argmax_case2) {
  const int N = 4;
  const int IN_C = 3;
  const int H = 7;
  const int W = 7;

  NetBuilder builder("net_builder");
  Placeholder input = builder.CreateInput(Float(32), {N, IN_C, H, W}, "In");
  Variable output = builder.Argmax(input, 1, false);
  auto program = builder.Build();

  Target target = common::DefaultHostTarget();
  std::unordered_set<std::string> fetch_ids;
  auto graph = Optimize(&program, fetch_ids, target);

  auto scope = BuildScope(target, graph);
  hlir::framework::CompilationContext context(graph, scope, target);
  hlir::framework::GraphCompiler gc(context);
  auto runtime_program = gc.Build();

  scope->Var<hlir::framework::Tensor>(std::string(input.id()));
  scope->Var<hlir::framework::Tensor>(std::string(output->id));

  auto input_tensor = scope->GetTensor(std::string(input.id()));
  SetRandData<float>(input_tensor, target);
  float* input_data = input_tensor->mutable_data<float>(target);
  VLOG(6) << "Visualize input_data";
  for (int n = 0; n < N; ++n) {
    for (int c = 0; c < IN_C; ++c) {
      VLOG(6) << "n = " << n << ", c = " << c;
      for (int h = 0; h < H; ++h) {
        std::string line;
        for (int w = 0; w < W; ++w) {
          int index = w + W * (h + H * (c + IN_C * n));
          line += (std::to_string(input_data[index]) + ", ");
        }
        VLOG(6) << line;
      }
    }
  }
  runtime_program->Execute();

  auto output_tensor = scope->GetTensor(std::string(output->id));
  const std::vector<int>& output_shape = output_tensor->shape().data();
  EXPECT_EQ(output_shape.size(), 3UL);
  EXPECT_EQ(output_shape[0], N);
  EXPECT_EQ(output_shape[1], H);
  EXPECT_EQ(output_shape[2], W);

  int* output_data = output_tensor->mutable_data<int>(target);
  VLOG(6) << "Visualize output_data";
  for (int n = 0; n < N; ++n) {
    for (int c = 0; c < IN_C; ++c) {
      VLOG(6) << "n = " << n << ", c = " << c;
      for (int h = 0; h < H; ++h) {
        std::string line;
        for (int w = 0; w < W; ++w) {
          int index = w + W * (h + H * (c + IN_C * n));
          int out_index = w + W * (h + H * n);
          float in_data = input_data[index];
          int out_data = output_data[out_index];
          EXPECT_LE(0, out_data);
          EXPECT_LT(out_data, IN_C);
          int max_index = w + W * (h + H * (out_data + IN_C * n));
          float max_value = input_data[max_index];
          line += (std::to_string(out_data) + ", ");
          EXPECT_LE(in_data, max_value);
        }
        VLOG(6) << line;
      }
    }
  }
}

TEST(net_build, program_argmin_case1) {
  const int N = 4;
  const int IN_C = 3;
  const int OUT_C = 1;
  const int H = 7;
  const int W = 7;

  NetBuilder builder("net_builder");
  Placeholder input = builder.CreateInput(Float(32), {N, IN_C, H, W}, "In");
  Variable output = builder.Argmin(input, 1, true);
  auto program = builder.Build();
#ifdef CINN_WITH_CUDA
  Target target = common::DefaultNVGPUTarget();
#else
  Target target = common::DefaultHostTarget();
#endif

  std::unordered_set<std::string> fetch_ids;
  auto graph = Optimize(&program, fetch_ids, target);

  auto scope = BuildScope(target, graph);
  hlir::framework::CompilationContext context(graph, scope, target);
  hlir::framework::GraphCompiler gc(context);
  auto runtime_program = gc.Build();

  scope->Var<hlir::framework::Tensor>(std::string(input.id()));
  scope->Var<hlir::framework::Tensor>(std::string(output->id));

  auto input_tensor = scope->GetTensor(std::string(input.id()));
  SetRandData<float>(input_tensor, target);
  std::vector<float> input_data = GetTensorData<float>(input_tensor, target);
  VLOG(6) << "Visualize input_data";
  for (int n = 0; n < N; ++n) {
    for (int c = 0; c < IN_C; ++c) {
      VLOG(6) << "n = " << n << ", c = " << c;
      for (int h = 0; h < H; ++h) {
        std::string line;
        for (int w = 0; w < W; ++w) {
          int index = w + W * (h + H * (c + IN_C * n));
          line += (std::to_string(input_data[index]) + ", ");
        }
        VLOG(6) << line;
      }
    }
  }
  runtime_program->Execute();

  auto output_tensor = scope->GetTensor(std::string(output->id));
  const std::vector<int>& output_shape = output_tensor->shape().data();
  EXPECT_EQ(output_shape.size(), 4UL);
  EXPECT_EQ(output_shape[0], N);
  EXPECT_EQ(output_shape[1], OUT_C);
  EXPECT_EQ(output_shape[2], H);
  EXPECT_EQ(output_shape[3], W);

  std::vector<int> output_data = GetTensorData<int>(output_tensor, target);
  VLOG(6) << "Visualize output_data";
  for (int n = 0; n < N; ++n) {
    for (int c = 0; c < IN_C; ++c) {
      VLOG(6) << "n = " << n << ", c = " << c;
      for (int h = 0; h < H; ++h) {
        std::string line;
        for (int w = 0; w < W; ++w) {
          int index = w + W * (h + H * (c + IN_C * n));
          int out_index = w + W * (h + H * n);
          float in_data = input_data[index];
          int out_data = output_data[out_index];
          EXPECT_LE(0, out_data);
          EXPECT_LT(out_data, IN_C);
          int max_index = w + W * (h + H * (out_data + IN_C * n));
          float max_value = input_data[max_index];
          line += (std::to_string(out_data) + ", ");
          EXPECT_GE(in_data, max_value);
        }
        VLOG(6) << line;
      }
    }
  }
}

TEST(net_build, program_argmin_case2) {
  const int N = 4;
  const int IN_C = 3;
  const int H = 7;
  const int W = 7;

  NetBuilder builder("net_builder");
  Placeholder input = builder.CreateInput(Float(32), {N, IN_C, H, W}, "In");
  Variable output = builder.Argmin(input, 1, false);
  auto program = builder.Build();
#ifdef CINN_WITH_CUDA
  Target target = common::DefaultNVGPUTarget();
#else
  Target target = common::DefaultHostTarget();
#endif
  std::unordered_set<std::string> fetch_ids;
  auto graph = Optimize(&program, fetch_ids, target);

  auto scope = BuildScope(target, graph);
  hlir::framework::CompilationContext context(graph, scope, target);
  hlir::framework::GraphCompiler gc(context);
  auto runtime_program = gc.Build();

  scope->Var<hlir::framework::Tensor>(std::string(input.id()));
  scope->Var<hlir::framework::Tensor>(std::string(output->id));

  auto input_tensor = scope->GetTensor(std::string(input.id()));
  SetRandData<float>(input_tensor, target);
  std::vector<float> input_data = GetTensorData<float>(input_tensor, target);
  VLOG(6) << "Visualize input_data";
  for (int n = 0; n < N; ++n) {
    for (int c = 0; c < IN_C; ++c) {
      VLOG(6) << "n = " << n << ", c = " << c;
      for (int h = 0; h < H; ++h) {
        std::string line;
        for (int w = 0; w < W; ++w) {
          int index = w + W * (h + H * (c + IN_C * n));
          line += (std::to_string(input_data[index]) + ", ");
        }
        VLOG(6) << line;
      }
    }
  }
  runtime_program->Execute();

  auto output_tensor = scope->GetTensor(std::string(output->id));
  const std::vector<int>& output_shape = output_tensor->shape().data();
  EXPECT_EQ(output_shape.size(), 3UL);
  EXPECT_EQ(output_shape[0], N);
  EXPECT_EQ(output_shape[1], H);
  EXPECT_EQ(output_shape[2], W);

  std::vector<int> output_data = GetTensorData<int>(output_tensor, target);
  VLOG(6) << "Visualize output_data";
  for (int n = 0; n < N; ++n) {
    for (int c = 0; c < IN_C; ++c) {
      VLOG(6) << "n = " << n << ", c = " << c;
      for (int h = 0; h < H; ++h) {
        std::string line;
        for (int w = 0; w < W; ++w) {
          int index = w + W * (h + H * (c + IN_C * n));
          int out_index = w + W * (h + H * n);
          float in_data = input_data[index];
          int out_data = output_data[out_index];
          EXPECT_LE(0, out_data);
          EXPECT_LT(out_data, IN_C);
          int max_index = w + W * (h + H * (out_data + IN_C * n));
          float max_value = input_data[max_index];
          line += (std::to_string(out_data) + ", ");
          EXPECT_GE(in_data, max_value);
        }
        VLOG(6) << line;
      }
    }
  }
}

TEST(net_build, program_execute_repeat_axis_0) {
  const int M = 4;
  const int N = 4;
  const int repeats = 3;
  const int axis = 0;

  NetBuilder builder("net_builder");
  Placeholder input = builder.CreateInput(Float(32), {M, N}, "In");
  Variable output = builder.Repeat(input, repeats, axis);
  auto program = builder.Build();

#ifdef CINN_WITH_CUDA
  Target target = common::DefaultNVGPUTarget();
#else
  Target target = common::DefaultHostTarget();
#endif
  std::unordered_set<std::string> fetch_ids;
  auto graph = Optimize(&program, fetch_ids, target);

  auto scope = BuildScope(target, graph);
  hlir::framework::CompilationContext context(graph, scope, target);
  hlir::framework::GraphCompiler gc(context);
  auto runtime_program = gc.Build();

  scope->Var<hlir::framework::Tensor>(std::string(input.id()));
  scope->Var<hlir::framework::Tensor>(std::string(output->id));

  auto input_tensor = scope->GetTensor(std::string(input.id()));
  SetRandData<float>(input_tensor, target);
  std::vector<float> input_data = GetTensorData<float>(input_tensor, target);

  runtime_program->Execute();

  auto output_tensor = scope->GetTensor(std::string(output->id));
  const std::vector<int>& output_shape = output_tensor->shape().data();

  const int new_M = M * repeats;
  const int new_N = N;
  EXPECT_EQ(output_tensor->type(), Float(32));
  EXPECT_EQ(output_shape.size(), 2UL);
  EXPECT_EQ(output_shape[0], new_M);
  EXPECT_EQ(output_shape[1], new_N);

  std::vector<float> output_data = GetTensorData<float>(output_tensor, target);
  for (int m = 0; m < new_M; ++m) {
    for (int n = 0; n < new_N; ++n) {
      int in_index =
          n + N * static_cast<int>(std::floor(static_cast<float>(m) / repeats));
      int out_index = n + new_N * m;
      float in_data = input_data[in_index];
      float out_data = output_data[out_index];
      EXPECT_EQ(in_data, out_data);
    }
  }
}

TEST(net_build, program_execute_repeat_axis_1) {
  const int M = 4;
  const int N = 4;
  const int repeats = 3;
  const int axis = 1;

  NetBuilder builder("net_builder");
  Placeholder input = builder.CreateInput(Float(32), {M, N}, "In");
  Variable output = builder.Repeat(input, repeats, axis);
  auto program = builder.Build();

#ifdef CINN_WITH_CUDA
  Target target = common::DefaultNVGPUTarget();
#else
  Target target = common::DefaultHostTarget();
#endif
  std::unordered_set<std::string> fetch_ids;
  auto graph = Optimize(&program, fetch_ids, target);

  auto scope = BuildScope(target, graph);
  hlir::framework::CompilationContext context(graph, scope, target);
  hlir::framework::GraphCompiler gc(context);
  auto runtime_program = gc.Build();

  scope->Var<hlir::framework::Tensor>(std::string(input.id()));
  scope->Var<hlir::framework::Tensor>(std::string(output->id));

  auto input_tensor = scope->GetTensor(std::string(input.id()));
  SetRandData<float>(input_tensor, target);
  std::vector<float> input_data = GetTensorData<float>(input_tensor, target);

  runtime_program->Execute();

  auto output_tensor = scope->GetTensor(std::string(output->id));
  const std::vector<int>& output_shape = output_tensor->shape().data();

  const int new_M = M;
  const int new_N = N * repeats;
  EXPECT_EQ(output_tensor->type(), Float(32));
  EXPECT_EQ(output_shape.size(), 2UL);
  EXPECT_EQ(output_shape[0], new_M);
  EXPECT_EQ(output_shape[1], new_N);

  std::vector<float> output_data = GetTensorData<float>(output_tensor, target);
  for (int m = 0; m < new_M; ++m) {
    for (int n = 0; n < new_N; ++n) {
      int in_index =
          N * m + static_cast<int>(std::floor(static_cast<float>(n) / repeats));
      int out_index = n + new_N * m;
      float in_data = input_data[in_index];
      float out_data = output_data[out_index];
      EXPECT_EQ(in_data, out_data);
    }
  }
}

TEST(net_build, program_execute_one_hot) {
  const int M = 4;
  const int N = 4;
  const int on_value = 1;
  const int off_value = 0;
  const int depth = 11;
  const int axis = 0;  // [-1 , M]
  const std::string dtype = "int32";

  NetBuilder builder("net_builder");
  Placeholder input = builder.CreateInput(Int(32), {M, N}, "In");
  Placeholder on_value_input = builder.CreateInput(Int(32), {1}, "OnValue");
  Placeholder off_value_input = builder.CreateInput(Int(32), {1}, "OffValue");
  Variable output = builder.OneHot(
      input, on_value_input, off_value_input, depth, axis, dtype);
  auto program = builder.Build();

#ifdef CINN_WITH_CUDA
  Target target = common::DefaultNVGPUTarget();
#else
  Target target = common::DefaultHostTarget();
#endif
  std::unordered_set<std::string> fetch_ids;
  auto graph = Optimize(&program, fetch_ids, target);

  auto scope = BuildScope(target, graph);
  hlir::framework::CompilationContext context(graph, scope, target);
  hlir::framework::GraphCompiler gc(context);
  auto runtime_program = gc.Build();

  scope->Var<hlir::framework::Tensor>(std::string(input.id()));
  scope->Var<hlir::framework::Tensor>(std::string(on_value_input.id()));
  scope->Var<hlir::framework::Tensor>(std::string(off_value_input.id()));
  scope->Var<hlir::framework::Tensor>(std::string(output->id));

  auto input_tensor = scope->GetTensor(std::string(input.id()));
  const std::vector<int>& intput_shape = input_tensor->shape().data();
  SetRandInt(input_tensor, target);
  std::vector<int> input_data = GetTensorData<int>(input_tensor, target);

  auto on_value_tensor = scope->GetTensor(std::string(on_value_input.id()));
  SetRandInt(on_value_tensor, target, -1, on_value, on_value + 1);

  auto off_value_tensor = scope->GetTensor(std::string(off_value_input.id()));
  SetRandInt(off_value_tensor, target, -1, off_value, off_value + 1);

  runtime_program->Execute();

  auto output_tensor = scope->GetTensor(std::string(output->id));
  const std::vector<int>& output_shape = output_tensor->shape().data();
  std::vector<int> output_data = GetTensorData<int>(output_tensor, target);

  EXPECT_EQ(output_tensor->type(), Int(32));
  EXPECT_EQ(output_shape.size(), intput_shape.size() + 1);

  const int true_axis = axis == -1 ? M : axis;
  int input_shape_index = 0;

  for (int i = 0; i < output_shape.size(); i++) {
    LOG(INFO) << output_shape[i];
    if (i == true_axis) {
      EXPECT_EQ(output_shape[i], depth);
    } else {
      EXPECT_EQ(output_shape[i], intput_shape[input_shape_index++]);
    }
  }

  for (int i = 0; i < output_shape[0]; ++i) {
    for (int j = 0; j < output_shape[1]; ++j) {
      for (int k = 0; k < output_shape[2]; ++k) {
        std::vector<int> s = {i, j, k};
        int input_index = 0;
        int output_index = 0;
        int base = 1;

        for (int x = s.size() - 1; x >= 0; --x) {
          if (x == true_axis) {
            continue;
          }
          input_index += base * s[x];
          base = base * output_shape[x];
        }

        base = 1;

        for (int x = s.size() - 1; x >= 0; --x) {
          output_index += base * s[x];
          base = base * output_shape[x];
        }

        if (s[true_axis] == input_data[input_index]) {
          EXPECT_EQ(output_data[output_index], on_value);
        } else {
          EXPECT_EQ(output_data[output_index], off_value);
        }
      }
    }
  }
}

}  // namespace frontend
}  // namespace cinn
