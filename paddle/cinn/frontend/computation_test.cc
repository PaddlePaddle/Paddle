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

#include "paddle/cinn/frontend/computation.h"

#include <gtest/gtest.h>

#include "paddle/cinn/common/target.h"
#include "paddle/cinn/frontend/decomposer/use_decomposer.h"
#include "paddle/cinn/frontend/decomposer_registry.h"
#include "paddle/cinn/frontend/net_builder.h"
#include "paddle/cinn/frontend/pass/use_program_pass.h"
#include "paddle/cinn/frontend/program_pass.h"

PD_DEFINE_string(model_dir, "", "");

namespace cinn {
namespace frontend {

Program CreateTestProgram() {
  constexpr int B = 8;
  constexpr int M = 32;
  constexpr int N = 24;

  NetBuilder builder("net_builder");
  auto a = builder.CreateInput(Float(32), {M, N / 2}, "A");
  auto b = builder.CreateInput(Float(32), {M, N / 2}, "B");
  auto t = builder.Transpose(b, {1, 0});
  auto r = builder.Reshape(t, {M, N / 2});
  auto c = builder.Add(a, r);
  auto x = builder.Divide(a, b);
  auto d = builder.Concat({c, x}, 1);
  auto e = builder.BroadcastTo(d, {B, M, N}, {1, 2});
  auto f = builder.Concat({a, b}, 1);
  auto g = builder.BroadcastTo(f, {B, M, N}, {1, 2});
  auto h = builder.Subtract(e, g);
  auto i = builder.Max(e, h);
  auto j = builder.Min(e, h);
  auto k = builder.Multiply(i, j);
  auto l = builder.Constant<bool>(1, "condition");
  auto m = builder.BroadcastTo(l, {B, M, N}, {0});
  auto n = builder.Select(m, j, k);
  auto o = builder.ReduceSum(n, {0, 1, 2});

  auto program = builder.Build();
  return program;
}

Program CreateAddProgram() {
  constexpr int M = 32;
  constexpr int N = 24;

  NetBuilder builder("net_builder");
  auto a = builder.CreateInput(Float(32), {M, N});
  auto b = builder.CreateInput(Float(32), {M, N});
  auto c = builder.Relu(a);
  auto d = builder.Add(b, c);
  auto program = builder.Build();

  return program;
}

TEST(cinn_computation, basic_cpu) {
  NetBuilder builder("basic");
  constexpr int M = 32;
  constexpr int N = 24;

  auto a = builder.CreateInput(Float(32), {M, N}, "A");
  auto b = builder.CreateInput(Float(32), {M, N}, "B");
  auto c = builder.Add(a, b);
  auto d = builder.Add(a, c);

  auto target = common::DefaultHostTarget();
  auto comp = CinnComputation::BuildAndCompile(target, builder);
  std::vector<float> hostA(M * N);
  std::vector<float> hostB(M * N);
  std::vector<float> hostD(M * N);
  std::vector<float> hostD_expected(M * N);
  for (int i = 0; i < M * N; i++) {
    hostA[i] = static_cast<float>(rand()) / INT_MAX;  // NOLINT
    hostB[i] = static_cast<float>(rand()) / INT_MAX;  // NOLINT
    hostD_expected[i] = hostA[i] * 2 + hostB[i];
  }

  comp->SetTensorData("A",
                      reinterpret_cast<void *>(hostA.data()),
                      hostA.size() * sizeof(float));
  comp->SetTensorData("B",
                      reinterpret_cast<void *>(hostB.data()),
                      hostB.size() * sizeof(float));
  comp->Execute();
  comp->GetTensorData(d->id,
                      reinterpret_cast<void *>(hostD.data()),
                      hostD.size() * sizeof(float));
  for (int i = 0; i < hostD.size(); i++) {
    ASSERT_NEAR(hostD[i], hostD_expected[i], 1e-5);
  }
}

#ifdef CINN_WITH_CUDA
TEST(cinn_computation, basic_gpu) {
  NetBuilder builder("basic");
  constexpr int M = 32;
  constexpr int N = 24;

  auto a = builder.CreateInput(Float(32), {M, N}, "A");
  auto b = builder.CreateInput(Float(32), {M, N}, "B");
  auto c = builder.Add(a, b);
  auto d = builder.Add(a, c);

  auto target = common::DefaultNVGPUTarget();
  auto comp = CinnComputation::BuildAndCompile(target, builder);
  std::vector<float> hostA(M * N);
  std::vector<float> hostB(M * N);
  std::vector<float> hostD(M * N);
  std::vector<float> hostD_expected(M * N);
  for (int i = 0; i < M * N; i++) {
    hostA[i] = static_cast<float>(rand()) / INT_MAX;  // NOLINT
    hostB[i] = static_cast<float>(rand()) / INT_MAX;  // NOLINT
    hostD_expected[i] = hostA[i] * 2 + hostB[i];
  }

  comp->SetTensorData("A",
                      reinterpret_cast<void *>(hostA.data()),
                      hostA.size() * sizeof(float));
  comp->SetTensorData("B",
                      reinterpret_cast<void *>(hostB.data()),
                      hostB.size() * sizeof(float));
  comp->Execute();
  comp->GetTensorData(d->id,
                      reinterpret_cast<void *>(hostD.data()),
                      hostD.size() * sizeof(float));
  for (int i = 0; i < hostD.size(); i++) {
    ASSERT_NEAR(hostD[i], hostD_expected[i], 1e-5);
  }
}
#endif

TEST(cinn_computation, net_builder_cpu) {
  auto program = CreateTestProgram();
  auto target = common::DefaultHostTarget();
  auto compute = CinnComputation::Compile(target, program);
  auto inputs = compute->GetInputTensors();
  ASSERT_EQ(inputs.size(), 2);
  auto tensorA = inputs[0];
  auto tensorB = inputs[1];
  ASSERT_EQ(tensorA->shape().numel(), 32 * 24 / 2);
  ASSERT_EQ(tensorB->shape().numel(), 32 * 24 / 2);

  auto outputs = compute->GetOutputTensors();
  ASSERT_EQ(outputs.size(), 1);
  auto tensorOut = outputs[0];

  auto load_input = [=](hlir::framework::Tensor t) {
    float *ptr = t->mutable_data<float>(target);
    for (int i = 0; i < t->shape().numel(); i++) {
      ptr[i] = static_cast<float>(rand()) / INT_MAX;  // NOLINT
    }
  };

  // run inference for 10 times
  for (int i = 0; i < 10; i++) {
    // load data directly to tensor's host memory
    load_input(tensorA);
    load_input(tensorB);
    // execute engine
    compute->Execute();
    // get outputs (ignored)
  }
}

#ifdef CINN_WITH_CUDA
TEST(cinn_computation, net_builder_gpu) {
  auto program = CreateTestProgram();
  auto target = common::DefaultNVGPUTarget();
  auto compute = CinnComputation::Compile(target, program);
  auto inputs = compute->GetInputTensors();
  ASSERT_EQ(inputs.size(), 2);
  auto tensorA = inputs[0];
  auto tensorB = inputs[1];
  ASSERT_EQ(tensorA->shape().numel(), 32 * 24 / 2);
  ASSERT_EQ(tensorB->shape().numel(), 32 * 24 / 2);
  auto outputs = compute->GetOutputTensors();
  ASSERT_EQ(outputs.size(), 1);
  auto tensorOut = outputs[0];

  // run inference for 10 times
  for (int i = 0; i < 10; i++) {
    // load data directly to tensor's host memory
    // assume tensorA is generated in GPU directly
    float *device_ptrA = tensorOut->mutable_data<float>(target);
    // ... generated data directly in device memory via gpu kernels
    // ... or async copy to device memory
    // ... not showed here

    // assume tensorB is generated in host memory, needs copy to GPU memory
    // (sync.)
    std::vector<float> hostB(32 * 24 / 2);
    compute->SetTensorData(tensorB,
                           reinterpret_cast<void *>(hostB.data()),
                           hostB.size() * sizeof(float));

    // execute engine
    compute->Execute();
    // get outputs
    std::vector<float> hostOut(tensorOut->shape().numel());
    compute->GetTensorData(tensorOut,
                           reinterpret_cast<void *>(hostOut.data()),
                           hostOut.size() * sizeof(float));
  }
}
#endif

TEST(cinn_computation, fc_execute_cpu) {
  auto target = common::DefaultHostTarget();
  ASSERT_NE(FLAGS_model_dir, "");
  auto compute = CinnComputation::CompilePaddleModel(
      target, FLAGS_model_dir, {"A"}, {{1, 30}}, false);
  auto inputs = compute->GetInputTensors();
  ASSERT_EQ(inputs.size(), 1);
  auto A = inputs[0];
  ASSERT_EQ(A->shape().numel(), 1 * 30);
  float *ptrA = A->mutable_data<float>(target);
  for (int i = 0; i < 30; i++)
    ptrA[i] = static_cast<float>(rand()) / INT_MAX;  // NOLINT
  for (int i = 0; i < 30; i++) ptrA[i] = static_cast<float>(0);
  compute->Execute();
}

#ifdef CINN_WITH_CUDA
TEST(cinn_computation, fc_execute_gpu) {
  auto target = common::DefaultNVGPUTarget();
  ASSERT_NE(FLAGS_model_dir, "");
  auto compute = CinnComputation::CompilePaddleModel(
      target, FLAGS_model_dir, {"A"}, {{1, 30}}, false);

  auto inputs = compute->GetInputTensors();
  ASSERT_EQ(inputs.size(), 1);
  auto A = inputs[0];
  ASSERT_EQ(A->shape().numel(), 1 * 30);
  auto outputs = compute->GetOutputTensors();
  ASSERT_EQ(outputs.size(), 1);
  auto out = outputs[0];

  std::vector<float> hostA(30);
  for (float &v : hostA) v = static_cast<float>(rand()) / INT_MAX;  // NOLINT
  compute->SetTensorData(
      A, reinterpret_cast<void *>(hostA.data()), hostA.size() * sizeof(float));

  compute->Execute();

  std::vector<float> hostOut(30);
  compute->GetTensorData(out,
                         reinterpret_cast<void *>(hostOut.data()),
                         hostOut.size() * sizeof(float));
}
#endif

TEST(cinn_computation, decomposer_cpu) {
  // this test only shows the API usage
  ASSERT_NE(cinn::frontend::ProgramPassRegistry::Global()->Find("Decomposer"),
            nullptr);
  // without decomposer
  {
    auto prog = CreateAddProgram();
    auto target = common::DefaultHostTarget();
    auto options = CinnComputation::DefaultCompileOptions();
    options.use_decomposer = false;
    auto compute = CinnComputation::Compile(target, prog, options);
    auto names = compute->GetAllTensorNames();
    ASSERT_EQ(names.size(), 3);
  }
  // with decomposer
  {
    auto prog = CreateAddProgram();
    auto target = common::DefaultHostTarget();
    auto options = CinnComputation::DefaultCompileOptions();
    options.use_decomposer = true;
    auto compute = CinnComputation::Compile(target, prog, options);
    auto names = compute->GetAllTensorNames();
  }
}

#ifdef CINN_WITH_CUDA
TEST(cinn_computation, gpu_stream) {
  // this test only shows the API usage
  auto target = common::DefaultNVGPUTarget();
  auto prog = CreateAddProgram();
  auto options = CinnComputation::DefaultCompileOptions();

  cudaStream_t streams[1];
  cudaStreamCreate(&streams[0]);
  auto compute = CinnComputation::Compile(
      target, prog, options, {}, static_cast<void *>(streams[0]));
  compute->Execute();
}
#endif

TEST(cinn_computation, without_instantiate_variables) {
  // this test only shows the API usage
  auto target = common::DefaultHostTarget();
  auto prog = CreateAddProgram();
  auto options = CinnComputation::DefaultCompileOptions();
  options.with_instantiate_variables = false;

  auto compute = CinnComputation::Compile(target, prog, options);
  auto names = compute->GetAllTensorNames();

  std::map<std::string, cinn_pod_value_t> pod2args;
  // compute->Execute(&pod2args);
}

}  // namespace frontend
}  // namespace cinn
