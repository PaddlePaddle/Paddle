/* Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/framework/ir/fusion_group/code_generator.h"
#include <gtest/gtest.h>
#include <cmath>
#include <string>
#include <vector>
#include "paddle/fluid/framework/ir/fusion_group/operation.h"
#include "paddle/fluid/framework/ir/pass_tester_helper.h"
#include "paddle/fluid/framework/lod_tensor.h"
#include "paddle/fluid/operators/math.h"
#include "paddle/fluid/platform/device_code.h"
#include "paddle/fluid/platform/init.h"

#ifdef PADDLE_WITH_CUDA
namespace fusion_group = paddle::framework::ir::fusion_group;

template <typename T>
void CheckOutput(T actual, T expect) {
  PADDLE_ENFORCE_LT(fabs(actual - expect), 1.E-05,
                    "Get %f vs %f (actual vs expect).", actual, expect);
}

template <typename T>
void SetupRandomCPUTensor(paddle::framework::LoDTensor* tensor) {
  static unsigned int seed = 100;
  std::mt19937 rng(seed++);
  std::uniform_real_distribution<double> uniform_dist(0, 1);

  T* ptr = tensor->data<T>();
  PADDLE_ENFORCE_NOT_NULL(
      ptr, "Call mutable_data to alloc memory for Tensor first.");
  for (int64_t i = 0; i < tensor->numel(); ++i) {
    ptr[i] = static_cast<T>(uniform_dist(rng)) - static_cast<T>(0.5);
  }
}

void TestMainImpl(std::string func_name, std::string code_str,
                  std::vector<paddle::framework::LoDTensor> cpu_tensors, int n,
                  std::vector<int> input_ids, std::vector<int> output_ids) {
  paddle::framework::InitDevices(false, {0});
  paddle::platform::CUDAPlace place = paddle::platform::CUDAPlace(0);
  paddle::platform::CUDADeviceCode device_code(place, func_name, code_str);
  device_code.Compile();

  std::vector<paddle::framework::LoDTensor> gpu_tensors(cpu_tensors.size());

  std::vector<float*> gpu_ptrs(gpu_tensors.size());
  std::vector<void*> args;
  args.push_back(&n);

  for (size_t i = 0; i < input_ids.size(); ++i) {
    gpu_ptrs[input_ids[i]] = gpu_tensors[input_ids[i]].mutable_data<float>(
        cpu_tensors[input_ids[i]].dims(), place);
    args.push_back(&gpu_ptrs[input_ids[i]]);

    SetupRandomCPUTensor<float>(&cpu_tensors[input_ids[i]]);
    TensorCopySync(cpu_tensors[input_ids[i]], place,
                   &gpu_tensors[input_ids[i]]);
  }

  for (size_t i = 0; i < output_ids.size(); ++i) {
    gpu_ptrs[output_ids[i]] = gpu_tensors[output_ids[i]].mutable_data<float>(
        cpu_tensors[output_ids[i]].dims(), place);
    args.push_back(&gpu_ptrs[output_ids[i]]);
  }

  device_code.SetNumThreads(1024);
  device_code.SetWorkloadPerThread(1);
  device_code.Launch(n, &args);

  auto* dev_ctx = reinterpret_cast<paddle::platform::CUDADeviceContext*>(
      paddle::platform::DeviceContextPool::Instance().Get(place));
  dev_ctx->Wait();

  for (size_t i = 0; i < output_ids.size(); ++i) {
    TensorCopySync(gpu_tensors[output_ids[i]], paddle::platform::CPUPlace(),
                   &cpu_tensors[output_ids[i]]);
  }
}

void TestMain(std::string func_name,
              std::vector<fusion_group::OperationExpression> expressions,
              std::vector<paddle::framework::LoDTensor> cpu_tensors, int n,
              std::vector<int> input_ids, std::vector<int> output_ids) {
  fusion_group::OperationMap::Init();
  fusion_group::CodeGenerator code_generator;
  std::string code_str = code_generator.Generate(func_name, expressions);
  VLOG(3) << code_str;

  TestMainImpl(func_name, code_str, cpu_tensors, n, input_ids, output_ids);
}

void TestMain(fusion_group::SubGraph* subgraph,
              std::vector<paddle::framework::LoDTensor> cpu_tensors, int n,
              std::vector<int> input_ids, std::vector<int> output_ids) {
  fusion_group::OperationMap::Init();
  fusion_group::CodeGenerator code_generator;
  std::string code_str = code_generator.Generate(subgraph);
  LOG(INFO) << code_str;

  TestMainImpl(subgraph->func_name, code_str, cpu_tensors, n, input_ids,
               output_ids);
}

TEST(code_generator, elementwise) {
  // t2 = t0 * t1
  // t4 = t2 + t3
  // t6 = t4 - t5
  // t7 = relu(t6)
  // t8 = sigmoid(t7)
  fusion_group::OperationExpression exp1("elementwise_mul", {0, 1}, {2});
  fusion_group::OperationExpression exp2("elementwise_add", {2, 3}, {4});
  fusion_group::OperationExpression exp3("elementwise_sub", {4, 5}, {6});
  fusion_group::OperationExpression exp4("relu", {6}, {7});
  fusion_group::OperationExpression exp5("sigmoid", {7}, {8});

  std::vector<fusion_group::OperationExpression> expressions = {
      exp1, exp2, exp3, exp4, exp5};

  // Prepare CPU tensors
  std::vector<paddle::framework::LoDTensor> cpu_tensors(9);
  std::vector<int> input_ids = {0, 1, 3, 5};
  std::vector<int> output_ids = {2, 4, 6, 7, 8};

  auto dims = paddle::framework::make_ddim(
      {static_cast<int64_t>(256), static_cast<int64_t>(1024)});
  for (size_t i = 0; i < cpu_tensors.size(); ++i) {
    cpu_tensors[i].mutable_data<float>(dims, paddle::platform::CPUPlace());
  }

  int n = cpu_tensors[0].numel();
  TestMain("fused_elementwise_0", expressions, cpu_tensors, n, input_ids,
           output_ids);

  auto cpu_kernel_handler = [&](float* var0, float* var1, float* var3,
                                float* var5, int i) -> float {
    float var2_i = var0[i] * var1[i];
    float var4_i = var2_i + var3[i];
    float var6_i = var4_i - var5[i];
    float var7_i = var6_i > 0.0 ? var6_i : 0.0;
    float var8_i = 1.0 / (1.0 + std::exp(-var7_i));
    return var8_i;
  };

  // Check the results
  for (int i = 0; i < n; i++) {
    float result = cpu_kernel_handler(
        cpu_tensors[0].data<float>(), cpu_tensors[1].data<float>(),
        cpu_tensors[3].data<float>(), cpu_tensors[5].data<float>(), i);
    CheckOutput(cpu_tensors[8].data<float>()[i], result);
  }
}

TEST(code_generator, elementwise_grad) {
  // The var order: t0, t1, t2, t3, t0', t1', t2', t3'
  // t2 = t0 * t1
  // t3 = relu(t2)
  // t2' = relu_grad(t2, t3, t3')
  // t0', t1' = elementwise_mul_grad(t0, t1, t2, t2')
  fusion_group::OperationExpression exp1("relu_grad", {2, 3, 7}, {6});
  fusion_group::OperationExpression exp2("elementwise_mul_grad", {0, 1, 2, 6},
                                         {4, 5});

  std::vector<fusion_group::OperationExpression> expressions = {exp1, exp2};

  // Prepare CPU tensors
  std::vector<paddle::framework::LoDTensor> cpu_tensors(8);
  std::vector<int> input_ids = {0, 1, 2, 3, 7};
  std::vector<int> output_ids = {4, 5, 6};

  auto dims = paddle::framework::make_ddim(
      {static_cast<int64_t>(256), static_cast<int64_t>(1024)});
  for (size_t i = 0; i < cpu_tensors.size(); ++i) {
    cpu_tensors[i].mutable_data<float>(dims, paddle::platform::CPUPlace());
  }

  int n = cpu_tensors[0].numel();
  TestMain("fused_elementwise_grad_0", expressions, cpu_tensors, n, input_ids,
           output_ids);

  auto cpu_kernel_handler = [&](float* var0, float* var1, float* var2,
                                float* var3, float* var7,
                                int i) -> std::vector<float> {
    float var6_i = var2[i] > 0 ? var7[i] : 0;
    float var4_i = var6_i * var1[i];
    float var5_i = var6_i * var0[i];
    return std::vector<float>{var4_i, var5_i, var6_i};
  };

  // Check the results
  for (int i = 0; i < n; i++) {
    std::vector<float> results = cpu_kernel_handler(
        cpu_tensors[0].data<float>(), cpu_tensors[1].data<float>(),
        cpu_tensors[2].data<float>(), cpu_tensors[3].data<float>(),
        cpu_tensors[7].data<float>(), i);
    CheckOutput(cpu_tensors[4].data<float>()[i], results[0]);
    CheckOutput(cpu_tensors[5].data<float>()[i], results[1]);
    CheckOutput(cpu_tensors[6].data<float>()[i], results[2]);
  }
}

TEST(code_generator, subgraph) {
  // inputs                     operator            output
  // --------------------------------------------------------
  // x0                         sigmoid          -> tmp_0
  // (tmp_0, x1)                elementwise_mul  -> tmp_1
  // x2                         tanh             -> tmp_2
  // (x3, tmp_2)                elementwise_mul  -> tmp_3
  // (tmp_1, tmp_3)             elementwise_add  -> tmp_4
  //
  // Expression: tmp_4 = sigmoid(x0) * x1 + tanh(x2) * x3
  // The var order: x0, x1, x2, x3, tmp_0, tmp_2, tmp_1, tmp_3, tmp_4
  paddle::framework::ir::Layers layers;
  auto* x0 = layers.data("x0", {16, 32});
  auto* tmp_0 = layers.sigmoid(x0);
  tmp_0->SetShape({16, 32});
  auto* x1 = layers.data("x1", {16, 32});
  auto* tmp_1 = layers.elementwise_mul(tmp_0, x1);
  tmp_1->SetShape({16, 32});
  auto* x2 = layers.data("x2", {16, 32});
  auto* tmp_2 = layers.tanh(x2);
  tmp_2->SetShape({16, 32});
  auto* x3 = layers.data("x3", {16, 32});
  auto* tmp_3 = layers.elementwise_mul(x3, tmp_2);
  tmp_3->SetShape({16, 32});
  layers.elementwise_add(tmp_1, tmp_3);

  std::unique_ptr<paddle::framework::ir::Graph> graph(
      new paddle::framework::ir::Graph(layers.main_program()));
  fusion_group::SubGraph subgraph(0, "elementwise_kernel_1", true,
                                  graph->Nodes());

  // Prepare CPU tensors
  std::vector<paddle::framework::LoDTensor> cpu_tensors(9);
  // input_ids and output_ds should be consistant with that parsed from
  // subgraph.
  std::vector<int> input_ids = {0, 1, 2, 3};
  std::vector<int> output_ids = {4, 5, 6, 7, 8};

  auto dims = paddle::framework::make_ddim(
      {static_cast<int64_t>(256), static_cast<int64_t>(1024)});
  for (size_t i = 0; i < cpu_tensors.size(); ++i) {
    cpu_tensors[i].mutable_data<float>(dims, paddle::platform::CPUPlace());
  }

  int n = cpu_tensors[0].numel();
  TestMain(&subgraph, cpu_tensors, n, input_ids, output_ids);

  auto cpu_kernel_handler = [&](float* var0, float* var1, float* var2,
                                float* var3, int i) -> float {
    float var4_i = 1.0 / (1.0 + std::exp(-var0[i]));
    float var6_i = var4_i * var1[i];
    float var5_i = std::tanh(var2[i]);
    float var7_i = var3[i] * var5_i;
    float var8_i = var6_i + var7_i;
    return var8_i;
  };

  // Check the results
  for (int i = 0; i < n; i++) {
    float result = cpu_kernel_handler(
        cpu_tensors[0].data<float>(), cpu_tensors[1].data<float>(),
        cpu_tensors[2].data<float>(), cpu_tensors[3].data<float>(), i);
    CheckOutput(cpu_tensors[8].data<float>()[i], result);
  }
}
#endif
