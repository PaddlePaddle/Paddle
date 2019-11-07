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
#include "paddle/fluid/framework/lod_tensor.h"
#include "paddle/fluid/operators/math.h"
#include "paddle/fluid/platform/device_code.h"
#include "paddle/fluid/platform/init.h"

#ifdef PADDLE_WITH_CUDA
namespace fusion_group = paddle::framework::ir::fusion_group;

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

void TestMain(std::string func_name,
              std::vector<fusion_group::OperationExpression> expressions,
              std::vector<paddle::framework::LoDTensor> cpu_tensors, int n,
              std::vector<int> input_ids, std::vector<int> output_ids) {
  fusion_group::OperationMap::Init();
  fusion_group::CodeGenerator code_generator;
  std::string code_str = code_generator.GenerateCode(func_name, expressions);
  VLOG(3) << code_str;

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
  TestMain("fused_elementwise_kernel", expressions, cpu_tensors, n, input_ids,
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
    PADDLE_ENFORCE_LT(fabs(cpu_tensors[8].data<float>()[i] - result), 1.E-05);
  }
}
#endif
