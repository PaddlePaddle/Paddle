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

void TestMain(std::string func_name,
              std::vector<fusion_group::OperationExpression> expressions,
              std::vector<paddle::framework::LoDTensor> cpu_tensors, int n,
              std::vector<int> input_ids, std::vector<int> output_ids) {
  fusion_group::OperationMap::Init();
  fusion_group::CodeGenerator code_generator;
  std::string code_str = code_generator.GenerateCode(func_name, expressions);
  std::cout << code_str << std::endl;

  paddle::framework::InitDevices(false, {0});
  paddle::platform::CUDAPlace place = paddle::platform::CUDAPlace(0);
  paddle::platform::CUDADeviceCode device_code(place, func_name, code_str);
  device_code.Compile();

  std::vector<paddle::framework::LoDTensor> gpu_tensors(cpu_tensors.size());
  VLOG(3) << gpu_tensors.size();

  std::vector<float*> gpu_ptrs(cpu_tensors.size());
  std::vector<void*> args;
  args.push_back(&n);
  for (size_t i = 0; i < input_ids.size(); ++i) {
    gpu_ptrs[input_ids[i]] = gpu_tensors[input_ids[i]].mutable_data<float>(
        cpu_tensors[input_ids[i]].dims(), place);
    args.push_back(&gpu_ptrs[input_ids[i]]);
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
  for (int i = 0; i < n; ++i) {
    cpu_tensors[input_ids[0]].data<float>()[i] = static_cast<float>(i);
    cpu_tensors[input_ids[1]].data<float>()[i] = static_cast<float>(0.5);
    cpu_tensors[input_ids[2]].data<float>()[i] = static_cast<float>(10.0);
    cpu_tensors[input_ids[3]].data<float>()[i] = static_cast<float>(0.0);
  }

  TestMain("fused_elementwise_kernel", expressions, cpu_tensors, n, input_ids,
           output_ids);

  // Check the results
  for (int i = 0; i < n; i++) {
    float result =
        (1.0 / (1.0 + std::exp(-std::max(
                          0.0, static_cast<float>(i) * 0.5 + 10.0 - 0.0))));
    PADDLE_ENFORCE_EQ(cpu_tensors[8].data<float>()[i], result);
  }
}
#endif
