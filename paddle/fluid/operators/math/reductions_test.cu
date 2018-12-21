/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/operators/math/reductions.h"

#include <iostream>
#include <vector>

#include "gtest/gtest.h"
#include "paddle/fluid/framework/tensor_util.h"

using CUDAPlace = paddle::platform::CUDAPlace;
using CPUPlace = paddle::platform::CPUPlace;
using CUDADeviceContext = paddle::platform::CUDADeviceContext;
using DeviceContextPool = paddle::platform::DeviceContextPool;
using Tensor = paddle::framework::Tensor;

TEST(RowReduction, GPU) {
  CUDAPlace gpu_place(0);
  CUDADeviceContext &ctx = *reinterpret_cast<CUDADeviceContext *>(
      DeviceContextPool::Instance().Get(gpu_place));

  paddle::framework::Tensor in_tensor, out_tensor;
  int num_rows = 3;
  int num_cols = 10;
  in_tensor.Resize({num_rows, num_cols});
  out_tensor.Resize({num_rows, 1});
  float *in_data = in_tensor.mutable_data<float>(gpu_place);
  float *out_data = out_tensor.mutable_data<float>(gpu_place);

  // prepare data
  Tensor cpu_tensor;
  cpu_tensor.Resize({num_rows, num_cols});
  float *cpu_data = cpu_tensor.mutable_data<float>(CPUPlace());
  std::vector<float> result;
  for (int i = 0; i < num_rows; ++i) {
    result.push_back(0.0f);
    for (int j = 0; j < num_cols; ++j) {
      cpu_data[i * num_cols + j] = i * num_cols + j + 1;
      result[i] += cpu_data[i * num_cols + j];
    }
  }
  paddle::framework::TensorCopy(cpu_tensor, gpu_place, &in_tensor);

  // Do reduction
  cub::Sum op;
  paddle::operators::math::LaunchRowReduction<float, float, cub::Sum>(
      &ctx, out_data, in_data, num_rows, num_cols, op, 0.0f);
  // copy result to cpu
  Tensor out_cpu_tensor;
  out_cpu_tensor.Resize({num_rows, 1});
  float *out_cpu_data = out_cpu_tensor.mutable_data<float>(CPUPlace());
  paddle::framework::TensorCopySync(out_tensor, CPUPlace(), &out_cpu_tensor);
  for (int i = 0; i < num_rows; ++i) {
    EXPECT_EQ(out_cpu_data[i], result[i]);
  }
}
