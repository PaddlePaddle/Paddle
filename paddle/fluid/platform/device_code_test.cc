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

#include "paddle/fluid/platform/device_code.h"
#include "gtest/gtest.h"
#include "paddle/fluid/framework/lod_tensor.h"
#include "paddle/fluid/platform/init.h"

constexpr auto saxpy_code = R"(
extern "C" __global__
void saxpy_kernel(float a, float *x, float* y, float* z, size_t n) {
  for (size_t tid = blockIdx.x * blockDim.x + threadIdx.x; tid < n;
       tid += blockDim.x * gridDim.x) {
    z[tid] = a * x[tid] + y[tid];
  }
}
)";

#ifdef PADDLE_WITH_CUDA
TEST(device_code, cuda) {
  paddle::framework::InitDevices(false, {0});
  paddle::platform::CUDAPlace place = paddle::platform::CUDAPlace(0);
  paddle::platform::CUDADeviceCode code(place, "saxpy_kernel", saxpy_code);

  paddle::framework::Tensor cpu_x;
  paddle::framework::Tensor cpu_y;
  paddle::framework::Tensor cpu_z;

  float scale = 2;
  auto dims = paddle::framework::make_ddim(
      {static_cast<int64_t>(256), static_cast<int64_t>(1024)});
  cpu_x.mutable_data<float>(dims, paddle::platform::CPUPlace());
  cpu_y.mutable_data<float>(dims, paddle::platform::CPUPlace());

  size_t n = cpu_x.numel();
  for (size_t i = 0; i < n; ++i) {
    cpu_x.data<float>()[i] = static_cast<float>(i);
  }
  for (size_t i = 0; i < n; ++i) {
    cpu_y.data<float>()[i] = static_cast<float>(0.5);
  }

  paddle::framework::Tensor x;
  paddle::framework::Tensor y;
  paddle::framework::Tensor z;

  float* x_data = x.mutable_data<float>(dims, place);
  float* y_data = y.mutable_data<float>(dims, place);
  float* z_data = z.mutable_data<float>(dims, place);

  TensorCopySync(cpu_x, place, &x);
  TensorCopySync(cpu_y, place, &y);

  code.Compile();

  std::vector<void*> args = {&scale, &x_data, &y_data, &z_data, &n};
  code.SetNumThreads(1024);
  code.SetWorkloadPerThread(1);
  code.Launch(n, &args);

  TensorCopySync(z, paddle::platform::CPUPlace(), &cpu_z);
  for (size_t i = 0; i < n; i++) {
    PADDLE_ENFORCE_EQ(cpu_z.data<float>()[i],
                      static_cast<float>(i) * scale + 0.5);
  }
}
#endif
