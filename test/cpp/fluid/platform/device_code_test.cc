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

#include "paddle/phi/backends/device_code.h"

#include <utility>

#include "gtest/gtest.h"
#include "paddle/fluid/framework/lod_tensor.h"
#include "paddle/fluid/platform/init.h"

#ifdef PADDLE_WITH_CUDA
constexpr auto saxpy_code = R"(
extern "C" __global__
void saxpy_kernel(float a, float *x, float* y, float* z, size_t n) {
  for (size_t tid = blockIdx.x * blockDim.x + threadIdx.x; tid < n;
       tid += blockDim.x * gridDim.x) {
    z[tid] = a * x[tid] + y[tid];
  }
}
)";
#endif

#ifdef PADDLE_WITH_HIP
constexpr auto saxpy_code = R"(
#include <hip/hip_runtime.h>
extern "C" __global__
void saxpy_kernel(float a, float *x, float* y, float* z, size_t n) {
  for (size_t tid = blockIdx.x * blockDim.x + threadIdx.x; tid < n;
       tid += blockDim.x * gridDim.x) {
    z[tid] = a * x[tid] + y[tid];
  }
}
)";
#endif

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
TEST(DeviceCode, cuda) {
  if (!phi::dynload::HasNVRTC() || !phi::dynload::HasCUDADriver()) {
    return;
  }

  paddle::framework::InitDevices({0});
  phi::GPUPlace place = phi::GPUPlace(0);
  phi::GPUDeviceCode code(place, "saxpy_kernel", saxpy_code);

  phi::DenseTensor cpu_x;
  phi::DenseTensor cpu_y;
  phi::DenseTensor cpu_z;

  float scale = 2;
  auto dims = common::make_ddim(
      {static_cast<int64_t>(256), static_cast<int64_t>(1024)});
  phi::DeviceContextPool& pool = phi::DeviceContextPool::Instance();
  auto* cpu_ctx = reinterpret_cast<phi::CPUContext*>(pool.Get(phi::CPUPlace()));
  cpu_x.Resize(dims);
  cpu_ctx->template Alloc<float>(&cpu_x);
  cpu_y.Resize(dims);
  cpu_ctx->template Alloc<float>(&cpu_y);

  size_t n = cpu_x.numel();
  for (size_t i = 0; i < n; ++i) {
    cpu_x.data<float>()[i] = static_cast<float>(i);
  }
  for (size_t i = 0; i < n; ++i) {
    cpu_y.data<float>()[i] = static_cast<float>(0.5);
  }

  phi::DenseTensor x;
  phi::DenseTensor y;
  phi::DenseTensor z;

  auto* dev_ctx = reinterpret_cast<phi::GPUContext*>(pool.Get(place));
  x.Resize(dims);
  float* x_data = dev_ctx->template Alloc<float>(&x);
  y.Resize(dims);
  float* y_data = dev_ctx->template Alloc<float>(&y);
  z.Resize(dims);
  float* z_data = dev_ctx->template Alloc<float>(&z);

  paddle::framework::TensorCopySync(cpu_x, place, &x);
  paddle::framework::TensorCopySync(cpu_y, place, &y);

  EXPECT_EQ(code.Compile(), true);

  std::vector<void*> args = {&scale, &x_data, &y_data, &z_data, &n};
  code.SetNumThreads(1024);
  code.SetWorkloadPerThread(1);
  code.Launch(n, &args);

  dev_ctx->Wait();

  paddle::framework::TensorCopySync(z, phi::CPUPlace(), &cpu_z);
  for (size_t i = 0; i < n; i++) {
    EXPECT_EQ(cpu_z.data<float>()[i], static_cast<float>(i) * scale + 0.5);
  }
}

TEST(DeviceCodePool, cuda) {
  if (!phi::dynload::HasNVRTC() || !phi::dynload::HasCUDADriver()) {
    return;
  }

  paddle::framework::InitDevices({0});
  phi::GPUPlace place = phi::GPUPlace(0);
  phi::DeviceCodePool& pool = phi::DeviceCodePool::Init({place});
  size_t num_device_codes_before = pool.size(place);
  EXPECT_EQ(num_device_codes_before, 0UL);

  std::unique_ptr<phi::DeviceCode> code(
      new phi::GPUDeviceCode(place, "saxpy_kernel", saxpy_code));
  LOG(INFO) << "origin ptr: " << code.get();
  pool.Set(std::move(code));
  size_t num_device_codes_after = pool.size(place);
  EXPECT_EQ(num_device_codes_after, 1UL);

  phi::DeviceCode* code_get = pool.Get(place, "saxpy_kernel");
  LOG(INFO) << "get ptr: " << code_get;
}
#endif
