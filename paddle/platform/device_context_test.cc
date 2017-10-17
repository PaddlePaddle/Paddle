/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserve.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/platform/device_context.h"
#include "gtest/gtest.h"

TEST(Device, Init) {
  using paddle::platform::DeviceContext;
  using paddle::platform::CUDADeviceContext;
  using paddle::platform::GPUPlace;

  int count = paddle::platform::GetCUDADeviceCount();
  for (int i = 0; i < count; i++) {
    DeviceContext* device_context = new CUDADeviceContext(GPUPlace(i));
    Eigen::GpuDevice* gpu_device =
        device_context->template GetEigenDevice<GPUPlace>();
    ASSERT_NE(nullptr, gpu_device);
    delete device_context;
  }
}

TEST(Device, CUDADeviceContext) {
  using paddle::platform::CUDADeviceContext;
  using paddle::platform::GPUPlace;

  int count = paddle::platform::GetCUDADeviceCount();
  for (int i = 0; i < count; i++) {
    CUDADeviceContext* device_context = new CUDADeviceContext(GPUPlace(i));
    Eigen::GpuDevice* gpu_device = device_context->eigen_device();
    ASSERT_NE(nullptr, gpu_device);
    cudnnHandle_t cudnn_handle = device_context->cudnn_handle();
    ASSERT_NE(nullptr, cudnn_handle);
    cublasHandle_t cublas_handle = device_context->cublas_handle();
    ASSERT_NE(nullptr, cublas_handle);
    ASSERT_NE(nullptr, device_context->stream());
    delete device_context;
  }
}
