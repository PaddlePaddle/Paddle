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

#include "paddle/platform/cuda_device.h"
#include "gtest/gtest.h"

TEST(Device, Init) {
  int count = paddle::platform::GetDeviceCount();
  for (int i = 0; i < count; i++) {
    paddle::platform::Device<DEVICE_GPU>* device =
        new paddle::platform::Device<DEVICE_GPU>(i);
    Eigen::GpuDevice gpu_device = device->eigen_device();
    ASSERT_NE(nullptr, gpu_device.stream());
    cudnnHandle_t cudnn_handle = device->cudnn_handle();
    ASSERT_NE(nullptr, cudnn_handle);
    cublasHandle_t cublas_handle = device->cublas_handle();
    ASSERT_NE(nullptr, cublas_handle);
    curandGenerator_t curand_handle = device->curand_generator();
    ASSERT_NE(nullptr, curand_handle);
    delete device;
  }
}
