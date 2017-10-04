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

#ifdef PADDLE_WITH_CUDA
TEST(DeviceContext, CUDA) {
  using paddle::platform::DeviceContext;
  using paddle::platform::CUDADeviceContext;
  using paddle::platform::GPUPlace;

  for (int i = 0; i < paddle::platform::GetDeviceCount(); i++) {
    DeviceContext dev_ctx(GPUPlace(i));
    ASSERT_NE(nullptr, boost::get<CUDADeviceContext>(dev_ctx).GetEigenDevice());
    ASSERT_NE(nullptr, boost::get<CUDADeviceContext>(dev_ctx).cudnn_handle());
    ASSERT_NE(nullptr, boost::get<CUDADeviceContext>(dev_ctx).cublas_handle());
    ASSERT_NE(nullptr, boost::get<CUDADeviceContext>(dev_ctx).stream());
  }
}
#endif  // PADDLE_WITH_CUDA

TEST(DeviceContext, CPU) {
  using paddle::platform::DeviceContext;
  using paddle::platform::CPUDeviceContext;

  DeviceContext dev_ctx;  // defaults to CPUPlace
  ASSERT_NE(nullptr, boost::get<CUDADeviceContext>(dev_ctx).GetEigenDevice());
}
