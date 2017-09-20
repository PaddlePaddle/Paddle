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

#include "paddle/platform/device_context_manager.h"
#include "gtest/gtest.h"

TEST(DeviceContextManager, CPU) {
  paddle::platform::CPUPlace place;
  paddle::platform::CPUDeviceContext* ctx =
      paddle::platform::DeviceContextManager::Get()->GetDeviceContext(place);
  ASSERT_NE(nullptr, ctx);
}

#ifndef PADDLE_ONLY_CPU
TEST(DeviceContextManager, GPU) {
  paddle::platform::GPUPlace place(0);
  paddle::platform::CUDADeviceContext* ctx =
      paddle::platform::DeviceContextManager::Get()->GetDeviceContext(place);
  ASSERT_NE(nullptr, ctx);
}
#endif
