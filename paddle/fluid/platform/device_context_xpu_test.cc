/* Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */
#include "paddle/fluid/platform/device_context.h"

#include <vector>

#include "glog/logging.h"
#include "gtest/gtest.h"

TEST(Device, Init) {
  using paddle::platform::DeviceContext;
  using paddle::platform::XPUDeviceContext;
  using paddle::platform::XPUPlace;

  int count = paddle::platform::GetXPUDeviceCount();
  for (int i = 0; i < count; i++) {
    XPUDeviceContext* device_context = new XPUDeviceContext(XPUPlace(i));
    xpu::Context* ctx = device_context->x_context();
    ASSERT_NE(nullptr, ctx);
    delete device_context;
  }
}

TEST(Device, DeviceContextPool) {
  using paddle::platform::DeviceContextPool;
  using paddle::platform::XPUDeviceContext;
  using paddle::platform::Place;
  using paddle::platform::CPUPlace;
  using paddle::platform::XPUPlace;

  DeviceContextPool& pool = DeviceContextPool::Instance();
  auto cpu_dev_ctx1 = pool.Get(CPUPlace());
  auto cpu_dev_ctx2 = pool.Get(CPUPlace());
  ASSERT_EQ(cpu_dev_ctx2, cpu_dev_ctx1);

  std::vector<Place> xpu_places;
  int count = paddle::platform::GetXPUDeviceCount();
  for (int i = 0; i < count; ++i) {
    auto dev_ctx = pool.Get(XPUPlace(i));
    ASSERT_NE(dev_ctx, nullptr);
  }
}
