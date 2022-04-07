/* Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */
#include "paddle/fluid/platform/device/mlu/device_context.h"

#include <vector>

#include "glog/logging.h"
#include "gtest/gtest.h"

TEST(Device, Init) {
  using paddle::platform::DeviceContext;
  using paddle::platform::MLUDeviceContext;
  using paddle::platform::MLUPlace;
  using paddle::platform::MLUContext;

  int count = paddle::platform::GetMLUDeviceCount();
  for (int i = 0; i < count; i++) {
    MLUDeviceContext* device_context = new MLUDeviceContext(MLUPlace(i));
    std::shared_ptr<MLUContext> ctx = device_context->context();
    ASSERT_NE(nullptr, ctx);
    delete device_context;
  }
}

TEST(Device, MLUDeviceContext) {
  using paddle::platform::MLUDeviceContext;
  using paddle::platform::MLUPlace;
  using paddle::mluCnnlHandle;

  int count = paddle::platform::GetMLUDeviceCount();
  for (int i = 0; i < count; i++) {
    MLUDeviceContext* device_context = new MLUDeviceContext(MLUPlace(i));
    mluCnnlHandle mlu_handle = device_context->cnnl_handle();
    ASSERT_NE(nullptr, mlu_handle);
    delete device_context;
  }
}

TEST(Device, MLUStream) {
  using paddle::platform::MLUDeviceContext;
  using paddle::platform::MLUPlace;
  using paddle::mluStream;

  int count = paddle::platform::GetMLUDeviceCount();
  for (int i = 0; i < count; i++) {
    MLUDeviceContext* device_context = new MLUDeviceContext(MLUPlace(i));
    mluStream mlu_stream = device_context->stream();
    ASSERT_NE(nullptr, mlu_stream);
    delete device_context;
  }
}

TEST(Device, DeviceContextPool) {
  using paddle::platform::DeviceContextPool;
  using paddle::platform::MLUDeviceContext;
  using paddle::platform::Place;
  using paddle::platform::CPUPlace;
  using paddle::platform::MLUPlace;

  DeviceContextPool& pool = DeviceContextPool::Instance();
  auto cpu_dev_ctx1 = pool.Get(CPUPlace());
  auto cpu_dev_ctx2 = pool.Get(CPUPlace());
  ASSERT_EQ(cpu_dev_ctx2, cpu_dev_ctx1);

  std::vector<Place> mlu_places;
  int count = paddle::platform::GetMLUDeviceCount();
  for (int i = 0; i < count; ++i) {
    auto dev_ctx = pool.Get(MLUPlace(i));
    ASSERT_NE(dev_ctx, nullptr);
  }
}
