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

#include <gtest/gtest.h>
#include <memory>

#include "paddle/pten/kernels/scale_kernel.h"

#include "paddle/pten/api/lib/utils/allocator.h"
#include "paddle/pten/core/dense_tensor.h"
#include "paddle/pten/core/kernel_registry.h"

namespace pten {
namespace tests {

namespace framework = paddle::framework;
using DDim = paddle::framework::DDim;

TEST(DEV_API, scale) {
  // 1. create tensor
  const auto alloc = std::make_unique<paddle::experimental::DefaultAllocator>(
      paddle::platform::CPUPlace());
  pten::DenseTensor dense_x(alloc.get(),
                            pten::DenseTensorMeta(pten::DataType::FLOAT32,
                                                  framework::make_ddim({3, 4}),
                                                  pten::DataLayout::NCHW));

  auto* dense_x_data = dense_x.mutable_data<float>();
  for (size_t i = 0; i < 12; ++i) {
    dense_x_data[i] = i * 1.0;
  }
  float scale = 2;
  float bias = 1;
  bool bias_after_scale = true;

  paddle::platform::DeviceContextPool& pool =
      paddle::platform::DeviceContextPool::Instance();
  auto* dev_ctx = pool.Get(paddle::platform::CPUPlace());

  // 2. test API
  auto out = pten::Scale<float>(
      *(static_cast<paddle::platform::CPUDeviceContext*>(dev_ctx)),
      dense_x,
      scale,
      bias,
      bias_after_scale);

  // 3. check result
  ASSERT_EQ(out.dims().size(), 2);
  ASSERT_EQ(out.numel(), 12);
  ASSERT_EQ(out.meta().dtype, pten::DataType::FLOAT32);
  ASSERT_EQ(out.meta().layout, pten::DataLayout::NCHW);

  auto expect_result = 23;
  auto actual_result = out.data<float>()[11];
  ASSERT_NEAR(expect_result, actual_result, 1e-6f);
}

TEST(DEV_API, scale_host) {
  // 1. create tensor
  const auto alloc = std::make_unique<paddle::experimental::DefaultAllocator>(
      paddle::platform::CPUPlace());
  pten::DenseTensor dense_x(alloc.get(),
                            pten::DenseTensorMeta(pten::DataType::FLOAT32,
                                                  framework::make_ddim({3, 4}),
                                                  pten::DataLayout::NCHW));
  auto* dense_x_data = dense_x.mutable_data<float>();
  for (size_t i = 0; i < 12; ++i) {
    dense_x_data[i] = i * 1.0;
  }

  pten::DenseTensor scale(alloc.get(),
                          pten::DenseTensorMeta(pten::DataType::FLOAT32,
                                                framework::make_ddim({1}),
                                                pten::DataLayout::NCHW));
  scale.mutable_data<float>()[0] = 2;
  float bias = 1;
  bool bias_after_scale = true;

  paddle::platform::DeviceContextPool& pool =
      paddle::platform::DeviceContextPool::Instance();
  auto* dev_ctx = pool.Get(paddle::platform::CPUPlace());

  // 2. test API
  auto out = pten::Scale<float>(
      *(static_cast<paddle::platform::CPUDeviceContext*>(dev_ctx)),
      dense_x,
      scale,
      bias,
      bias_after_scale);

  // 3. check result
  ASSERT_EQ(out.dims().size(), 2);
  ASSERT_EQ(out.numel(), 12);
  ASSERT_EQ(out.meta().dtype, pten::DataType::FLOAT32);
  ASSERT_EQ(out.meta().layout, pten::DataLayout::NCHW);

  auto expect_result = 23;
  auto actual_result = out.data<float>()[11];
  ASSERT_NEAR(expect_result, actual_result, 1e-6f);
}

}  // namespace tests
}  // namespace pten
