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

#include "paddle/pten/hapi/include/math.h"

#include "paddle/pten/core/dense_tensor.h"
#include "paddle/pten/core/kernel_registry.h"

#include "paddle/pten/api/include/math.h"

PT_DECLARE_MODULE(MathCPU);

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
PT_DECLARE_MODULE(MathCUDA);
#endif

namespace framework = paddle::framework;
using DDim = paddle::framework::DDim;

// TODO(chenweihang): Remove this test after the API is used in the dygraph
TEST(API, mean) {
  // 1. create tensor
  auto dense_x = std::make_shared<pten::DenseTensor>(
      pten::TensorMeta(framework::make_ddim({3, 4}),
                       pten::Backend::CPU,
                       pten::DataType::FLOAT32,
                       pten::DataLayout::NCHW),
      pten::TensorStatus());
  auto* dense_x_data = dense_x->mutable_data<float>();

  float sum = 0.0;
  for (size_t i = 0; i < 12; ++i) {
    dense_x_data[i] = i * 1.0;
    sum += i * 1.0;
  }

  paddle::experimental::Tensor x(dense_x);

  // 2. test API
  auto out = paddle::experimental::mean(x);

  // 3. check result
  ASSERT_EQ(out.shape().size(), 1);
  ASSERT_EQ(out.shape()[0], 1);
  ASSERT_EQ(out.numel(), 1);
  ASSERT_EQ(out.is_cpu(), true);
  ASSERT_EQ(out.type(), pten::DataType::FLOAT32);
  ASSERT_EQ(out.layout(), pten::DataLayout::NCHW);
  ASSERT_EQ(out.initialized(), true);

  auto expect_result = sum / 12;
  auto dense_out = std::dynamic_pointer_cast<pten::DenseTensor>(out.impl());
  auto actual_result = dense_out->data<float>()[0];
  ASSERT_NEAR(expect_result, actual_result, 1e-6f);
}

TEST(DEV_API, mean) {
  // 1. create tensor
  pten::DenseTensor dense_x(pten::TensorMeta(framework::make_ddim({3, 4}),
                                             pten::Backend::CPU,
                                             pten::DataType::FLOAT32,
                                             pten::DataLayout::NCHW),
                            pten::TensorStatus());
  auto* dense_x_data = dense_x.mutable_data<float>();

  float sum = 0.0;
  for (size_t i = 0; i < 12; ++i) {
    dense_x_data[i] = i * 1.0;
    sum += i * 1.0;
  }
  paddle::platform::DeviceContextPool& pool =
      paddle::platform::DeviceContextPool::Instance();
  auto* dev_ctx = pool.Get(paddle::platform::CPUPlace());
  // 2. test API
  auto out = pten::Mean<float>(
      *(static_cast<paddle::platform::CPUDeviceContext*>(dev_ctx)), dense_x);

  // 3. check result
  ASSERT_EQ(out.dims().size(), 1);
  ASSERT_EQ(out.numel(), 1);
  ASSERT_EQ(out.meta().backend, pten::Backend::CPU);
  ASSERT_EQ(out.meta().type, pten::DataType::FLOAT32);
  ASSERT_EQ(out.meta().layout, pten::DataLayout::NCHW);

  auto expect_result = sum / 12;
  auto actual_result = out.data<float>()[0];
  ASSERT_NEAR(expect_result, actual_result, 1e-6f);
}
