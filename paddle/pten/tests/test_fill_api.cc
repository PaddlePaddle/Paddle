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

#include "paddle/pten/hapi/include/creation.h"

#include "paddle/pten/core/dense_tensor.h"
#include "paddle/pten/core/kernel_registry.h"
#include "paddle/pten/hapi/lib/utils/allocator.h"

#include "paddle/pten/api/include/creation.h"

PT_DECLARE_MODULE(CreationCPU);

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
PT_DECLARE_MODULE(CreationCUDA);
#endif

namespace framework = paddle::framework;
using DDim = paddle::framework::DDim;

// TODO(chenweihang): Remove this test after the API is used in the dygraph
TEST(API, full_like) {
  // 1. create tensor
  const auto alloc = std::make_shared<paddle::experimental::DefaultAllocator>(
      paddle::platform::CPUPlace());
  auto dense_x = std::make_shared<pten::DenseTensor>(
      alloc,
      pten::DenseTensorMeta(pten::DataType::FLOAT32,
                            framework::make_ddim({3, 2}),
                            pten::DataLayout::NCHW));
  auto* dense_x_data = dense_x->mutable_data<float>();
  dense_x_data[0] = 0;

  float val = 1.0;

  paddle::experimental::Tensor x(dense_x);

  // 2. test API
  auto out = paddle::experimental::full_like(x, val, pten::DataType::FLOAT32);

  // 3. check result
  ASSERT_EQ(out.shape().size(), 2);
  ASSERT_EQ(out.shape()[0], 3);
  ASSERT_EQ(out.numel(), 6);
  ASSERT_EQ(out.is_cpu(), true);
  ASSERT_EQ(out.type(), pten::DataType::FLOAT32);
  ASSERT_EQ(out.layout(), pten::DataLayout::NCHW);
  ASSERT_EQ(out.initialized(), true);

  auto dense_out = std::dynamic_pointer_cast<pten::DenseTensor>(out.impl());
  auto* actual_result = dense_out->data<float>();
  for (auto i = 0; i < 6; i++) {
    ASSERT_NEAR(actual_result[i], val, 1e-6f);
  }
}

TEST(API, zeros_like) {
  // 1. create tensor
  const auto alloc = std::make_shared<paddle::experimental::DefaultAllocator>(
      paddle::platform::CPUPlace());
  auto dense_x = std::make_shared<pten::DenseTensor>(
      alloc,
      pten::DenseTensorMeta(pten::DataType::FLOAT32,
                            framework::make_ddim({3, 2}),
                            pten::DataLayout::NCHW));
  auto* dense_x_data = dense_x->mutable_data<float>();
  dense_x_data[0] = 1;

  paddle::experimental::Tensor x(dense_x);

  // 2. test API
  auto out = paddle::experimental::zeros_like(x, pten::DataType::FLOAT32);

  // 3. check result
  ASSERT_EQ(out.shape().size(), 2);
  ASSERT_EQ(out.shape()[0], 3);
  ASSERT_EQ(out.numel(), 6);
  ASSERT_EQ(out.is_cpu(), true);
  ASSERT_EQ(out.type(), pten::DataType::FLOAT32);
  ASSERT_EQ(out.layout(), pten::DataLayout::NCHW);
  ASSERT_EQ(out.initialized(), true);

  auto dense_out = std::dynamic_pointer_cast<pten::DenseTensor>(out.impl());
  auto* actual_result = dense_out->data<float>();
  for (auto i = 0; i < 6; i++) {
    ASSERT_NEAR(actual_result[i], 0, 1e-6f);
  }
}

TEST(API, ones_like) {
  // 1. create tensor
  const auto alloc = std::make_shared<paddle::experimental::DefaultAllocator>(
      paddle::platform::CPUPlace());
  auto dense_x = std::make_shared<pten::DenseTensor>(
      alloc,
      pten::DenseTensorMeta(pten::DataType::INT32,
                            framework::make_ddim({3, 2}),
                            pten::DataLayout::NCHW));
  auto* dense_x_data = dense_x->mutable_data<int32_t>();
  dense_x_data[0] = 0;

  paddle::experimental::Tensor x(dense_x);

  // 2. test API
  auto out = paddle::experimental::ones_like(x, pten::DataType::INT32);

  // 3. check result
  ASSERT_EQ(out.shape().size(), 2);
  ASSERT_EQ(out.shape()[0], 3);
  ASSERT_EQ(out.numel(), 6);
  ASSERT_EQ(out.is_cpu(), true);
  ASSERT_EQ(out.type(), pten::DataType::INT32);
  ASSERT_EQ(out.layout(), pten::DataLayout::NCHW);
  ASSERT_EQ(out.initialized(), true);

  auto dense_out = std::dynamic_pointer_cast<pten::DenseTensor>(out.impl());
  auto* actual_result = dense_out->data<int32_t>();
  for (auto i = 0; i < 6; i++) {
    ASSERT_EQ(actual_result[i], 1);
  }
}

TEST(DEV_API, fill_any_like) {
  // 1. create tensor
  const auto alloc = std::make_shared<paddle::experimental::DefaultAllocator>(
      paddle::platform::CPUPlace());
  pten::DenseTensor dense_x(alloc,
                            pten::DenseTensorMeta(pten::DataType::FLOAT32,
                                                  framework::make_ddim({3, 2}),
                                                  pten::DataLayout::NCHW));
  auto* dense_x_data = dense_x.mutable_data<float>();
  dense_x_data[0] = 0;
  float val = 1.0;

  paddle::platform::DeviceContextPool& pool =
      paddle::platform::DeviceContextPool::Instance();
  auto* dev_ctx = pool.Get(paddle::platform::CPUPlace());

  // 2. test API
  auto out = pten::FillAnyLike<float>(
      *(static_cast<paddle::platform::CPUDeviceContext*>(dev_ctx)),
      dense_x,
      val);

  // 3. check result
  ASSERT_EQ(out.dims().size(), 2);
  ASSERT_EQ(out.dims()[0], 3);
  ASSERT_EQ(out.numel(), 6);
  ASSERT_EQ(out.meta().type, pten::DataType::FLOAT32);
  ASSERT_EQ(out.meta().layout, pten::DataLayout::NCHW);

  auto* actual_result = out.data<float>();
  for (auto i = 0; i < 6; i++) {
    ASSERT_NEAR(actual_result[i], val, 1e-6f);
  }
}
