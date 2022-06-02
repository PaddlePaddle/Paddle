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

#include "paddle/phi/api/include/api.h"

#include "paddle/phi/api/lib/utils/allocator.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/kernel_registry.h"

PD_DECLARE_KERNEL(add, CPU, ALL_LAYOUT);
PD_DECLARE_KERNEL(subtract, CPU, ALL_LAYOUT);
PD_DECLARE_KERNEL(multiply, CPU, ALL_LAYOUT);
PD_DECLARE_KERNEL(divide, CPU, ALL_LAYOUT);

namespace paddle {
namespace tests {

namespace framework = paddle::framework;
using DDim = phi::DDim;

// TODO(chenweihang): Remove this test after the API is used in the dygraph
TEST(API, add) {
  // 1. create tensor
  const auto alloc = std::make_unique<paddle::experimental::DefaultAllocator>(
      paddle::platform::CPUPlace());
  auto dense_x = std::make_shared<phi::DenseTensor>(
      alloc.get(),
      phi::DenseTensorMeta(phi::DataType::FLOAT32,
                           phi::make_ddim({3, 10}),
                           phi::DataLayout::NCHW));
  auto* dense_x_data =
      dense_x->mutable_data<float>(paddle::platform::CPUPlace());

  auto dense_y = std::make_shared<phi::DenseTensor>(
      alloc.get(),
      phi::DenseTensorMeta(
          phi::DataType::FLOAT32, phi::make_ddim({10}), phi::DataLayout::NCHW));
  auto* dense_y_data =
      dense_y->mutable_data<float>(paddle::platform::CPUPlace());

  float sum[3][10] = {0.0};
  for (size_t i = 0; i < 3; ++i) {
    for (size_t j = 0; j < 10; ++j) {
      dense_x_data[i * 10 + j] = (i * 10 + j) * 1.0;
      sum[i][j] = (i * 10 + j) * 1.0 + j * 2.0;
    }
  }
  for (size_t i = 0; i < 10; ++i) {
    dense_y_data[i] = i * 2.0;
  }
  paddle::experimental::Tensor x(dense_x);
  paddle::experimental::Tensor y(dense_y);

  // 2. test API
  auto out = paddle::experimental::add(x, y);

  // 3. check result
  ASSERT_EQ(out.shape().size(), 2UL);
  ASSERT_EQ(out.shape()[0], 3);
  ASSERT_EQ(out.numel(), 30);
  ASSERT_EQ(out.is_cpu(), true);
  ASSERT_EQ(out.type(), phi::DataType::FLOAT32);
  ASSERT_EQ(out.layout(), phi::DataLayout::NCHW);
  ASSERT_EQ(out.initialized(), true);

  auto expect_result = sum;
  auto dense_out = std::dynamic_pointer_cast<phi::DenseTensor>(out.impl());
  auto actual_result0 = dense_out->data<float>()[0];
  auto actual_result1 = dense_out->data<float>()[1];
  auto actual_result2 = dense_out->data<float>()[10];
  ASSERT_NEAR(expect_result[0][0], actual_result0, 1e-6f);
  ASSERT_NEAR(expect_result[0][1], actual_result1, 1e-6f);
  ASSERT_NEAR(expect_result[1][0], actual_result2, 1e-6f);
}

// TODO(chenweihang): Remove this test after the API is used in the dygraph
TEST(API, subtract) {
  // 1. create tensor
  const auto alloc = std::make_unique<paddle::experimental::DefaultAllocator>(
      paddle::platform::CPUPlace());
  auto dense_x = std::make_shared<phi::DenseTensor>(
      alloc.get(),
      phi::DenseTensorMeta(phi::DataType::FLOAT32,
                           phi::make_ddim({3, 10}),
                           phi::DataLayout::NCHW));
  auto* dense_x_data =
      dense_x->mutable_data<float>(paddle::platform::CPUPlace());

  auto dense_y = std::make_shared<phi::DenseTensor>(
      alloc.get(),
      phi::DenseTensorMeta(
          phi::DataType::FLOAT32, phi::make_ddim({10}), phi::DataLayout::NCHW));
  auto* dense_y_data =
      dense_y->mutable_data<float>(paddle::platform::CPUPlace());

  float sub[3][10] = {0.0};
  for (size_t i = 0; i < 3; ++i) {
    for (size_t j = 0; j < 10; ++j) {
      dense_x_data[i * 10 + j] = (i * 10 + j) * 1.0;
      sub[i][j] = (i * 10 + j) * 1.0 - j * 2.0;
    }
  }
  for (size_t i = 0; i < 10; ++i) {
    dense_y_data[i] = i * 2.0;
  }
  paddle::experimental::Tensor x(dense_x);
  paddle::experimental::Tensor y(dense_y);

  // 2. test API
  auto out = paddle::experimental::subtract(x, y);

  // 3. check result
  ASSERT_EQ(out.shape().size(), 2UL);
  ASSERT_EQ(out.shape()[0], 3);
  ASSERT_EQ(out.numel(), 30);
  ASSERT_EQ(out.is_cpu(), true);
  ASSERT_EQ(out.type(), phi::DataType::FLOAT32);
  ASSERT_EQ(out.layout(), phi::DataLayout::NCHW);
  ASSERT_EQ(out.initialized(), true);

  auto expect_result = sub;
  auto dense_out = std::dynamic_pointer_cast<phi::DenseTensor>(out.impl());
  auto actual_result0 = dense_out->data<float>()[0];
  auto actual_result1 = dense_out->data<float>()[1];
  auto actual_result2 = dense_out->data<float>()[10];
  ASSERT_NEAR(expect_result[0][0], actual_result0, 1e-6f);
  ASSERT_NEAR(expect_result[0][1], actual_result1, 1e-6f);
  ASSERT_NEAR(expect_result[1][0], actual_result2, 1e-6f);
}

// TODO(chenweihang): Remove this test after the API is used in the dygraph
TEST(API, divide) {
  // 1. create tensor
  const auto alloc = std::make_unique<paddle::experimental::DefaultAllocator>(
      paddle::platform::CPUPlace());
  auto dense_x = std::make_shared<phi::DenseTensor>(
      alloc.get(),
      phi::DenseTensorMeta(phi::DataType::FLOAT32,
                           phi::make_ddim({3, 10}),
                           phi::DataLayout::NCHW));
  auto* dense_x_data =
      dense_x->mutable_data<float>(paddle::platform::CPUPlace());

  auto dense_y = std::make_shared<phi::DenseTensor>(
      alloc.get(),
      phi::DenseTensorMeta(
          phi::DataType::FLOAT32, phi::make_ddim({10}), phi::DataLayout::NCHW));
  auto* dense_y_data =
      dense_y->mutable_data<float>(paddle::platform::CPUPlace());

  float div[3][10] = {0.0};
  for (size_t i = 0; i < 3; ++i) {
    for (size_t j = 0; j < 10; ++j) {
      dense_x_data[i * 10 + j] = (i * 10 + j) * 1.0;
      div[i][j] = (i * 10 + j) * 1.0 / (j * 2.0 + 1);
    }
  }
  for (size_t i = 0; i < 10; ++i) {
    dense_y_data[i] = i * 2.0 + 1;
  }

  paddle::experimental::Tensor x(dense_x);
  paddle::experimental::Tensor y(dense_y);

  // 2. test API
  auto out = paddle::experimental::divide(x, y);

  // 3. check result
  ASSERT_EQ(out.shape().size(), 2UL);
  ASSERT_EQ(out.shape()[0], 3);
  ASSERT_EQ(out.numel(), 30);
  ASSERT_EQ(out.is_cpu(), true);
  ASSERT_EQ(out.type(), phi::DataType::FLOAT32);
  ASSERT_EQ(out.layout(), phi::DataLayout::NCHW);
  ASSERT_EQ(out.initialized(), true);

  auto expect_result = div;
  auto dense_out = std::dynamic_pointer_cast<phi::DenseTensor>(out.impl());
  auto actual_result0 = dense_out->data<float>()[0];
  auto actual_result1 = dense_out->data<float>()[1];
  auto actual_result2 = dense_out->data<float>()[10];
  ASSERT_NEAR(expect_result[0][0], actual_result0, 1e-6f);
  ASSERT_NEAR(expect_result[0][1], actual_result1, 1e-6f);
  ASSERT_NEAR(expect_result[1][0], actual_result2, 1e-6f);
}

TEST(API, multiply) {
  // 1. create tensor
  const auto alloc = std::make_unique<paddle::experimental::DefaultAllocator>(
      paddle::platform::CPUPlace());
  auto dense_x = std::make_shared<phi::DenseTensor>(
      alloc.get(),
      phi::DenseTensorMeta(phi::DataType::FLOAT32,
                           phi::make_ddim({3, 10}),
                           phi::DataLayout::NCHW));
  auto* dense_x_data =
      dense_x->mutable_data<float>(paddle::platform::CPUPlace());

  auto dense_y = std::make_shared<phi::DenseTensor>(
      alloc.get(),
      phi::DenseTensorMeta(
          phi::DataType::FLOAT32, phi::make_ddim({10}), phi::DataLayout::NCHW));
  auto* dense_y_data =
      dense_y->mutable_data<float>(paddle::platform::CPUPlace());

  float mul[3][10] = {0.0};
  for (size_t i = 0; i < 3; ++i) {
    for (size_t j = 0; j < 10; ++j) {
      dense_x_data[i * 10 + j] = (i * 10 + j) * 1.0;
      mul[i][j] = (i * 10 + j) * 1.0 * j * 2.0;
    }
  }
  for (size_t i = 0; i < 10; ++i) {
    dense_y_data[i] = i * 2.0;
  }
  paddle::experimental::Tensor x(dense_x);
  paddle::experimental::Tensor y(dense_y);

  // 2. test API
  auto out = paddle::experimental::multiply(x, y);

  // 3. check result
  ASSERT_EQ(out.shape().size(), 2UL);
  ASSERT_EQ(out.shape()[0], 3);
  ASSERT_EQ(out.numel(), 30);
  ASSERT_EQ(out.is_cpu(), true);
  ASSERT_EQ(out.type(), phi::DataType::FLOAT32);
  ASSERT_EQ(out.layout(), phi::DataLayout::NCHW);
  ASSERT_EQ(out.initialized(), true);

  auto expect_result = mul;
  auto dense_out = std::dynamic_pointer_cast<phi::DenseTensor>(out.impl());
  auto actual_result0 = dense_out->data<float>()[0];
  auto actual_result1 = dense_out->data<float>()[1];
  auto actual_result2 = dense_out->data<float>()[10];
  ASSERT_NEAR(expect_result[0][0], actual_result0, 1e-6f);
  ASSERT_NEAR(expect_result[0][1], actual_result1, 1e-6f);
  ASSERT_NEAR(expect_result[1][0], actual_result2, 1e-6f);
}
}  // namespace tests
}  // namespace paddle
