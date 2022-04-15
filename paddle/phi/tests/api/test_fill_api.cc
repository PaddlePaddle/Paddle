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

PD_DECLARE_KERNEL(full, CPU, ALL_LAYOUT);

namespace paddle {
namespace tests {

namespace framework = paddle::framework;
using DDim = phi::DDim;

// TODO(chenweihang): Remove this test after the API is used in the dygraph
TEST(API, full_like) {
  // 1. create tensor
  const auto alloc = std::make_unique<paddle::experimental::DefaultAllocator>(
      paddle::platform::CPUPlace());
  auto dense_x = std::make_shared<phi::DenseTensor>(
      alloc.get(),
      phi::DenseTensorMeta(phi::DataType::FLOAT32,
                           phi::make_ddim({3, 2}),
                           phi::DataLayout::NCHW));
  auto* dense_x_data =
      dense_x->mutable_data<float>(paddle::platform::CPUPlace());
  dense_x_data[0] = 0;

  float val = 1.0;

  paddle::experimental::Tensor x(dense_x);

  // 2. test API
  auto out = paddle::experimental::full_like(x, val, phi::DataType::FLOAT32);

  // 3. check result
  ASSERT_EQ(out.dims().size(), 2);
  ASSERT_EQ(out.dims()[0], 3);
  ASSERT_EQ(out.numel(), 6);
  ASSERT_EQ(out.is_cpu(), true);
  ASSERT_EQ(out.type(), phi::DataType::FLOAT32);
  ASSERT_EQ(out.layout(), phi::DataLayout::NCHW);
  ASSERT_EQ(out.initialized(), true);

  auto dense_out = std::dynamic_pointer_cast<phi::DenseTensor>(out.impl());
  auto* actual_result = dense_out->data<float>();
  for (auto i = 0; i < 6; i++) {
    ASSERT_NEAR(actual_result[i], val, 1e-6f);
  }
}

TEST(API, zeros_like) {
  // 1. create tensor
  const auto alloc = std::make_unique<paddle::experimental::DefaultAllocator>(
      paddle::platform::CPUPlace());
  auto dense_x = std::make_shared<phi::DenseTensor>(
      alloc.get(),
      phi::DenseTensorMeta(phi::DataType::FLOAT32,
                           phi::make_ddim({3, 2}),
                           phi::DataLayout::NCHW));
  auto* dense_x_data =
      dense_x->mutable_data<float>(paddle::platform::CPUPlace());
  dense_x_data[0] = 1;

  paddle::experimental::Tensor x(dense_x);

  // 2. test API
  auto out = paddle::experimental::zeros_like(x, phi::DataType::INT32);

  // 3. check result
  ASSERT_EQ(out.dims().size(), 2);
  ASSERT_EQ(out.dims()[0], 3);
  ASSERT_EQ(out.numel(), 6);
  ASSERT_EQ(out.is_cpu(), true);
  ASSERT_EQ(out.type(), phi::DataType::INT32);
  ASSERT_EQ(out.layout(), phi::DataLayout::NCHW);
  ASSERT_EQ(out.initialized(), true);

  auto dense_out = std::dynamic_pointer_cast<phi::DenseTensor>(out.impl());
  auto* actual_result = dense_out->data<int32_t>();
  for (auto i = 0; i < 6; i++) {
    ASSERT_EQ(actual_result[i], 0);
  }
}

TEST(API, ones_like) {
  // 1. create tensor
  const auto alloc = std::make_unique<paddle::experimental::DefaultAllocator>(
      paddle::platform::CPUPlace());
  auto dense_x = std::make_shared<phi::DenseTensor>(
      alloc.get(),
      phi::DenseTensorMeta(
          phi::DataType::INT32, phi::make_ddim({3, 2}), phi::DataLayout::NCHW));
  auto* dense_x_data =
      dense_x->mutable_data<int32_t>(paddle::platform::CPUPlace());
  dense_x_data[0] = 0;

  paddle::experimental::Tensor x(dense_x);

  // 2. test API
  auto out = paddle::experimental::ones_like(x, phi::DataType::INT32);

  // 3. check result
  ASSERT_EQ(out.dims().size(), 2);
  ASSERT_EQ(out.dims()[0], 3);
  ASSERT_EQ(out.numel(), 6);
  ASSERT_EQ(out.is_cpu(), true);
  ASSERT_EQ(out.type(), phi::DataType::INT32);
  ASSERT_EQ(out.layout(), phi::DataLayout::NCHW);
  ASSERT_EQ(out.initialized(), true);

  auto dense_out = std::dynamic_pointer_cast<phi::DenseTensor>(out.impl());
  auto* actual_result = dense_out->data<int32_t>();
  for (auto i = 0; i < 6; i++) {
    ASSERT_EQ(actual_result[i], 1);
  }
}

TEST(API, full1) {
  // 1. create tensor
  const auto alloc = std::make_unique<paddle::experimental::DefaultAllocator>(
      paddle::platform::CPUPlace());

  auto dense_shape = std::make_shared<phi::DenseTensor>(
      alloc.get(),
      phi::DenseTensorMeta(
          phi::DataType::INT64, phi::make_ddim({2}), phi::DataLayout::NCHW));
  auto* shape_data =
      dense_shape->mutable_data<int64_t>(paddle::platform::CPUPlace());
  shape_data[0] = 2;
  shape_data[1] = 3;

  auto dense_scalar = std::make_shared<phi::DenseTensor>(
      alloc.get(),
      phi::DenseTensorMeta(
          phi::DataType::FLOAT32, phi::make_ddim({1}), phi::DataLayout::NCHW));
  dense_scalar->mutable_data<float>(paddle::platform::CPUPlace())[0] = 1.0;

  paddle::experimental::Tensor value(dense_scalar);

  paddle::experimental::Tensor tensor_shape(dense_shape);

  float val = 1.0;

  // 2. test API
  auto out =
      paddle::experimental::full(tensor_shape, value, phi::DataType::FLOAT32);

  // 3. check result
  ASSERT_EQ(out.shape().size(), 2UL);
  ASSERT_EQ(out.shape()[0], 2);
  ASSERT_EQ(out.numel(), 6);
  ASSERT_EQ(out.is_cpu(), true);
  ASSERT_EQ(out.type(), phi::DataType::FLOAT32);
  ASSERT_EQ(out.layout(), phi::DataLayout::NCHW);
  ASSERT_EQ(out.initialized(), true);

  auto dense_out = std::dynamic_pointer_cast<phi::DenseTensor>(out.impl());
  auto* actual_result = dense_out->data<float>();
  for (auto i = 0; i < 6; i++) {
    ASSERT_NEAR(actual_result[i], val, 1e-6f);
  }
}

TEST(API, full2) {
  const auto alloc = std::make_unique<paddle::experimental::DefaultAllocator>(
      paddle::platform::CPUPlace());

  auto dense_scalar = std::make_shared<phi::DenseTensor>(
      alloc.get(),
      phi::DenseTensorMeta(
          phi::DataType::INT32, phi::make_ddim({1}), phi::DataLayout::NCHW));
  dense_scalar->mutable_data<int>(paddle::platform::CPUPlace())[0] = 2;

  paddle::experimental::Tensor shape_scalar1(dense_scalar);
  paddle::experimental::Tensor shape_scalar2(dense_scalar);
  std::vector<paddle::experimental::Tensor> list_shape{shape_scalar1,
                                                       shape_scalar2};

  float val = 1.0;

  auto out =
      paddle::experimental::full(list_shape, val, phi::DataType::FLOAT32);

  ASSERT_EQ(out.shape().size(), 2UL);
  ASSERT_EQ(out.shape()[0], 2);
  ASSERT_EQ(out.numel(), 4);
  ASSERT_EQ(out.is_cpu(), true);
  ASSERT_EQ(out.type(), phi::DataType::FLOAT32);
  ASSERT_EQ(out.layout(), phi::DataLayout::NCHW);
  ASSERT_EQ(out.initialized(), true);

  auto dense_out = std::dynamic_pointer_cast<phi::DenseTensor>(out.impl());
  auto* actual_result = dense_out->data<float>();
  for (auto i = 0; i < 4; i++) {
    ASSERT_NEAR(actual_result[i], val, 1e-6f);
  }
}

TEST(API, full3) {
  std::vector<int64_t> vector_shape{2, 3};

  float val = 1.0;

  auto out =
      paddle::experimental::full(vector_shape, val, phi::DataType::INT32);

  ASSERT_EQ(out.shape().size(), 2UL);
  ASSERT_EQ(out.shape()[0], 2);
  ASSERT_EQ(out.numel(), 6);
  ASSERT_EQ(out.is_cpu(), true);
  ASSERT_EQ(out.type(), phi::DataType::INT32);
  ASSERT_EQ(out.layout(), phi::DataLayout::NCHW);
  ASSERT_EQ(out.initialized(), true);

  auto dense_out = std::dynamic_pointer_cast<phi::DenseTensor>(out.impl());
  auto* actual_result = dense_out->data<int>();
  for (auto i = 0; i < 6; i++) {
    ASSERT_EQ(actual_result[i], 1);
  }
}

}  // namespace tests
}  // namespace paddle
