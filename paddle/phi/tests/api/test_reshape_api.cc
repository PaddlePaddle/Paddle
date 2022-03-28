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
PD_DECLARE_KERNEL(reshape, CPU, ALL_LAYOUT);

namespace paddle {
namespace tests {

namespace framework = paddle::framework;
using DDim = phi::DDim;

// TODO(chenweihang): Remove this test after the API is used in the dygraph
TEST(API, reshape) {
  // 1. create tensor
  const auto alloc = std::make_unique<paddle::experimental::DefaultAllocator>(
      paddle::platform::CPUPlace());
  auto dense_x = std::make_shared<phi::DenseTensor>(
      alloc.get(),
      phi::DenseTensorMeta(phi::DataType::FLOAT32,
                           phi::make_ddim({3, 2, 2, 3}),
                           phi::DataLayout::NCHW));
  auto* dense_x_data =
      dense_x->mutable_data<float>(paddle::platform::CPUPlace());

  for (int i = 0; i < dense_x->numel(); i++) {
    dense_x_data[i] = i;
  }

  paddle::experimental::Tensor x(dense_x);
  std::vector<int64_t> shape{12, 3};
  // 2. test API
  auto out = paddle::experimental::reshape(x, shape);
  // 3. check result
  std::vector<int64_t> expect_shape = {12, 3};
  ASSERT_EQ(out.shape()[0], expect_shape[0]);
  ASSERT_EQ(out.shape()[1], expect_shape[1]);
  ASSERT_EQ(out.numel(), 36);
  ASSERT_EQ(out.is_cpu(), true);
  ASSERT_EQ(out.type(), phi::DataType::FLOAT32);
  ASSERT_EQ(out.layout(), phi::DataLayout::NCHW);
  ASSERT_EQ(out.initialized(), true);
  bool value_equal = true;
  auto dense_out = std::dynamic_pointer_cast<phi::DenseTensor>(out.impl());
  auto* dense_out_data = dense_out->data<float>();
  for (int i = 0; i < dense_x->numel(); i++) {
    if (std::abs(dense_x_data[i] - dense_out_data[i]) > 1e-6f)
      value_equal = false;
  }
  ASSERT_EQ(value_equal, true);
}

TEST(API, reshape_) {
  // 1. create tensor
  auto x = paddle::experimental::full(
      {3, 2, 2, 3}, 1.0, experimental::DataType::FLOAT32);

  // 2. test API
  paddle::experimental::Tensor out = paddle::experimental::reshape_(x, {12, 3});
  // 3. check result
  std::vector<int64_t> expect_shape = {12, 3};
  ASSERT_EQ(out.shape()[0], expect_shape[0]);
  ASSERT_EQ(out.shape()[1], expect_shape[1]);
  ASSERT_EQ(out.numel(), 36);
  ASSERT_EQ(out.is_cpu(), true);
  ASSERT_EQ(out.type(), phi::DataType::FLOAT32);
  ASSERT_EQ(out.layout(), phi::DataLayout::NCHW);
  ASSERT_EQ(out.initialized(), true);
  ASSERT_EQ(out.data<float>(), x.data<float>());
}

TEST(Tensor, old_reshape) {
  paddle::experimental::Tensor x(paddle::PlaceType::kCPU);
  x.reshape({3, 4});
  x.mutable_data<float>(paddle::PlaceType::kCPU);

  ASSERT_EQ(x.shape()[0], 3);
  ASSERT_EQ(x.shape()[1], 4);
  ASSERT_EQ(x.numel(), 12);
  ASSERT_EQ(x.is_cpu(), true);
  ASSERT_EQ(x.type(), phi::DataType::FLOAT32);
  ASSERT_EQ(x.layout(), phi::DataLayout::NCHW);
  ASSERT_EQ(x.initialized(), true);
}

}  // namespace tests
}  // namespace paddle
