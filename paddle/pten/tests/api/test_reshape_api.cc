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

#include "paddle/pten/api/include/api.h"

#include "paddle/pten/api/lib/utils/allocator.h"
#include "paddle/pten/core/dense_tensor.h"
#include "paddle/pten/core/kernel_registry.h"

namespace paddle {
namespace tests {

namespace framework = paddle::framework;
using DDim = paddle::framework::DDim;

// TODO(chenweihang): Remove this test after the API is used in the dygraph
TEST(API, reshape) {
  // 1. create tensor
  const auto alloc = std::make_shared<paddle::experimental::DefaultAllocator>(
      paddle::platform::CPUPlace());
  auto dense_x = std::make_shared<pten::DenseTensor>(
      alloc,
      pten::DenseTensorMeta(pten::DataType::FLOAT32,
                            framework::make_ddim({3, 2, 2, 3}),
                            pten::DataLayout::NCHW));
  auto* dense_x_data = dense_x->mutable_data<float>();

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
  ASSERT_EQ(out.type(), pten::DataType::FLOAT32);
  ASSERT_EQ(out.layout(), pten::DataLayout::NCHW);
  ASSERT_EQ(out.initialized(), true);
  bool value_equal = true;
  auto dense_out = std::dynamic_pointer_cast<pten::DenseTensor>(out.impl());
  auto* dense_out_data = dense_out->data<float>();
  for (int i = 0; i < dense_x->numel(); i++) {
    if (std::abs(dense_x_data[i] - dense_out_data[i]) > 1e-6f)
      value_equal = false;
  }
  ASSERT_EQ(value_equal, true);
}

TEST(Tensor, old_reshape) {
  paddle::experimental::Tensor x(paddle::PlaceType::kCPU);
  x.reshape({3, 4});

  ASSERT_EQ(x.shape()[0], 3);
  ASSERT_EQ(x.shape()[1], 4);
  ASSERT_EQ(x.numel(), 12);
  ASSERT_EQ(x.is_cpu(), true);
  ASSERT_EQ(x.type(), pten::DataType::UNDEFINED);
  ASSERT_EQ(x.layout(), pten::DataLayout::NCHW);
  ASSERT_EQ(x.initialized(), false);
}

}  // namespace tests
}  // namespace paddle
