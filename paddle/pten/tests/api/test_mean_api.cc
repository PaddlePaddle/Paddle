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
using DDim = pten::framework::DDim;

// TODO(chenweihang): Remove this test after the API is used in the dygraph
TEST(API, mean) {
  // 1. create tensor
  const auto alloc = std::make_unique<paddle::experimental::DefaultAllocator>(
      paddle::platform::CPUPlace());
  auto dense_x = std::make_shared<pten::DenseTensor>(
      alloc.get(),
      pten::DenseTensorMeta(pten::DataType::FLOAT32,
                            framework::make_ddim({3, 4}),
                            pten::DataLayout::NCHW));
  auto* dense_x_data = dense_x->mutable_data<float>();

  float sum = 0.0;
  for (size_t i = 0; i < 12; ++i) {
    dense_x_data[i] = i * 1.0;
    sum += i * 1.0;
  }

  paddle::experimental::Tensor x(dense_x);
  std::vector<int64_t> axis = {0, 1};

  // 2. test API
  auto out = paddle::experimental::mean(x, axis, false);

  // 3. check result
  ASSERT_EQ(out.dims().size(), 1);
  ASSERT_EQ(out.dims()[0], 1);
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

}  // namespace tests
}  // namespace paddle
