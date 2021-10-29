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
#include "paddle/pten/hapi/include/manipulation.h"

#include "paddle/pten/core/dense_tensor.h"
#include "paddle/pten/core/kernel_registry.h"
#include "paddle/pten/hapi/lib/utils/allocator.h"

PT_DECLARE_MODULE(ManipulationCPU);

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
PT_DECLARE_MODULE(ManipulationCUDA);
#endif

namespace framework = paddle::framework;
using DDim = paddle::framework::DDim;

// TODO(chenweihang): Remove this test after the API is used in the dygraph
TEST(API, concat) {
  // 1. create tensor
  const auto alloc1 = std::make_shared<paddle::experimental::DefaultAllocator>(
      paddle::platform::CPUPlace());
  auto x1 = std::make_shared<pten::DenseTensor>(
      alloc1,
      pten::DenseTensorMeta(pten::DataType::FLOAT32,
                            framework::make_ddim({2, 3}),
                            pten::DataLayout::NCHW));
  auto* x1_data = x1->mutable_data<float>();

  for (int i = 0; i < x1->numel(); i++) {
    x1_data[i] = i;
  }

  const auto alloc2 = std::make_shared<paddle::experimental::DefaultAllocator>(
      paddle::platform::CPUPlace());
  auto x2 = std::make_shared<pten::DenseTensor>(
      alloc2,
      pten::DenseTensorMeta(pten::DataType::FLOAT32,
                            framework::make_ddim({2, 3}),
                            pten::DataLayout::NCHW));
  auto* x2_data = x2->mutable_data<float>();

  for (int i = 0; i < x2->numel(); i++) {
    x2_data[i] = i;
  }

  paddle::experimental::Tensor x_t1(x1);
  paddle::experimental::Tensor x_t2(x2);
  std::vector<paddle::experimental::Tensor> x = {x_t1, x_t2};
  int axis = 0;
  // 2. test API
  auto out = paddle::experimental::concat(x, axis);

  // 3. check result
  std::vector<int> expect_shape = {4, 3};
  ASSERT_EQ(out.shape()[0], expect_shape[0]);
  ASSERT_EQ(out.shape()[1], expect_shape[1]);
  ASSERT_EQ(out.numel(), 12);
  ASSERT_EQ(out.is_cpu(), true);
  ASSERT_EQ(out.type(), pten::DataType::FLOAT32);
  ASSERT_EQ(out.layout(), pten::DataLayout::NCHW);
  ASSERT_EQ(out.initialized(), true);

  auto dense_out = std::dynamic_pointer_cast<pten::DenseTensor>(out.impl());
  auto out_data = dense_out->data<float>();

  for (int i = 0; i < 2; ++i) {
    for (int j = 0; j < 3; ++j) {
      ASSERT_NEAR(x1_data[i * 3 + j], out_data[i * 3 + j], 1e-6f);
    }
  }

  for (int i = 2; i < 4; ++i) {
    for (int j = 0; j < 3; ++j) {
      ASSERT_NEAR(x2_data[(i - 2) * 3 + j], out_data[i * 3 + j], 1e-6f);
    }
  }
}
