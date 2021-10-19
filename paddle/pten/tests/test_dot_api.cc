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

#include "paddle/pten/hapi/include/linalg.h"

#include "paddle/pten/core/dense_tensor.h"
#include "paddle/pten/core/kernel_registry.h"

PT_DECLARE_MODULE(LinalgCPU);

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
PT_DECLARE_MODULE(LinalgCUDA);
#endif

namespace framework = paddle::framework;
using DDim = paddle::framework::DDim;

TEST(API, dot) {
  // 1. create tensor
  auto dense_x = std::make_shared<pten::DenseTensor>(
      pten::TensorMeta(framework::make_ddim({3, 10}),
                       pten::Backend::kCPU,
                       pten::DataType::kFLOAT32,
                       pten::DataLayout::kNCHW),
      pten::TensorStatus());
  auto* dense_x_data = dense_x->mutable_data<float>();

  auto dense_y = std::make_shared<pten::DenseTensor>(
      pten::TensorMeta(framework::make_ddim({3, 10}),
                       pten::Backend::kCPU,
                       pten::DataType::kFLOAT32,
                       pten::DataLayout::kNCHW),
      pten::TensorStatus());
  auto* dense_y_data = dense_y->mutable_data<float>();

  float sum[3] = {0.0, 0.0, 0.0};
  for (size_t i = 0; i < 3; ++i) {
    for (size_t j = 0; j < 10; ++j) {
      dense_x_data[i * 10 + j] = (i * 10 + j) * 1.0;
      dense_y_data[i * 10 + j] = (i * 10 + j) * 1.0;
      sum[i] += (i * 10 + j) * (i * 10 + j) * 1.0;
    }
  }

  paddle::experimental::Tensor x(dense_x);
  paddle::experimental::Tensor y(dense_y);

  // 2. test API
  auto out = paddle::experimental::dot(x, y);

  // 3. check result
  ASSERT_EQ(out.shape().size(), 2);
  ASSERT_EQ(out.shape()[0], 3);
  ASSERT_EQ(out.numel(), 3);
  ASSERT_EQ(out.is_cpu(), true);
  ASSERT_EQ(out.type(), pten::DataType::kFLOAT32);
  ASSERT_EQ(out.layout(), pten::DataLayout::kNCHW);
  ASSERT_EQ(out.initialized(), true);

  auto expect_result = sum;
  auto dense_out = std::dynamic_pointer_cast<pten::DenseTensor>(out.impl());
  auto actual_result0 = dense_out->data<float>()[0];
  auto actual_result1 = dense_out->data<float>()[1];
  auto actual_result2 = dense_out->data<float>()[2];
  ASSERT_NEAR(expect_result[0], actual_result0, 1e-6f);
  ASSERT_NEAR(expect_result[1], actual_result1, 1e-6f);
  ASSERT_NEAR(expect_result[2], actual_result2, 1e-6f);
}
