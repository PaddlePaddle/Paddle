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

#include "paddle/tcmpt/api/include/math.h"
#include "paddle/tcmpt/core/dense_tensor.h"

namespace framework = paddle::framework;
using DDim = paddle::framework::DDim;

TEST(API, mean) {
  // 1. create tensor
  auto dense_x = std::make_shared<pt::DenseTensor>(
      pt::TensorMeta(framework::make_ddim({3, 4}),
                     pt::Backend::kCPU,
                     pt::DataType::kFLOAT32,
                     pt::DataLayout::kNCHW),
      pt::TensorStatus());
  auto* dense_x_data = dense_x->mutable_data<float>();

  float sum = 0.0;
  for (size_t i = 0; i < 12; ++i) {
    dense_x_data[i] = i * 1.0;
    sum += i * 1.0;
  }

  pt::Tensor x(dense_x);

  // 2. test API
  auto out = pt::mean(x);

  // 3. check result
  ASSERT_EQ(out.shape().size(), 1);
  ASSERT_EQ(out.shape()[0], 1);
  ASSERT_EQ(out.numel(), 1);
  ASSERT_EQ(out.is_cpu(), true);
  ASSERT_EQ(out.type(), pt::DataType::kFLOAT32);
  ASSERT_EQ(out.layout(), pt::DataLayout::kNCHW);
  ASSERT_EQ(out.initialized(), true);

  auto expect_result = sum / 12;
  auto dense_out = std::dynamic_pointer_cast<pt::DenseTensor>(out.impl());
  auto actual_result = dense_out->data<float>()[0];
  ASSERT_NEAR(expect_result, actual_result, 1e-6f);
}
