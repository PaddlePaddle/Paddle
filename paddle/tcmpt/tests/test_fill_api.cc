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

#include "paddle/tcmpt/hapi/include/creation.h"

#include "paddle/tcmpt/core/dense_tensor.h"
#include "paddle/tcmpt/core/kernel_registry.h"

PT_DECLARE_MODULE(CreationCPU);

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
PT_DECLARE_MODULE(CreationCUDA);
#endif

namespace framework = paddle::framework;
using DDim = paddle::framework::DDim;

TEST(API, full_like) {
  // 1. create tensor
  auto dense_x = std::make_shared<pt::DenseTensor>(
      pt::TensorMeta(framework::make_ddim({3, 2}),
                     pt::Backend::kCPU,
                     pt::DataType::kFLOAT32,
                     pt::DataLayout::kNCHW),
      pt::TensorStatus());
  auto* dense_x_data = dense_x->mutable_data<float>();
  dense_x_data[0] = 0;

  float val = 1.0;

  paddle::experimental::Tensor x(dense_x);

  // 2. test API
  auto out = paddle::experimental::full_like(x, val, pt::DataType::kFLOAT32);

  // 3. check result
  ASSERT_EQ(out.shape().size(), 2);
  ASSERT_EQ(out.shape()[0], 3);
  ASSERT_EQ(out.numel(), 6);
  ASSERT_EQ(out.is_cpu(), true);
  ASSERT_EQ(out.type(), pt::DataType::kFLOAT32);
  ASSERT_EQ(out.layout(), pt::DataLayout::kNCHW);
  ASSERT_EQ(out.initialized(), true);

  auto dense_out = std::dynamic_pointer_cast<pt::DenseTensor>(out.impl());
  auto* actual_result = dense_out->data<float>();
  for (auto i = 0; i < 6; i++) {
    ASSERT_NEAR(actual_result[i], val, 1e-6f);
  }
}

TEST(API, zeros_like) {
  // 1. create tensor
  auto dense_x = std::make_shared<pt::DenseTensor>(
      pt::TensorMeta(framework::make_ddim({3, 2}),
                     pt::Backend::kCPU,
                     pt::DataType::kFLOAT32,
                     pt::DataLayout::kNCHW),
      pt::TensorStatus());
  auto* dense_x_data = dense_x->mutable_data<float>();
  dense_x_data[0] = 1;

  paddle::experimental::Tensor x(dense_x);

  // 2. test API
  auto out = paddle::experimental::zeros_like(x, pt::DataType::kFLOAT32);

  // 3. check result
  ASSERT_EQ(out.shape().size(), 2);
  ASSERT_EQ(out.shape()[0], 3);
  ASSERT_EQ(out.numel(), 6);
  ASSERT_EQ(out.is_cpu(), true);
  ASSERT_EQ(out.type(), pt::DataType::kFLOAT32);
  ASSERT_EQ(out.layout(), pt::DataLayout::kNCHW);
  ASSERT_EQ(out.initialized(), true);

  auto dense_out = std::dynamic_pointer_cast<pt::DenseTensor>(out.impl());
  auto* actual_result = dense_out->data<float>();
  for (auto i = 0; i < 6; i++) {
    ASSERT_NEAR(actual_result[i], 0, 1e-6f);
  }
}

TEST(API, ones_like) {
  // 1. create tensor
  auto dense_x = std::make_shared<pt::DenseTensor>(
      pt::TensorMeta(framework::make_ddim({3, 2}),
                     pt::Backend::kCPU,
                     pt::DataType::kFLOAT32,
                     pt::DataLayout::kNCHW),
      pt::TensorStatus());
  auto* dense_x_data = dense_x->mutable_data<float>();
  dense_x_data[0] = 0;

  paddle::experimental::Tensor x(dense_x);

  // 2. test API
  auto out = paddle::experimental::ones_like(x, pt::DataType::kINT32);

  // 3. check result
  ASSERT_EQ(out.shape().size(), 2);
  ASSERT_EQ(out.shape()[0], 3);
  ASSERT_EQ(out.numel(), 6);
  ASSERT_EQ(out.is_cpu(), true);
  ASSERT_EQ(out.type(), pt::DataType::kINT32);
  ASSERT_EQ(out.layout(), pt::DataLayout::kNCHW);
  ASSERT_EQ(out.initialized(), true);

  auto dense_out = std::dynamic_pointer_cast<pt::DenseTensor>(out.impl());
  auto* actual_result = dense_out->data<float>();
  for (auto i = 0; i < 6; i++) {
    ASSERT_EQ(actual_result[i], 1);
  }
}
