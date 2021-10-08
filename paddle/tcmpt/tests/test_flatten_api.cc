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

#include "paddle/tcmpt/api/include/dev/symbols.h"
#include "paddle/tcmpt/api/include/manipulation.h"

#include "paddle/tcmpt/core/dense_tensor.h"

namespace framework = paddle::framework;
using DDim = paddle::framework::DDim;

TEST(API, flatten) {
  // 1. create tensor
  auto dense_x = std::make_shared<pt::DenseTensor>(
      pt::TensorMeta(framework::make_ddim({3, 2, 2, 3}),
                     pt::Backend::kCPU,
                     pt::DataType::kFLOAT32,
                     pt::DataLayout::kNCHW),
      pt::TensorStatus());
  auto* dense_x_data = dense_x->mutable_data<float>();

  for (int i = 0; i < dense_x->numel(); i++) {
    dense_x_data[i] = i;
  }

  pt::Tensor x(dense_x);
  int start_axis = 1, stop_axis = 2;
  // 2. test API
  auto out = pt::flatten(x, start_axis, stop_axis);

  // 3. check result
  std::vector<int> expect_shape = {3, 4, 3};
  ASSERT_EQ(out.shape()[0], expect_shape[0]);
  ASSERT_EQ(out.shape()[1], expect_shape[1]);
  ASSERT_EQ(out.shape()[2], expect_shape[2]);
  ASSERT_EQ(out.numel(), 36);
  ASSERT_EQ(out.is_cpu(), true);
  ASSERT_EQ(out.type(), pt::DataType::kFLOAT32);
  ASSERT_EQ(out.layout(), pt::DataLayout::kNCHW);
  ASSERT_EQ(out.initialized(), true);
  bool value_equal = true;
  auto dense_out = std::dynamic_pointer_cast<pt::DenseTensor>(out.impl());
  auto* dense_out_data = dense_out->data<float>();
  for (int i = 0; i < dense_x->numel(); i++) {
    if (std::abs(dense_x_data[i] - dense_out_data[i]) > 1e-6f)
      value_equal = false;
  }
  ASSERT_EQ(value_equal, true);
}
