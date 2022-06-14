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
#include "paddle/phi/core/selected_rows.h"

PD_DECLARE_KERNEL(full, CPU, ALL_LAYOUT);
PD_DECLARE_KERNEL(scale, CPU, ALL_LAYOUT);
PD_DECLARE_KERNEL(scale_sr, CPU, ALL_LAYOUT);
PD_DECLARE_KERNEL(copy, CPU, ALL_LAYOUT);

namespace paddle {
namespace tests {

namespace framework = paddle::framework;
using DDim = phi::DDim;

void CheckScaleResult(const experimental::Tensor* out) {
  ASSERT_EQ(out->dims().size(), 2);
  ASSERT_EQ(out->dims()[0], 3);
  ASSERT_EQ(out->dims()[1], 4);
  ASSERT_EQ(out->numel(), 12);
  ASSERT_EQ(out->is_cpu(), true);
  ASSERT_EQ(out->type(), phi::DataType::FLOAT32);
  ASSERT_EQ(out->layout(), phi::DataLayout::NCHW);
  ASSERT_EQ(out->initialized(), true);
  for (int64_t i = 0; i < out->numel(); ++i) {
    ASSERT_NEAR(3.0, out->data<float>()[i], 1e-6f);
  }
}

TEST(API, scale) {
  // 1. check `scale` is float value
  auto x = experimental::full({3, 4}, 1.0, phi::DataType::FLOAT32);
  auto out1 = experimental::scale(x, 2.0, 1.0, true);
  CheckScaleResult(&out1);

  // 2. check `scale` is Tensor with shape [1]
  auto scale = experimental::full({1}, 2.0, phi::DataType::FLOAT32);
  auto out2 = experimental::scale(x, scale, 1.0, true);
  CheckScaleResult(&out2);
}

TEST(API, scale_sr) {
  // 1. check `scale` is float value
  std::vector<int64_t> rows{0, 4, 7};
  int64_t height = 10;
  auto selected_rows = std::make_shared<phi::SelectedRows>(rows, height);
  auto dense_tensor = std::dynamic_pointer_cast<phi::DenseTensor>(
      experimental::full({3, 4}, 1.0, phi::DataType::FLOAT32).impl());
  *(selected_rows->mutable_value()) = *dense_tensor;
  experimental::Tensor x(selected_rows);
  auto out = experimental::scale(x, 2.0, 1.0, true);

  ASSERT_EQ(out.dims().size(), 2);
  ASSERT_EQ(out.dims()[0], 3);
  ASSERT_EQ(out.dims()[1], 4);
  ASSERT_EQ(out.numel(), 12);
  ASSERT_EQ(out.is_cpu(), true);
  ASSERT_EQ(out.type(), phi::DataType::FLOAT32);
  ASSERT_EQ(out.layout(), phi::DataLayout::NCHW);
  ASSERT_EQ(out.initialized(), true);
  for (int64_t i = 0; i < out.numel(); ++i) {
    ASSERT_NEAR(3.0, out.data<float>()[i], 1e-6f);
  }
}

}  // namespace tests
}  // namespace paddle
