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

#include "paddle/fluid/memory/allocation/allocator_facade.h"
#include "paddle/phi/api/include/tensor.h"
#include "paddle/phi/api/lib/api_custom_impl.h"
#include "paddle/phi/common/place.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/core/selected_rows.h"

PD_DECLARE_KERNEL(add_n_sr, CPU, ALL_LAYOUT);

namespace paddle {
namespace tests {

TEST(API, add_n) {
  // 1. create tensor
  std::vector<int64_t> rows = {0, 1, 2, 3, 4, 5, 6};
  int64_t row_numel = 12;
  auto x_sr = std::make_shared<phi::SelectedRows>(rows, 10);
  auto x_meta = phi::DenseTensorMeta(
      phi::DataType::FLOAT32,
      phi::make_ddim({static_cast<int64_t>(rows.size()), row_numel}),
      phi::DataLayout::NCHW);
  x_sr->mutable_value()->set_meta(x_meta);
  x_sr->AllocateFrom(paddle::memory::allocation::AllocatorFacade::Instance()
                         .GetAllocator(paddle::platform::CPUPlace())
                         .get(),
                     phi::DataType::FLOAT32);
  auto* dense_x_data = x_sr->mutable_value()->data<float>();

  auto y_sr = std::make_shared<phi::SelectedRows>(rows, 10);
  y_sr->mutable_value()->set_meta(x_meta);
  y_sr->AllocateFrom(paddle::memory::allocation::AllocatorFacade::Instance()
                         .GetAllocator(paddle::platform::CPUPlace())
                         .get(),
                     phi::DataType::FLOAT32);
  auto* dense_y_data = y_sr->mutable_value()->data<float>();

  float sum[84] = {0.0};
  for (size_t i = 0; i < 7; ++i) {
    for (size_t j = 0; j < 12; ++j) {
      dense_x_data[i * 12 + j] = (i * 4 + j);
      dense_y_data[i * 12 + j] = (i * 4 + j);
      sum[i * 12 + j] += (i * 4 + j) * 2;
    }
  }

  paddle::experimental::Tensor x(x_sr);
  paddle::experimental::Tensor y(y_sr);
  auto out = paddle::experimental::add_n_impl({x, y});

  // check slice result
  ASSERT_EQ(
      static_cast<int>(std::dynamic_pointer_cast<phi::SelectedRows>(out.impl())
                           ->rows()
                           .size()),
      7);
  for (int64_t i = 0; i < 84; ++i) {
    ASSERT_EQ(sum[i],
              std::dynamic_pointer_cast<phi::SelectedRows>(out.impl())
                  ->value()
                  .data<float>()[i]);
  }
}

}  // namespace tests
}  // namespace paddle
