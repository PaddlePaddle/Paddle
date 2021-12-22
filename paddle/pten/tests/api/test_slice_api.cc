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
#include "paddle/pten/api/include/tensor.h"
#include "paddle/pten/core/kernel_registry.h"

namespace paddle {
namespace tests {

TEST(Tensor, slice) {
  auto x = paddle::experimental::full({4, 3}, 1, pten::DataType::INT64);
  auto slice_x = x.slice(1, 2);

  // check slice result
  ASSERT_EQ(slice_x.dims().size(), 2);
  ASSERT_EQ(slice_x.dims()[0], 1);
  ASSERT_EQ(slice_x.dims()[1], 3);
  ASSERT_EQ(slice_x.numel(), 3);
  ASSERT_EQ(slice_x.is_cpu(), true);
  ASSERT_EQ(slice_x.type(), pten::DataType::INT64);
  ASSERT_EQ(slice_x.layout(), pten::DataLayout::NCHW);
  ASSERT_EQ(slice_x.initialized(), true);
  for (int64_t i = 0; i < slice_x.numel(); ++i) {
    ASSERT_EQ(slice_x.mutable_data<int64_t>()[i], 1);
  }
}

}  // namespace tests
}  // namespace paddle
