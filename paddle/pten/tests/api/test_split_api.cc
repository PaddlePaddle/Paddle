// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <gtest/gtest.h>
#include <memory>

#include "paddle/pten/api/include/api.h"

#include "paddle/pten/api/include/manual_api.h"
#include "paddle/pten/api/lib/utils/allocator.h"
#include "paddle/pten/core/dense_tensor.h"
#include "paddle/pten/core/kernel_registry.h"

namespace paddle {
namespace tests {

namespace framework = paddle::framework;
using DDim = pten::framework::DDim;

// TODO(chentianyu03): Remove this test after the API is used in the dygraph
TEST(API, split) {
  // 1. create tensor
  const auto alloc = std::make_unique<paddle::experimental::DefaultAllocator>(
      paddle::platform::CPUPlace());
  auto dense_x = std::make_shared<pten::DenseTensor>(
      alloc.get(),
      pten::DenseTensorMeta(pten::DataType::FLOAT32,
                            pten::framework::make_ddim({4, 10}),
                            pten::DataLayout::NCHW));
  auto* dense_x_data =
      dense_x->mutable_data<float>(paddle::platform::CPUPlace());

  for (size_t i = 0; i < 4; ++i) {
    for (size_t j = 0; j < 10; ++j) {
      dense_x_data[i * 10 + j] = (i * 10 + j) * 1.0;
    }
  }

  paddle::experimental::Tensor x(dense_x);

  // 2. test API
  auto out = paddle::experimental::split(x, {2, 2}, 0);

  // 3. check result
  ASSERT_EQ(out.size(), static_cast<size_t>(2));
  ASSERT_EQ(out[0].dims().size(), 2);
  ASSERT_EQ(out[0].dims()[0], 2);
  ASSERT_EQ(out[0].dims()[1], 10);
  ASSERT_EQ(out[0].type(), pten::DataType::FLOAT32);
  ASSERT_EQ(out[0].layout(), pten::DataLayout::NCHW);

  ASSERT_EQ(out[1].dims().size(), 2);
  ASSERT_EQ(out[1].dims()[0], 2);
  ASSERT_EQ(out[1].dims()[1], 10);
  ASSERT_EQ(out[1].type(), pten::DataType::FLOAT32);
  ASSERT_EQ(out[1].layout(), pten::DataLayout::NCHW);

  auto out_data_0 = std::dynamic_pointer_cast<pten::DenseTensor>(out[0].impl())
                        ->data<float>();
  auto out_data_1 = std::dynamic_pointer_cast<pten::DenseTensor>(out[1].impl())
                        ->data<float>();
  for (size_t i = 0; i < 4; ++i) {
    if (i < 20) {
      ASSERT_NEAR(dense_x_data[i], out_data_0[i], 1e-6);
    } else {
      ASSERT_NEAR(dense_x_data[i], out_data_1[i - 20], 1e-6);
    }
  }
}

}  // namespace tests
}  // namespace paddle
