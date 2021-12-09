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

namespace paddle {
namespace tests {

namespace framework = paddle::framework;
using DDim = paddle::framework::DDim;

void CheckScaleResult(Tensor* out) {
  ASSERT_EQ(out->dims().size(), 2);
  ASSERT_EQ(out->dims()[0], 3);
  ASSERT_EQ(out->dims()[1], 4);
  ASSERT_EQ(out->numel(), 12);
  ASSERT_EQ(out->is_cpu(), true);
  ASSERT_EQ(out->type(), pten::DataType::FLOAT32);
  ASSERT_EQ(out->layout(), pten::DataLayout::NCHW);
  ASSERT_EQ(out->initialized(), true);
  for (int64_t i = 0; i < out->numel(); ++i) {
    ASSERT_EQ(out->mutable_data<float>()[i], 3.0);
  }
}

TEST(API, scale) {
  // 1. check `scale` is float value
  auto x = experimental::full({3, 4}, 1.0, pten::DataType::FLOAT32);
  auto out1 = experimental::scale(x, 2.0, 1.0, true);
  CheckScaleResult(&out1);

  // 2. check `scale` is Tensor with shape [1]
  auto scale = experimental::full({1}, 2.0, pten::DataType::FLOAT32);
  auto out2 = experimental::scale(x, scale, 1.0, true);
  CheckScaleResult(&out2);
}

}  // namespace tests
}  // namespace paddle
