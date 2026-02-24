// Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
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

#include <ATen/Functions.h>
#include <ATen/core/TensorBody.h>
#include <ATen/ops/tensor.h>
#include <c10/core/ScalarType.h>
#include <c10/core/TensorOptions.h>

#include "ATen/ATen.h"
#include "gtest/gtest.h"
#include "torch/all.h"

TEST(TensorAsStridedTest, AsStrided) {
  at::Tensor t = at::arange(12, at::kFloat);
  // Create a 2x3 view with strides {3, 1}
  at::Tensor result = t.as_strided({2, 3}, {3, 1});

  ASSERT_EQ(result.sizes(), c10::IntArrayRef({2, 3}));
  float* data = result.data_ptr<float>();
  // Should see [0,1,2,3,4,5] (first 6 elements with strides {3,1})
  ASSERT_FLOAT_EQ(data[0], 0.0f);
  ASSERT_FLOAT_EQ(data[1], 1.0f);
}

TEST(TensorAsStridedTest, AsStridedWithOffset) {
  at::Tensor t = at::arange(12, at::kFloat);
  at::Tensor result = t.as_strided({2, 3}, {3, 1}, 2);

  ASSERT_EQ(result.sizes(), c10::IntArrayRef({2, 3}));
}

TEST(TensorAsStridedTest, AsStridedInplace) {
  at::Tensor t = at::arange(12, at::kFloat);
  t.as_strided_({2, 6}, {6, 1});

  ASSERT_EQ(t.sizes(), c10::IntArrayRef({2, 6}));
}

TEST(TensorAsStridedTest, AsStridedInplaceWithOffset) {
  at::Tensor t = at::arange(12, at::kFloat);
  t.as_strided_({2, 3}, {3, 1}, 1);

  ASSERT_EQ(t.sizes(), c10::IntArrayRef({2, 3}));
}

TEST(TensorAsStridedTest, AsStridedScatter) {
  at::Tensor t = at::arange(12, at::kFloat);
  at::Tensor src = at::full({2, 3}, 99.0f, at::kFloat);
  at::Tensor result = t.as_strided_scatter(src, {2, 3}, {3, 1});

  ASSERT_EQ(result.sizes(), c10::IntArrayRef({2, 3}));
}
