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

TEST(TensorClampTest, ClampWithScalar) {
  // Create tensor with values [0, 1, 2, 3, 4, 5]
  at::Tensor t = at::arange(6, at::kFloat).reshape({2, 3});
  at::Tensor result = t.clamp(at::Scalar(1.0), at::Scalar(4.0));

  float* data = result.data_ptr<float>();
  // Expected: [1, 1, 2, 3, 4, 4]
  ASSERT_FLOAT_EQ(data[0], 1.0f);
  ASSERT_FLOAT_EQ(data[1], 1.0f);
  ASSERT_FLOAT_EQ(data[2], 2.0f);
  ASSERT_FLOAT_EQ(data[3], 3.0f);
  ASSERT_FLOAT_EQ(data[4], 4.0f);
  ASSERT_FLOAT_EQ(data[5], 4.0f);
}

TEST(TensorClampTest, ClampWithTensor) {
  at::Tensor t = at::arange(6, at::kFloat).reshape({2, 3});
  at::Tensor min_t = at::full({2, 3}, 1.0f, at::kFloat);
  at::Tensor max_t = at::full({2, 3}, 4.0f, at::kFloat);

  at::Tensor result = t.clamp(::std::optional<at::Tensor>(min_t),
                              ::std::optional<at::Tensor>(max_t));

  float* data = result.data_ptr<float>();
  ASSERT_FLOAT_EQ(data[0], 1.0f);
  ASSERT_FLOAT_EQ(data[5], 4.0f);
}

TEST(TensorClampTest, ClampInplaceScalar) {
  at::Tensor t = at::arange(6, at::kFloat).reshape({2, 3});
  t.clamp_(at::Scalar(2.0), at::Scalar(3.0));

  float* data = t.data_ptr<float>();
  // Expected: [2, 2, 2, 3, 3, 3]
  ASSERT_FLOAT_EQ(data[0], 2.0f);
  ASSERT_FLOAT_EQ(data[1], 2.0f);
  ASSERT_FLOAT_EQ(data[2], 2.0f);
  ASSERT_FLOAT_EQ(data[3], 3.0f);
  ASSERT_FLOAT_EQ(data[4], 3.0f);
  ASSERT_FLOAT_EQ(data[5], 3.0f);
}

TEST(TensorClampTest, ClampInplaceTensor) {
  at::Tensor t = at::arange(6, at::kFloat).reshape({2, 3});
  at::Tensor min_t = at::full({2, 3}, 1.0f, at::kFloat);
  at::Tensor max_t = at::full({2, 3}, 4.0f, at::kFloat);

  t.clamp_(::std::optional<at::Tensor>(min_t),
           ::std::optional<at::Tensor>(max_t));

  float* data = t.data_ptr<float>();
  ASSERT_FLOAT_EQ(data[0], 1.0f);
  ASSERT_FLOAT_EQ(data[5], 4.0f);
}

TEST(TensorClampTest, ClampMaxScalar) {
  at::Tensor t = at::arange(6, at::kFloat);
  at::Tensor result = t.clamp_max(at::Scalar(3.0));

  float* data = result.data_ptr<float>();
  // Expected: [0, 1, 2, 3, 3, 3]
  ASSERT_FLOAT_EQ(data[4], 3.0f);
  ASSERT_FLOAT_EQ(data[5], 3.0f);
}

TEST(TensorClampTest, ClampMaxTensor) {
  at::Tensor t = at::arange(6, at::kFloat);
  at::Tensor max_t = at::full({6}, 3.0f, at::kFloat);
  at::Tensor result = t.clamp_max(max_t);

  float* data = result.data_ptr<float>();
  ASSERT_FLOAT_EQ(data[4], 3.0f);
  ASSERT_FLOAT_EQ(data[5], 3.0f);
}

TEST(TensorClampTest, ClampMaxInplaceScalar) {
  at::Tensor t = at::arange(6, at::kFloat);
  t.clamp_max_(at::Scalar(3.0));

  float* data = t.data_ptr<float>();
  ASSERT_FLOAT_EQ(data[4], 3.0f);
  ASSERT_FLOAT_EQ(data[5], 3.0f);
}

TEST(TensorClampTest, ClampMaxInplaceTensor) {
  at::Tensor t = at::arange(6, at::kFloat);
  at::Tensor max_t = at::full({6}, 3.0f, at::kFloat);
  t.clamp_max_(max_t);

  float* data = t.data_ptr<float>();
  ASSERT_FLOAT_EQ(data[4], 3.0f);
  ASSERT_FLOAT_EQ(data[5], 3.0f);
}

TEST(TensorClampTest, ClampMinScalar) {
  at::Tensor t = at::arange(6, at::kFloat);
  at::Tensor result = t.clamp_min(at::Scalar(2.0));

  float* data = result.data_ptr<float>();
  // Expected: [2, 2, 2, 3, 4, 5]
  ASSERT_FLOAT_EQ(data[0], 2.0f);
  ASSERT_FLOAT_EQ(data[1], 2.0f);
}

TEST(TensorClampTest, ClampMinTensor) {
  at::Tensor t = at::arange(6, at::kFloat);
  at::Tensor min_t = at::full({6}, 2.0f, at::kFloat);
  at::Tensor result = t.clamp_min(min_t);

  float* data = result.data_ptr<float>();
  ASSERT_FLOAT_EQ(data[0], 2.0f);
  ASSERT_FLOAT_EQ(data[1], 2.0f);
}

TEST(TensorClampTest, ClampMinInplaceScalar) {
  at::Tensor t = at::arange(6, at::kFloat);
  t.clamp_min_(at::Scalar(2.0));

  float* data = t.data_ptr<float>();
  ASSERT_FLOAT_EQ(data[0], 2.0f);
  ASSERT_FLOAT_EQ(data[1], 2.0f);
}

TEST(TensorClampTest, ClampMinInplaceTensor) {
  at::Tensor t = at::arange(6, at::kFloat);
  at::Tensor min_t = at::full({6}, 2.0f, at::kFloat);
  t.clamp_min_(min_t);

  float* data = t.data_ptr<float>();
  ASSERT_FLOAT_EQ(data[0], 2.0f);
  ASSERT_FLOAT_EQ(data[1], 2.0f);
}
