// Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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
#include <ATen/ops/vstack.h>
#include <c10/core/ScalarType.h>
#include <c10/core/TensorOptions.h>
#include <vector>

#include "ATen/ATen.h"
#include "gtest/gtest.h"
#include "torch/all.h"

TEST(ATenVStackTest, Basic2D) {
  auto t1 = at::ones({2, 3}, at::kFloat);
  auto t2 = at::zeros({2, 3}, at::kFloat);
  std::vector<at::Tensor> tensors = {t1, t2};
  auto result = at::vstack(tensors);

  EXPECT_EQ(result.dim(), 2);
  EXPECT_EQ(result.size(0), 4);
  EXPECT_EQ(result.size(1), 3);

  auto slice0 = result.slice(0, 0, 2);
  auto slice1 = result.slice(0, 2, 4);
  EXPECT_TRUE(at::allclose(slice0, t1));
  EXPECT_TRUE(at::allclose(slice1, t2));
}

TEST(ATenVStackTest, Basic1D) {
  auto t1 = at::ones({3}, at::kFloat);
  auto t2 = at::zeros({3}, at::kFloat);
  std::vector<at::Tensor> tensors = {t1, t2};
  auto result = at::vstack(tensors);

  EXPECT_EQ(result.dim(), 2);
  EXPECT_EQ(result.size(0), 2);
  EXPECT_EQ(result.size(1), 3);

  auto slice0 = result[0];
  auto slice1 = result[1];
  EXPECT_TRUE(at::allclose(slice0, t1));
  EXPECT_TRUE(at::allclose(slice1, t2));
}

TEST(ATenVStackTest, ScalarTo2D) {
  auto t1 = at::full({}, 1.0f, at::kFloat);
  auto t2 = at::full({}, 2.0f, at::kFloat);
  std::vector<at::Tensor> tensors = {t1, t2};
  auto result = at::vstack(tensors);

  EXPECT_EQ(result.dim(), 2);
  EXPECT_EQ(result.size(0), 2);
  EXPECT_EQ(result.size(1), 1);

  EXPECT_FLOAT_EQ(result[0][0].item<float>(), 1.0f);
  EXPECT_FLOAT_EQ(result[1][0].item<float>(), 2.0f);
}

TEST(ATenVStackTest, MixedDims) {
  auto t1 = at::ones({3}, at::kFloat);
  auto t2 = at::zeros({1, 3}, at::kFloat);
  std::vector<at::Tensor> tensors = {t1, t2};
  auto result = at::vstack(tensors);

  EXPECT_EQ(result.dim(), 2);
  EXPECT_EQ(result.size(0), 2);
  EXPECT_EQ(result.size(1), 3);

  auto slice0 = result[0];
  auto slice1 = result[1];
  EXPECT_TRUE(at::allclose(slice0, t1));
  EXPECT_TRUE(at::allclose(slice1, t2.squeeze(0)));
}

TEST(ATenVStackTest, EmptyListThrows) {
  std::vector<at::Tensor> tensors = {};
  ASSERT_THROW(at::vstack(tensors), std::exception);
}

TEST(ATenVStackTest, DtypeFloat64) {
  auto t1 = at::ones({2, 3}, at::kDouble);
  auto t2 = at::zeros({2, 3}, at::kDouble);
  std::vector<at::Tensor> tensors = {t1, t2};
  auto result = at::vstack(tensors);

  EXPECT_EQ(result.scalar_type(), at::kDouble);
  EXPECT_EQ(result.size(0), 4);
  EXPECT_EQ(result.size(1), 3);

  auto slice0 = result.slice(0, 0, 2);
  auto slice1 = result.slice(0, 2, 4);
  EXPECT_TRUE(at::allclose(slice0, t1));
  EXPECT_TRUE(at::allclose(slice1, t2));
}

TEST(ATenVStackTest, DtypeInt32) {
  auto t1 = at::ones({2, 3}, at::kInt);
  auto t2 = at::zeros({2, 3}, at::kInt);
  std::vector<at::Tensor> tensors = {t1, t2};
  auto result = at::vstack(tensors);

  EXPECT_EQ(result.scalar_type(), at::kInt);
  EXPECT_EQ(result.size(0), 4);
  EXPECT_EQ(result.size(1), 3);

  auto slice0 = result.slice(0, 0, 2);
  auto slice1 = result.slice(0, 2, 4);
  EXPECT_TRUE(at::equal(slice0, t1));
  EXPECT_TRUE(at::equal(slice1, t2));
}

TEST(ATenVStackTest, DtypeInt64) {
  auto t1 = at::ones({2, 3}, at::kLong);
  auto t2 = at::zeros({2, 3}, at::kLong);
  std::vector<at::Tensor> tensors = {t1, t2};
  auto result = at::vstack(tensors);

  EXPECT_EQ(result.scalar_type(), at::kLong);
  EXPECT_EQ(result.size(0), 4);
  EXPECT_EQ(result.size(1), 3);

  auto slice0 = result.slice(0, 0, 2);
  auto slice1 = result.slice(0, 2, 4);
  EXPECT_TRUE(at::equal(slice0, t1));
  EXPECT_TRUE(at::equal(slice1, t2));
}
