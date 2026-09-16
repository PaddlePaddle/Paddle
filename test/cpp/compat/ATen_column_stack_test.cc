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
#include <ATen/ops/column_stack.h>
#include <gtest/gtest.h>

#include <vector>

TEST(ColumnStackTest, Basic1D) {
  at::Tensor v1 = at::arange(3, at::kFloat);
  at::Tensor v2 = at::arange(3, 6, at::kFloat);
  std::vector<at::Tensor> tensors = {v1, v2};
  at::Tensor result = at::column_stack(tensors);

  ASSERT_EQ(result.dim(), 2);
  ASSERT_EQ(result.sizes()[0], 3);
  ASSERT_EQ(result.sizes()[1], 2);
  ASSERT_EQ(result.numel(), 6);

  ASSERT_FLOAT_EQ(result[0][0].item<float>(), 0.0f);
  ASSERT_FLOAT_EQ(result[0][1].item<float>(), 3.0f);
  ASSERT_FLOAT_EQ(result[1][0].item<float>(), 1.0f);
  ASSERT_FLOAT_EQ(result[1][1].item<float>(), 4.0f);
  ASSERT_FLOAT_EQ(result[2][0].item<float>(), 2.0f);
  ASSERT_FLOAT_EQ(result[2][1].item<float>(), 5.0f);
}

TEST(ColumnStackTest, Basic2D) {
  at::Tensor m1 = at::ones({2, 3}, at::kFloat);
  at::Tensor m2 = at::ones({2, 2}, at::kFloat);
  std::vector<at::Tensor> tensors = {m1, m2};
  at::Tensor result = at::column_stack(tensors);

  ASSERT_EQ(result.dim(), 2);
  ASSERT_EQ(result.sizes()[0], 2);
  ASSERT_EQ(result.sizes()[1], 5);
}

TEST(ColumnStackTest, Mixed1DAnd2D) {
  at::Tensor vec = at::zeros({3}, at::kFloat);
  at::Tensor mat = at::zeros({3, 2}, at::kFloat);
  std::vector<at::Tensor> tensors = {vec, mat};
  at::Tensor result = at::column_stack(tensors);

  ASSERT_EQ(result.dim(), 2);
  ASSERT_EQ(result.sizes()[0], 3);
  ASSERT_EQ(result.sizes()[1], 3);
}

TEST(ColumnStackTest, ScalarTensors) {
  at::Tensor s1 = at::ones({}, at::kFloat);
  at::Tensor s2 = at::ones({}, at::kFloat);
  std::vector<at::Tensor> tensors = {s1, s2};
  at::Tensor result = at::column_stack(tensors);

  ASSERT_EQ(result.dim(), 2);
  ASSERT_EQ(result.sizes()[0], 1);
  ASSERT_EQ(result.sizes()[1], 2);
}

TEST(ColumnStackTest, ScalarMixedWithVectorThrows) {
  at::Tensor scalar = at::ones({}, at::kFloat);
  at::Tensor vector = at::zeros({3}, at::kFloat);
  std::vector<at::Tensor> tensors = {scalar, vector};

  ASSERT_THROW(at::column_stack(tensors), std::exception);
}

TEST(ColumnStackTest, ScalarMixedWithSingleRowMatrix) {
  at::Tensor scalar = at::full({}, 2.0f, at::kFloat);
  at::Tensor matrix = at::arange(2, at::kFloat).reshape({1, 2});
  std::vector<at::Tensor> tensors = {scalar, matrix};
  at::Tensor result = at::column_stack(tensors);

  ASSERT_EQ(result.dim(), 2);
  ASSERT_EQ(result.sizes()[0], 1);
  ASSERT_EQ(result.sizes()[1], 3);
  ASSERT_FLOAT_EQ(result[0][0].item<float>(), 2.0f);
  ASSERT_FLOAT_EQ(result[0][1].item<float>(), 0.0f);
  ASSERT_FLOAT_EQ(result[0][2].item<float>(), 1.0f);
}

TEST(ColumnStackTest, SingleTensor) {
  at::Tensor vec = at::zeros({3}, at::kFloat);
  std::vector<at::Tensor> tensors = {vec};
  at::Tensor result = at::column_stack(tensors);

  ASSERT_EQ(result.dim(), 2);
  ASSERT_EQ(result.sizes()[0], 3);
  ASSERT_EQ(result.sizes()[1], 1);
}

TEST(ColumnStackTest, DtypeDouble) {
  at::Tensor t1 = at::zeros({3}, at::kDouble);
  at::Tensor t2 = at::zeros({3}, at::kDouble);
  std::vector<at::Tensor> tensors = {t1, t2};
  at::Tensor result = at::column_stack(tensors);

  ASSERT_EQ(result.scalar_type(), at::kDouble);
  ASSERT_EQ(result.dim(), 2);
  ASSERT_EQ(result.sizes()[0], 3);
  ASSERT_EQ(result.sizes()[1], 2);
}

TEST(ColumnStackTest, DtypeInt) {
  at::Tensor t1 = at::zeros({3}, at::kInt);
  at::Tensor t2 = at::zeros({3}, at::kInt);
  std::vector<at::Tensor> tensors = {t1, t2};
  at::Tensor result = at::column_stack(tensors);

  ASSERT_EQ(result.scalar_type(), at::kInt);
  ASSERT_EQ(result.dim(), 2);
  ASSERT_EQ(result.sizes()[0], 3);
  ASSERT_EQ(result.sizes()[1], 2);
}

TEST(ColumnStackTest, DtypeLong) {
  at::Tensor t1 = at::zeros({3}, at::kLong);
  at::Tensor t2 = at::zeros({3}, at::kLong);
  std::vector<at::Tensor> tensors = {t1, t2};
  at::Tensor result = at::column_stack(tensors);

  ASSERT_EQ(result.scalar_type(), at::kLong);
  ASSERT_EQ(result.dim(), 2);
  ASSERT_EQ(result.sizes()[0], 3);
  ASSERT_EQ(result.sizes()[1], 2);
}

TEST(ColumnStackTest, EmptyListThrows) {
  std::vector<at::Tensor> tensors = {};
  ASSERT_THROW(at::column_stack(tensors), std::exception);
}
