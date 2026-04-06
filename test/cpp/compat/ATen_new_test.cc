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
#include <ATen/ops/new_empty.h>
#include <ATen/ops/new_full.h>
#include <ATen/ops/new_ones.h>
#include <ATen/ops/new_zeros.h>
#include <ATen/ops/tensor.h>
#include <c10/core/ScalarType.h>
#include <c10/core/TensorOptions.h>

#include "ATen/ATen.h"
#include "gtest/gtest.h"
#include "torch/all.h"

// ======================== new_empty tests ========================

TEST(TensorNewTest, NewEmptyBasic) {
  at::Tensor t = at::arange(6, at::kFloat).reshape({2, 3});

  at::Tensor result = t.new_empty({4, 5});

  ASSERT_EQ(result.sizes()[0], 4);
  ASSERT_EQ(result.sizes()[1], 5);
  ASSERT_EQ(result.dtype(), at::kFloat);
}

TEST(TensorNewTest, NewEmptyWithOptions) {
  at::Tensor t = at::arange(6, at::kFloat).reshape({2, 3});

  at::TensorOptions options = at::TensorOptions().dtype(at::kInt);
  at::Tensor result = t.new_empty({3, 4}, options);

  ASSERT_EQ(result.sizes()[0], 3);
  ASSERT_EQ(result.dtype(), at::kInt);
}

TEST(TensorNewTest, NewEmptyWithDtype) {
  at::Tensor t = at::arange(6, at::kFloat).reshape({2, 3});

  at::Tensor result = t.new_empty({2, 2}, at::kDouble);

  ASSERT_EQ(result.dtype(), at::kDouble);
}

TEST(TensorNewTest, NewEmptyWithDevice) {
  at::Tensor t = at::arange(6, at::kFloat);

  at::Tensor result = t.new_empty(
      {3, 3}, at::kFloat, ::std::nullopt, at::Device(at::kCPU), false);

  ASSERT_EQ(result.device().type(), at::kCPU);
}

TEST(TensorNewTest, NewEmptyNulloptDtypeInheritsSourceDtype) {
  at::Tensor t = at::arange(6, at::kInt);

  at::Tensor result =
      t.new_empty({2, 2}, std::nullopt, std::nullopt, at::kCPU, false);

  ASSERT_EQ(result.scalar_type(), at::kInt);
}

// ======================== new_full tests ========================

TEST(TensorNewTest, NewFullBasic) {
  at::Tensor t = at::arange(6, at::kFloat).reshape({2, 3});

  at::Tensor result = t.new_full({3, 4}, 7.5f);

  ASSERT_EQ(result.sizes()[0], 3);
  ASSERT_EQ(result.sizes()[1], 4);
  ASSERT_FLOAT_EQ(result.data_ptr<float>()[0], 7.5f);
}

TEST(TensorNewTest, NewFullWithOptions) {
  at::Tensor t = at::arange(6, at::kFloat);

  at::TensorOptions options = at::TensorOptions().dtype(at::kInt);
  at::Tensor result = t.new_full({2, 2}, 42, options);

  ASSERT_EQ(result.dtype(), at::kInt);
  ASSERT_EQ(result.data_ptr<int>()[0], 42);
}

TEST(TensorNewTest, NewFullWithDtype) {
  at::Tensor t = at::arange(6, at::kFloat);

  at::Tensor result = t.new_full({2, 2}, 100, at::kInt);

  ASSERT_EQ(result.dtype(), at::kInt);
}

TEST(TensorNewTest, NewFullScalarValue) {
  at::Tensor t = at::arange(6, at::kFloat);

  at::Scalar scalar_val(3.14);
  at::Tensor result = t.new_full({2, 2}, scalar_val);

  ASSERT_FLOAT_EQ(result.data_ptr<float>()[0], 3.14f);
}

TEST(TensorNewTest, NewFullNulloptDtypeInheritsSourceDtype) {
  at::Tensor t = at::arange(6, at::kLong);

  at::Tensor result =
      t.new_full({2, 2}, 9, std::nullopt, std::nullopt, at::kCPU, false);

  ASSERT_EQ(result.scalar_type(), at::kLong);
  ASSERT_EQ(result.data_ptr<int64_t>()[0], 9);
}

// ======================== new_zeros tests ========================

TEST(TensorNewTest, NewZerosBasic) {
  at::Tensor t = at::arange(6, at::kFloat).reshape({2, 3});

  at::Tensor result = t.new_zeros({4, 5});

  ASSERT_EQ(result.sizes()[0], 4);
  ASSERT_EQ(result.sizes()[1], 5);
  ASSERT_EQ(result.dtype(), at::kFloat);
}

TEST(TensorNewTest, NewZerosWithOptions) {
  at::Tensor t = at::arange(6, at::kFloat);

  at::TensorOptions options = at::TensorOptions().dtype(at::kInt);
  at::Tensor result = t.new_zeros({3, 3}, options);

  ASSERT_EQ(result.dtype(), at::kInt);
  ASSERT_EQ(result.data_ptr<int>()[0], 0);
}

TEST(TensorNewTest, NewZerosWithDtype) {
  at::Tensor t = at::arange(6, at::kFloat);

  at::Tensor result = t.new_zeros({2, 2}, at::kDouble);

  ASSERT_EQ(result.dtype(), at::kDouble);
}

TEST(TensorNewTest, NewZerosIntType) {
  at::Tensor t = at::arange(6, at::kInt);

  at::Tensor result = t.new_zeros({3, 3});

  ASSERT_EQ(result.dtype(), at::kInt);
  ASSERT_EQ(result.data_ptr<int>()[0], 0);
}

TEST(TensorNewTest, NewZerosNulloptDtypeInheritsSourceDtype) {
  at::Tensor t = at::arange(6, at::kDouble);

  at::Tensor result =
      t.new_zeros({2, 3}, std::nullopt, std::nullopt, at::kCPU, false);

  ASSERT_EQ(result.scalar_type(), at::kDouble);
  ASSERT_EQ(result.data_ptr<double>()[0], 0.0);
}

// ======================== new_ones tests ========================

TEST(TensorNewTest, NewOnesBasic) {
  at::Tensor t = at::arange(6, at::kFloat).reshape({2, 3});

  at::Tensor result = t.new_ones({4, 5});

  ASSERT_EQ(result.sizes()[0], 4);
  ASSERT_EQ(result.sizes()[1], 5);
  ASSERT_EQ(result.dtype(), at::kFloat);
}

TEST(TensorNewTest, NewOnesWithOptions) {
  at::Tensor t = at::arange(6, at::kFloat);

  at::TensorOptions options = at::TensorOptions().dtype(at::kInt);
  at::Tensor result = t.new_ones({3, 3}, options);

  ASSERT_EQ(result.dtype(), at::kInt);
  ASSERT_EQ(result.data_ptr<int>()[0], 1);
}

TEST(TensorNewTest, NewOnesWithDtype) {
  at::Tensor t = at::arange(6, at::kFloat);

  at::Tensor result = t.new_ones({2, 2}, at::kDouble);

  ASSERT_EQ(result.dtype(), at::kDouble);
  ASSERT_EQ(result.data_ptr<double>()[0], 1.0);
}

TEST(TensorNewTest, NewOnesIntType) {
  at::Tensor t = at::arange(6, at::kInt);

  at::Tensor result = t.new_ones({3, 3});

  ASSERT_EQ(result.dtype(), at::kInt);
  ASSERT_EQ(result.data_ptr<int>()[0], 1);
}

TEST(TensorNewTest, NewOnesNulloptDtypeInheritsSourceDtype) {
  at::Tensor t = at::ones({6}, at::kDouble);

  at::Tensor result =
      t.new_ones({2, 3}, std::nullopt, std::nullopt, at::kCPU, false);

  ASSERT_EQ(result.scalar_type(), at::kDouble);
  ASSERT_EQ(result.data_ptr<double>()[0], 1.0);
}
