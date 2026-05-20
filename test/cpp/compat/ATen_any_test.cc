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

// ======================== any tests ========================

TEST(TensorAnyTest, AnyNoDim) {
  at::Tensor t = at::zeros({3, 3}, at::kFloat);
  t.data_ptr<float>()[4] = 1.0f;  // Set one element to non-zero

  at::Tensor result = t.any();
  ASSERT_EQ(result.numel(), 1);
  ASSERT_TRUE(result.item<bool>());
}

TEST(TensorAnyTest, AnyNoDimAllZero) {
  at::Tensor t = at::zeros({3, 3}, at::kFloat);

  at::Tensor result = t.any();
  ASSERT_EQ(result.numel(), 1);
  ASSERT_FALSE(result.item<bool>());
}

TEST(TensorAnyTest, AnyWithDim) {
  at::Tensor t = at::zeros({2, 3}, at::kFloat);
  t.data_ptr<float>()[0] = 1.0f;  // First row has non-zero

  at::Tensor result = t.any(0);
  ASSERT_EQ(result.numel(), 3);
}

TEST(TensorAnyTest, AnyWithDimKeepdim) {
  at::Tensor t = at::zeros({2, 3}, at::kFloat);
  t.data_ptr<float>()[0] = 1.0f;

  at::Tensor result = t.any(0, true);
  ASSERT_EQ(result.sizes().size(), 2);
  ASSERT_EQ(result.size(0), 1);
  ASSERT_EQ(result.size(1), 3);
}

TEST(TensorAnyTest, AnyWithOptionalDim) {
  at::Tensor t = at::zeros({2, 3}, at::kFloat);
  t.data_ptr<float>()[3] = 1.0f;  // Second row has non-zero

  at::Tensor result = t.any(at::OptionalIntArrayRef(1));
  ASSERT_EQ(result.numel(), 2);
}

TEST(TensorAnyTest, AnyInt) {
  at::Tensor t = at::zeros({2, 3}, at::kInt);
  t.data_ptr<int>()[0] = 1;

  at::Tensor result = t.any(1);
  ASSERT_EQ(result.numel(), 2);
}

TEST(TensorAnyTest, AnyBool) {
  at::Tensor t = at::zeros({3, 3}, at::kBool);
  t.data_ptr<bool>()[4] = true;

  at::Tensor result = t.any();
  ASSERT_TRUE(result.item<bool>());
}
