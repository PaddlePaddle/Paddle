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
#include <cmath>

#include "ATen/ATen.h"
#include "gtest/gtest.h"
#include "torch/all.h"

// ======================== std tests ========================

TEST(TensorStdTest, StdDefault) {
  // Create tensor [1, 2, 3, 4, 5, 6]
  at::Tensor t = at::arange(1, 7, at::kFloat);
  at::Tensor result = t.std();

  // std of [1,2,3,4,5,6] with unbiased=true (ddof=1) = sqrt(3.5)
  ASSERT_EQ(result.numel(), 1);
  float val = result.item<float>();
  ASSERT_NEAR(val, std::sqrt(3.5f), 1e-4);
}

TEST(TensorStdTest, StdBiased) {
  at::Tensor t = at::arange(1, 7, at::kFloat);
  at::Tensor result = t.std(false);  // unbiased=false

  // std with ddof=0: sqrt(sum((x-mean)^2)/N)
  // mean=3.5, sum_sq_diff = 17.5, var=17.5/6, std=sqrt(17.5/6)
  ASSERT_EQ(result.numel(), 1);
  float val = result.item<float>();
  ASSERT_NEAR(val, std::sqrt(17.5f / 6.0f), 1e-4);
}

TEST(TensorStdTest, StdWithDim) {
  // Create 2x3 tensor
  at::Tensor t = at::arange(1, 7, at::kFloat).reshape({2, 3});
  at::Tensor result =
      t.std(at::OptionalIntArrayRef({1}), /*unbiased=*/true, /*keepdim=*/false);

  ASSERT_EQ(result.numel(), 2);
}

TEST(TensorStdTest, StdWithDimAndCorrection) {
  at::Tensor t = at::arange(1, 7, at::kFloat).reshape({2, 3});
  at::Tensor result = t.std(
      at::OptionalIntArrayRef({1}), ::std::optional<at::Scalar>(1.0), false);

  ASSERT_EQ(result.numel(), 2);
}

TEST(TensorStdTest, StdSingleDim) {
  at::Tensor t = at::arange(1, 7, at::kFloat).reshape({2, 3});
  at::Tensor result = t.std(1);

  ASSERT_EQ(result.numel(), 2);
}

// ======================== var tests ========================

TEST(TensorVarTest, VarDefault) {
  at::Tensor t = at::arange(1, 7, at::kFloat);
  at::Tensor result = t.var();

  // var of [1,2,3,4,5,6] with unbiased=true: 17.5/5 = 3.5
  float val = result.item<float>();
  ASSERT_NEAR(val, 3.5f, 1e-4);
}

TEST(TensorVarTest, VarBiased) {
  at::Tensor t = at::arange(1, 7, at::kFloat);
  at::Tensor result = t.var(false);

  // var with unbiased=false: 17.5/6
  float val = result.item<float>();
  ASSERT_NEAR(val, 17.5f / 6.0f, 1e-4);
}

TEST(TensorVarTest, VarWithDim) {
  at::Tensor t = at::arange(1, 7, at::kFloat).reshape({2, 3});
  at::Tensor result =
      t.var(at::OptionalIntArrayRef({1}), /*unbiased=*/true, /*keepdim=*/false);

  ASSERT_EQ(result.numel(), 2);
}

TEST(TensorVarTest, VarWithCorrection) {
  at::Tensor t = at::arange(1, 7, at::kFloat).reshape({2, 3});
  at::Tensor result = t.var(
      at::OptionalIntArrayRef({0}), ::std::optional<at::Scalar>(1.0), false);

  ASSERT_EQ(result.numel(), 3);
}

TEST(TensorVarTest, VarSingleDim) {
  at::Tensor t = at::arange(1, 7, at::kFloat).reshape({2, 3});
  at::Tensor result = t.var(0);

  ASSERT_EQ(result.numel(), 3);
}

// ======================= Additional std edge case tests
// ========================

TEST(TensorStdTest, StdWithKeepdim) {
  at::Tensor t = at::arange(1, 7, at::kFloat).reshape({2, 3});
  at::Tensor result =
      t.std(at::OptionalIntArrayRef({1}), /*unbiased=*/true, /*keepdim=*/true);

  // keepdim should preserve dimension
  ASSERT_EQ(result.sizes().size(), 2);
  ASSERT_EQ(result.size(0), 2);
  ASSERT_EQ(result.size(1), 1);
}

TEST(TensorStdTest, StdWithMultipleDims) {
  at::Tensor t = at::arange(1, 13, at::kFloat).reshape({2, 2, 3});
  at::Tensor result = t.std(
      at::OptionalIntArrayRef({0, 2}), /*unbiased=*/true, /*keepdim=*/false);

  ASSERT_EQ(result.numel(), 2);
}

TEST(TensorStdTest, StdWithCorrectionValue) {
  at::Tensor t = at::arange(1, 7, at::kFloat);
  // Test with custom correction value (ddof)
  at::Tensor result = t.std(
      at::OptionalIntArrayRef({}), ::std::optional<at::Scalar>(2.0), false);

  ASSERT_EQ(result.numel(), 1);
}

TEST(TensorStdTest, StdNegativeDim) {
  at::Tensor t = at::arange(1, 7, at::kFloat).reshape({2, 3});
  // Test with negative dimension (-1 means last dimension)
  at::Tensor result = t.std(-1);

  ASSERT_EQ(result.numel(), 2);
}

TEST(TensorVarTest, VarWithKeepdim) {
  at::Tensor t = at::arange(1, 7, at::kFloat).reshape({2, 3});
  at::Tensor result =
      t.var(at::OptionalIntArrayRef({1}), /*unbiased=*/true, /*keepdim=*/true);

  ASSERT_EQ(result.sizes().size(), 2);
  ASSERT_EQ(result.size(0), 2);
  ASSERT_EQ(result.size(1), 1);
}

TEST(TensorVarTest, VarWithMultipleDims) {
  at::Tensor t = at::arange(1, 13, at::kFloat).reshape({2, 2, 3});
  at::Tensor result = t.var(
      at::OptionalIntArrayRef({0, 2}), /*unbiased=*/true, /*keepdim=*/false);

  ASSERT_EQ(result.numel(), 2);
}

TEST(TensorVarTest, VarWithCorrectionValue) {
  at::Tensor t = at::arange(1, 7, at::kFloat);
  at::Tensor result = t.var(
      at::OptionalIntArrayRef({}), ::std::optional<at::Scalar>(2.0), false);

  ASSERT_EQ(result.numel(), 1);
}

TEST(TensorVarTest, VarNegativeDim) {
  at::Tensor t = at::arange(1, 7, at::kFloat).reshape({2, 3});
  at::Tensor result = t.var(-1);

  ASSERT_EQ(result.numel(), 2);
}
