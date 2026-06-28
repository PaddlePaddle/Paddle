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

#include <ATen/ATen.h>
#include <ATen/Functions.h>
#include <ATen/core/TensorBody.h>
#include "gtest/gtest.h"

// ============================================================================
// RepeatInterleave Tests - Scalar repeats
// ============================================================================

TEST(TestRepeatInterleave, ScalarRepeatsWithDim) {
  at::Tensor tensor = at::arange(24, at::kFloat).reshape({2, 3, 4});
  at::Tensor result = tensor.repeat_interleave(2, 1);
  // Shape: {2, 3*2, 4} = {2, 6, 4}
  ASSERT_EQ(result.sizes(), c10::IntArrayRef({2, 6, 4}));
  ASSERT_EQ(result.numel(), 48);
}

TEST(TestRepeatInterleave, ScalarRepeatsWithoutDim) {
  at::Tensor tensor = at::arange(6, at::kFloat).reshape({2, 3});
  at::Tensor result = tensor.repeat_interleave(2);
  // Flattens to {6}, then repeats each element 2 times -> {12}
  ASSERT_EQ(result.sizes(), c10::IntArrayRef({12}));
  ASSERT_EQ(result.numel(), 12);
}

TEST(TestRepeatInterleave, ScalarRepeatsNegativeDim) {
  at::Tensor tensor = at::arange(24, at::kFloat).reshape({2, 3, 4});
  at::Tensor result = tensor.repeat_interleave(2, -1);
  // Shape: {2, 3, 4*2} = {2, 3, 8}
  ASSERT_EQ(result.sizes(), c10::IntArrayRef({2, 3, 8}));
}

TEST(TestRepeatInterleave, ZeroScalarRepeats) {
  at::Tensor tensor = at::ones({2, 3}, at::kFloat);
  at::Tensor result = tensor.repeat_interleave(0, 0);
  // Shape: {2*0, 3} = {0, 3}
  ASSERT_EQ(result.sizes(), c10::IntArrayRef({0, 3}));
}

// ============================================================================
// RepeatInterleave Tests - Tensor repeats
// ============================================================================

TEST(TestRepeatInterleave, TensorRepeatsWithDim) {
  at::Tensor tensor = at::ones({2, 3, 4}, at::kFloat);
  at::Tensor repeats = at::zeros({3}, at::kLong);
  repeats.data_ptr<int64_t>()[0] = 2;
  repeats.data_ptr<int64_t>()[1] = 1;
  repeats.data_ptr<int64_t>()[2] = 3;
  at::Tensor result = tensor.repeat_interleave(repeats, 1);
  // Shape: {2, 2+1+3, 4} = {2, 6, 4}
  ASSERT_EQ(result.sizes(), c10::IntArrayRef({2, 6, 4}));
}

TEST(TestRepeatInterleave, TensorRepeatsWithoutDim) {
  at::Tensor tensor = at::ones({2, 3}, at::kFloat);
  at::Tensor repeats = at::zeros({6}, at::kLong);
  for (int64_t i = 0; i < 6; ++i) {
    repeats.data_ptr<int64_t>()[i] = i % 3 + 1;
  }
  at::Tensor result = tensor.repeat_interleave(repeats);
  // Flattens to {6}, then each element repeated according to repeats
  int64_t total = 0;
  for (int64_t i = 0; i < 6; ++i) {
    total += (i % 3 + 1);
  }
  ASSERT_EQ(result.numel(), total);
}

TEST(TestRepeatInterleave, ScalarTensorRepeats) {
  at::Tensor tensor = at::ones({2, 3, 4}, at::kFloat);
  at::Tensor repeats = at::zeros({}, at::kLong);
  repeats.data_ptr<int64_t>()[0] = 2;
  at::Tensor result = tensor.repeat_interleave(repeats, 1);
  // 0-dim repeats reshaped to {1} and expanded to {3} -> all 2s
  // Shape: {2, 3*2, 4} = {2, 6, 4}
  ASSERT_EQ(result.sizes(), c10::IntArrayRef({2, 6, 4}));
}

// ============================================================================
// RepeatInterleave Tests - Standalone function
// ============================================================================

TEST(TestRepeatInterleave, StandaloneFunction) {
  at::Tensor repeats = at::zeros({3}, at::kLong);
  repeats.data_ptr<int64_t>()[0] = 2;
  repeats.data_ptr<int64_t>()[1] = 1;
  repeats.data_ptr<int64_t>()[2] = 3;
  at::Tensor result = at::repeat_interleave(repeats);
  // Returns indices [0, 0, 1, 2, 2, 2]
  ASSERT_EQ(result.sizes(), c10::IntArrayRef({6}));
  ASSERT_EQ(result.scalar_type(), at::kLong);
}

// ============================================================================
// RepeatInterleave Tests - Dtype coverage
// ============================================================================

TEST(TestRepeatInterleave, Float64Dtype) {
  at::Tensor tensor = at::ones({2, 3}, at::kDouble);
  at::Tensor result = tensor.repeat_interleave(2, 0);
  ASSERT_EQ(result.scalar_type(), at::kDouble);
}

TEST(TestRepeatInterleave, Int32Dtype) {
  at::Tensor tensor = at::ones({2, 3}, at::kInt);
  at::Tensor result = tensor.repeat_interleave(2, 0);
  ASSERT_EQ(result.scalar_type(), at::kInt);
}

TEST(TestRepeatInterleave, Int64Dtype) {
  at::Tensor tensor = at::ones({2, 3}, at::kLong);
  at::Tensor result = tensor.repeat_interleave(2, 0);
  ASSERT_EQ(result.scalar_type(), at::kLong);
}

// ============================================================================
// RepeatInterleave Tests - Exception cases
// ============================================================================

TEST(TestRepeatInterleave, NegativeRepeatsThrows) {
  at::Tensor tensor = at::ones({2, 3}, at::kFloat);
  ASSERT_THROW(tensor.repeat_interleave(-1, 0), std::exception);
}

TEST(TestRepeatInterleave, InvalidRepeatsDimThrows) {
  at::Tensor tensor = at::ones({2, 3, 4}, at::kFloat);
  at::Tensor repeats = at::ones({2, 3}, at::kLong);
  ASSERT_THROW(tensor.repeat_interleave(repeats, 0), std::exception);
}

TEST(TestRepeatInterleave, RepeatsSizeMismatchThrows) {
  at::Tensor tensor = at::ones({2, 3, 4}, at::kFloat);
  at::Tensor repeats = at::zeros({2}, at::kLong);
  repeats.data_ptr<int64_t>()[0] = 1;
  repeats.data_ptr<int64_t>()[1] = 2;
  // repeats size 2, but dim 1 size is 3
  ASSERT_THROW(tensor.repeat_interleave(repeats, 1), std::exception);
}

TEST(TestRepeatInterleave, DimOutOfRangePositiveThrows) {
  at::Tensor tensor = at::ones({2, 3}, at::kFloat);
  ASSERT_THROW(tensor.repeat_interleave(2, 5), std::exception);
}

TEST(TestRepeatInterleave, DimOutOfRangeNegativeThrows) {
  at::Tensor tensor = at::ones({2, 3}, at::kFloat);
  ASSERT_THROW(tensor.repeat_interleave(2, -5), std::exception);
}

TEST(TestRepeatInterleave, StandaloneEmptyRepeatsWithOutputSize) {
  at::Tensor repeats = at::empty({0}, at::kLong);
  at::Tensor result = at::repeat_interleave(repeats, 1);
  ASSERT_EQ(result.dim(), 1);
  ASSERT_EQ(result.numel(), 0);
  ASSERT_EQ(result.sizes()[0], 0);
}

TEST(TestRepeatInterleave, ScalarZeroRepeatsInvalidOutputSizeThrows) {
  at::Tensor tensor = at::ones({2, 3}, at::kFloat);
  ASSERT_THROW(tensor.repeat_interleave(0, 0, 1), std::exception);
}

TEST(TestRepeatInterleave, ScalarRepeatsNegativeOutputSizeThrows) {
  at::Tensor tensor = at::ones({2, 3}, at::kFloat);
  ASSERT_THROW(tensor.repeat_interleave(2, 0, -1), std::exception);
}

TEST(TestRepeatInterleave, TensorRepeatsNegativeOutputSizeThrows) {
  at::Tensor tensor = at::ones({3}, at::kFloat);
  at::Tensor repeats = at::ones({3}, at::kLong);
  ASSERT_THROW(tensor.repeat_interleave(repeats, 0, -1), std::exception);
}

TEST(TestRepeatInterleave, TensorRepeatsZeroOutputSizeMismatchThrows) {
  at::Tensor tensor = at::ones({3}, at::kFloat);
  at::Tensor repeats = at::ones({3}, at::kLong);
  ASSERT_THROW(tensor.repeat_interleave(repeats, 0, 0), std::exception);
}

TEST(TestRepeatInterleave, StandaloneNonEmptyRepeatsZeroOutputSize) {
  at::Tensor repeats = at::ones({3}, at::kLong);
  at::Tensor result = at::repeat_interleave(repeats, 0);
  ASSERT_EQ(result.dim(), 1);
  ASSERT_EQ(result.numel(), 0);
  ASSERT_EQ(result.sizes()[0], 0);
}

// ============================================================================
// RepeatInterleave Tests - Data integrity
// ============================================================================

TEST(TestRepeatInterleave, ScalarDataIntegrity) {
  at::Tensor t = at::zeros({3}, at::kFloat);
  t.data_ptr<float>()[0] = 1.0f;
  t.data_ptr<float>()[1] = 2.0f;
  t.data_ptr<float>()[2] = 3.0f;
  at::Tensor result = t.repeat_interleave(2);
  const float* data = result.data_ptr<float>();
  ASSERT_EQ(data[0], 1.0f);
  ASSERT_EQ(data[1], 1.0f);
  ASSERT_EQ(data[2], 2.0f);
  ASSERT_EQ(data[3], 2.0f);
  ASSERT_EQ(data[4], 3.0f);
  ASSERT_EQ(data[5], 3.0f);
}

TEST(TestRepeatInterleave, TensorDataIntegrity) {
  at::Tensor t = at::zeros({3}, at::kFloat);
  t.data_ptr<float>()[0] = 1.0f;
  t.data_ptr<float>()[1] = 2.0f;
  t.data_ptr<float>()[2] = 3.0f;
  at::Tensor repeats = at::zeros({3}, at::kLong);
  repeats.data_ptr<int64_t>()[0] = 2;
  repeats.data_ptr<int64_t>()[1] = 1;
  repeats.data_ptr<int64_t>()[2] = 3;
  at::Tensor result = t.repeat_interleave(repeats);
  const float* data = result.data_ptr<float>();
  ASSERT_EQ(data[0], 1.0f);
  ASSERT_EQ(data[1], 1.0f);
  ASSERT_EQ(data[2], 2.0f);
  ASSERT_EQ(data[3], 3.0f);
  ASSERT_EQ(data[4], 3.0f);
  ASSERT_EQ(data[5], 3.0f);
}
