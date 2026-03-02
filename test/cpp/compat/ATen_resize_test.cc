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
#include <ATen/ops/resize.h>
#include <ATen/ops/tensor.h>
#include <c10/core/ScalarType.h>
#include <c10/core/TensorOptions.h>

#include "ATen/ATen.h"
#include "gtest/gtest.h"
#include "torch/all.h"

// ======================== resize_ tests ========================
// Note: Paddle's resize_ is implemented via reshape, which requires
// total element count to remain unchanged.

TEST(TensorResizeTest, ResizeBasic) {
  // Create a 2x3 tensor
  at::Tensor t = at::arange(6, at::kFloat).reshape({2, 3});

  // Resize to 3x2 (same 6 elements)
  t.resize_({3, 2});

  // Verify the shape
  ASSERT_EQ(t.sizes()[0], 3);
  ASSERT_EQ(t.sizes()[1], 2);
}

TEST(TensorResizeTest, ResizeFlatten) {
  // Create a 2x3 tensor
  at::Tensor t = at::arange(6, at::kFloat).reshape({2, 3});

  // Resize to flat 1D (same 6 elements)
  t.resize_({6});

  ASSERT_EQ(t.sizes()[0], 6);
}

TEST(TensorResizeTest, ResizeSameSize) {
  // Create a tensor
  at::Tensor t = at::arange(6, at::kFloat).reshape({2, 3});

  // Resize to same size
  t.resize_({2, 3});

  ASSERT_EQ(t.sizes()[0], 2);
  ASSERT_EQ(t.sizes()[1], 3);
}

TEST(TensorResizeTest, ResizeTo1D) {
  // Create a 2x3 tensor
  at::Tensor t = at::arange(6, at::kFloat).reshape({2, 3});

  // Resize to 1D (6 elements)
  t.resize_({6});

  ASSERT_EQ(t.sizes()[0], 6);
}

TEST(TensorResizeTest, ResizeTo2D) {
  // Create a 6-element tensor
  at::Tensor t = at::arange(6, at::kFloat);

  // Resize to 2x3 (6 elements)
  t.resize_({2, 3});

  ASSERT_EQ(t.sizes()[0], 2);
  ASSERT_EQ(t.sizes()[1], 3);
}

TEST(TensorResizeTest, ResizeSquare) {
  // Create a 2x3 tensor
  at::Tensor t = at::arange(6, at::kFloat).reshape({2, 3});

  // Resize to 1x6 (6 elements)
  t.resize_({1, 6});

  ASSERT_EQ(t.sizes()[0], 1);
  ASSERT_EQ(t.sizes()[1], 6);
}

TEST(TensorResizeTest, ResizePreservesData) {
  // Create tensor with known values
  at::Tensor t = at::arange(6, at::kFloat).reshape({2, 3});

  // Resize to 3x2
  t.resize_({3, 2});

  // Verify data is preserved (in row-major order)
  float* data = t.data_ptr<float>();
  ASSERT_FLOAT_EQ(data[0], 0.0f);
  ASSERT_FLOAT_EQ(data[1], 1.0f);
  ASSERT_FLOAT_EQ(data[2], 2.0f);
  ASSERT_FLOAT_EQ(data[3], 3.0f);
  ASSERT_FLOAT_EQ(data[4], 4.0f);
  ASSERT_FLOAT_EQ(data[5], 5.0f);
}

TEST(TensorResizeTest, ResizeReturnReference) {
  // Create a tensor
  at::Tensor t = at::zeros({2, 3});

  // Resize in-place, returns reference to same tensor
  const at::Tensor& result = t.resize_({3, 2});

  // Verify returned reference points to same tensor
  ASSERT_EQ(result.sizes()[0], 3);
  ASSERT_EQ(result.sizes()[1], 2);

  // Verify original tensor was modified
  ASSERT_EQ(t.sizes()[0], 3);
  ASSERT_EQ(t.sizes()[1], 2);
}

TEST(TensorResizeTest, ResizePreserveDtype) {
  // Create an int tensor
  at::Tensor t = at::zeros({2, 3}, at::kInt);

  // Resize to same element count (3x2)
  t.resize_({3, 2});

  // Verify dtype is preserved
  ASSERT_EQ(t.dtype(), at::kInt);
}

TEST(TensorResizeTest, ResizeLargeTensor) {
  // Create a larger tensor 4x5 = 20 elements
  at::Tensor t = at::arange(20, at::kFloat).reshape({4, 5});

  // Resize to 2x10 (20 elements)
  t.resize_({2, 10});

  ASSERT_EQ(t.sizes()[0], 2);
  ASSERT_EQ(t.sizes()[1], 10);
}

TEST(TensorResizeTest, ResizeChain) {
  // Multiple consecutive resizes
  at::Tensor t = at::arange(12, at::kFloat).reshape({3, 4});

  // Resize to 4x3
  t.resize_({4, 3});
  ASSERT_EQ(t.sizes()[0], 4);
  ASSERT_EQ(t.sizes()[1], 3);

  // Resize to 2x6
  t.resize_({2, 6});
  ASSERT_EQ(t.sizes()[0], 2);
  ASSERT_EQ(t.sizes()[1], 6);

  // Resize back to 3x4
  t.resize_({3, 4});
  ASSERT_EQ(t.sizes()[0], 3);
  ASSERT_EQ(t.sizes()[1], 4);
}
