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

// ======================== resize_ tests ========================

TEST(TensorResizeTest, ResizeBasic) {
  // Create a 2x3 tensor
  at::Tensor t = at::arange(6, at::kFloat).reshape({2, 3});

  // Resize to 3x2
  t.resize_({3, 2});

  // Verify the shape
  ASSERT_EQ(t.sizes()[0], 3);
  ASSERT_EQ(t.sizes()[1], 2);
}

TEST(TensorResizeTest, ResizeFlatten) {
  // Create a 2x3 tensor
  at::Tensor t = at::arange(6, at::kFloat).reshape({2, 3});

  // Resize to flat 1D
  t.resize_({6});

  ASSERT_EQ(t.sizes()[0], 6);
}

TEST(TensorResizeTest, ResizeExpand) {
  // Create a 2x3 tensor
  at::Tensor t = at::arange(6, at::kFloat).reshape({2, 3});

  // Resize to larger size (6 elements -> 12 elements)
  t.resize_({12});

  ASSERT_EQ(t.sizes()[0], 12);
}

TEST(TensorResizeTest, ResizeInPlace) {
  // Create a tensor
  at::Tensor t = at::zeros({2, 3});

  // Resize in-place
  const at::Tensor& result = t.resize_({4, 5});

  // Verify returned reference points to same tensor
  ASSERT_EQ(result.sizes()[0], 4);
  ASSERT_EQ(result.sizes()[1], 5);

  // Verify original tensor was modified
  ASSERT_EQ(t.sizes()[0], 4);
  ASSERT_EQ(t.sizes()[1], 5);
}

TEST(TensorResizeTest, ResizePreserveDtype) {
  // Create an int tensor
  at::Tensor t = at::zeros({2, 3}, at::kInt);

  // Resize
  t.resize_({3, 4});

  // Verify dtype is preserved
  ASSERT_EQ(t.dtype(), at::kInt);
}

TEST(TensorResizeTest, ResizeSameSize) {
  // Create a tensor
  at::Tensor t = at::arange(6, at::kFloat).reshape({2, 3});

  // Resize to same size
  t.resize_({2, 3});

  ASSERT_EQ(t.sizes()[0], 2);
  ASSERT_EQ(t.sizes()[1], 3);
}

TEST(TensorResizeTest, ResizeSingleDimension) {
  // Create a tensor
  at::Tensor t = at::zeros({10});

  // Resize to different 1D size
  t.resize_({20});

  ASSERT_EQ(t.sizes()[0], 20);
}
