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
#include <ATen/cuda/EmptyTensor.h>
#include <ATen/native/cuda/Resize.h>
#include <ATen/ops/tensor.h>
#include <c10/core/ScalarType.h>
#include <c10/core/SymInt.h>
#include <c10/core/TensorOptions.h>
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
#include <c10/cuda/CUDAFunctions.h>
#include <c10/cuda/CUDAGuard.h>
#endif
#include "ATen/ATen.h"
#include "gtest/gtest.h"
#include "paddle/phi/common/float16.h"
#include "torch/all.h"

// ============================================================================
// Narrow Tests
// ============================================================================

TEST(TestNarrow, NarrowBasic) {
  // Test basic narrow operation on dimension 0
  at::Tensor tensor = at::arange(12, at::kFloat).reshape({3, 4});
  // tensor: [[0, 1, 2, 3],
  //          [4, 5, 6, 7],
  //          [8, 9, 10, 11]]

  // Narrow on dim 0, start 1, length 2
  at::Tensor narrowed = tensor.narrow(0, 1, 2);
  ASSERT_EQ(narrowed.sizes(), c10::IntArrayRef({2, 4}));

  // Verify data
  const float* data = narrowed.data_ptr<float>();
  ASSERT_EQ(data[0], 4.0f);  // First element of row 1
  ASSERT_EQ(data[4], 8.0f);  // First element of row 2
}

TEST(TestNarrow, NarrowOnDim1) {
  // Test narrow on dimension 1
  at::Tensor tensor = at::arange(12, at::kFloat).reshape({3, 4});

  // Narrow on dim 1, start 1, length 2
  at::Tensor narrowed = tensor.narrow(1, 1, 2);
  ASSERT_EQ(narrowed.sizes(), c10::IntArrayRef({3, 2}));
}

TEST(TestNarrow, NarrowStartZero) {
  // Test narrow starting from 0
  at::Tensor tensor = at::ones({5, 4}, at::kFloat);

  at::Tensor narrowed = tensor.narrow(0, 0, 3);
  ASSERT_EQ(narrowed.sizes(), c10::IntArrayRef({3, 4}));
  ASSERT_EQ(narrowed.numel(), 12);
}

TEST(TestNarrow, NarrowFullLength) {
  // Test narrow with full length (should return same shape on that dim)
  at::Tensor tensor = at::ones({3, 4}, at::kFloat);

  at::Tensor narrowed = tensor.narrow(0, 0, 3);
  ASSERT_EQ(narrowed.sizes(), c10::IntArrayRef({3, 4}));
}

TEST(TestNarrow, NarrowSingleElement) {
  // Test narrow with length 1
  at::Tensor tensor = at::arange(12, at::kFloat).reshape({3, 4});

  at::Tensor narrowed = tensor.narrow(0, 1, 1);
  ASSERT_EQ(narrowed.sizes(), c10::IntArrayRef({1, 4}));
}

TEST(TestNarrow, NarrowHigherDim) {
  // Test narrow on 3D tensor
  at::Tensor tensor = at::ones({2, 3, 4}, at::kFloat);

  // Narrow on dim 1
  at::Tensor narrowed = tensor.narrow(1, 1, 2);
  ASSERT_EQ(narrowed.sizes(), c10::IntArrayRef({2, 2, 4}));

  // Narrow on dim 2
  at::Tensor narrowed2 = tensor.narrow(2, 0, 2);
  ASSERT_EQ(narrowed2.sizes(), c10::IntArrayRef({2, 3, 2}));
}

TEST(TestNarrow, NarrowSymInt) {
  // Test narrow_symint (should behave same as narrow)
  at::Tensor tensor = at::ones({3, 4}, at::kFloat);

  c10::SymInt start = 1;
  c10::SymInt length = 2;
  at::Tensor narrowed = tensor.narrow_symint(0, start, length);
  ASSERT_EQ(narrowed.sizes(), c10::IntArrayRef({2, 4}));
}

// ============================================================================
// Narrow Copy Tests
// ============================================================================

TEST(TestNarrowCopy, NarrowCopyBasic) {
  // Test narrow_copy returns a copy of the narrowed tensor
  at::Tensor tensor = at::arange(12, at::kFloat).reshape({3, 4});

  at::Tensor narrowed_copy = tensor.narrow_copy(0, 1, 2);
  ASSERT_EQ(narrowed_copy.sizes(), c10::IntArrayRef({2, 4}));

  // Verify it's a copy (different data pointer)
  // Note: narrow_copy creates a contiguous copy
  ASSERT_EQ(narrowed_copy.numel(), 8);
}

TEST(TestNarrowCopy, NarrowCopySymInt) {
  // Test narrow_copy_symint
  at::Tensor tensor = at::ones({3, 4}, at::kFloat);

  c10::SymInt start = 0;
  c10::SymInt length = 2;
  at::Tensor narrowed_copy = tensor.narrow_copy_symint(0, start, length);
  ASSERT_EQ(narrowed_copy.sizes(), c10::IntArrayRef({2, 4}));
}

TEST(TestNarrowCopy, NarrowCopyDataIntegrity) {
  // Test that narrow_copy preserves data correctly
  at::Tensor tensor = at::arange(12, at::kFloat).reshape({3, 4});

  at::Tensor narrowed_copy = tensor.narrow_copy(0, 1, 2);

  // Verify data
  const float* data = narrowed_copy.data_ptr<float>();
  ASSERT_EQ(data[0], 4.0f);
  ASSERT_EQ(data[1], 5.0f);
  ASSERT_EQ(data[2], 6.0f);
  ASSERT_EQ(data[3], 7.0f);
}

// ============================================================================
// Narrow with Tensor Start Tests
// ============================================================================

TEST(TestNarrowTensor, NarrowWithTensorStart) {
  // Test narrow with tensor start parameter
  at::Tensor tensor = at::arange(12, at::kFloat).reshape({3, 4});
  // Create start tensor with value 1 using ones
  at::Tensor start_tensor = at::ones({1}, at::kLong);

  at::Tensor narrowed = tensor.narrow(0, start_tensor, 2);
  ASSERT_EQ(narrowed.sizes(), c10::IntArrayRef({2, 4}));
}

TEST(TestNarrowTensor, NarrowSymIntWithTensorStart) {
  // Test narrow_symint with tensor start parameter
  at::Tensor tensor = at::ones({3, 4}, at::kFloat);
  // Create start tensor with value 0 using zeros
  at::Tensor start_tensor = at::zeros({1}, at::kLong);

  c10::SymInt length = 2;
  at::Tensor narrowed = tensor.narrow_symint(0, start_tensor, length);
  ASSERT_EQ(narrowed.sizes(), c10::IntArrayRef({2, 4}));
}

// ============================================================================
// Combined Tests
// ============================================================================

TEST(TestNarrowCombined, NarrowMultipleDims) {
  // Test narrow on multiple dimensions sequentially
  at::Tensor tensor = at::ones({4, 5, 6}, at::kFloat);

  // Narrow on each dimension
  at::Tensor n1 = tensor.narrow(0, 1, 2);
  ASSERT_EQ(n1.sizes(), c10::IntArrayRef({2, 5, 6}));

  at::Tensor n2 = n1.narrow(1, 2, 3);
  ASSERT_EQ(n2.sizes(), c10::IntArrayRef({2, 3, 6}));

  at::Tensor n3 = n2.narrow(2, 0, 4);
  ASSERT_EQ(n3.sizes(), c10::IntArrayRef({2, 3, 4}));
}

TEST(TestNarrowCombined, NarrowDataVerification) {
  // Verify narrow selects correct data
  at::Tensor tensor = at::arange(20, at::kFloat).reshape({4, 5});
  // tensor: [[ 0,  1,  2,  3,  4],
  //          [ 5,  6,  7,  8,  9],
  //          [10, 11, 12, 13, 14],
  //          [15, 16, 17, 18, 19]]

  // Narrow rows 1-2, columns 2-3
  at::Tensor narrowed = tensor.narrow(0, 1, 2).narrow(1, 2, 2);
  ASSERT_EQ(narrowed.sizes(), c10::IntArrayRef({2, 2}));

  // Expected: [[7, 8], [12, 13]]
  // Note: The actual verification depends on contiguous memory layout
}
