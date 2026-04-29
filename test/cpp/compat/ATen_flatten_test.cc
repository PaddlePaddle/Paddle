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
// Flatten Tests
// ============================================================================

TEST(TestFlatten, FlattenAllDims) {
  // Test flatten with start_dim=0, end_dim=-1
  // Flattens the entire tensor to 1D
  at::Tensor tensor = at::ones({2, 3, 4}, at::kFloat);
  at::Tensor flattened = tensor.flatten(0, -1);

  ASSERT_EQ(flattened.sizes(), c10::IntArrayRef({24}));
  ASSERT_EQ(flattened.numel(), tensor.numel());
}

TEST(TestFlatten, FlattenPartialDims) {
  // Test flatten with specific start and end dimensions
  at::Tensor tensor = at::ones({2, 3, 4, 5}, at::kFloat);

  // Flatten dimensions 1 to 2 (3*4 = 12)
  at::Tensor flattened = tensor.flatten(1, 2);
  ASSERT_EQ(flattened.sizes(), c10::IntArrayRef({2, 12, 5}));
  ASSERT_EQ(flattened.numel(), tensor.numel());
}

TEST(TestFlatten, FlattenSingleDim) {
  // Test flatten when start_dim == end_dim (should be no-op)
  at::Tensor tensor = at::ones({2, 3, 4}, at::kFloat);

  at::Tensor flattened = tensor.flatten(1, 1);
  ASSERT_EQ(flattened.sizes(), c10::IntArrayRef({2, 3, 4}));
}

TEST(TestFlatten, FlattenNegativeDims) {
  // Test flatten with negative dimension indices
  at::Tensor tensor = at::ones({2, 3, 4, 5}, at::kFloat);

  // Flatten from -3 to -2 (dimensions 1 to 2)
  at::Tensor flattened = tensor.flatten(-3, -2);
  ASSERT_EQ(flattened.sizes(), c10::IntArrayRef({2, 12, 5}));
}

TEST(TestFlatten, FlattenFirstTwoDims) {
  // Test flatten on first two dimensions
  at::Tensor tensor = at::ones({2, 3, 4}, at::kFloat);

  at::Tensor flattened = tensor.flatten(0, 1);
  ASSERT_EQ(flattened.sizes(), c10::IntArrayRef({6, 4}));
}

TEST(TestFlatten, FlattenLastTwoDims) {
  // Test flatten on last two dimensions
  at::Tensor tensor = at::ones({2, 3, 4}, at::kFloat);

  at::Tensor flattened = tensor.flatten(1, 2);
  ASSERT_EQ(flattened.sizes(), c10::IntArrayRef({2, 12}));
}

TEST(TestFlatten, FlattenDataIntegrity) {
  // Test that flatten preserves data
  at::Tensor tensor = at::arange(24, at::kFloat).reshape({2, 3, 4});
  at::Tensor flattened = tensor.flatten(0, -1);

  const float* original_data = tensor.data_ptr<float>();
  const float* flattened_data = flattened.data_ptr<float>();

  for (int64_t i = 0; i < tensor.numel(); ++i) {
    ASSERT_EQ(original_data[i], flattened_data[i]);
  }
}

// ============================================================================
// Unflatten Tests
// ============================================================================

TEST(TestUnflatten, UnflattenBasic) {
  // Test basic unflatten operation
  at::Tensor tensor = at::ones({4, 6, 8}, at::kFloat);

  // Unflatten dimension 1 (size 6) into (2, 3)
  at::Tensor unflattened = tensor.unflatten(1, c10::IntArrayRef({2, 3}));
  ASSERT_EQ(unflattened.sizes(), c10::IntArrayRef({4, 2, 3, 8}));
  ASSERT_EQ(unflattened.numel(), tensor.numel());
}

TEST(TestUnflatten, UnflattenFirstDim) {
  // Test unflatten on first dimension
  at::Tensor tensor = at::ones({6, 4}, at::kFloat);

  // Unflatten dimension 0 (size 6) into (2, 3)
  at::Tensor unflattened = tensor.unflatten(0, c10::IntArrayRef({2, 3}));
  ASSERT_EQ(unflattened.sizes(), c10::IntArrayRef({2, 3, 4}));
}

TEST(TestUnflatten, UnflattenLastDim) {
  // Test unflatten on last dimension
  at::Tensor tensor = at::ones({2, 12}, at::kFloat);

  // Unflatten dimension 1 (size 12) into (3, 4)
  at::Tensor unflattened = tensor.unflatten(1, c10::IntArrayRef({3, 4}));
  ASSERT_EQ(unflattened.sizes(), c10::IntArrayRef({2, 3, 4}));
}

TEST(TestUnflatten, UnflattenNegativeDim) {
  // Test unflatten with negative dimension index
  at::Tensor tensor = at::ones({4, 6, 8}, at::kFloat);

  // Unflatten dimension -1 (last dim, size 8) into (4, 2)
  at::Tensor unflattened = tensor.unflatten(-1, c10::IntArrayRef({4, 2}));
  ASSERT_EQ(unflattened.sizes(), c10::IntArrayRef({4, 6, 4, 2}));
}

TEST(TestUnflatten, UnflattenSymInt) {
  // Test unflatten_symint (should behave same as unflatten)
  at::Tensor tensor = at::ones({4, 6, 8}, at::kFloat);

  // Unflatten dimension 1 using symint version
  // Note: Must keep the underlying data alive
  std::vector<c10::SymInt> sizes_vec = {2, 3};
  c10::SymIntArrayRef sizes(sizes_vec);
  at::Tensor unflattened = tensor.unflatten_symint(1, sizes);
  ASSERT_EQ(unflattened.sizes(), c10::IntArrayRef({4, 2, 3, 8}));
}

TEST(TestUnflatten, UnflattenDataIntegrity) {
  // Test that unflatten preserves data
  at::Tensor tensor = at::arange(24, at::kFloat).reshape({2, 12});
  at::Tensor unflattened = tensor.unflatten(1, c10::IntArrayRef({3, 4}));

  // Verify shape
  ASSERT_EQ(unflattened.sizes(), c10::IntArrayRef({2, 3, 4}));

  // Verify numel
  ASSERT_EQ(unflattened.numel(), tensor.numel());
}

// ============================================================================
// Flatten and Unflatten Combined Tests
// ============================================================================

TEST(TestFlattenUnflatten, RoundTrip) {
  // Test that flatten followed by unflatten restores original shape
  at::Tensor tensor = at::arange(24, at::kFloat).reshape({2, 3, 4});

  // Flatten dimensions 1 and 2
  at::Tensor flattened = tensor.flatten(1, 2);
  ASSERT_EQ(flattened.sizes(), c10::IntArrayRef({2, 12}));

  // Unflatten back to original shape
  at::Tensor unflattened = flattened.unflatten(1, c10::IntArrayRef({3, 4}));
  ASSERT_EQ(unflattened.sizes(), c10::IntArrayRef({2, 3, 4}));

  // Verify data integrity
  ASSERT_EQ(tensor.numel(), unflattened.numel());
}

TEST(TestFlattenUnflatten, MultipleOperations) {
  // Test multiple flatten/unflatten operations
  at::Tensor tensor = at::ones({2, 3, 4, 5}, at::kFloat);

  // Flatten all dimensions
  at::Tensor flattened = tensor.flatten(0, -1);
  ASSERT_EQ(flattened.sizes(), c10::IntArrayRef({120}));

  // Unflatten into different shape
  at::Tensor unflattened = flattened.unflatten(0, c10::IntArrayRef({6, 20}));
  ASSERT_EQ(unflattened.sizes(), c10::IntArrayRef({6, 20}));

  // Unflatten again
  at::Tensor final_tensor = unflattened.unflatten(1, c10::IntArrayRef({4, 5}));
  ASSERT_EQ(final_tensor.sizes(), c10::IntArrayRef({6, 4, 5}));
}
