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

// Helper macro for float comparison with tolerance
#define ASSERT_FLOAT_NEAR(a, b, tol) \
  ASSERT_NEAR(static_cast<double>(a), static_cast<double>(b), tol)

TEST(TensorAsStridedTest, AsStridedBasic) {
  // Test basic as_strided functionality: create 2x3 view with strides {3, 1}
  // Original tensor: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
  // View should be:
  //   [[0, 1, 2],
  //    [3, 4, 5]]
  at::Tensor t = at::arange(12, at::kFloat);
  at::Tensor result = t.as_strided({2, 3}, {3, 1});

  // Verify shape
  ASSERT_EQ(result.sizes(), c10::IntArrayRef({2, 3}));

  // Verify data content
  float* data = result.data_ptr<float>();
  ASSERT_FLOAT_NEAR(data[0], 0.0f, 1e-5);
  ASSERT_FLOAT_NEAR(data[1], 1.0f, 1e-5);
  ASSERT_FLOAT_NEAR(data[2], 2.0f, 1e-5);
  ASSERT_FLOAT_NEAR(data[3], 3.0f, 1e-5);
  ASSERT_FLOAT_NEAR(data[4], 4.0f, 1e-5);
  ASSERT_FLOAT_NEAR(data[5], 5.0f, 1e-5);

  // Verify memory is shared (view, not copy)
  ASSERT_TRUE(result.is_same(t));
}

TEST(TensorAsStridedTest, AsStridedWithOffset) {
  // Test as_strided with offset: skip first 2 elements
  // Original tensor: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
  // With offset=2, view should be:
  //   [[2, 3, 4],
  //    [5, 6, 7]]
  at::Tensor t = at::arange(12, at::kFloat);
  at::Tensor result = t.as_strided({2, 3}, {3, 1}, 2);

  // Verify shape
  ASSERT_EQ(result.sizes(), c10::IntArrayRef({2, 3}));

  // Verify offset is applied - data should start from index 2
  float* data = result.data_ptr<float>();
  ASSERT_FLOAT_NEAR(data[0], 2.0f, 1e-5);
  ASSERT_FLOAT_NEAR(data[1], 3.0f, 1e-5);
  ASSERT_FLOAT_NEAR(data[2], 4.0f, 1e-5);
  ASSERT_FLOAT_NEAR(data[3], 5.0f, 1e-5);
  ASSERT_FLOAT_NEAR(data[4], 6.0f, 1e-5);
  ASSERT_FLOAT_NEAR(data[5], 7.0f, 1e-5);

  // Verify memory is shared
  ASSERT_TRUE(result.is_same(t));
}

TEST(TensorAsStridedTest, AsStridedWithDifferentStrides) {
  // Test with non-contiguous strides: {2, 1} on 12 elements
  // Should produce: [[0,1,2], [3,4,5], [6,7,8], [9,10,11]] -> actually needs
  // proper calculation With shape {4, 2} and stride {2, 1}:
  //   [[0,1],
  //    [2,3],
  //    [4,5],
  //    [6,7]]
  at::Tensor t = at::arange(12, at::kFloat);
  at::Tensor result = t.as_strided({4, 2}, {2, 1});

  ASSERT_EQ(result.sizes(), c10::IntArrayRef({4, 2}));

  float* data = result.data_ptr<float>();
  ASSERT_FLOAT_NEAR(data[0], 0.0f, 1e-5);
  ASSERT_FLOAT_NEAR(data[1], 1.0f, 1e-5);
  ASSERT_FLOAT_NEAR(data[2], 2.0f, 1e-5);
  ASSERT_FLOAT_NEAR(data[3], 3.0f, 1e-5);
  ASSERT_FLOAT_NEAR(data[4], 4.0f, 1e-5);
  ASSERT_FLOAT_NEAR(data[5], 5.0f, 1e-5);
  ASSERT_FLOAT_NEAR(data[6], 6.0f, 1e-5);
  ASSERT_FLOAT_NEAR(data[7], 7.0f, 1e-5);

  ASSERT_TRUE(result.is_same(t));
}

TEST(TensorAsStridedTest, AsStridedInplace) {
  // Test inplace as_strided_: modifies tensor in-place, shares memory
  // Original tensor: [0,1,2,3,4,5,6,7,8,9,10,11], shape {12}
  // After as_strided_({2, 6}, {6, 1}): shape {2, 6}
  //   [[0,1,2,3,4,5],
  //    [6,7,8,9,10,11]]
  at::Tensor t = at::arange(12, at::kFloat);
  float* original_data_ptr = t.data_ptr<float>();

  t.as_strided_({2, 6}, {6, 1});

  // Verify shape changed
  ASSERT_EQ(t.sizes(), c10::IntArrayRef({2, 6}));

  // Verify data pointer unchanged (memory shared)
  ASSERT_EQ(t.data_ptr<float>(), original_data_ptr);

  // Verify data content is correct
  float* data = t.data_ptr<float>();
  ASSERT_FLOAT_NEAR(data[0], 0.0f, 1e-5);
  ASSERT_FLOAT_NEAR(data[1], 1.0f, 1e-5);
  ASSERT_FLOAT_NEAR(data[5], 5.0f, 1e-5);
  ASSERT_FLOAT_NEAR(data[6], 6.0f, 1e-5);
  ASSERT_FLOAT_NEAR(data[11], 11.0f, 1e-5);
}

TEST(TensorAsStridedTest, AsStridedInplaceWithOffset) {
  // Test inplace with offset
  // Original tensor: [0,1,2,3,4,5,6,7,8,9,10,11]
  // After as_strided_({2, 3}, {3, 1}, 1): shape {2, 3}, starting from index 1
  //   [[1,2,3],
  //    [4,5,6]]
  at::Tensor t = at::arange(12, at::kFloat);
  float* original_data_ptr = t.data_ptr<float>();

  t.as_strided_({2, 3}, {3, 1}, 1);

  // Verify shape changed
  ASSERT_EQ(t.sizes(), c10::IntArrayRef({2, 3}));

  // Verify data pointer unchanged
  ASSERT_EQ(t.data_ptr<float>(), original_data_ptr);

  // Verify data content - should start from index 1 due to offset
  float* data = t.data_ptr<float>();
  ASSERT_FLOAT_NEAR(data[0], 1.0f, 1e-5);
  ASSERT_FLOAT_NEAR(data[1], 2.0f, 1e-5);
  ASSERT_FLOAT_NEAR(data[2], 3.0f, 1e-5);
  ASSERT_FLOAT_NEAR(data[3], 4.0f, 1e-5);
  ASSERT_FLOAT_NEAR(data[4], 5.0f, 1e-5);
  ASSERT_FLOAT_NEAR(data[5], 6.0f, 1e-5);
}

TEST(TensorAsStridedTest, AsStridedInplaceModifiesView) {
  // Test that modifying inplace view affects original tensor
  at::Tensor t = at::arange(12, at::kFloat);
  at::Tensor view = t.as_strided({2, 3}, {3, 1});

  // Modify the view
  view.data_ptr<float>()[0] = 99.0f;

  // Verify original tensor is modified
  ASSERT_FLOAT_NEAR(t.data_ptr<float>()[0], 99.0f, 1e-5);
}

TEST(TensorAsStridedTest, AsStridedScatterBasic) {
  // Test as_strided_scatter: write src tensor into t at specified strides
  // Original t: [0,1,2,3,4,5,6,7,8,9,10,11]
  // src: 2x3 filled with 99
  // With shape {2,3} and stride {3,1}:
  //   t[0:2, 0:3] = 99
  // Result:
  //   [[99,99,99],
  //    [99,99,99],
  //    [6, 7, 8],
  //    [9,10,11]]
  at::Tensor t = at::arange(12, at::kFloat);
  at::Tensor src = at::full({2, 3}, 99.0f, at::kFloat);
  at::Tensor result = t.as_strided_scatter(src, {2, 3}, {3, 1});

  // Verify output shape
  ASSERT_EQ(result.sizes(), c10::IntArrayRef({2, 3}));

  // Verify scattered data
  float* data = result.data_ptr<float>();
  for (int i = 0; i < 6; ++i) {
    ASSERT_FLOAT_NEAR(data[i], 99.0f, 1e-5);
  }
}

TEST(TensorAsStridedTest, AsStridedScatterOriginalUnchanged) {
  // Verify as_strided_scatter returns new tensor, original unchanged
  at::Tensor t = at::arange(12, at::kFloat);
  at::Tensor src = at::full({2, 3}, 99.0f, at::kFloat);
  at::Tensor result = t.as_strided_scatter(src, {2, 3}, {3, 1});

  // Verify result is a new tensor (not same storage)
  // Note: Some implementations may return the same tensor, check implementation
  ASSERT_NE(result.data_ptr<float>(), t.data_ptr<float>());

  // Verify original t is unchanged
  ASSERT_FLOAT_NEAR(t.data_ptr<float>()[0], 0.0f, 1e-5);
  ASSERT_FLOAT_NEAR(t.data_ptr<float>()[5], 5.0f, 1e-5);
}

TEST(TensorAsStridedTest, AsStridedScatterWithOffset) {
  // Test scatter with offset
  // Original t: [0,1,2,3,4,5,6,7,8,9,10,11]
  // src: 2x2 filled with 88
  // With shape {2,2}, stride {2,1}, offset=2:
  //   t[0:2, 0:2] starting at index 2 -> [2,3,4,5] = 88
  // Result:
  //   [0,1,88,88,88,88,6,7,8,9,10,11]
  at::Tensor t = at::arange(12, at::kFloat);
  at::Tensor src = at::full({2, 2}, 88.0f, at::kFloat);
  at::Tensor result = t.as_strided_scatter(src, {2, 2}, {2, 1}, 2);

  ASSERT_EQ(result.sizes(), c10::IntArrayRef({2, 2}));

  float* data = result.data_ptr<float>();
  ASSERT_FLOAT_NEAR(data[0], 88.0f, 1e-5);
  ASSERT_FLOAT_NEAR(data[1], 88.0f, 1e-5);
  ASSERT_FLOAT_NEAR(data[2], 88.0f, 1e-5);
  ASSERT_FLOAT_NEAR(data[3], 88.0f, 1e-5);
}

TEST(TensorAsStridedTest, AsStridedTranspose) {
  // Test creating transposed view using as_strided
  // Original: [0,1,2,3,4,5] (2x3 matrix: [[0,1,2],[3,4,5]])
  // Transposed: [[0,3],[1,4],[2,5]] -> shape {3,2}, stride {1,2}
  at::Tensor t = at::arange(6, at::kFloat).view({2, 3});
  at::Tensor result = t.as_strided({3, 2}, {1, 2});

  ASSERT_EQ(result.sizes(), c10::IntArrayRef({3, 2}));

  float* data = result.data_ptr<float>();
  ASSERT_FLOAT_NEAR(data[0], 0.0f, 1e-5);
  ASSERT_FLOAT_NEAR(data[1], 3.0f, 1e-5);
  ASSERT_FLOAT_NEAR(data[2], 1.0f, 1e-5);
  ASSERT_FLOAT_NEAR(data[3], 4.0f, 1e-5);
  ASSERT_FLOAT_NEAR(data[4], 2.0f, 1e-5);
  ASSERT_FLOAT_NEAR(data[5], 5.0f, 1e-5);

  ASSERT_TRUE(result.is_same(t));
}

TEST(TensorAsStridedTest, AsStridedContiguous) {
  // Test contiguous flag after as_strided
  at::Tensor t = at::arange(12, at::kFloat);

  // Contiguous view
  at::Tensor contig = t.as_strided({2, 6}, {6, 1});
  ASSERT_TRUE(contig.is_contiguous());

  // Non-contiguous view (transposed)
  at::Tensor non_contig = t.as_strided({3, 2}, {1, 3});
  ASSERT_FALSE(non_contig.is_contiguous());
}
