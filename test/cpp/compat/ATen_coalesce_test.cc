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

// ============================================================
// Tests for at::Tensor::coalesce() and at::Tensor::is_coalesced()
// ============================================================

// Helper: build a 2-D sparse COO tensor from indices and values.
// indices shape: [sparse_dim, nnz], values shape: [nnz]
static at::Tensor make_sparse(at::Tensor indices,
                              at::Tensor values,
                              c10::IntArrayRef size) {
  return at::sparse_coo_tensor(indices, values, size);
}

TEST(TensorCoalesceTest, NewSparseNotCoalesced) {
  // A freshly created sparse COO tensor reports is_coalesced() == false.
  at::Tensor indices =
      at::tensor({0, 0, 1, 1, 1, 2}, at::kLong).reshape({2, 3});
  at::Tensor values = at::tensor({1.0f, 2.0f, 3.0f}, at::kFloat);
  at::Tensor sparse = make_sparse(indices, values, {3, 3});

  ASSERT_FALSE(sparse.is_coalesced());
}

TEST(TensorCoalesceTest, CoalesceReturnsSparse) {
  // coalesce() returns a sparse COO tensor.
  at::Tensor indices =
      at::tensor({0, 0, 1, 1, 1, 2}, at::kLong).reshape({2, 3});
  at::Tensor values = at::tensor({1.0f, 2.0f, 3.0f}, at::kFloat);
  at::Tensor sparse = make_sparse(indices, values, {3, 3});

  at::Tensor coalesced = sparse.coalesce();

  ASSERT_EQ(coalesced.layout(), c10::kSparse);
}

TEST(TensorCoalesceTest, CoalescedTensorIsCoalesced) {
  // After calling coalesce(), is_coalesced() must return true.
  at::Tensor indices =
      at::tensor({0, 0, 1, 1, 1, 2}, at::kLong).reshape({2, 3});
  at::Tensor values = at::tensor({1.0f, 2.0f, 3.0f}, at::kFloat);
  at::Tensor sparse = make_sparse(indices, values, {3, 3});

  at::Tensor coalesced = sparse.coalesce();

  ASSERT_TRUE(coalesced.is_coalesced());
}

TEST(TensorCoalesceTest, CoalesceDuplicateIndices_SumsValues) {
  // Duplicate indices [(0,1) appears twice] are merged; values are summed.
  // indices = [[0,0],[1,1]]  (both at (0,1))
  at::Tensor indices = at::tensor({0, 0, 1, 1}, at::kLong).reshape({2, 2});
  at::Tensor values = at::tensor({1.0f, 2.0f}, at::kFloat);
  at::Tensor sparse = make_sparse(indices, values, {3, 3});

  at::Tensor coalesced = sparse.coalesce();
  ASSERT_TRUE(coalesced.is_coalesced());
  // After coalescing, nnz should be 1 (duplicates merged)
  ASSERT_EQ(coalesced._nnz(), 1);
  // The merged value at (0,1) should be 1+2 = 3
  ASSERT_FLOAT_EQ(coalesced._values()[0].item<float>(), 3.0f);
}

TEST(TensorCoalesceTest, CoalesceIdempotent) {
  // Calling coalesce() on an already-coalesced tensor returns the same tensor.
  at::Tensor indices = at::tensor({0, 1, 1, 2}, at::kLong).reshape({2, 2});
  at::Tensor values = at::tensor({1.0f, 2.0f}, at::kFloat);
  at::Tensor sparse = make_sparse(indices, values, {3, 3});

  at::Tensor coalesced1 = sparse.coalesce();
  at::Tensor coalesced2 = coalesced1.coalesce();  // already coalesced

  ASSERT_TRUE(coalesced2.is_coalesced());
}

TEST(TensorCoalesceTest, CoalesceOnDenseTensorThrows) {
  // coalesce() on a dense tensor must throw.
  at::Tensor dense = at::ones({3, 3}, at::kFloat);
  ASSERT_THROW(dense.coalesce(), std::exception);
}

TEST(TensorCoalesceTest, IsCoalescedOnDenseTensorThrows) {
  // is_coalesced() on a dense tensor must throw.
  at::Tensor dense = at::ones({3, 3}, at::kFloat);
  ASSERT_THROW(dense.is_coalesced(), std::exception);
}
