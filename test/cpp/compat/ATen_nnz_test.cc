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
#include <ATen/ops/_nnz.h>
#include <ATen/ops/tensor.h>
#include <c10/core/Layout.h>
#include <c10/core/ScalarType.h>
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
// Tests for at::Tensor::_nnz()
// ============================================================

// ---- Sparse COO ----

TEST(TensorNNZTest, SparseCOO_NNZ_Correct) {
  // indices: [[0,1],[1,2]] -> 2 entries in a 3x3 matrix
  at::Tensor indices = at::tensor({0, 1, 1, 2}, at::kLong).reshape({2, 2});
  at::Tensor values = at::tensor({1.0f, 2.0f}, at::kFloat);
  at::Tensor sparse = at::sparse_coo_tensor(indices, values, {3, 3});

  ASSERT_EQ(sparse._nnz(), 2);
}

TEST(TensorNNZTest, SparseCOO_NNZ_ZeroWhenAllZero) {
  // Empty sparse tensor: 0 explicit entries.
  at::Tensor indices = at::zeros({2, 0}, at::kLong);
  at::Tensor values = at::zeros({0}, at::kFloat);
  at::Tensor sparse = at::sparse_coo_tensor(indices, values, {3, 3});

  ASSERT_EQ(sparse._nnz(), 0);
}

TEST(TensorNNZTest, SparseCOO_NNZ_AfterCoalesce_DuplicatesMerged) {
  // Two entries at the same position: after coalescing they become one.
  at::Tensor indices = at::tensor({0, 0, 1, 1}, at::kLong).reshape({2, 2});
  at::Tensor values = at::tensor({1.0f, 2.0f}, at::kFloat);
  at::Tensor sparse = at::sparse_coo_tensor(indices, values, {3, 3});

  at::Tensor coalesced = sparse.coalesce();

  ASSERT_EQ(coalesced._nnz(), 1);
}

// ---- Sparse CSR ----

TEST(TensorNNZTest, SparseCsr_NNZ_Correct) {
  // 3x3 identity: 3 non-zeros.
  at::Tensor crow = at::tensor({0, 1, 2, 3}, at::kInt);
  at::Tensor col = at::tensor({0, 1, 2}, at::kInt);
  at::Tensor vals = at::tensor({1.0f, 1.0f, 1.0f}, at::kFloat);
  at::Tensor sparse_csr = at::sparse_csr_tensor(crow, col, vals, {3, 3});

  ASSERT_EQ(sparse_csr._nnz(), 3);
}

TEST(TensorNNZTest, SparseCsr_NNZ_SingleRow) {
  // 1x3 row with 2 non-zeros.
  at::Tensor crow = at::tensor({0, 2}, at::kInt);
  at::Tensor col = at::tensor({0, 2}, at::kInt);
  at::Tensor vals = at::tensor({5.0f, 7.0f}, at::kFloat);
  at::Tensor sparse_csr = at::sparse_csr_tensor(crow, col, vals, {1, 3});

  ASSERT_EQ(sparse_csr._nnz(), 2);
}

// ---- Dense tensor must throw ----

TEST(TensorNNZTest, DenseTensor_Throws) {
  at::Tensor dense = at::ones({3, 3}, at::kFloat);

  ASSERT_THROW(dense._nnz(), std::exception);
}
