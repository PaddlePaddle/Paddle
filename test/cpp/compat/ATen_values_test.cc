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
#include <ATen/ops/_values.h>
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
// Tests for at::Tensor::_values()
// ============================================================

// Helper: build 2-D sparse COO tensor from dense indices + float values.
static at::Tensor make_sparse_coo(at::Tensor indices,
                                  at::Tensor values,
                                  c10::IntArrayRef size) {
  return at::sparse_coo_tensor(indices, values, size);
}

// ---- COO sparse ----

TEST(TensorValuesTest, SparseCOO_ValuesHasCorrectNumel) {
  // 3x3 matrix, 2 non-zeros at (0,1)=1.5 and (2,0)=3.0
  at::Tensor indices =
      at::tensor({0, 2, 1, 0}, at::kLong).reshape({2, 2});  // [sparse_dim, nnz]
  at::Tensor values = at::tensor({1.5f, 3.0f}, at::kFloat);
  at::Tensor sparse = make_sparse_coo(indices, values, {3, 3});

  at::Tensor vals = sparse._values();

  ASSERT_EQ(vals.numel(), 2);
}

TEST(TensorValuesTest, SparseCOO_ValuesCorrectContent) {
  at::Tensor indices = at::tensor({0, 2, 1, 0}, at::kLong).reshape({2, 2});
  at::Tensor values = at::tensor({1.5f, 3.0f}, at::kFloat);
  at::Tensor sparse = make_sparse_coo(indices, values, {3, 3});

  at::Tensor vals = sparse.coalesce()._values();

  ASSERT_NEAR(vals[0].item<float>(), 1.5f, 1e-5f);
  ASSERT_NEAR(vals[1].item<float>(), 3.0f, 1e-5f);
}

TEST(TensorValuesTest, SparseCOO_ValuesIsDense) {
  // The values() tensor of a sparse tensor is itself a dense (strided) tensor.
  at::Tensor indices = at::tensor({0, 0, 1, 1}, at::kLong).reshape({2, 2});
  at::Tensor values = at::tensor({7.0f, 8.0f}, at::kFloat);
  at::Tensor sparse = make_sparse_coo(indices, values, {3, 3});

  at::Tensor vals = sparse._values();

  ASSERT_EQ(vals.layout(), c10::kStrided);
}

TEST(TensorValuesTest, SparseCOO_ValuesScalarType) {
  at::Tensor indices = at::tensor({0, 0, 1, 2}, at::kLong).reshape({2, 2});
  at::Tensor values = at::tensor({1, 2}, at::kInt);
  at::Tensor sparse = make_sparse_coo(indices, values, {3, 3});

  at::Tensor vals = sparse._values();

  ASSERT_EQ(vals.scalar_type(), at::kInt);
}

// ---- Dense tensor must throw ----

TEST(TensorValuesTest, DenseTensor_Throws) {
  at::Tensor dense = at::ones({3, 3}, at::kFloat);

  ASSERT_THROW(dense._values(), std::exception);
}

// ---- CSR sparse ----

TEST(TensorValuesTest, SparseCsr_ValuesCorrect) {
  // 3x3 identity in CSR: values=[1,1,1], col_indices=[0,1,2],
  //   row_ptrs=[0,1,2,3]
  at::Tensor crow = at::tensor({0, 1, 2, 3}, at::kInt);
  at::Tensor col = at::tensor({0, 1, 2}, at::kInt);
  at::Tensor vals_in = at::tensor({1.0f, 1.0f, 1.0f}, at::kFloat);
  at::Tensor sparse_csr =
      at::sparse_csr_tensor(crow, col, vals_in, {3, 3}, at::TensorOptions());

  // PyTorch does not dispatch _values for SparseCsr tensors
  ASSERT_THROW(sparse_csr._values(), std::exception);
}
