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
#include <c10/core/Layout.h>
#include <c10/core/ScalarType.h>
#include <c10/core/SymInt.h>
#include <c10/core/TensorOptions.h>
#include <vector>
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
#include <c10/cuda/CUDAFunctions.h>
#include <c10/cuda/CUDAGuard.h>
#endif
#include "ATen/ATen.h"
#include "gtest/gtest.h"
#include "paddle/phi/common/float16.h"
#include "torch/all.h"

// ============== at::zeros sparse tests ==============

TEST(SparseZerosTest, SparseCOO) {
  // Dense tensor should return false
  at::TensorBase tensor = at::zeros({2, 3}, at::kFloat);
  ASSERT_FALSE(tensor.is_sparse());
  ASSERT_EQ(tensor.layout(), c10::kStrided);

  // Create sparse COO tensor
  auto options = c10::TensorOptions().dtype(at::kFloat).layout(at::kSparse);
  at::TensorBase sparse_tensor = at::zeros({2, 3}, options);
  ASSERT_TRUE(sparse_tensor.is_sparse());
  ASSERT_EQ(sparse_tensor.layout(), c10::kSparse);
}

TEST(SparseZerosTest, SparseCsr) {
  // Create sparse CSR tensor
  auto options = c10::TensorOptions().dtype(at::kFloat).layout(at::kSparseCsr);
  at::TensorBase sparse_csr_tensor = at::zeros({2, 3}, options);
  ASSERT_TRUE(sparse_csr_tensor.is_sparse_csr());
  ASSERT_TRUE(sparse_csr_tensor.is_sparse());
  ASSERT_EQ(sparse_csr_tensor.layout(), c10::kSparseCsr);
}

TEST(SparseZerosTest, WithOptionalParams) {
  // Test zeros with optional parameters
  at::Tensor sparse_tensor = at::zeros(
      {2, 3}, at::kFloat, at::kSparse, at::kCPU, /*pin_memory=*/false);
  ASSERT_TRUE(sparse_tensor.is_sparse());
  ASSERT_EQ(sparse_tensor.layout(), c10::kSparse);

  at::Tensor sparse_csr_tensor = at::zeros(
      {2, 3}, at::kFloat, at::kSparseCsr, at::kCPU, /*pin_memory=*/false);
  ASSERT_TRUE(sparse_csr_tensor.is_sparse_csr());
  ASSERT_EQ(sparse_csr_tensor.layout(), c10::kSparseCsr);
}

// ============== at::empty sparse tests ==============

TEST(SparseEmptyTest, SparseCOO) {
  // Dense tensor should return false
  at::TensorBase tensor = at::empty({2, 3}, at::kFloat);
  ASSERT_FALSE(tensor.is_sparse());
  ASSERT_EQ(tensor.layout(), c10::kStrided);

  // Create sparse COO tensor
  auto options = c10::TensorOptions().dtype(at::kFloat).layout(at::kSparse);
  at::TensorBase sparse_tensor = at::empty({2, 3}, options);
  ASSERT_TRUE(sparse_tensor.is_sparse());
  ASSERT_EQ(sparse_tensor.layout(), c10::kSparse);
}

TEST(SparseEmptyTest, SparseCsr) {
  // Create sparse CSR tensor
  auto options = c10::TensorOptions().dtype(at::kFloat).layout(at::kSparseCsr);
  at::TensorBase sparse_csr_tensor = at::empty({2, 3}, options);
  ASSERT_TRUE(sparse_csr_tensor.is_sparse_csr());
  ASSERT_TRUE(sparse_csr_tensor.is_sparse());
  ASSERT_EQ(sparse_csr_tensor.layout(), c10::kSparseCsr);
}

TEST(SparseEmptyTest, WithOptionalParams) {
  // Test empty with optional parameters
  at::Tensor sparse_tensor = at::empty({2, 3},
                                       at::kFloat,
                                       at::kSparse,
                                       at::kCPU,
                                       /*pin_memory=*/false,
                                       /*memory_format=*/std::nullopt);
  ASSERT_TRUE(sparse_tensor.is_sparse());
  ASSERT_EQ(sparse_tensor.layout(), c10::kSparse);

  at::Tensor sparse_csr_tensor = at::empty({2, 3},
                                           at::kFloat,
                                           at::kSparseCsr,
                                           at::kCPU,
                                           /*pin_memory=*/false,
                                           /*memory_format=*/std::nullopt);
  ASSERT_TRUE(sparse_csr_tensor.is_sparse_csr());
  ASSERT_EQ(sparse_csr_tensor.layout(), c10::kSparseCsr);
}

// ============== at::empty_like sparse tests ==============

TEST(SparseEmptyLikeTest, SparseCOO) {
  at::Tensor base_tensor = at::ones({2, 3}, at::kFloat);

  // Dense empty_like should return false
  at::TensorBase tensor = at::empty_like(base_tensor);
  ASSERT_FALSE(tensor.is_sparse());
  ASSERT_EQ(tensor.layout(), c10::kStrided);

  // Create sparse COO tensor using empty_like
  auto options = c10::TensorOptions().dtype(at::kFloat).layout(at::kSparse);
  at::TensorBase sparse_tensor = at::empty_like(base_tensor, options);
  ASSERT_TRUE(sparse_tensor.is_sparse());
  ASSERT_EQ(sparse_tensor.layout(), c10::kSparse);
}

TEST(SparseEmptyLikeTest, SparseCsr) {
  at::Tensor base_tensor = at::ones({2, 3}, at::kFloat);

  // Create sparse CSR tensor using empty_like
  auto options = c10::TensorOptions().dtype(at::kFloat).layout(at::kSparseCsr);
  at::TensorBase sparse_csr_tensor = at::empty_like(base_tensor, options);
  ASSERT_TRUE(sparse_csr_tensor.is_sparse_csr());
  ASSERT_TRUE(sparse_csr_tensor.is_sparse());
  ASSERT_EQ(sparse_csr_tensor.layout(), c10::kSparseCsr);
}

TEST(SparseEmptyLikeTest, WithOptionalParams) {
  at::Tensor base_tensor = at::ones({2, 3}, at::kFloat);

  // Test empty_like with optional parameters
  at::Tensor sparse_tensor = at::empty_like(base_tensor,
                                            at::kFloat,
                                            at::kSparse,
                                            at::kCPU,
                                            /*pin_memory=*/false,
                                            /*memory_format=*/std::nullopt);
  ASSERT_TRUE(sparse_tensor.is_sparse());
  ASSERT_EQ(sparse_tensor.layout(), c10::kSparse);

  at::Tensor sparse_csr_tensor = at::empty_like(base_tensor,
                                                at::kFloat,
                                                at::kSparseCsr,
                                                at::kCPU,
                                                /*pin_memory=*/false,
                                                /*memory_format=*/std::nullopt);
  ASSERT_TRUE(sparse_csr_tensor.is_sparse_csr());
  ASSERT_EQ(sparse_csr_tensor.layout(), c10::kSparseCsr);
}

// ============== at::zeros_like sparse tests ==============

TEST(SparseZerosLikeTest, SparseCOO) {
  at::Tensor base_tensor = at::ones({2, 3}, at::kFloat);

  // Dense zeros_like should return false
  at::TensorBase tensor = at::zeros_like(base_tensor);
  ASSERT_FALSE(tensor.is_sparse());
  ASSERT_EQ(tensor.layout(), c10::kStrided);

  // Create sparse COO tensor using zeros_like
  auto options = c10::TensorOptions().dtype(at::kFloat).layout(at::kSparse);
  at::TensorBase sparse_tensor = at::zeros_like(base_tensor, options);
  ASSERT_TRUE(sparse_tensor.is_sparse());
  ASSERT_EQ(sparse_tensor.layout(), c10::kSparse);
}

TEST(SparseZerosLikeTest, SparseCsr) {
  at::Tensor base_tensor = at::ones({2, 3}, at::kFloat);

  // Create sparse CSR tensor using zeros_like
  auto options = c10::TensorOptions().dtype(at::kFloat).layout(at::kSparseCsr);
  at::TensorBase sparse_csr_tensor = at::zeros_like(base_tensor, options);
  ASSERT_TRUE(sparse_csr_tensor.is_sparse_csr());
  ASSERT_TRUE(sparse_csr_tensor.is_sparse());
  ASSERT_EQ(sparse_csr_tensor.layout(), c10::kSparseCsr);
}

TEST(SparseZerosLikeTest, WithOptionalParams) {
  at::Tensor base_tensor = at::ones({2, 3}, at::kFloat);

  // Test zeros_like with optional parameters
  at::Tensor sparse_tensor = at::zeros_like(base_tensor,
                                            at::kFloat,
                                            at::kSparse,
                                            at::kCPU,
                                            /*pin_memory=*/false,
                                            /*memory_format=*/std::nullopt);
  ASSERT_TRUE(sparse_tensor.is_sparse());
  ASSERT_EQ(sparse_tensor.layout(), c10::kSparse);

  at::Tensor sparse_csr_tensor = at::zeros_like(base_tensor,
                                                at::kFloat,
                                                at::kSparseCsr,
                                                at::kCPU,
                                                /*pin_memory=*/false,
                                                /*memory_format=*/std::nullopt);
  ASSERT_TRUE(sparse_csr_tensor.is_sparse_csr());
  ASSERT_EQ(sparse_csr_tensor.layout(), c10::kSparseCsr);
}

// ============== at::sparse_coo_tensor tests ==============

TEST(SparseConstructorTest, SparseCooTensorBasic) {
  // Create indices: 2D tensor of shape [sparse_dim, nnz]
  // For a 3x4 sparse tensor with 2 non-zero elements at (0,1) and (2,3)
  at::Tensor indices = at::empty({2, 2}, c10::TensorOptions().dtype(at::kLong));
  int64_t* indices_ptr = indices.data_ptr<int64_t>();
  indices_ptr[0] = 0;  // row index of first non-zero
  indices_ptr[1] = 2;  // row index of second non-zero
  indices_ptr[2] = 1;  // col index of first non-zero
  indices_ptr[3] = 3;  // col index of second non-zero

  // Create values: tensor of shape [nnz]
  at::Tensor values = at::empty({2}, c10::TensorOptions().dtype(at::kFloat));
  float* values_ptr = values.data_ptr<float>();
  values_ptr[0] = 1.0f;
  values_ptr[1] = 2.0f;

  // Create sparse COO tensor
  at::Tensor sparse = at::sparse_coo_tensor(indices, values, {3, 4});

  ASSERT_TRUE(sparse.is_sparse());
  ASSERT_EQ(sparse.layout(), c10::kSparse);
}

TEST(SparseConstructorTest, SparseCooTensorWithOptions) {
  at::Tensor indices = at::empty({2, 2}, c10::TensorOptions().dtype(at::kLong));
  int64_t* indices_ptr = indices.data_ptr<int64_t>();
  indices_ptr[0] = 0;
  indices_ptr[1] = 1;
  indices_ptr[2] = 0;
  indices_ptr[3] = 1;

  at::Tensor values = at::empty({2}, c10::TensorOptions().dtype(at::kFloat));
  float* values_ptr = values.data_ptr<float>();
  values_ptr[0] = 3.0f;
  values_ptr[1] = 4.0f;

  // Create with optional parameters
  at::Tensor sparse = at::sparse_coo_tensor(indices,
                                            values,
                                            {2, 2},
                                            at::kFloat,
                                            at::kSparse,
                                            at::kCPU,
                                            /*pin_memory=*/false);

  ASSERT_TRUE(sparse.is_sparse());
  ASSERT_EQ(sparse.layout(), c10::kSparse);
}

// ============== at::sparse_csr_tensor tests ==============

TEST(SparseConstructorTest, SparseCsrTensorBasic) {
  // Create a 3x4 sparse CSR tensor with 4 non-zero elements:
  // Row 0: values at columns 0, 2
  // Row 1: value at column 1
  // Row 2: value at column 3

  // crow_indices: [0, 2, 3, 4] - compressed row pointers
  at::Tensor crow_indices =
      at::empty({4}, c10::TensorOptions().dtype(at::kLong));
  int64_t* crow_ptr = crow_indices.data_ptr<int64_t>();
  crow_ptr[0] = 0;
  crow_ptr[1] = 2;
  crow_ptr[2] = 3;
  crow_ptr[3] = 4;

  // col_indices: [0, 2, 1, 3] - column indices
  at::Tensor col_indices =
      at::empty({4}, c10::TensorOptions().dtype(at::kLong));
  int64_t* col_ptr = col_indices.data_ptr<int64_t>();
  col_ptr[0] = 0;
  col_ptr[1] = 2;
  col_ptr[2] = 1;
  col_ptr[3] = 3;

  // values: [1.0, 2.0, 3.0, 4.0]
  at::Tensor values = at::empty({4}, c10::TensorOptions().dtype(at::kFloat));
  float* values_ptr = values.data_ptr<float>();
  values_ptr[0] = 1.0f;
  values_ptr[1] = 2.0f;
  values_ptr[2] = 3.0f;
  values_ptr[3] = 4.0f;

  // Create sparse CSR tensor
  at::Tensor sparse =
      at::sparse_csr_tensor(crow_indices, col_indices, values, {3, 4});

  ASSERT_TRUE(sparse.is_sparse_csr());
  ASSERT_TRUE(sparse.is_sparse());
  ASSERT_EQ(sparse.layout(), c10::kSparseCsr);
}

TEST(SparseConstructorTest, SparseCsrTensorWithOptions) {
  // Create a simple 2x2 sparse CSR tensor
  at::Tensor crow_indices =
      at::empty({3}, c10::TensorOptions().dtype(at::kLong));
  int64_t* crow_ptr = crow_indices.data_ptr<int64_t>();
  crow_ptr[0] = 0;
  crow_ptr[1] = 1;
  crow_ptr[2] = 2;

  at::Tensor col_indices =
      at::empty({2}, c10::TensorOptions().dtype(at::kLong));
  int64_t* col_ptr = col_indices.data_ptr<int64_t>();
  col_ptr[0] = 0;
  col_ptr[1] = 1;

  at::Tensor values = at::empty({2}, c10::TensorOptions().dtype(at::kFloat));
  float* values_ptr = values.data_ptr<float>();
  values_ptr[0] = 5.0f;
  values_ptr[1] = 6.0f;

  // Create with optional parameters
  at::Tensor sparse = at::sparse_csr_tensor(crow_indices,
                                            col_indices,
                                            values,
                                            {2, 2},
                                            at::kFloat,
                                            at::kSparseCsr,
                                            at::kCPU,
                                            /*pin_memory=*/false);

  ASSERT_TRUE(sparse.is_sparse_csr());
  ASSERT_TRUE(sparse.is_sparse());
  ASSERT_EQ(sparse.layout(), c10::kSparseCsr);
}

TEST(SparseConstructorTest, SparseCsrTensorMismatchedOptionsDtypeIgnored) {
  // PyTorch ignores dtype mismatch in sparse_csr_tensor;
  // the resulting tensor uses values' original dtype.
  at::Tensor crow_indices =
      at::empty({3}, c10::TensorOptions().dtype(at::kLong));
  int64_t* crow_ptr = crow_indices.data_ptr<int64_t>();
  crow_ptr[0] = 0;
  crow_ptr[1] = 1;
  crow_ptr[2] = 2;

  at::Tensor col_indices =
      at::empty({2}, c10::TensorOptions().dtype(at::kLong));
  int64_t* col_ptr = col_indices.data_ptr<int64_t>();
  col_ptr[0] = 0;
  col_ptr[1] = 1;

  at::Tensor values = at::empty({2}, c10::TensorOptions().dtype(at::kFloat));
  float* values_ptr = values.data_ptr<float>();
  values_ptr[0] = 5.0f;
  values_ptr[1] = 6.0f;

  std::vector<int64_t> size = {2, 2};
  auto options = c10::TensorOptions().dtype(at::kDouble);

  at::Tensor sparse =
      at::sparse_csr_tensor(crow_indices, col_indices, values, size, options);

  // Result should use values' dtype (float), not options' dtype (double).
  ASSERT_EQ(sparse.dtype(), at::kFloat);
}

// ============== Additional sparse_coo_tensor tests ==============

TEST(SparseConstructorTest, SparseCooTensorInferSize) {
  // Test sparse_coo_tensor with inferred size (no explicit size parameter)
  at::Tensor indices = at::empty({2, 3}, c10::TensorOptions().dtype(at::kLong));
  int64_t* indices_ptr = indices.data_ptr<int64_t>();
  // 3 non-zero elements at (0,0), (1,1), (2,2)
  indices_ptr[0] = 0;
  indices_ptr[1] = 1;
  indices_ptr[2] = 2;
  indices_ptr[3] = 0;
  indices_ptr[4] = 1;
  indices_ptr[5] = 2;

  at::Tensor values = at::empty({3}, c10::TensorOptions().dtype(at::kFloat));
  float* values_ptr = values.data_ptr<float>();
  values_ptr[0] = 1.0f;
  values_ptr[1] = 2.0f;
  values_ptr[2] = 3.0f;

  // Create sparse COO tensor with inferred size
  at::Tensor sparse = at::sparse_coo_tensor(indices, values);

  ASSERT_TRUE(sparse.is_sparse());
  ASSERT_EQ(sparse.layout(), c10::kSparse);
  ASSERT_EQ(sparse.dim(), 2);
  ASSERT_EQ(sparse.size(0), 3);
  ASSERT_EQ(sparse.size(1), 3);
}

TEST(SparseConstructorTest, SparseCooTensorDouble) {
  // Test sparse_coo_tensor with double dtype
  at::Tensor indices = at::empty({2, 2}, c10::TensorOptions().dtype(at::kLong));
  int64_t* indices_ptr = indices.data_ptr<int64_t>();
  indices_ptr[0] = 0;
  indices_ptr[1] = 1;
  indices_ptr[2] = 0;
  indices_ptr[3] = 1;

  at::Tensor values = at::empty({2}, c10::TensorOptions().dtype(at::kDouble));
  double* values_ptr = values.data_ptr<double>();
  values_ptr[0] = 1.5;
  values_ptr[1] = 2.5;

  at::Tensor sparse = at::sparse_coo_tensor(indices, values, {2, 2});

  ASSERT_TRUE(sparse.is_sparse());
  ASSERT_EQ(sparse.layout(), c10::kSparse);
}

TEST(SparseConstructorTest, SparseCooTensor3D) {
  // Test 3D sparse COO tensor
  at::Tensor indices = at::empty({3, 2}, c10::TensorOptions().dtype(at::kLong));
  int64_t* indices_ptr = indices.data_ptr<int64_t>();
  // 2 non-zero elements at (0,1,2) and (1,0,1)
  indices_ptr[0] = 0;
  indices_ptr[1] = 1;
  indices_ptr[2] = 1;
  indices_ptr[3] = 0;
  indices_ptr[4] = 2;
  indices_ptr[5] = 1;

  at::Tensor values = at::empty({2}, c10::TensorOptions().dtype(at::kFloat));
  float* values_ptr = values.data_ptr<float>();
  values_ptr[0] = 5.0f;
  values_ptr[1] = 6.0f;

  at::Tensor sparse = at::sparse_coo_tensor(indices, values, {2, 2, 3});

  ASSERT_TRUE(sparse.is_sparse());
  ASSERT_EQ(sparse.layout(), c10::kSparse);
}

// ============== Additional sparse_csr_tensor tests ==============

TEST(SparseConstructorTest, SparseCsrTensorDouble) {
  // Test sparse_csr_tensor with double dtype
  at::Tensor crow_indices =
      at::empty({3}, c10::TensorOptions().dtype(at::kLong));
  int64_t* crow_ptr = crow_indices.data_ptr<int64_t>();
  crow_ptr[0] = 0;
  crow_ptr[1] = 1;
  crow_ptr[2] = 2;

  at::Tensor col_indices =
      at::empty({2}, c10::TensorOptions().dtype(at::kLong));
  int64_t* col_ptr = col_indices.data_ptr<int64_t>();
  col_ptr[0] = 0;
  col_ptr[1] = 1;

  at::Tensor values = at::empty({2}, c10::TensorOptions().dtype(at::kDouble));
  double* values_ptr = values.data_ptr<double>();
  values_ptr[0] = 1.5;
  values_ptr[1] = 2.5;

  at::Tensor sparse =
      at::sparse_csr_tensor(crow_indices, col_indices, values, {2, 2});

  ASSERT_TRUE(sparse.is_sparse_csr());
  ASSERT_TRUE(sparse.is_sparse());
  ASSERT_EQ(sparse.layout(), c10::kSparseCsr);
}

TEST(SparseConstructorTest, SparseCsrTensorLarger) {
  // Test larger sparse CSR tensor (4x5 with 6 non-zero elements)
  // Row 0: values at columns 1, 3
  // Row 1: value at column 2
  // Row 2: values at columns 0, 4
  // Row 3: value at column 2

  at::Tensor crow_indices =
      at::empty({5}, c10::TensorOptions().dtype(at::kLong));
  int64_t* crow_ptr = crow_indices.data_ptr<int64_t>();
  crow_ptr[0] = 0;
  crow_ptr[1] = 2;
  crow_ptr[2] = 3;
  crow_ptr[3] = 5;
  crow_ptr[4] = 6;

  at::Tensor col_indices =
      at::empty({6}, c10::TensorOptions().dtype(at::kLong));
  int64_t* col_ptr = col_indices.data_ptr<int64_t>();
  col_ptr[0] = 1;
  col_ptr[1] = 3;
  col_ptr[2] = 2;
  col_ptr[3] = 0;
  col_ptr[4] = 4;
  col_ptr[5] = 2;

  at::Tensor values = at::empty({6}, c10::TensorOptions().dtype(at::kFloat));
  float* values_ptr = values.data_ptr<float>();
  values_ptr[0] = 1.0f;
  values_ptr[1] = 2.0f;
  values_ptr[2] = 3.0f;
  values_ptr[3] = 4.0f;
  values_ptr[4] = 5.0f;
  values_ptr[5] = 6.0f;

  at::Tensor sparse =
      at::sparse_csr_tensor(crow_indices, col_indices, values, {4, 5});

  ASSERT_TRUE(sparse.is_sparse_csr());
  ASSERT_TRUE(sparse.is_sparse());
  ASSERT_EQ(sparse.layout(), c10::kSparseCsr);
}

TEST(SparseConstructorTest, SparseCsrTensorEmpty) {
  // Test sparse CSR tensor with no non-zero elements
  at::Tensor crow_indices =
      at::empty({4}, c10::TensorOptions().dtype(at::kLong));
  int64_t* crow_ptr = crow_indices.data_ptr<int64_t>();
  crow_ptr[0] = 0;
  crow_ptr[1] = 0;
  crow_ptr[2] = 0;
  crow_ptr[3] = 0;

  at::Tensor col_indices =
      at::empty({0}, c10::TensorOptions().dtype(at::kLong));

  at::Tensor values = at::empty({0}, c10::TensorOptions().dtype(at::kFloat));

  at::Tensor sparse =
      at::sparse_csr_tensor(crow_indices, col_indices, values, {3, 3});

  ASSERT_TRUE(sparse.is_sparse_csr());
  ASSERT_TRUE(sparse.is_sparse());
  ASSERT_EQ(sparse.layout(), c10::kSparseCsr);
}

// ============== Sparse tensor interoperability tests ==============

TEST(SparseInteropTest, SparseCsrFromZeros) {
  // Create sparse CSR tensor from zeros
  auto options = c10::TensorOptions().dtype(at::kFloat).layout(at::kSparseCsr);
  at::Tensor sparse = at::zeros({4, 4}, options);

  ASSERT_TRUE(sparse.is_sparse_csr());
  ASSERT_TRUE(sparse.is_sparse());
  ASSERT_EQ(sparse.layout(), c10::kSparseCsr);
}
