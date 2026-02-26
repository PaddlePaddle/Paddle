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
// Tests for at::Tensor::item() / at::Tensor::item<T>()
// ============================================================

TEST(TensorItemTest, ItemFloat_ReturnsScalar) {
  // item() on a single-element float tensor returns an at::Scalar
  at::Tensor t = at::tensor({3.14f}, at::kFloat);
  at::Scalar s = t.item();

  ASSERT_NEAR(s.to<float>(), 3.14f, 1e-5f);
}

TEST(TensorItemTest, ItemDouble_ReturnsScalar) {
  // item() on a single-element double tensor
  at::Tensor t = at::tensor({2.718281828}, at::kDouble);
  at::Scalar s = t.item();

  ASSERT_NEAR(s.to<double>(), 2.718281828, 1e-9);
}

TEST(TensorItemTest, ItemInt32_ReturnsScalar) {
  // item() on a single-element int32 tensor
  at::Tensor t = at::tensor({42}, at::kInt);
  at::Scalar s = t.item();

  ASSERT_EQ(s.to<int32_t>(), 42);
}

TEST(TensorItemTest, ItemInt64_ReturnsScalar) {
  // item() on a single-element int64 tensor
  at::Tensor t = at::tensor({static_cast<int64_t>(1234567890)}, at::kLong);
  at::Scalar s = t.item();

  ASSERT_EQ(s.to<int64_t>(), 1234567890LL);
}

TEST(TensorItemTest, ItemTemplated_Float) {
  // item<float>() returns float directly
  at::Tensor t = at::tensor({1.5f}, at::kFloat);
  float val = t.item<float>();

  ASSERT_FLOAT_EQ(val, 1.5f);
}

TEST(TensorItemTest, ItemTemplated_Double) {
  // item<double>() returns double directly
  at::Tensor t = at::tensor({1.0 / 3.0}, at::kDouble);
  double val = t.item<double>();

  ASSERT_NEAR(val, 1.0 / 3.0, 1e-15);
}

TEST(TensorItemTest, ItemTemplated_Int32) {
  // item<int32_t>() on int32 tensor
  at::Tensor t = at::tensor({-7}, at::kInt);
  int32_t val = t.item<int32_t>();

  ASSERT_EQ(val, -7);
}

TEST(TensorItemTest, ItemFromSqueezed1D) {
  // item() works on a tensor that has been reshaped to single element via
  // squeeze / indexing
  at::Tensor t = at::arange(6, at::kFloat).reshape({2, 3});
  at::Tensor elem = t[1][2];  // value = 5.0

  ASSERT_FLOAT_EQ(elem.item<float>(), 5.0f);
}

TEST(TensorItemTest, ItemOnMultiElementTensorThrows) {
  // item() on a tensor with more than one element must throw.
  at::Tensor t = at::ones({2, 3}, at::kFloat);
  ASSERT_THROW(t.item(), std::exception);
}

// ============================================================
// Tests for at::Tensor::is_variable()
// ============================================================

TEST(TensorIsVariableTest, AlwaysReturnsTrue) {
  // is_variable() is always true in the eager execution mode.
  at::Tensor t = at::ones({3, 4}, at::kFloat);
  ASSERT_TRUE(t.is_variable());
}

TEST(TensorIsVariableTest, AlwaysTrueForScalarTensor) {
  at::Tensor t = at::tensor({1.0f}, at::kFloat);
  ASSERT_TRUE(t.is_variable());
}

TEST(TensorIsVariableTest, AlwaysTrueFor1D) {
  at::Tensor t = at::arange(10, at::kFloat);
  ASSERT_TRUE(t.is_variable());
}

// ============================================================
// Tests for at::Tensor::item() — sparse tensor paths
// ============================================================

TEST(TensorItemSparseTest, EmptySparseCOO_ItemReturnsZero) {
  // A sparse tensor with nnz == 0: item() must return zero (Scalar(0)).
  at::Tensor indices = at::zeros({2, 0}, at::kLong);
  at::Tensor values = at::zeros({0}, at::kFloat);
  // 1x1 empty sparse tensor
  at::Tensor sparse = at::sparse_coo_tensor(indices, values, {1, 1});
  sparse = sparse.coalesce();

  at::Scalar s = sparse.item();

  ASSERT_NEAR(s.to<float>(), 0.0f, 1e-6f);
}

TEST(TensorItemSparseTest, CoalescedSparseCOO_SingleNonZero_ReturnsValue) {
  // 1x1 sparse COO with one non-zero at (0,0) = 5.0.
  at::Tensor indices = at::tensor({0, 0}, at::kLong).reshape({2, 1});
  at::Tensor values = at::tensor({5.0f}, at::kFloat);
  at::Tensor sparse = at::sparse_coo_tensor(indices, values, {1, 1});
  sparse = sparse.coalesce();

  ASSERT_TRUE(sparse.is_coalesced());
  at::Scalar s = sparse.item();

  ASSERT_NEAR(s.to<float>(), 5.0f, 1e-5f);
}

TEST(TensorItemSparseTest, NonCoalescedSparseCOO_DuplicateIndices_SumsValues) {
  // Two entries both at (0,0): item() must sum them (3 + 7 = 10).
  at::Tensor indices = at::tensor({0, 0, 0, 0}, at::kLong).reshape({2, 2});
  at::Tensor values = at::tensor({3.0f, 7.0f}, at::kFloat);
  at::Tensor sparse = at::sparse_coo_tensor(indices, values, {1, 1});

  // Do NOT coalesce — exercising the non-coalesced path.
  ASSERT_FALSE(sparse.is_coalesced());
  at::Scalar s = sparse.item();

  ASSERT_NEAR(s.to<float>(), 10.0f, 1e-5f);
}
