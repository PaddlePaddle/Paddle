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
// Tests for at::Tensor::t() and at::Tensor::t_()
// ============================================================

TEST(TensorTTest, T1D_ReturnsSameShape) {
  // t() on a 1D tensor: transposing a 1D tensor returns itself (same shape)
  at::Tensor t = at::arange(5, at::kFloat);
  at::Tensor result = t.t();

  ASSERT_EQ(result.dim(), 1);
  ASSERT_EQ(result.sizes(), c10::IntArrayRef({5}));
  ASSERT_EQ(result.numel(), t.numel());
}

TEST(TensorTTest, T2D_TransposesShape) {
  // t() on a 2D tensor: returns transposed shape
  at::Tensor t = at::ones({3, 4}, at::kFloat);
  at::Tensor result = t.t();

  ASSERT_EQ(result.dim(), 2);
  ASSERT_EQ(result.sizes(), c10::IntArrayRef({4, 3}));
  ASSERT_EQ(result.numel(), t.numel());
}

TEST(TensorTTest, T2D_PreservesValues) {
  // t() on 2D tensor: verify element access after transpose
  at::Tensor t = at::arange(6, at::kFloat).reshape({2, 3});
  // t = [[0,1,2],[3,4,5]]
  at::Tensor result = t.t();
  // result = [[0,3],[1,4],[2,5]]

  ASSERT_EQ(result.sizes(), c10::IntArrayRef({3, 2}));
  // Check [0][0] == 0, [1][0] == 1, [0][1] == 3
  ASSERT_FLOAT_EQ(result[0][0].item<float>(), 0.0f);
  ASSERT_FLOAT_EQ(result[1][0].item<float>(), 1.0f);
  ASSERT_FLOAT_EQ(result[0][1].item<float>(), 3.0f);
  ASSERT_FLOAT_EQ(result[2][1].item<float>(), 5.0f);
}

TEST(TensorTTest, TInplace1D_DoesNotChangeShape) {
  // t_() on a 1D tensor: shape remains the same, returns self
  at::Tensor t = at::arange(5, at::kFloat);
  void* original_ptr = t.data_ptr();
  at::Tensor& ref = t.t_();

  ASSERT_EQ(t.dim(), 1);
  ASSERT_EQ(t.sizes(), c10::IntArrayRef({5}));
  // Must return *this by reference
  ASSERT_EQ(&ref, &t);
  // Data must remain in place
  ASSERT_EQ(t.data_ptr(), original_ptr);
}

TEST(TensorTTest, TInplace2D_TransposesInPlace) {
  // t_() on 2D tensor: shape becomes transposed, data pointer unchanged
  at::Tensor t = at::ones({3, 4}, at::kFloat);
  void* original_ptr = t.data_ptr();
  t.t_();

  ASSERT_EQ(t.dim(), 2);
  ASSERT_EQ(t.sizes(), c10::IntArrayRef({4, 3}));
  ASSERT_EQ(t.data_ptr(), original_ptr);
}

TEST(TensorTTest, TInplace2D_PreservesValues) {
  // t_() on 2D tensor: values are correct after in-place transpose
  at::Tensor t = at::arange(6, at::kFloat).reshape({2, 3});
  // t = [[0,1,2],[3,4,5]]
  t.t_();
  // After t_: shape is {3,2}, t = [[0,3],[1,4],[2,5]]

  ASSERT_EQ(t.sizes(), c10::IntArrayRef({3, 2}));
  ASSERT_FLOAT_EQ(t[0][0].item<float>(), 0.0f);
  ASSERT_FLOAT_EQ(t[0][1].item<float>(), 3.0f);
  ASSERT_FLOAT_EQ(t[2][1].item<float>(), 5.0f);
}

// ============================================================
// High-dimensional tests (dim > 2):
// t() / t_() always swap axes 0 and 1 only; remaining axes stay in place.
// ============================================================

TEST(TensorTTest, T3D_SwapsOnlyDim0AndDim1) {
  // For a 3D tensor {A, B, C}, t() should produce shape {B, A, C}.
  // The innermost axis (dim 2) must NOT be touched.
  at::Tensor t = at::ones({2, 3, 4}, at::kFloat);
  at::Tensor result = t.t();

  ASSERT_EQ(result.dim(), 3);
  ASSERT_EQ(result.sizes(), c10::IntArrayRef({3, 2, 4}));
}

TEST(TensorTTest, T3D_PreservesValues) {
  // Verify that element access is consistent after transposing a 3D tensor.
  // t = arange(24).reshape({2,3,4})
  // t[i][j][k] = i*12 + j*4 + k
  // After t(): result[j][i][k] should still equal i*12 + j*4 + k.
  at::Tensor t = at::arange(24, at::kFloat).reshape({2, 3, 4});
  at::Tensor r = t.t();

  ASSERT_EQ(r.sizes(), c10::IntArrayRef({3, 2, 4}));
  // r[j][i][k] == t[i][j][k]
  for (int64_t i = 0; i < 2; ++i) {
    for (int64_t j = 0; j < 3; ++j) {
      for (int64_t k = 0; k < 4; ++k) {
        ASSERT_FLOAT_EQ(r[j][i][k].item<float>(), t[i][j][k].item<float>());
      }
    }
  }
}

TEST(TensorTTest, T4D_SwapsOnlyDim0AndDim1) {
  // For a 4D tensor {A, B, C, D}, t() should produce shape {B, A, C, D}.
  at::Tensor t = at::ones({2, 5, 3, 4}, at::kFloat);
  at::Tensor result = t.t();

  ASSERT_EQ(result.dim(), 4);
  ASSERT_EQ(result.sizes(), c10::IntArrayRef({5, 2, 3, 4}));
}

TEST(TensorTTest, TInplace3D_SwapsOnlyDim0AndDim1) {
  // t_() on a 3D tensor: shape {A,B,C} -> {B,A,C}, data pointer unchanged.
  at::Tensor t = at::ones({2, 3, 4}, at::kFloat);
  void* original_ptr = t.data_ptr();
  t.t_();

  ASSERT_EQ(t.dim(), 3);
  ASSERT_EQ(t.sizes(), c10::IntArrayRef({3, 2, 4}));
  ASSERT_EQ(t.data_ptr(), original_ptr);
}

TEST(TensorTTest, TInplace3D_HigherDimsUnchanged) {
  // After t_() on a 3D tensor, verify that dim 2 is not touched.
  at::Tensor t = at::arange(24, at::kFloat).reshape({2, 3, 4});
  t.t_();
  // Shape must be {3, 2, 4}: C=4 must be preserved.
  ASSERT_EQ(t.size(2), 4);
}

TEST(TensorTTest, TInplace4D_SwapsOnlyDim0AndDim1) {
  // t_() on a 4D tensor: shape {A,B,C,D} -> {B,A,C,D}.
  at::Tensor t = at::ones({2, 5, 3, 4}, at::kFloat);
  void* original_ptr = t.data_ptr();
  t.t_();

  ASSERT_EQ(t.sizes(), c10::IntArrayRef({5, 2, 3, 4}));
  ASSERT_EQ(t.data_ptr(), original_ptr);
}
