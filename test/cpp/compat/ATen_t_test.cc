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
