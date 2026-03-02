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
// Tests for at::Tensor::view_as(const at::Tensor& other)
// ============================================================

TEST(TensorViewAsTest, ViewAsSameShape) {
  // view_as with same shape: result has identical shape
  at::Tensor t = at::arange(12, at::kFloat).reshape({3, 4});
  at::Tensor other = at::zeros({3, 4}, at::kFloat);
  at::Tensor result = t.view_as(other);

  ASSERT_EQ(result.sizes(), other.sizes());
  ASSERT_EQ(result.numel(), t.numel());
}

TEST(TensorViewAsTest, ViewAsDifferentShape_CompatibleNumel) {
  // view_as with a different but numel-compatible shape
  at::Tensor t = at::arange(12, at::kFloat);
  at::Tensor other = at::zeros({3, 4}, at::kFloat);
  at::Tensor result = t.view_as(other);

  ASSERT_EQ(result.dim(), 2);
  ASSERT_EQ(result.sizes(), c10::IntArrayRef({3, 4}));
}

TEST(TensorViewAsTest, ViewAsPreservesData) {
  // Elements are accessible with the new shape and preserve original values
  at::Tensor t = at::arange(6, at::kFloat);
  // t = [0, 1, 2, 3, 4, 5]
  at::Tensor other = at::zeros({2, 3}, at::kFloat);
  at::Tensor result = t.view_as(other);

  // result[0] = [0,1,2], result[1] = [3,4,5]
  ASSERT_FLOAT_EQ(result[0][0].item<float>(), 0.0f);
  ASSERT_FLOAT_EQ(result[0][2].item<float>(), 2.0f);
  ASSERT_FLOAT_EQ(result[1][0].item<float>(), 3.0f);
  ASSERT_FLOAT_EQ(result[1][2].item<float>(), 5.0f);
}

TEST(TensorViewAsTest, ViewAs1D_Flattens) {
  // view_as a 1-D tensor to flatten a higher-rank tensor
  at::Tensor t = at::ones({2, 3, 4}, at::kFloat);
  at::Tensor flat_ref = at::zeros({24}, at::kFloat);
  at::Tensor result = t.view_as(flat_ref);

  ASSERT_EQ(result.dim(), 1);
  ASSERT_EQ(result.sizes(), c10::IntArrayRef({24}));
}

TEST(TensorViewAsTest, ViewAs_SameDataPointer) {
  // view_as should share the underlying data (no copy)
  at::Tensor t = at::arange(12, at::kFloat);
  at::Tensor other = at::zeros({3, 4}, at::kFloat);
  at::Tensor result = t.view_as(other);

  ASSERT_EQ(result.data_ptr(), t.data_ptr());
}
