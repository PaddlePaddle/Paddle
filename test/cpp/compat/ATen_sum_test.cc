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
// Tests for at::Tensor::sum()
// ============================================================

TEST(TensorSumTest, SumAllElementsNoArgs) {
  // sum() without arguments: sum of all elements, keeps original dtype
  at::Tensor t = at::ones({2, 3}, at::kFloat);
  at::Tensor result = t.sum();

  ASSERT_EQ(result.numel(), 1);
  ASSERT_FLOAT_EQ(result.item<float>(), 6.0f);
}

TEST(TensorSumTest, SumAllElementsWithDtype) {
  // sum(dtype): reduce all elements, cast result to given dtype
  at::Tensor t = at::ones({4, 4}, at::kFloat);
  at::Tensor result = t.sum(at::kDouble);

  ASSERT_EQ(result.numel(), 1);
  ASSERT_EQ(result.scalar_type(), at::kDouble);
  ASSERT_DOUBLE_EQ(result.item<double>(), 16.0);
}

TEST(TensorSumTest, SumAlongDim0) {
  // sum(dim={0}): reduce along first dimension
  at::Tensor t = at::ones({3, 4}, at::kFloat);
  at::Tensor result = t.sum(at::IntArrayRef{0}, /*keepdim=*/false);

  ASSERT_EQ(result.dim(), 1);
  ASSERT_EQ(result.sizes(), c10::IntArrayRef({4}));
  for (int64_t i = 0; i < 4; ++i) {
    ASSERT_FLOAT_EQ(result[i].item<float>(), 3.0f);
  }
}

TEST(TensorSumTest, SumAlongDim1) {
  // sum(dim={1}): reduce along second dimension
  at::Tensor t = at::ones({3, 4}, at::kFloat);
  at::Tensor result = t.sum(at::IntArrayRef{1}, /*keepdim=*/false);

  ASSERT_EQ(result.dim(), 1);
  ASSERT_EQ(result.sizes(), c10::IntArrayRef({3}));
  for (int64_t i = 0; i < 3; ++i) {
    ASSERT_FLOAT_EQ(result[i].item<float>(), 4.0f);
  }
}

TEST(TensorSumTest, SumAlongDimKeepDim) {
  // sum(dim, keepdim=true): result keeps reduced dimension as size 1
  at::Tensor t = at::ones({3, 4}, at::kFloat);
  at::Tensor result = t.sum(at::IntArrayRef{1}, /*keepdim=*/true);

  ASSERT_EQ(result.dim(), 2);
  ASSERT_EQ(result.sizes(), c10::IntArrayRef({3, 1}));
  for (int64_t i = 0; i < 3; ++i) {
    ASSERT_FLOAT_EQ(result[i][0].item<float>(), 4.0f);
  }
}

TEST(TensorSumTest, SumAlongDimWithDtypeCast) {
  // sum(dim, keepdim, dtype): reduce and cast to specified dtype
  at::Tensor t = at::ones({2, 5}, at::kFloat);
  at::Tensor result =
      t.sum(at::IntArrayRef{0}, /*keepdim=*/false, /*dtype=*/at::kDouble);

  ASSERT_EQ(result.scalar_type(), at::kDouble);
  ASSERT_EQ(result.sizes(), c10::IntArrayRef({5}));
  for (int64_t i = 0; i < 5; ++i) {
    ASSERT_DOUBLE_EQ(result[i].item<double>(), 2.0);
  }
}

TEST(TensorSumTest, SumPreservesNumel) {
  // Verify that sum of known values is correct
  at::Tensor t = at::arange(6, at::kFloat).reshape({2, 3});
  // t = [[0,1,2],[3,4,5]], total = 15
  at::Tensor result = t.sum();
  ASSERT_FLOAT_EQ(result.item<float>(), 15.0f);
}
