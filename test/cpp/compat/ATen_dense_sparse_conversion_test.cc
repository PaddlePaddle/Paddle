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
#include <c10/core/TensorOptions.h>
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
#include <c10/cuda/CUDAFunctions.h>
#include <c10/cuda/CUDAGuard.h>
#endif
#include "ATen/ATen.h"
#include "gtest/gtest.h"
#include "paddle/phi/common/float16.h"
#include "torch/all.h"
#include "utils/dense_sparse_conversion.h"

// ============================================================
// Tests for compat::_PD_ConvertToSparseIfNeeded()
// ============================================================

// Helper: create a small 2-D dense float tensor.
static paddle::Tensor make_dense_2d(int rows = 3, int cols = 3) {
  // Use at::ones, but get the underlying paddle::Tensor.
  at::Tensor t = at::ones({rows, cols}, at::kFloat);
  return t._PD_GetInner();
}

// ---- kStrided -> dense (no conversion) ----

TEST(DenseSparseConversionTest, kStrided_ReturnsDense) {
  paddle::Tensor dense = make_dense_2d();
  at::Tensor result = compat::_PD_ConvertToSparseIfNeeded(dense, c10::kStrided);

  ASSERT_EQ(result.layout(), c10::kStrided);
}

// ---- unsupported layout throws ----

TEST(DenseSparseConversionTest, UnsupportedLayout_Throws) {
  paddle::Tensor dense = make_dense_2d();

  // kSparseBsr is not handled in the switch => PD_CHECK(false, ...) fires.
  ASSERT_THROW(compat::_PD_ConvertToSparseIfNeeded(dense, c10::kSparseBsr),
               std::exception);
}
