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
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/EmptyTensor.h>
#include <ATen/native/cuda/Resize.h>
#include <ATen/ops/tensor.h>
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
// Tests for at::eye()
// ============================================================

// Helper: verify that a 2-D tensor is an identity-like matrix
// (diagonal == 1, off-diagonal == 0).
static void CheckEye(const at::Tensor& t, int64_t rows, int64_t cols) {
  ASSERT_EQ(t.dim(), 2);
  ASSERT_EQ(t.size(0), rows);
  ASSERT_EQ(t.size(1), cols);
  for (int64_t i = 0; i < rows; ++i) {
    for (int64_t j = 0; j < cols; ++j) {
      float expected = (i == j) ? 1.0f : 0.0f;
      ASSERT_FLOAT_EQ(t[i][j].item<float>(), expected)
          << "Mismatch at (" << i << ", " << j << ")";
    }
  }
}

// ---- eye(n) -------------------------------------------------------

TEST(ATenEyeTest, SquareDefaultDtype) {
  // eye(n) should produce an n×n float32 identity matrix.
  at::Tensor t = at::eye(4);
  ASSERT_EQ(t.scalar_type(), at::kFloat);
  CheckEye(t, 4, 4);
}

TEST(ATenEyeTest, SquareTensorOptionsFloat) {
  // eye(n, TensorOptions) — explicit float32.
  at::Tensor t = at::eye(3, at::TensorOptions().dtype(at::kFloat));
  ASSERT_EQ(t.scalar_type(), at::kFloat);
  CheckEye(t, 3, 3);
}

TEST(ATenEyeTest, SquareTensorOptionsDouble) {
  // eye(n, TensorOptions) — explicit float64.
  at::Tensor t = at::eye(5, at::TensorOptions().dtype(at::kDouble));
  ASSERT_EQ(t.scalar_type(), at::kDouble);
  ASSERT_EQ(t.size(0), 5);
  ASSERT_EQ(t.size(1), 5);
  for (int64_t i = 0; i < 5; ++i) {
    ASSERT_DOUBLE_EQ(t[i][i].item<double>(), 1.0);
    if (i + 1 < 5) {
      ASSERT_DOUBLE_EQ(t[i][i + 1].item<double>(), 0.0);
    }
  }
}

// eye(n, dtype, layout, device, pin_memory) — separate-params overload

TEST(ATenEyeTest, SquareSeparateParamsFloat) {
  at::Tensor t =
      at::eye(4, at::kFloat, /*layout=*/std::nullopt, at::kCPU, false);
  ASSERT_EQ(t.scalar_type(), at::kFloat);
  CheckEye(t, 4, 4);
}

TEST(ATenEyeTest, SquareSeparateParamsNulloptDtype) {
  // When dtype is nullopt the default dtype (float32) should be used.
  at::Tensor t =
      at::eye(3, std::nullopt, /*layout=*/std::nullopt, at::kCPU, false);
  ASSERT_EQ(t.scalar_type(), at::kFloat);
  CheckEye(t, 3, 3);
}

// ---- eye(n, m) -------------------------------------------------------

TEST(ATenEyeTest, RectangularWiderThanTall) {
  // n < m: identity portion fits entirely within row range.
  at::Tensor t = at::eye(3, 5);
  ASSERT_EQ(t.scalar_type(), at::kFloat);
  CheckEye(t, 3, 5);
}

TEST(ATenEyeTest, RectangularTallerThanWide) {
  // n > m: identity portion fits entirely within column range.
  at::Tensor t = at::eye(5, 3);
  ASSERT_EQ(t.scalar_type(), at::kFloat);
  CheckEye(t, 5, 3);
}

TEST(ATenEyeTest, RectangularSquareEquivalent) {
  // eye(n, n) should behave like eye(n).
  at::Tensor t2 = at::eye(4, 4);
  at::Tensor t1 = at::eye(4);
  CheckEye(t2, 4, 4);
  for (int64_t i = 0; i < 4; ++i)
    for (int64_t j = 0; j < 4; ++j)
      ASSERT_FLOAT_EQ(t1[i][j].item<float>(), t2[i][j].item<float>());
}

TEST(ATenEyeTest, RectangularTensorOptionsDouble) {
  // eye(n, m, TensorOptions) — float64.
  at::Tensor t = at::eye(2, 4, at::TensorOptions().dtype(at::kDouble));
  ASSERT_EQ(t.scalar_type(), at::kDouble);
  ASSERT_EQ(t.size(0), 2);
  ASSERT_EQ(t.size(1), 4);
  ASSERT_DOUBLE_EQ(t[0][0].item<double>(), 1.0);
  ASSERT_DOUBLE_EQ(t[1][1].item<double>(), 1.0);
  ASSERT_DOUBLE_EQ(t[0][1].item<double>(), 0.0);
}

TEST(ATenEyeTest, RectangularSeparateParams) {
  // eye(n, m, dtype, layout, device, pin_memory)
  at::Tensor t =
      at::eye(3, 5, at::kDouble, /*layout=*/std::nullopt, at::kCPU, false);
  ASSERT_EQ(t.scalar_type(), at::kDouble);
  CheckEye(t, 3, 5);
}

TEST(ATenEyeTest, RectangularSeparateParamsNulloptDtype) {
  at::Tensor t =
      at::eye(4, 6, std::nullopt, /*layout=*/std::nullopt, at::kCPU, false);
  ASSERT_EQ(t.scalar_type(), at::kFloat);
  CheckEye(t, 4, 6);
}

// ---- 1×1 edge case -------------------------------------------------------

TEST(ATenEyeTest, OneByOne) {
  at::Tensor t = at::eye(1);
  ASSERT_EQ(t.numel(), 1);
  ASSERT_FLOAT_EQ(t[0][0].item<float>(), 1.0f);
}

// ---- GPU tests (compiled only when CUDA / HIP is available) --------------

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
TEST(ATenEyeTest, SquareOnGPU) {
  if (!at::cuda::is_available()) {
    return;
  }
  at::Tensor t =
      at::eye(4, at::TensorOptions().dtype(at::kFloat).device(at::kCUDA));
  at::Tensor t_cpu = t.to(at::kCPU);
  CheckEye(t_cpu, 4, 4);
}

TEST(ATenEyeTest, RectangularOnGPU) {
  if (!at::cuda::is_available()) {
    return;
  }
  at::Tensor t =
      at::eye(3, 5, at::TensorOptions().dtype(at::kFloat).device(at::kCUDA));
  at::Tensor t_cpu = t.to(at::kCPU);
  CheckEye(t_cpu, 3, 5);
}
#endif
