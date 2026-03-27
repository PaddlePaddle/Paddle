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
#include <ATen/ops/_local_scalar_dense.h>
#include <ATen/ops/tensor.h>
#include <c10/core/ScalarType.h>
#include <c10/core/TensorOptions.h>
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
#include <c10/cuda/CUDAFunctions.h>
#include <c10/cuda/CUDAGuard.h>
#endif
#include "ATen/ATen.h"
#include "gtest/gtest.h"
#include "paddle/phi/common/bfloat16.h"
#include "paddle/phi/common/float16.h"
#include "test/cpp/compat/cuda_test_utils.h"
#include "torch/all.h"

// ============================================================
// Tests for at::_local_scalar_dense()
// ============================================================

TEST(LocalScalarDenseTest, Float32_ReturnsCorrectValue) {
  at::Tensor t = at::tensor({2.5f}, at::kFloat);
  at::Scalar s = at::_local_scalar_dense(t);

  ASSERT_NEAR(s.to<float>(), 2.5f, 1e-6f);
}

TEST(LocalScalarDenseTest, Float64_ReturnsCorrectValue) {
  at::Tensor t = at::tensor({3.141592653589793}, at::kDouble);
  at::Scalar s = at::_local_scalar_dense(t);

  ASSERT_NEAR(s.to<double>(), 3.141592653589793, 1e-12);
}

TEST(LocalScalarDenseTest, Float16_ReturnsCorrectValue) {
  // Create FP16 tensor from float, then read back via _local_scalar_dense.
  at::Tensor t = at::tensor({1.5f}, at::kHalf);
  at::Scalar s = at::_local_scalar_dense(t);

  // Float16 has ~3 significant decimal digits.
  ASSERT_NEAR(s.to<float>(), 1.5f, 1e-2f);
}

TEST(LocalScalarDenseTest, BFloat16_ReturnsCorrectValue) {
  at::Tensor t = at::tensor({1.0f}, at::kBFloat16);
  at::Scalar s = at::_local_scalar_dense(t);

  ASSERT_NEAR(s.to<float>(), 1.0f, 1e-2f);
}

TEST(LocalScalarDenseTest, Int8_ReturnsCorrectValue) {
  at::Tensor t = at::tensor({static_cast<int8_t>(-7)}, at::kChar);
  at::Scalar s = at::_local_scalar_dense(t);

  ASSERT_EQ(s.to<int8_t>(), static_cast<int8_t>(-7));
}

TEST(LocalScalarDenseTest, Int16_ReturnsCorrectValue) {
  at::Tensor t = at::tensor({static_cast<int16_t>(300)}, at::kShort);
  at::Scalar s = at::_local_scalar_dense(t);

  ASSERT_EQ(s.to<int16_t>(), static_cast<int16_t>(300));
}

TEST(LocalScalarDenseTest, Int32_ReturnsCorrectValue) {
  at::Tensor t = at::tensor({42}, at::kInt);
  at::Scalar s = at::_local_scalar_dense(t);

  ASSERT_EQ(s.to<int32_t>(), 42);
}

TEST(LocalScalarDenseTest, Int64_ReturnsCorrectValue) {
  at::Tensor t = at::tensor({static_cast<int64_t>(9876543210LL)}, at::kLong);
  at::Scalar s = at::_local_scalar_dense(t);

  ASSERT_EQ(s.to<int64_t>(), 9876543210LL);
}

TEST(LocalScalarDenseTest, UInt8_ReturnsCorrectValue) {
  at::Tensor t = at::tensor({static_cast<uint8_t>(255)}, at::kByte);
  at::Scalar s = at::_local_scalar_dense(t);

  ASSERT_EQ(s.to<uint8_t>(), static_cast<uint8_t>(255));
}

TEST(LocalScalarDenseTest, Bool_True_ReturnsCorrectValue) {
  at::Tensor t = at::tensor({true}, at::kBool);
  at::Scalar s = at::_local_scalar_dense(t);

  ASSERT_TRUE(s.to<bool>());
}

TEST(LocalScalarDenseTest, Bool_False_ReturnsCorrectValue) {
  at::Tensor t = at::tensor({false}, at::kBool);
  at::Scalar s = at::_local_scalar_dense(t);

  ASSERT_FALSE(s.to<bool>());
}

TEST(LocalScalarDenseTest, NegativeValue_Float) {
  at::Tensor t = at::tensor({-99.0f}, at::kFloat);
  at::Scalar s = at::_local_scalar_dense(t);

  ASSERT_NEAR(s.to<float>(), -99.0f, 1e-5f);
}

TEST(LocalScalarDenseTest, ZeroValue_Int32) {
  at::Tensor t = at::tensor({0}, at::kInt);
  at::Scalar s = at::_local_scalar_dense(t);

  ASSERT_EQ(s.to<int32_t>(), 0);
}

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
TEST(LocalScalarDenseTest, GPU_Float32_ReturnsCorrectValue) {
  SKIP_IF_CUDA_RUNTIME_UNAVAILABLE();
  // _local_scalar_dense must copy to CPU when the tensor is on GPU.
  at::Tensor t = at::tensor(
      {7.0f},
      at::TensorOptions().dtype(at::kFloat).device(c10::Device(c10::kCUDA, 0)));
  at::Scalar s = at::_local_scalar_dense(t);

  ASSERT_NEAR(s.to<float>(), 7.0f, 1e-5f);
}
#endif

TEST(LocalScalarDenseTest, EmptyTensor_ThrowsCheck) {
  // Passing an empty tensor should trigger PD_CHECK in the implementation.
  at::Tensor t = at::empty({0}, at::kFloat);
  ASSERT_THROW(at::_local_scalar_dense(t), std::exception);
}
