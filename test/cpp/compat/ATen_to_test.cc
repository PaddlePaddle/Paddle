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
#include <c10/core/Device.h>
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
// Tests for at::Tensor::to() overloads
// ============================================================

// ---- Overload 4: to(ScalarType) ----

TEST(TensorToTest, ToDtype_FloatToDouble) {
  at::Tensor t = at::tensor({1.0f, 2.0f, 3.0f}, at::kFloat);
  at::Tensor result = t.to(at::kDouble);

  ASSERT_EQ(result.scalar_type(), at::kDouble);
  ASSERT_EQ(result.numel(), 3);
  ASSERT_NEAR(result[0].item<double>(), 1.0, 1e-10);
  ASSERT_NEAR(result[2].item<double>(), 3.0, 1e-10);
}

TEST(TensorToTest, ToDtype_DoubleToFloat) {
  at::Tensor t = at::tensor({1.5, 2.5}, at::kDouble);
  at::Tensor result = t.to(at::kFloat);

  ASSERT_EQ(result.scalar_type(), at::kFloat);
  ASSERT_NEAR(result[0].item<float>(), 1.5f, 1e-5f);
}

TEST(TensorToTest, ToDtype_FloatToInt32) {
  at::Tensor t = at::tensor({1.9f, 2.1f, 3.7f}, at::kFloat);
  at::Tensor result = t.to(at::kInt);

  ASSERT_EQ(result.scalar_type(), at::kInt);
}

TEST(TensorToTest, ToDtype_SameType_NoAllocation) {
  // When target dtype == current dtype and copy=false, returns self.
  at::Tensor t = at::tensor({4.0f}, at::kFloat);
  at::Tensor result = t.to(at::kFloat, /*non_blocking=*/false, /*copy=*/false);

  ASSERT_EQ(result.scalar_type(), at::kFloat);
  ASSERT_NEAR(result.item<float>(), 4.0f, 1e-6f);
}

TEST(TensorToTest, ToDtype_Int32ToInt64) {
  at::Tensor t = at::tensor({10, 20, 30}, at::kInt);
  at::Tensor result = t.to(at::kLong);

  ASSERT_EQ(result.scalar_type(), at::kLong);
  ASSERT_EQ(result[1].item<int64_t>(), 20LL);
}

TEST(TensorToTest, ToDtype_FloatToHalf) {
  at::Tensor t = at::tensor({1.0f, 2.0f}, at::kFloat);
  at::Tensor result = t.to(at::kHalf);

  ASSERT_EQ(result.scalar_type(), at::kHalf);
}

// ---- Overload 1: to(TensorOptions) ----

TEST(TensorToTest, ToOptions_DtypeOnly) {
  at::Tensor t = at::tensor({5.0f}, at::kFloat);
  at::TensorOptions opts = at::TensorOptions().dtype(at::kDouble);

  at::Tensor result = t.to(opts);

  ASSERT_EQ(result.scalar_type(), at::kDouble);
  ASSERT_NEAR(result.item<double>(), 5.0, 1e-9);
}

TEST(TensorToTest, ToOptions_DeviceCPU) {
  at::Tensor t = at::tensor({3.0f}, at::kFloat);
  at::TensorOptions opts = at::TensorOptions().device(c10::Device(c10::kCPU));

  at::Tensor result = t.to(opts);

  ASSERT_EQ(result.device().type(), c10::DeviceType::CPU);
}

// ---- Overload 2: to(optional<ScalarType>, optional<Layout>, ...) ----

TEST(TensorToTest, ToOptionalArgs_DtypeSet) {
  at::Tensor t = at::ones({3}, at::kFloat);
  at::Tensor result = t.to(at::kDouble,
                           /*layout=*/std::nullopt,
                           /*device=*/std::nullopt,
                           /*pin_memory=*/std::nullopt,
                           /*non_blocking=*/false,
                           /*copy=*/false,
                           /*memory_format=*/std::nullopt);

  ASSERT_EQ(result.scalar_type(), at::kDouble);
}

TEST(TensorToTest, ToOptionalArgs_NothingSet_ReturnsSameType) {
  at::Tensor t = at::ones({3}, at::kFloat);
  at::Tensor result = t.to(std::nullopt,
                           std::nullopt,
                           std::nullopt,
                           std::nullopt,
                           /*non_blocking=*/false,
                           /*copy=*/false,
                           std::nullopt);

  ASSERT_EQ(result.scalar_type(), at::kFloat);
}

TEST(TensorToTest, ToCopyAndUnsupportedDeviceBranches) {
  at::Tensor t = at::ones({2, 3}, at::kFloat);

  at::Tensor copied =
      t.to(at::TensorOptions().dtype(at::kFloat), false, true, std::nullopt);
  EXPECT_TRUE(copied.equal(t));

  at::Tensor pinned = t.to(std::nullopt,
                           std::nullopt,
                           std::nullopt,
                           true,
                           false,
                           false,
                           std::nullopt);
  EXPECT_TRUE(pinned.equal(t));

  EXPECT_THROW(
      t.to(at::TensorOptions().device(c10::Device(c10::DeviceType::XPU, 0))),
      ::std::exception);
}

// ---- Overload 3: to(Device, ScalarType) ----

TEST(TensorToTest, ToDeviceAndDtype) {
  at::Tensor t = at::tensor({1.0f, 2.0f}, at::kFloat);
  at::Tensor result = t.to(c10::Device(c10::kCPU),
                           at::kDouble,
                           /*non_blocking=*/false,
                           /*copy=*/false);

  ASSERT_EQ(result.scalar_type(), at::kDouble);
  ASSERT_EQ(result.device().type(), c10::DeviceType::CPU);
}

// ---- Overload 5: to(const Tensor& other) ----

TEST(TensorToTest, ToOtherTensor_MatchesDtype) {
  at::Tensor src = at::ones({2, 3}, at::kFloat);
  at::Tensor target_template = at::zeros({1}, at::kDouble);

  at::Tensor result = src.to(target_template);

  ASSERT_EQ(result.scalar_type(), at::kDouble);
}

TEST(TensorToTest, ToOtherTensor_MatchesDevice) {
  at::Tensor src = at::ones({3}, at::kFloat);
  at::Tensor target_template =
      at::zeros({1}, at::TensorOptions().dtype(at::kFloat).device(c10::kCPU));

  at::Tensor result = src.to(target_template);

  ASSERT_EQ(result.device().type(), c10::DeviceType::CPU);
}

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
TEST(TensorToTest, ToDtype_GPU_FloatToDouble) {
  if (!at::cuda::is_available()) {
    return;
  }
  at::Tensor t = at::tensor(
      {1.0f, 2.0f},
      at::TensorOptions().dtype(at::kFloat).device(c10::Device(c10::kCUDA, 0)));
  at::Tensor result = t.to(at::kDouble);

  ASSERT_EQ(result.scalar_type(), at::kDouble);
  ASSERT_EQ(result.device().type(), c10::DeviceType::CUDA);
}

TEST(TensorToTest, ToDevice_CPUToGPU) {
  if (!at::cuda::is_available()) {
    return;
  }
  at::Tensor t = at::tensor({5.0f}, at::kFloat);
  at::Tensor result = t.to(c10::Device(c10::kCUDA, 0),
                           at::kFloat,
                           /*non_blocking=*/false,
                           /*copy=*/false);

  ASSERT_EQ(result.device().type(), c10::DeviceType::CUDA);
}

TEST(TensorToTest, ToDevice_GPUToCPU) {
  if (!at::cuda::is_available()) {
    return;
  }
  at::Tensor t = at::tensor(
      {7.0f},
      at::TensorOptions().dtype(at::kFloat).device(c10::Device(c10::kCUDA, 0)));
  at::Tensor result = t.to(at::TensorOptions().device(c10::Device(c10::kCPU)));

  ASSERT_EQ(result.device().type(), c10::DeviceType::CPU);
  ASSERT_NEAR(result.item<float>(), 7.0f, 1e-5f);
}
#endif
