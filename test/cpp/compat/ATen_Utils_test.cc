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
#include <ATen/Utils.h>
#include <ATen/core/TensorBody.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/EmptyTensor.h>
#include <ATen/native/cuda/Resize.h>
#include <ATen/ops/tensor.h>
#include <c10/core/ScalarType.h>
#include <c10/core/TensorOptions.h>
#include <c10/util/ArrayRef.h>
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
#include <c10/cuda/CUDAFunctions.h>
#include <c10/cuda/CUDAGuard.h>
#endif
#include "ATen/ATen.h"
#include "gtest/gtest.h"
#include "paddle/phi/common/float16.h"
#include "torch/all.h"

// ============================================================
// Tests for at::detail::tensor_cpu / tensor_backend / complex variants
// and the at::tensor() factory macro-generated overloads (ATen/Utils.h)
// ============================================================

// ---- tensor_cpu (via at::tensor public API) ----

TEST(ATenUtilsTest, TensorCPU_Float) {
  std::vector<float> data = {1.0f, 2.0f, 3.0f};
  at::Tensor t = at::tensor(c10::ArrayRef<float>(data),
                            at::TensorOptions().dtype(at::kFloat));

  ASSERT_EQ(t.scalar_type(), at::kFloat);
  ASSERT_EQ(t.numel(), 3);
  ASSERT_NEAR(t[0].item<float>(), 1.0f, 1e-6f);
  ASSERT_NEAR(t[2].item<float>(), 3.0f, 1e-6f);
}

TEST(ATenUtilsTest, TensorCPU_Double) {
  std::vector<double> data = {1.1, 2.2, 3.3};
  at::Tensor t = at::tensor(c10::ArrayRef<double>(data),
                            at::TensorOptions().dtype(at::kDouble));

  ASSERT_EQ(t.scalar_type(), at::kDouble);
  ASSERT_NEAR(t[1].item<double>(), 2.2, 1e-10);
}

TEST(ATenUtilsTest, TensorCPU_Int32) {
  std::vector<int32_t> data = {10, 20, 30};
  at::Tensor t = at::tensor(c10::ArrayRef<int32_t>(data),
                            at::TensorOptions().dtype(at::kInt));

  ASSERT_EQ(t.scalar_type(), at::kInt);
  ASSERT_EQ(t[0].item<int32_t>(), 10);
  ASSERT_EQ(t[2].item<int32_t>(), 30);
}

TEST(ATenUtilsTest, TensorCPU_Int64) {
  std::vector<int64_t> data = {100LL, 200LL};
  at::Tensor t = at::tensor(c10::ArrayRef<int64_t>(data),
                            at::TensorOptions().dtype(at::kLong));

  ASSERT_EQ(t.scalar_type(), at::kLong);
  ASSERT_EQ(t[1].item<int64_t>(), 200LL);
}

TEST(ATenUtilsTest, TensorCPU_Int8) {
  std::vector<int8_t> data = {-1, 0, 1};
  at::Tensor t = at::tensor(c10::ArrayRef<int8_t>(data),
                            at::TensorOptions().dtype(at::kChar));

  ASSERT_EQ(t.scalar_type(), at::kChar);
  ASSERT_EQ(t[0].item<int8_t>(), static_cast<int8_t>(-1));
}

TEST(ATenUtilsTest, TensorCPU_Int16) {
  std::vector<int16_t> data = {256, 512};
  at::Tensor t = at::tensor(c10::ArrayRef<int16_t>(data),
                            at::TensorOptions().dtype(at::kShort));

  ASSERT_EQ(t.scalar_type(), at::kShort);
  ASSERT_EQ(t[0].item<int16_t>(), static_cast<int16_t>(256));
}

TEST(ATenUtilsTest, TensorCPU_UInt8) {
  std::vector<uint8_t> data = {200, 255};
  at::Tensor t = at::tensor(c10::ArrayRef<uint8_t>(data),
                            at::TensorOptions().dtype(at::kByte));

  ASSERT_EQ(t.scalar_type(), at::kByte);
  ASSERT_EQ(t[1].item<uint8_t>(), static_cast<uint8_t>(255));
}

TEST(ATenUtilsTest, TensorCPU_Bool) {
  // std::vector<bool> is a bitfield specialization without data(), so use a
  // plain C array to construct c10::ArrayRef<bool>.
  bool data[] = {true, false, true};
  at::Tensor t = at::tensor(c10::ArrayRef<bool>(data),
                            at::TensorOptions().dtype(at::kBool));

  ASSERT_EQ(t.scalar_type(), at::kBool);
  ASSERT_TRUE(t[0].item<bool>());
  ASSERT_FALSE(t[1].item<bool>());
}

// ---- dtype promotion: values stored as native type then cast ----

TEST(ATenUtilsTest, TensorCPU_DtypePromotion_IntToFloat) {
  // Store int32 values, but request float32 output – should auto-cast.
  std::vector<int32_t> data = {1, 2, 3};
  at::Tensor t = at::tensor(c10::ArrayRef<int32_t>(data),
                            at::TensorOptions().dtype(at::kFloat));

  ASSERT_EQ(t.scalar_type(), at::kFloat);
  ASSERT_NEAR(t[0].item<float>(), 1.0f, 1e-6f);
}

// ---- contiguity ----

TEST(ATenUtilsTest, TensorCPU_IsContiguous) {
  std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f};
  at::Tensor t = at::tensor(c10::ArrayRef<float>(data),
                            at::TensorOptions().dtype(at::kFloat));

  ASSERT_TRUE(t.is_contiguous());
}

// ---- tensor_backend (CPU -> same result since default is CPU in tests) ----

TEST(ATenUtilsTest, TensorBackend_CPUDevice_MatchesTensorCPU) {
  std::vector<float> data = {5.0f, 6.0f};
  at::TensorOptions opts =
      at::TensorOptions().dtype(at::kFloat).device(c10::Device(c10::kCPU));
  at::Tensor t = at::tensor(c10::ArrayRef<float>(data), opts);

  ASSERT_EQ(t.scalar_type(), at::kFloat);
  ASSERT_EQ(t.device().type(), c10::DeviceType::CPU);
  ASSERT_NEAR(t[0].item<float>(), 5.0f, 1e-6f);
}

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
TEST(ATenUtilsTest, TensorBackend_GPUDevice) {
  if (!at::cuda::is_available()) {
    return;
  }
  std::vector<float> data = {7.0f, 8.0f};
  at::TensorOptions opts =
      at::TensorOptions().dtype(at::kFloat).device(c10::Device(c10::kCUDA, 0));
  at::Tensor t = at::tensor(c10::ArrayRef<float>(data), opts);

  ASSERT_EQ(t.scalar_type(), at::kFloat);
  ASSERT_EQ(t.device().type(), c10::DeviceType::CUDA);
}

TEST(ATenUtilsTest, TensorComplexBackend_GPUDevice) {
  if (!at::cuda::is_available()) {
    return;
  }
  std::vector<c10::complex<float>> data = {{1.0f, 0.0f}};
  at::TensorOptions opts = at::TensorOptions()
                               .dtype(at::kComplexFloat)
                               .device(c10::Device(c10::kCUDA, 0));
  at::Tensor t = at::tensor(c10::ArrayRef<c10::complex<float>>(data), opts);

  ASSERT_EQ(t.scalar_type(), at::kComplexFloat);
  ASSERT_EQ(t.device().type(), c10::DeviceType::CUDA);
}
#endif
