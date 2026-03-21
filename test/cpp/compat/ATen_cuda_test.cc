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
#include <ATen/ops/tensor.h>
#include <c10/core/Device.h>
#include <c10/core/DeviceType.h>
#include <c10/core/ScalarType.h>
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
#include <c10/cuda/CUDAFunctions.h>
#endif

#include "ATen/ATen.h"
#include "gtest/gtest.h"
#include "torch/all.h"

// ============================================================
// Tests for at::Tensor::cuda()
// ============================================================

// After cuda(), the tensor should reside on a GPU device.
TEST(TensorCudaTest, CpuTensorMovesToCuda) {
  at::Tensor cpu_t = at::tensor({1.0f, 2.0f, 3.0f}, at::kFloat);
  ASSERT_TRUE(cpu_t.is_cpu());

  at::Tensor cuda_t = cpu_t.cuda();
  ASSERT_TRUE(cuda_t.is_cuda());
  ASSERT_FALSE(cuda_t.is_cpu());
}

// dtype and numel must be preserved.
TEST(TensorCudaTest, DtypeAndNumelPreserved) {
  at::Tensor cpu_t = at::tensor({1, 2, 3, 4}, at::kInt);
  at::Tensor cuda_t = cpu_t.cuda();

  ASSERT_EQ(cuda_t.scalar_type(), at::kInt);
  ASSERT_EQ(cuda_t.numel(), 4);
}

// Values should round-trip back to CPU intact.
TEST(TensorCudaTest, ValuesPreservedAfterRoundTrip) {
  std::vector<float> data = {1.0f, 2.5f, -3.0f, 4.75f};
  at::Tensor cpu_t = at::tensor(data, at::kFloat);
  at::Tensor cuda_t = cpu_t.cuda();
  at::Tensor back = cuda_t.cpu();

  ASSERT_EQ(back.numel(), static_cast<int64_t>(data.size()));
  for (int64_t i = 0; i < back.numel(); ++i) {
    ASSERT_NEAR(back[i].item<float>(), data[static_cast<size_t>(i)], 1e-5f);
  }
}

// shape (sizes) should be preserved.
TEST(TensorCudaTest, ShapePreserved) {
  at::Tensor cpu_t = at::zeros({2, 3, 4}, at::kFloat);
  at::Tensor cuda_t = cpu_t.cuda();

  ASSERT_EQ(cuda_t.dim(), 3);
  ASSERT_EQ(cuda_t.size(0), 2);
  ASSERT_EQ(cuda_t.size(1), 3);
  ASSERT_EQ(cuda_t.size(2), 4);
}

// An already-CUDA tensor should still be CUDA after another cuda() call.
TEST(TensorCudaTest, AlreadyCudaTensorStaysCuda) {
  at::Tensor cpu_t = at::tensor({7.0f}, at::kFloat);
  at::Tensor cuda_t = cpu_t.cuda();
  at::Tensor cuda_t2 = cuda_t.cuda();

  ASSERT_TRUE(cuda_t2.is_cuda());
  ASSERT_NEAR(cuda_t2.cpu().item<float>(), 7.0f, 1e-6f);
}

// device() should report a CUDA device.
TEST(TensorCudaTest, DeviceIsCuda) {
  at::Tensor cpu_t = at::tensor({0.0f}, at::kFloat);
  at::Tensor cuda_t = cpu_t.cuda();

  ASSERT_EQ(cuda_t.device().type(), c10::DeviceType::CUDA);
}

// is_cuda() / is_cpu() are mutually exclusive.
TEST(TensorCudaTest, IsCudaAndIsCpuMutuallyExclusive) {
  at::Tensor cpu_t = at::tensor({1.0f, 2.0f}, at::kFloat);
  at::Tensor cuda_t = cpu_t.cuda();

  ASSERT_TRUE(cuda_t.is_cuda());
  ASSERT_FALSE(cuda_t.is_cpu());
}
