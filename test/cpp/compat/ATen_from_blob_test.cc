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

#include <ATen/cuda/CUDAContext.h>
#include <ATen/ops/from_blob.h>
#include <c10/core/ScalarType.h>
#include <c10/core/TensorOptions.h>

#include "ATen/ATen.h"
#include "gtest/gtest.h"
#include "paddle/common/macros.h"
#include "torch/all.h"

COMMON_DECLARE_bool(use_stride_kernel);

#if defined(PADDLE_WITH_CUDA)
#include <cuda_runtime.h>
#elif defined(PADDLE_WITH_HIP)
#include <hip/hip_runtime.h>
#endif

// ======================== CPU place detection ========================

// No device specified: CPU pointer → tensor must be on CPU.
TEST(ATenFromBlobTest, CpuPtrDefaultsToCpu) {
  float data[4] = {1.0f, 2.0f, 3.0f, 4.0f};
  at::Tensor t = at::from_blob(data, {4}, at::kFloat);
  ASSERT_TRUE(t.is_cpu());
  ASSERT_EQ(t.scalar_type(), at::kFloat);
  ASSERT_EQ(t.numel(), 4);
}

// Explicitly pass CPU options: still CPU.
TEST(ATenFromBlobTest, CpuPtrWithCpuOptions) {
  float data[3] = {1.0f, 2.0f, 3.0f};
  at::Tensor t = at::from_blob(
      data, {3}, at::TensorOptions().dtype(at::kFloat).device(at::kCPU));
  ASSERT_TRUE(t.is_cpu());
}

// Data pointer must be preserved (no copy).
TEST(ATenFromBlobTest, DataPtrPreserved) {
  float data[4] = {10.f, 20.f, 30.f, 40.f};
  at::Tensor t = at::from_blob(data, {4}, at::kFloat);
  ASSERT_EQ(t.data_ptr<float>(), data);
}

// Shape and strides are correctly set.
TEST(ATenFromBlobTest, ShapeAndStrides) {
  float data[6] = {};
  at::Tensor t = at::from_blob(data, {2, 3}, at::kFloat);
  ASSERT_EQ(t.sizes()[0], 2);
  ASSERT_EQ(t.sizes()[1], 3);
  // contiguous strides: [3, 1]
  ASSERT_EQ(t.strides()[0], 3);
  ASSERT_EQ(t.strides()[1], 1);
}

// Explicit strides overload.
TEST(ATenFromBlobTest, ExplicitStrides) {
  if (!FLAGS_use_stride_kernel) {
    return;
  }
  // Row-major 2×3 laid out in memory, but we interpret as column-major strides
  float data[6] = {1, 2, 3, 4, 5, 6};
  at::Tensor t = at::from_blob(data, {2, 3}, {1, 2}, at::kFloat);
  ASSERT_EQ(t.strides()[0], 1);
  ASSERT_EQ(t.strides()[1], 2);
  ASSERT_TRUE(t.is_cpu());
}

// Deleter is called when the tensor is destroyed.
TEST(ATenFromBlobTest, DeleterCalled) {
  bool deleted = false;
  {
    float* data = new float[4]{};
    at::Tensor t = at::from_blob(
        data,
        {4},
        [&deleted](void* p) {
          deleted = true;
          delete[] static_cast<float*>(p);
        },
        at::kFloat);
    ASSERT_FALSE(deleted);
  }
  ASSERT_TRUE(deleted);
}

// Deleter + strides overload.
TEST(ATenFromBlobTest, DeleterWithStrides) {
  bool deleted = false;
  {
    float* data = new float[6]{};
    at::Tensor t = at::from_blob(
        data,
        {2, 3},
        {3, 1},
        [&deleted](void* p) {
          deleted = true;
          delete[] static_cast<float*>(p);
        },
        at::kFloat);
    ASSERT_FALSE(deleted);
    ASSERT_TRUE(t.is_cpu());
  }
  ASSERT_TRUE(deleted);
}

// ======================== GPU place detection ========================

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)

// No device specified: GPU pointer → tensor must be on CUDA automatically.
TEST(ATenFromBlobTest, GpuPtrDefaultsToCuda) {
  if (!at::cuda::is_available()) {
    return;
  }
  float* d_data = nullptr;
#if defined(PADDLE_WITH_CUDA)
  cudaMalloc(&d_data, 4 * sizeof(float));
#else
  hipMalloc(&d_data, 4 * sizeof(float));
#endif

  at::Tensor t = at::from_blob(d_data, {4}, at::kFloat);
  ASSERT_TRUE(t.is_cuda())
      << "Expected GPU tensor when data pointer lives on device";
  ASSERT_FALSE(t.is_cpu());
  ASSERT_EQ(t.scalar_type(), at::kFloat);
  ASSERT_EQ(t.numel(), 4);
  ASSERT_EQ(t.data_ptr<float>(), d_data);

#if defined(PADDLE_WITH_CUDA)
  cudaFree(d_data);
#else
  hipFree(d_data);
#endif
}

// Explicit CUDA device option + GPU pointer → still CUDA.
TEST(ATenFromBlobTest, GpuPtrWithCudaOptions) {
  if (!at::cuda::is_available()) {
    return;
  }
  float* d_data = nullptr;
#if defined(PADDLE_WITH_CUDA)
  cudaMalloc(&d_data, 4 * sizeof(float));
#else
  hipMalloc(&d_data, 4 * sizeof(float));
#endif

  at::Tensor t = at::from_blob(
      d_data, {4}, at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0));
  ASSERT_TRUE(t.is_cuda());

#if defined(PADDLE_WITH_CUDA)
  cudaFree(d_data);
#else
  hipFree(d_data);
#endif
}

// target_device overrides auto-detection.
TEST(ATenFromBlobTest, TargetDeviceOverride) {
  if (!at::cuda::is_available()) {
    return;
  }
  float* d_data = nullptr;
#if defined(PADDLE_WITH_CUDA)
  cudaMalloc(&d_data, 4 * sizeof(float));
#else
  hipMalloc(&d_data, 4 * sizeof(float));
#endif

  at::Tensor t = at::for_blob(d_data, {4})
                     .options(at::kFloat)
                     .target_device(at::Device(at::kCUDA, 0))
                     .make_tensor();
  ASSERT_TRUE(t.is_cuda());

#if defined(PADDLE_WITH_CUDA)
  cudaFree(d_data);
#else
  hipFree(d_data);
#endif
}

#endif  // PADDLE_WITH_CUDA || PADDLE_WITH_HIP
