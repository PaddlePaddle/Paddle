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

// Test for TensorBase::accessor()
TEST(TensorAccessorTest, AccessorBasic) {
  // Create a 2D tensor with known values
  at::Tensor tensor = at::arange(12, at::kFloat).reshape({3, 4});

  // Get accessor
  auto accessor = tensor.accessor<float, 2>();

  // Verify accessor dimensions
  ASSERT_EQ(accessor.size(0), 3);
  ASSERT_EQ(accessor.size(1), 4);

  // Verify accessor values
  float expected = 0.0f;
  for (int64_t i = 0; i < 3; ++i) {
    for (int64_t j = 0; j < 4; ++j) {
      ASSERT_EQ(accessor[i][j], expected);
      expected += 1.0f;
    }
  }
}

TEST(TensorAccessorTest, AccessorWithConstType) {
  // Create a tensor
  at::Tensor tensor = at::ones({2, 3}, at::kFloat);

  // Get const accessor
  auto accessor = tensor.accessor<const float, 2>();

  // Verify values are all ones
  for (int64_t i = 0; i < 2; ++i) {
    for (int64_t j = 0; j < 3; ++j) {
      ASSERT_EQ(accessor[i][j], 1.0f);
    }
  }
}

TEST(TensorAccessorTest, Accessor3D) {
  // Create a 3D tensor
  at::Tensor tensor = at::arange(24, at::kFloat).reshape({2, 3, 4});

  // Get accessor
  auto accessor = tensor.accessor<float, 3>();

  // Verify dimensions
  ASSERT_EQ(accessor.size(0), 2);
  ASSERT_EQ(accessor.size(1), 3);
  ASSERT_EQ(accessor.size(2), 4);

  // Verify a few values
  ASSERT_EQ(accessor[0][0][0], 0.0f);
  ASSERT_EQ(accessor[0][0][3], 3.0f);
  ASSERT_EQ(accessor[1][2][3], 23.0f);
}

TEST(TensorAccessorTest, AccessorModifyValues) {
  // Create a tensor
  at::Tensor tensor = at::zeros({2, 3}, at::kFloat);

  // Get mutable accessor
  auto accessor = tensor.accessor<float, 2>();

  // Modify values through accessor
  for (int64_t i = 0; i < 2; ++i) {
    for (int64_t j = 0; j < 3; ++j) {
      accessor[i][j] = static_cast<float>(i * 3 + j);
    }
  }

  // Verify modifications via data_ptr
  float* data = tensor.data_ptr<float>();
  for (int64_t i = 0; i < 6; ++i) {
    ASSERT_EQ(data[i], static_cast<float>(i));
  }
}

// Test for TensorBase::packed_accessor64()
TEST(TensorAccessorTest, PackedAccessor64Basic) {
  // Create a 2D tensor
  at::Tensor tensor = at::arange(12, at::kFloat).reshape({3, 4});

  // Get packed accessor with int64_t index type
  auto packed = tensor.packed_accessor64<float, 2>();

  // Verify dimensions
  ASSERT_EQ(packed.size(0), 3);
  ASSERT_EQ(packed.size(1), 4);

  // Verify strides
  ASSERT_EQ(packed.stride(0), 4);
  ASSERT_EQ(packed.stride(1), 1);

  // Verify values
  float expected = 0.0f;
  for (int64_t i = 0; i < 3; ++i) {
    for (int64_t j = 0; j < 4; ++j) {
      ASSERT_EQ(packed[i][j], expected);
      expected += 1.0f;
    }
  }
}

// Test for TensorBase::packed_accessor32()
TEST(TensorAccessorTest, PackedAccessor32Basic) {
  // Create a small 2D tensor (within int32_t range)
  at::Tensor tensor = at::arange(6, at::kFloat).reshape({2, 3});

  // Get packed accessor with int32_t index type
  auto packed = tensor.packed_accessor32<float, 2>();

  // Verify dimensions
  ASSERT_EQ(packed.size(0), 2);
  ASSERT_EQ(packed.size(1), 3);

  // Verify strides
  ASSERT_EQ(packed.stride(0), 3);
  ASSERT_EQ(packed.stride(1), 1);

  // Verify values
  ASSERT_EQ(packed[0][0], 0.0f);
  ASSERT_EQ(packed[0][2], 2.0f);
  ASSERT_EQ(packed[1][0], 3.0f);
  ASSERT_EQ(packed[1][2], 5.0f);
}

// Test for TensorBase::generic_packed_accessor()
TEST(TensorAccessorTest, GenericPackedAccessor) {
  // Create a 3D tensor
  at::Tensor tensor = at::arange(24, at::kDouble).reshape({2, 3, 4});

  // Get generic packed accessor with default template parameters
  auto packed = tensor.generic_packed_accessor<double, 3>();

  // Verify dimensions
  ASSERT_EQ(packed.size(0), 2);
  ASSERT_EQ(packed.size(1), 3);
  ASSERT_EQ(packed.size(2), 4);

  // Verify strides
  ASSERT_EQ(packed.stride(0), 12);  // 3*4
  ASSERT_EQ(packed.stride(1), 4);
  ASSERT_EQ(packed.stride(2), 1);

  // Verify corner values
  ASSERT_DOUBLE_EQ(packed[0][0][0], 0.0);
  ASSERT_DOUBLE_EQ(packed[1][2][3], 23.0);
}

TEST(TensorAccessorTest, PackedAccessorWithIntType) {
  // Test with integer tensor
  at::Tensor tensor = at::arange(10, at::kInt).reshape({2, 5});

  auto packed = tensor.packed_accessor64<int, 2>();

  ASSERT_EQ(packed.size(0), 2);
  ASSERT_EQ(packed.size(1), 5);

  int expected = 0;
  for (int64_t i = 0; i < 2; ++i) {
    for (int64_t j = 0; j < 5; ++j) {
      ASSERT_EQ(packed[i][j], expected);
      expected++;
    }
  }
}

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
TEST(TensorAccessorTest, PackedAccessorCUDA) {
  if (torch::cuda::is_available()) {
    // Create CUDA tensor
    at::Tensor tensor =
        at::arange(12, at::TensorOptions().dtype(at::kFloat).device(at::kCUDA))
            .reshape({3, 4});

    // Get packed accessor (typically used to pass to CUDA kernels)
    auto packed = tensor.packed_accessor64<float, 2>();

    // Verify dimensions
    ASSERT_EQ(packed.size(0), 3);
    ASSERT_EQ(packed.size(1), 4);

    // Verify strides
    ASSERT_EQ(packed.stride(0), 4);
    ASSERT_EQ(packed.stride(1), 1);
  }
}
#endif
