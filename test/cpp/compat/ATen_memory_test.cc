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

// ==================== is_pinned tests ====================

// Test is_pinned for CPU tensor (should be false)
TEST(IsPinnedTest, CPUTensorNotPinned) {
  auto tensor = at::arange(10, at::TensorOptions().dtype(at::kFloat));

  EXPECT_FALSE(tensor.is_pinned());
}

// Test is_pinned for empty tensor
TEST(IsPinnedTest, EmptyTensorNotPinned) {
  auto tensor = at::empty({0}, at::TensorOptions().dtype(at::kFloat));

  EXPECT_FALSE(tensor.is_pinned());
}

// Test is_pinned for multi-dimensional tensor
TEST(IsPinnedTest, MultiDimTensorNotPinned) {
  auto tensor = at::empty({2, 3, 4}, at::TensorOptions().dtype(at::kFloat));

  EXPECT_FALSE(tensor.is_pinned());
}

// ==================== reciprocal tests ====================

// Test reciprocal for simple values
TEST(ReciprocalTest, ReciprocalSimple) {
  auto tensor = at::empty({4}, at::TensorOptions().dtype(at::kFloat));
  tensor.data_ptr<float>()[0] = 1.0f;
  tensor.data_ptr<float>()[1] = 2.0f;
  tensor.data_ptr<float>()[2] = 4.0f;
  tensor.data_ptr<float>()[3] = 0.5f;

  auto result = tensor.reciprocal();

  // Check that original tensor is unchanged
  EXPECT_FLOAT_EQ(tensor.data_ptr<float>()[0], 1.0f);
  EXPECT_FLOAT_EQ(tensor.data_ptr<float>()[1], 2.0f);

  // Check reciprocal values: 1/1=1, 1/2=0.5, 1/4=0.25, 1/0.5=2
  EXPECT_FLOAT_EQ(result.data_ptr<float>()[0], 1.0f);
  EXPECT_FLOAT_EQ(result.data_ptr<float>()[1], 0.5f);
  EXPECT_FLOAT_EQ(result.data_ptr<float>()[2], 0.25f);
  EXPECT_FLOAT_EQ(result.data_ptr<float>()[3], 2.0f);
}

// Test reciprocal for 2D tensor
TEST(ReciprocalTest, Reciprocal2D) {
  auto tensor = at::empty({2, 2}, at::TensorOptions().dtype(at::kFloat));
  tensor.data_ptr<float>()[0] = 1.0f;
  tensor.data_ptr<float>()[1] = 2.0f;
  tensor.data_ptr<float>()[2] = 5.0f;
  tensor.data_ptr<float>()[3] = 10.0f;

  auto result = tensor.reciprocal();

  EXPECT_EQ(result.dim(), 2);
  EXPECT_EQ(result.size(0), 2);
  EXPECT_EQ(result.size(1), 2);

  EXPECT_FLOAT_EQ(result.data_ptr<float>()[0], 1.0f);
  EXPECT_FLOAT_EQ(result.data_ptr<float>()[1], 0.5f);
  EXPECT_FLOAT_EQ(result.data_ptr<float>()[2], 0.2f);
  EXPECT_FLOAT_EQ(result.data_ptr<float>()[3], 0.1f);
}

// Test reciprocal with double dtype
TEST(ReciprocalTest, ReciprocalDouble) {
  auto tensor = at::empty({3}, at::TensorOptions().dtype(at::kDouble));
  tensor.data_ptr<double>()[0] = 1.0;
  tensor.data_ptr<double>()[1] = 3.0;
  tensor.data_ptr<double>()[2] = 8.0;

  auto result = tensor.reciprocal();

  EXPECT_DOUBLE_EQ(result.data_ptr<double>()[0], 1.0);
  EXPECT_NEAR(result.data_ptr<double>()[1], 1.0 / 3.0, 1e-10);
  EXPECT_DOUBLE_EQ(result.data_ptr<double>()[2], 0.125);
}

// Test reciprocal preserves dtype
TEST(ReciprocalTest, ReciprocalPreservesDtype) {
  auto tensor_float = at::empty({2}, at::TensorOptions().dtype(at::kFloat));
  tensor_float.fill_(2.0f);

  auto tensor_double = at::empty({2}, at::TensorOptions().dtype(at::kDouble));
  tensor_double.fill_(2.0);

  auto result_float = tensor_float.reciprocal();
  auto result_double = tensor_double.reciprocal();

  EXPECT_EQ(result_float.dtype(), at::kFloat);
  EXPECT_EQ(result_double.dtype(), at::kDouble);
}

// ==================== reciprocal_ (in-place) tests ====================

// Test reciprocal_ modifies tensor in-place
TEST(ReciprocalInplaceTest, ReciprocalInplaceSimple) {
  auto tensor = at::empty({4}, at::TensorOptions().dtype(at::kFloat));
  tensor.data_ptr<float>()[0] = 1.0f;
  tensor.data_ptr<float>()[1] = 2.0f;
  tensor.data_ptr<float>()[2] = 4.0f;
  tensor.data_ptr<float>()[3] = 0.5f;

  void* original_ptr = tensor.data_ptr();

  auto result = tensor.reciprocal_();

  // Should return reference to same tensor
  EXPECT_EQ(result.data_ptr(), original_ptr);

  // Check in-place modification: 1/1=1, 1/2=0.5, 1/4=0.25, 1/0.5=2
  EXPECT_FLOAT_EQ(tensor.data_ptr<float>()[0], 1.0f);
  EXPECT_FLOAT_EQ(tensor.data_ptr<float>()[1], 0.5f);
  EXPECT_FLOAT_EQ(tensor.data_ptr<float>()[2], 0.25f);
  EXPECT_FLOAT_EQ(tensor.data_ptr<float>()[3], 2.0f);
}

// Test reciprocal_ on 2D tensor
TEST(ReciprocalInplaceTest, ReciprocalInplace2D) {
  auto tensor = at::empty({2, 3}, at::TensorOptions().dtype(at::kFloat));
  for (int i = 0; i < 6; ++i) {
    tensor.data_ptr<float>()[i] =
        static_cast<float>(i + 1);  // [1, 2, 3, 4, 5, 6]
  }

  tensor.reciprocal_();

  EXPECT_FLOAT_EQ(tensor.data_ptr<float>()[0], 1.0f);           // 1/1
  EXPECT_FLOAT_EQ(tensor.data_ptr<float>()[1], 0.5f);           // 1/2
  EXPECT_NEAR(tensor.data_ptr<float>()[2], 1.0f / 3.0f, 1e-6);  // 1/3
  EXPECT_FLOAT_EQ(tensor.data_ptr<float>()[3], 0.25f);          // 1/4
  EXPECT_FLOAT_EQ(tensor.data_ptr<float>()[4], 0.2f);           // 1/5
  EXPECT_NEAR(tensor.data_ptr<float>()[5], 1.0f / 6.0f, 1e-6);  // 1/6
}

// Test chaining reciprocal_ twice returns original values
TEST(ReciprocalInplaceTest, ReciprocalInplaceTwice) {
  auto tensor = at::empty({3}, at::TensorOptions().dtype(at::kFloat));
  tensor.data_ptr<float>()[0] = 2.0f;
  tensor.data_ptr<float>()[1] = 4.0f;
  tensor.data_ptr<float>()[2] = 8.0f;

  tensor.reciprocal_().reciprocal_();

  // Should return to original values
  EXPECT_FLOAT_EQ(tensor.data_ptr<float>()[0], 2.0f);
  EXPECT_FLOAT_EQ(tensor.data_ptr<float>()[1], 4.0f);
  EXPECT_FLOAT_EQ(tensor.data_ptr<float>()[2], 8.0f);
}

// ==================== detach tests ====================

// Test detach creates a new tensor sharing data
TEST(DetachTest, DetachSharesData) {
  auto tensor = at::arange(5, at::TensorOptions().dtype(at::kFloat));

  auto detached = tensor.detach();

  // Should have same shape and dtype
  EXPECT_EQ(detached.dim(), tensor.dim());
  EXPECT_EQ(detached.size(0), tensor.size(0));
  EXPECT_EQ(detached.dtype(), tensor.dtype());

  // Should have same values
  for (int i = 0; i < 5; ++i) {
    EXPECT_FLOAT_EQ(detached.data_ptr<float>()[i], tensor.data_ptr<float>()[i]);
  }
}

// Test detach on 2D tensor
TEST(DetachTest, Detach2D) {
  auto tensor =
      at::arange(12, at::TensorOptions().dtype(at::kFloat)).reshape({3, 4});

  auto detached = tensor.detach();

  EXPECT_EQ(detached.dim(), 2);
  EXPECT_EQ(detached.size(0), 3);
  EXPECT_EQ(detached.size(1), 4);
  EXPECT_EQ(detached.numel(), 12);
}

// Test detach preserves device
TEST(DetachTest, DetachPreservesDevice) {
  auto tensor = at::arange(5, at::TensorOptions().dtype(at::kFloat));

  auto detached = tensor.detach();

  EXPECT_TRUE(tensor.is_cpu());
  EXPECT_TRUE(detached.is_cpu());
}

// Test detach with different dtypes
TEST(DetachTest, DetachDifferentDtypes) {
  auto tensor_float = at::arange(5, at::TensorOptions().dtype(at::kFloat));
  auto tensor_int = at::arange(5, at::TensorOptions().dtype(at::kInt));
  auto tensor_double = at::arange(5, at::TensorOptions().dtype(at::kDouble));

  auto detached_float = tensor_float.detach();
  auto detached_int = tensor_int.detach();
  auto detached_double = tensor_double.detach();

  EXPECT_EQ(detached_float.dtype(), at::kFloat);
  EXPECT_EQ(detached_int.dtype(), at::kInt);
  EXPECT_EQ(detached_double.dtype(), at::kDouble);
}

// Test multiple detach calls
TEST(DetachTest, DetachMultipleTimes) {
  auto tensor = at::arange(5, at::TensorOptions().dtype(at::kFloat));

  auto detached1 = tensor.detach();
  auto detached2 = detached1.detach();

  EXPECT_EQ(detached2.numel(), 5);
  EXPECT_EQ(detached2.dtype(), at::kFloat);
}

// ==================== detach_ (in-place) tests ====================

// Test detach_ returns reference to self
TEST(DetachInplaceTest, DetachInplaceReturnsSelf) {
  auto tensor = at::arange(5, at::TensorOptions().dtype(at::kFloat));
  void* original_ptr = tensor.data_ptr();

  auto result = tensor.detach_();

  // Should return reference to same tensor
  EXPECT_EQ(result.data_ptr(), original_ptr);
}

// Test detach_ preserves data
TEST(DetachInplaceTest, DetachInplacePreservesData) {
  auto tensor = at::arange(5, at::TensorOptions().dtype(at::kFloat));

  tensor.detach_();

  // Data should be unchanged
  for (int i = 0; i < 5; ++i) {
    EXPECT_FLOAT_EQ(tensor.data_ptr<float>()[i], static_cast<float>(i));
  }
}

// Test detach_ preserves shape
TEST(DetachInplaceTest, DetachInplacePreservesShape) {
  auto tensor =
      at::arange(12, at::TensorOptions().dtype(at::kFloat)).reshape({3, 4});

  tensor.detach_();

  EXPECT_EQ(tensor.dim(), 2);
  EXPECT_EQ(tensor.size(0), 3);
  EXPECT_EQ(tensor.size(1), 4);
}

// Test detach_ preserves dtype
TEST(DetachInplaceTest, DetachInplacePreservesDtype) {
  auto tensor_float = at::empty({5}, at::TensorOptions().dtype(at::kFloat));
  auto tensor_double = at::empty({5}, at::TensorOptions().dtype(at::kDouble));

  tensor_float.detach_();
  tensor_double.detach_();

  EXPECT_EQ(tensor_float.dtype(), at::kFloat);
  EXPECT_EQ(tensor_double.dtype(), at::kDouble);
}

// Test chaining detach_ calls
TEST(DetachInplaceTest, DetachInplaceChained) {
  auto tensor = at::arange(5, at::TensorOptions().dtype(at::kFloat));

  tensor.detach_().detach_();

  // Should still have valid data
  EXPECT_EQ(tensor.numel(), 5);
  EXPECT_FLOAT_EQ(tensor.data_ptr<float>()[0], 0.0f);
  EXPECT_FLOAT_EQ(tensor.data_ptr<float>()[4], 4.0f);
}

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
// Test reciprocal on CUDA
TEST(ReciprocalTest, ReciprocalCUDA) {
  auto tensor =
      at::empty({4}, at::TensorOptions().dtype(at::kFloat).device(at::kCUDA));
  auto cpu_tensor = at::empty({4}, at::TensorOptions().dtype(at::kFloat));
  cpu_tensor.data_ptr<float>()[0] = 1.0f;
  cpu_tensor.data_ptr<float>()[1] = 2.0f;
  cpu_tensor.data_ptr<float>()[2] = 4.0f;
  cpu_tensor.data_ptr<float>()[3] = 0.5f;
  tensor.copy_(cpu_tensor);

  auto result = tensor.reciprocal();

  EXPECT_TRUE(result.is_cuda());

  auto cpu_result = result.cpu();
  EXPECT_FLOAT_EQ(cpu_result.data_ptr<float>()[0], 1.0f);
  EXPECT_FLOAT_EQ(cpu_result.data_ptr<float>()[1], 0.5f);
  EXPECT_FLOAT_EQ(cpu_result.data_ptr<float>()[2], 0.25f);
  EXPECT_FLOAT_EQ(cpu_result.data_ptr<float>()[3], 2.0f);
}

// Test detach on CUDA
TEST(DetachTest, DetachCUDA) {
  auto tensor =
      at::arange(5, at::TensorOptions().dtype(at::kFloat).device(at::kCUDA));

  auto detached = tensor.detach();

  EXPECT_TRUE(detached.is_cuda());
  EXPECT_EQ(detached.numel(), 5);
}
#endif
