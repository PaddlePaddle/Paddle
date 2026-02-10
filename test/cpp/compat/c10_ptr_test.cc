// Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

TEST(TensorBaseTest, IsSameAPI) {
  // Test is_same() API - checks if two tensors share the same underlying data
  at::TensorBase tensor1 = at::ones({2, 3}, at::kFloat);
  at::TensorBase tensor2 = tensor1;                       // Same tensor
  at::TensorBase tensor3 = at::ones({2, 3}, at::kFloat);  // Different tensor

  // tensor1 and tensor2 should point to the same underlying tensor
  ASSERT_TRUE(tensor1.is_same(tensor2));
  ASSERT_TRUE(tensor2.is_same(tensor1));

  // tensor1 and tensor3 should be different tensors
  ASSERT_FALSE(tensor1.is_same(tensor3));
  ASSERT_FALSE(tensor3.is_same(tensor1));

  // A tensor should be the same as itself
  ASSERT_TRUE(tensor1.is_same(tensor1));

  // Test with view - in Paddle, view creates a new tensor implementation
  // even though they may share underlying data storage
  at::TensorBase view_tensor = tensor1.view({6});
  // View tensor has different impl pointer in Paddle
  ASSERT_FALSE(tensor1.is_same(view_tensor));

  // Test with undefined tensors
  at::TensorBase undefined1;
  at::TensorBase undefined2;
  ASSERT_TRUE(undefined1.is_same(undefined2));  // Both undefined
}

TEST(TensorBaseTest, UseCountAPI) {
  // Test use_count() API - returns reference count of underlying tensor
  at::TensorBase tensor1 = at::ones({2, 3}, at::kFloat);

  // Initial reference count should be 1
  ASSERT_EQ(tensor1.use_count(), 1);

  // Create a copy - reference count should increase
  at::TensorBase tensor2 = tensor1;
  ASSERT_EQ(tensor1.use_count(), 2);
  ASSERT_EQ(tensor2.use_count(), 2);

  // Create another copy
  at::TensorBase tensor3 = tensor1;
  ASSERT_EQ(tensor1.use_count(), 3);
  ASSERT_EQ(tensor2.use_count(), 3);
  ASSERT_EQ(tensor3.use_count(), 3);

  // Reset one copy - reference count should decrease
  tensor2.reset();
  ASSERT_EQ(tensor1.use_count(), 2);
  ASSERT_EQ(tensor3.use_count(), 2);

  // Reset another copy
  tensor3.reset();
  ASSERT_EQ(tensor1.use_count(), 1);

  // Test with view - in Paddle, view creates a new tensor with separate impl
  // So the use_count remains 1 for each
  {
    at::TensorBase view_tensor = tensor1.view({6});
    // Each tensor has its own impl, so use_count is 1 for each
    ASSERT_EQ(tensor1.use_count(), 1);
    ASSERT_EQ(view_tensor.use_count(), 1);
  }
  // After view goes out of scope
  ASSERT_EQ(tensor1.use_count(), 1);
}

TEST(TensorBaseTest, WeakUseCountAPI) {
  // Test weak_use_count() API
  // Note: Currently returns 0 as Paddle uses std::shared_ptr instead of
  // c10::intrusive_ptr
  at::TensorBase tensor1 = at::ones({2, 3}, at::kFloat);

  // Should return 0 (not implemented yet)
  ASSERT_EQ(tensor1.weak_use_count(), 0);

  at::TensorBase tensor2 = tensor1;
  ASSERT_EQ(tensor1.weak_use_count(), 0);
  ASSERT_EQ(tensor2.weak_use_count(), 0);

  // Test with undefined tensor
  at::TensorBase undefined_tensor;
  ASSERT_EQ(undefined_tensor.weak_use_count(), 0);
}
