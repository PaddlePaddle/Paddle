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

TEST(TensorBaseTest, IsContiguousOrFalseAPI) {
  // Test with regular contiguous tensor
  at::Tensor contiguous_tensor = at::ones({2, 3, 4}, at::kFloat);
  ASSERT_TRUE(contiguous_tensor.is_contiguous_or_false());
  ASSERT_EQ(contiguous_tensor.is_contiguous_or_false(),
            contiguous_tensor.is_contiguous());

  // Test with view tensor (should still be contiguous if strides allow)
  at::TensorBase view_tensor = contiguous_tensor.view({6, 4});
  ASSERT_TRUE(view_tensor.is_contiguous_or_false());
  ASSERT_EQ(view_tensor.is_contiguous_or_false(), view_tensor.is_contiguous());

  // Test with different shapes
  at::TensorBase flat_tensor = at::ones({24}, at::kFloat);
  ASSERT_TRUE(flat_tensor.is_contiguous_or_false());

  at::TensorBase multi_dim_tensor = at::ones({2, 3, 4, 5}, at::kFloat);
  ASSERT_TRUE(multi_dim_tensor.is_contiguous_or_false());

  // Test with contiguous() method
  at::TensorBase made_contiguous = contiguous_tensor.contiguous();
  ASSERT_TRUE(made_contiguous.is_contiguous_or_false());

  // Test consistency between is_contiguous() and is_contiguous_or_false()
  // They should return the same value for all cases
  at::TensorBase tensor1 = at::ones({5, 6}, at::kDouble);
  ASSERT_EQ(tensor1.is_contiguous(), tensor1.is_contiguous_or_false());

  at::TensorBase tensor2 = at::ones({3, 4, 5}, at::kInt);
  ASSERT_EQ(tensor2.is_contiguous(), tensor2.is_contiguous_or_false());

  // Test with different dtypes
  at::TensorBase bool_tensor = at::ones({2, 3}, at::kBool);
  ASSERT_TRUE(bool_tensor.is_contiguous_or_false());

  at::TensorBase long_tensor = at::ones({2, 3}, at::kLong);
  ASSERT_TRUE(long_tensor.is_contiguous_or_false());
}
