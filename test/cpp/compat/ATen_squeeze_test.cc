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

TEST(TestSqueeze, SqueezeNoArg) {
  // Test squeeze() without arguments - removes all dimensions of size 1
  at::Tensor tensor = at::ones({1, 3, 1, 4, 1}, at::kFloat);
  at::Tensor squeezed = tensor.squeeze();

  ASSERT_EQ(squeezed.sizes(), c10::IntArrayRef({3, 4}));
  ASSERT_EQ(squeezed.numel(), tensor.numel());
}

TEST(TestSqueeze, SqueezeSingleDim) {
  // Test squeeze(int64_t dim) - removes specific dimension if size is 1
  at::Tensor tensor = at::ones({1, 3, 1, 4}, at::kFloat);

  // Squeeze dimension 0 (size 1)
  at::Tensor squeezed_0 = tensor.squeeze(0);
  ASSERT_EQ(squeezed_0.sizes(), c10::IntArrayRef({3, 1, 4}));

  // Squeeze dimension 2 (size 1)
  at::Tensor squeezed_2 = tensor.squeeze(2);
  ASSERT_EQ(squeezed_2.sizes(), c10::IntArrayRef({1, 3, 4}));

  // Squeeze dimension 1 (size 3, should not change)
  at::Tensor squeezed_1 = tensor.squeeze(1);
  ASSERT_EQ(squeezed_1.sizes(), c10::IntArrayRef({1, 3, 1, 4}));
}

TEST(TestSqueeze, SqueezeMultipleDims) {
  // Test squeeze(IntArrayRef dim) - removes specified dimensions if size is 1
  at::Tensor tensor = at::ones({1, 3, 1, 4, 1}, at::kFloat);

  // Squeeze dimensions 0 and 2
  at::Tensor squeezed = tensor.squeeze(c10::IntArrayRef({0, 2}));
  ASSERT_EQ(squeezed.sizes(), c10::IntArrayRef({3, 4, 1}));

  // Squeeze all size-1 dimensions
  at::Tensor squeezed_all = tensor.squeeze(c10::IntArrayRef({0, 2, 4}));
  ASSERT_EQ(squeezed_all.sizes(), c10::IntArrayRef({3, 4}));
}

TEST(TestSqueeze, SqueezeInplaceNoArg) {
  // Test squeeze_() without arguments - in-place operation
  at::Tensor tensor = at::ones({1, 3, 1, 4, 1}, at::kFloat);
  void* original_ptr = tensor.data_ptr();

  tensor.squeeze_();

  ASSERT_EQ(tensor.sizes(), c10::IntArrayRef({3, 4}));
  ASSERT_EQ(tensor.data_ptr(), original_ptr);  // Same data pointer
}

TEST(TestSqueeze, SqueezeInplaceSingleDim) {
  // Test squeeze_(int64_t dim) - in-place operation on specific dimension
  at::Tensor tensor = at::ones({1, 3, 1, 4}, at::kFloat);
  void* original_ptr = tensor.data_ptr();

  tensor.squeeze_(2);

  ASSERT_EQ(tensor.sizes(), c10::IntArrayRef({1, 3, 4}));
  ASSERT_EQ(tensor.data_ptr(), original_ptr);
}

TEST(TestSqueeze, SqueezeInplaceMultipleDims) {
  // Test squeeze_(IntArrayRef dim) - in-place operation on multiple dimensions
  at::Tensor tensor = at::ones({1, 3, 1, 4, 1}, at::kFloat);
  void* original_ptr = tensor.data_ptr();

  tensor.squeeze_(c10::IntArrayRef({0, 2, 4}));

  ASSERT_EQ(tensor.sizes(), c10::IntArrayRef({3, 4}));
  ASSERT_EQ(tensor.data_ptr(), original_ptr);
}

TEST(TestUnsqueeze, UnsqueezeNoArg) {
  // Test unsqueeze() without arguments
  at::Tensor tensor = at::ones({3, 4}, at::kFloat);
  at::Tensor unsqueezed = tensor.unsqueeze();

  // The behavior depends on the implementation
  ASSERT_EQ(unsqueezed.numel(), tensor.numel());
}

TEST(TestUnsqueeze, UnsqueezeSingleDim) {
  // Test unsqueeze(int64_t dim) - adds dimension at specified position
  at::Tensor tensor = at::ones({3, 4}, at::kFloat);

  // Unsqueeze at dimension 0
  at::Tensor unsqueezed_0 = tensor.unsqueeze(0);
  ASSERT_EQ(unsqueezed_0.sizes(), c10::IntArrayRef({1, 3, 4}));

  // Unsqueeze at dimension 1
  at::Tensor unsqueezed_1 = tensor.unsqueeze(1);
  ASSERT_EQ(unsqueezed_1.sizes(), c10::IntArrayRef({3, 1, 4}));

  // Unsqueeze at dimension 2
  at::Tensor unsqueezed_2 = tensor.unsqueeze(2);
  ASSERT_EQ(unsqueezed_2.sizes(), c10::IntArrayRef({3, 4, 1}));

  // Unsqueeze at negative dimension
  at::Tensor unsqueezed_neg = tensor.unsqueeze(-1);
  ASSERT_EQ(unsqueezed_neg.sizes(), c10::IntArrayRef({3, 4, 1}));
}

TEST(TestUnsqueeze, UnsqueezeMultipleDims) {
  // Test unsqueeze(IntArrayRef dim) - adds dimensions at specified positions
  at::Tensor tensor = at::ones({3, 4}, at::kFloat);

  // Unsqueeze at dimensions 0 and 2
  at::Tensor unsqueezed = tensor.unsqueeze(c10::IntArrayRef({0, 2}));
  ASSERT_EQ(unsqueezed.numel(), tensor.numel());
}

TEST(TestUnsqueeze, UnsqueezeInplaceNoArg) {
  // Test unsqueeze_() without arguments - in-place operation
  at::Tensor tensor = at::ones({3, 4}, at::kFloat);
  void* original_ptr = tensor.data_ptr();

  tensor.unsqueeze_();

  ASSERT_EQ(tensor.numel(), 12);  // 3 * 4
  ASSERT_EQ(tensor.data_ptr(), original_ptr);
}

TEST(TestUnsqueeze, UnsqueezeInplaceSingleDim) {
  // Test unsqueeze_(int64_t dim) - in-place operation
  at::Tensor tensor = at::ones({3, 4}, at::kFloat);
  void* original_ptr = tensor.data_ptr();

  tensor.unsqueeze_(1);

  ASSERT_EQ(tensor.sizes(), c10::IntArrayRef({3, 1, 4}));
  ASSERT_EQ(tensor.data_ptr(), original_ptr);
}

TEST(TestUnsqueeze, UnsqueezeInplaceMultipleDims) {
  // Test unsqueeze_(IntArrayRef dim) - in-place operation on multiple
  // dimensions
  at::Tensor tensor = at::ones({3, 4}, at::kFloat);
  void* original_ptr = tensor.data_ptr();

  tensor.unsqueeze_(c10::IntArrayRef({0, 2}));

  ASSERT_EQ(tensor.numel(), 12);
  ASSERT_EQ(tensor.data_ptr(), original_ptr);
}

TEST(TestSqueezeUnsqueeze, CombinedOperations) {
  // Test combining squeeze and unsqueeze operations
  at::Tensor tensor = at::ones({2, 3, 4}, at::kFloat);

  // Add a dimension then remove it
  at::Tensor unsqueezed = tensor.unsqueeze(1);
  ASSERT_EQ(unsqueezed.sizes(), c10::IntArrayRef({2, 1, 3, 4}));

  at::Tensor squeezed = unsqueezed.squeeze(1);
  ASSERT_EQ(squeezed.sizes(), c10::IntArrayRef({2, 3, 4}));

  // Verify data integrity
  ASSERT_EQ(tensor.numel(), squeezed.numel());
}

TEST(TestSqueezeUnsqueeze, DataIntegrity) {
  // Test that squeeze and unsqueeze preserve data
  at::Tensor tensor = at::arange(24, at::kFloat).reshape({2, 3, 4});

  // Unsqueeze and squeeze
  at::Tensor unsqueezed = tensor.unsqueeze(0);
  at::Tensor squeezed = unsqueezed.squeeze(0);

  // Check data is preserved
  const float* original_data = tensor.data_ptr<float>();
  const float* result_data = squeezed.data_ptr<float>();

  for (int64_t i = 0; i < tensor.numel(); ++i) {
    ASSERT_EQ(original_data[i], result_data[i]);
  }
}
