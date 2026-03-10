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
#include <ATen/ops/detach.h>
#include <ATen/ops/reciprocal.h>

#include "ATen/ATen.h"
#include "gtest/gtest.h"
#include "paddle/phi/common/float16.h"
#include "torch/all.h"

// Test detach member function: tensor.detach()
TEST(TestDetach, MemberFunction) {
  at::Tensor tensor = at::ones({2, 3}, at::kFloat);

  // Detach creates a new tensor that shares data but has no autograd history
  at::Tensor detached = tensor.detach();

  ASSERT_EQ(detached.sizes(), tensor.sizes());
  ASSERT_EQ(detached.numel(), tensor.numel());
  ASSERT_EQ(detached.dtype(), tensor.dtype());

  // Both tensors should share the same data
  float* original_ptr = tensor.data_ptr<float>();
  float* detached_ptr = detached.data_ptr<float>();
  ASSERT_EQ(original_ptr, detached_ptr);
}

// Test detach free function: at::detach(tensor)
TEST(TestDetach, FreeFunction) {
  at::Tensor tensor = at::ones({3, 4}, at::kFloat);
  at::Tensor detached = at::detach(tensor);

  ASSERT_EQ(detached.sizes(), tensor.sizes());
  ASSERT_EQ(detached.numel(), tensor.numel());

  // Verify data is shared
  float* original_ptr = tensor.data_ptr<float>();
  float* detached_ptr = detached.data_ptr<float>();
  ASSERT_EQ(original_ptr, detached_ptr);
}

// Test that both methods produce identical results (shared implementation)
TEST(TestDetach, SharedImplementation) {
  at::Tensor tensor = at::ones({2, 3, 4}, at::kFloat);

  // Call both detach methods
  at::Tensor detached_member = tensor.detach();
  at::Tensor detached_free = at::detach(tensor);

  // Both should have the same properties
  ASSERT_EQ(detached_member.sizes(), detached_free.sizes());
  ASSERT_EQ(detached_member.numel(), detached_free.numel());
  ASSERT_EQ(detached_member.dtype(), detached_free.dtype());

  // All three should share the same data
  float* original_ptr = tensor.data_ptr<float>();
  float* member_ptr = detached_member.data_ptr<float>();
  float* free_ptr = detached_free.data_ptr<float>();

  ASSERT_EQ(original_ptr, member_ptr);
  ASSERT_EQ(original_ptr, free_ptr);
}

// Test detach_ in-place member function: tensor.detach_()
TEST(TestDetach, InplaceMemberFunction) {
  at::Tensor tensor = at::ones({2, 3}, at::kFloat);
  void* original_ptr = tensor.data_ptr();

  // detach_() modifies the tensor in-place
  at::Tensor& result = tensor.detach_();

  // Should return reference to the same tensor
  ASSERT_EQ(&result, &tensor);
  ASSERT_EQ(result.data_ptr(), original_ptr);
  ASSERT_EQ(result.sizes(), c10::IntArrayRef({2, 3}));
}

// Test detach preserves data values
TEST(TestDetach, PreservesData) {
  at::Tensor tensor = at::ones({2, 3}, at::kFloat);
  float* data = tensor.data_ptr<float>();
  data[0] = 1.0f;
  data[1] = 2.0f;
  data[2] = 3.0f;
  data[3] = 4.0f;
  data[4] = 5.0f;
  data[5] = 6.0f;

  at::Tensor detached = tensor.detach();

  // Verify data is preserved
  float* detached_data = detached.data_ptr<float>();
  ASSERT_EQ(detached_data[0], 1.0f);
  ASSERT_EQ(detached_data[1], 2.0f);
  ASSERT_EQ(detached_data[2], 3.0f);
  ASSERT_EQ(detached_data[3], 4.0f);
  ASSERT_EQ(detached_data[4], 5.0f);
  ASSERT_EQ(detached_data[5], 6.0f);
}

// Test detach with different dtypes
TEST(TestDetach, DifferentDtypes) {
  // Float32
  at::Tensor float_tensor = at::ones({2, 3}, at::kFloat);
  at::Tensor float_detached = float_tensor.detach();
  ASSERT_EQ(float_detached.dtype(), at::kFloat);
  ASSERT_EQ(float_detached.sizes(), float_tensor.sizes());

  // Float64
  at::Tensor double_tensor = at::ones({2, 3}, at::kDouble);
  at::Tensor double_detached = double_tensor.detach();
  ASSERT_EQ(double_detached.dtype(), at::kDouble);
  ASSERT_EQ(double_detached.sizes(), double_tensor.sizes());

  // Int32
  at::Tensor int_tensor = at::ones({2, 3}, at::kInt);
  at::Tensor int_detached = int_tensor.detach();
  ASSERT_EQ(int_detached.dtype(), at::kInt);
  ASSERT_EQ(int_detached.sizes(), int_tensor.sizes());

  // Int64
  at::Tensor long_tensor = at::ones({2, 3}, at::kLong);
  at::Tensor long_detached = long_tensor.detach();
  ASSERT_EQ(long_detached.dtype(), at::kLong);
  ASSERT_EQ(long_detached.sizes(), long_tensor.sizes());
}

// Test detach with various shapes
TEST(TestDetach, VariousShapes) {
  // 1D tensor
  at::Tensor tensor_1d = at::ones({10}, at::kFloat);
  at::Tensor detached_1d = tensor_1d.detach();
  ASSERT_EQ(detached_1d.sizes(), c10::IntArrayRef({10}));

  // 2D tensor
  at::Tensor tensor_2d = at::ones({3, 4}, at::kFloat);
  at::Tensor detached_2d = tensor_2d.detach();
  ASSERT_EQ(detached_2d.sizes(), c10::IntArrayRef({3, 4}));

  // 3D tensor
  at::Tensor tensor_3d = at::ones({2, 3, 4}, at::kFloat);
  at::Tensor detached_3d = tensor_3d.detach();
  ASSERT_EQ(detached_3d.sizes(), c10::IntArrayRef({2, 3, 4}));

  // 4D tensor
  at::Tensor tensor_4d = at::ones({2, 3, 4, 5}, at::kFloat);
  at::Tensor detached_4d = tensor_4d.detach();
  ASSERT_EQ(detached_4d.sizes(), c10::IntArrayRef({2, 3, 4, 5}));
}

// Test modifications affect both tensors (shared data)
TEST(TestDetach, SharedDataModification) {
  at::Tensor tensor = at::ones({2, 3}, at::kFloat);
  at::Tensor detached = tensor.detach();

  // Modify original tensor
  float* tensor_data = tensor.data_ptr<float>();
  tensor_data[0] = 99.0f;

  // Check that detached tensor sees the change
  float* detached_data = detached.data_ptr<float>();
  ASSERT_EQ(detached_data[0], 99.0f);

  // Modify detached tensor
  detached_data[1] = 88.0f;

  // Check that original tensor sees the change
  ASSERT_EQ(tensor_data[1], 88.0f);
}

// ============================================================================
// Reciprocal Tests
// ============================================================================

// Test reciprocal member function: tensor.reciprocal()
TEST(TestReciprocal, MemberFunction) {
  at::Tensor tensor = at::full({2, 3}, 2.0f, at::kFloat);
  at::Tensor result = tensor.reciprocal();

  ASSERT_EQ(result.sizes(), tensor.sizes());
  ASSERT_EQ(result.numel(), tensor.numel());

  // Verify reciprocal calculation: 1/2 = 0.5
  float* result_data = result.data_ptr<float>();
  for (int i = 0; i < result.numel(); i++) {
    ASSERT_NEAR(result_data[i], 0.5f, 1e-6);
  }
}

// Test reciprocal free function: at::reciprocal(tensor)
TEST(TestReciprocal, FreeFunction) {
  at::Tensor tensor = at::full({3, 4}, 4.0f, at::kFloat);
  at::Tensor result = at::reciprocal(tensor);

  ASSERT_EQ(result.sizes(), tensor.sizes());
  ASSERT_EQ(result.numel(), tensor.numel());

  // Verify reciprocal calculation: 1/4 = 0.25
  float* result_data = result.data_ptr<float>();
  for (int i = 0; i < result.numel(); i++) {
    ASSERT_NEAR(result_data[i], 0.25f, 1e-6);
  }
}

// Test that both methods produce identical results (shared implementation)
TEST(TestReciprocal, SharedImplementation) {
  at::Tensor tensor = at::full({2, 3, 4}, 5.0f, at::kFloat);

  // Call both reciprocal methods
  at::Tensor result_member = tensor.reciprocal();
  at::Tensor result_free = at::reciprocal(tensor);

  // Both should have the same shape and values
  ASSERT_EQ(result_member.sizes(), result_free.sizes());
  ASSERT_EQ(result_member.numel(), result_free.numel());

  // Verify both produce same values: 1/5 = 0.2
  float* member_data = result_member.data_ptr<float>();
  float* free_data = result_free.data_ptr<float>();
  for (int i = 0; i < result_member.numel(); i++) {
    ASSERT_NEAR(member_data[i], 0.2f, 1e-6);
    ASSERT_NEAR(free_data[i], 0.2f, 1e-6);
    ASSERT_EQ(member_data[i], free_data[i]);
  }
}

// Test reciprocal_ in-place member function: tensor.reciprocal_()
TEST(TestReciprocal, InplaceMemberFunction) {
  at::Tensor tensor = at::full({2, 3}, 2.0f, at::kFloat);
  void* original_ptr = tensor.data_ptr();

  // reciprocal_() modifies the tensor in-place
  at::Tensor& result = tensor.reciprocal_();

  // Should return reference to the same tensor
  ASSERT_EQ(&result, &tensor);
  ASSERT_EQ(result.data_ptr(), original_ptr);
  ASSERT_EQ(result.sizes(), c10::IntArrayRef({2, 3}));

  // Verify reciprocal calculation: 1/2 = 0.5
  float* result_data = result.data_ptr<float>();
  for (int i = 0; i < result.numel(); i++) {
    ASSERT_NEAR(result_data[i], 0.5f, 1e-6);
  }
}

// Test reciprocal with various input values
TEST(TestReciprocal, VariousValues) {
  at::Tensor tensor = at::ones({5}, at::kFloat);
  float* data = tensor.data_ptr<float>();
  data[0] = 1.0f;
  data[1] = 2.0f;
  data[2] = 4.0f;
  data[3] = 0.5f;
  data[4] = 10.0f;

  at::Tensor result = tensor.reciprocal();
  float* result_data = result.data_ptr<float>();

  // Verify reciprocals
  ASSERT_NEAR(result_data[0], 1.0f, 1e-6);   // 1/1 = 1
  ASSERT_NEAR(result_data[1], 0.5f, 1e-6);   // 1/2 = 0.5
  ASSERT_NEAR(result_data[2], 0.25f, 1e-6);  // 1/4 = 0.25
  ASSERT_NEAR(result_data[3], 2.0f, 1e-6);   // 1/0.5 = 2
  ASSERT_NEAR(result_data[4], 0.1f, 1e-6);   // 1/10 = 0.1
}

// Test reciprocal with different dtypes
TEST(TestReciprocal, DifferentDtypes) {
  // Float32
  at::Tensor float_tensor = at::full({2, 3}, 2.0f, at::kFloat);
  at::Tensor float_result = float_tensor.reciprocal();
  ASSERT_EQ(float_result.dtype(), at::kFloat);
  float* float_data = float_result.data_ptr<float>();
  ASSERT_NEAR(float_data[0], 0.5f, 1e-6);

  // Float64
  at::Tensor double_tensor = at::full({2, 3}, 2.0, at::kDouble);
  at::Tensor double_result = double_tensor.reciprocal();
  ASSERT_EQ(double_result.dtype(), at::kDouble);
  double* double_data = double_result.data_ptr<double>();
  ASSERT_NEAR(double_data[0], 0.5, 1e-10);
}

// Test reciprocal with various shapes
TEST(TestReciprocal, VariousShapes) {
  // 1D tensor
  at::Tensor tensor_1d = at::full({10}, 2.0f, at::kFloat);
  at::Tensor result_1d = tensor_1d.reciprocal();
  ASSERT_EQ(result_1d.sizes(), c10::IntArrayRef({10}));
  ASSERT_NEAR(result_1d.data_ptr<float>()[0], 0.5f, 1e-6);

  // 2D tensor
  at::Tensor tensor_2d = at::full({3, 4}, 2.0f, at::kFloat);
  at::Tensor result_2d = tensor_2d.reciprocal();
  ASSERT_EQ(result_2d.sizes(), c10::IntArrayRef({3, 4}));
  ASSERT_NEAR(result_2d.data_ptr<float>()[0], 0.5f, 1e-6);

  // 3D tensor
  at::Tensor tensor_3d = at::full({2, 3, 4}, 2.0f, at::kFloat);
  at::Tensor result_3d = tensor_3d.reciprocal();
  ASSERT_EQ(result_3d.sizes(), c10::IntArrayRef({2, 3, 4}));
  ASSERT_NEAR(result_3d.data_ptr<float>()[0], 0.5f, 1e-6);

  // 4D tensor
  at::Tensor tensor_4d = at::full({2, 3, 4, 5}, 2.0f, at::kFloat);
  at::Tensor result_4d = tensor_4d.reciprocal();
  ASSERT_EQ(result_4d.sizes(), c10::IntArrayRef({2, 3, 4, 5}));
  ASSERT_NEAR(result_4d.data_ptr<float>()[0], 0.5f, 1e-6);
}

// Test reciprocal_ modifies original tensor
TEST(TestReciprocal, InplaceModifiesOriginal) {
  at::Tensor tensor = at::full({3, 3}, 4.0f, at::kFloat);

  // Store original data pointer
  void* original_ptr = tensor.data_ptr();

  // Call in-place reciprocal
  tensor.reciprocal_();

  // Same memory location
  ASSERT_EQ(tensor.data_ptr(), original_ptr);

  // Values should be modified: 1/4 = 0.25
  float* data = tensor.data_ptr<float>();
  for (int i = 0; i < tensor.numel(); i++) {
    ASSERT_NEAR(data[i], 0.25f, 1e-6);
  }
}

// Test reciprocal creates new tensor (non-inplace)
TEST(TestReciprocal, CreatesNewTensor) {
  at::Tensor tensor = at::full({2, 3}, 2.0f, at::kFloat);
  void* original_ptr = tensor.data_ptr();

  // Non-inplace reciprocal should create new tensor
  at::Tensor result = tensor.reciprocal();

  // Different memory location
  ASSERT_NE(result.data_ptr(), original_ptr);

  // Original tensor unchanged
  float* original_data = tensor.data_ptr<float>();
  ASSERT_NEAR(original_data[0], 2.0f, 1e-6);

  // Result has reciprocal values
  float* result_data = result.data_ptr<float>();
  ASSERT_NEAR(result_data[0], 0.5f, 1e-6);
}

// Test reciprocal with negative values
TEST(TestReciprocal, NegativeValues) {
  at::Tensor tensor = at::ones({4}, at::kFloat);
  float* data = tensor.data_ptr<float>();
  data[0] = -1.0f;
  data[1] = -2.0f;
  data[2] = -0.5f;
  data[3] = -4.0f;

  at::Tensor result = tensor.reciprocal();
  float* result_data = result.data_ptr<float>();

  // Verify reciprocals of negative numbers
  ASSERT_NEAR(result_data[0], -1.0f, 1e-6);   // 1/(-1) = -1
  ASSERT_NEAR(result_data[1], -0.5f, 1e-6);   // 1/(-2) = -0.5
  ASSERT_NEAR(result_data[2], -2.0f, 1e-6);   // 1/(-0.5) = -2
  ASSERT_NEAR(result_data[3], -0.25f, 1e-6);  // 1/(-4) = -0.25
}
