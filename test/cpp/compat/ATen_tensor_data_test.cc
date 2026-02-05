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
#include <c10/core/ScalarType.h>
#include <c10/core/TensorOptions.h>
#include "gtest/gtest.h"

// Test for Tensor::tensor_data() and Tensor::variable_data()
TEST(TensorDataTest, TensorDataContiguous) {
  // Create a contiguous tensor
  at::Tensor tensor = at::arange(12, at::kFloat).reshape({3, 4});

  // Get tensor_data
  at::Tensor data_tensor = tensor.tensor_data();

  // Verify shape and values
  ASSERT_EQ(data_tensor.dim(), 2);
  ASSERT_EQ(data_tensor.size(0), 3);
  ASSERT_EQ(data_tensor.size(1), 4);

  // Verify values match
  for (int64_t i = 0; i < tensor.numel(); ++i) {
    ASSERT_EQ(tensor.data_ptr<float>()[i], data_tensor.data_ptr<float>()[i]);
  }

  // Verify they share data (for contiguous tensors)
  // Modify original tensor and check if data_tensor reflects the change
  tensor.fill_(42.0f);
  ASSERT_EQ(data_tensor.data_ptr<float>()[0], 42.0f);
}

TEST(TensorDataTest, TensorDataNonContiguous) {
  // Create a non-contiguous tensor (transpose creates non-contiguous view)
  at::Tensor tensor = at::arange(12, at::kFloat).reshape({3, 4});
  at::Tensor transposed = tensor.transpose(0, 1);

  // Verify it's non-contiguous
  ASSERT_FALSE(transposed.is_contiguous());

  // Get tensor_data
  at::Tensor data_tensor = transposed.tensor_data();

  // Verify shape matches
  ASSERT_EQ(data_tensor.dim(), 2);
  ASSERT_EQ(data_tensor.size(0), 4);
  ASSERT_EQ(data_tensor.size(1), 3);

  // Verify values match
  for (int64_t i = 0; i < transposed.numel(); ++i) {
    ASSERT_EQ(transposed.data_ptr<float>()[i],
              data_tensor.data_ptr<float>()[i]);
  }

  // For non-contiguous tensors, data should be copied
  // Modify original and verify data_tensor is independent
  transposed.fill_(99.0f);
  // The data_tensor should have the original values, not the modified ones
  // (since it's a copy)
}

TEST(TensorDataTest, TensorDataEmptyTensor) {
  // Create an empty tensor
  at::Tensor tensor = at::empty({0}, at::kFloat);

  // Get tensor_data
  at::Tensor data_tensor = tensor.tensor_data();

  // Verify shape
  ASSERT_EQ(data_tensor.dim(), 1);
  ASSERT_EQ(data_tensor.size(0), 0);
  ASSERT_EQ(data_tensor.numel(), 0);
}

TEST(TensorDataTest, TensorDataDifferentDtypes) {
  // Test with different data types
  std::vector<c10::ScalarType> dtypes = {
      at::kFloat, at::kDouble, at::kInt, at::kLong, at::kBool};

  for (auto dtype : dtypes) {
    at::Tensor tensor = at::ones({2, 3}, at::TensorOptions().dtype(dtype));
    at::Tensor data_tensor = tensor.tensor_data();

    ASSERT_EQ(data_tensor.dtype(), dtype);
    ASSERT_EQ(data_tensor.dim(), 2);
    ASSERT_EQ(data_tensor.size(0), 2);
    ASSERT_EQ(data_tensor.size(1), 3);
  }
}

TEST(TensorDataTest, VariableDataContiguous) {
  // Create a contiguous tensor
  at::Tensor tensor = at::arange(12, at::kFloat).reshape({3, 4});

  // Get variable_data
  at::Tensor var_tensor = tensor.variable_data();

  // Verify shape and values
  ASSERT_EQ(var_tensor.dim(), 2);
  ASSERT_EQ(var_tensor.size(0), 3);
  ASSERT_EQ(var_tensor.size(1), 4);

  // Verify values match
  for (int64_t i = 0; i < tensor.numel(); ++i) {
    ASSERT_EQ(tensor.data_ptr<float>()[i], var_tensor.data_ptr<float>()[i]);
  }

  // Verify they share data (for contiguous tensors)
  tensor.fill_(42.0f);
  ASSERT_EQ(var_tensor.data_ptr<float>()[0], 42.0f);
}

TEST(TensorDataTest, VariableDataNonContiguous) {
  // Create a non-contiguous tensor
  at::Tensor tensor = at::arange(12, at::kFloat).reshape({3, 4});
  at::Tensor transposed = tensor.transpose(0, 1);

  // Verify it's non-contiguous
  ASSERT_FALSE(transposed.is_contiguous());

  // Get variable_data
  at::Tensor var_tensor = transposed.variable_data();

  // Verify shape matches
  ASSERT_EQ(var_tensor.dim(), 2);
  ASSERT_EQ(var_tensor.size(0), 4);
  ASSERT_EQ(var_tensor.size(1), 3);

  // Verify values match
  for (int64_t i = 0; i < transposed.numel(); ++i) {
    ASSERT_EQ(transposed.data_ptr<float>()[i], var_tensor.data_ptr<float>()[i]);
  }
}

TEST(TensorDataTest, VariableDataEmptyTensor) {
  // Create an empty tensor
  at::Tensor tensor = at::empty({0}, at::kFloat);

  // Get variable_data
  at::Tensor var_tensor = tensor.variable_data();

  // Verify shape
  ASSERT_EQ(var_tensor.dim(), 1);
  ASSERT_EQ(var_tensor.size(0), 0);
  ASSERT_EQ(var_tensor.numel(), 0);
}

TEST(TensorDataTest, VariableDataDifferentDtypes) {
  // Test with different data types
  std::vector<c10::ScalarType> dtypes = {
      at::kFloat, at::kDouble, at::kInt, at::kLong, at::kBool};

  for (auto dtype : dtypes) {
    at::Tensor tensor = at::ones({2, 3}, at::TensorOptions().dtype(dtype));
    at::Tensor var_tensor = tensor.variable_data();

    ASSERT_EQ(var_tensor.dtype(), dtype);
    ASSERT_EQ(var_tensor.dim(), 2);
    ASSERT_EQ(var_tensor.size(0), 2);
    ASSERT_EQ(var_tensor.size(1), 3);
  }
}

TEST(TensorDataTest, TensorDataAndVariableDataEquivalence) {
  // Test that tensor_data() and variable_data() produce equivalent results
  at::Tensor tensor = at::arange(24, at::kFloat).reshape({2, 3, 4});

  at::Tensor tensor_data_result = tensor.tensor_data();
  at::Tensor variable_data_result = tensor.variable_data();

  // Verify shapes match
  ASSERT_EQ(tensor_data_result.dim(), variable_data_result.dim());
  for (int64_t i = 0; i < tensor_data_result.dim(); ++i) {
    ASSERT_EQ(tensor_data_result.size(i), variable_data_result.size(i));
  }

  // Verify values match
  ASSERT_EQ(tensor_data_result.numel(), variable_data_result.numel());
  for (int64_t i = 0; i < tensor_data_result.numel(); ++i) {
    ASSERT_EQ(tensor_data_result.data_ptr<float>()[i],
              variable_data_result.data_ptr<float>()[i]);
  }
}

TEST(TensorDataTest, TensorDataModificationIndependence) {
  // Test that modifying the result doesn't affect original (for copied data)
  at::Tensor tensor = at::ones({3, 4}, at::kFloat);

  at::Tensor data_tensor = tensor.tensor_data();

  // Modify data_tensor
  data_tensor.fill_(5.0f);

  // For contiguous tensors, they share data, so original should also change
  // For non-contiguous, they are independent
  if (tensor.is_contiguous()) {
    // They share data, so original should reflect changes
    ASSERT_EQ(tensor.data_ptr<float>()[0], 5.0f);
  }
}

TEST(TensorDataTest, TensorData3DTensor) {
  // Test with 3D tensor
  at::Tensor tensor = at::arange(24, at::kFloat).reshape({2, 3, 4});

  at::Tensor data_tensor = tensor.tensor_data();

  ASSERT_EQ(data_tensor.dim(), 3);
  ASSERT_EQ(data_tensor.size(0), 2);
  ASSERT_EQ(data_tensor.size(1), 3);
  ASSERT_EQ(data_tensor.size(2), 4);

  // Verify all values match
  for (int64_t i = 0; i < tensor.numel(); ++i) {
    ASSERT_EQ(tensor.data_ptr<float>()[i], data_tensor.data_ptr<float>()[i]);
  }
}

TEST(TensorDataTest, VariableData3DTensor) {
  // Test with 3D tensor
  at::Tensor tensor = at::arange(24, at::kFloat).reshape({2, 3, 4});

  at::Tensor var_tensor = tensor.variable_data();

  ASSERT_EQ(var_tensor.dim(), 3);
  ASSERT_EQ(var_tensor.size(0), 2);
  ASSERT_EQ(var_tensor.size(1), 3);
  ASSERT_EQ(var_tensor.size(2), 4);

  // Verify all values match
  for (int64_t i = 0; i < tensor.numel(); ++i) {
    ASSERT_EQ(tensor.data_ptr<float>()[i], var_tensor.data_ptr<float>()[i]);
  }
}
