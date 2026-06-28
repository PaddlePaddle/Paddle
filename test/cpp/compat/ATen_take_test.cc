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

#include <vector>

#include "ATen/ATen.h"
#include "ATen/Functions.h"
#include "ATen/core/TensorBody.h"
#include "ATen/ops/full.h"
#include "ATen/ops/take.h"
#include "ATen/ops/tensor.h"
#include "gtest/gtest.h"
#include "torch/all.h"

// ==================== take tests ====================

static at::Tensor make_long_index(const std::vector<int64_t>& values) {
  return at::tensor(at::ArrayRef<int64_t>(values),
                    at::TensorOptions().dtype(at::kLong));
}

// Test for take on 1D tensor
TEST(TakeTest, Take1D) {
  auto tensor = at::arange(10, at::TensorOptions().dtype(at::kFloat));
  auto index = make_long_index({2, 5, 7});

  auto result = at::take(tensor, index);

  EXPECT_EQ(result.dim(), 1);
  EXPECT_EQ(result.size(0), 3);
  EXPECT_FLOAT_EQ(result[0].item<float>(), 2.0f);
  EXPECT_FLOAT_EQ(result[1].item<float>(), 5.0f);
  EXPECT_FLOAT_EQ(result[2].item<float>(), 7.0f);
}

// Test for take on 2D tensor (treated as flattened)
TEST(TakeTest, Take2DFlattened) {
  auto tensor =
      at::arange(12, at::TensorOptions().dtype(at::kFloat)).reshape({3, 4});

  // Flattened indices: element at flattened position 0 is 0, position 5 is 5,
  // position 11 is 11
  auto index = make_long_index({0, 5, 11});

  auto result = at::take(tensor, index);

  EXPECT_EQ(result.dim(), 1);
  EXPECT_EQ(result.size(0), 3);
  EXPECT_FLOAT_EQ(result[0].item<float>(), 0.0f);
  EXPECT_FLOAT_EQ(result[1].item<float>(), 5.0f);
  EXPECT_FLOAT_EQ(result[2].item<float>(), 11.0f);
}

// Test for take with multi-dimensional index
TEST(TakeTest, TakeMultiDimIndex) {
  auto tensor = at::arange(12, at::TensorOptions().dtype(at::kFloat));

  auto index = make_long_index({0, 3, 7, 10}).reshape({2, 2});

  auto result = at::take(tensor, index);

  EXPECT_EQ(result.dim(), 2);
  EXPECT_EQ(result.size(0), 2);
  EXPECT_EQ(result.size(1), 2);
  EXPECT_FLOAT_EQ(result[0][0].item<float>(), 0.0f);
  EXPECT_FLOAT_EQ(result[0][1].item<float>(), 3.0f);
  EXPECT_FLOAT_EQ(result[1][0].item<float>(), 7.0f);
  EXPECT_FLOAT_EQ(result[1][1].item<float>(), 10.0f);
}

// Test for take with duplicate indices
TEST(TakeTest, TakeDuplicateIndices) {
  auto tensor = at::arange(5, at::TensorOptions().dtype(at::kFloat));

  auto index = make_long_index({1, 1, 3, 1});

  auto result = at::take(tensor, index);

  EXPECT_EQ(result.size(0), 4);
  EXPECT_FLOAT_EQ(result[0].item<float>(), 1.0f);
  EXPECT_FLOAT_EQ(result[1].item<float>(), 1.0f);
  EXPECT_FLOAT_EQ(result[2].item<float>(), 3.0f);
  EXPECT_FLOAT_EQ(result[3].item<float>(), 1.0f);
}

// Test for take with scalar tensor (0-dim index)
TEST(TakeTest, TakeScalarIndex) {
  auto tensor = at::arange(10, at::TensorOptions().dtype(at::kFloat));

  auto index = at::full({}, 7, at::kLong);

  auto result = at::take(tensor, index);

  // Result should be scalar (0-dim tensor)
  EXPECT_EQ(result.dim(), 0);
  EXPECT_FLOAT_EQ(result.item<float>(), 7.0f);
}

// Test for take with different dtypes
TEST(TakeTest, TakeDifferentDtypes) {
  // Test with int64
  auto tensor_int = at::arange(10, at::TensorOptions().dtype(at::kLong));
  auto index = make_long_index({1, 3, 8});

  auto result = at::take(tensor_int, index);

  EXPECT_EQ(result.scalar_type(), at::kLong);
  EXPECT_EQ(result[0].item<int64_t>(), 1);
  EXPECT_EQ(result[1].item<int64_t>(), 3);
  EXPECT_EQ(result[2].item<int64_t>(), 8);

  // Test with double
  auto tensor_double = at::arange(10, at::TensorOptions().dtype(at::kDouble));
  auto result_double = at::take(tensor_double, index);

  EXPECT_EQ(result_double.scalar_type(), at::kDouble);
  EXPECT_DOUBLE_EQ(result_double[0].item<double>(), 1.0);
  EXPECT_DOUBLE_EQ(result_double[1].item<double>(), 3.0);
  EXPECT_DOUBLE_EQ(result_double[2].item<double>(), 8.0);
}

// Test for take on empty index
TEST(TakeTest, TakeEmptyIndex) {
  auto tensor = at::arange(10, at::TensorOptions().dtype(at::kFloat));

  // Empty index
  auto index = at::empty({0}, at::TensorOptions().dtype(at::kLong));

  auto result = at::take(tensor, index);

  EXPECT_EQ(result.numel(), 0);
}

TEST(TakeTest, TakeEmptyInputNonEmptyIndexThrows) {
  auto tensor = at::empty({0}, at::TensorOptions().dtype(at::kFloat));
  auto index = make_long_index({0});

  EXPECT_THROW(at::take(tensor, index), std::exception);
}

// Test for take with non-Long index dtype (should throw)
TEST(TakeTest, TakeNonLongIndexThrows) {
  auto tensor = at::arange(10, at::TensorOptions().dtype(at::kFloat));

  // INT32 index should be rejected
  auto index_int32 = at::tensor({0, 1, 2}, at::TensorOptions().dtype(at::kInt));
  EXPECT_THROW(at::take(tensor, index_int32), std::exception);

  // FLOAT index should be rejected
  auto index_float =
      at::tensor({0.0f, 1.0f, 2.0f}, at::TensorOptions().dtype(at::kFloat));
  EXPECT_THROW(at::take(tensor, index_float), std::exception);
}

TEST(TakeTest, TakeNegativeIndexWraps) {
  auto tensor = at::arange(10, at::TensorOptions().dtype(at::kFloat));
  auto index = make_long_index({-1, -10});

  auto result = at::take(tensor, index);

  EXPECT_FLOAT_EQ(result[0].item<float>(), 9.0f);
  EXPECT_FLOAT_EQ(result[1].item<float>(), 0.0f);
}

TEST(TakeTest, TakeOutOfRangeThrows) {
  auto tensor = at::arange(10, at::TensorOptions().dtype(at::kFloat));

  auto positive_oob = make_long_index({10});
  EXPECT_THROW(at::take(tensor, positive_oob), std::exception);

  auto negative_oob = make_long_index({-11});
  EXPECT_THROW(at::take(tensor, negative_oob), std::exception);
}

// Test for take member function
TEST(TakeTest, TakeMemberFunction) {
  auto tensor = at::arange(10, at::TensorOptions().dtype(at::kFloat));

  auto index = make_long_index({4, 6});

  auto result = tensor.take(index);

  EXPECT_EQ(result.size(0), 2);
  EXPECT_FLOAT_EQ(result[0].item<float>(), 4.0f);
  EXPECT_FLOAT_EQ(result[1].item<float>(), 6.0f);
}

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
// Test for take on CUDA
TEST(TakeTest, TakeCUDA) {
  auto tensor =
      at::arange(10, at::TensorOptions().dtype(at::kFloat).device(at::kCUDA));

  auto index = make_long_index({1, 3, 7}).cuda();

  auto result = at::take(tensor, index);

  EXPECT_TRUE(result.is_cuda());
  EXPECT_EQ(result.size(0), 3);

  auto cpu_result = result.cpu();
  EXPECT_FLOAT_EQ(cpu_result[0].item<float>(), 1.0f);
  EXPECT_FLOAT_EQ(cpu_result[1].item<float>(), 3.0f);
  EXPECT_FLOAT_EQ(cpu_result[2].item<float>(), 7.0f);
}

TEST(TakeTest, TakeCUDANegativeIndexAndOutOfRange) {
  auto tensor =
      at::arange(10, at::TensorOptions().dtype(at::kFloat).device(at::kCUDA));

  auto index = make_long_index({-1}).cuda();
  auto result = at::take(tensor, index).cpu();
  EXPECT_FLOAT_EQ(result[0].item<float>(), 9.0f);

  index = make_long_index({10}).cuda();
  EXPECT_THROW((void)at::take(tensor, index).cpu(), std::exception);
}
#endif
