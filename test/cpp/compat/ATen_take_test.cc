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

#include <complex>
#include <cstdint>
#include <vector>

#include "ATen/ATen.h"
#include "ATen/Functions.h"
#include "ATen/core/TensorBody.h"
#include "ATen/ops/as_strided.h"
#include "ATen/ops/full.h"
#include "ATen/ops/take.h"
#include "ATen/ops/tensor.h"
#include "c10/util/complex.h"
#include "gtest/gtest.h"
#include "torch/all.h"

// ==================== take tests ====================

static at::Tensor make_long_index(const std::vector<int64_t>& values) {
  return at::tensor(at::ArrayRef<int64_t>(values),
                    at::TensorOptions().dtype(at::kLong));
}

static at::Tensor make_float_tensor(const std::vector<float>& values) {
  return at::tensor(at::ArrayRef<float>(values),
                    at::TensorOptions().dtype(at::kFloat));
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

TEST(TakeTest, TakeExtendedDtypes) {
  auto index = make_long_index({0, 3, 1});

  auto tensor_bool = make_long_index({0, 1, 0, 1}).to(at::kBool);
  auto result_bool = at::take(tensor_bool, index);
  EXPECT_EQ(result_bool.scalar_type(), at::kBool);
  EXPECT_FALSE(result_bool.data_ptr<bool>()[0]);
  EXPECT_TRUE(result_bool.data_ptr<bool>()[1]);
  EXPECT_TRUE(result_bool.data_ptr<bool>()[2]);

  std::vector<int8_t> char_values = {-4, -2, 3, 7};
  auto tensor_char = at::tensor(at::ArrayRef<int8_t>(char_values),
                                at::TensorOptions().dtype(at::kChar));
  auto result_char = at::take(tensor_char, index);
  EXPECT_EQ(result_char.scalar_type(), at::kChar);
  EXPECT_EQ(result_char.data_ptr<int8_t>()[0], static_cast<int8_t>(-4));
  EXPECT_EQ(result_char.data_ptr<int8_t>()[1], static_cast<int8_t>(7));
  EXPECT_EQ(result_char.data_ptr<int8_t>()[2], static_cast<int8_t>(-2));

  auto tensor_half = make_float_tensor({1.5f, -2.0f, 0.5f, 4.0f}).to(at::kHalf);
  auto result_half = at::take(tensor_half, index);
  EXPECT_EQ(result_half.scalar_type(), at::kHalf);
  auto result_half_float = result_half.to(at::kFloat);
  EXPECT_FLOAT_EQ(result_half_float.data_ptr<float>()[0], 1.5f);
  EXPECT_FLOAT_EQ(result_half_float.data_ptr<float>()[1], 4.0f);
  EXPECT_FLOAT_EQ(result_half_float.data_ptr<float>()[2], -2.0f);

  auto tensor_bfloat =
      make_float_tensor({1.25f, -3.5f, 2.0f, 8.0f}).to(at::kBFloat16);
  auto result_bfloat = at::take(tensor_bfloat, index);
  EXPECT_EQ(result_bfloat.scalar_type(), at::kBFloat16);
  auto result_bfloat_float = result_bfloat.to(at::kFloat);
  EXPECT_FLOAT_EQ(result_bfloat_float.data_ptr<float>()[0], 1.25f);
  EXPECT_FLOAT_EQ(result_bfloat_float.data_ptr<float>()[1], 8.0f);
  EXPECT_FLOAT_EQ(result_bfloat_float.data_ptr<float>()[2], -3.5f);

  std::vector<c10::complex<float>> complex_float_values = {
      {1.0f, 2.0f}, {3.0f, -4.0f}, {-5.0f, 6.0f}, {7.0f, 8.0f}};
  auto tensor_complex_float =
      at::tensor(at::ArrayRef<c10::complex<float>>(complex_float_values),
                 at::TensorOptions().dtype(at::kComplexFloat));
  auto result_complex_float = at::take(tensor_complex_float, index);
  EXPECT_EQ(result_complex_float.scalar_type(), at::kComplexFloat);
  auto* complex_float_data =
      result_complex_float.data_ptr<c10::complex<float>>();
  auto complex_float_0 =
      static_cast<std::complex<float>>(complex_float_data[0]);
  auto complex_float_1 =
      static_cast<std::complex<float>>(complex_float_data[1]);
  auto complex_float_2 =
      static_cast<std::complex<float>>(complex_float_data[2]);
  EXPECT_FLOAT_EQ(complex_float_0.real(), 1.0f);
  EXPECT_FLOAT_EQ(complex_float_0.imag(), 2.0f);
  EXPECT_FLOAT_EQ(complex_float_1.real(), 7.0f);
  EXPECT_FLOAT_EQ(complex_float_1.imag(), 8.0f);
  EXPECT_FLOAT_EQ(complex_float_2.real(), 3.0f);
  EXPECT_FLOAT_EQ(complex_float_2.imag(), -4.0f);

  std::vector<c10::complex<double>> complex_double_values = {
      {1.0, -2.0}, {-3.0, 4.0}, {5.0, -6.0}, {-7.0, -8.0}};
  auto tensor_complex_double =
      at::tensor(at::ArrayRef<c10::complex<double>>(complex_double_values),
                 at::TensorOptions().dtype(at::kComplexDouble));
  auto result_complex_double = at::take(tensor_complex_double, index);
  EXPECT_EQ(result_complex_double.scalar_type(), at::kComplexDouble);
  auto* complex_double_data =
      result_complex_double.data_ptr<c10::complex<double>>();
  auto complex_double_0 =
      static_cast<std::complex<double>>(complex_double_data[0]);
  auto complex_double_1 =
      static_cast<std::complex<double>>(complex_double_data[1]);
  auto complex_double_2 =
      static_cast<std::complex<double>>(complex_double_data[2]);
  EXPECT_DOUBLE_EQ(complex_double_0.real(), 1.0);
  EXPECT_DOUBLE_EQ(complex_double_0.imag(), -2.0);
  EXPECT_DOUBLE_EQ(complex_double_1.real(), -7.0);
  EXPECT_DOUBLE_EQ(complex_double_1.imag(), -8.0);
  EXPECT_DOUBLE_EQ(complex_double_2.real(), -3.0);
  EXPECT_DOUBLE_EQ(complex_double_2.imag(), 4.0);
}

TEST(TakeTest, TakeCpuFloat8Throws) {
  auto index = make_long_index({0, 3, 1});

  auto tensor_float8_e5m2 =
      make_float_tensor({1.0f, 2.0f, 4.0f, 8.0f}).to(at::kFloat8_e5m2);
  EXPECT_THROW(at::take(tensor_float8_e5m2, index), std::exception);

  auto tensor_float8_e4m3fn =
      make_float_tensor({1.0f, 2.0f, 4.0f, 8.0f}).to(at::kFloat8_e4m3fn);
  EXPECT_THROW(at::take(tensor_float8_e4m3fn, index), std::exception);
}

TEST(TakeTest, TakeNonContiguousInput) {
  auto base = at::arange(6, at::TensorOptions().dtype(at::kFloat));
  auto tensor = base.as_strided({3}, {2});
  auto index = make_long_index({1});

  auto result = at::take(tensor, index);

  EXPECT_EQ(result.dim(), 1);
  EXPECT_EQ(result.size(0), 1);
  EXPECT_FLOAT_EQ(result[0].item<float>(), 2.0f);
}

TEST(TakeTest, TakeNonContiguousIndex) {
  auto tensor = at::arange(6, at::TensorOptions().dtype(at::kFloat));
  auto index_base = make_long_index({0, 1, 2, 3, 4, 5});
  auto index = index_base.as_strided({3}, {2});

  auto result = at::take(tensor, index);

  EXPECT_EQ(result.dim(), 1);
  EXPECT_EQ(result.size(0), 3);
  EXPECT_FLOAT_EQ(result[0].item<float>(), 0.0f);
  EXPECT_FLOAT_EQ(result[1].item<float>(), 2.0f);
  EXPECT_FLOAT_EQ(result[2].item<float>(), 4.0f);
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
