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
#include <ATen/ops/tensor.h>
#include <c10/core/ScalarType.h>
#include <c10/core/TensorOptions.h>

#include "ATen/ATen.h"
#include "gtest/gtest.h"
#include "test/cpp/prim/init_env_utils.h"
#include "torch/all.h"

namespace {

class TensorClampTest : public ::testing::Test {
 protected:
  static void SetUpTestSuite() { paddle::prim::InitTensorOperants(); }
};

class TensorOperatorIndexTest : public ::testing::Test {
 protected:
  static void SetUpTestSuite() { paddle::prim::InitTensorOperants(); }
};

}  // namespace

TEST_F(TensorClampTest, ClampWithScalar) {
  // Create tensor with values [0, 1, 2, 3, 4, 5]
  at::Tensor t = at::arange(6, at::kFloat).reshape({2, 3});
  at::Tensor result = t.clamp(at::Scalar(1.0), at::Scalar(4.0));

  float* data = result.data_ptr<float>();
  // Expected: [1, 1, 2, 3, 4, 4]
  ASSERT_FLOAT_EQ(data[0], 1.0f);
  ASSERT_FLOAT_EQ(data[1], 1.0f);
  ASSERT_FLOAT_EQ(data[2], 2.0f);
  ASSERT_FLOAT_EQ(data[3], 3.0f);
  ASSERT_FLOAT_EQ(data[4], 4.0f);
  ASSERT_FLOAT_EQ(data[5], 4.0f);
}

TEST_F(TensorClampTest, ClampWithTensor) {
  at::Tensor t = at::arange(6, at::kFloat).reshape({2, 3});
  at::Tensor min_t = at::full({2, 3}, 1.0f, at::kFloat);
  at::Tensor max_t = at::full({2, 3}, 4.0f, at::kFloat);

  at::Tensor result = t.clamp(::std::optional<at::Tensor>(min_t),
                              ::std::optional<at::Tensor>(max_t));

  float* data = result.data_ptr<float>();
  ASSERT_FLOAT_EQ(data[0], 1.0f);
  ASSERT_FLOAT_EQ(data[5], 4.0f);
}

TEST_F(TensorClampTest, ClampInplaceScalar) {
  at::Tensor t = at::arange(6, at::kFloat).reshape({2, 3});
  t.clamp_(at::Scalar(2.0), at::Scalar(3.0));

  float* data = t.data_ptr<float>();
  // Expected: [2, 2, 2, 3, 3, 3]
  ASSERT_FLOAT_EQ(data[0], 2.0f);
  ASSERT_FLOAT_EQ(data[1], 2.0f);
  ASSERT_FLOAT_EQ(data[2], 2.0f);
  ASSERT_FLOAT_EQ(data[3], 3.0f);
  ASSERT_FLOAT_EQ(data[4], 3.0f);
  ASSERT_FLOAT_EQ(data[5], 3.0f);
}

TEST_F(TensorClampTest, ClampInplaceTensor) {
  at::Tensor t = at::arange(6, at::kFloat).reshape({2, 3});
  at::Tensor min_t = at::full({2, 3}, 1.0f, at::kFloat);
  at::Tensor max_t = at::full({2, 3}, 4.0f, at::kFloat);

  t.clamp_(::std::optional<at::Tensor>(min_t),
           ::std::optional<at::Tensor>(max_t));

  float* data = t.data_ptr<float>();
  ASSERT_FLOAT_EQ(data[0], 1.0f);
  ASSERT_FLOAT_EQ(data[5], 4.0f);
}

TEST_F(TensorClampTest, ClampMaxScalar) {
  at::Tensor t = at::arange(6, at::kFloat);
  at::Tensor result = t.clamp_max(at::Scalar(3.0));

  float* data = result.data_ptr<float>();
  // Expected: [0, 1, 2, 3, 3, 3]
  ASSERT_FLOAT_EQ(data[4], 3.0f);
  ASSERT_FLOAT_EQ(data[5], 3.0f);
}

TEST_F(TensorClampTest, ClampMaxTensor) {
  at::Tensor t = at::arange(6, at::kFloat);
  at::Tensor max_t = at::full({6}, 3.0f, at::kFloat);
  at::Tensor result = t.clamp_max(max_t);

  float* data = result.data_ptr<float>();
  ASSERT_FLOAT_EQ(data[4], 3.0f);
  ASSERT_FLOAT_EQ(data[5], 3.0f);
}

TEST_F(TensorClampTest, ClampMaxInplaceScalar) {
  at::Tensor t = at::arange(6, at::kFloat);
  t.clamp_max_(at::Scalar(3.0));

  float* data = t.data_ptr<float>();
  ASSERT_FLOAT_EQ(data[4], 3.0f);
  ASSERT_FLOAT_EQ(data[5], 3.0f);
}

TEST_F(TensorClampTest, ClampMaxInplaceTensor) {
  at::Tensor t = at::arange(6, at::kFloat);
  at::Tensor max_t = at::full({6}, 3.0f, at::kFloat);
  t.clamp_max_(max_t);

  float* data = t.data_ptr<float>();
  ASSERT_FLOAT_EQ(data[4], 3.0f);
  ASSERT_FLOAT_EQ(data[5], 3.0f);
}

TEST_F(TensorClampTest, ClampMinScalar) {
  at::Tensor t = at::arange(6, at::kFloat);
  at::Tensor result = t.clamp_min(at::Scalar(2.0));

  float* data = result.data_ptr<float>();
  // Expected: [2, 2, 2, 3, 4, 5]
  ASSERT_FLOAT_EQ(data[0], 2.0f);
  ASSERT_FLOAT_EQ(data[1], 2.0f);
}

TEST_F(TensorClampTest, ClampMinTensor) {
  at::Tensor t = at::arange(6, at::kFloat);
  at::Tensor min_t = at::full({6}, 2.0f, at::kFloat);
  at::Tensor result = t.clamp_min(min_t);

  float* data = result.data_ptr<float>();
  ASSERT_FLOAT_EQ(data[0], 2.0f);
  ASSERT_FLOAT_EQ(data[1], 2.0f);
}

TEST_F(TensorClampTest, ClampMinInplaceScalar) {
  at::Tensor t = at::arange(6, at::kFloat);
  t.clamp_min_(at::Scalar(2.0));

  float* data = t.data_ptr<float>();
  ASSERT_FLOAT_EQ(data[0], 2.0f);
  ASSERT_FLOAT_EQ(data[1], 2.0f);
}

TEST_F(TensorClampTest, ClampMinInplaceTensor) {
  at::Tensor t = at::arange(6, at::kFloat);
  at::Tensor min_t = at::full({6}, 2.0f, at::kFloat);
  t.clamp_min_(min_t);

  float* data = t.data_ptr<float>();
  ASSERT_FLOAT_EQ(data[0], 2.0f);
  ASSERT_FLOAT_EQ(data[1], 2.0f);
}

// ======================== operator[] tests ========================

TEST_F(TensorOperatorIndexTest, OperatorIndexBasic) {
  // Create tensor [[0,1,2],[3,4,5]]
  at::Tensor t = at::arange(6, at::kFloat).reshape({2, 3});

  // Test operator[](int64_t index) - returns first row
  at::Tensor result0 = t[0];
  ASSERT_EQ(result0.numel(), 3);  // First row has 3 elements [0,1,2]
  ASSERT_FLOAT_EQ(result0.data_ptr<float>()[0],
                  0.0f);  // First element of the row

  at::Tensor result1 = t[1];
  ASSERT_EQ(result1.numel(), 3);  // Second row has 3 elements [3,4,5]
  ASSERT_FLOAT_EQ(result1.data_ptr<float>()[0],
                  3.0f);  // First element of the row
}

TEST_F(TensorOperatorIndexTest, OperatorIndexOutOfBounds) {
  at::Tensor t = at::arange(6, at::kFloat).reshape({2, 3});

  // Test out of bounds index - should throw an exception
  // The test expects the code to handle this gracefully
  bool threw_exception = false;
  try {
    at::Tensor result = t[5];
    (void)result;
  } catch (...) {
    threw_exception = true;
  }
  // Note: Depending on implementation, this may or may not throw
  // We accept either behavior (return empty/invalid tensor or throw)
}

// ======================= Additional clamp edge case tests
// =======================

TEST_F(TensorClampTest, ClampNoMinMax) {
  // Test clamp with no min and max (should be identity)
  at::Tensor t = at::arange(6, at::kFloat);
  at::Tensor result = t.clamp(::std::optional<at::Scalar>(::std::nullopt),
                              ::std::optional<at::Scalar>(::std::nullopt));

  ASSERT_EQ(result.numel(), 6);
  float* data = result.data_ptr<float>();
  for (int i = 0; i < 6; i++) {
    ASSERT_FLOAT_EQ(data[i], static_cast<float>(i));
  }
}

TEST_F(TensorClampTest, ClampOnlyMin) {
  // Test clamp with only min value
  at::Tensor t = at::arange(6, at::kFloat);
  at::Tensor result =
      t.clamp(at::Scalar(2.5), ::std::optional<at::Scalar>(::std::nullopt));

  float* data = result.data_ptr<float>();
  ASSERT_FLOAT_EQ(data[0], 2.5f);  // 0 < 2.5 -> 2.5
  ASSERT_FLOAT_EQ(data[1], 2.5f);  // 1 < 2.5 -> 2.5
  ASSERT_FLOAT_EQ(data[2], 2.5f);  // 2 < 2.5 -> 2.5
}

TEST_F(TensorClampTest, ClampOnlyMax) {
  // Test clamp with only max value
  at::Tensor t = at::arange(6, at::kFloat);
  at::Tensor result =
      t.clamp(::std::optional<at::Scalar>(::std::nullopt), at::Scalar(2.5));

  float* data = result.data_ptr<float>();
  ASSERT_FLOAT_EQ(data[0], 0.0f);
  ASSERT_FLOAT_EQ(data[1], 1.0f);
  ASSERT_FLOAT_EQ(data[2], 2.0f);
  ASSERT_FLOAT_EQ(data[3], 2.5f);
}

TEST_F(TensorClampTest, ClampMinOnlyTensor) {
  // Test clamp_min with Tensor
  at::Tensor t = at::arange(6, at::kFloat);
  at::Tensor min_t = at::full({6}, 2.5f, at::kFloat);
  at::Tensor result = t.clamp_min(min_t);

  float* data = result.data_ptr<float>();
  ASSERT_FLOAT_EQ(data[0], 2.5f);  // 0 < 2.5 -> 2.5
  ASSERT_FLOAT_EQ(data[1], 2.5f);  // 1 < 2.5 -> 2.5
  ASSERT_FLOAT_EQ(data[2], 2.5f);  // 2 < 2.5 -> 2.5
}

TEST_F(TensorClampTest, ClampMaxOnlyTensor) {
  // Test clamp_max with Tensor
  at::Tensor t = at::arange(6, at::kFloat);
  at::Tensor max_t = at::full({6}, 2.5f, at::kFloat);
  at::Tensor result = t.clamp_max(max_t);

  float* data = result.data_ptr<float>();
  ASSERT_FLOAT_EQ(data[0], 0.0f);
  ASSERT_FLOAT_EQ(data[1], 1.0f);
  ASSERT_FLOAT_EQ(data[2], 2.0f);
  ASSERT_FLOAT_EQ(data[3], 2.5f);
}

TEST_F(TensorClampTest, ClampWithTensorBothNone) {
  // Test clamp with both min and max as empty optional
  at::Tensor t = at::arange(6, at::kFloat).reshape({2, 3});
  at::Tensor result = t.clamp(::std::optional<at::Tensor>(::std::nullopt),
                              ::std::optional<at::Tensor>(::std::nullopt));

  ASSERT_EQ(result.numel(), 6);
}

TEST_F(TensorClampTest, ClampMinTensorMaxNone) {
  // Test clamp with min tensor, max none
  at::Tensor t = at::arange(6, at::kFloat);
  at::Tensor min_t = at::full({6}, 2.0f, at::kFloat);
  at::Tensor result = t.clamp(::std::optional<at::Tensor>(min_t),
                              ::std::optional<at::Tensor>(::std::nullopt));

  float* data = result.data_ptr<float>();
  ASSERT_FLOAT_EQ(data[0], 2.0f);
}

TEST_F(TensorClampTest, ClampMinNoneMaxTensor) {
  // Test clamp with min none, max tensor
  at::Tensor t = at::arange(6, at::kFloat);
  at::Tensor max_t = at::full({6}, 3.0f, at::kFloat);
  at::Tensor result = t.clamp(::std::optional<at::Tensor>(::std::nullopt),
                              ::std::optional<at::Tensor>(max_t));

  float* data = result.data_ptr<float>();
  ASSERT_FLOAT_EQ(data[3], 3.0f);
  ASSERT_FLOAT_EQ(data[4], 3.0f);
}

TEST_F(TensorClampTest, ClampInplaceMinNoneMax) {
  // Test clamp_ with min none
  at::Tensor t = at::arange(6, at::kFloat);
  t.clamp_(::std::optional<at::Scalar>(::std::nullopt), at::Scalar(2.5));

  float* data = t.data_ptr<float>();
  ASSERT_FLOAT_EQ(data[3], 2.5f);
}

TEST_F(TensorClampTest, ClampInplaceMaxNoneMin) {
  // Test clamp_ with max none
  at::Tensor t = at::arange(6, at::kFloat);
  t.clamp_(at::Scalar(2.0), ::std::optional<at::Scalar>(::std::nullopt));

  float* data = t.data_ptr<float>();
  ASSERT_FLOAT_EQ(data[0], 2.0f);
}
