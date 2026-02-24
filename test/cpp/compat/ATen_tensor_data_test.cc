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
#include "torch/all.h"

// ======================== tensor_data / variable_data tests
// ========================

TEST(TensorDataTest, TensorData) {
  at::Tensor t = at::arange(6, at::kFloat).reshape({2, 3});
  at::Tensor td = t.tensor_data();

  ASSERT_EQ(td.sizes(), t.sizes());
  ASSERT_EQ(td.dtype(), t.dtype());
  ASSERT_EQ(td.numel(), t.numel());

  // Values should be the same
  float* orig = t.data_ptr<float>();
  float* copy = td.data_ptr<float>();
  for (int i = 0; i < t.numel(); i++) {
    ASSERT_FLOAT_EQ(orig[i], copy[i]);
  }
}

TEST(TensorDataTest, VariableData) {
  at::Tensor t = at::arange(6, at::kFloat).reshape({2, 3});
  at::Tensor vd = t.variable_data();

  ASSERT_EQ(vd.sizes(), t.sizes());
  ASSERT_EQ(vd.dtype(), t.dtype());
  ASSERT_EQ(vd.numel(), t.numel());

  float* orig = t.data_ptr<float>();
  float* copy = vd.data_ptr<float>();
  for (int i = 0; i < t.numel(); i++) {
    ASSERT_FLOAT_EQ(orig[i], copy[i]);
  }
}

// ======================== item tests ========================

TEST(TensorItemTest, ItemScalar) {
  at::Tensor t = at::full({}, 3.14f, at::kFloat);
  at::Scalar s = t.item();
  ASSERT_NEAR(s.to<float>(), 3.14f, 1e-5);
}

TEST(TensorItemTest, ItemTyped) {
  at::Tensor t = at::full({1}, 42.0f, at::kFloat);
  float val = t.item<float>();
  ASSERT_FLOAT_EQ(val, 42.0f);
}

TEST(TensorItemTest, ItemInt) {
  at::Tensor t = at::full({1}, 7, at::kInt);
  at::Scalar s = t.item();
  ASSERT_EQ(s.to<int>(), 7);
}

TEST(TensorItemTest, ItemDouble) {
  at::Tensor t = at::full({1}, 2.718, at::kDouble);
  at::Scalar s = t.item();
  ASSERT_NEAR(s.to<double>(), 2.718, 1e-6);
}

TEST(TensorItemTest, ItemInt64) {
  at::Tensor t = at::full({1}, 12345, at::kLong);
  at::Scalar s = t.item();
  ASSERT_EQ(s.to<int64_t>(), 12345);
}

TEST(TensorItemTest, ItemBool) {
  at::Tensor t = at::full({1}, true, at::kBool);
  at::Scalar s = t.item();
  ASSERT_TRUE(s.to<bool>());
}

TEST(TensorItemTest, ItemInt8) {
  at::Tensor t = at::full({1}, 5, at::kChar);
  at::Scalar s = t.item();
  ASSERT_EQ(s.to<int8_t>(), 5);
}

TEST(TensorItemTest, ItemUint8) {
  at::Tensor t = at::full({1}, 200, at::kByte);
  at::Scalar s = t.item();
  ASSERT_EQ(s.to<uint8_t>(), 200);
}

TEST(TensorItemTest, ItemInt16) {
  at::Tensor t = at::full({1}, 300, at::kShort);
  at::Scalar s = t.item();
  ASSERT_EQ(s.to<int16_t>(), 300);
}
