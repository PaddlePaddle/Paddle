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

class TensorExpandTest : public ::testing::Test {
 protected:
  static void SetUpTestSuite() { paddle::prim::InitTensorOperants(); }
};

}  // namespace

// ======================== expand tests ========================

TEST_F(TensorExpandTest, ExpandBasic) {
  at::Tensor t = at::arange(3, at::kFloat);

  at::Tensor result = t.expand({3, 4});

  ASSERT_EQ(result.sizes()[0], 3);
  ASSERT_EQ(result.sizes()[1], 4);
}

TEST_F(TensorExpandTest, ExpandSingleDim) {
  at::Tensor t = at::full({1}, 5.0f, at::kFloat);

  at::Tensor result = t.expand({5});

  ASSERT_EQ(result.numel(), 5);
}

TEST_F(TensorExpandTest, ExpandMultipleDims) {
  at::Tensor t = at::full({1, 3}, 1.0f, at::kFloat);

  at::Tensor result = t.expand({2, 3, 4});

  ASSERT_EQ(result.sizes()[0], 2);
  ASSERT_EQ(result.sizes()[1], 3);
  ASSERT_EQ(result.sizes()[2], 4);
}

TEST_F(TensorExpandTest, ExpandWithImplicit) {
  at::Tensor t = at::arange(3, at::kFloat);

  at::Tensor result = t.expand({3, 4}, true);

  ASSERT_EQ(result.sizes()[0], 3);
  ASSERT_EQ(result.sizes()[1], 4);
}

TEST_F(TensorExpandTest, ExpandPreservesValue) {
  at::Tensor t = at::full({3}, 7.0f, at::kFloat);

  at::Tensor result = t.expand({3, 4});

  ASSERT_FLOAT_EQ(result.data_ptr<float>()[0], 7.0f);
  ASSERT_FLOAT_EQ(result.data_ptr<float>()[7], 7.0f);
}

// ======================== expand_as tests ========================

TEST_F(TensorExpandTest, ExpandAsBasic) {
  at::Tensor t = at::arange(3, at::kFloat).reshape({1, 3});
  at::Tensor other = at::zeros({2, 3}, at::kFloat);

  at::Tensor result = t.expand_as(other);

  ASSERT_EQ(result.sizes()[0], 2);
  ASSERT_EQ(result.sizes()[1], 3);
}

TEST_F(TensorExpandTest, ExpandAsMatchSize) {
  at::Tensor t = at::full({1}, 7.0f, at::kFloat);
  at::Tensor other = at::zeros({3, 3, 3}, at::kFloat);

  at::Tensor result = t.expand_as(other);

  ASSERT_EQ(result.sizes().size(), 3);
  ASSERT_EQ(result.numel(), other.numel());
}

TEST_F(TensorExpandTest, ExpandAsPreservesValue) {
  at::Tensor t = at::full({2, 1}, 5.0f, at::kFloat);
  at::Tensor other = at::zeros({2, 3}, at::kFloat);

  at::Tensor result = t.expand_as(other);

  ASSERT_FLOAT_EQ(result.data_ptr<float>()[0], 5.0f);
}
