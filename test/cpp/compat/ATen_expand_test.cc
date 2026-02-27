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

// ======================== Additional tests for coverage
// ========================

// Test tile fallback path when input_rank < target_rank
// This triggers lines 86-100 in expand.h
TEST_F(TensorExpandTest, ExpandTileFallbackLowRank) {
  // {2,1}.expand({1,4}) - input has 2 dims, target has 2 dims
  // but reshape_vec would be {2,1}, which cannot directly use paddle expand
  // because 2 != 1 and 2 != 4
  at::Tensor t = at::full({2, 1}, 1.0f, at::kFloat);
  at::Tensor result = t.expand({1, 4});

  ASSERT_EQ(result.sizes()[0], 1);
  ASSERT_EQ(result.sizes()[1], 4);
  ASSERT_FLOAT_EQ(result.data_ptr<float>()[0], 1.0f);
}

// Test tile fallback when input_rank == target_rank
// This triggers lines 119-130 in expand.h
TEST_F(TensorExpandTest, ExpandSameRankTileFallback) {
  // Same rank but dimensions don't match, needs tile
  // {2,3}.expand({2,6}) - dim 0 matches (2), dim 1 needs expand (3->6)
  at::Tensor t = at::full({2, 3}, 2.0f, at::kFloat);
  at::Tensor result = t.expand({2, 6});

  ASSERT_EQ(result.sizes()[0], 2);
  ASSERT_EQ(result.sizes()[1], 6);
  ASSERT_FLOAT_EQ(result.data_ptr<float>()[0], 2.0f);
  ASSERT_FLOAT_EQ(result.data_ptr<float>()[3], 2.0f);
}

// Test zero dimension handling
// This triggers lines 90-94 and 122-126 in expand.h
TEST_F(TensorExpandTest, ExpandZeroDim) {
  at::Tensor t = at::full({0}, 1.0f, at::kFloat);
  at::Tensor result = t.expand({0, 3});

  ASSERT_EQ(result.sizes()[0], 0);
  ASSERT_EQ(result.sizes()[1], 3);
}

// Test input_rank > target_rank branch
// This triggers lines 131-136 in expand.h
TEST_F(TensorExpandTest, ExpandHighRankToLowRank) {
  // Input has more dimensions than target
  at::Tensor t = at::full({2, 3, 4}, 1.0f, at::kFloat);
  at::Tensor result = t.expand({3, 4});

  ASSERT_EQ(result.sizes()[0], 3);
  ASSERT_EQ(result.sizes()[1], 4);
}

// Test expand_as with tile fallback
TEST_F(TensorExpandTest, ExpandAsTileFallback) {
  // {2,1}.expand_as({1,4}) should trigger tile fallback
  at::Tensor t = at::full({2, 1}, 3.0f, at::kFloat);
  at::Tensor other = at::zeros({1, 4}, at::kFloat);

  at::Tensor result = t.expand_as(other);

  ASSERT_EQ(result.sizes()[0], 1);
  ASSERT_EQ(result.sizes()[1], 4);
  ASSERT_FLOAT_EQ(result.data_ptr<float>()[0], 3.0f);
}

// Test preserve non-singleton dimension (matching dimension)
TEST_F(TensorExpandTest, ExpandPreserveNonSingleton) {
  // {3,1}.expand({3,4}) - dim 0 matches (3), dim 1 expands (1->4)
  at::Tensor t = at::full({3, 1}, 5.0f, at::kFloat);
  at::Tensor result = t.expand({3, 4});

  ASSERT_EQ(result.sizes()[0], 3);
  ASSERT_EQ(result.sizes()[1], 4);
  ASSERT_FLOAT_EQ(result.data_ptr<float>()[0], 5.0f);
  ASSERT_FLOAT_EQ(result.data_ptr<float>()[3], 5.0f);
}

// Test expand function (not member function)
TEST_F(TensorExpandTest, ExpandFunction) {
  at::Tensor t = at::full({1}, 7.0f, at::kFloat);

  at::Tensor result = at::expand(t, {3, 4});

  ASSERT_EQ(result.sizes()[0], 3);
  ASSERT_EQ(result.sizes()[1], 4);
  ASSERT_FLOAT_EQ(result.data_ptr<float>()[0], 7.0f);
}

// Test expand_as function (not member function)
TEST_F(TensorExpandTest, ExpandAsFunction) {
  at::Tensor t = at::full({1, 2}, 4.0f, at::kFloat);
  at::Tensor other = at::zeros({3, 2}, at::kFloat);

  at::Tensor result = at::expand_as(t, other);

  ASSERT_EQ(result.sizes()[0], 3);
  ASSERT_EQ(result.sizes()[1], 2);
  ASSERT_FLOAT_EQ(result.data_ptr<float>()[0], 4.0f);
}
