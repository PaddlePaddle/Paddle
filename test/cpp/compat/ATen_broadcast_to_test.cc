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
#include "paddle/common/macros.h"
#include "test/cpp/prim/init_env_utils.h"
#include "torch/all.h"

COMMON_DECLARE_bool(use_stride_kernel);

namespace {

class TensorBroadcastToTest : public ::testing::Test {
 protected:
  static void SetUpTestSuite() { paddle::prim::InitTensorOperants(); }
};

}  // namespace

// Helper: compute linear offset from logical index using strides
static inline int64_t compute_offset(int64_t flat_idx,
                                     const at::Tensor& tensor) {
  int64_t offset = 0;
  int64_t remainder = flat_idx;
  for (int64_t d = tensor.dim() - 1; d >= 0; --d) {
    int64_t coord = remainder % tensor.sizes()[d];
    remainder /= tensor.sizes()[d];
    offset += coord * tensor.strides()[d];
  }
  return offset;
}

// ======================== broadcast_to tests ========================

TEST_F(TensorBroadcastToTest, BroadcastToBasic) {
  at::Tensor t = at::full({1, 3}, 5.0f, at::kFloat);

  at::Tensor result = t.broadcast_to({2, 3});

  ASSERT_EQ(result.sizes()[0], 2);
  ASSERT_EQ(result.sizes()[1], 3);
}

TEST_F(TensorBroadcastToTest, BroadcastToSingleDim) {
  at::Tensor t = at::full({1}, 7.0f, at::kFloat);

  at::Tensor result = t.broadcast_to({5});

  ASSERT_EQ(result.numel(), 5);
}

TEST_F(TensorBroadcastToTest, BroadcastToMultipleDims) {
  at::Tensor t = at::full({1, 1}, 3.0f, at::kFloat);

  at::Tensor result = t.broadcast_to({2, 3});

  ASSERT_EQ(result.sizes()[0], 2);
  ASSERT_EQ(result.sizes()[1], 3);
}

TEST_F(TensorBroadcastToTest, BroadcastToPreservesValue) {
  if (!FLAGS_use_stride_kernel) {
    return;
  }
  at::Tensor t = at::empty({1, 2}, at::kFloat);
  float* input_data = t.data_ptr<float>();
  input_data[0] = 3.0f;
  input_data[1] = 7.0f;

  at::Tensor result = t.broadcast_to({3, 2});

  ASSERT_EQ(result.strides()[0], 0);
  float* data = result.data_ptr<float>();
  ASSERT_FLOAT_EQ(data[compute_offset(0, result)], 3.0f);  // [0,0]
  ASSERT_FLOAT_EQ(data[compute_offset(1, result)], 7.0f);  // [0,1]
  ASSERT_FLOAT_EQ(data[compute_offset(2, result)], 3.0f);  // [1,0]
  ASSERT_FLOAT_EQ(data[compute_offset(5, result)], 7.0f);  // [2,1]
}

TEST_F(TensorBroadcastToTest, BroadcastToRankLess) {
  at::Tensor t = at::full({1}, 4.0f, at::kFloat);

  at::Tensor result = t.broadcast_to({2, 3});

  ASSERT_EQ(result.sizes()[0], 2);
  ASSERT_EQ(result.sizes()[1], 3);
}

TEST_F(TensorBroadcastToTest, BroadcastToSameRank) {
  at::Tensor t = at::full({3, 1}, 6.0f, at::kFloat);

  at::Tensor result = t.broadcast_to({3, 4});

  ASSERT_EQ(result.sizes()[0], 3);
  ASSERT_EQ(result.sizes()[1], 4);
}

TEST_F(TensorBroadcastToTest, BroadcastToNonSingletonMatch) {
  at::Tensor t = at::full({2, 1}, 8.0f, at::kFloat);

  at::Tensor result = t.broadcast_to({2, 3});

  ASSERT_EQ(result.sizes()[0], 2);
  ASSERT_EQ(result.sizes()[1], 3);
}

TEST_F(TensorBroadcastToTest, BroadcastToFunction) {
  if (!FLAGS_use_stride_kernel) {
    return;
  }
  at::Tensor t = at::full({1, 2}, 2.0f, at::kFloat);

  at::Tensor result = at::broadcast_to(t, {3, 2});

  ASSERT_EQ(result.sizes()[0], 3);
  ASSERT_EQ(result.sizes()[1], 2);
  float* data = result.data_ptr<float>();
  ASSERT_FLOAT_EQ(data[compute_offset(0, result)], 2.0f);  // [0,0]
}

TEST_F(TensorBroadcastToTest, BroadcastToZeroDim) {
  at::Tensor t = at::full({1, 0}, 1.0f, at::kFloat);

  at::Tensor result = t.broadcast_to({2, 0});

  ASSERT_EQ(result.sizes()[0], 2);
  ASSERT_EQ(result.sizes()[1], 0);
}

TEST_F(TensorBroadcastToTest, BroadcastToInvalidNonSingleton) {
  at::Tensor t = at::full({2, 3}, 1.0f, at::kFloat);

  ASSERT_THROW(t.broadcast_to({2, 4}), std::exception);
}

TEST_F(TensorBroadcastToTest, BroadcastToHighRankToLowRank) {
  at::Tensor t = at::full({2, 3, 4}, 1.0f, at::kFloat);

  ASSERT_THROW(t.broadcast_to({3, 4}), std::exception);
}

TEST_F(TensorBroadcastToTest, BroadcastToSymInt) {
  if (!FLAGS_use_stride_kernel) {
    return;
  }
  at::Tensor t = at::full({1, 3}, 5.0f, at::kFloat);

  std::vector<c10::SymInt> sym_sizes = {c10::SymInt(2), c10::SymInt(3)};
  at::Tensor result = t.broadcast_to_symint(c10::SymIntArrayRef(sym_sizes));

  ASSERT_EQ(result.sizes()[0], 2);
  ASSERT_EQ(result.sizes()[1], 3);
  float* data = result.data_ptr<float>();
  ASSERT_FLOAT_EQ(data[compute_offset(0, result)], 5.0f);  // [0,0]
}

// broadcast_to supports -1 (same as expand)
TEST_F(TensorBroadcastToTest, BroadcastToNegativeOne) {
  at::Tensor t = at::full({3}, 5.0f, at::kFloat);
  at::Tensor result = t.broadcast_to({-1});
  ASSERT_EQ(result.sizes()[0], 3);
}

// broadcast_to_symint supports -1 (same as expand)
TEST_F(TensorBroadcastToTest, BroadcastToSymIntNegativeOne) {
  at::Tensor t = at::full({3}, 5.0f, at::kFloat);
  std::vector<c10::SymInt> sym_sizes = {c10::SymInt(-1)};
  at::Tensor result = t.broadcast_to_symint(c10::SymIntArrayRef(sym_sizes));
  ASSERT_EQ(result.sizes()[0], 3);
}
