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

class TensorExpandTest : public ::testing::Test {
 protected:
  static void SetUpTestSuite() { paddle::prim::InitTensorOperants(); }
};

class UseStrideKernelGuard {
 public:
  explicit UseStrideKernelGuard(bool value)
      : previous_(FLAGS_use_stride_kernel) {
    FLAGS_use_stride_kernel = value;
  }

  ~UseStrideKernelGuard() { FLAGS_use_stride_kernel = previous_; }

 private:
  bool previous_;
};

}  // namespace

// ======================== expand tests ========================

TEST_F(TensorExpandTest, ExpandBasic) {
  // {3}.expand({3,4}) - PyTorch rejects non-singleton expansion (3 != 4)
  at::Tensor t = at::arange(3, at::kFloat);
  ASSERT_THROW(t.expand({3, 4}), std::exception);
}

TEST_F(TensorExpandTest, ExpandSingleDim) {
  at::Tensor t = at::full({1}, 5.0f, at::kFloat);

  at::Tensor result = t.expand({5});

  ASSERT_EQ(result.numel(), 5);
}

TEST_F(TensorExpandTest, ExpandMultipleDims) {
  // {1,3}.expand({2,3,4}) - PyTorch rejects non-singleton expansion (3 != 4)
  at::Tensor t = at::full({1, 3}, 1.0f, at::kFloat);
  ASSERT_THROW(t.expand({2, 3, 4}), std::exception);
}

TEST_F(TensorExpandTest, ExpandWithImplicit) {
  // {3}.expand({3,4}) - PyTorch rejects non-singleton expansion (3 != 4)
  at::Tensor t = at::arange(3, at::kFloat);
  ASSERT_THROW(t.expand({3, 4}, true), std::exception);
}

TEST_F(TensorExpandTest, ExpandPreservesValue) {
  // {3}.expand({3,4}) - PyTorch rejects non-singleton expansion (3 != 4)
  at::Tensor t = at::full({3}, 7.0f, at::kFloat);
  ASSERT_THROW(t.expand({3, 4}), std::exception);
}

// Test scalar expand (tensor_dim == 0 in compute_expand_strides)
TEST_F(TensorExpandTest, ExpandScalar) {
  at::Tensor t = at::full({}, 5.0f, at::kFloat);

  at::Tensor result = t.expand({2, 3});

  ASSERT_EQ(result.sizes()[0], 2);
  ASSERT_EQ(result.sizes()[1], 3);
}

// Test target_size == -1 (keep original size)
TEST_F(TensorExpandTest, ExpandNegativeOne) {
  at::Tensor t = at::full({3}, 7.0f, at::kFloat);

  at::Tensor result = t.expand({-1});

  ASSERT_EQ(result.sizes()[0], 3);
}

// Test target_size == -1 in leading non-existing dimension (error)
TEST_F(TensorExpandTest, ExpandNegativeOneLeadingError) {
  at::Tensor t = at::full({3}, 7.0f, at::kFloat);

  ASSERT_THROW(t.expand({-1, 4}), std::exception);
}

TEST_F(TensorExpandTest, ExpandMaterializedWhenStrideKernelDisabled) {
  UseStrideKernelGuard guard(false);
  at::Tensor t = at::empty({1, 2}, at::kFloat);
  float* input_data = t.data_ptr<float>();
  input_data[0] = 3.0f;
  input_data[1] = 7.0f;

  at::Tensor result = t.expand({3, 2});

  ASSERT_EQ(result.sizes()[0], 3);
  ASSERT_EQ(result.sizes()[1], 2);
  ASSERT_NE(result.strides()[0], 0);
  float* data = result.data_ptr<float>();
  ASSERT_FLOAT_EQ(data[0], 3.0f);
  ASSERT_FLOAT_EQ(data[1], 7.0f);
  ASSERT_FLOAT_EQ(data[2], 3.0f);
  ASSERT_FLOAT_EQ(data[5], 7.0f);
}

TEST_F(TensorExpandTest, ExpandLowRankMaterializedWhenStrideKernelDisabled) {
  UseStrideKernelGuard guard(false);
  at::Tensor t = at::full({1}, 5.0f, at::kFloat);

  at::Tensor result = t.expand({2, 3});

  ASSERT_EQ(result.sizes()[0], 2);
  ASSERT_EQ(result.sizes()[1], 3);
  ASSERT_NE(result.strides()[0], 0);
  float* data = result.data_ptr<float>();
  ASSERT_FLOAT_EQ(data[0], 5.0f);
  ASSERT_FLOAT_EQ(data[5], 5.0f);
}

TEST_F(TensorExpandTest, ExpandInvalidStillThrowsWhenStrideKernelDisabled) {
  UseStrideKernelGuard guard(false);
  at::Tensor t = at::full({2, 3}, 1.0f, at::kFloat);

  ASSERT_THROW(t.expand({2, 4}), std::exception);
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
  if (!FLAGS_use_stride_kernel) {
    return;
  }
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
  // {2,1}.expand({1,4}) - PyTorch rejects shrinking non-singleton dims
  at::Tensor t = at::full({2, 1}, 1.0f, at::kFloat);
  ASSERT_THROW(t.expand({1, 4}), std::exception);
}

// Test tile fallback when input_rank == target_rank
// This triggers lines 119-130 in expand.h
TEST_F(TensorExpandTest, ExpandSameRankTileFallback) {
  // {2,3}.expand({2,6}) - PyTorch only allows expanding singleton dims
  at::Tensor t = at::full({2, 3}, 2.0f, at::kFloat);
  ASSERT_THROW(t.expand({2, 6}), std::exception);
}

// Test zero dimension handling
// This triggers lines 90-94 and 122-126 in expand.h
TEST_F(TensorExpandTest, ExpandZeroDim) {
  // {0}.expand({0,3}) - PyTorch rejects non-singleton expansion (0 != 3)
  at::Tensor t = at::full({0}, 1.0f, at::kFloat);
  ASSERT_THROW(t.expand({0, 3}), std::exception);
}

// Test input_rank > target_rank branch
// This triggers lines 131-136 in expand.h
TEST_F(TensorExpandTest, ExpandHighRankToLowRank) {
  // Input has more dimensions than target - PyTorch rejects this
  at::Tensor t = at::full({2, 3, 4}, 1.0f, at::kFloat);
  ASSERT_THROW(t.expand({3, 4}), std::exception);
}

// Test expand_as with tile fallback
TEST_F(TensorExpandTest, ExpandAsTileFallback) {
  // {2,1}.expand_as({1,4}) - PyTorch rejects shrinking non-singleton dims
  at::Tensor t = at::full({2, 1}, 3.0f, at::kFloat);
  at::Tensor other = at::zeros({1, 4}, at::kFloat);

  ASSERT_THROW(t.expand_as(other), std::exception);
}

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

// Test preserve non-singleton dimension (matching dimension)
TEST_F(TensorExpandTest, ExpandPreserveNonSingleton) {
  if (!FLAGS_use_stride_kernel) {
    return;
  }
  // {3,1}.expand({3,4}) - dim 0 matches (3), dim 1 expands (1->4)
  at::Tensor t = at::full({3, 1}, 5.0f, at::kFloat);
  at::Tensor result = t.expand({3, 4});

  ASSERT_EQ(result.sizes()[0], 3);
  ASSERT_EQ(result.sizes()[1], 4);
  // Use strides-aware access because expand returns a view with stride=0
  float* data = result.data_ptr<float>();
  ASSERT_FLOAT_EQ(data[compute_offset(0, result)], 5.0f);  // [0,0]
  ASSERT_FLOAT_EQ(data[compute_offset(3, result)], 5.0f);  // [0,3]
}

// Test expand function (not member function)
TEST_F(TensorExpandTest, ExpandFunction) {
  if (!FLAGS_use_stride_kernel) {
    return;
  }
  at::Tensor t = at::full({1}, 7.0f, at::kFloat);

  at::Tensor result = at::expand(t, {3, 4});

  ASSERT_EQ(result.sizes()[0], 3);
  ASSERT_EQ(result.sizes()[1], 4);
  ASSERT_FLOAT_EQ(result.data_ptr<float>()[0], 7.0f);
}

TEST_F(TensorExpandTest, ExpandAsMemberFunction) {
  if (!FLAGS_use_stride_kernel) {
    return;
  }
  at::Tensor t = at::full({1, 2}, 4.0f, at::kFloat);
  at::Tensor other = at::zeros({3, 2}, at::kFloat);

  at::Tensor result = t.expand_as(other);

  ASSERT_EQ(result.sizes()[0], 3);
  ASSERT_EQ(result.sizes()[1], 2);
  ASSERT_FLOAT_EQ(result.data_ptr<float>()[0], 4.0f);
}
