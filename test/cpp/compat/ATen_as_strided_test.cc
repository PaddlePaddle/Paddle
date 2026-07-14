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
#include "torch/all.h"

COMMON_DECLARE_bool(use_stride_kernel);

namespace {

class TensorAsStridedTest : public ::testing::Test {};

}  // namespace

TEST_F(TensorAsStridedTest, AsStridedBasic) {
  // shape {2,3}, stride {3,1}: [[0,1,2],[3,4,5]]
  at::Tensor t = at::arange(12, at::kFloat);
  at::Tensor result = t.as_strided({2, 3}, {3, 1});

  ASSERT_EQ(result.sizes(), c10::IntArrayRef({2, 3}));
  float* data = result.data_ptr<float>();
  ASSERT_FLOAT_EQ(data[0], 0.0f);
  ASSERT_FLOAT_EQ(data[1], 1.0f);
  ASSERT_FLOAT_EQ(data[5], 5.0f);
}

TEST_F(TensorAsStridedTest, AsStridedWithOffset) {
  // offset=2: [[2,3,4],[5,6,7]]
  at::Tensor t = at::arange(12, at::kFloat);
  at::Tensor result = t.as_strided({2, 3}, {3, 1}, 2);

  ASSERT_EQ(result.sizes(), c10::IntArrayRef({2, 3}));
  float* data = result.data_ptr<float>();
  ASSERT_FLOAT_EQ(data[5], 7.0f);
}

TEST_F(TensorAsStridedTest, AsStridedWithDifferentStrides) {
  // shape {4,2}, stride {2,1}: [[0,1],[2,3],[4,5],[6,7]]
  at::Tensor t = at::arange(12, at::kFloat);
  at::Tensor result = t.as_strided({4, 2}, {2, 1});

  ASSERT_EQ(result.sizes(), c10::IntArrayRef({4, 2}));
  float* data = result.data_ptr<float>();
  ASSERT_FLOAT_EQ(data[0], 0.0f);
  ASSERT_FLOAT_EQ(data[7], 7.0f);
}

TEST_F(TensorAsStridedTest, AsStridedInplace) {
  // inplace: shape {12} -> {2,6}
  at::Tensor t = at::arange(12, at::kFloat);
  float* original_data_ptr = t.data_ptr<float>();

  t.as_strided_({2, 6}, {6, 1});

  ASSERT_EQ(t.sizes(), c10::IntArrayRef({2, 6}));
  ASSERT_EQ(t.data_ptr<float>(), original_data_ptr);

  float* data = t.data_ptr<float>();
  ASSERT_FLOAT_EQ(data[0], 0.0f);
  ASSERT_FLOAT_EQ(data[11], 11.0f);
}

TEST_F(TensorAsStridedTest, AsStridedInplaceWithOffset) {
  // inplace with offset=1: [[1,2,3],[4,5,6]]
  at::Tensor t = at::arange(12, at::kFloat);
  float* original_data_ptr = t.data_ptr<float>();

  t.as_strided_({2, 3}, {3, 1}, 1);

  ASSERT_EQ(t.sizes(), c10::IntArrayRef({2, 3}));
  ASSERT_NE(t.data_ptr<float>(), original_data_ptr);

  float* data = t.data_ptr<float>();
  ASSERT_FLOAT_EQ(data[0], 1.0f);
  ASSERT_FLOAT_EQ(data[5], 6.0f);
}

TEST_F(TensorAsStridedTest, AsStridedInplaceModifiesView) {
  // Modify view, verify original is affected
  at::Tensor t = at::arange(12, at::kFloat);
  at::Tensor view = t.as_strided({2, 3}, {3, 1});

  view.data_ptr<float>()[0] = 99.0f;
  ASSERT_FLOAT_EQ(t.data_ptr<float>()[0], 99.0f);
}

TEST_F(TensorAsStridedTest, AsStridedScatterBasic) {
  // Scatter 2x3 99s into t: [[99,99,99],[99,99,99],...]]
  at::Tensor t = at::arange(12, at::kFloat);
  at::Tensor src = at::full({2, 3}, 99.0f, at::kFloat);
  at::Tensor result = t.as_strided_scatter(src, {2, 3}, {3, 1});

  ASSERT_EQ(result.sizes(), c10::IntArrayRef({12}));
  float* data = result.data_ptr<float>();
  for (int i = 0; i < 6; ++i) {
    ASSERT_FLOAT_EQ(data[i], 99.0f);
  }
}

TEST_F(TensorAsStridedTest, AsStridedScatterOriginalUnchanged) {
  // Scatter returns new tensor, original unchanged
  at::Tensor t = at::arange(12, at::kFloat);
  at::Tensor src = at::full({2, 3}, 99.0f, at::kFloat);
  at::Tensor result = t.as_strided_scatter(src, {2, 3}, {3, 1});

  ASSERT_FLOAT_EQ(t.data_ptr<float>()[0], 0.0f);
}

TEST_F(TensorAsStridedTest, AsStridedScatterWithOffset) {
  // Scatter with offset=2: [[88,88],[88,88]]
  at::Tensor t = at::arange(12, at::kFloat);
  at::Tensor src = at::full({2, 2}, 88.0f, at::kFloat);
  at::Tensor result = t.as_strided_scatter(src, {2, 2}, {2, 1}, 2);

  ASSERT_EQ(result.sizes(), c10::IntArrayRef({12}));
  float* data = result.data_ptr<float>();
  ASSERT_FLOAT_EQ(data[2], 88.0f);
  ASSERT_FLOAT_EQ(data[5], 88.0f);
}

TEST_F(TensorAsStridedTest, AsStridedTranspose) {
  if (!FLAGS_use_stride_kernel) {
    return;
  }
  // Transpose: shape {2,3} -> {3,2}, stride {1,2}
  // [[0,1,2],[3,4,5]] -> [[0,3],[1,4],[2,5]]
  at::Tensor t = at::arange(6, at::kFloat).view({2, 3});
  at::Tensor result = t.as_strided({3, 2}, {1, 2});

  ASSERT_EQ(result.sizes(), c10::IntArrayRef({3, 2}));
  float* data = result.data_ptr<float>();
  ASSERT_FLOAT_EQ(data[0], 0.0f);
  ASSERT_FLOAT_EQ(data[5], 5.0f);
}

TEST_F(TensorAsStridedTest, AsStridedContiguous) {
  if (!FLAGS_use_stride_kernel) {
    return;
  }
  at::Tensor t = at::arange(12, at::kFloat);

  // Contiguous: {2,6}, stride {6,1}
  at::Tensor contig = t.as_strided({2, 6}, {6, 1});
  ASSERT_TRUE(contig.is_contiguous());

  // Non-contiguous: {3,2}, stride {1,3}
  at::Tensor non_contig = t.as_strided({3, 2}, {1, 3});
  ASSERT_FALSE(non_contig.is_contiguous());
}
