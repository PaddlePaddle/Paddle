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
#include <ATen/indexing.h>
#include <ATen/ops/tensor.h>
#include <c10/core/List.h>
#include <c10/core/ScalarType.h>
#include <c10/core/TensorOptions.h>

#include "ATen/ATen.h"
#include "gtest/gtest.h"
#include "torch/all.h"

// ======================== index tests ========================

TEST(TensorIndexTest, IndexWithSingleTensor) {
  // Create tensor [0, 10, 20, 30, 40]
  at::Tensor t = at::arange(5, at::kFloat);
  for (int i = 0; i < 5; i++) {
    t.data_ptr<float>()[i] = static_cast<float>(i * 10);
  }

  // Index with [0, 2, 4]
  at::Tensor idx = at::empty({3}, at::kLong);
  int64_t* idx_data = idx.data_ptr<int64_t>();
  idx_data[0] = 0;
  idx_data[1] = 2;
  idx_data[2] = 4;

  c10::List<::std::optional<at::Tensor>> indices;
  indices.push_back(idx);

  at::Tensor result = t.index(indices);
  ASSERT_EQ(result.numel(), 3);

  float* result_data = result.data_ptr<float>();
  ASSERT_FLOAT_EQ(result_data[0], 0.0f);
  ASSERT_FLOAT_EQ(result_data[1], 20.0f);
  ASSERT_FLOAT_EQ(result_data[2], 40.0f);
}

TEST(TensorIndexTest, SliceKeepsStrideWithoutContiguousCopy) {
  at::Tensor base = at::arange(24, at::kFloat).reshape({4, 6});
  at::Tensor transposed = base.t();  // shape: [6, 4], strides: [1, 6]
  ASSERT_FALSE(transposed.is_contiguous());

  at::Tensor sliced =
      transposed.index({at::indexing::Slice(1, 5), at::indexing::Slice(0, 3)});

  ASSERT_EQ(sliced.sizes(), c10::IntArrayRef({4, 3}));
  ASSERT_EQ(sliced.strides(), c10::IntArrayRef({1, 6}));
  ASSERT_EQ(sliced.stride(0), transposed.stride(0));
  ASSERT_EQ(sliced.stride(1), transposed.stride(1));
  ASSERT_FALSE(sliced.is_contiguous());
}

TEST(TensorIndexTest, IndexWithEmptyInitializerListReturnsSelf) {
  at::Tensor t = at::arange(5, at::kFloat);

  at::Tensor result =
      at::index(t, std::initializer_list<at::indexing::TensorIndex>{});

  ASSERT_EQ(result.numel(), t.numel());
  ASSERT_EQ(result.data_ptr<float>(), t.data_ptr<float>());
}

TEST(TensorIndexTest, IndexWithTensorInitializerList) {
  at::Tensor t = at::arange(5, at::kFloat);

  at::Tensor idx = at::empty({3}, at::kLong);
  int64_t* idx_data = idx.data_ptr<int64_t>();
  idx_data[0] = 0;
  idx_data[1] = 2;
  idx_data[2] = 4;

  at::Tensor result = at::index(t, {idx});

  ASSERT_EQ(result.numel(), 3);
  float* result_data = result.data_ptr<float>();
  ASSERT_FLOAT_EQ(result_data[0], 0.0f);
  ASSERT_FLOAT_EQ(result_data[1], 2.0f);
  ASSERT_FLOAT_EQ(result_data[2], 4.0f);
}

TEST(TensorIndexTest, MemberIndexWithArrayRefTensorIndices) {
  at::Tensor base = at::arange(24, at::kFloat).reshape({4, 6});
  at::Tensor transposed = base.t();
  std::vector<at::indexing::TensorIndex> indices = {at::indexing::Slice(1, 5),
                                                    at::indexing::Slice(0, 3)};

  at::Tensor sliced = transposed.index(indices);

  ASSERT_EQ(sliced.sizes(), c10::IntArrayRef({4, 3}));
  ASSERT_EQ(sliced.strides(), c10::IntArrayRef({1, 6}));
}

TEST(TensorIndexTest, MixedSliceAndTensorIndicesThrows) {
  at::Tensor t = at::arange(12, at::kFloat).reshape({3, 4});

  at::Tensor idx = at::empty({2}, at::kLong);
  idx.data_ptr<int64_t>()[0] = 0;
  idx.data_ptr<int64_t>()[1] = 2;

  ASSERT_THROW(at::index(t, {at::indexing::Slice(0, 2), idx}), std::exception);
}

// ======================== index_put_ tests ========================

TEST(TensorIndexPutTest, IndexPutInplaceWithTensor) {
  at::Tensor t = at::zeros({5}, at::kFloat);
  float* original_data_ptr = t.data_ptr<float>();

  // Create index tensor [1, 3]
  at::Tensor idx = at::empty({2}, at::kLong);
  int64_t* idx_data = idx.data_ptr<int64_t>();
  idx_data[0] = 1;
  idx_data[1] = 3;

  // Values to put
  at::Tensor values = at::full({2}, 99.0f, at::kFloat);

  c10::List<::std::optional<at::Tensor>> indices;
  indices.push_back(idx);

  t.index_put_(indices, values);

  // Verify data pointer unchanged (inplace)
  ASSERT_EQ(t.data_ptr<float>(), original_data_ptr);

  float* data = t.data_ptr<float>();
  ASSERT_FLOAT_EQ(data[0], 0.0f);
  ASSERT_FLOAT_EQ(data[1], 99.0f);
  ASSERT_FLOAT_EQ(data[2], 0.0f);
  ASSERT_FLOAT_EQ(data[3], 99.0f);
  ASSERT_FLOAT_EQ(data[4], 0.0f);
}

TEST(TensorIndexPutTest, IndexPutInplaceWithScalar) {
  at::Tensor t = at::zeros({5}, at::kFloat);
  float* original_data_ptr = t.data_ptr<float>();

  at::Tensor idx = at::empty({2}, at::kLong);
  int64_t* idx_data = idx.data_ptr<int64_t>();
  idx_data[0] = 0;
  idx_data[1] = 4;

  c10::List<::std::optional<at::Tensor>> indices;
  indices.push_back(idx);

  t.index_put_(indices, at::Scalar(7.0));

  // Verify data pointer unchanged (inplace)
  ASSERT_EQ(t.data_ptr<float>(), original_data_ptr);

  float* data = t.data_ptr<float>();
  ASSERT_FLOAT_EQ(data[0], 7.0f);
  ASSERT_FLOAT_EQ(data[1], 0.0f);
  ASSERT_FLOAT_EQ(data[4], 7.0f);
}

TEST(TensorIndexPutTest, IndexPutNonInplace) {
  at::Tensor t = at::zeros({5}, at::kFloat);

  at::Tensor idx = at::empty({2}, at::kLong);
  int64_t* idx_data = idx.data_ptr<int64_t>();
  idx_data[0] = 1;
  idx_data[1] = 3;

  at::Tensor values = at::full({2}, 42.0f, at::kFloat);

  c10::List<::std::optional<at::Tensor>> indices;
  indices.push_back(idx);

  at::Tensor result = t.index_put(indices, values);

  // Original should be unchanged
  ASSERT_FLOAT_EQ(t.data_ptr<float>()[1], 0.0f);

  // Result should have the values
  float* rdata = result.data_ptr<float>();
  ASSERT_FLOAT_EQ(rdata[1], 42.0f);
  ASSERT_FLOAT_EQ(rdata[3], 42.0f);
}

// ======================= Additional index edge case tests
// =======================

TEST(TensorIndexTest, IndexWithEmptyList) {
  // Test index with empty indices list (should return self)
  at::Tensor t = at::arange(5, at::kFloat);
  c10::List<::std::optional<at::Tensor>> indices;

  at::Tensor result = t.index(indices);
  ASSERT_EQ(result.numel(), 5);
}

TEST(TensorIndexTest, IndexWithMultipleIndices) {
  // Test index with multiple indices (2D indexing)
  at::Tensor t = at::arange(9, at::kFloat).reshape({3, 3});

  at::Tensor idx0 = at::empty({2}, at::kLong);
  int64_t* idx0_data = idx0.data_ptr<int64_t>();
  idx0_data[0] = 0;
  idx0_data[1] = 1;

  at::Tensor idx1 = at::empty({2}, at::kLong);
  int64_t* idx1_data = idx1.data_ptr<int64_t>();
  idx1_data[0] = 0;
  idx1_data[1] = 2;

  c10::List<::std::optional<at::Tensor>> indices;
  indices.push_back(idx0);
  indices.push_back(idx1);

  at::Tensor result = t.index(indices);
  ASSERT_EQ(result.numel(), 2);
}

TEST(TensorIndexTest, IndexWithOptionalNone) {
  // Test index with optional None in indices
  // None means "select all" along that dimension
  at::Tensor t = at::arange(9, at::kFloat).reshape({3, 3});

  at::Tensor idx = at::empty({2}, at::kLong);
  idx.data_ptr<int64_t>()[0] = 0;
  idx.data_ptr<int64_t>()[1] = 2;

  c10::List<::std::optional<at::Tensor>> indices;
  indices.push_back(::std::nullopt);  // None = select all rows
  indices.push_back(idx);             // [0, 2] = select columns 0 and 2

  at::Tensor result = t.index(indices);
  // Result should be shape {3, 2} = 6 elements
  // Columns 0 and 2 from all rows: [[0,2], [3,5], [6,8]]
  ASSERT_EQ(result.numel(), 6);
}

TEST(TensorIndexPutTest, IndexPutAccumulate) {
  // Test index_put_ with accumulate=true
  at::Tensor t = at::zeros({5}, at::kFloat);
  float* original_data_ptr = t.data_ptr<float>();

  at::Tensor idx = at::empty({2}, at::kLong);
  idx.data_ptr<int64_t>()[0] = 1;
  idx.data_ptr<int64_t>()[1] = 1;

  at::Tensor values = at::full({2}, 5.0f, at::kFloat);

  c10::List<::std::optional<at::Tensor>> indices;
  indices.push_back(idx);

  t.index_put_(indices, values, true);  // accumulate=true

  // Verify data pointer unchanged (inplace)
  ASSERT_EQ(t.data_ptr<float>(), original_data_ptr);

  float* data = t.data_ptr<float>();
  ASSERT_FLOAT_EQ(data[0], 0.0f);
  ASSERT_FLOAT_EQ(data[1], 10.0f);  // 5 + 5 (accumulated)
  ASSERT_FLOAT_EQ(data[2], 0.0f);
}

TEST(TensorIndexPutTest, IndexPutWith2D) {
  // Test index_put_ with 2D tensor
  at::Tensor t = at::zeros({3, 3}, at::kFloat);
  float* original_data_ptr = t.data_ptr<float>();

  at::Tensor idx0 = at::arange(2, at::kLong);
  idx0.data_ptr<int64_t>()[0] = 0;
  idx0.data_ptr<int64_t>()[1] = 1;
  at::Tensor idx1 = at::arange(2, at::kLong);
  idx1.data_ptr<int64_t>()[0] = 0;
  idx1.data_ptr<int64_t>()[1] = 1;

  c10::List<::std::optional<at::Tensor>> indices;
  indices.push_back(idx0);
  indices.push_back(idx1);

  at::Tensor values = at::full({2}, 9.0f, at::kFloat);

  t.index_put_(indices, values);

  // Verify data pointer unchanged (inplace)
  ASSERT_EQ(t.data_ptr<float>(), original_data_ptr);

  float* data = t.data_ptr<float>();
  ASSERT_FLOAT_EQ(data[0], 9.0f);  // [0,0]
  ASSERT_FLOAT_EQ(data[4], 9.0f);  // [1,1]
}

TEST(TensorIndexPutTest, IndexPutNonInplaceAccumulate) {
  // Test index_put with accumulate=true (non-inplace)
  at::Tensor t = at::zeros({5}, at::kFloat);

  at::Tensor idx = at::empty({2}, at::kLong);
  idx.data_ptr<int64_t>()[0] = 1;
  idx.data_ptr<int64_t>()[1] = 1;
  at::Tensor values = at::full({2}, 3.0f, at::kFloat);

  c10::List<::std::optional<at::Tensor>> indices;
  indices.push_back(idx);

  at::Tensor result = t.index_put(indices, values, true);

  // Original unchanged
  ASSERT_FLOAT_EQ(t.data_ptr<float>()[1], 0.0f);
  // Result has accumulated
  ASSERT_FLOAT_EQ(result.data_ptr<float>()[1], 6.0f);
}
