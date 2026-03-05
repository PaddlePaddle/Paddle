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
#include "gtest/gtest.h"

TEST(TensorIndexTest, SliceKeepsStrideWithoutContiguousCopy) {
  at::Tensor base = at::arange(24, at::kFloat).reshape({4, 6});
  at::Tensor transposed = base.t();  // shape: [6, 4], strides: [1, 6]
  ASSERT_FALSE(transposed.is_contiguous());

  at::Tensor sliced = transposed.index(
      {at::indexing::Slice(1, 5), at::indexing::Slice(0, 3)});

  ASSERT_EQ(sliced.sizes(), std::vector<int64_t>({4, 3}));
  ASSERT_EQ(sliced.strides(), std::vector<int64_t>({1, 6}));
  ASSERT_EQ(sliced.stride(0), transposed.stride(0));
  ASSERT_EQ(sliced.stride(1), transposed.stride(1));
  ASSERT_FALSE(sliced.is_contiguous());
}
