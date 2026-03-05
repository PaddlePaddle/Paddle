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

// ======================== chunk tests ========================

TEST(TensorChunkTest, ChunkBasic) {
  at::Tensor t = at::arange(12, at::kFloat).reshape({3, 4});

  std::vector<at::Tensor> chunks = t.chunk(3, 0);

  ASSERT_EQ(chunks.size(), 3);
  ASSERT_EQ(chunks[0].size(0), 1);
  ASSERT_EQ(chunks[1].size(0), 1);
  ASSERT_EQ(chunks[2].size(0), 1);
}

TEST(TensorChunkTest, ChunkDim1) {
  at::Tensor t = at::arange(12, at::kFloat).reshape({3, 4});

  std::vector<at::Tensor> chunks = t.chunk(2, 1);

  ASSERT_EQ(chunks.size(), 2);
  ASSERT_EQ(chunks[0].size(1), 2);
  ASSERT_EQ(chunks[1].size(1), 2);
}

TEST(TensorChunkTest, ChunkUneven) {
  at::Tensor t = at::arange(10, at::kFloat).reshape({2, 5});

  std::vector<at::Tensor> chunks = t.chunk(3, 1);

  ASSERT_EQ(chunks.size(), 3);
}

TEST(TensorChunkTest, ChunkMoreChunksThanSize) {
  at::Tensor t = at::arange(6, at::kFloat).reshape({2, 3});

  std::vector<at::Tensor> chunks = t.chunk(5, 0);

  ASSERT_EQ(chunks.size(), 5);
}

TEST(TensorChunkTest, ChunkDefaultDim) {
  at::Tensor t = at::arange(12, at::kFloat).reshape({3, 4});

  std::vector<at::Tensor> chunks = t.chunk(3);

  ASSERT_EQ(chunks.size(), 3);
  ASSERT_EQ(chunks[0].size(0), 1);
}

TEST(TensorChunkTest, ChunkIntType) {
  at::Tensor t = at::arange(12, at::kInt).reshape({3, 4});

  std::vector<at::Tensor> chunks = t.chunk(3, 0);

  ASSERT_EQ(chunks.size(), 3);
  ASSERT_EQ(chunks[0].dtype(), at::kInt);
}
