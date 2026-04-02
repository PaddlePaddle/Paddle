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
#include <ATen/cuda/EmptyTensor.h>
#include <ATen/native/cuda/Resize.h>
#include <ATen/ops/tensor.h>
#include <c10/core/Layout.h>
#include <c10/core/ScalarType.h>
#include <c10/core/SymInt.h>
#include <c10/core/TensorOptions.h>
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
#include <c10/cuda/CUDAFunctions.h>
#include <c10/cuda/CUDAGuard.h>
#endif
#include "ATen/ATen.h"
#include "gtest/gtest.h"
#include "paddle/phi/common/float16.h"
#include "test/cpp/compat/cuda_test_utils.h"
#include "torch/all.h"

// Test for tensor_split with sections
TEST(TensorSplitTest, TensorSplitWithSections) {
  // Create a test tensor [0, 1, 2, ..., 8] (9 elements, evenly divisible by 3)
  auto tensor = at::arange(9, at::TensorOptions().dtype(at::kFloat));

  // Split into 3 sections along dim 0
  auto splits = tensor.tensor_split(3, 0);

  EXPECT_EQ(splits.size(), 3);
  EXPECT_EQ(splits[0].numel(), 3);  // [0, 1, 2]
  EXPECT_EQ(splits[1].numel(), 3);  // [3, 4, 5]
  EXPECT_EQ(splits[2].numel(), 3);  // [6, 7, 8]

  // Verify first split values
  EXPECT_FLOAT_EQ(splits[0][0].item<float>(), 0.0f);
  EXPECT_FLOAT_EQ(splits[0][2].item<float>(), 2.0f);
}

// Test for tensor_split with indices (PyTorch semantics: indices are positions)
TEST(TensorSplitTest, TensorSplitWithIndices) {
  auto tensor = at::arange(10, at::TensorOptions().dtype(at::kFloat));

  // Split at indices [2, 3, 5] -> [0:2], [2:3], [3:5], [5:10]
  std::vector<int64_t> indices = {2, 3, 5};
  auto splits = tensor.tensor_split(at::IntArrayRef(indices), 0);

  EXPECT_EQ(splits.size(), 4);      // 4 segments
  EXPECT_EQ(splits[0].numel(), 2);  // [0, 1]
  EXPECT_EQ(splits[1].numel(), 1);  // [2]
  EXPECT_EQ(splits[2].numel(), 2);  // [3, 4]
  EXPECT_EQ(splits[3].numel(), 5);  // [5, 6, 7, 8, 9]
}

// Test for tensor_split with tensor indices (1D tensor = indices)
TEST(TensorSplitTest, TensorSplitWithTensorIndices) {
  auto tensor = at::arange(10, at::TensorOptions().dtype(at::kFloat));

  // Create 1D tensor with indices [3, 7] -> split at positions 3 and 7
  // Result: [0:3], [3:7], [7:10] -> sizes: 3, 4, 3
  auto indices_tensor = at::empty({2}, at::TensorOptions().dtype(at::kLong));
  indices_tensor.data_ptr<int64_t>()[0] = 3;
  indices_tensor.data_ptr<int64_t>()[1] = 7;
  auto splits = tensor.tensor_split(indices_tensor, 0);

  EXPECT_EQ(splits.size(), 3);      // 3 segments
  EXPECT_EQ(splits[0].numel(), 3);  // [0, 1, 2]
  EXPECT_EQ(splits[1].numel(), 4);  // [3, 4, 5, 6]
  EXPECT_EQ(splits[2].numel(), 3);  // [7, 8, 9]
}

// Test for tensor_split with tensor scalar (0D tensor = number of sections)
TEST(TensorSplitTest, TensorSplitWithTensorScalar) {
  auto tensor = at::arange(9, at::TensorOptions().dtype(at::kFloat));

  // Create 0D scalar tensor with value 3 -> split into 3 sections
  auto sections_tensor = at::empty({}, at::TensorOptions().dtype(at::kLong));
  sections_tensor.data_ptr<int64_t>()[0] = 3;
  auto splits = tensor.tensor_split(sections_tensor, 0);

  EXPECT_EQ(splits.size(), 3);
  EXPECT_EQ(splits[0].numel(), 3);  // [0, 1, 2]
  EXPECT_EQ(splits[1].numel(), 3);  // [3, 4, 5]
  EXPECT_EQ(splits[2].numel(), 3);  // [6, 7, 8]
}

// Test for split with split_size
TEST(SplitTest, SplitWithSize) {
  auto tensor = at::arange(10, at::TensorOptions().dtype(at::kFloat));

  // Split with size 3
  auto splits = tensor.split(3, 0);

  EXPECT_EQ(splits.size(), 4);
  EXPECT_EQ(splits[0].numel(), 3);  // [0, 1, 2]
  EXPECT_EQ(splits[1].numel(), 3);  // [3, 4, 5]
  EXPECT_EQ(splits[2].numel(), 3);  // [6, 7, 8]
  EXPECT_EQ(splits[3].numel(), 1);  // [9]
}

// Test for split with split_sizes
TEST(SplitTest, SplitWithSizes) {
  auto tensor = at::arange(10, at::TensorOptions().dtype(at::kFloat));

  // Split with sizes [2, 3, 5]
  std::vector<int64_t> sizes = {2, 3, 5};
  auto splits = tensor.split(at::IntArrayRef(sizes), 0);

  EXPECT_EQ(splits.size(), 3);
  EXPECT_EQ(splits[0].numel(), 2);  // [0, 1]
  EXPECT_EQ(splits[1].numel(), 3);  // [2, 3, 4]
  EXPECT_EQ(splits[2].numel(), 5);  // [5, 6, 7, 8, 9]
}

// Test for split_with_sizes
TEST(SplitTest, SplitWithSizesMethod) {
  auto tensor = at::arange(12, at::TensorOptions().dtype(at::kFloat));

  std::vector<int64_t> sizes = {4, 4, 4};
  auto splits = tensor.split_with_sizes(at::IntArrayRef(sizes), 0);

  EXPECT_EQ(splits.size(), 3);
  for (size_t i = 0; i < splits.size(); ++i) {
    EXPECT_EQ(splits[i].numel(), 4);
  }
}

// Test for unsafe_split
TEST(SplitTest, UnsafeSplit) {
  auto tensor = at::arange(10, at::TensorOptions().dtype(at::kFloat));

  auto splits = tensor.unsafe_split(4, 0);

  EXPECT_EQ(splits.size(), 3);
  EXPECT_EQ(splits[0].numel(), 4);
  EXPECT_EQ(splits[1].numel(), 4);
  EXPECT_EQ(splits[2].numel(), 2);
}

// Test for unsafe_split_with_sizes
TEST(SplitTest, UnsafeSplitWithSizes) {
  auto tensor = at::arange(10, at::TensorOptions().dtype(at::kFloat));

  std::vector<int64_t> sizes = {3, 3, 4};
  auto splits = tensor.unsafe_split_with_sizes(at::IntArrayRef(sizes), 0);

  EXPECT_EQ(splits.size(), 3);
  EXPECT_EQ(splits[0].numel(), 3);
  EXPECT_EQ(splits[1].numel(), 3);
  EXPECT_EQ(splits[2].numel(), 4);
}

// Test for hsplit with 1D tensor
TEST(SplitTest, HSplit1D) {
  auto tensor = at::arange(6, at::TensorOptions().dtype(at::kFloat));

  // For 1D tensor, hsplit splits along dim 0
  auto splits = tensor.hsplit(3);

  EXPECT_EQ(splits.size(), 3);
  EXPECT_EQ(splits[0].numel(), 2);
  EXPECT_EQ(splits[1].numel(), 2);
  EXPECT_EQ(splits[2].numel(), 2);
}

// Test for hsplit with 2D tensor
TEST(SplitTest, HSplit2D) {
  auto tensor =
      at::arange(12, at::TensorOptions().dtype(at::kFloat)).reshape({3, 4});

  // For 2D tensor, hsplit splits along dim 1
  auto splits = tensor.hsplit(2);

  EXPECT_EQ(splits.size(), 2);
  EXPECT_EQ(splits[0].size(0), 3);
  EXPECT_EQ(splits[0].size(1), 2);
  EXPECT_EQ(splits[1].size(0), 3);
  EXPECT_EQ(splits[1].size(1), 2);
}

// Test for hsplit with indices (PyTorch semantics: indices are positions)
TEST(SplitTest, HSplitWithIndices) {
  auto tensor = at::arange(10, at::TensorOptions().dtype(at::kFloat));

  // Split at indices [2, 5] -> [0:2], [2:5], [5:10]
  std::vector<int64_t> indices = {2, 5};
  auto splits = tensor.hsplit(at::IntArrayRef(indices));

  EXPECT_EQ(splits.size(), 3);
  EXPECT_EQ(splits[0].numel(), 2);  // [0, 1]
  EXPECT_EQ(splits[1].numel(), 3);  // [2, 3, 4]
  EXPECT_EQ(splits[2].numel(), 5);  // [5, 6, 7, 8, 9]
}

// Test for vsplit
TEST(SplitTest, VSplit) {
  auto tensor =
      at::arange(12, at::TensorOptions().dtype(at::kFloat)).reshape({6, 2});

  // vsplit splits along dim 0
  auto splits = tensor.vsplit(3);

  EXPECT_EQ(splits.size(), 3);
  EXPECT_EQ(splits[0].size(0), 2);
  EXPECT_EQ(splits[1].size(0), 2);
  EXPECT_EQ(splits[2].size(0), 2);
}

// Test for vsplit with indices (PyTorch semantics: indices are positions)
TEST(SplitTest, VSplitWithIndices) {
  auto tensor =
      at::arange(12, at::TensorOptions().dtype(at::kFloat)).reshape({6, 2});

  // Split at indices [2, 4] along dim 0 -> [0:2], [2:4], [4:6]
  std::vector<int64_t> indices = {2, 4};
  auto splits = tensor.vsplit(at::IntArrayRef(indices));

  EXPECT_EQ(splits.size(), 3);
  EXPECT_EQ(splits[0].size(0), 2);
  EXPECT_EQ(splits[1].size(0), 2);
  EXPECT_EQ(splits[2].size(0), 2);
}

// Test for dsplit
TEST(SplitTest, DSplit) {
  auto tensor =
      at::arange(24, at::TensorOptions().dtype(at::kFloat)).reshape({2, 3, 4});

  // dsplit splits along dim 2
  auto splits = tensor.dsplit(2);

  EXPECT_EQ(splits.size(), 2);
  EXPECT_EQ(splits[0].size(2), 2);
  EXPECT_EQ(splits[1].size(2), 2);
}

// Test for dsplit with indices (PyTorch semantics: indices are positions)
TEST(SplitTest, DSplitWithIndices) {
  auto tensor =
      at::arange(30, at::TensorOptions().dtype(at::kFloat)).reshape({2, 3, 5});

  // Split at indices [2, 3] along dim 2 -> [0:2], [2:3], [3:5]
  std::vector<int64_t> indices = {2, 3};
  auto splits = tensor.dsplit(at::IntArrayRef(indices));

  EXPECT_EQ(splits.size(), 3);
  EXPECT_EQ(splits[0].size(2), 2);
  EXPECT_EQ(splits[1].size(2), 1);
  EXPECT_EQ(splits[2].size(2), 2);
}

// Test for split along different dimensions
TEST(SplitTest, SplitAlongDifferentDims) {
  auto tensor =
      at::arange(24, at::TensorOptions().dtype(at::kFloat)).reshape({2, 3, 4});

  // Split along dim 1
  auto splits_dim1 = tensor.split(2, 1);
  EXPECT_EQ(splits_dim1.size(), 2);
  EXPECT_EQ(splits_dim1[0].size(1), 2);
  EXPECT_EQ(splits_dim1[1].size(1), 1);

  // Split along dim 2
  auto splits_dim2 = tensor.split(2, 2);
  EXPECT_EQ(splits_dim2.size(), 2);
  EXPECT_EQ(splits_dim2[0].size(2), 2);
  EXPECT_EQ(splits_dim2[1].size(2), 2);
}

// Test for tensor_split_symint
TEST(TensorSplitTest, TensorSplitSymInt) {
  // Use 9 elements to be evenly divisible by 3
  auto tensor = at::arange(9, at::TensorOptions().dtype(at::kFloat));

  c10::SymInt sections(3);
  auto splits = tensor.tensor_split_symint(sections, 0);

  EXPECT_EQ(splits.size(), 3);
  EXPECT_EQ(splits[0].numel(), 3);
  EXPECT_EQ(splits[1].numel(), 3);
  EXPECT_EQ(splits[2].numel(), 3);
}

// Test for split_symint
TEST(SplitTest, SplitSymInt) {
  auto tensor = at::arange(10, at::TensorOptions().dtype(at::kFloat));

  c10::SymInt split_size(3);
  auto splits = tensor.split_symint(split_size, 0);

  EXPECT_EQ(splits.size(), 4);
  EXPECT_EQ(splits[0].numel(), 3);
  EXPECT_EQ(splits[1].numel(), 3);
  EXPECT_EQ(splits[2].numel(), 3);
  EXPECT_EQ(splits[3].numel(), 1);
}

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
// Test for split on CUDA
TEST(SplitTest, SplitCUDA) {
  SKIP_IF_CUDA_RUNTIME_UNAVAILABLE();
  auto tensor =
      at::arange(10, at::TensorOptions().dtype(at::kFloat).device(at::kCUDA));

  auto splits = tensor.split(3, 0);

  EXPECT_EQ(splits.size(), 4);
  EXPECT_TRUE(splits[0].is_cuda());
  EXPECT_EQ(splits[0].numel(), 3);

  // Copy to CPU and verify
  auto cpu_split = splits[0].cpu();
  EXPECT_FLOAT_EQ(cpu_split[0].item<float>(), 0.0f);
  EXPECT_FLOAT_EQ(cpu_split[1].item<float>(), 1.0f);
  EXPECT_FLOAT_EQ(cpu_split[2].item<float>(), 2.0f);
}

// Test for tensor_split on CUDA
TEST(TensorSplitTest, TensorSplitCUDA) {
  SKIP_IF_CUDA_RUNTIME_UNAVAILABLE();
  auto tensor =
      at::arange(12, at::TensorOptions().dtype(at::kFloat).device(at::kCUDA));

  auto splits = tensor.tensor_split(4, 0);

  EXPECT_EQ(splits.size(), 4);
  for (const auto& split : splits) {
    EXPECT_TRUE(split.is_cuda());
  }
  EXPECT_EQ(splits[0].numel(), 3);
  EXPECT_EQ(splits[1].numel(), 3);
  EXPECT_EQ(splits[2].numel(), 3);
  EXPECT_EQ(splits[3].numel(), 3);
}
#endif
