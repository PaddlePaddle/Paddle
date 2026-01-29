// Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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
#include "torch/all.h"

TEST(TensorBaseTest, DimensionAPIs) {
  // Test dimension related APIs
  at::TensorBase tensor = at::ones({2, 3, 4}, at::kFloat);

  // Test sizes()
  auto sizes = tensor.sizes();
  ASSERT_EQ(sizes.size(), 3);
  ASSERT_EQ(sizes[0], 2);
  ASSERT_EQ(sizes[1], 3);
  ASSERT_EQ(sizes[2], 4);

  // Test size(dim)
  ASSERT_EQ(tensor.size(0), 2);
  ASSERT_EQ(tensor.size(1), 3);
  ASSERT_EQ(tensor.size(2), 4);

  // Test strides()
  auto strides = tensor.strides();
  ASSERT_EQ(strides.size(), 3);
  ASSERT_EQ(strides[0], 12);  // 3*4
  ASSERT_EQ(strides[1], 4);   // 4
  ASSERT_EQ(strides[2], 1);   // contiguous

  // Test stride(dim)
  ASSERT_EQ(tensor.stride(0), 12);
  ASSERT_EQ(tensor.stride(1), 4);
  ASSERT_EQ(tensor.stride(2), 1);

  // Test numel()
  ASSERT_EQ(tensor.numel(), 24);  // 2*3*4

  // Test dim()/ndimension()
  ASSERT_EQ(tensor.dim(), 3);
  ASSERT_EQ(tensor.ndimension(), 3);
}

TEST(TestSymbolicInt, SymSizeAPI) {
  // Test sym_size() API
  at::TensorBase tensor = at::ones({2, 3, 4}, at::kFloat);

  // Test sym_size(dim) returns c10::SymInt
  c10::SymInt sym_size_0 = tensor.sym_size(0);
  c10::SymInt sym_size_1 = tensor.sym_size(1);
  c10::SymInt sym_size_2 = tensor.sym_size(2);

  ASSERT_EQ(sym_size_0, 2);
  ASSERT_EQ(sym_size_1, 3);
  ASSERT_EQ(sym_size_2, 4);

  // Test sym_size with negative index
  c10::SymInt sym_size_neg1 = tensor.sym_size(-1);
  c10::SymInt sym_size_neg2 = tensor.sym_size(-2);
  c10::SymInt sym_size_neg3 = tensor.sym_size(-3);

  ASSERT_EQ(sym_size_neg1, 4);
  ASSERT_EQ(sym_size_neg2, 3);
  ASSERT_EQ(sym_size_neg3, 2);
}

TEST(TestSymbolicInt, SymSizesAPI) {
  // Test sym_sizes() API
  at::TensorBase tensor = at::ones({2, 3, 4, 5}, at::kFloat);

  // Test sym_sizes() returns c10::SymIntArrayRef
  c10::SymIntArrayRef sym_sizes = tensor.sym_sizes();

  ASSERT_EQ(sym_sizes.size(), 4);
  ASSERT_EQ(sym_sizes[0], 2);
  ASSERT_EQ(sym_sizes[1], 3);
  ASSERT_EQ(sym_sizes[2], 4);
  ASSERT_EQ(sym_sizes[3], 5);
}

TEST(TestSymbolicInt, SymStrideAPI) {
  // Test sym_stride() API
  at::TensorBase tensor = at::ones({2, 3, 4}, at::kFloat);

  // Test sym_stride(dim) returns c10::SymInt
  c10::SymInt sym_stride_0 = tensor.sym_stride(0);
  c10::SymInt sym_stride_1 = tensor.sym_stride(1);
  c10::SymInt sym_stride_2 = tensor.sym_stride(2);

  ASSERT_EQ(sym_stride_0, 12);  // 3*4
  ASSERT_EQ(sym_stride_1, 4);   // 4
  ASSERT_EQ(sym_stride_2, 1);   // contiguous

  // Test sym_stride with negative index
  c10::SymInt sym_stride_neg1 = tensor.sym_stride(-1);
  c10::SymInt sym_stride_neg2 = tensor.sym_stride(-2);

  ASSERT_EQ(sym_stride_neg1, 1);
  ASSERT_EQ(sym_stride_neg2, 4);
}

TEST(TestSymbolicInt, SymStridesAPI) {
  // Test sym_strides() API
  at::TensorBase tensor = at::ones({2, 3, 4}, at::kFloat);

  // Test sym_strides() returns c10::SymIntArrayRef
  c10::SymIntArrayRef sym_strides = tensor.sym_strides();

  ASSERT_EQ(sym_strides.size(), 3);
  ASSERT_EQ(sym_strides[0], 12);  // 3*4
  ASSERT_EQ(sym_strides[1], 4);   // 4
  ASSERT_EQ(sym_strides[2], 1);   // contiguous
}

TEST(TestSymbolicInt, SymNumelAPI) {
  // Test sym_numel() API
  at::TensorBase tensor = at::ones({2, 3, 4}, at::kFloat);

  // Test sym_numel() returns c10::SymInt
  c10::SymInt sym_numel = tensor.sym_numel();

  ASSERT_EQ(sym_numel, 24);  // 2*3*4

  // Test with different shape
  at::TensorBase tensor2 = at::ones({5, 6, 7, 8}, at::kFloat);
  c10::SymInt sym_numel2 = tensor2.sym_numel();

  ASSERT_EQ(sym_numel2, 1680);  // 5*6*7*8
}

TEST(TestSymbolicInt, SymAPIsConsistency) {
  // Test that sym_* APIs return values consistent with non-sym APIs
  at::TensorBase tensor = at::ones({3, 4, 5, 6}, at::kFloat);

  // Test sym_size vs size
  for (int64_t i = 0; i < tensor.dim(); ++i) {
    ASSERT_EQ(tensor.sym_size(i), tensor.size(i));
  }

  // Test sym_stride vs stride
  for (int64_t i = 0; i < tensor.dim(); ++i) {
    ASSERT_EQ(tensor.sym_stride(i), tensor.stride(i));
  }

  // Test sym_numel vs numel
  ASSERT_EQ(tensor.sym_numel(), tensor.numel());

  // Test sym_sizes vs sizes
  auto sizes = tensor.sizes();
  auto sym_sizes = tensor.sym_sizes();
  ASSERT_EQ(sizes.size(), sym_sizes.size());
  for (size_t i = 0; i < sizes.size(); ++i) {
    ASSERT_EQ(sym_sizes[i], sizes[i]);
  }

  // Test sym_strides vs strides
  auto strides = tensor.strides();
  auto sym_strides = tensor.sym_strides();
  ASSERT_EQ(strides.size(), sym_strides.size());
  for (size_t i = 0; i < strides.size(); ++i) {
    ASSERT_EQ(sym_strides[i], strides[i]);
  }
}
