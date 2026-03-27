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

// ==================== select tests ====================

// Test for select on 1D tensor
TEST(SelectTest, Select1D) {
  auto tensor = at::arange(10, at::TensorOptions().dtype(at::kFloat));

  // Select element at index 5
  auto selected = tensor.select(0, 5);

  // Result should be a scalar (0-dim tensor)
  EXPECT_EQ(selected.dim(), 0);
  EXPECT_FLOAT_EQ(selected.item<float>(), 5.0f);
}

// Test for select on 2D tensor along dim 0
TEST(SelectTest, Select2DDim0) {
  auto tensor =
      at::arange(12, at::TensorOptions().dtype(at::kFloat)).reshape({3, 4});

  // Select row at index 1 (second row)
  auto selected = tensor.select(0, 1);

  // Result should be 1D tensor of size 4
  EXPECT_EQ(selected.dim(), 1);
  EXPECT_EQ(selected.size(0), 4);

  // Second row should be [4, 5, 6, 7]
  EXPECT_FLOAT_EQ(selected[0].item<float>(), 4.0f);
  EXPECT_FLOAT_EQ(selected[1].item<float>(), 5.0f);
  EXPECT_FLOAT_EQ(selected[2].item<float>(), 6.0f);
  EXPECT_FLOAT_EQ(selected[3].item<float>(), 7.0f);
}

// Test for select on 2D tensor along dim 1
TEST(SelectTest, Select2DDim1) {
  auto tensor =
      at::arange(12, at::TensorOptions().dtype(at::kFloat)).reshape({3, 4});

  // Select column at index 2 (third column)
  auto selected = tensor.select(1, 2);

  // Result should be 1D tensor of size 3
  EXPECT_EQ(selected.dim(), 1);
  EXPECT_EQ(selected.size(0), 3);

  // Third column should be [2, 6, 10]
  EXPECT_FLOAT_EQ(selected[0].item<float>(), 2.0f);
  EXPECT_FLOAT_EQ(selected[1].item<float>(), 6.0f);
  EXPECT_FLOAT_EQ(selected[2].item<float>(), 10.0f);
}

// Test for select on 3D tensor
TEST(SelectTest, Select3D) {
  auto tensor =
      at::arange(24, at::TensorOptions().dtype(at::kFloat)).reshape({2, 3, 4});

  // Select along dim 0
  auto selected_dim0 = tensor.select(0, 1);
  EXPECT_EQ(selected_dim0.dim(), 2);
  EXPECT_EQ(selected_dim0.size(0), 3);
  EXPECT_EQ(selected_dim0.size(1), 4);

  // Select along dim 1
  auto selected_dim1 = tensor.select(1, 2);
  EXPECT_EQ(selected_dim1.dim(), 2);
  EXPECT_EQ(selected_dim1.size(0), 2);
  EXPECT_EQ(selected_dim1.size(1), 4);

  // Select along dim 2
  auto selected_dim2 = tensor.select(2, 3);
  EXPECT_EQ(selected_dim2.dim(), 2);
  EXPECT_EQ(selected_dim2.size(0), 2);
  EXPECT_EQ(selected_dim2.size(1), 3);
}

// Note: Negative index is not supported by Paddle's slice implementation
// Test for select with last index using positive index
TEST(SelectTest, SelectLastIndex) {
  auto tensor = at::arange(10, at::TensorOptions().dtype(at::kFloat));

  // Select last element using positive index (size - 1)
  auto selected = tensor.select(0, 9);

  EXPECT_EQ(selected.dim(), 0);
  EXPECT_FLOAT_EQ(selected.item<float>(), 9.0f);
}

// Test for select with first and last indices
TEST(SelectTest, SelectBoundary) {
  auto tensor = at::arange(5, at::TensorOptions().dtype(at::kFloat));

  // Select first element
  auto first = tensor.select(0, 0);
  EXPECT_FLOAT_EQ(first.item<float>(), 0.0f);

  // Select last element
  auto last = tensor.select(0, 4);
  EXPECT_FLOAT_EQ(last.item<float>(), 4.0f);
}

// ==================== select_symint tests ====================

// Test for select_symint
TEST(SelectTest, SelectSymInt) {
  auto tensor = at::arange(10, at::TensorOptions().dtype(at::kFloat));

  c10::SymInt index(5);
  auto selected = tensor.select_symint(0, index);

  EXPECT_EQ(selected.dim(), 0);
  EXPECT_FLOAT_EQ(selected.item<float>(), 5.0f);
}

// Test for select_symint on 2D tensor
TEST(SelectTest, SelectSymInt2D) {
  auto tensor =
      at::arange(12, at::TensorOptions().dtype(at::kFloat)).reshape({3, 4});

  c10::SymInt index(1);
  auto selected = tensor.select_symint(0, index);

  EXPECT_EQ(selected.dim(), 1);
  EXPECT_EQ(selected.size(0), 4);
  EXPECT_FLOAT_EQ(selected[0].item<float>(), 4.0f);
}

TEST(SelectTest, SelectNegativeIndexBranches) {
  auto tensor =
      at::arange(12, at::TensorOptions().dtype(at::kFloat)).reshape({3, 4});

  auto selected = tensor.select(-1, -1);
  EXPECT_EQ(selected.dim(), 1);
  EXPECT_EQ(selected.size(0), 3);
  EXPECT_FLOAT_EQ(selected[0].item<float>(), 3.0f);
  EXPECT_FLOAT_EQ(selected[2].item<float>(), 11.0f);

  c10::SymInt index(-1);
  auto selected_symint = tensor.select_symint(-1, index);
  EXPECT_EQ(selected_symint.size(0), 3);
  EXPECT_FLOAT_EQ(selected_symint[1].item<float>(), 7.0f);
}

// ==================== index_select tests ====================

// Test for index_select on 1D tensor
TEST(IndexSelectTest, IndexSelect1D) {
  auto tensor = at::arange(10, at::TensorOptions().dtype(at::kFloat));

  // Create index tensor [2, 5, 7]
  auto index = at::empty({3}, at::TensorOptions().dtype(at::kLong));
  index.data_ptr<int64_t>()[0] = 2;
  index.data_ptr<int64_t>()[1] = 5;
  index.data_ptr<int64_t>()[2] = 7;

  auto selected = tensor.index_select(0, index);

  EXPECT_EQ(selected.dim(), 1);
  EXPECT_EQ(selected.size(0), 3);
  EXPECT_FLOAT_EQ(selected[0].item<float>(), 2.0f);
  EXPECT_FLOAT_EQ(selected[1].item<float>(), 5.0f);
  EXPECT_FLOAT_EQ(selected[2].item<float>(), 7.0f);
}

// Test for index_select on 2D tensor along dim 0 (select rows)
TEST(IndexSelectTest, IndexSelect2DDim0) {
  auto tensor =
      at::arange(12, at::TensorOptions().dtype(at::kFloat)).reshape({3, 4});

  // Select rows [0, 2]
  auto index = at::empty({2}, at::TensorOptions().dtype(at::kLong));
  index.data_ptr<int64_t>()[0] = 0;
  index.data_ptr<int64_t>()[1] = 2;

  auto selected = tensor.index_select(0, index);

  EXPECT_EQ(selected.dim(), 2);
  EXPECT_EQ(selected.size(0), 2);
  EXPECT_EQ(selected.size(1), 4);

  // First selected row [0, 1, 2, 3]
  EXPECT_FLOAT_EQ(selected[0][0].item<float>(), 0.0f);
  EXPECT_FLOAT_EQ(selected[0][3].item<float>(), 3.0f);

  // Second selected row [8, 9, 10, 11]
  EXPECT_FLOAT_EQ(selected[1][0].item<float>(), 8.0f);
  EXPECT_FLOAT_EQ(selected[1][3].item<float>(), 11.0f);
}

// Test for index_select on 2D tensor along dim 1 (select columns)
TEST(IndexSelectTest, IndexSelect2DDim1) {
  auto tensor =
      at::arange(12, at::TensorOptions().dtype(at::kFloat)).reshape({3, 4});

  // Select columns [1, 3]
  auto index = at::empty({2}, at::TensorOptions().dtype(at::kLong));
  index.data_ptr<int64_t>()[0] = 1;
  index.data_ptr<int64_t>()[1] = 3;

  auto selected = tensor.index_select(1, index);

  EXPECT_EQ(selected.dim(), 2);
  EXPECT_EQ(selected.size(0), 3);
  EXPECT_EQ(selected.size(1), 2);

  // Check values
  EXPECT_FLOAT_EQ(selected[0][0].item<float>(), 1.0f);
  EXPECT_FLOAT_EQ(selected[0][1].item<float>(), 3.0f);
  EXPECT_FLOAT_EQ(selected[1][0].item<float>(), 5.0f);
  EXPECT_FLOAT_EQ(selected[1][1].item<float>(), 7.0f);
}

// Test for index_select with duplicate indices
TEST(IndexSelectTest, IndexSelectDuplicateIndices) {
  auto tensor = at::arange(5, at::TensorOptions().dtype(at::kFloat));

  // Select with duplicate indices [1, 1, 3, 1]
  auto index = at::empty({4}, at::TensorOptions().dtype(at::kLong));
  index.data_ptr<int64_t>()[0] = 1;
  index.data_ptr<int64_t>()[1] = 1;
  index.data_ptr<int64_t>()[2] = 3;
  index.data_ptr<int64_t>()[3] = 1;

  auto selected = tensor.index_select(0, index);

  EXPECT_EQ(selected.size(0), 4);
  EXPECT_FLOAT_EQ(selected[0].item<float>(), 1.0f);
  EXPECT_FLOAT_EQ(selected[1].item<float>(), 1.0f);
  EXPECT_FLOAT_EQ(selected[2].item<float>(), 3.0f);
  EXPECT_FLOAT_EQ(selected[3].item<float>(), 1.0f);
}

// Test for index_select on 3D tensor
TEST(IndexSelectTest, IndexSelect3D) {
  auto tensor =
      at::arange(24, at::TensorOptions().dtype(at::kFloat)).reshape({2, 3, 4});

  // Select along dim 1
  auto index = at::empty({2}, at::TensorOptions().dtype(at::kLong));
  index.data_ptr<int64_t>()[0] = 0;
  index.data_ptr<int64_t>()[1] = 2;

  auto selected = tensor.index_select(1, index);

  EXPECT_EQ(selected.dim(), 3);
  EXPECT_EQ(selected.size(0), 2);
  EXPECT_EQ(selected.size(1), 2);
  EXPECT_EQ(selected.size(2), 4);
}

// ==================== masked_select tests ====================

// Test for masked_select on 1D tensor
TEST(MaskedSelectTest, MaskedSelect1D) {
  auto tensor = at::arange(10, at::TensorOptions().dtype(at::kFloat));

  // Create mask for elements > 5
  auto mask = at::empty({10}, at::TensorOptions().dtype(at::kBool));
  for (int i = 0; i < 10; ++i) {
    mask.data_ptr<bool>()[i] = (i > 5);
  }

  auto selected = tensor.masked_select(mask);

  // Should select [6, 7, 8, 9]
  EXPECT_EQ(selected.dim(), 1);
  EXPECT_EQ(selected.numel(), 4);
  EXPECT_FLOAT_EQ(selected[0].item<float>(), 6.0f);
  EXPECT_FLOAT_EQ(selected[1].item<float>(), 7.0f);
  EXPECT_FLOAT_EQ(selected[2].item<float>(), 8.0f);
  EXPECT_FLOAT_EQ(selected[3].item<float>(), 9.0f);
}

// Test for masked_select on 2D tensor
TEST(MaskedSelectTest, MaskedSelect2D) {
  auto tensor =
      at::arange(12, at::TensorOptions().dtype(at::kFloat)).reshape({3, 4});

  // Create mask - select even numbers
  auto mask = at::empty({3, 4}, at::TensorOptions().dtype(at::kBool));
  for (int i = 0; i < 12; ++i) {
    mask.data_ptr<bool>()[i] = (i % 2 == 0);
  }

  auto selected = tensor.masked_select(mask);

  // Should select [0, 2, 4, 6, 8, 10]
  EXPECT_EQ(selected.dim(), 1);  // Result is always 1D
  EXPECT_EQ(selected.numel(), 6);
  EXPECT_FLOAT_EQ(selected[0].item<float>(), 0.0f);
  EXPECT_FLOAT_EQ(selected[1].item<float>(), 2.0f);
  EXPECT_FLOAT_EQ(selected[2].item<float>(), 4.0f);
  EXPECT_FLOAT_EQ(selected[3].item<float>(), 6.0f);
  EXPECT_FLOAT_EQ(selected[4].item<float>(), 8.0f);
  EXPECT_FLOAT_EQ(selected[5].item<float>(), 10.0f);
}

// Test for masked_select with all true mask
TEST(MaskedSelectTest, MaskedSelectAllTrue) {
  auto tensor = at::arange(5, at::TensorOptions().dtype(at::kFloat));

  // All true mask
  auto mask = at::empty({5}, at::TensorOptions().dtype(at::kBool));
  for (int i = 0; i < 5; ++i) {
    mask.data_ptr<bool>()[i] = true;
  }

  auto selected = tensor.masked_select(mask);

  EXPECT_EQ(selected.numel(), 5);
  for (int i = 0; i < 5; ++i) {
    EXPECT_FLOAT_EQ(selected[i].item<float>(), static_cast<float>(i));
  }
}

// Test for masked_select with all false mask
TEST(MaskedSelectTest, MaskedSelectAllFalse) {
  auto tensor = at::arange(5, at::TensorOptions().dtype(at::kFloat));

  // All false mask
  auto mask = at::empty({5}, at::TensorOptions().dtype(at::kBool));
  for (int i = 0; i < 5; ++i) {
    mask.data_ptr<bool>()[i] = false;
  }

  auto selected = tensor.masked_select(mask);

  EXPECT_EQ(selected.numel(), 0);
}

// Test for masked_select with different dtypes
TEST(MaskedSelectTest, MaskedSelectDifferentDtypes) {
  // Test with int64
  auto tensor_int = at::arange(10, at::TensorOptions().dtype(at::kLong));
  auto mask = at::empty({10}, at::TensorOptions().dtype(at::kBool));
  for (int i = 0; i < 10; ++i) {
    mask.data_ptr<bool>()[i] = (i >= 7);
  }

  auto selected = tensor_int.masked_select(mask);

  EXPECT_EQ(selected.numel(), 3);
  EXPECT_EQ(selected[0].item<int64_t>(), 7);
  EXPECT_EQ(selected[1].item<int64_t>(), 8);
  EXPECT_EQ(selected[2].item<int64_t>(), 9);
}

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
// Test for select on CUDA
TEST(SelectTest, SelectCUDA) {
  SKIP_IF_CUDA_RUNTIME_UNAVAILABLE();
  auto tensor =
      at::arange(10, at::TensorOptions().dtype(at::kFloat).device(at::kCUDA));

  auto selected = tensor.select(0, 5);

  EXPECT_TRUE(selected.is_cuda());
  EXPECT_EQ(selected.dim(), 0);

  auto cpu_selected = selected.cpu();
  EXPECT_FLOAT_EQ(cpu_selected.item<float>(), 5.0f);
}

// Test for index_select on CUDA
TEST(IndexSelectTest, IndexSelectCUDA) {
  SKIP_IF_CUDA_RUNTIME_UNAVAILABLE();
  auto tensor =
      at::arange(10, at::TensorOptions().dtype(at::kFloat).device(at::kCUDA));

  auto index =
      at::empty({3}, at::TensorOptions().dtype(at::kLong).device(at::kCUDA));
  auto cpu_index = index.cpu();
  cpu_index.data_ptr<int64_t>()[0] = 1;
  cpu_index.data_ptr<int64_t>()[1] = 3;
  cpu_index.data_ptr<int64_t>()[2] = 5;
  index.copy_(cpu_index);

  auto selected = tensor.index_select(0, index);

  EXPECT_TRUE(selected.is_cuda());
  EXPECT_EQ(selected.size(0), 3);

  auto cpu_selected = selected.cpu();
  EXPECT_FLOAT_EQ(cpu_selected[0].item<float>(), 1.0f);
  EXPECT_FLOAT_EQ(cpu_selected[1].item<float>(), 3.0f);
  EXPECT_FLOAT_EQ(cpu_selected[2].item<float>(), 5.0f);
}

// Test for masked_select on CUDA
TEST(MaskedSelectTest, MaskedSelectCUDA) {
  SKIP_IF_CUDA_RUNTIME_UNAVAILABLE();
  auto tensor =
      at::arange(10, at::TensorOptions().dtype(at::kFloat).device(at::kCUDA));

  // Create mask on CUDA and copy data from CPU
  auto mask =
      at::empty({10}, at::TensorOptions().dtype(at::kBool).device(at::kCUDA));
  auto cpu_mask = mask.cpu();
  for (int i = 0; i < 10; ++i) {
    cpu_mask.data_ptr<bool>()[i] = (i % 2 == 0);
  }
  mask.copy_(cpu_mask);

  auto selected = tensor.masked_select(mask);

  EXPECT_TRUE(selected.is_cuda());
  EXPECT_EQ(selected.numel(), 5);

  auto cpu_selected = selected.cpu();
  float val0 = cpu_selected[0].item<float>();
  float val1 = cpu_selected[1].item<float>();
  float val2 = cpu_selected[2].item<float>();
  float val3 = cpu_selected[3].item<float>();
  float val4 = cpu_selected[4].item<float>();
  EXPECT_FLOAT_EQ(val0, 0.0f);
  EXPECT_FLOAT_EQ(val1, 2.0f);
  EXPECT_FLOAT_EQ(val2, 4.0f);
  EXPECT_FLOAT_EQ(val3, 6.0f);
  EXPECT_FLOAT_EQ(val4, 8.0f);
}
#endif
