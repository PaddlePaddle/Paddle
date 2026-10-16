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
#include <ATen/ops/index_reduce.h>
#include <c10/core/ScalarType.h>
#include <c10/core/TensorOptions.h>

#include "ATen/ATen.h"
#include "gtest/gtest.h"

// ======================== index_reduce tests ========================

TEST(TensorIndexReduceTest, SumInvalidThrows) {
  at::Tensor self = at::zeros({3, 4}, at::kFloat);
  at::Tensor index = at::empty({2}, at::kLong);
  index.data_ptr<int64_t>()[0] = 0;
  index.data_ptr<int64_t>()[1] = 2;
  at::Tensor source = at::full({2, 4}, 3.0f, at::kFloat);

  // "sum" is not a valid reduce mode for index_reduce in PyTorch
  ASSERT_THROW(self.index_reduce(0, index, source, "sum"), std::exception);
}

TEST(TensorIndexReduceTest, PaddleReduceNamesInvalidThrow) {
  at::Tensor self = at::ones({3, 4}, at::kFloat);
  at::Tensor index = at::empty({2}, at::kLong);
  index.data_ptr<int64_t>()[0] = 0;
  index.data_ptr<int64_t>()[1] = 2;
  at::Tensor source = at::full({2, 4}, 2.0f, at::kFloat);

  ASSERT_THROW(self.index_reduce(0, index, source, "add"), std::exception);
  ASSERT_THROW(self.index_reduce(0, index, source, "assign"), std::exception);
  ASSERT_THROW(self.index_reduce(0, index, source, "multiply"), std::exception);
}

TEST(TensorIndexReduceTest, IndexDtypeInvalidThrows) {
  at::Tensor self = at::ones({3, 4}, at::kFloat);
  at::Tensor index = at::empty({2}, at::kFloat);
  index.data_ptr<float>()[0] = 0.0f;
  index.data_ptr<float>()[1] = 2.0f;
  at::Tensor source = at::full({2, 4}, 2.0f, at::kFloat);

  // index dtype must be int32/int64
  ASSERT_THROW(self.index_reduce(0, index, source, "prod"), std::exception);
}

TEST(TensorIndexReduceTest, SourceDtypeMismatchThrows) {
  at::Tensor self = at::ones({3, 4}, at::kFloat);
  at::Tensor index = at::empty({2}, at::kLong);
  index.data_ptr<int64_t>()[0] = 0;
  index.data_ptr<int64_t>()[1] = 2;
  at::Tensor source = at::full({2, 4}, 2.0, at::kDouble);

  // self and source must have the same scalar type
  ASSERT_THROW(self.index_reduce(0, index, source, "prod"), std::exception);
}

TEST(TensorIndexReduceTest, IntIndexDtypeSucceeds) {
  at::Tensor self = at::ones({3, 4}, at::kFloat);
  at::Tensor index = at::empty({2}, at::kInt);
  index.data_ptr<int>()[0] = 0;
  index.data_ptr<int>()[1] = 2;
  at::Tensor source = at::full({2, 4}, 2.0f, at::kFloat);

  at::Tensor result = self.index_reduce(0, index, source, "prod");

  ASSERT_EQ(result.sizes(), c10::IntArrayRef({3, 4}));
  float* data = result.data_ptr<float>();
  ASSERT_FLOAT_EQ(data[0], 2.0f);
  ASSERT_FLOAT_EQ(data[4], 1.0f);
  ASSERT_FLOAT_EQ(data[8], 2.0f);
}

TEST(TensorIndexReduceTest, NegativeIndexThrows) {
  at::Tensor self = at::ones({3, 4}, at::kFloat);
  at::Tensor index = at::empty({2}, at::kLong);
  index.data_ptr<int64_t>()[0] = 0;
  index.data_ptr<int64_t>()[1] = -1;
  at::Tensor source = at::full({2, 4}, 2.0f, at::kFloat);

  ASSERT_THROW(self.index_reduce(0, index, source, "prod"), std::exception);

  at::Tensor inplace_self = at::ones({3, 4}, at::kFloat);
  ASSERT_THROW(inplace_self.index_reduce_(0, index, source, "prod"),
               std::exception);
}

TEST(TensorIndexReduceTest, IndexUpperBoundThrows) {
  at::Tensor self = at::ones({3, 4}, at::kFloat);
  at::Tensor index = at::empty({2}, at::kInt);
  index.data_ptr<int>()[0] = 0;
  index.data_ptr<int>()[1] = 3;
  at::Tensor source = at::full({2, 4}, 2.0f, at::kFloat);

  ASSERT_THROW(self.index_reduce(0, index, source, "prod"), std::exception);

  at::Tensor inplace_self = at::ones({3, 4}, at::kFloat);
  ASSERT_THROW(inplace_self.index_reduce_(0, index, source, "prod"),
               std::exception);
}

TEST(TensorIndexReduceTest, NegativeDim) {
  at::Tensor self = at::zeros({4, 6}, at::kFloat);
  at::Tensor index = at::empty({3}, at::kLong);
  index.data_ptr<int64_t>()[0] = 1;
  index.data_ptr<int64_t>()[1] = 3;
  index.data_ptr<int64_t>()[2] = 5;
  at::Tensor source = at::full({4, 3}, 4.0f, at::kFloat);

  // dim=-1 is equivalent to dim=1 for a 2D tensor
  at::Tensor result = self.index_reduce(-1, index, source, "mean");

  ASSERT_EQ(result.sizes(), c10::IntArrayRef({4, 6}));
  float* data = result.data_ptr<float>();
  // col 1: (0 + 4) / 2 = 2
  ASSERT_FLOAT_EQ(data[1], 2.0f);
  // col 3: (0 + 4) / 2 = 2
  ASSERT_FLOAT_EQ(data[3], 2.0f);
  // col 5: (0 + 4) / 2 = 2
  ASSERT_FLOAT_EQ(data[5], 2.0f);
  // col 0: unchanged = 0
  ASSERT_FLOAT_EQ(data[0], 0.0f);
}

TEST(TensorIndexReduceTest, ProdBasic2D) {
  at::Tensor self = at::ones({3, 4}, at::kFloat);
  at::Tensor index = at::empty({2}, at::kLong);
  index.data_ptr<int64_t>()[0] = 0;
  index.data_ptr<int64_t>()[1] = 2;
  at::Tensor source = at::full({2, 4}, 2.0f, at::kFloat);

  at::Tensor result = self.index_reduce(0, index, source, "prod");

  ASSERT_EQ(result.sizes(), c10::IntArrayRef({3, 4}));
  float* data = result.data_ptr<float>();
  // row 0: 1 * 2 = 2
  for (int j = 0; j < 4; ++j) {
    ASSERT_FLOAT_EQ(data[j], 2.0f);
  }
  // row 1: unchanged = 1
  for (int j = 0; j < 4; ++j) {
    ASSERT_FLOAT_EQ(data[4 + j], 1.0f);
  }
  // row 2: 1 * 2 = 2
  for (int j = 0; j < 4; ++j) {
    ASSERT_FLOAT_EQ(data[8 + j], 2.0f);
  }
}

TEST(TensorIndexReduceTest, MeanBasic2D) {
  at::Tensor self = at::zeros({3, 4}, at::kFloat);
  at::Tensor index = at::empty({3}, at::kLong);
  index.data_ptr<int64_t>()[0] = 0;
  index.data_ptr<int64_t>()[1] = 1;
  index.data_ptr<int64_t>()[2] = 0;
  at::Tensor source = at::full({3, 4}, 3.0f, at::kFloat);

  at::Tensor result = self.index_reduce(0, index, source, "mean");

  ASSERT_EQ(result.sizes(), c10::IntArrayRef({3, 4}));
  float* data = result.data_ptr<float>();
  // row 0: (0 + 3 + 3) / 3 = 2
  for (int j = 0; j < 4; ++j) {
    ASSERT_FLOAT_EQ(data[j], 2.0f);
  }
  // row 1: (0 + 3) / 2 = 1.5
  for (int j = 0; j < 4; ++j) {
    ASSERT_FLOAT_EQ(data[4 + j], 1.5f);
  }
  // row 2: unchanged = 0
  for (int j = 0; j < 4; ++j) {
    ASSERT_FLOAT_EQ(data[8 + j], 0.0f);
  }
}

TEST(TensorIndexReduceTest, AmaxBasic) {
  at::Tensor self = at::arange(0, 12, at::kFloat).reshape({3, 4});
  at::Tensor index = at::empty({3}, at::kLong);
  index.data_ptr<int64_t>()[0] = 0;
  index.data_ptr<int64_t>()[1] = 2;
  index.data_ptr<int64_t>()[2] = 0;
  at::Tensor source = at::full({3, 4}, 10.0f, at::kFloat);

  at::Tensor result = self.index_reduce(0, index, source, "amax");

  ASSERT_EQ(result.sizes(), c10::IntArrayRef({3, 4}));
  float* data = result.data_ptr<float>();
  // row 0: max(self[0], source[0], source[2]) = max(0,3,6,9, 10,10,10,10,
  // 10,10,10,10) col 0: max(0, 10, 10) = 10
  ASSERT_FLOAT_EQ(data[0], 10.0f);
  // col 3: max(3, 10, 10) = 10
  ASSERT_FLOAT_EQ(data[3], 10.0f);
  // row 1: unchanged
  ASSERT_FLOAT_EQ(data[4], 4.0f);
  ASSERT_FLOAT_EQ(data[7], 7.0f);
  // row 2: max(self[2], source[1]) = max(8,9,10,11, 10,10,10,10)
  ASSERT_FLOAT_EQ(data[8], 10.0f);
  ASSERT_FLOAT_EQ(data[11], 11.0f);
}

TEST(TensorIndexReduceTest, AminBasic) {
  at::Tensor self = at::arange(0, 12, at::kFloat).reshape({3, 4});
  at::Tensor index = at::empty({3}, at::kLong);
  index.data_ptr<int64_t>()[0] = 1;
  index.data_ptr<int64_t>()[1] = 1;
  index.data_ptr<int64_t>()[2] = 0;
  at::Tensor source = at::full({3, 4}, -5.0f, at::kFloat);

  at::Tensor result = self.index_reduce(0, index, source, "amin");

  ASSERT_EQ(result.sizes(), c10::IntArrayRef({3, 4}));
  float* data = result.data_ptr<float>();
  // row 0: min(self[0], source[2]) = min(0,1,2,3, -5,-5,-5,-5) = -5
  ASSERT_FLOAT_EQ(data[0], -5.0f);
  // row 1: min(self[1], source[0], source[1]) = min(4,5,6,7, -5,-5,-5,-5,
  // -5,-5,-5,-5)
  ASSERT_FLOAT_EQ(data[4], -5.0f);
  // row 2: unchanged
  ASSERT_FLOAT_EQ(data[8], 8.0f);
}

TEST(TensorIndexReduceTest, ProdExcludeSelf) {
  at::Tensor self = at::full({3, 4}, 5.0f, at::kFloat);
  at::Tensor index = at::empty({2}, at::kLong);
  index.data_ptr<int64_t>()[0] = 0;
  index.data_ptr<int64_t>()[1] = 2;
  at::Tensor source = at::full({2, 4}, 2.0f, at::kFloat);

  at::Tensor result = self.index_reduce(0, index, source, "prod", false);

  ASSERT_EQ(result.sizes(), c10::IntArrayRef({3, 4}));
  float* data = result.data_ptr<float>();
  // row 0: prod(source[0]) = 2 (exclude self)
  for (int j = 0; j < 4; ++j) {
    ASSERT_FLOAT_EQ(data[j], 2.0f);
  }
  // row 1: unchanged = 5 (not indexed)
  for (int j = 0; j < 4; ++j) {
    ASSERT_FLOAT_EQ(data[4 + j], 5.0f);
  }
  // row 2: prod(source[1]) = 2 (exclude self)
  for (int j = 0; j < 4; ++j) {
    ASSERT_FLOAT_EQ(data[8 + j], 2.0f);
  }
}

TEST(TensorIndexReduceTest, InplaceProd) {
  at::Tensor self = at::ones({3, 4}, at::kFloat);
  float* original_data_ptr = self.data_ptr<float>();

  at::Tensor index = at::empty({2}, at::kLong);
  index.data_ptr<int64_t>()[0] = 0;
  index.data_ptr<int64_t>()[1] = 2;
  at::Tensor source = at::full({2, 4}, 3.0f, at::kFloat);

  self.index_reduce_(0, index, source, "prod");

  // Verify data pointer unchanged (inplace)
  ASSERT_EQ(self.data_ptr<float>(), original_data_ptr);

  float* data = self.data_ptr<float>();
  // row 0: 1 * 3 = 3
  for (int j = 0; j < 4; ++j) {
    ASSERT_FLOAT_EQ(data[j], 3.0f);
  }
  // row 1: unchanged = 1
  for (int j = 0; j < 4; ++j) {
    ASSERT_FLOAT_EQ(data[4 + j], 1.0f);
  }
  // row 2: 1 * 3 = 3
  for (int j = 0; j < 4; ++j) {
    ASSERT_FLOAT_EQ(data[8 + j], 3.0f);
  }
}

TEST(TensorIndexReduceTest, Dim1) {
  at::Tensor self = at::zeros({4, 6}, at::kFloat);
  at::Tensor index = at::empty({3}, at::kLong);
  index.data_ptr<int64_t>()[0] = 1;
  index.data_ptr<int64_t>()[1] = 3;
  index.data_ptr<int64_t>()[2] = 5;
  at::Tensor source = at::full({4, 3}, 4.0f, at::kFloat);

  at::Tensor result = self.index_reduce(1, index, source, "mean");

  ASSERT_EQ(result.sizes(), c10::IntArrayRef({4, 6}));
  float* data = result.data_ptr<float>();
  // col 1: (0 + 4) / 2 = 2
  ASSERT_FLOAT_EQ(data[1], 2.0f);
  // col 3: (0 + 4) / 2 = 2
  ASSERT_FLOAT_EQ(data[3], 2.0f);
  // col 5: (0 + 4) / 2 = 2
  ASSERT_FLOAT_EQ(data[5], 2.0f);
  // col 0: unchanged = 0
  ASSERT_FLOAT_EQ(data[0], 0.0f);
}

TEST(TensorIndexReduceTest, OneDTensor) {
  at::Tensor self = at::ones({5}, at::kFloat);
  at::Tensor index = at::empty({3}, at::kLong);
  index.data_ptr<int64_t>()[0] = 0;
  index.data_ptr<int64_t>()[1] = 2;
  index.data_ptr<int64_t>()[2] = 4;
  at::Tensor source = at::full({3}, 3.0f, at::kFloat);

  at::Tensor result = self.index_reduce(0, index, source, "prod");

  ASSERT_EQ(result.sizes(), c10::IntArrayRef({5}));
  float* data = result.data_ptr<float>();
  ASSERT_FLOAT_EQ(data[0], 3.0f);
  ASSERT_FLOAT_EQ(data[1], 1.0f);
  ASSERT_FLOAT_EQ(data[2], 3.0f);
  ASSERT_FLOAT_EQ(data[3], 1.0f);
  ASSERT_FLOAT_EQ(data[4], 3.0f);
}

TEST(TensorIndexReduceTest, DoubleDtype) {
  at::Tensor self = at::zeros({3, 4}, at::kDouble);
  at::Tensor index = at::empty({3}, at::kLong);
  index.data_ptr<int64_t>()[0] = 0;
  index.data_ptr<int64_t>()[1] = 1;
  index.data_ptr<int64_t>()[2] = 0;
  at::Tensor source = at::full({3, 4}, 2.0, at::kDouble);

  at::Tensor result = self.index_reduce(0, index, source, "mean");

  ASSERT_EQ(result.scalar_type(), at::kDouble);
  double* data = result.data_ptr<double>();
  // row 0: (0 + 2 + 2) / 3 = 1.333...
  ASSERT_DOUBLE_EQ(data[0], 4.0 / 3.0);
}

TEST(TensorIndexReduceTest, IntDtype) {
  at::Tensor self = at::arange(0, 12, at::kInt).reshape({3, 4});
  at::Tensor index = at::empty({3}, at::kLong);
  index.data_ptr<int64_t>()[0] = 0;
  index.data_ptr<int64_t>()[1] = 2;
  index.data_ptr<int64_t>()[2] = 0;
  at::Tensor source = at::full({3, 4}, 20, at::kInt);

  at::Tensor result = self.index_reduce(0, index, source, "amax");

  ASSERT_EQ(result.scalar_type(), at::kInt);
  int* data = result.data_ptr<int>();
  // row 0: max(0,1,2,3, 20,20,20,20, 20,20,20,20) = 20
  ASSERT_EQ(data[0], 20);
}

TEST(TensorIndexReduceTest, LongDtype) {
  at::Tensor self = at::arange(0, 12, at::kLong).reshape({3, 4});
  at::Tensor index = at::empty({3}, at::kLong);
  index.data_ptr<int64_t>()[0] = 1;
  index.data_ptr<int64_t>()[1] = 1;
  index.data_ptr<int64_t>()[2] = 0;
  at::Tensor source = at::full({3, 4}, static_cast<int64_t>(-3), at::kLong);

  at::Tensor result = self.index_reduce(0, index, source, "amin");

  ASSERT_EQ(result.scalar_type(), at::kLong);
  int64_t* data = result.data_ptr<int64_t>();
  // row 0: min(0,1,2,3, -3,-3,-3,-3) = -3
  ASSERT_EQ(data[0], -3L);
}

TEST(TensorIndexReduceTest, NonMemberFunction) {
  at::Tensor self = at::ones({3, 4}, at::kFloat);
  at::Tensor index = at::empty({2}, at::kLong);
  index.data_ptr<int64_t>()[0] = 0;
  index.data_ptr<int64_t>()[1] = 2;
  at::Tensor source = at::full({2, 4}, 2.0f, at::kFloat);

  at::Tensor result = at::index_reduce(self, 0, index, source, "prod");

  ASSERT_EQ(result.sizes(), c10::IntArrayRef({3, 4}));
  float* data = result.data_ptr<float>();
  ASSERT_FLOAT_EQ(data[0], 2.0f);
  ASSERT_FLOAT_EQ(data[4], 1.0f);
  ASSERT_FLOAT_EQ(data[8], 2.0f);
}

TEST(TensorIndexReduceTest, InvalidReduceThrows) {
  at::Tensor self = at::ones({3, 4}, at::kFloat);
  at::Tensor index = at::empty({2}, at::kLong);
  index.data_ptr<int64_t>()[0] = 0;
  index.data_ptr<int64_t>()[1] = 2;
  at::Tensor source = at::full({2, 4}, 2.0f, at::kFloat);

  ASSERT_THROW(self.index_reduce(0, index, source, "invalid"), std::exception);
}
