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
#include <ATen/ops/cholesky.h>
#include <c10/core/ScalarType.h>
#include <c10/core/TensorOptions.h>

#include "ATen/ATen.h"
#include "gtest/gtest.h"
#include "test/cpp/prim/init_env_utils.h"
#include "torch/all.h"

namespace {

class TensorCholeskyTest : public ::testing::Test {
 protected:
  static void SetUpTestSuite() { paddle::prim::InitTensorOperants(); }
};

}  // namespace

// Build a symmetric positive definite matrix using diagonal dominance:
// diagonal element > sum of absolute values of other elements in the row.
static at::Tensor make_spd_matrix(const std::vector<int64_t>& shape,
                                  at::ScalarType dtype) {
  at::Tensor A = at::zeros(shape, dtype);
  int64_t n = shape[shape.size() - 1];
  int64_t m = shape[shape.size() - 2];
  int64_t batch = 1;
  for (size_t i = 0; i + 2 < shape.size(); ++i) {
    batch *= shape[i];
  }

  if (dtype == at::kFloat) {
    float* data = A.data_ptr<float>();
    for (int64_t b = 0; b < batch; ++b) {
      for (int64_t i = 0; i < m; ++i) {
        for (int64_t j = 0; j < n; ++j) {
          int64_t idx = b * m * n + i * n + j;
          if (i == j) {
            data[idx] = static_cast<float>(n);
          } else {
            data[idx] = 0.5f;
          }
        }
      }
    }
  } else if (dtype == at::kDouble) {
    double* data = A.data_ptr<double>();
    for (int64_t b = 0; b < batch; ++b) {
      for (int64_t i = 0; i < m; ++i) {
        for (int64_t j = 0; j < n; ++j) {
          int64_t idx = b * m * n + i * n + j;
          if (i == j) {
            data[idx] = static_cast<double>(n);
          } else {
            data[idx] = 0.5;
          }
        }
      }
    }
  }
  return A;
}

TEST_F(TensorCholeskyTest, BasicCholesky) {
  at::Tensor A = make_spd_matrix({3, 3}, at::kFloat);
  at::Tensor result = at::cholesky(A);

  ASSERT_EQ(result.dim(), 2);
  ASSERT_EQ(result.sizes()[0], 3);
  ASSERT_EQ(result.sizes()[1], 3);

  // Verify result is lower triangular (upper part should be 0)
  float* data = result.data_ptr<float>();
  ASSERT_FLOAT_EQ(data[1], 0.0f);
  ASSERT_FLOAT_EQ(data[2], 0.0f);
  ASSERT_FLOAT_EQ(data[5], 0.0f);
}

TEST_F(TensorCholeskyTest, CholeskyUpper) {
  at::Tensor A = make_spd_matrix({3, 3}, at::kFloat);
  at::Tensor result = at::cholesky(A, /*upper=*/true);

  ASSERT_EQ(result.dim(), 2);
  ASSERT_EQ(result.sizes()[0], 3);
  ASSERT_EQ(result.sizes()[1], 3);

  // Verify result is upper triangular (lower part should be 0)
  float* data = result.data_ptr<float>();
  ASSERT_FLOAT_EQ(data[3], 0.0f);
  ASSERT_FLOAT_EQ(data[6], 0.0f);
  ASSERT_FLOAT_EQ(data[7], 0.0f);
}

TEST_F(TensorCholeskyTest, MethodCholesky) {
  at::Tensor A = make_spd_matrix({3, 3}, at::kFloat);
  at::Tensor result = A.cholesky();

  ASSERT_EQ(result.dim(), 2);
  ASSERT_EQ(result.sizes()[0], 3);
  ASSERT_EQ(result.sizes()[1], 3);
}

TEST_F(TensorCholeskyTest, MethodCholeskyUpper) {
  at::Tensor A = make_spd_matrix({3, 3}, at::kFloat);
  at::Tensor result = A.cholesky(/*upper=*/true);

  ASSERT_EQ(result.dim(), 2);
  ASSERT_EQ(result.sizes()[0], 3);
  ASSERT_EQ(result.sizes()[1], 3);
}

TEST_F(TensorCholeskyTest, Float64Dtype) {
  at::Tensor A = make_spd_matrix({3, 3}, at::kDouble);
  at::Tensor result = at::cholesky(A);

  ASSERT_EQ(result.scalar_type(), at::kDouble);
  ASSERT_EQ(result.dim(), 2);
}

TEST_F(TensorCholeskyTest, BatchMatrix) {
  at::Tensor A = make_spd_matrix({2, 3, 3}, at::kFloat);
  at::Tensor result = at::cholesky(A);

  ASSERT_EQ(result.dim(), 3);
  ASSERT_EQ(result.sizes()[0], 2);
  ASSERT_EQ(result.sizes()[1], 3);
  ASSERT_EQ(result.sizes()[2], 3);
}

TEST_F(TensorCholeskyTest, NonPositiveDefinite) {
  at::Tensor A = at::zeros({3, 3}, at::kFloat);
  ASSERT_THROW((void)at::cholesky(A), std::exception);
}
