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

#include <ATen/ops/svd.h>

#include <ATen/Functions.h>
#include <ATen/core/TensorBody.h>
#include <c10/core/ScalarType.h>
#include <c10/core/TensorOptions.h>

#include <cmath>

#include "ATen/ATen.h"
#include "gtest/gtest.h"
#include "test/cpp/prim/init_env_utils.h"

namespace {

class TensorSvdTest : public ::testing::Test {
 protected:
  static void SetUpTestSuite() { paddle::prim::InitTensorOperants(); }
};

}  // namespace

static void AssertSvdReconstructsFloat(const at::Tensor& input,
                                       const at::Tensor& U,
                                       const at::Tensor& S,
                                       const at::Tensor& V,
                                       float tolerance) {
  const int64_t m = input.size(0);
  const int64_t n = input.size(1);
  const int64_t k = S.size(0);
  const at::Tensor input_cont = input.contiguous();
  const at::Tensor u_cont = U.contiguous();
  const at::Tensor s_cont = S.contiguous();
  const at::Tensor v_cont = V.contiguous();
  const float* input_data = input_cont.data_ptr<float>();
  const float* u_data = u_cont.data_ptr<float>();
  const float* s_data = s_cont.data_ptr<float>();
  const float* v_data = v_cont.data_ptr<float>();
  for (int64_t i = 0; i < m; ++i) {
    for (int64_t j = 0; j < n; ++j) {
      float value = 0.0f;
      for (int64_t r = 0; r < k; ++r) {
        value += u_data[i * k + r] * s_data[r] * v_data[j * k + r];
      }
      ASSERT_NEAR(value, input_data[i * n + j], tolerance);
    }
  }
}

static void AssertSvdReconstructsComplexFloat(const at::Tensor& input,
                                              const at::Tensor& U,
                                              const at::Tensor& S,
                                              const at::Tensor& V,
                                              float tolerance) {
  const int64_t m = input.size(0);
  const int64_t n = input.size(1);
  const int64_t k = S.size(0);
  const at::Tensor input_cont = input.contiguous();
  const at::Tensor u_cont = U.contiguous();
  const at::Tensor s_cont = S.contiguous();
  const at::Tensor v_cont = V.contiguous();
  const auto* input_data = input_cont.data_ptr<at::complex<float>>();
  const auto* u_data = u_cont.data_ptr<at::complex<float>>();
  const float* s_data = s_cont.data_ptr<float>();
  const auto* v_data = v_cont.data_ptr<at::complex<float>>();
  for (int64_t i = 0; i < m; ++i) {
    for (int64_t j = 0; j < n; ++j) {
      at::complex<float> value(0.0f, 0.0f);
      for (int64_t r = 0; r < k; ++r) {
        auto v = v_data[j * k + r];
        auto v_conj = at::complex<float>(v.real, -v.imag);
        value +=
            u_data[i * k + r] * at::complex<float>(s_data[r], 0.0f) * v_conj;
      }
      auto expected = input_data[i * n + j];
      ASSERT_NEAR(value.real, expected.real, tolerance);
      ASSERT_NEAR(value.imag, expected.imag, tolerance);
    }
  }
}

TEST_F(TensorSvdTest, SvdBasic) {
  at::Tensor t = at::zeros({3, 4}, at::kFloat);
  float* data = t.data_ptr<float>();
  for (int i = 0; i < 12; ++i) {
    data[i] = static_cast<float>(i + 1);
  }

  auto [U, S, V] = at::svd(t);

  // U shape: {3, 3} (some=true, k=min(3,4)=3)
  ASSERT_EQ(U.dim(), 2);
  ASSERT_EQ(U.size(0), 3);
  ASSERT_EQ(U.size(1), 3);

  // S shape: {3} (k=min(3,4)=3)
  ASSERT_EQ(S.dim(), 1);
  ASSERT_EQ(S.size(0), 3);

  // V shape: {4, 3} (some=true, k=min(3,4)=3)
  ASSERT_EQ(V.dim(), 2);
  ASSERT_EQ(V.size(0), 4);
  ASSERT_EQ(V.size(1), 3);
  AssertSvdReconstructsFloat(t, U, S, V, 1e-4f);
}

TEST_F(TensorSvdTest, SvdFullMatrices) {
  at::Tensor t = at::zeros({3, 4}, at::kFloat);
  float* data = t.data_ptr<float>();
  for (int i = 0; i < 12; ++i) {
    data[i] = static_cast<float>(i + 1);
  }

  auto [U, S, V] = at::svd(t, /*some=*/false);

  // U shape: {3, 3} (some=false, full_matrices=true, m=3)
  ASSERT_EQ(U.dim(), 2);
  ASSERT_EQ(U.size(0), 3);
  ASSERT_EQ(U.size(1), 3);

  // S shape: {3} (k=min(3,4)=3)
  ASSERT_EQ(S.dim(), 1);
  ASSERT_EQ(S.size(0), 3);

  // V shape: {4, 4} (some=false, full_matrices=true, n=4)
  ASSERT_EQ(V.dim(), 2);
  ASSERT_EQ(V.size(0), 4);
  ASSERT_EQ(V.size(1), 4);
}

TEST_F(TensorSvdTest, SvdNoComputeUv) {
  at::Tensor t = at::zeros({3, 4}, at::kFloat);
  float* data = t.data_ptr<float>();
  for (int i = 0; i < 12; ++i) {
    data[i] = static_cast<float>(i + 1);
  }

  auto [U, S, V] = at::svd(t, /*some=*/true, /*compute_uv=*/false);

  // U shape: {3, 3}, all zeros (compute_uv=false ignores 'some')
  ASSERT_EQ(U.dim(), 2);
  ASSERT_EQ(U.size(0), 3);
  ASSERT_EQ(U.size(1), 3);
  ASSERT_FLOAT_EQ(U.abs().sum().item().to<float>(), 0.0f);

  // S shape: {3}
  ASSERT_EQ(S.dim(), 1);
  ASSERT_EQ(S.size(0), 3);

  // V shape: {4, 4}, all zeros (compute_uv=false ignores 'some')
  ASSERT_EQ(V.dim(), 2);
  ASSERT_EQ(V.size(0), 4);
  ASSERT_EQ(V.size(1), 4);
  ASSERT_FLOAT_EQ(V.abs().sum().item().to<float>(), 0.0f);
}

TEST_F(TensorSvdTest, SvdMethod) {
  at::Tensor t = at::zeros({3, 4}, at::kFloat);
  float* data = t.data_ptr<float>();
  for (int i = 0; i < 12; ++i) {
    data[i] = static_cast<float>(i + 1);
  }

  auto [U, S, V] = t.svd();

  ASSERT_EQ(U.dim(), 2);
  ASSERT_EQ(U.size(0), 3);
  ASSERT_EQ(U.size(1), 3);

  ASSERT_EQ(S.dim(), 1);
  ASSERT_EQ(S.size(0), 3);

  ASSERT_EQ(V.dim(), 2);
  ASSERT_EQ(V.size(0), 4);
  ASSERT_EQ(V.size(1), 3);
}

TEST_F(TensorSvdTest, SvdBatch) {
  at::Tensor t = at::zeros({2, 3, 4}, at::kFloat);
  float* data = t.data_ptr<float>();
  for (int i = 0; i < 24; ++i) {
    data[i] = static_cast<float>(i + 1);
  }

  auto [U, S, V] = at::svd(t);

  // U shape: {2, 3, 3}
  ASSERT_EQ(U.dim(), 3);
  ASSERT_EQ(U.size(0), 2);
  ASSERT_EQ(U.size(1), 3);
  ASSERT_EQ(U.size(2), 3);

  // S shape: {2, 3}
  ASSERT_EQ(S.dim(), 2);
  ASSERT_EQ(S.size(0), 2);
  ASSERT_EQ(S.size(1), 3);

  // V shape: {2, 4, 3}
  ASSERT_EQ(V.dim(), 3);
  ASSERT_EQ(V.size(0), 2);
  ASSERT_EQ(V.size(1), 4);
  ASSERT_EQ(V.size(2), 3);
}

TEST_F(TensorSvdTest, SvdDoubleDtype) {
  at::Tensor t = at::zeros({3, 4}, at::kDouble);
  double* data = t.data_ptr<double>();
  for (int i = 0; i < 12; ++i) {
    data[i] = static_cast<double>(i + 1);
  }

  auto [U, S, V] = at::svd(t);

  ASSERT_EQ(U.scalar_type(), at::kDouble);
  ASSERT_EQ(S.scalar_type(), at::kDouble);
  ASSERT_EQ(V.scalar_type(), at::kDouble);
}

TEST_F(TensorSvdTest, SvdWideMatrix) {
  at::Tensor t = at::zeros({3, 5}, at::kFloat);
  float* data = t.data_ptr<float>();
  for (int i = 0; i < 15; ++i) {
    data[i] = static_cast<float>(i + 1);
  }

  auto [U, S, V] = at::svd(t);

  // m=3, n=5, k=3
  ASSERT_EQ(U.size(0), 3);
  ASSERT_EQ(U.size(1), 3);
  ASSERT_EQ(S.size(0), 3);
  ASSERT_EQ(V.size(0), 5);
  ASSERT_EQ(V.size(1), 3);
}

TEST_F(TensorSvdTest, SvdComplexDtype) {
  at::Tensor t = at::zeros({2, 3}, at::kComplexFloat);
  auto* data = t.data_ptr<at::complex<float>>();
  data[0] = at::complex<float>(1.0f, 2.0f);
  data[1] = at::complex<float>(2.0f, -1.0f);
  data[2] = at::complex<float>(3.0f, 0.0f);
  data[3] = at::complex<float>(4.0f, 1.0f);
  data[4] = at::complex<float>(5.0f, -2.0f);
  data[5] = at::complex<float>(6.0f, 3.0f);

  auto [U, S, V] = at::svd(t);
  ASSERT_EQ(U.scalar_type(), at::kComplexFloat);
  ASSERT_EQ(S.scalar_type(), at::kFloat);
  ASSERT_EQ(V.scalar_type(), at::kComplexFloat);
  ASSERT_EQ(V.size(0), 3);
  ASSERT_EQ(V.size(1), 2);
  AssertSvdReconstructsComplexFloat(t, U, S, V, 1e-4f);
}

TEST_F(TensorSvdTest, SvdFullNoComputeUv) {
  at::Tensor t = at::zeros({3, 4}, at::kFloat);
  float* data = t.data_ptr<float>();
  for (int i = 0; i < 12; ++i) {
    data[i] = static_cast<float>(i + 1);
  }

  auto [U, S, V] = at::svd(t, /*some=*/false, /*compute_uv=*/false);

  // U shape: {3, 3}, all zeros
  ASSERT_EQ(U.size(0), 3);
  ASSERT_EQ(U.size(1), 3);
  ASSERT_FLOAT_EQ(U.abs().sum().item().to<float>(), 0.0f);

  // S shape: {3}
  ASSERT_EQ(S.size(0), 3);

  // V shape: {4, 4}, all zeros
  ASSERT_EQ(V.size(0), 4);
  ASSERT_EQ(V.size(1), 4);
  ASSERT_FLOAT_EQ(V.abs().sum().item().to<float>(), 0.0f);
}
