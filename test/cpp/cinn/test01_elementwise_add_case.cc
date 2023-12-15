// Copyright (c) 2021 CINN Authors. All Rights Reserved.
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

#include <glog/logging.h>
#include <gtest/gtest.h>

#include "paddle/cinn/common/test_helper.h"
#include "paddle/cinn/runtime/cinn_runtime.h"
#include "test/cpp/cinn/test01_elementwise_add.h"
#include "test/cpp/cinn/test01_elementwise_add_compute_at.h"
#include "test/cpp/cinn/test01_elementwise_add_compute_at_level1.h"
#include "test/cpp/cinn/test01_elementwise_add_vectorize.h"

namespace cinn {

TEST(test01, basic) {
  auto* A = cinn::common::BufferBuilder(Float(32), {100, 32})
                .set_align(32)
                .set_random()
                .Build();
  auto* B = cinn::common::BufferBuilder(Float(32), {100, 32})
                .set_align(32)
                .set_random()
                .Build();
  auto* C = cinn::common::BufferBuilder(Float(32), {100, 32})
                .set_align(32)
                .set_zero()
                .Build();

  float* Ad = reinterpret_cast<float*>(A->memory);
  float* Bd = reinterpret_cast<float*>(B->memory);
  float* Cd = reinterpret_cast<float*>(C->memory);
  ASSERT_EQ(C->num_elements(), A->num_elements());

  auto check = [&] {
    for (int i = 0; i < C->num_elements(); i++) {
      EXPECT_EQ(Ad[i] + Bd[i], Cd[i]);
    }
  };

  auto args = common::ArgsBuilder().Add(A).Add(B).Add(C).Build();

  LOG(INFO) << "test1 basic";
  add1(args.data(), args.size());
  check();

  LOG(INFO) << "test1 vectorize";
  add1_vectorize(args.data(), args.size());
  check();
}

TEST(test01, compute_at) {
  const int M = 100;
  const int N = 32;
  auto* A = cinn::common::BufferBuilder(Float(32), {M, N})
                .set_align(32)
                .set_random()
                .Build();
  auto* B = cinn::common::BufferBuilder(Float(32), {M, N})
                .set_align(32)
                .set_random()
                .Build();
  auto* C = cinn::common::BufferBuilder(Float(32), {M, N})
                .set_align(32)
                .set_zero()
                .Build();

  float* Ad = reinterpret_cast<float*>(A->memory);
  float* Bd = reinterpret_cast<float*>(B->memory);
  float* Cd = reinterpret_cast<float*>(C->memory);
  ASSERT_EQ(C->num_elements(), A->num_elements());

  auto check_add = [&] {
    for (int i = 0; i < M; i++) {
      for (int j = 0; j < N; j++) {
        ASSERT_NEAR(Ad[i * N + j] + Bd[i * N + j], Cd[i * N + j], 1e-5);
      }
    }
  };

  auto check_compute = [&] {
    for (int i = 0; i < M; i++) {
      for (int j = 0; j < N; j++) {
        float first = i > 0 ? Ad[(i - 1) * N + j] : 0.f;
        float last = i < M - 1 ? Ad[(i + 1) * N + j] : 0.f;
        float left = first + last + Ad[i * N + j] + Bd[i * N + j];
        ASSERT_NEAR(left, Cd[i * N + j], 1e-5);
      }
    }
  };

  auto reset = [&]() {
    for (int i = 0; i < M; i++) {
      for (int j = 0; j < N; j++) {
        Cd[i * N + j] = 0.f;
      }
    }
  };

  auto args = common::ArgsBuilder().Add(A).Add(B).Add(C).Build();

  LOG(INFO) << "test1 basic";
  add1(args.data(), args.size());
  check_add();
  reset();
  LOG(INFO) << "test1 compute_at";
  fn_compute_at(args.data(), args.size());
  check_compute();

  cinn_buffer_free(nullptr, A);
  cinn_buffer_free(nullptr, B);
  cinn_buffer_free(nullptr, C);
}

TEST(test01, compute_at_level1) {
  const int M = 100;
  const int N = 32;
  auto* A = cinn::common::BufferBuilder(Float(32), {M, N})
                .set_align(32)
                .set_random()
                .Build();
  auto* B = cinn::common::BufferBuilder(Float(32), {M, N})
                .set_align(32)
                .set_random()
                .Build();
  auto* C = cinn::common::BufferBuilder(Float(32), {M, N})
                .set_align(32)
                .set_zero()
                .Build();

  float* Ad = reinterpret_cast<float*>(A->memory);
  float* Bd = reinterpret_cast<float*>(B->memory);
  float* Cd = reinterpret_cast<float*>(C->memory);
  ASSERT_EQ(C->num_elements(), A->num_elements());

  auto check_add = [&] {
    for (int i = 0; i < M; i++) {
      for (int j = 0; j < N; j++) {
        ASSERT_NEAR(Ad[i * N + j] + Bd[i * N + j], Cd[i * N + j], 1e-5);
      }
    }
  };

  auto check_compute = [&] {
    for (int i = 0; i < M; i++) {
      for (int j = 0; j < N; j++) {
        float first = i > 0 ? Ad[(i - 1) * N + j] : 0.f;
        float last = i < M - 1 ? Ad[(i + 1) * N + j] : 0.f;
        float left = first + last + Ad[i * N + j] + Bd[i * N + j];
        ASSERT_NEAR(left, Cd[i * N + j], 1e-5);
      }
    }
  };

  auto args = common::ArgsBuilder().Add(A).Add(B).Add(C).Build();

  LOG(INFO) << "test1 basic";
  add1(args.data(), args.size());
  check_add();

  LOG(INFO) << "test1 compute_at_level1";
  fn_compute_at_level1(args.data(), args.size());
  check_compute();

  cinn_buffer_free(nullptr, A);
  cinn_buffer_free(nullptr, B);
  cinn_buffer_free(nullptr, C);
}

}  // namespace cinn

// include the generated C source code:
// @{
#include "test/cpp/cinn/test01_elementwise_add.cc"
#include "test/cpp/cinn/test01_elementwise_add_compute_at.cc"
#include "test/cpp/cinn/test01_elementwise_add_compute_at_level1.cc"
#include "test/cpp/cinn/test01_elementwise_add_vectorize.cc"
// @}
