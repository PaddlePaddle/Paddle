// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/lite/kernels/arm/mul_compute.h"
#include <gtest/gtest.h>
#include <algorithm>
#include <iostream>
#include <memory>
#include <random>
#include <utility>
#include <vector>
#include "paddle/fluid/lite/arm/math/funcs.h"
#include "paddle/fluid/lite/core/op_registry.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace arm {

#define A(i, j) a[i * lda + j]
#define B(i, j) b[i * ldb + j]
#define C(i, j) c[i * ldc + j]

template <typename T>
void mul_gemm(const T* a, const int M, const int K, const T* b, const int K_,
              const int N, T* c) {
  EXPECT_TRUE(K_ == K && M > 0 && N > 0 && K > 0);
  EXPECT_TRUE(a && b && c);
  const int lda = K;
  const int ldb = N;
  const int ldc = N;
  for (int m = 0; m < M; ++m) {
    for (int n = 0; n < N; ++n) {
      C(m, n) = 0.0f;
      for (int k = 0; k < K; ++k) {
        C(m, n) += A(m, k) * B(k, n);
      }
    }
  }
}

template <typename T>
void FillData(T* a, const int n, const T lower = static_cast<T>(-2.f),
              const T upper = static_cast<T>(2.f)) {
  static unsigned int seed = 100;
  std::mt19937 rng(seed++);
  std::uniform_real_distribution<double> uniform_dist(0, 1);
  for (int i = 0; i < n; ++i) {
    a[i] = static_cast<T>(uniform_dist(rng) * (upper - lower) + lower);
  }
}

TEST(mul_arm, retrive_op) {
  auto mul =
      KernelRegistry::Global().Create<TARGET(kARM), PRECISION(kFloat)>("mul");
  ASSERT_FALSE(mul.empty());
  ASSERT_TRUE(mul.front());
}

TEST(mul_arm, init) {
  MulCompute mul;
  ASSERT_EQ(mul.precision(), PRECISION(kFloat));
  ASSERT_EQ(mul.target(), TARGET(kARM));
}

TEST(mul_arm, compare_test) {
  using T = float;

  for (int m : {1, 2, 3, 4}) {
    for (int n : {1, 2, 3, 4}) {
      for (int k : {1, 2, 3, 4}) {
        VLOG(3) << "m: " << m << ", n: " << n << ", k: " << k;
        lite::Tensor x, y, out, ref;
        x.Resize({m, k});
        y.Resize({k, n});
        out.Resize({m, n});
        ref.Resize({m, n});

        auto* x_data = x.mutable_data<T>();
        auto* y_data = y.mutable_data<T>();
        auto* out_data = out.mutable_data<T>();
        auto* ref_data = ref.mutable_data<T>();

        FillData<T>(x_data, x.dims().production());
        FillData<T>(y_data, y.dims().production());
        FillData<T>(out_data, out.dims().production(), 0, 0);
        FillData<T>(ref_data, ref.dims().production(), 0, 0);

        MulCompute mul;
        operators::MulParam param;

        param.x = &x;
        param.y = &y;
        param.output = &out;

        DeviceInfo::Init();
        std::unique_ptr<KernelContext> ctx(new KernelContext);
        ctx->As<ARMContext>();
        mul.SetParam(param);
        mul.SetContext(std::move(ctx));
        mul.PrepareForRun();

        mul.Run();

        mul_gemm<T>(x_data, m, k, y_data, k, n, ref_data);

        for (int i = 0; i < out.dims().production(); i++) {
          EXPECT_NEAR(out_data[i], ref_data[i], 1e-3);
        }
      }
    }
  }
}

TEST(mul_arm, num_col_dims) {
  using T = float;

  lite::Tensor x, y, out, ref;
  x.Resize({2, 3, 4});
  y.Resize({3, 4, 5});
  out.Resize({2, 5});
  ref.Resize({2, 5});

  auto* x_data = x.mutable_data<T>();
  auto* y_data = y.mutable_data<T>();
  auto* out_data = out.mutable_data<T>();
  auto* ref_data = ref.mutable_data<T>();

  FillData<T>(x_data, x.dims().production());
  FillData<T>(y_data, y.dims().production());
  FillData<T>(out_data, out.dims().production());
  FillData<T>(ref_data, out.dims().production());

  MulCompute mul;
  operators::MulParam param;

  param.x = &x;
  param.y = &y;
  param.output = &out;
  param.x_num_col_dims = 1;
  param.y_num_col_dims = 2;

  DeviceInfo::Init();
  std::unique_ptr<KernelContext> ctx(new KernelContext);
  ctx->As<ARMContext>();
  mul.SetParam(param);
  mul.SetContext(std::move(ctx));
  mul.PrepareForRun();

  mul.Run();

  mul_gemm<T>(x_data, 2, 12, y_data, 12, 5, ref_data);

  for (int i = 0; i < out.dims().production(); i++) {
    EXPECT_NEAR(out_data[i], ref_data[i], 1e-3);
  }
}

}  // namespace arm
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

USE_LITE_KERNEL(mul, kARM, kFloat, kNCHW, def);
