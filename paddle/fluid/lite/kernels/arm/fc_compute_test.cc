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

#include "paddle/fluid/lite/kernels/arm/fc_compute.h"
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
void gemm_bias(const T* a, const int M, const int K, const T* b, const int K_,
               const int N, T* biases, T* c) {
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
  if (biases) {
    for (int m = 0; m < M; ++m) {
      for (int n = 0; n < N; ++n) {
        C(m, n) += biases[n];
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

TEST(fc_arm, retrive_op) {
  auto fc =
      KernelRegistry::Global().Create<TARGET(kARM), PRECISION(kFloat)>("fc");
  ASSERT_FALSE(fc.empty());
  ASSERT_TRUE(fc.front());
}

TEST(fc_arm, init) {
  FcCompute fc;
  ASSERT_EQ(fc.precision(), PRECISION(kFloat));
  ASSERT_EQ(fc.target(), TARGET(kARM));
}

TEST(fc_arm, compare_test) {
  using T = float;

  for (int m : {1, 2, 3, 4}) {
    for (int n : {1, 2, 3, 4}) {
      for (int k : {1, 2, 3, 4}) {
        for (bool with_bias : {true, false}) {
          VLOG(3) << "m: " << m << ", n: " << n << ", k: " << k
                  << (with_bias ? ", with bias" : "");
          lite::Tensor x, w, b, out, ref;

          x.Resize({m, k});
          w.Resize({k, n});
          b.Resize({1, n});
          out.Resize({m, n});
          ref.Resize({m, n});

          auto* x_data = x.mutable_data<T>();
          auto* w_data = w.mutable_data<T>();
          auto* b_data = with_bias ? b.mutable_data<T>() : nullptr;

          auto* out_data = out.mutable_data<T>();
          auto* ref_data = ref.mutable_data<T>();

          FillData<T>(x_data, x.dims().production());
          FillData<T>(w_data, w.dims().production());
          FillData<T>(out_data, out.dims().production(), 0, 0);
          FillData<T>(ref_data, ref.dims().production(), 0, 0);

          if (with_bias) {
            FillData<T>(b_data, b.dims().production());
          }

          FcCompute fc;
          operators::FcParam param;

          param.input = &x;
          param.w = &w;
          param.bias = with_bias ? &b : nullptr;
          param.output = &out;
          param.in_num_col_dims = 1;
          param.in_mat_dims = x.dims();

          DeviceInfo::Init();
          std::unique_ptr<KernelContext> ctx(new KernelContext);
          ctx->As<ARMContext>();
          fc.SetParam(param);
          fc.SetContext(std::move(ctx));
          fc.PrepareForRun();
          fc.Run();

          gemm_bias<T>(x_data, m, k, w_data, k, n, b_data, ref_data);

          for (int i = 0; i < out.dims().production(); i++) {
            EXPECT_NEAR(out_data[i], ref_data[i], 1e-3);
          }
        }
      }
    }
  }
}

TEST(fc_arm, num_col_dims) {
  using T = float;

  for (bool with_bias : {true, false}) {
    lite::Tensor x, w, b, out, ref;

    x.Resize({1, 2, 3});
    w.Resize({3, 4});
    b.Resize({1, 4});
    out.Resize({2, 4});
    ref.Resize({2, 4});

    auto* x_data = x.mutable_data<float>();
    auto* w_data = w.mutable_data<float>();
    auto* b_data = with_bias ? b.mutable_data<T>() : nullptr;

    auto* out_data = out.mutable_data<T>();
    auto* ref_data = ref.mutable_data<T>();

    FillData<T>(x_data, x.dims().production());
    FillData<T>(w_data, w.dims().production());
    FillData<T>(out_data, out.dims().production(), 0, 0);
    FillData<T>(ref_data, ref.dims().production(), 0, 0);
    if (with_bias) {
      FillData<T>(b_data, b.dims().production());
    }
    FcCompute fc;
    operators::FcParam param;
    param.input = &x;
    param.w = &w;
    param.bias = with_bias ? &b : nullptr;
    param.output = &out;
    param.in_num_col_dims = 2;
    param.in_mat_dims = x.dims();

    std::unique_ptr<KernelContext> ctx(new KernelContext);
    ctx->As<ARMContext>();
    DeviceInfo::Init();

    fc.SetParam(param);
    fc.SetContext(std::move(ctx));
    fc.PrepareForRun();
    fc.Run();

    gemm_bias<T>(x_data, 2, 3, w_data, 3, 4, b_data, ref_data);

    for (int i = 0; i < out.dims().production(); i++) {
      EXPECT_NEAR(out_data[i], ref_data[i], 1e-3);
    }
  }
}

}  // namespace arm
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

USE_LITE_KERNEL(fc, kARM, kFloat, kNCHW, def);
