/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/operators/math/jit_kernel.h"
#include <sys/time.h>
#include <string>
#include <vector>
#include "gflags/gflags.h"
#include "glog/logging.h"
#include "gtest/gtest.h"

#ifdef PADDLE_WITH_MKLML
#include "paddle/fluid/platform/dynload/mklml.h"
#endif

#ifdef __AVX__
#include <immintrin.h>
#endif

inline double GetCurrentUS() {
  struct timeval time;
  gettimeofday(&time, NULL);
  return 1e+6 * time.tv_sec + time.tv_usec;
}

template <typename T>
void RandomVec(const int n, T* a) {
  static unsigned int seed = 100;
  std::mt19937 rng(seed++);
  std::uniform_real_distribution<double> uniform_dist(0, 1);
  const T lower = static_cast<T>(-20.f);
  const T upper = static_cast<T>(20.f);
  for (int i = 0; i < n; ++i) {
    a[i] = static_cast<T>(uniform_dist(rng) * (upper - lower) + lower);
  }
}

constexpr int repeat = 20000;

void vmul_ref(const int n, const float* x, const float* y, float* z) {
  for (int i = 0; i < n; ++i) {
    z[i] = x[i] * y[i];
  }
}

#if defined __AVX__ || defined __AVX2__
void vmul_intri8(const int n, const float* x, const float* y, float* z) {
  __m256 tmpx, tmpy;
  tmpx = _mm256_loadu_ps(x);
  tmpy = _mm256_loadu_ps(y);
  tmpx = _mm256_mul_ps(tmpx, tmpy);
  _mm256_storeu_ps(z, tmpx);
}
#endif

#ifdef PADDLE_WITH_MKLML
void vmul_mkl(const int n, const float* x, const float* y, float* z) {
  paddle::platform::dynload::vsMul(n, x, y, z);
}
#endif

TEST(JitKernel, vmul) {
  namespace jit = paddle::operators::math::jitkernel;
  for (int d : {7, 8, 15, 16, 30, 256, 512}) {
    std::vector<float> x(d), y(d);
    std::vector<float> zref(d), ztgt(d);
    RandomVec<float>(d, x.data());
    RandomVec<float>(d, y.data());
    const auto& ker =
        jit::KernelPool::Instance().template Get<jit::VMulKernel<float>>(d);

    const float* x_data = x.data();
    const float* y_data = y.data();
    float* ztgt_data = ztgt.data();
    float* zref_data = zref.data();

    auto trefs = GetCurrentUS();
    for (int i = 0; i < repeat; ++i) {
      vmul_ref(d, x_data, y_data, zref_data);
    }
    auto trefe = GetCurrentUS();

#ifdef PADDLE_WITH_MKLML
    auto tmkls = GetCurrentUS();
    for (int i = 0; i < repeat; ++i) {
      vmul_mkl(d, x_data, y_data, zref_data);
    }
    auto tmkle = GetCurrentUS();
#endif

#if defined __AVX__ || defined __AVX2__
    if (d == 8) {
      auto si0 = GetCurrentUS();
      for (int i = 0; i < repeat; ++i) {
        vmul_intri8(d, x_data, y_data, zref_data);
      }
      auto si1 = GetCurrentUS();
      VLOG(3) << "Vec size 8 intr takes: " << (si1 - si0) / repeat;
    }
#endif

    auto ttgts = GetCurrentUS();
    for (int i = 0; i < repeat; ++i) {
      ker->Compute(d, x_data, y_data, ztgt_data);
    }
    auto ttgte = GetCurrentUS();

    VLOG(3) << "Vec size " << d << ": refer takes: " << (trefe - trefs) / repeat
#ifdef PADDLE_WITH_MKLML
            << " us, mkl takes: " << (tmkle - tmkls) / repeat << " us, "
#else
            << " us, "
#endif
            << "tgt takes: " << (ttgte - ttgts) / repeat;
    for (int i = 0; i < d; ++i) {
      EXPECT_NEAR(ztgt_data[i], zref_data[i], 1e-3);
    }
  }
}

TEST(JitKernel, pool) {
  namespace jit = paddle::operators::math::jitkernel;
  const int frame_size = 4;
  std::string act_gate = "sigmoid", act_cand = "tanh", act_cell = "tanh";
  const auto& plstm1 =
      jit::KernelPool::Instance()
          .template Get<jit::LSTMKernel<float>, int, const std::string&,
                        const std::string&, const std::string&>(
              frame_size, act_gate, act_cand, act_cell);
  const auto& plstm2 =
      jit::KernelPool::Instance()
          .template Get<jit::LSTMKernel<float>, int, const std::string&,
                        const std::string&, const std::string&>(
              frame_size, act_gate, act_cand, act_cell);
  EXPECT_EQ(plstm1, plstm2);

  const auto& pvmul_f =
      jit::KernelPool::Instance().template Get<jit::VMulKernel<float>>(4);
  EXPECT_TRUE(std::dynamic_pointer_cast<jit::Kernel>(plstm2) !=
              std::dynamic_pointer_cast<jit::Kernel>(pvmul_f));

  const auto& pvmul_d =
      jit::KernelPool::Instance().template Get<jit::VMulKernel<double>>(4);
  EXPECT_TRUE(std::dynamic_pointer_cast<jit::Kernel>(pvmul_f) !=
              std::dynamic_pointer_cast<jit::Kernel>(pvmul_d));

  const auto& pvmul_from_key = jit::KernelPool::Instance().Get("vmulf4");
  EXPECT_TRUE(pvmul_f == pvmul_from_key);
  const auto& pvmul_from_key2 = jit::KernelPool::Instance().Get("vmulf5");
  EXPECT_TRUE(pvmul_from_key2 == nullptr);
}
