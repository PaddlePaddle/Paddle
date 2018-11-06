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
#include <cmath>    // for exp
#include <cstring>  // for memcpy
#include <random>
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

constexpr int repeat = 20000;

inline double GetCurrentUS() {
  struct timeval time;
  gettimeofday(&time, NULL);
  return 1e+6 * time.tv_sec + time.tv_usec;
}

template <typename T>
void RandomVec(const int n, T* a, const T lower = static_cast<T>(-20.f),
               const T upper = static_cast<T>(20.f)) {
  static unsigned int seed = 100;
  std::mt19937 rng(seed++);
  std::uniform_real_distribution<double> uniform_dist(0, 1);
  for (int i = 0; i < n; ++i) {
    a[i] = static_cast<T>(uniform_dist(rng) * (upper - lower) + lower);
  }
}

void vrelu_ref(const int n, const float* x, float* y) {
  for (int i = 0; i < n; ++i) {
    y[i] = x[i] > 0.f ? x[i] : 0.f;
  }
}

#if defined __AVX__ || defined __AVX2__
void vrelu_intri8(const int n, const float* x, float* y) {
  __m256 tmp = _mm256_loadu_ps(x);
  tmp = _mm256_max_ps(tmp, _mm256_setzero_ps());
  _mm256_storeu_ps(y, tmp);
}
#endif

TEST(JitKernel, vrelu) {
  namespace jit = paddle::operators::math::jitkernel;
  for (int d : {7, 8, 15, 16, 30, 256, 512}) {
    std::vector<float> x(d);
    std::vector<float> zref(d), ztgt(d);
    RandomVec<float>(d, x.data(), -10.f, 1.f);
    const auto& ker =
        jit::KernelPool::Instance().template Get<jit::VReluKernel<float>>(d);
    const float* x_data = x.data();
    float* ztgt_data = ztgt.data();
    float* zref_data = zref.data();
    auto trefs = GetCurrentUS();
    for (int i = 0; i < repeat; ++i) {
      vrelu_ref(d, x_data, zref_data);
    }
    auto trefe = GetCurrentUS();
#if defined __AVX__ || defined __AVX2__
    if (d == 8) {
      auto si0 = GetCurrentUS();
      for (int i = 0; i < repeat; ++i) {
        vrelu_intri8(d, x_data, zref_data);
      }
      auto si1 = GetCurrentUS();
      VLOG(3) << "Vec size 8 intr takes: " << (si1 - si0) / repeat;
    }
#endif
    auto ttgts = GetCurrentUS();
    for (int i = 0; i < repeat; ++i) {
      ker->Compute(x_data, ztgt_data);
    }
    auto ttgte = GetCurrentUS();
    VLOG(3) << "Vec size " << d << ": refer takes: " << (trefe - trefs) / repeat
            << " us, tgt takes: " << (ttgte - ttgts) / repeat;
    for (int i = 0; i < d; ++i) {
      EXPECT_NEAR(ztgt_data[i], zref_data[i], 1e-3);
    }
  }
}

void vaddbias_ref(const int n, const float a, const float* x, float* y) {
  for (int i = 0; i < n; ++i) {
    y[i] = x[i] + a;
  }
}

TEST(JitKernel, vaddbias) {
  namespace jit = paddle::operators::math::jitkernel;
  for (int d : {7, 8, 15, 16, 30, 64, 100, 128, 256}) {
    std::vector<float> x(d);
    std::vector<float> zref(d), ztgt(d);
    RandomVec<float>(d, x.data(), -2.f, 2.f);
    const auto& ker =
        jit::KernelPool::Instance().template Get<jit::VAddBiasKernel<float>>(d);
    const float a = 2.f;
    const float* x_data = x.data();
    float* ztgt_data = ztgt.data();
    float* zref_data = zref.data();
    auto trefs = GetCurrentUS();
    for (int i = 0; i < repeat; ++i) {
      vaddbias_ref(d, a, x_data, zref_data);
    }
    auto trefe = GetCurrentUS();
    auto ttgts = GetCurrentUS();
    for (int i = 0; i < repeat; ++i) {
      ker->Compute(a, x_data, ztgt_data);
    }
    auto ttgte = GetCurrentUS();

    VLOG(3) << "Vec size " << d << ": refer takes: " << (trefe - trefs) / repeat
            << " us, tgt takes: " << (ttgte - ttgts) / repeat;
    for (int i = 0; i < d; ++i) {
      EXPECT_NEAR(ztgt_data[i], zref_data[i], 1e-3);
    }
  }
}

void vexp_ref(const int n, const float* x, float* y) {
  for (int i = 0; i < n; ++i) {
    y[i] = std::exp(x[i]);
  }
}

#ifdef PADDLE_WITH_MKLML
void vexp_mkl(const int n, const float* x, float* y) {
  paddle::platform::dynload::vsExp(n, x, y);
}
#endif

TEST(JitKernel, vexp) {
  namespace jit = paddle::operators::math::jitkernel;
  for (int d : {7, 8, 15, 16, 30, 128, 256}) {
    std::vector<float> x(d);
    std::vector<float> zref(d), ztgt(d);
    RandomVec<float>(d, x.data(), -2.f, 2.f);
    const auto& ker =
        jit::KernelPool::Instance().template Get<jit::VExpKernel<float>>(d);
    const float* x_data = x.data();
    float* ztgt_data = ztgt.data();
    float* zref_data = zref.data();
    auto trefs = GetCurrentUS();
    for (int i = 0; i < repeat; ++i) {
      vexp_ref(d, x_data, zref_data);
    }
    auto trefe = GetCurrentUS();

#ifdef PADDLE_WITH_MKLML
    auto tmkls = GetCurrentUS();
    for (int i = 0; i < repeat; ++i) {
      vexp_mkl(d, x_data, zref_data);
    }
    auto tmkle = GetCurrentUS();
#endif

    auto ttgts = GetCurrentUS();
    for (int i = 0; i < repeat; ++i) {
      ker->Compute(x_data, ztgt_data);
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

inline float _sigmoid(float x) {
  const float min = SIGMOID_THRESHOLD_MIN;
  const float max = SIGMOID_THRESHOLD_MAX;
  float tmp = (x < min) ? min : ((x > max) ? max : x);
  return 1.f / (1.f + std::exp(-tmp));
}

void vsigmoid_ref(const int n, const float* x, float* y) {
  for (int i = 0; i < n; ++i) {
    y[i] = _sigmoid(x[i]);
  }
}

void vsigmoid_better(
    const std::shared_ptr<
        const paddle::operators::math::jitkernel::VExpKernel<float>>& vexp,
    const int n, const float* x, float* y) {
  const float min = SIGMOID_THRESHOLD_MIN;
  const float max = SIGMOID_THRESHOLD_MAX;
  for (int i = 0; i < n; ++i) {
    y[i] = (x[i] < min) ? min : ((x[i] > max) ? max : x[i]);
    y[i] = 0.f - y[i];
  }
  vexp->Compute(y, y);
  for (int i = 0; i < n; ++i) {
    y[i] = 1.f / (1.f + y[i]);
  }
}

TEST(JitKernel, vsigmoid) {
  namespace jit = paddle::operators::math::jitkernel;
  for (int d : {7, 8, 15, 16, 30, 32, 64, 100, 128, 256}) {
    std::vector<float> x(d);
    std::vector<float> zref(d), ztgt(d);
    RandomVec<float>(d, x.data(), -2.f, 2.f);
    const auto& ker =
        jit::KernelPool::Instance().template Get<jit::VSigmoidKernel<float>>(d);
    const auto& vexp =
        jit::KernelPool::Instance().template Get<jit::VExpKernel<float>>(d);
    const float* x_data = x.data();
    float* ztgt_data = ztgt.data();
    float* zref_data = zref.data();
    auto tmkls = GetCurrentUS();
    for (int i = 0; i < repeat; ++i) {
      vsigmoid_better(vexp, d, x_data, zref_data);
    }
    auto tmkle = GetCurrentUS();
    auto trefs = GetCurrentUS();
    for (int i = 0; i < repeat; ++i) {
      vsigmoid_ref(d, x_data, zref_data);
    }
    auto trefe = GetCurrentUS();
    auto ttgts = GetCurrentUS();
    for (int i = 0; i < repeat; ++i) {
      ker->Compute(x_data, ztgt_data);
    }
    auto ttgte = GetCurrentUS();

    VLOG(3) << "Vec size " << d << ": refer takes: " << (trefe - trefs) / repeat
            << " us, better(jit exp) takes: " << (tmkle - tmkls) / repeat
            << " us, tgt takes: " << (ttgte - ttgts) / repeat;
    for (int i = 0; i < d; ++i) {
      EXPECT_NEAR(ztgt_data[i], zref_data[i], 1e-3);
    }
  }
}

inline float _tanh(float x) { return 2.f * _sigmoid(2.f * x) - 1.f; }

void vtanh_ref(const int n, const float* x, float* y) {
  for (int i = 0; i < n; ++i) {
    y[i] = _tanh(x[i]);
  }
}

void vtanh_better(
    const std::shared_ptr<
        const paddle::operators::math::jitkernel::VScalKernel<float>>& vscal,
    const std::shared_ptr<
        const paddle::operators::math::jitkernel::VSigmoidKernel<float>>&
        vsigmoid,
    const std::shared_ptr<
        const paddle::operators::math::jitkernel::VAddBiasKernel<float>>&
        vaddbias,
    const int n, const float* x, float* y) {
  vscal->Compute(2.f, x, y);
  vsigmoid->Compute(y, y);
  vscal->Compute(2.f, y);
  vaddbias->Compute(-1.f, y, y);
}

TEST(JitKernel, vtanh) {
  namespace jit = paddle::operators::math::jitkernel;
  for (int d : {7, 8, 15, 16, 30, 32, 64, 100, 128, 256}) {
    std::vector<float> x(d);
    std::vector<float> zref(d), ztgt(d);
    RandomVec<float>(d, x.data(), -2.f, 2.f);
    const auto& ker =
        jit::KernelPool::Instance().template Get<jit::VTanhKernel<float>>(d);
    const auto& vscal =
        jit::KernelPool::Instance().template Get<jit::VScalKernel<float>>(d);
    const auto& vsigmoid =
        jit::KernelPool::Instance().template Get<jit::VSigmoidKernel<float>>(d);
    const auto& vaddbias =
        jit::KernelPool::Instance().template Get<jit::VAddBiasKernel<float>>(d);
    const float* x_data = x.data();
    float* ztgt_data = ztgt.data();
    float* zref_data = zref.data();
    auto tmkls = GetCurrentUS();
    for (int i = 0; i < repeat; ++i) {
      vtanh_better(vscal, vsigmoid, vaddbias, d, x_data, zref_data);
    }
    auto tmkle = GetCurrentUS();
    auto trefs = GetCurrentUS();
    for (int i = 0; i < repeat; ++i) {
      vtanh_ref(d, x_data, zref_data);
    }
    auto trefe = GetCurrentUS();
    auto ttgts = GetCurrentUS();
    for (int i = 0; i < repeat; ++i) {
      ker->Compute(x_data, ztgt_data);
    }
    auto ttgte = GetCurrentUS();

    VLOG(3) << "Vec size " << d << ": refer takes: " << (trefe - trefs) / repeat
            << " us, better(jit exp) takes: " << (tmkle - tmkls) / repeat
            << " us, tgt takes: " << (ttgte - ttgts) / repeat;
    for (int i = 0; i < d; ++i) {
      EXPECT_NEAR(ztgt_data[i], zref_data[i], 1e-3);
    }
  }
}

void lstm_ctht_ref(
    const std::shared_ptr<
        const paddle::operators::math::jitkernel::VSigmoidKernel<float>>&
        vsigmoid_3d,
    const std::shared_ptr<
        const paddle::operators::math::jitkernel::VTanhKernel<float>>& vtanh_d,
    const std::shared_ptr<
        const paddle::operators::math::jitkernel::VExpKernel<float>>& vexp_1,
    const int d, float* gates, const float* ct_1, float* ct, float* ht) {
  vsigmoid_3d->Compute(gates + d, gates + d);
  vtanh_d->Compute(gates, gates);
  const float *i = gates + d, *f = gates + d * 2, *o = gates + d * 3;
  const float min = SIGMOID_THRESHOLD_MIN;
  const float max = SIGMOID_THRESHOLD_MAX;
  for (int k = 0; k < d; ++k) {
    // C_t = C_t-1 * fgated + cand_gated * igated
    ct[k] = ct_1[k] * f[k] + gates[k] * i[k];
    // H_t = act_cell(C_t) * ogated
    float tmp = ct[k] * 2;
    tmp = 0.f - ((tmp < min) ? min : ((tmp > max) ? max : tmp));
    vexp_1->Compute(&tmp, &tmp);
    tmp = 2.f / (1.f + tmp) - 1.f;
    ht[k] = tmp * o[k];
  }
}

void lstm_ctht_better(
    const std::shared_ptr<
        const paddle::operators::math::jitkernel::VSigmoidKernel<float>>&
        vsigmoid_3d,
    const std::shared_ptr<
        const paddle::operators::math::jitkernel::VTanhKernel<float>>& vtanh_d,
    const std::shared_ptr<
        const paddle::operators::math::jitkernel::VMulKernel<float>>& vmul_d,
    const std::shared_ptr<
        const paddle::operators::math::jitkernel::VAddKernel<float>>& vadd_d,
    const int d, float* gates, const float* ct_1, float* ct, float* ht) {
  int d2 = d * 2;
  vsigmoid_3d->Compute(gates + d, gates + d);
  vtanh_d->Compute(gates, gates);
  vmul_d->Compute(gates, gates + d, gates + d, d);
  vmul_d->Compute(ct_1, gates + d2, gates + d2, d);
  vadd_d->Compute(gates + d, gates + d2, ct);
  /* H_t = act_cell(C_t) * ogated */
  vtanh_d->Compute(ct, gates + d2);
  vmul_d->Compute(gates + d2, gates + d * 3, ht, d);
}

TEST(JitKernel, lstm) {
  namespace jit = paddle::operators::math::jitkernel;
  for (int d : {7, 8, 15, 16, 30, 32, 64, 100}) {
    int d4 = d * 4;
    int d3 = d * 3;
    std::vector<float> x(d4), xref(d4);
    std::vector<float> ct_1(d), ct_tgt(d), ht_tgt(d);
    std::vector<float> ct_ref(d), ht_ref(d);
    RandomVec<float>(d4, x.data(), -2.f, 2.f);
    RandomVec<float>(d, ct_1.data(), -2.f, 2.f);
    memcpy(xref.data(), x.data(), sizeof(float) * d4);
    std::string act_gate = "sigmoid", act_cand = "tanh", act_cell = "tanh";
    const auto& ker =
        jit::KernelPool::Instance()
            .template Get<jit::LSTMKernel<float>, const std::string&,
                          const std::string&, const std::string&>(
                act_gate, act_cand, act_cell, d, false);
    // below kernels are used to compute refer
    const auto& vsigmoid_3d =
        jit::KernelPool::Instance().template Get<jit::VSigmoidKernel<float>>(
            d3);
    const auto& vtanh_d =
        jit::KernelPool::Instance().template Get<jit::VTanhKernel<float>>(d);
    const auto& vexp_1 =
        jit::KernelPool::Instance().template Get<jit::VExpKernel<float>>(1);
    const auto& vmul_d =
        jit::KernelPool::Instance().template Get<jit::VMulKernel<float>>(d);
    const auto& vadd_d =
        jit::KernelPool::Instance().template Get<jit::VAddKernel<float>>(d);

    float* x_data = x.data();
    float* xref_data = xref.data();
    const float* ct_1_data = ct_1.data();
    float* ct_tgt_data = ct_tgt.data();
    float* ht_tgt_data = ht_tgt.data();
    float* ct_ref_data = ct_ref.data();
    float* ht_ref_data = ht_ref.data();
    // compute once to check correctness
    lstm_ctht_ref(vsigmoid_3d, vtanh_d, vexp_1, d, xref_data, ct_1_data,
                  ct_ref_data, ht_ref_data);
    ker->ComputeCtHt(x_data, ct_1_data, ct_tgt_data, ht_tgt_data);
    for (int i = 0; i < d; ++i) {
      EXPECT_NEAR(ct_tgt_data[i], ct_ref_data[i], 1e-3);
      EXPECT_NEAR(ht_tgt_data[i], ht_ref_data[i], 1e-3);
    }

    auto tmkls = GetCurrentUS();
    for (int i = 0; i < repeat; ++i) {
      lstm_ctht_better(vsigmoid_3d, vtanh_d, vmul_d, vadd_d, d, xref_data,
                       ct_1_data, ct_ref_data, ht_ref_data);
    }
    auto tmkle = GetCurrentUS();
    auto trefs = GetCurrentUS();
    for (int i = 0; i < repeat; ++i) {
      lstm_ctht_ref(vsigmoid_3d, vtanh_d, vexp_1, d, xref_data, ct_1_data,
                    ct_ref_data, ht_ref_data);
    }
    auto trefe = GetCurrentUS();
    auto ttgts = GetCurrentUS();
    for (int i = 0; i < repeat; ++i) {
      ker->ComputeCtHt(x_data, ct_1_data, ct_tgt_data, ht_tgt_data);
    }
    auto ttgte = GetCurrentUS();
    VLOG(3) << "Vec size " << d << ": refer takes: " << (trefe - trefs) / repeat
            << " us, better(jit) takes: " << (tmkle - tmkls) / repeat
            << " us, tgt takes: " << (ttgte - ttgts) / repeat;
  }
}

void vscal_ref(const int n, const float a, const float* x, float* y) {
  for (int i = 0; i < n; ++i) {
    y[i] = a * x[i];
  }
}
void vscal_inp_ref(const int n, const float a, float* x) {
  for (int i = 0; i < n; ++i) {
    x[i] = a * x[i];
  }
}
#if defined __AVX__ || defined __AVX2__
void vscal_intri8(const int n, const float a, const float* x, float* y) {
  __m256 tmp;
  __m256 scalar = _mm256_set1_ps(a);
  tmp = _mm256_loadu_ps(x);
  tmp = _mm256_mul_ps(tmp, scalar);
  _mm256_storeu_ps(y, tmp);
}
void vscal_inp_intri8(const int n, const float a, float* x) {
  __m256 tmp;
  __m256 scalar = _mm256_set1_ps(a);
  tmp = _mm256_loadu_ps(x);
  tmp = _mm256_mul_ps(tmp, scalar);
  _mm256_storeu_ps(x, tmp);
}
#endif

#ifdef PADDLE_WITH_MKLML
void vscal_inp_mkl(const int n, const float a, float* x) {
  paddle::platform::dynload::cblas_sscal(n, a, x, 1);
}
#endif

TEST(JitKernel, vscal) {
  namespace jit = paddle::operators::math::jitkernel;
  for (int d : {7, 8, 15, 16, 30, 256, 512}) {
    std::vector<float> x(d), y(d);
    std::vector<float> zref(d), ztgt(d);
    RandomVec<float>(d, x.data());
    std::memcpy(y.data(), x.data(), sizeof(float) * d);
    float a = 2.f;
    const auto& ker =
        jit::KernelPool::Instance().template Get<jit::VScalKernel<float>>(d);
    const float* x_data = x.data();
    float* y_data = y.data();
    float* ztgt_data = ztgt.data();
    float* zref_data = zref.data();
    auto trefs = GetCurrentUS();
    for (int i = 0; i < repeat; ++i) {
      vscal_ref(d, a, x_data, zref_data);
    }
    auto trefe = GetCurrentUS();
    auto trefs1 = GetCurrentUS();
    for (int i = 0; i < repeat; ++i) {
      vscal_inp_ref(d, a, y_data);
    }
    auto trefe1 = GetCurrentUS();

#ifdef PADDLE_WITH_MKLML
    auto tmkls = GetCurrentUS();
    for (int i = 0; i < repeat; ++i) {
      vscal_inp_mkl(d, a, y_data);
    }
    auto tmkle = GetCurrentUS();
#endif

#if defined __AVX__ || defined __AVX2__
    if (d == 8) {
      auto si0 = GetCurrentUS();
      for (int i = 0; i < repeat; ++i) {
        vscal_intri8(d, a, x_data, zref_data);
      }
      auto si1 = GetCurrentUS();
      auto si2 = GetCurrentUS();
      for (int i = 0; i < repeat; ++i) {
        vscal_inp_intri8(d, a, y_data);
      }
      auto si3 = GetCurrentUS();
      VLOG(3) << "Vec size 8 intr takes: " << (si1 - si0) / repeat
              << " us, inplace: " << (si3 - si2) / repeat;
    }
#endif

    auto ttgts = GetCurrentUS();
    for (int i = 0; i < repeat; ++i) {
      ker->Compute(a, x_data, ztgt_data);
    }
    auto ttgte = GetCurrentUS();
    auto ttgts1 = GetCurrentUS();
    for (int i = 0; i < repeat; ++i) {
      ker->Compute(a, y_data);
    }
    auto ttgte1 = GetCurrentUS();
    VLOG(3) << "Vec size " << d << ": refer takes: " << (trefe - trefs) / repeat
            << " us, inplace takes: " << (trefe1 - trefs1) / repeat
#ifdef PADDLE_WITH_MKLML
            << " us, mkl inplace takes: " << (tmkle - tmkls) / repeat << " us, "
#else
            << " us, "
#endif
            << "tgt takes: " << (ttgte - ttgts) / repeat
            << "us, tgt inplace takes: " << (ttgte1 - ttgts1) / repeat;
    for (int i = 0; i < d; ++i) {
      EXPECT_NEAR(ztgt_data[i], zref_data[i], 1e-3);
    }
  }
}

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
  for (int d : {7, 8, 15, 16, 20, 30, 256, 512, 1000, 1024}) {
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
      ker->Compute(x_data, y_data, ztgt_data, d);
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

void vadd_ref(const int n, const float* x, const float* y, float* z) {
  for (int i = 0; i < n; ++i) {
    z[i] = x[i] + y[i];
  }
}

#if defined __AVX__ || defined __AVX2__
void vadd_intri8(const int n, const float* x, const float* y, float* z) {
  __m256 tmpx, tmpy;
  tmpx = _mm256_loadu_ps(x);
  tmpy = _mm256_loadu_ps(y);
  tmpx = _mm256_add_ps(tmpx, tmpy);
  _mm256_storeu_ps(z, tmpx);
}
#endif

#ifdef PADDLE_WITH_MKLML
void vadd_mkl(const int n, const float* x, const float* y, float* z) {
  paddle::platform::dynload::vsAdd(n, x, y, z);
}
#endif

TEST(JitKernel, vadd) {
  namespace jit = paddle::operators::math::jitkernel;
  for (int d : {7, 8, 15, 16, 30, 256, 512}) {
    std::vector<float> x(d), y(d);
    std::vector<float> zref(d), ztgt(d);
    RandomVec<float>(d, x.data());
    RandomVec<float>(d, y.data());
    const auto& ker =
        jit::KernelPool::Instance().template Get<jit::VAddKernel<float>>(d);
    const float* x_data = x.data();
    const float* y_data = y.data();
    float* ztgt_data = ztgt.data();
    float* zref_data = zref.data();
    auto trefs = GetCurrentUS();
    for (int i = 0; i < repeat; ++i) {
      vadd_ref(d, x_data, y_data, zref_data);
    }
    auto trefe = GetCurrentUS();

#ifdef PADDLE_WITH_MKLML
    auto tmkls = GetCurrentUS();
    for (int i = 0; i < repeat; ++i) {
      vadd_mkl(d, x_data, y_data, zref_data);
    }
    auto tmkle = GetCurrentUS();
#endif

#if defined __AVX__ || defined __AVX2__
    if (d == 8) {
      auto si0 = GetCurrentUS();
      for (int i = 0; i < repeat; ++i) {
        vadd_intri8(d, x_data, y_data, zref_data);
      }
      auto si1 = GetCurrentUS();
      VLOG(3) << "Vec size 8 intr takes: " << (si1 - si0) / repeat;
    }
#endif

    auto ttgts = GetCurrentUS();
    for (int i = 0; i < repeat; ++i) {
      ker->Compute(x_data, y_data, ztgt_data);
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

void vaddrelu_ref(const int n, const float* x, const float* y, float* z) {
  for (int i = 0; i < n; ++i) {
    z[i] = x[i] + y[i];
    z[i] = z[i] > 0 ? z[i] : 0;
  }
}
void vaddrelu_better(
    const std::shared_ptr<
        const paddle::operators::math::jitkernel::VAddKernel<float>>& vadd,
    const std::shared_ptr<
        const paddle::operators::math::jitkernel::VReluKernel<float>>& vrelu,
    const float* x, const float* y, float* z) {
  vadd->Compute(x, y, z);
  vrelu->Compute(z, z);
}

TEST(JitKernel, vaddrelu) {
  namespace jit = paddle::operators::math::jitkernel;
  for (int d : {7, 8, 15, 16, 30, 256, 512}) {
    std::vector<float> x(d), y(d);
    std::vector<float> zref(d), ztgt(d);
    RandomVec<float>(d, x.data());
    RandomVec<float>(d, y.data());
    const auto& ker =
        jit::KernelPool::Instance().template Get<jit::VAddReluKernel<float>>(d);
    const auto& vadd =
        jit::KernelPool::Instance().template Get<jit::VAddKernel<float>>(d);
    const auto& vrelu =
        jit::KernelPool::Instance().template Get<jit::VReluKernel<float>>(d);
    const float* x_data = x.data();
    const float* y_data = y.data();
    float* ztgt_data = ztgt.data();
    float* zref_data = zref.data();
    auto trefs = GetCurrentUS();
    for (int i = 0; i < repeat; ++i) {
      vadd_ref(d, x_data, y_data, zref_data);
    }
    auto trefe = GetCurrentUS();
    auto tmkls = GetCurrentUS();
    for (int i = 0; i < repeat; ++i) {
      vaddrelu_better(vadd, vrelu, x_data, y_data, zref_data);
    }
    auto tmkle = GetCurrentUS();
    auto ttgts = GetCurrentUS();
    for (int i = 0; i < repeat; ++i) {
      ker->Compute(x_data, y_data, ztgt_data);
    }
    auto ttgte = GetCurrentUS();
    VLOG(3) << "Vec size " << d << ": refer takes: " << (trefe - trefs) / repeat
            << " us, better takes: " << (tmkle - tmkls) / repeat << " us, "
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
          .template Get<jit::LSTMKernel<float>, const std::string&,
                        const std::string&, const std::string&>(
              act_gate, act_cand, act_cell, frame_size, false);
  const auto& plstm2 =
      jit::KernelPool::Instance()
          .template Get<jit::LSTMKernel<float>, const std::string&,
                        const std::string&, const std::string&>(
              act_gate, act_cand, act_cell, frame_size, false);
  const auto& peephole =
      jit::KernelPool::Instance()
          .template Get<jit::LSTMKernel<float>, const std::string&,
                        const std::string&, const std::string&>(
              act_gate, act_cand, act_cell, frame_size, true);
  EXPECT_TRUE(plstm1 != peephole);

  const auto& pvmul_f =
      jit::KernelPool::Instance().template Get<jit::VMulKernel<float>>(4);
  EXPECT_TRUE(std::dynamic_pointer_cast<const jit::Kernel>(plstm2) !=
              std::dynamic_pointer_cast<const jit::Kernel>(pvmul_f));

  const auto& pvmul_d =
      jit::KernelPool::Instance().template Get<jit::VMulKernel<double>>(4);
  EXPECT_TRUE(std::dynamic_pointer_cast<const jit::Kernel>(pvmul_f) !=
              std::dynamic_pointer_cast<const jit::Kernel>(pvmul_d));

  const auto& pvmul_from_key = jit::KernelPool::Instance().Get("vmulfjit4");
  EXPECT_EQ(pvmul_f, pvmul_from_key);
  const auto& pvmul_from_key2 = jit::KernelPool::Instance().Get("vmulfjit");
  EXPECT_TRUE(pvmul_from_key2 == nullptr);
}
