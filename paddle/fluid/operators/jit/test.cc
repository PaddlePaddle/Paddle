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

#include <iostream>
#include <random>

#include "gflags/gflags.h"
#include "glog/logging.h"
#include "gtest/gtest.h"
#include "paddle/fluid/operators/jit/kernels.h"
#include "paddle/fluid/platform/cpu_info.h"
#include "paddle/fluid/platform/place.h"

DEFINE_double(acc, 1e-5, "Test accuracy threshold.");

template <typename T>
void RandomVec(const int n, T* a, const T lower = static_cast<T>(-2.f),
               const T upper = static_cast<T>(2.f)) {
  static unsigned int seed = 100;
  std::mt19937 rng(seed++);
  std::uniform_real_distribution<double> uniform_dist(0, 1);
  for (int i = 0; i < n; ++i) {
    a[i] = static_cast<T>(uniform_dist(rng) * (upper - lower) + lower);
  }
}

template <typename T>
void ExpectEQ(const T* target, const T* refer, size_t n) {
  if (std::is_floating_point<T>::value) {
    for (size_t i = 0; i < n; ++i) {
      EXPECT_NEAR(target[i], refer[i], FLAGS_acc) << " at index : " << i;
    }
  } else {
    for (size_t i = 0; i < n; ++i) {
      EXPECT_EQ(target[i], refer[i]) << " at index : " << i;
    }
  }
}

std::vector<int> TestSizes() {
  std::vector<int> s;
  for (int i = 1; i < 32; ++i) {
    s.push_back(i);
  }
  // test some large size
  s.push_back(100);
  s.push_back(1000);
  s.push_back(2000);
  return s;
}

namespace jit = paddle::operators::jit;
using CPUPlace = paddle::platform::CPUPlace;

template <typename KernelTuple, typename PlaceType, typename Tester,
          typename... Args>
void TestAllImpls(const typename KernelTuple::attr_type& attr,
                  const Tester& verifier, const Args&... args) {
  auto funcs = jit::GetAllCandidateFuncsWithTypes<KernelTuple, PlaceType>(attr);
  for (auto f : funcs) {
    VLOG(10) << "Test Kernel " << f.first;
    verifier(f.second, args...);
  }
}

template <typename KernelTuple, typename PlaceType>
void TestKernelXYZN() {
  using T = typename KernelTuple::data_type;
  VLOG(10) << "Test JITKernel: " << jit::to_string(KernelTuple::kernel_type);
  for (int d : TestSizes()) {
    auto ref = jit::GetReferFunc<KernelTuple>();
    EXPECT_TRUE(ref != nullptr);

    std::vector<T> x(d), y(d), zref(d);
    RandomVec<T>(d, x.data());
    RandomVec<T>(d, y.data());

    std::vector<T> xinp(d), yinp(d);  // inplace test
    std::copy(x.begin(), x.end(), xinp.begin());
    std::copy(y.begin(), y.end(), yinp.begin());

    const T* x_data = x.data();
    const T* y_data = y.data();
    T* zref_data = zref.data();
    T* xinp_data = xinp.data();
    T* yinp_data = yinp.data();

    // test refer code inplace
    ref(x_data, y_data, zref_data, d);
    ref(x_data, yinp_data, yinp_data, d);
    ref(xinp_data, y_data, xinp_data, d);
    ExpectEQ<T>(xinp_data, zref_data, d);
    ExpectEQ<T>(yinp_data, zref_data, d);

    auto verifier = [](const typename KernelTuple::func_type tgt,
                       const std::vector<T>& x, const std::vector<T>& y,
                       const std::vector<T>& zref) {
      EXPECT_TRUE(tgt != nullptr);
      EXPECT_EQ(zref.size(), x.size());
      EXPECT_EQ(zref.size(), y.size());
      const T* x_data = x.data();
      const T* y_data = y.data();
      const T* zref_data = zref.data();
      const int d = zref.size();

      std::vector<T> ztgt(d);
      T* ztgt_data = ztgt.data();
      // test normal
      tgt(x_data, y_data, ztgt_data, d);
      ExpectEQ<T>(ztgt_data, zref_data, d);
      // test inplace x
      std::copy(x.begin(), x.end(), ztgt.begin());
      tgt(ztgt_data, y_data, ztgt_data, d);
      ExpectEQ<T>(ztgt_data, zref_data, d);
      // test inplace y
      std::copy(y.begin(), y.end(), ztgt.begin());
      tgt(x_data, ztgt_data, ztgt_data, d);
      ExpectEQ<T>(ztgt_data, zref_data, d);
    };

    TestAllImpls<KernelTuple, PlaceType>(d, verifier, x, y, zref);
  }
}

template <typename KernelTuple, typename PlaceType>
void TestKernelAXYN() {
  using T = typename KernelTuple::data_type;
  VLOG(10) << "Test JITKernel: " << jit::to_string(KernelTuple::kernel_type);
  for (int d : TestSizes()) {
    auto ref = jit::GetReferFunc<KernelTuple>();
    EXPECT_TRUE(ref != nullptr);

    const T a = static_cast<T>(3);
    std::vector<T> x(d), yref(d);
    std::vector<T> xinp(d);  // inplace test
    RandomVec<T>(d, x.data());
    std::copy(x.begin(), x.end(), xinp.begin());

    const T* x_data = x.data();
    T* yref_data = yref.data();
    T* xinp_data = xinp.data();
    // test refer code inplace
    ref(&a, x_data, yref_data, d);
    ref(&a, xinp_data, xinp_data, d);
    ExpectEQ<T>(xinp_data, yref_data, d);

    auto verifier = [](const typename KernelTuple::func_type tgt, const T a,
                       const std::vector<T>& x, const std::vector<T>& yref) {
      EXPECT_TRUE(tgt != nullptr);
      EXPECT_EQ(yref.size(), x.size());
      const T* x_data = x.data();
      const T* yref_data = yref.data();
      const int d = yref.size();
      std::vector<T> ytgt(d);
      T* ytgt_data = ytgt.data();
      // test normal
      tgt(&a, x_data, ytgt_data, d);
      ExpectEQ<T>(ytgt_data, yref_data, d);
      // test inplace x
      std::copy(x.begin(), x.end(), ytgt.begin());
      tgt(&a, ytgt_data, ytgt_data, d);
      ExpectEQ<T>(ytgt_data, yref_data, d);
    };
    TestAllImpls<KernelTuple, PlaceType>(d, verifier, a, x, yref);
  }
}

template <typename KernelTuple, typename PlaceType>
void TestKernelXYN() {
  using T = typename KernelTuple::data_type;
  VLOG(10) << "Test JITKernel: " << jit::to_string(KernelTuple::kernel_type);
  for (int d : TestSizes()) {
    auto ref = jit::GetReferFunc<KernelTuple>();
    EXPECT_TRUE(ref != nullptr);

    std::vector<T> x(d), yref(d);
    std::vector<T> xinp(d);  // inplace test
    RandomVec<T>(d, x.data());
    std::copy(x.begin(), x.end(), xinp.begin());

    const T* x_data = x.data();
    T* yref_data = yref.data();
    T* xinp_data = xinp.data();
    // test refer code inplace
    ref(x_data, yref_data, d);
    ref(xinp_data, xinp_data, d);
    ExpectEQ<T>(xinp_data, yref_data, d);
    auto verifier = [](const typename KernelTuple::func_type tgt,
                       const std::vector<T>& x, const std::vector<T>& yref) {
      EXPECT_TRUE(tgt != nullptr);
      EXPECT_EQ(yref.size(), x.size());
      const T* x_data = x.data();
      const T* yref_data = yref.data();
      const int d = yref.size();
      std::vector<T> ytgt(d);
      T* ytgt_data = ytgt.data();
      // test normal
      tgt(x_data, ytgt_data, d);
      ExpectEQ<T>(ytgt_data, yref_data, d);
      // test inplace x
      std::copy(x.begin(), x.end(), ytgt.begin());
      tgt(ytgt_data, ytgt_data, d);
      ExpectEQ<T>(ytgt_data, yref_data, d);
    };
    TestAllImpls<KernelTuple, PlaceType>(d, verifier, x, yref);
  }
}

template <typename KernelTuple, typename PlaceType>
void TestKernelXRN() {
  using T = typename KernelTuple::data_type;
  VLOG(10) << "Test JITKernel: " << jit::to_string(KernelTuple::kernel_type);
  auto last_acc = FLAGS_acc;
  FLAGS_acc = 1e-4;
  for (int d : TestSizes()) {
    auto ref = jit::GetReferFunc<KernelTuple>();
    EXPECT_TRUE(ref != nullptr);
    std::vector<T> x(d);
    RandomVec<T>(d, x.data());
    T ref_res;
    ref(x.data(), &ref_res, d);

    auto verifier = [](const typename KernelTuple::func_type tgt,
                       const std::vector<T>& x, const T ref_res) {
      EXPECT_TRUE(tgt != nullptr);
      T tgt_res;
      tgt(x.data(), &tgt_res, x.size());
      ExpectEQ<T>(&tgt_res, &ref_res, 1);
    };
    TestAllImpls<KernelTuple, PlaceType>(d, verifier, x, ref_res);
  }
  FLAGS_acc = last_acc;
}

template <typename KernelTuple, typename PlaceType>
void TestKernelLSTM() {
  using T = typename KernelTuple::data_type;
  VLOG(10) << "Test JITKernel: " << jit::to_string(KernelTuple::kernel_type);
  std::vector<std::string> all_acts = {"sigmoid", "tanh", "relu", "identity"};
  auto test_sizes = TestSizes();
  test_sizes.erase(std::remove(test_sizes.begin(), test_sizes.end(), 1000));
  for (int d : test_sizes) {
    for (bool use_peephole : {true, false}) {
      for (auto& act_gate : all_acts) {
        for (auto& act_cand : all_acts) {
          for (auto& act_cell : all_acts) {
            const jit::lstm_attr_t attr(
                d, jit::to_kerneltype(act_gate), jit::to_kerneltype(act_cand),
                jit::to_kerneltype(act_cell), use_peephole);
            auto ref = jit::GetReferFunc<KernelTuple>();
            EXPECT_TRUE(ref != nullptr);
            std::vector<T> xsrc(4 * d), wp(3 * d), ct_1(d);
            std::vector<T> ct_ref(d), ht_ref(d), checked(2 * d);
            RandomVec<T>(4 * d, xsrc.data());
            RandomVec<T>(3 * d, wp.data(), -1.f, 1.f);
            RandomVec<T>(d, ct_1.data(), -1.f, 1.f);
            // x could be changed after compute, so copy to save src
            std::vector<T> x(xsrc.size());
            std::copy(xsrc.begin(), xsrc.end(), x.begin());
            const T* ct_1_data = ct_1.data();
            const T* wp_data = wp.data();
            T* x_data = x.data();
            T* checked_data = checked.data();
            T* ct_ref_data = ct_ref.data();
            T* ht_ref_data = ht_ref.data();
            jit::lstm_t step;
            step.gates = x_data;
            step.ct_1 = ct_1_data;
            step.ct = ct_ref_data;
            step.ht = ht_ref_data;
            if (use_peephole) {
              step.wp = wp_data;
              step.checked = checked_data;
            }
            ref(&step, &attr);
            VLOG(10) << attr;

            auto verifier = [](
                const typename KernelTuple::func_type tgt,
                const std::vector<T>& xsrc, const std::vector<T>& wp,
                const std::vector<T>& ct_1, const std::vector<T>& ct_ref,
                const std::vector<T>& ht_ref,
                const typename KernelTuple::attr_type& attr) {
              EXPECT_TRUE(tgt != nullptr);
              EXPECT_EQ(ct_ref.size(), ht_ref.size());
              EXPECT_EQ(ct_1.size(), ht_ref.size());
              EXPECT_EQ(xsrc.size(), 4 * ht_ref.size());
              EXPECT_EQ(wp.size(), 3 * ht_ref.size());

              // x could be changed after compute, so copy to save src
              int d = ht_ref.size();
              std::vector<T> x(xsrc.size()), ct(ct_ref.size()),
                  ht(ht_ref.size());
              std::vector<T> checked(2 * d);
              std::copy(xsrc.begin(), xsrc.end(), x.begin());

              const T* ct_1_data = ct_1.data();
              const T* wp_data = wp.data();
              const T* ct_ref_data = ct_ref.data();
              const T* ht_ref_data = ht_ref.data();
              T* x_data = x.data();
              T* ct_data = ct.data();
              T* ht_data = ht.data();
              T* checked_data = checked.data();

              jit::lstm_t step;
              step.gates = x_data;
              step.ct_1 = ct_1_data;
              step.ct = ct_data;
              step.ht = ht_data;
              if (attr.use_peephole) {
                step.wp = wp_data;
                step.checked = checked_data;
              }

              tgt(&step, &attr);
              ExpectEQ<T>(ct_data, ct_ref_data, d);
              ExpectEQ<T>(ht_data, ht_ref_data, d);
            };
            TestAllImpls<KernelTuple, PlaceType>(attr, verifier, xsrc, wp, ct_1,
                                                 ct_ref, ht_ref, attr);
          }
        }
      }
    }
  }
}

template <typename KernelTuple, typename PlaceType>
void TestKernelGRU() {
  using T = typename KernelTuple::data_type;
  VLOG(10) << "Test JITKernel: " << jit::to_string(KernelTuple::kernel_type);
  std::vector<std::string> all_acts = {"sigmoid", "tanh", "relu", "identity"};
  auto test_sizes = TestSizes();
  test_sizes.erase(std::remove(test_sizes.begin(), test_sizes.end(), 1000));
  for (int d : test_sizes) {
    for (auto& act_gate : all_acts) {
      for (auto& act_cand : all_acts) {
        const jit::gru_attr_t attr(d, jit::to_kerneltype(act_gate),
                                   jit::to_kerneltype(act_cand));
        auto ref = jit::GetReferFunc<KernelTuple>();
        EXPECT_TRUE(ref != nullptr);
        std::vector<T> xsrc(3 * d), ht_1(d), ht_ref(d);
        RandomVec<T>(3 * d, xsrc.data());
        RandomVec<T>(d, ht_1.data());
        // x could be changed after compute, so copy to save src
        std::vector<T> x(xsrc.size());
        std::copy(xsrc.begin(), xsrc.end(), x.begin());
        const T* ht_1_data = ht_1.data();
        T* x_data = x.data();
        T* ht_ref_data = ht_ref.data();
        jit::gru_t step;
        step.gates = x_data;
        step.ht_1 = ht_1_data;
        step.ht = ht_ref_data;
        ref(&step, &attr);
        VLOG(10) << attr;
        auto verifier = [](const typename KernelTuple::func_type tgt,
                           const std::vector<T>& xsrc,
                           const std::vector<T>& ht_1,
                           const std::vector<T>& ht_ref,
                           const typename KernelTuple::attr_type& attr) {
          EXPECT_TRUE(tgt != nullptr);
          EXPECT_EQ(ht_1.size(), ht_ref.size());
          EXPECT_EQ(xsrc.size(), 3 * ht_ref.size());

          // x could be changed after compute, so copy to save src
          int d = ht_ref.size();
          std::vector<T> x(xsrc.size()), ht(ht_ref.size());
          std::copy(xsrc.begin(), xsrc.end(), x.begin());
          const T* ht_1_data = ht_1.data();
          const T* ht_ref_data = ht_ref.data();
          T* x_data = x.data();
          T* ht_data = ht.data();
          jit::gru_t step;
          step.gates = x_data;
          step.ht_1 = ht_1_data;
          step.ht = ht_data;
          tgt(&step, &attr);
          ExpectEQ<T>(ht_data, ht_ref_data, d);
        };
        TestAllImpls<KernelTuple, PlaceType>(attr, verifier, xsrc, ht_1, ht_ref,
                                             attr);
      }
    }
  }
}

template <typename KernelTuple, typename PlaceType>
void TestKernelNCHW16CMulNC() {
  using T = typename KernelTuple::data_type;
  VLOG(10) << "Test JITKernel: " << jit::to_string(KernelTuple::kernel_type);
  const int n = 3, c = 16 * 4, h = 10, w = 10;
  auto ref = jit::GetReferFunc<KernelTuple>();
  EXPECT_TRUE(ref != nullptr);
  int sz = n * c * h * w;
  std::vector<T> x(sz), y(n * c), zref(sz);
  std::vector<T> ztgt(sz), zjit(sz);
  RandomVec<T>(sz, x.data());
  RandomVec<T>(n * c, y.data());

  const T* x_data = x.data();
  const T* y_data = y.data();
  T* zref_data = zref.data();
  T* ztgt_data = ztgt.data();
  T* zjit_data = zjit.data();
  constexpr int simd_width = ZMM_FLOAT_BLOCK;
  int C = c / simd_width;
  auto tgt = jit::KernelFuncs<KernelTuple, PlaceType>::Cache().At(0);
  auto funcs = jit::GetAllCandidateFuncs<KernelTuple, PlaceType>(0);
  EXPECT_GT(funcs.size(), 0UL);
  auto jitcode = funcs[0];
  EXPECT_TRUE(tgt != nullptr);

  if (std::is_same<T, float>::value &&
      paddle::platform::MayIUse(paddle::platform::avx512f)) {
    EXPECT_TRUE(jitcode != nullptr);
  }
  for (int ni = 0; ni < n; ni++) {
    for (int ci = 0; ci < C; ci++) {
      auto ptr_x =
          x_data + ni * C * h * w * simd_width + ci * h * w * simd_width;
      auto ptr_y = y_data + ni * C * simd_width + ci * simd_width;
      auto ptr_zref =
          zref_data + ni * C * h * w * simd_width + ci * h * w * simd_width;
      auto ptr_ztgt =
          ztgt_data + ni * C * h * w * simd_width + ci * h * w * simd_width;

      ref(ptr_x, ptr_y, ptr_zref, h, w);
      tgt(ptr_x, ptr_y, ptr_ztgt, h, w);

      if (jitcode) {
        auto ptr_zjit =
            zjit_data + ni * C * h * w * simd_width + ci * h * w * simd_width;
        jitcode(ptr_x, ptr_y, ptr_zjit, h, w);
      }
    }
  }
  ExpectEQ<T>(ztgt_data, zref_data, sz);
  if (jitcode) {
    ExpectEQ<T>(zjit_data, zref_data, sz);
  }
}

template <typename KernelTuple, typename PlaceType>
void TestKernelLayerNorm() {
  using T = typename KernelTuple::data_type;
  VLOG(10) << "Test JITKernel: " << jit::to_string(KernelTuple::kernel_type);
  const T epsilon = 9.99999975e-06;
  for (int n : {1, 2, 10}) {
    for (int x_dim_0 : {1, 9, 17, 50}) {
      int left = n * x_dim_0;
      for (int x_dim_1 : TestSizes()) {
        int right = x_dim_1;
        auto ref = jit::GetReferFunc<KernelTuple>();
        EXPECT_TRUE(ref != nullptr);
        int sz = left * right;
        std::vector<T> x(sz), mean(left), var(left), scale(right), bias(right),
            outref(sz);
        RandomVec<T>(sz, x.data());
        RandomVec<T>(left, mean.data());
        RandomVec<T>(left, var.data());
        RandomVec<T>(right, scale.data());
        RandomVec<T>(right, bias.data());

        const T* scale_data = scale.data();
        const T* bias_data = bias.data();
        T* x_data = x.data();
        T* mean_data = mean.data();
        T* var_data = var.data();
        T* outref_data = outref.data();

        ref(x_data, outref_data, mean_data, var_data, scale_data, bias_data,
            left, epsilon, right);

        auto verifier = [](
            const typename KernelTuple::func_type tgt, const std::vector<T>& x_,
            const std::vector<T>& outref_, const std::vector<T>& mean_,
            const std::vector<T>& var_, const std::vector<T>& scale,
            const std::vector<T>& bias, const int& left, const float& epsilon,
            const typename KernelTuple::attr_type& right) {
          EXPECT_TRUE(tgt != nullptr);
          std::vector<T> outtgt(outref_.size());
          std::vector<T> x(x_.size());
          std::vector<T> mean(mean_.size());
          std::vector<T> var(var_.size());
          std::vector<T> outref(outref_.size());
          std::copy(x_.begin(), x_.end(), x.begin());
          std::copy(mean_.begin(), mean_.end(), mean.begin());
          std::copy(var_.begin(), var_.end(), var.begin());
          std::copy(outref_.begin(), outref_.end(), outref.begin());

          EXPECT_EQ(x.size(), static_cast<size_t>(left * right));
          EXPECT_EQ(outref.size(), static_cast<size_t>(left * right));
          EXPECT_EQ(mean.size(), static_cast<size_t>(left));
          EXPECT_EQ(var.size(), static_cast<size_t>(left));
          EXPECT_EQ(scale.size(), static_cast<size_t>(right));
          EXPECT_EQ(bias.size(), static_cast<size_t>(right));

          const T* scale_data = scale.data();
          const T* bias_data = bias.data();
          T* x_data = x.data();
          T* mean_data = mean.data();
          T* var_data = var.data();
          T* outref_data = outref.data();
          T* outtgt_data = outtgt.data();
          tgt(x_data, outtgt_data, mean_data, var_data, scale_data, bias_data,
              left, epsilon, right);
          ExpectEQ<T>(outtgt_data, outref_data, left * right);
        };
        TestAllImpls<KernelTuple, PlaceType>(right, verifier, x, outref, mean,
                                             var, scale, bias, left, epsilon,
                                             right);
      }
    }
  }
}

template <typename KernelTuple, typename PlaceType>
void TestKernelCRFDecoding() {
  using T = typename KernelTuple::data_type;
  VLOG(10) << "Test JITKernel: " << jit::to_string(KernelTuple::kernel_type);
  constexpr int state_trans_base_idx = 2;
  auto test_sizes = TestSizes();
  test_sizes.erase(std::remove(test_sizes.begin(), test_sizes.end(), 2000));
  for (int seq_len : {1, 11, 17, 50}) {
    for (int tag_num : test_sizes) {
      auto ref = jit::GetReferFunc<KernelTuple>();
      EXPECT_TRUE(ref != nullptr);
      int x_sz = seq_len * tag_num;
      int w_sz = (tag_num + state_trans_base_idx) * tag_num;
      std::vector<T> x(x_sz), w(w_sz), alpharef(x_sz);
      std::vector<int> trackref(x_sz);
      RandomVec<T>(x_sz, x.data());
      RandomVec<T>(w_sz, w.data());

      ref(seq_len, (const T*)x.data(), (const T*)w.data(), alpharef.data(),
          trackref.data(), tag_num);

      auto verifier = [](
          const typename KernelTuple::func_type tgt, const int& seq_len,
          const std::vector<T>& x, const std::vector<T>& w,
          const std::vector<T>& alpharef, const std::vector<int>& trackref,
          const typename KernelTuple::attr_type& tag_num) {
        constexpr int state_trans_base_idx = 2;
        EXPECT_TRUE(tgt != nullptr);
        EXPECT_EQ(x.size(), static_cast<size_t>(seq_len * tag_num));
        EXPECT_EQ(w.size(), static_cast<size_t>(
                                (tag_num + state_trans_base_idx) * tag_num));
        EXPECT_EQ(alpharef.size(), static_cast<size_t>(seq_len * tag_num));
        EXPECT_EQ(trackref.size(), static_cast<size_t>(seq_len * tag_num));
        std::vector<T> alphatgt(alpharef.size());
        std::vector<int> tracktgt(trackref.size());
        memcpy(tracktgt.data(), trackref.data(), tag_num * sizeof(int));
        tgt(seq_len, (const T*)x.data(), (const T*)w.data(), alphatgt.data(),
            tracktgt.data(), tag_num);
        ExpectEQ<T>(alpharef.data(), alphatgt.data(), seq_len * tag_num);
        ExpectEQ<int>(trackref.data(), tracktgt.data(), seq_len * tag_num);
      };
      TestAllImpls<KernelTuple, PlaceType>(tag_num, verifier, seq_len, x, w,
                                           alpharef, trackref, tag_num);
    }
  }
}

template <typename KernelTuple, typename PlaceType>
void TestKernelSeqPool() {
  using T = typename KernelTuple::data_type;
  VLOG(10) << "Test JITKernel: " << jit::to_string(KernelTuple::kernel_type);
  std::vector<jit::SeqPoolType> pool_types = {
      jit::SeqPoolType::kSum, jit::SeqPoolType::kAvg, jit::SeqPoolType::kSqrt};
  auto test_sizes = TestSizes();
  test_sizes.erase(std::remove(test_sizes.begin(), test_sizes.end(), 1000));
  for (auto type : pool_types) {
    for (int w : test_sizes) {
      jit::seq_pool_attr_t attr(w, type);
      for (int h : test_sizes) {
        attr.h = h;
        auto ref = jit::GetReferFunc<KernelTuple>();
        EXPECT_TRUE(ref != nullptr);
        std::vector<T> x(h * w), yref(w);
        RandomVec<T>(h * w, x.data());
        const T* x_data = x.data();
        T* yref_data = yref.data();
        ref(x_data, yref_data, &attr);
        VLOG(10) << attr;
        auto verifier = [](const typename KernelTuple::func_type tgt,
                           const std::vector<T>& x, const std::vector<T>& yref,
                           const typename KernelTuple::attr_type& attr) {
          EXPECT_TRUE(tgt != nullptr);
          EXPECT_EQ(x.size() % yref.size(), static_cast<size_t>(0));
          int w = yref.size();
          std::vector<T> y(w);
          const T* x_data = x.data();
          const T* yref_data = yref.data();
          T* y_data = y.data();
          tgt(x_data, y_data, &attr);
          ExpectEQ<T>(y_data, yref_data, w);
        };
        TestAllImpls<KernelTuple, PlaceType>(attr, verifier, x, yref, attr);
      }
    }
  }
}

template <typename KernelTuple, typename PlaceType>
void TestKernelEmbSeqPool() {
  using T = typename KernelTuple::data_type;
  VLOG(10) << "Test JITKernel: " << jit::to_string(KernelTuple::kernel_type);
  int64_t tbl_h = 1e4;
  std::vector<jit::SeqPoolType> pool_types = {
      jit::SeqPoolType::kSum};  // only support sum yet
  auto test_sizes = TestSizes();
  test_sizes.erase(std::remove(test_sizes.begin(), test_sizes.end(), 1000));
  for (int tbl_w : test_sizes) {
    std::vector<T> table(tbl_h * tbl_w);
    RandomVec<T>(tbl_h * tbl_w, table.data());
    const T* table_data = table.data();
    for (auto type : pool_types) {
      for (int idx_w : {1, 2, 10, 16}) {
        for (int idx_h : {1, 2, 9, 13, 16}) {
          auto ref = jit::GetReferFunc<KernelTuple>();
          EXPECT_TRUE(ref != nullptr);
          std::vector<int64_t> idx(idx_h * idx_w);
          RandomVec<int64_t>(idx_h * idx_w, idx.data(), 0, tbl_h - 1);
          int64_t out_w = tbl_w * idx_w;
          std::vector<T> oref(out_w);
          const int64_t* idx_data = idx.data();
          T* o_data = oref.data();
          jit::emb_seq_pool_attr_t attr(tbl_h, tbl_w, idx_h, idx_w, out_w,
                                        type);
          ref(table_data, idx_data, o_data, &attr);

          auto verifier = [](const typename KernelTuple::func_type tgt,
                             const std::vector<T>& table,
                             const std::vector<int64_t>& idx,
                             const std::vector<T>& oref,
                             const typename KernelTuple::attr_type& attr) {
            EXPECT_TRUE(tgt != nullptr);
            EXPECT_EQ(table.size(), static_cast<size_t>(attr.table_height *
                                                        attr.table_width));
            EXPECT_EQ(idx.size(), static_cast<size_t>(attr.index_height *
                                                      attr.index_width));
            EXPECT_EQ(oref.size(),
                      static_cast<size_t>(attr.table_width * attr.index_width));
            const T* table_data = table.data();
            const int64_t* idx_data = idx.data();
            const T* oref_data = oref.data();
            int o_w = oref.size();
            std::vector<T> out(o_w);
            T* o_data = out.data();
            tgt(table_data, idx_data, o_data, &attr);
            ExpectEQ<T>(o_data, oref_data, o_w);
          };
          TestAllImpls<KernelTuple, PlaceType>(attr, verifier, table, idx, oref,
                                               attr);
        }
      }
    }
  }
}

template <typename KernelTuple, typename PlaceType>
void TestKernelMatMul() {
  using T = typename KernelTuple::data_type;
  VLOG(10) << "Test JITKernel: " << jit::to_string(KernelTuple::kernel_type);
  auto last_acc = FLAGS_acc;
  // export MKL_CBWR=AVX would make MKL force to use AVX
  // export KMP_DETERMINISTIC_REDUCTION=yes would make the result deterministic
  FLAGS_acc = 1e-3;
  for (int m : {1, 2, 3, 4}) {
    for (int n : {1, 2, 3, 4}) {
      for (int k : TestSizes()) {
        auto ref = jit::GetReferFunc<KernelTuple>();
        EXPECT_TRUE(ref != nullptr);
        std::vector<T> a(m * k), b(k * n), c(m * n);
        RandomVec<T>(m * k, a.data());
        RandomVec<T>(k * n, b.data());
        const T* a_data = a.data();
        const T* b_data = b.data();
        T* c_data = c.data();
        const jit::matmul_attr_t attr{m, n, k};
        ref(a_data, b_data, c_data, &attr);
        auto verifier = [](const typename KernelTuple::func_type tgt,
                           const std::vector<T>& a, const std::vector<T>& b,
                           const std::vector<T>& cref,
                           const typename KernelTuple::attr_type& attr) {
          EXPECT_TRUE(tgt != nullptr);
          EXPECT_EQ(a.size(), static_cast<size_t>(attr.m * attr.k));
          EXPECT_EQ(b.size(), static_cast<size_t>(attr.k * attr.n));
          EXPECT_EQ(cref.size(), static_cast<size_t>(attr.m * attr.n));
          std::vector<T> c(cref.size());
          const T* a_data = a.data();
          const T* b_data = b.data();
          const T* cref_data = cref.data();
          T* c_data = c.data();
          tgt(a_data, b_data, c_data, &attr);
          ExpectEQ<T>(c_data, cref_data, attr.m * attr.n);
        };
        TestAllImpls<KernelTuple, PlaceType>(attr, verifier, a, b, c, attr);
      }
    }
  }
  FLAGS_acc = last_acc;
}

template <typename KernelTuple, typename PlaceType>
void TestKernelSoftmax() {
  using T = typename KernelTuple::data_type;
  VLOG(10) << "Test JITKernel: " << jit::to_string(KernelTuple::kernel_type);
  for (int bs : {1, 2, 10}) {
    for (int n : TestSizes()) {
      for (int m : {1, 2, 3}) {  // remain
        if (m > n || n % m != 0) {
          continue;
        }
        auto ref = jit::GetReferFunc<KernelTuple>();
        EXPECT_TRUE(ref != nullptr);
        std::vector<T> x(bs * n), y(bs * n);
        RandomVec<T>(bs * n, x.data());
        const T* x_data = x.data();
        T* y_data = y.data();

        std::vector<T> xinp(x.size());  // inplace test
        std::copy(x.begin(), x.end(), xinp.begin());
        ref(x_data, y_data, n, bs, m);
        T* xinp_data = xinp.data();
        ref(xinp_data, xinp_data, n, bs, m);
        ExpectEQ<T>(xinp_data, y_data, n * bs);

        auto verifier = [](const typename KernelTuple::func_type tgt,
                           const std::vector<T>& x, const std::vector<T>& yref,
                           int n, int bs, int m) {
          EXPECT_TRUE(tgt != nullptr);
          EXPECT_EQ(yref.size(), x.size());
          EXPECT_EQ(x.size(), static_cast<size_t>(n * bs));
          const T* x_data = x.data();
          const T* yref_data = yref.data();
          std::vector<T> ytgt(n * bs);
          T* ytgt_data = ytgt.data();
          // test normal
          tgt(x_data, ytgt_data, n, bs, m);
          ExpectEQ<T>(ytgt_data, yref_data, n * bs);
          // test inplace x
          std::copy(x.begin(), x.end(), ytgt.begin());
          tgt(ytgt_data, ytgt_data, n, bs, m);
          ExpectEQ<T>(ytgt_data, yref_data, n * bs);
        };
        TestAllImpls<KernelTuple, PlaceType>(n, verifier, x, y, n, bs, m);
      }
    }
  }
}

template <typename KernelTuple, typename PlaceType>
void TestKernelStrideASum() {
  using T = typename KernelTuple::data_type;
  VLOG(10) << "Test JITKernel: " << jit::to_string(KernelTuple::kernel_type);
  for (int d : TestSizes()) {
    for (int m : {1, 2, 3}) {  // stride
      if (m > d || d % m != 0) {
        continue;
      }
      auto ref = jit::GetReferFunc<KernelTuple>();
      EXPECT_TRUE(ref != nullptr);
      std::vector<T> x(d);
      RandomVec<T>(d, x.data());
      T ref_res;
      ref(x.data(), &ref_res, d, m);

      auto verifier = [](const typename KernelTuple::func_type tgt,
                         const std::vector<T>& x, const T ref_res,
                         const int m) {
        EXPECT_TRUE(tgt != nullptr);
        T tgt_res;
        tgt(x.data(), &tgt_res, x.size(), m);
        ExpectEQ<T>(&tgt_res, &ref_res, 1);
      };
      TestAllImpls<KernelTuple, PlaceType>(d, verifier, x, ref_res, m);
    }
  }
}

template <typename KernelTuple, typename PlaceType>
void TestKernelStrideScal() {
  using T = typename KernelTuple::data_type;
  VLOG(10) << "Test JITKernel: " << jit::to_string(KernelTuple::kernel_type);
  for (int d : TestSizes()) {
    for (int m : {1, 2, 3}) {  // stride
      if (m > d || d % m != 0) {
        continue;
      }
      auto ref = jit::GetReferFunc<KernelTuple>();
      EXPECT_TRUE(ref != nullptr);

      const T a = static_cast<T>(3);
      std::vector<T> x(d), yref(d);
      std::vector<T> xinp(d);  // inplace test
      RandomVec<T>(d, x.data());
      std::copy(x.begin(), x.end(), xinp.begin());

      const T* x_data = x.data();
      T* yref_data = yref.data();
      T* xinp_data = xinp.data();
      // test refer code inplace
      ref(&a, x_data, yref_data, d, m);
      ref(&a, xinp_data, xinp_data, d, m);
      ExpectEQ<T>(xinp_data, yref_data, d);

      auto verifier = [](const typename KernelTuple::func_type tgt, const T a,
                         const std::vector<T>& x, const std::vector<T>& yref,
                         const int m) {
        EXPECT_TRUE(tgt != nullptr);
        EXPECT_EQ(yref.size(), x.size());
        const T* x_data = x.data();
        const T* yref_data = yref.data();
        const int d = yref.size();
        std::vector<T> ytgt(d);
        T* ytgt_data = ytgt.data();
        // test normal
        tgt(&a, x_data, ytgt_data, d, m);
        ExpectEQ<T>(ytgt_data, yref_data, d);
        // test inplace x
        std::copy(x.begin(), x.end(), ytgt.begin());
        tgt(&a, ytgt_data, ytgt_data, d, m);
        ExpectEQ<T>(ytgt_data, yref_data, d);
      };
      TestAllImpls<KernelTuple, PlaceType>(d, verifier, a, x, yref, m);
    }
  }
}

template <typename KernelTuple, typename PlaceType>
void TestKernelAdam() {
  using T = typename KernelTuple::data_type;
  VLOG(10) << "Test JITKernel: " << jit::to_string(KernelTuple::kernel_type);
  const T lr = 0.1;
  const T beta1 = 0.99;
  const T beta2 = 0.95;
  const T beta1_pow = beta1 * beta1;
  const T beta2_pow = beta2 * beta2;

  const T epsilon = 0.000001;
  const int64_t numel = 123;

  T learning_rate = lr * (sqrt(1 - beta2_pow) / (1 - beta1_pow));
  T eps = epsilon * sqrt(1 - beta2_pow);

  std::vector<T> param(numel);
  std::vector<T> grad(numel);
  std::vector<T> mom1(numel);
  std::vector<T> mom2(numel);

  std::vector<T> param_out(param.size());
  std::vector<T> mom1_out(mom1.size());
  std::vector<T> mom2_out(mom2.size());

  RandomVec<T>(numel, param.data(), 0.5f);
  RandomVec<T>(numel, grad.data(), 0.5f);
  RandomVec<T>(numel, mom1.data(), 0.5f);
  RandomVec<T>(numel, mom2.data(), 0.5f);

  auto ref = jit::GetReferFunc<KernelTuple>();
  EXPECT_TRUE(ref != nullptr);
  jit::adam_attr_t attr(beta1, beta2);
  ref(beta1, beta2, -learning_rate, eps, numel, grad.data(), mom1.data(),
      mom2.data(), param.data(), mom1_out.data(), mom2_out.data(),
      param_out.data());

  auto verifier = [](
      const typename KernelTuple::func_type tgt, T beta1, T beta2, T lr, T eps,
      int64_t numel, const std::vector<T>& grad, const std::vector<T>& mom1,
      const std::vector<T>& mom2, const std::vector<T>& param,
      const std::vector<T>& ref_mom1_out, const std::vector<T>& ref_mom2_out,
      const std::vector<T>& ref_param_out) {
    EXPECT_TRUE(tgt != nullptr);
    EXPECT_EQ(param.size(), static_cast<size_t>(numel));
    EXPECT_EQ(grad.size(), static_cast<size_t>(numel));
    EXPECT_EQ(mom1.size(), static_cast<size_t>(numel));
    EXPECT_EQ(mom2.size(), static_cast<size_t>(numel));

    std::vector<T> jit_mom1_out(ref_mom1_out.size());
    std::vector<T> jit_mom2_out(ref_mom2_out.size());
    std::vector<T> jit_param_out(ref_param_out.size());

    tgt(beta1, beta2, -lr, eps, numel, grad.data(), mom1.data(), mom2.data(),
        param.data(), jit_mom1_out.data(), jit_mom2_out.data(),
        jit_param_out.data());

    ExpectEQ<T>(ref_mom1_out.data(), jit_mom1_out.data(), numel);
    ExpectEQ<T>(ref_mom2_out.data(), jit_mom2_out.data(), numel);
    ExpectEQ<T>(ref_param_out.data(), jit_param_out.data(), numel);
  };
  TestAllImpls<KernelTuple, PlaceType>(
      attr, verifier, beta1, beta2, learning_rate, eps, numel, grad, mom1, mom2,
      param, mom1_out, mom2_out, param_out);
}

template <typename KernelTuple, typename PlaceType>
void TestKernelSgd() {
  using T = typename KernelTuple::data_type;
  VLOG(10) << "Test JITKernel: " << jit::to_string(KernelTuple::kernel_type);
  const T lr = 0.1;
  auto UnDuplicatedRandomVec = [](int n, const int64_t lower,
                                  const int64_t upper) -> std::vector<int64_t> {
    PADDLE_ENFORCE_LE(static_cast<size_t>(upper - lower), n - 1,
                      paddle::platform::errors::InvalidArgument(
                          "The range of Sgd (upper - lower) should be lower "
                          "than n-1 (Sgd size -1). But the upper - lower is %d "
                          "and n-1 is %d.",
                          static_cast<size_t>(upper - lower), n - 1));
    PADDLE_ENFORCE_GT(
        n, 0, paddle::platform::errors::InvalidArgument(
                  "The Sgd size should be larger than 0. But the n is %d.", n));
    std::vector<int64_t> all, out;
    for (int i = 0; i < n; ++i) {
      all.push_back(i);
    }
    std::random_device rnd;
    int64_t seed_tmp = rnd();
    std::default_random_engine rng(seed_tmp);
    std::shuffle(all.begin(), all.end(), rng);
    out.insert(out.begin(), all.begin(), all.begin() + n);
    return out;
  };
  for (int param_h : {1, 10}) {
    for (int grad_w : TestSizes()) {
      std::vector<T> param(param_h * grad_w);
      std::vector<T> param_out(param_h * grad_w);
      RandomVec<T>(param_h * grad_w, param.data());
      const T* param_data = param.data();
      T* out_data = param_out.data();
      for (int rows_size = 1; rows_size <= param_h; ++rows_size) {
        std::vector<T> grad(rows_size * grad_w);
        std::vector<int64_t> rows =
            UnDuplicatedRandomVec(rows_size, 0, rows_size - 1);
        RandomVec<T>(rows_size * grad_w, grad.data());
        const int64_t* rows_data = rows.data();
        const T* grad_data = grad.data();
        auto ref = jit::GetReferFunc<KernelTuple>();
        EXPECT_TRUE(ref != nullptr);
        jit::sgd_attr_t attr(param_h, grad_w, rows_size, grad_w, rows_size);
        ref(&lr, param_data, grad_data, rows_data, out_data, &attr);

        // inplace test
        std::vector<T> inp(param.size());
        std::copy(param.begin(), param.end(), inp.begin());
        T* inp_data = inp.data();
        ref(&lr, inp_data, grad_data, rows_data, inp_data, &attr);
        // only the selected rows should be equal
        for (int i = 0; i < rows_size; ++i) {
          ExpectEQ<T>(inp_data + rows[i] * grad_w, out_data + rows[i] * grad_w,
                      grad_w);
        }

        auto verifier = [](
            const typename KernelTuple::func_type tgt, const T lr,
            const std::vector<T>& param, const std::vector<T>& grad,
            const std::vector<int64_t>& rows, const std::vector<T>& oref,
            const typename KernelTuple::attr_type& attr) {
          EXPECT_TRUE(tgt != nullptr);
          EXPECT_EQ(param.size(),
                    static_cast<size_t>(attr.param_height * attr.param_width));
          EXPECT_EQ(grad.size(),
                    static_cast<size_t>(attr.grad_height * attr.grad_width));
          EXPECT_EQ(rows.size(), static_cast<size_t>(attr.selected_rows_size));
          EXPECT_EQ(param.size(), oref.size());
          const T* param_data = param.data();
          const T* grad_data = grad.data();
          const int64_t* rows_data = rows.data();
          const T* oref_data = oref.data();

          std::vector<T> out(oref.size());
          T* o_data = out.data();
          tgt(&lr, param_data, grad_data, rows_data, o_data, &attr);
          // only the selected rows should be equal
          for (size_t i = 0; i < rows.size(); ++i) {
            ExpectEQ<T>(o_data + rows[i] * attr.grad_width,
                        oref_data + rows[i] * attr.grad_width, attr.grad_width);
          }

          // inplace
          std::copy(param.begin(), param.end(), out.begin());
          tgt(&lr, o_data, grad_data, rows_data, o_data, &attr);
          for (size_t i = 0; i < rows.size(); ++i) {
            ExpectEQ<T>(o_data + rows[i] * attr.grad_width,
                        oref_data + rows[i] * attr.grad_width, attr.grad_width);
          }
        };
        TestAllImpls<KernelTuple, PlaceType>(attr, verifier, lr, param, grad,
                                             rows, param_out, attr);
      }
    }
  }
}

template <typename KernelTuple, typename PlaceType>
void TestKernelVBroadcast() {
  using T = typename KernelTuple::data_type;
  VLOG(10) << "Test JITKernel: " << jit::to_string(KernelTuple::kernel_type);
  for (int w : TestSizes()) {
    std::vector<T> x(w);
    RandomVec<T>(w, x.data());
    const T* x_data = x.data();
    for (int64_t h : {1, 2, 6}) {
      auto ref = jit::GetReferFunc<KernelTuple>();
      EXPECT_TRUE(ref != nullptr);
      std::vector<T> y(w * h);
      T* y_data = y.data();
      ref(x_data, y_data, h, w);

      auto verifier = [](const typename KernelTuple::func_type tgt,
                         const std::vector<T>& x, const std::vector<T>& yref,
                         const int64_t& h,
                         const typename KernelTuple::attr_type& attr) {
        EXPECT_TRUE(tgt != nullptr);
        EXPECT_EQ(x.size(), static_cast<size_t>(attr));
        EXPECT_EQ(yref.size(), x.size() * h);
        std::vector<T> y(yref.size());
        const T* x_data = x.data();
        const T* yref_data = yref.data();
        T* y_data = y.data();
        tgt(x_data, y_data, h, attr);
        ExpectEQ<T>(y_data, yref_data, yref.size());
      };
      TestAllImpls<KernelTuple, PlaceType>(static_cast<int64_t>(w), verifier, x,
                                           y, h, static_cast<int64_t>(w));
    }
  }
}

// test pool
TEST(JITKernel_pool, jitcreator) {
  const auto& jitcreators = jit::JitCodeCreatorPool::Instance().AllCreators();
#if defined(_WIN32) || defined(__APPLE__) || defined(__OSX__)
  EXPECT_EQ(jitcreators.size(), 0UL);
#else
  EXPECT_EQ(jitcreators.size(), 26UL);
#endif
}

TEST(JITKernel_pool, jitpool) {
  // jitpool is related with attr
  const auto& kers = jit::JitCodePool<jit::kVAdd>().Instance().AllKernels();
  EXPECT_EQ(kers.size(), 0UL);
  jit::GetAllCandidateKernels<jit::VAddTuple<float>, CPUPlace>(3);
// after call GetAllCandidateKernels, it will create jitcode Automatically
#if defined(_WIN32) || defined(__APPLE__) || defined(__OSX__)
  EXPECT_EQ(kers.size(), 0UL);
#else
  EXPECT_EQ(kers.size(), 1UL);
#endif
}

TEST(JITKernel_pool, more) {
  const auto& kers = jit::KernelPool::Instance().AllKernels();
  size_t target_num = 8;

#ifdef __AVX__
  target_num += 2;
#endif

#ifdef PADDLE_WITH_MKLML
  target_num += 12;
#endif

  EXPECT_EQ(kers.size(), target_num);
}

TEST(JITKernel_pool, refer) {
  const auto& kers = jit::ReferKernelPool::Instance().AllKernels();
  EXPECT_EQ(kers.size(), 32UL);
}

// test helper
TEST(JITKernel_helper, GetAllCandidateKernels) {
  auto fp_kers =
      jit::GetAllCandidateKernels<jit::VExpTuple<float>, CPUPlace>(10);
#if defined(_WIN32) || defined(__APPLE__) || defined(__OSX__)
  EXPECT_GE(fp_kers.size(), 1UL);  // refer
#else
#ifdef PADDLE_WITH_MKLML
  EXPECT_GE(fp_kers.size(), 3UL);  // jitcode, mkl, refer
#else
  EXPECT_GE(fp_kers.size(), 2UL);  // jitcode, refer
#endif
#endif

  auto db_kers =
      jit::GetAllCandidateKernels<jit::VExpTuple<double>, CPUPlace>(10);
#if defined(_WIN32) || defined(__APPLE__) || defined(__OSX__)
  EXPECT_GE(db_kers.size(), 1UL);  // refer
#else
#ifdef PADDLE_WITH_MKLML
  EXPECT_GE(db_kers.size(), 2UL);  // mkl, refer
#else
  EXPECT_GE(db_kers.size(), 1UL);  // refer
#endif
#endif
}

TEST(JITKernel_helper, GetAllCandidateFuncsWithTypes) {
  auto fp_kers =
      jit::GetAllCandidateFuncsWithTypes<jit::VExpTuple<float>, CPUPlace>(10);
#if defined(__APPLE__) || defined(__OSX__)
  EXPECT_GE(fp_kers.size(), 1UL);  // refer
#else
#if !defined(PADDLE_WITH_MKLML) || defined(_WIN32)
  EXPECT_GE(fp_kers.size(), 2UL);  // jitcode/mkl, refer
#else
  EXPECT_GE(fp_kers.size(), 3UL);  // jitcode, mkl, refer
#endif
#endif

  auto db_kers =
      jit::GetAllCandidateFuncsWithTypes<jit::VExpTuple<double>, CPUPlace>(10);
#if defined(__APPLE__) || defined(__OSX__) || !defined(PADDLE_WITH_MKLML)
  EXPECT_GE(db_kers.size(), 1UL);  // refer
#else
  EXPECT_GE(db_kers.size(), 2UL);  // mkl, refer
#endif
}

TEST(JITKernel_helper, KernelFuncs) {
  auto f1 = jit::KernelFuncs<jit::VAddTuple<float>, CPUPlace>::Cache().At(3);
  auto f2 = jit::KernelFuncs<jit::VAddTuple<float>, CPUPlace>::Cache()[3];
  EXPECT_TRUE(f1 != nullptr);
  EXPECT_TRUE(f1 == f2);

  auto f3 = jit::KernelFuncs<jit::VAddTuple<float>, CPUPlace>::Cache()[5];
#if defined(_WIN32) || defined(__APPLE__) || defined(__OSX__)
  EXPECT_TRUE(f2 == f3);
#else
  EXPECT_TRUE(f2 != f3);
#endif
}

TEST(JITKernel_helper, GetAllCandidateFuncs) {
  auto funcs = jit::GetAllCandidateFuncs<jit::VExpTuple<float>, CPUPlace>(10);
  auto kers = jit::GetAllCandidateKernels<jit::VExpTuple<float>, CPUPlace>(10);
  EXPECT_EQ(funcs.size(), kers.size());

  std::vector<float> x(10), tgt(10);
  RandomVec<float>(10, x.data());
  auto best = jit::GetDefaultBestFunc<jit::VExpTuple<float>, CPUPlace>(10);
  best(x.data(), tgt.data(), 10);
  for (auto f : funcs) {
    std::vector<float> y(10);
    f(x.data(), y.data(), 10);
    ExpectEQ<float>(y.data(), tgt.data(), 10);
  }
}

TEST(JITKernel_helper, pack_weights) {
  const int N = 8 * 60, K = 2;
  float src[K][N], yref[K][N], y[K * N];
  float* x = &(src[0][0]);
  float* ref = &(yref[0][0]);
  for (int i = 0; i < N * K; ++i) {
    *(x + i) = static_cast<float>(i);
  }
  int block = 0;
  std::vector<int> groups;
  if (paddle::platform::MayIUse(paddle::platform::avx512f)) {
    block = ZMM_FLOAT_BLOCK;
    groups.push_back(30);
  } else {
    block = YMM_FLOAT_BLOCK;
    groups.insert(groups.end(), {14, 14, 14, 14, 4});
  }

  int offset = 0;
  int acc = 0;
  for (int g : groups) {
    g = g * block;
    for (int k = 0; k < K; ++k) {
      for (int i = 0; i < g; ++i) {
        *(ref + offset) = src[k][i + acc];
        offset++;
      }
    }
    acc += g;
  }

  jit::pack_weights<float>(x, y, N, K);
  ExpectEQ<float>(y, ref, N * K);
}

TEST(JITKernel_helper, attr) {
  std::ostringstream out;
  // KernelTypes
  out << jit::to_string(jit::kNone) << jit::to_string(jit::kCRFDecoding)
      << jit::to_string(jit::kEmbSeqPool) << jit::to_string(jit::kGRUH1)
      << jit::to_string(jit::kGRUHtPart1) << jit::to_string(jit::kGRUHtPart2)
      << jit::to_string(jit::kHSum) << jit::to_string(jit::kHMax)
      << jit::to_string(jit::kLSTMCtHt) << jit::to_string(jit::kLSTMC1H1)
      << jit::to_string(jit::kLayerNorm) << jit::to_string(jit::kMatMul)
      << jit::to_string(jit::kNCHW16CMulNC) << jit::to_string(jit::kSeqPool)
      << jit::to_string(jit::kSoftmax) << jit::to_string(jit::kVAdd)
      << jit::to_string(jit::kVAddBias) << jit::to_string(jit::kVAddRelu)
      << jit::to_string(jit::kVBroadcast) << jit::to_string(jit::kVCopy)
      << jit::to_string(jit::kVExp) << jit::to_string(jit::kVIdentity)
      << jit::to_string(jit::kVMul) << jit::to_string(jit::kVRelu)
      << jit::to_string(jit::kVScal) << jit::to_string(jit::kSgd)
      << jit::to_string(jit::kAdam) << jit::to_string(jit::kVSigmoid)
      << jit::to_string(jit::kVSquare) << jit::to_string(jit::kVSub)
      << jit::to_string(jit::kVTanh);
  EXPECT_EQ(out.str().size(), 239UL);

  // SeqPoolTypes
  out.str("");
  out << jit::to_string(jit::kSum) << jit::to_string(jit::kAvg)
      << jit::to_string(jit::kSqrt);
  EXPECT_EQ(out.str().size(), 13UL);

  EXPECT_EQ(jit::to_kerneltype("relu"), jit::kVRelu);
  EXPECT_EQ(jit::to_kerneltype("Identity"), jit::kVIdentity);
  EXPECT_EQ(jit::to_kerneltype("VEXP"), jit::kVExp);
  EXPECT_EQ(jit::to_kerneltype("SigmoiD"), jit::kVSigmoid);
  EXPECT_EQ(jit::to_kerneltype("VTanh"), jit::kVTanh);

  out.str("");
  out << jit::lstm_attr_t(8, jit::kVIdentity, jit::kVSigmoid, jit::kVTanh);
  EXPECT_EQ(out.str().size(), 89UL);

  out.str("");
  out << jit::gru_attr_t(8, jit::kVIdentity, jit::kVSigmoid);
  EXPECT_EQ(out.str().size(), 52UL);

  out.str("");
  out << jit::seq_pool_attr_t(8, jit::SeqPoolType::kSum);
  EXPECT_EQ(out.str().size(), 44UL);

  out.str("");
  out << jit::emb_seq_pool_attr_t(1, 2, 3, 4, 5, jit::SeqPoolType::kAvg);
  EXPECT_EQ(out.str().size(), 93UL);

  out.str("");
  out << jit::sgd_attr_t(1, 2, 3, 4, 5);
  EXPECT_EQ(out.str().size(), 81UL);

  out.str("");
  out << jit::matmul_attr_t(1, 2, 3);
  EXPECT_EQ(out.str().size(), 14UL);
}

// test keys
TEST(JITKernel_key, int) {
  EXPECT_TRUE(jit::JitCodeKey<int>(2) == jit::JitCodeKey<int>(2));
  EXPECT_TRUE(jit::JitCodeKey<int>(2) == jit::JitCodeKey<int64_t>(2));
  EXPECT_TRUE(jit::JitCodeKey<int>(2) != jit::JitCodeKey<int>(3));
}

TEST(JITKernel_key, gru) {
  jit::gru_attr_t attr1(8, jit::kVSigmoid, jit::kVTanh);
  jit::gru_attr_t attr2(8, jit::kVSigmoid, jit::kVTanh);
  jit::gru_attr_t attr3(9, jit::kVSigmoid, jit::kVTanh);
  jit::gru_attr_t attr4(9, jit::kVSigmoid, jit::kVIdentity);
  jit::gru_attr_t attr5(9, jit::kVTanh, jit::kVIdentity);

  auto key1 = jit::JitCodeKey<jit::gru_attr_t>(attr1);
  auto key2 = jit::JitCodeKey<jit::gru_attr_t>(attr2);
  auto key3 = jit::JitCodeKey<jit::gru_attr_t>(attr3);
  auto key4 = jit::JitCodeKey<jit::gru_attr_t>(attr4);
  auto key5 = jit::JitCodeKey<jit::gru_attr_t>(attr5);

  EXPECT_TRUE(key1 == key2);
  EXPECT_TRUE(key2 != key3);
  EXPECT_TRUE(key2 != key4);
  EXPECT_TRUE(key2 != key5);
  EXPECT_TRUE(key3 != key4);
  EXPECT_TRUE(key3 != key5);
  EXPECT_TRUE(key4 != key5);
}

TEST(JITKernel_key, lstm) {
  jit::lstm_attr_t attr1(8, jit::kVIdentity, jit::kVSigmoid, jit::kVTanh);
  jit::lstm_attr_t attr2(8, jit::kVIdentity, jit::kVSigmoid, jit::kVTanh);
  jit::lstm_attr_t attr3(9, jit::kVIdentity, jit::kVSigmoid, jit::kVTanh);
  jit::lstm_attr_t attr4(9, jit::kVRelu, jit::kVSigmoid, jit::kVTanh);
  jit::lstm_attr_t attr5(9, jit::kVRelu, jit::kVSigmoid, jit::kVTanh, true);
  jit::lstm_attr_t attr6(9, jit::kVRelu, jit::kVSigmoid, jit::kVTanh, true);

  auto key1 = jit::JitCodeKey<jit::lstm_attr_t>(attr1);
  auto key2 = jit::JitCodeKey<jit::lstm_attr_t>(attr2);
  auto key3 = jit::JitCodeKey<jit::lstm_attr_t>(attr3);
  auto key4 = jit::JitCodeKey<jit::lstm_attr_t>(attr4);
  auto key5 = jit::JitCodeKey<jit::lstm_attr_t>(attr5);
  auto key6 = jit::JitCodeKey<jit::lstm_attr_t>(attr6);

  EXPECT_TRUE(key1 == key2);
  EXPECT_TRUE(key2 != key3);
  EXPECT_TRUE(key2 != key4);
  EXPECT_TRUE(key2 != key5);
  EXPECT_TRUE(key3 != key4);
  EXPECT_TRUE(key3 != key5);
  EXPECT_TRUE(key4 != key5);
  EXPECT_TRUE(key5 == key6);
}

TEST(JITKernel_key, seq_pool) {
  jit::seq_pool_attr_t attr1(2, jit::SeqPoolType::kSum, 1);
  jit::seq_pool_attr_t attr2(2, jit::SeqPoolType::kSum, 3);
  jit::seq_pool_attr_t attr3(3, jit::SeqPoolType::kSum, 3);
  jit::seq_pool_attr_t attr4(3, jit::SeqPoolType::kAvg, 3);

  auto key1 = jit::JitCodeKey<jit::seq_pool_attr_t>(attr1);
  auto key2 = jit::JitCodeKey<jit::seq_pool_attr_t>(attr2);
  auto key3 = jit::JitCodeKey<jit::seq_pool_attr_t>(attr3);
  auto key4 = jit::JitCodeKey<jit::seq_pool_attr_t>(attr4);

  EXPECT_TRUE(key1 == key2);
  EXPECT_TRUE(key2 != key3);
  EXPECT_TRUE(key2 != key4);
  EXPECT_TRUE(key3 != key4);
}

TEST(JITKernel_key, matmul) {
  jit::matmul_attr_t attr1(1, 2, 3);
  jit::matmul_attr_t attr2(1, 2, 3);
  jit::matmul_attr_t attr3(1, 3, 3);
  jit::matmul_attr_t attr4(2, 3, 4);

  auto key1 = jit::JitCodeKey<jit::matmul_attr_t>(attr1);
  auto key2 = jit::JitCodeKey<jit::matmul_attr_t>(attr2);
  auto key3 = jit::JitCodeKey<jit::matmul_attr_t>(attr3);
  auto key4 = jit::JitCodeKey<jit::matmul_attr_t>(attr4);

  EXPECT_TRUE(key1 == key2);
  EXPECT_TRUE(key2 != key3);
  EXPECT_TRUE(key2 != key4);
  EXPECT_TRUE(key3 != key4);
}

TEST(JITKernel_key, emb_seq_pool) {
  jit::emb_seq_pool_attr_t attr1(1, 2, 3, 4, 5, jit::SeqPoolType::kSum);
  jit::emb_seq_pool_attr_t attr2(1, 2, 3, 4, 5, jit::SeqPoolType::kSum);
  jit::emb_seq_pool_attr_t attr3(10, 2, 9, 8, 7, jit::SeqPoolType::kAvg);
  jit::emb_seq_pool_attr_t attr4(10, 3, 9, 8, 7, jit::SeqPoolType::kSum);
  jit::emb_seq_pool_attr_t attr5(1, 6, 3, 4, 5, jit::SeqPoolType::kSum);

  auto key1 = jit::JitCodeKey<jit::emb_seq_pool_attr_t>(attr1);
  auto key2 = jit::JitCodeKey<jit::emb_seq_pool_attr_t>(attr2);
  auto key3 = jit::JitCodeKey<jit::emb_seq_pool_attr_t>(attr3);
  auto key4 = jit::JitCodeKey<jit::emb_seq_pool_attr_t>(attr4);
  auto key5 = jit::JitCodeKey<jit::emb_seq_pool_attr_t>(attr5);

  EXPECT_TRUE(key1 == key2);
  EXPECT_TRUE(key2 == key3);
  EXPECT_TRUE(key2 != key4);
  EXPECT_TRUE(key2 != key5);
  EXPECT_TRUE(key4 != key5);
}

TEST(JITKernel_key, adam) {
  jit::adam_attr_t attr1(0.4f, 0.9f);
  jit::adam_attr_t attr2(0.4f, 0.9f);
  jit::adam_attr_t attr3(0.1f, 0.3f);

  auto key1 = jit::JitCodeKey<jit::adam_attr_t>(attr1);
  auto key2 = jit::JitCodeKey<jit::adam_attr_t>(attr2);
  auto key3 = jit::JitCodeKey<jit::adam_attr_t>(attr3);

  EXPECT_TRUE(key1 == key2);
  EXPECT_TRUE(key2 != key3);
}

TEST(JITKernel_key, sgd) {
  jit::sgd_attr_t attr1(1, 2, 3, 4, 5);
  jit::sgd_attr_t attr2(1, 2, 3, 4, 5);
  jit::sgd_attr_t attr3(9, 8, 7, 4, 6);
  jit::sgd_attr_t attr4(1, 2, 3, 6, 5);
  jit::sgd_attr_t attr5(10, 9, 8, 7, 6);

  auto key1 = jit::JitCodeKey<jit::sgd_attr_t>(attr1);
  auto key2 = jit::JitCodeKey<jit::sgd_attr_t>(attr2);
  auto key3 = jit::JitCodeKey<jit::sgd_attr_t>(attr3);
  auto key4 = jit::JitCodeKey<jit::sgd_attr_t>(attr4);
  auto key5 = jit::JitCodeKey<jit::sgd_attr_t>(attr5);

  EXPECT_TRUE(key1 == key2);
  EXPECT_TRUE(key2 == key3);
  EXPECT_TRUE(key3 != key4);
  EXPECT_TRUE(key3 != key5);
  EXPECT_TRUE(key4 != key5);
}

// test kernels
#define TestKernelVMul TestKernelXYZN
#define TestKernelVAdd TestKernelXYZN
#define TestKernelVAddRelu TestKernelXYZN
#define TestKernelVSub TestKernelXYZN

#define TestKernelVScal TestKernelAXYN
#define TestKernelVAddBias TestKernelAXYN

#define TestKernelVRelu TestKernelXYN
#define TestKernelVIdentity TestKernelXYN
#define TestKernelVSquare TestKernelXYN
#define TestKernelVExp TestKernelXYN
#define TestKernelVSigmoid TestKernelXYN
#define TestKernelVTanh TestKernelXYN
#define TestKernelVCopy TestKernelXYN

#define TestKernelHMax TestKernelXRN
#define TestKernelHSum TestKernelXRN

#define TestKernelLSTMCtHt TestKernelLSTM
#define TestKernelLSTMC1H1 TestKernelLSTM

#define TestKernelGRUH1 TestKernelGRU
#define TestKernelGRUHtPart1 TestKernelGRU
#define TestKernelGRUHtPart2 TestKernelGRU

#define TEST_CPU_KERNEL(kernel_type)                                      \
  TEST(JITKernel, kernel_type) {                                          \
    TestKernel##kernel_type<jit::kernel_type##Tuple<float>, CPUPlace>();  \
    TestKernel##kernel_type<jit::kernel_type##Tuple<double>, CPUPlace>(); \
  }

TEST_CPU_KERNEL(VMul);
TEST_CPU_KERNEL(VAdd);
TEST_CPU_KERNEL(VAddRelu);
TEST_CPU_KERNEL(VSub);

TEST_CPU_KERNEL(VScal);
TEST_CPU_KERNEL(VAddBias);

TEST_CPU_KERNEL(VRelu);
TEST_CPU_KERNEL(VIdentity);
TEST_CPU_KERNEL(VSquare);
TEST_CPU_KERNEL(VExp);
TEST_CPU_KERNEL(VSigmoid);
TEST_CPU_KERNEL(VTanh);
TEST_CPU_KERNEL(VCopy);

TEST_CPU_KERNEL(HMax);
TEST_CPU_KERNEL(HSum);

TEST_CPU_KERNEL(LSTMCtHt);
TEST_CPU_KERNEL(LSTMC1H1);

TEST_CPU_KERNEL(GRUH1);
TEST_CPU_KERNEL(GRUHtPart1);
TEST_CPU_KERNEL(GRUHtPart2);

TEST_CPU_KERNEL(NCHW16CMulNC);
TEST_CPU_KERNEL(LayerNorm);
TEST_CPU_KERNEL(CRFDecoding);

TEST_CPU_KERNEL(SeqPool);
TEST_CPU_KERNEL(EmbSeqPool);
TEST_CPU_KERNEL(MatMul);
TEST_CPU_KERNEL(Softmax);
TEST_CPU_KERNEL(Adam);
TEST_CPU_KERNEL(Sgd);
TEST_CPU_KERNEL(VBroadcast);

TEST_CPU_KERNEL(StrideASum);
TEST_CPU_KERNEL(StrideScal);
