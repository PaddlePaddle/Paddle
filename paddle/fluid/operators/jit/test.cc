/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License. */

#include <random>
#include <string>
#include <vector>
#include "gflags/gflags.h"
#include "glog/logging.h"
#include "gtest/gtest.h"
#include "paddle/fluid/operators/jit/kernels.h"
#include "paddle/fluid/platform/cpu_info.h"
#include "paddle/fluid/platform/place.h"

static double acc = 1e-5;

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

template <typename T>
void ExpectEQ(const T* target, const T* refer, int n) {
  if (std::is_floating_point<T>::value) {
    for (int i = 0; i < n; ++i) {
      EXPECT_NEAR(target[i], refer[i], acc);
    }
  } else {
    for (int i = 0; i < n; ++i) {
      EXPECT_EQ(target[i], refer[i]);
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

template <typename KernelTuples, typename... Args>
struct TestFuncWithRefer {
  void operator()(const typename KernelTuples::func_type tgt, Args... args) {
    LOG(FATAL) << "Should specify this function.";
  }
};

template <typename T>
struct TestFuncWithRefer<jit::XYZNTuples<T>, std::vector<T>, std::vector<T>,
                         std::vector<T>> {
  void operator()(const typename jit::XYZNTuples<T>::func_type tgt,
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
  }
};

template <typename T>
struct TestFuncWithRefer<jit::AXYNTuples<T>, T, std::vector<T>,
                         std::vector<T>> {
  void operator()(const typename jit::AXYNTuples<T>::func_type tgt, const T a,
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
  }
};

template <typename T>
struct TestFuncWithRefer<jit::XYNTuples<T>, std::vector<T>, std::vector<T>> {
  void operator()(const typename jit::XYNTuples<T>::func_type tgt,
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
  }
};

template <typename T>
struct TestFuncWithRefer<jit::LSTMTuples<T>, std::vector<T>, std::vector<T>,
                         std::vector<T>, std::vector<T>, std::vector<T>,
                         typename jit::LSTMTuples<T>::attr_type> {
  void operator()(const typename jit::LSTMTuples<T>::func_type tgt,
                  const std::vector<T>& xsrc, const std::vector<T>& wp,
                  const std::vector<T>& ct_1, const std::vector<T>& ct_ref,
                  const std::vector<T>& ht_ref,
                  const typename jit::LSTMTuples<T>::attr_type& attr) {
    EXPECT_TRUE(tgt != nullptr);
    EXPECT_EQ(ct_ref.size(), ht_ref.size());
    EXPECT_EQ(ct_1.size(), ht_ref.size());
    EXPECT_EQ(xsrc.size(), 4 * ht_ref.size());
    EXPECT_EQ(wp.size(), 3 * ht_ref.size());

    // x could be changed after compute, so copy to save src
    int d = ht_ref.size();
    std::vector<T> x(xsrc.size()), ct(ct_ref.size()), ht(ht_ref.size());
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

    paddle::operators::jit::lstm_t step;
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
  }
};

template <typename T>
struct TestFuncWithRefer<jit::GRUTuples<T>, std::vector<T>, std::vector<T>,
                         std::vector<T>,
                         typename jit::GRUTuples<T>::attr_type> {
  void operator()(const typename jit::GRUTuples<T>::func_type tgt,
                  const std::vector<T>& xsrc, const std::vector<T>& ht_1,
                  const std::vector<T>& ht_ref,
                  const typename jit::GRUTuples<T>::attr_type& attr) {
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
    paddle::operators::jit::gru_t step;
    step.gates = x_data;
    step.ht_1 = ht_1_data;
    step.ht = ht_data;
    tgt(&step, &attr);
    ExpectEQ<T>(ht_data, ht_ref_data, d);
  }
};

template <typename T>
struct TestFuncWithRefer<jit::SeqPoolTuples<T>, std::vector<T>, std::vector<T>,
                         typename jit::SeqPoolTuples<T>::attr_type> {
  void operator()(const typename jit::SeqPoolTuples<T>::func_type tgt,
                  const std::vector<T>& x, const std::vector<T>& yref,
                  const typename jit::SeqPoolTuples<T>::attr_type& attr) {
    EXPECT_TRUE(tgt != nullptr);
    EXPECT_EQ(x.size() % yref.size(), 0);
    int w = yref.size();
    std::vector<T> y(w);
    const T* x_data = x.data();
    const T* yref_data = yref.data();
    T* y_data = y.data();
    tgt(x_data, y_data, &attr);
    ExpectEQ<T>(y_data, yref_data, w);
  }
};

template <typename T>
struct TestFuncWithRefer<jit::MatMulTuples<T>, std::vector<T>, std::vector<T>,
                         std::vector<T>, int, int, int> {
  void operator()(const typename jit::MatMulTuples<T>::func_type tgt,
                  const std::vector<T>& a, const std::vector<T>& b,
                  const std::vector<T>& cref, int m, int n, int k) {
    EXPECT_TRUE(tgt != nullptr);
    EXPECT_EQ(a.size(), static_cast<size_t>(m * k));
    EXPECT_EQ(b.size(), static_cast<size_t>(k * n));
    EXPECT_EQ(cref.size(), static_cast<size_t>(m * n));
    std::vector<T> c(cref.size());
    const T* a_data = a.data();
    const T* b_data = b.data();
    const T* cref_data = cref.data();
    T* c_data = c.data();
    tgt(a_data, b_data, c_data, m, n, k);
    ExpectEQ<T>(c_data, cref_data, m * n);
  }
};

template <paddle::operators::jit::KernelType KT, typename KernelTuples,
          typename PlaceType, typename... Args>
void TestAllImpls(const typename KernelTuples::attr_type& attr, Args... args) {
  TestFuncWithRefer<KernelTuples, Args...> test;
  // test jitcode
  auto jitcode = jit::GetJitCode<KT, KernelTuples, PlaceType>(attr);
  if (jitcode) {
    VLOG(10) << "Test Jitcode Kernel ";
    test(jitcode, args...);
  }
  // test all impls in more
  jit::KernelKey kkey(KT, PlaceType());
  auto& pool = jit::KernelPool().Instance().AllKernels();
  auto iter = pool.find(kkey);
  if (iter != pool.end()) {
    auto& impls = iter->second;
    for (auto& impl : impls) {
      auto i = dynamic_cast<const jit::KernelMore<KernelTuples>*>(impl.get());
      if (i && i->UseMe(attr)) {
        auto more = i->GetFunc();
        VLOG(10) << "Test More Kernel : " << i->ImplType();
        test(more, args...);
      }
    }
  }
  // test result from Get function
  // VLOG(10) << "Test Get function ";
  auto tgt = jit::Get<KT, KernelTuples, PlaceType>(attr);
  test(tgt, args...);
}

template <paddle::operators::jit::KernelType KT, typename T, typename PlaceType>
void TestXYZNKernel() {
  namespace jit = paddle::operators::jit;
  VLOG(10) << "===== Test JITKernel " << jit::to_string(KT);
  for (int d : TestSizes()) {
    auto ref = jit::GetRefer<KT, jit::XYZNTuples<T>>();
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

    TestAllImpls<KT, jit::XYZNTuples<T>, PlaceType, std::vector<T>,
                 std::vector<T>, std::vector<T>>(d, x, y, zref);
  }
}

template <paddle::operators::jit::KernelType KT, typename T, typename PlaceType>
void TestAXYNKernel() {
  namespace jit = paddle::operators::jit;
  VLOG(10) << "===== Test JITKernel " << jit::to_string(KT);
  for (int d : TestSizes()) {
    auto ref = jit::GetRefer<KT, jit::AXYNTuples<T>>();
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

    TestAllImpls<KT, jit::AXYNTuples<T>, PlaceType, T, std::vector<T>,
                 std::vector<T>>(d, a, x, yref);
  }
}

template <paddle::operators::jit::KernelType KT, typename T, typename PlaceType>
void TestXYNKernel() {
  namespace jit = paddle::operators::jit;
  VLOG(10) << "===== Test JITKernel " << jit::to_string(KT);
  for (int d : TestSizes()) {
    auto ref = jit::GetRefer<KT, jit::XYNTuples<T>>();
    EXPECT_TRUE(ref != nullptr);

    std::vector<T> x(d), yref(d);
    std::vector<T> xinp(d);  // inplace test
    RandomVec<T>(d, x.data(), -2.f, 2.f);
    std::copy(x.begin(), x.end(), xinp.begin());

    const T* x_data = x.data();
    T* yref_data = yref.data();
    T* xinp_data = xinp.data();
    // test refer code inplace
    ref(x_data, yref_data, d);
    ref(xinp_data, xinp_data, d);
    ExpectEQ<T>(xinp_data, yref_data, d);

    TestAllImpls<KT, jit::XYNTuples<T>, PlaceType, std::vector<T>,
                 std::vector<T>>(d, x, yref);
  }
}

template <paddle::operators::jit::KernelType KT, typename T, typename PlaceType>
void TestLSTMKernel() {
  namespace jit = paddle::operators::jit;
  VLOG(10) << "===== Test JITKernel " << jit::to_string(KT);
  std::vector<std::string> all_acts = {"sigmoid", "tanh", "relu", "identity"};
  for (int d : TestSizes()) {
    for (bool use_peephole : {true, false}) {
      for (auto& act_gate : all_acts) {
        for (auto& act_cand : all_acts) {
          for (auto& act_cell : all_acts) {
            const jit::lstm_attr_t attr(
                d, jit::to_kerneltype(act_gate), jit::to_kerneltype(act_cand),
                jit::to_kerneltype(act_cell), use_peephole);
            auto ref = jit::GetRefer<KT, jit::LSTMTuples<T>>();
            EXPECT_TRUE(ref != nullptr);
            std::vector<T> xsrc(4 * d), wp(3 * d), ct_1(d);
            std::vector<T> ct_ref(d), ht_ref(d), checked(2 * d);
            RandomVec<T>(4 * d, xsrc.data(), -2.f, 2.f);
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
            TestAllImpls<KT, jit::LSTMTuples<T>, PlaceType, std::vector<T>,
                         std::vector<T>, std::vector<T>, std::vector<T>,
                         std::vector<T>>(attr, xsrc, wp, ct_1, ct_ref, ht_ref,
                                         attr);
          }
        }
      }
    }
  }
}

template <paddle::operators::jit::KernelType KT, typename T, typename PlaceType>
void TestGRUKernel() {
  namespace jit = paddle::operators::jit;
  VLOG(10) << "===== Test JITKernel " << jit::to_string(KT);
  std::vector<std::string> all_acts = {"sigmoid", "tanh", "relu", "identity"};
  for (int d : TestSizes()) {
    for (auto& act_gate : all_acts) {
      for (auto& act_cand : all_acts) {
        const jit::gru_attr_t attr(d, jit::to_kerneltype(act_gate),
                                   jit::to_kerneltype(act_cand));
        auto ref = jit::GetRefer<KT, jit::GRUTuples<T>>();
        EXPECT_TRUE(ref != nullptr);
        std::vector<T> xsrc(3 * d), ht_1(d), ht_ref(d);
        RandomVec<T>(3 * d, xsrc.data(), -2.f, 2.f);
        RandomVec<T>(d, ht_1.data(), -2.f, 2.f);
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
        TestAllImpls<KT, jit::GRUTuples<T>, PlaceType, std::vector<T>,
                     std::vector<T>, std::vector<T>>(attr, xsrc, ht_1, ht_ref,
                                                     attr);
      }
    }
  }
}

template <paddle::operators::jit::KernelType KT, typename T, typename PlaceType>
void TestSeqPoolKernel() {
  VLOG(10) << "===== Test JITKernel " << jit::to_string(KT);
  std::vector<jit::SeqPoolType> pool_types = {
      jit::SeqPoolType::kSum, jit::SeqPoolType::kAvg, jit::SeqPoolType::kSqrt};
  for (auto type : pool_types) {
    for (int w : TestSizes()) {
      jit::seq_pool_attr_t attr(w, type);
      for (int h : TestSizes()) {
        attr.h = h;
        auto ref = jit::GetRefer<KT, jit::SeqPoolTuples<T>>();
        EXPECT_TRUE(ref != nullptr);
        std::vector<T> x(h * w), yref(w);
        RandomVec<T>(h * w, x.data(), -2.f, 2.f);
        const T* x_data = x.data();
        T* yref_data = yref.data();
        ref(x_data, yref_data, &attr);
        VLOG(10) << attr;
        TestAllImpls<KT, jit::SeqPoolTuples<T>, PlaceType, std::vector<T>,
                     std::vector<T>>(attr, x, yref, attr);
      }
    }
  }
}

template <paddle::operators::jit::KernelType KT, typename T, typename PlaceType>
void TestMatMulKernel() {
  VLOG(10) << "===== Test JITKernel " << jit::to_string(KT);
  auto last_acc = acc;
  // TODO(intel): this should be acc issue of MKL
  acc = 1e-3;
  for (int m : {1, 2, 3, 4}) {
    for (int n : {1, 2, 3, 4}) {
      for (int k : TestSizes()) {
        auto ref = jit::GetRefer<KT, jit::MatMulTuples<T>>();
        EXPECT_TRUE(ref != nullptr);
        std::vector<T> a(m * k), b(k * n), c(m * n);
        RandomVec<T>(m * k, a.data(), -2.f, 2.f);
        RandomVec<T>(k * n, b.data(), -2.f, 2.f);
        const T* a_data = a.data();
        const T* b_data = b.data();
        T* c_data = c.data();
        ref(a_data, b_data, c_data, m, n, k);
        TestAllImpls<KT, jit::MatMulTuples<T>, PlaceType, std::vector<T>,
                     std::vector<T>, std::vector<T>>(k, a, b, c, m, n, k);
      }
    }
  }
  acc = last_acc;
}

template <paddle::operators::jit::KernelType KT, typename T, typename PlaceType>
void TestNCHW16CMulNCKernel() {
  VLOG(10) << "===== Test JITKernel " << jit::to_string(KT);
  const int n = 3, c = 16 * 4, h = 10, w = 10;
  auto ref = jit::GetRefer<KT, jit::NCHW16CMulNCTuples<T>>();
  EXPECT_TRUE(ref != nullptr);
  int sz = n * c * h * w;
  std::vector<T> x(sz), y(n * c), zref(sz);
  std::vector<T> ztgt(sz), zjit(sz);
  RandomVec<T>(sz, x.data(), -2.f, 2.f);
  RandomVec<T>(n * c, y.data(), -2.f, 2.f);

  const T* x_data = x.data();
  const T* y_data = y.data();
  T* zref_data = zref.data();
  T* ztgt_data = ztgt.data();
  T* zjit_data = zjit.data();
  constexpr int simd_width = ZMM_FLOAT_BLOCK;
  int C = c / simd_width;
  auto tgt = jit::Get<KT, jit::NCHW16CMulNCTuples<T>, PlaceType>(0);
  auto jitcode = jit::GetJitCode<KT, jit::NCHW16CMulNCTuples<T>, PlaceType>(0);
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

// XYZNTuple
TEST(JITKernel, kVMul) {
  namespace jit = paddle::operators::jit;
  TestXYZNKernel<jit::kVMul, float, paddle::platform::CPUPlace>();
  TestXYZNKernel<jit::kVMul, double, paddle::platform::CPUPlace>();
}

TEST(JITKernel, kVAdd) {
  namespace jit = paddle::operators::jit;
  TestXYZNKernel<jit::kVAdd, float, paddle::platform::CPUPlace>();
  TestXYZNKernel<jit::kVAdd, double, paddle::platform::CPUPlace>();
}

TEST(JITKernel, kVAddRelu) {
  namespace jit = paddle::operators::jit;
  TestXYZNKernel<jit::kVAddRelu, float, paddle::platform::CPUPlace>();
  TestXYZNKernel<jit::kVAddRelu, double, paddle::platform::CPUPlace>();
}

TEST(JITKernel, kVSub) {
  namespace jit = paddle::operators::jit;
  TestXYZNKernel<jit::kVSub, float, paddle::platform::CPUPlace>();
  TestXYZNKernel<jit::kVSub, double, paddle::platform::CPUPlace>();
}

// AXYNTuples
TEST(JITKernel, kVScal) {
  namespace jit = paddle::operators::jit;
  TestAXYNKernel<jit::kVScal, float, paddle::platform::CPUPlace>();
  TestAXYNKernel<jit::kVScal, double, paddle::platform::CPUPlace>();
}

TEST(JITKernel, kVAddBias) {
  namespace jit = paddle::operators::jit;
  TestAXYNKernel<jit::kVAddBias, float, paddle::platform::CPUPlace>();
  TestAXYNKernel<jit::kVAddBias, double, paddle::platform::CPUPlace>();
}

// XYNTuples
TEST(JITKernel, kVRelu) {
  namespace jit = paddle::operators::jit;
  TestXYNKernel<jit::kVRelu, float, paddle::platform::CPUPlace>();
  TestXYNKernel<jit::kVRelu, double, paddle::platform::CPUPlace>();
}

TEST(JITKernel, kVIdentity) {
  namespace jit = paddle::operators::jit;
  TestXYNKernel<jit::kVIdentity, float, paddle::platform::CPUPlace>();
  TestXYNKernel<jit::kVIdentity, double, paddle::platform::CPUPlace>();
}

TEST(JITKernel, kVSquare) {
  namespace jit = paddle::operators::jit;
  TestXYNKernel<jit::kVSquare, float, paddle::platform::CPUPlace>();
  TestXYNKernel<jit::kVSquare, double, paddle::platform::CPUPlace>();
}

TEST(JITKernel, kVExp) {
  namespace jit = paddle::operators::jit;
  TestXYNKernel<jit::kVExp, float, paddle::platform::CPUPlace>();
  TestXYNKernel<jit::kVExp, double, paddle::platform::CPUPlace>();
}

TEST(JITKernel, kVSigmoid) {
  namespace jit = paddle::operators::jit;
  TestXYNKernel<jit::kVSigmoid, float, paddle::platform::CPUPlace>();
  TestXYNKernel<jit::kVSigmoid, double, paddle::platform::CPUPlace>();
}

TEST(JITKernel, kVTanh) {
  namespace jit = paddle::operators::jit;
  TestXYNKernel<jit::kVTanh, float, paddle::platform::CPUPlace>();
  TestXYNKernel<jit::kVTanh, double, paddle::platform::CPUPlace>();
}

// LSTM
TEST(JITKernel, kLSTMCtHt) {
  namespace jit = paddle::operators::jit;
  TestLSTMKernel<jit::kLSTMCtHt, float, paddle::platform::CPUPlace>();
  TestLSTMKernel<jit::kLSTMCtHt, double, paddle::platform::CPUPlace>();
}

TEST(JITKernel, kLSTMC1H1) {
  namespace jit = paddle::operators::jit;
  TestLSTMKernel<jit::kLSTMC1H1, float, paddle::platform::CPUPlace>();
  TestLSTMKernel<jit::kLSTMC1H1, double, paddle::platform::CPUPlace>();
}

// GRU
TEST(JITKernel, kGRUH1) {
  namespace jit = paddle::operators::jit;
  TestGRUKernel<jit::kGRUH1, float, paddle::platform::CPUPlace>();
  TestGRUKernel<jit::kGRUH1, double, paddle::platform::CPUPlace>();
}

TEST(JITKernel, kGRUHtPart1) {
  namespace jit = paddle::operators::jit;
  TestGRUKernel<jit::kGRUHtPart1, float, paddle::platform::CPUPlace>();
  TestGRUKernel<jit::kGRUHtPart1, double, paddle::platform::CPUPlace>();
}

TEST(JITKernel, kGRUHtPart2) {
  namespace jit = paddle::operators::jit;
  TestGRUKernel<jit::kGRUHtPart2, float, paddle::platform::CPUPlace>();
  TestGRUKernel<jit::kGRUHtPart2, double, paddle::platform::CPUPlace>();
}

TEST(JITKernel, kSeqPool) {
  namespace jit = paddle::operators::jit;
  TestSeqPoolKernel<jit::kSeqPool, float, paddle::platform::CPUPlace>();
  TestSeqPoolKernel<jit::kSeqPool, double, paddle::platform::CPUPlace>();
}

TEST(JITKernel, kMatMul) {
  namespace jit = paddle::operators::jit;
  TestMatMulKernel<jit::kMatMul, float, paddle::platform::CPUPlace>();
  TestMatMulKernel<jit::kMatMul, double, paddle::platform::CPUPlace>();
}

TEST(JITKernel, kNCHW16CMulNC) {
  namespace jit = paddle::operators::jit;
  TestNCHW16CMulNCKernel<jit::kNCHW16CMulNC, float,
                         paddle::platform::CPUPlace>();
  TestNCHW16CMulNCKernel<jit::kNCHW16CMulNC, double,
                         paddle::platform::CPUPlace>();
}

// TODO(yihua/TJ): add crf decoding and layer norm unit tests

TEST(JITKernel, pool) {
  // TODO(TJ): add some test
}
