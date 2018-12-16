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
#include "paddle/fluid/platform/place.h"

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
      EXPECT_NEAR(target[i], refer[i], 1e-3);
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
  void operator()(const typename KernelTuples::func_type tgt, Args... args) {}
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
                         std::vector<T>, std::vector<T>, std::vector<T>> {
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
                         std::vector<T>> {
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
      auto i = dynamic_cast<const jit::KernelImpl<KernelTuples>*>(impl.get());
      if (i && i->UseMe(attr)) {
        auto more = i->GetFunc();
        VLOG(10) << "Test More Kernel ";
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
            RandomVec<T>(3 * d, wp.data(), -2.f, 2.f);
            RandomVec<T>(d, ct_1.data(), -2.f, 2.f);
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

// XYZNTuple
TEST(JITKernel, vmul) {
  namespace jit = paddle::operators::jit;
  TestXYZNKernel<jit::vmul, float, paddle::platform::CPUPlace>();
  TestXYZNKernel<jit::vmul, double, paddle::platform::CPUPlace>();
}

TEST(JITKernel, vadd) {
  namespace jit = paddle::operators::jit;
  TestXYZNKernel<jit::vadd, float, paddle::platform::CPUPlace>();
  TestXYZNKernel<jit::vadd, double, paddle::platform::CPUPlace>();
}

TEST(JITKernel, vaddrelu) {
  namespace jit = paddle::operators::jit;
  TestXYZNKernel<jit::vaddrelu, float, paddle::platform::CPUPlace>();
  TestXYZNKernel<jit::vaddrelu, double, paddle::platform::CPUPlace>();
}

TEST(JITKernel, vsub) {
  namespace jit = paddle::operators::jit;
  TestXYZNKernel<jit::vsub, float, paddle::platform::CPUPlace>();
  TestXYZNKernel<jit::vsub, double, paddle::platform::CPUPlace>();
}

// AXYNTuples
TEST(JITKernel, vscal) {
  namespace jit = paddle::operators::jit;
  TestAXYNKernel<jit::vscal, float, paddle::platform::CPUPlace>();
  TestAXYNKernel<jit::vscal, double, paddle::platform::CPUPlace>();
}

TEST(JITKernel, vaddbias) {
  namespace jit = paddle::operators::jit;
  TestAXYNKernel<jit::vaddbias, float, paddle::platform::CPUPlace>();
  TestAXYNKernel<jit::vaddbias, double, paddle::platform::CPUPlace>();
}

// XYNTuples
TEST(JITKernel, vrelu) {
  namespace jit = paddle::operators::jit;
  TestXYNKernel<jit::vrelu, float, paddle::platform::CPUPlace>();
  TestXYNKernel<jit::vrelu, double, paddle::platform::CPUPlace>();
}

TEST(JITKernel, videntity) {
  namespace jit = paddle::operators::jit;
  TestXYNKernel<jit::videntity, float, paddle::platform::CPUPlace>();
  TestXYNKernel<jit::videntity, double, paddle::platform::CPUPlace>();
}

TEST(JITKernel, vexp) {
  namespace jit = paddle::operators::jit;
  TestXYNKernel<jit::vexp, float, paddle::platform::CPUPlace>();
  TestXYNKernel<jit::vexp, double, paddle::platform::CPUPlace>();
}

TEST(JITKernel, vsigmoid) {
  namespace jit = paddle::operators::jit;
  TestXYNKernel<jit::vsigmoid, float, paddle::platform::CPUPlace>();
  TestXYNKernel<jit::vsigmoid, double, paddle::platform::CPUPlace>();
}

TEST(JITKernel, vtanh) {
  namespace jit = paddle::operators::jit;
  TestXYNKernel<jit::vtanh, float, paddle::platform::CPUPlace>();
  TestXYNKernel<jit::vtanh, double, paddle::platform::CPUPlace>();
}

// LSTM
TEST(JITKernel, lstmctht) {
  namespace jit = paddle::operators::jit;
  TestLSTMKernel<jit::lstmctht, float, paddle::platform::CPUPlace>();
  TestLSTMKernel<jit::lstmctht, double, paddle::platform::CPUPlace>();
}

TEST(JITKernel, lstmc1h1) {
  namespace jit = paddle::operators::jit;
  TestLSTMKernel<jit::lstmc1h1, float, paddle::platform::CPUPlace>();
  TestLSTMKernel<jit::lstmc1h1, double, paddle::platform::CPUPlace>();
}

// GRU
TEST(JITKernel, gruh1) {
  namespace jit = paddle::operators::jit;
  TestGRUKernel<jit::gruh1, float, paddle::platform::CPUPlace>();
  TestGRUKernel<jit::gruh1, double, paddle::platform::CPUPlace>();
}

TEST(JITKernel, gruhtpart1) {
  namespace jit = paddle::operators::jit;
  TestGRUKernel<jit::gruhtpart1, float, paddle::platform::CPUPlace>();
  TestGRUKernel<jit::gruhtpart1, double, paddle::platform::CPUPlace>();
}

TEST(JITKernel, gruhtpart2) {
  namespace jit = paddle::operators::jit;
  TestGRUKernel<jit::gruhtpart2, float, paddle::platform::CPUPlace>();
  TestGRUKernel<jit::gruhtpart2, double, paddle::platform::CPUPlace>();
}

// TODO(TJ): refine the tests template

TEST(JITKernel, pool) {
  // TODO(TJ): add some test
}
