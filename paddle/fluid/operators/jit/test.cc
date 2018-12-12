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

template <typename T, typename KernelTuples>
void TestXYZNFunc(const typename KernelTuples::func_type tgt,
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

    // test jitcode
    auto jitcode = jit::GetJitCode<KT, jit::XYZNTuples<T>, PlaceType>(d);
    if (jitcode) {
      VLOG(10) << "Test Jitcode Kernel, size: " << d;
      TestXYZNFunc<T, jit::XYZNTuples<T>>(jitcode, x, y, zref);
    }

    // test all impls in more
    jit::KernelKey kkey(KT, PlaceType());
    auto& pool = jit::KernelPool().Instance().AllKernels();
    auto iter = pool.find(kkey);
    if (iter != pool.end()) {
      auto& impls = iter->second;
      for (auto& impl : impls) {
        auto i = dynamic_cast<const jit::KernelImpl<jit::XYZNTuples<T>>*>(
            impl.get());
        if (i && i->UseMe(d)) {
          auto more = i->GetFunc();
          VLOG(10) << "Test More Kernel, size: " << d;
          TestXYZNFunc<T, jit::XYZNTuples<T>>(more, x, y, zref);
        }
      }
    }
    // Test result from Get function
    VLOG(10) << "Test Get function, size: " << d;
    auto tgt = jit::Get<KT, jit::XYZNTuples<T>, PlaceType>(d);
    TestXYZNFunc<T, jit::XYZNTuples<T>>(tgt, x, y, zref);
  }
}

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

template <typename T, typename KernelTuples>
void TestAXYNFunc(const typename KernelTuples::func_type tgt, const T a,
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

    // test jitcode
    auto jitcode = jit::GetJitCode<KT, jit::AXYNTuples<T>, PlaceType>(d);
    if (jitcode) {
      VLOG(10) << "Test Jitcode Kernel, size: " << d;
      TestAXYNFunc<T, jit::AXYNTuples<T>>(jitcode, a, x, yref);
    }

    // test all impls in more
    jit::KernelKey kkey(KT, PlaceType());
    auto& pool = jit::KernelPool().Instance().AllKernels();
    auto iter = pool.find(kkey);
    if (iter != pool.end()) {
      auto& impls = iter->second;
      for (auto& impl : impls) {
        auto i = dynamic_cast<const jit::KernelImpl<jit::AXYNTuples<T>>*>(
            impl.get());
        if (i && i->UseMe(d)) {
          auto more = i->GetFunc();
          VLOG(10) << "Test More Kernel, size: " << d;
          TestAXYNFunc<T, jit::AXYNTuples<T>>(more, a, x, yref);
        }
      }
    }
    // Test result from Get function
    VLOG(10) << "Test Get function, size: " << d;
    auto tgt = jit::Get<KT, jit::AXYNTuples<T>, PlaceType>(d);
    TestAXYNFunc<T, jit::AXYNTuples<T>>(tgt, a, x, yref);
  }
}

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

TEST(JITKernel, pool) {
  // TODO(TJ): add some test
}
