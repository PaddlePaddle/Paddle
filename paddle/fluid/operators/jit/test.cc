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

#include <cstring>  // for memcpy
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
  for (int i = 1; i < 30; ++i) {
    s.push_back(i);
  }
  // test some large size
  s.push_back(100);
  s.push_back(1000);
  return s;
}

template <typename T, typename Func>
void TestTartgetFunc(const Func tgt, const std::vector<T>& x,
                     const std::vector<T>& y, const std::vector<T>& zref) {
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

TEST(JitKernel, vmul) {
  using T = float;
  using PlaceType = paddle::platform::CPUPlace;
  namespace jit = paddle::operators::jit;
  const auto KT = jit::vmul;
  for (int d : TestSizes()) {
    auto ref = jit::GetRefer<KT, T, jit::VMulTuples<T>::func_type,
                             jit::VMulTuples<T>::attr_type>();
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
    auto jitcode = jit::GetJitCode<KT, T, jit::VMulTuples<T>::func_type,
                                   jit::VMulTuples<T>::attr_type, PlaceType>(d);
    if (jitcode) {
      VLOG(10) << "Test jitcode, size: " << d;
      TestTartgetFunc<T, jit::VMulTuples<T>::func_type>(jitcode, x, y, zref);
    }

    // test all impls in more
    jit::KernelKey kkey(KT, PlaceType());
    auto& pool = jit::KernelPool().Instance().AllKernels();
    auto iter = pool.find(kkey);
    if (iter != pool.end()) {
      auto& impls = iter->second;
      for (auto& impl : impls) {
        auto i =
            dynamic_cast<const jit::KernelImpl<T, jit::VMulTuples<T>::func_type,
                                               jit::VMulTuples<T>::attr_type>*>(
                impl.get());
        if (i && i->UseMe(d)) {
          auto more = i->GetFunc();
          VLOG(10) << "Test More Kernel, size: " << d;
          TestTartgetFunc<T, jit::VMulTuples<T>::func_type>(more, x, y, zref);
        }
      }
    }
    // Test result from Get function
    VLOG(10) << "Test Get function, size: " << d;
    auto tgt = jit::Get<KT, T, jit::VMulTuples<T>::func_type,
                        jit::VMulTuples<T>::attr_type, PlaceType>(d);
    TestTartgetFunc<T, jit::VMulTuples<T>::func_type>(tgt, x, y, zref);
  }
}

TEST(JitKernel, pool) {}
