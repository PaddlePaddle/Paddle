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
#include "paddle/fluid/operators/jitkernels/kernels.h"
// TODO(TJ): remove me
#include "paddle/fluid/operators/jitkernels/registry.h"

#include "paddle/fluid/platform/place.h"
#include "paddle/fluid/platform/port.h"

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

// TODO(TJ): remove me
USE_JITKERNEL_MORE(vmul, mkl);
USE_JITKERNEL_REFER(vmul);

TEST(JitKernel, vmul) {
  using T = float;
  using PlaceType = paddle::platform::CPUPlace;

  namespace jit = paddle::operators::jitkernels;
  // TODO(TJ): test more vector size
  for (int d = 1; d < 30; ++d) {
    auto ref = jit::GetRefer<jit::vmul, T,
                             void (*)(const T*, const T*, T*, int), int>();
    auto tgt = jit::Get<jit::vmul, T, void (*)(const T*, const T*, T*, int),
                        int, PlaceType>(d);
    EXPECT_TRUE(ref != nullptr);
    EXPECT_TRUE(tgt != nullptr);

    std::vector<T> x(d), y(d);
    std::vector<T> zref(d), ztgt(d);
    RandomVec<T>(d, x.data());
    RandomVec<T>(d, y.data());
    const float* x_data = x.data();
    const float* y_data = y.data();
    float* ztgt_data = ztgt.data();
    float* zref_data = zref.data();

    tgt(x_data, y_data, ztgt_data, d);
    ref(x_data, y_data, zref_data, d);
    ExpectEQ<T>(ztgt_data, zref_data, d);

    // test inplace x
    std::copy(x.begin(), x.end(), zref.begin());
    std::copy(x.begin(), x.end(), ztgt.begin());
    tgt(ztgt_data, y_data, ztgt_data, d);
    ref(zref_data, y_data, zref_data, d);
    ExpectEQ<T>(ztgt_data, zref_data, d);

    // test inplace y
    std::copy(y.begin(), y.end(), zref.begin());
    std::copy(y.begin(), y.end(), ztgt.begin());
    tgt(x_data, ztgt_data, ztgt_data, d);
    ref(x_data, zref_data, zref_data, d);
    ExpectEQ<T>(ztgt_data, zref_data, d);
  }
}

TEST(JitKernel, pool) {}
