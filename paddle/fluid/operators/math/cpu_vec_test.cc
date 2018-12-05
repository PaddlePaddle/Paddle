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

#include <cmath>
#include <cstring>
#include <random>
#include <vector>
#include "gflags/gflags.h"
#include "glog/logging.h"
#include "gtest/gtest.h"

#include "paddle/fluid/operators/math/cpu_vec.h"
#include "paddle/fluid/platform/port.h"

inline double GetCurrentUS() {
  struct timeval time;
  gettimeofday(&time, NULL);
  return 1e+6 * time.tv_sec + time.tv_usec;
}
constexpr int repeat = 1000;

template <typename T>
inline T _sigmoid(T x) {
  const T min = SIGMOID_THRESHOLD_MIN;
  const T max = SIGMOID_THRESHOLD_MAX;
  T tmp = (x < min) ? min : ((x > max) ? max : x);
  return static_cast<T>(1) / (static_cast<T>(1) + std::exp(-tmp));
}

template <typename T>
inline T _tanh(T x) {
  return static_cast<T>(2) * _sigmoid<T>(static_cast<T>(2) * x) -
         static_cast<T>(1);
}

template <typename T>
void ref_sigmoid(const int n, const T* x, T* y) {
  for (int i = 0; i < n; ++i) {
    y[i] = _sigmoid(x[i]);
  }
}

template <typename T>
void ref_tanh(const int n, const T* x, T* y) {
  for (int i = 0; i < n; ++i) {
    y[i] = _tanh(x[i]);
  }
}
template <typename T>
void ref_relu(const int n, const T* x, T* y) {
  for (int i = 0; i < n; ++i) {
    y[i] = x[i] > 0 ? x[i] : 0;
  }
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

template <typename T>
void TestAndBench(const int n, std::function<void(const int, const T*, T*)> tgt,
                  std::function<void(const int, const T*, T*)> ref) {
  std::vector<T> x(n);
  std::vector<T> ytgt(n), yref(n);
  RandomVec<T>(n, x.data());

  const T* x_data = x.data();
  T* ytgt_data = ytgt.data();
  T* yref_data = yref.data();
  auto st = GetCurrentUS();
  for (int i = 0; i < repeat; ++i) {
    tgt(n, x_data, ytgt_data);
  }
  auto mt = GetCurrentUS();
  for (int i = 0; i < repeat; ++i) {
    ref(n, x_data, yref_data);
  }
  auto et = GetCurrentUS();

  VLOG(3) << "Vec size " << n << ": refer takes: " << (et - mt) / repeat
          << " us, tgt takes: " << (mt - st) / repeat;
  for (int i = 0; i < n; ++i) {
    EXPECT_NEAR(ytgt_data[i], yref_data[i], 1e-3);
  }
}

TEST(CpuVecTest, sigmoid) {
  namespace platform = paddle::platform;
  using namespace paddle::operators::math;  // NOLINT
  for (auto sz : {1, 2, 15, 16, 30, 32, 128, 200, 512}) {
    TestAndBench<float>(sz, vec_sigmoid<float>, ref_sigmoid<float>);
    TestAndBench<float>(sz, vec_sigmoid<float, platform::avx>,
                        ref_sigmoid<float>);
    TestAndBench<float>(sz, vec_sigmoid<float, platform::avx2>,
                        ref_sigmoid<float>);
    TestAndBench<float>(sz, vec_sigmoid<float, platform::avx512f>,
                        ref_sigmoid<float>);
  }
  TestAndBench<double>(30, vec_sigmoid<double>, ref_sigmoid<double>);
}

TEST(CpuVecTest, tanh) {
  namespace platform = paddle::platform;
  using namespace paddle::operators::math;  // NOLINT
  for (auto sz : {1, 2, 15, 16, 30, 32, 128, 200, 512}) {
    TestAndBench<float>(sz, vec_tanh<float>, ref_tanh<float>);
    TestAndBench<float>(sz, vec_tanh<float, platform::avx>, ref_tanh<float>);
    TestAndBench<float>(sz, vec_tanh<float, platform::avx2>, ref_tanh<float>);
    TestAndBench<float>(sz, vec_tanh<float, platform::avx512f>,
                        ref_tanh<float>);
  }
  TestAndBench<double>(30, vec_tanh<double>, ref_tanh<double>);
}

TEST(CpuVecTest, relu) {
  namespace platform = paddle::platform;
  using namespace paddle::operators::math;  // NOLINT
  for (auto sz : {1, 2, 15, 16, 30, 32, 128, 200, 512}) {
    TestAndBench<float>(sz, vec_relu<float>, ref_relu<float>);
    TestAndBench<float>(sz, vec_relu<float, platform::avx>, ref_relu<float>);
    TestAndBench<float>(sz, vec_relu<float, platform::avx2>, ref_relu<float>);
    TestAndBench<float>(sz, vec_relu<float, platform::avx512f>,
                        ref_relu<float>);
  }
  TestAndBench<double>(30, vec_relu<double>, ref_relu<double>);
}

template <typename T>
void TestInplace(const int n, std::function<void(const int, const T*, T*)> tgt,
                 std::function<void(const int, const T*, T*)> ref) {
  std::vector<T> x(n);
  std::vector<T> ytgt(n), yref(n);
  RandomVec<T>(n, x.data());

  const T* x_data = x.data();
  T* yref_data = yref.data();
  T* ytgt_data = ytgt.data();
  std::memcpy(yref_data, x_data, sizeof(T) * n);
  std::memcpy(ytgt_data, x_data, sizeof(T) * n);

  ref(n, yref_data, yref_data);
  tgt(n, ytgt_data, ytgt_data);

  for (int i = 0; i < n; ++i) {
    EXPECT_NEAR(ytgt_data[i], yref_data[i], 1e-3);
  }
}

TEST(CpuVecTest, inplace_sigmoid) {
  namespace platform = paddle::platform;
  using namespace paddle::operators::math;  // NOLINT
  for (auto sz : {1, 2, 15, 16, 30, 32, 128, 200, 512}) {
    TestInplace<float>(sz, vec_sigmoid<float>, ref_sigmoid<float>);
    TestInplace<float>(sz, vec_sigmoid<float, platform::avx>,
                       ref_sigmoid<float>);
    TestInplace<float>(sz, vec_sigmoid<float, platform::avx2>,
                       ref_sigmoid<float>);
    TestInplace<float>(sz, vec_sigmoid<float, platform::avx512f>,
                       ref_sigmoid<float>);
  }
  TestInplace<double>(30, vec_sigmoid<double>, ref_sigmoid<double>);
}

TEST(CpuVecTest, inplace_tanh) {
  namespace platform = paddle::platform;
  using namespace paddle::operators::math;  // NOLINT
  for (auto sz : {1, 2, 15, 16, 30, 32, 128, 200, 512}) {
    TestInplace<float>(sz, vec_tanh<float>, ref_tanh<float>);
    TestInplace<float>(sz, vec_tanh<float, platform::avx>, ref_tanh<float>);
    TestInplace<float>(sz, vec_tanh<float, platform::avx2>, ref_tanh<float>);
    TestInplace<float>(sz, vec_tanh<float, platform::avx512f>, ref_tanh<float>);
  }
  TestInplace<double>(30, vec_tanh<double>, ref_tanh<double>);
}

TEST(CpuVecTest, inplace_relu) {
  namespace platform = paddle::platform;
  using namespace paddle::operators::math;  // NOLINT
  for (auto sz : {1, 2, 15, 16, 30, 32, 128, 200, 512}) {
    TestInplace<float>(sz, vec_relu<float>, ref_relu<float>);
    TestInplace<float>(sz, vec_relu<float, platform::avx>, ref_relu<float>);
    TestInplace<float>(sz, vec_relu<float, platform::avx2>, ref_relu<float>);
    TestInplace<float>(sz, vec_relu<float, platform::avx512f>, ref_relu<float>);
  }
  TestInplace<double>(30, vec_relu<double>, ref_relu<double>);
}
