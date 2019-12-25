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

#include "paddle/fluid/platform/parallel.h"
#include <vector>
#include "gtest/gtest.h"

void FillRandom(std::vector<float>* x) {
  static unsigned int seed = 100;
  std::mt19937 rng(seed++);
  std::uniform_real_distribution<double> uniform_dist(0, 1);
  for (size_t i = 0; i < x->size(); ++i) {
    x->at(i) = static_cast<float>(uniform_dist(rng)) - 0.5;
  }
}

void FillZero(std::vector<float>* x) {
  for (size_t i = 0; i < x->size(); ++i) {
    x->at(i) = 0;
  }
}

TEST(Parallel, compute) {
  std::vector<float> x;
  std::vector<float> y;
  std::vector<float> z;

  auto parallel_add = [&](int64_t begin, int64_t end) {
    for (int64_t i = begin; i < end; ++i) {
      z[i] = x[i] + y[i];
    }
  };

  for (int64_t length : {1, 2, 3, 4, 255}) {
    x.resize(length);
    y.resize(length);
    z.resize(length);

    FillRandom(&x);
    FillRandom(&y);
    FillZero(&z);

    for (int num_threads : {1, 2, 3, 4}) {
      paddle::platform::SetNumThreads(num_threads);
      ASSERT_EQ(paddle::platform::GetMaxThreads(), num_threads);
      paddle::platform::RunParallelFor(0, length, parallel_add);

      for (int64_t i = 0; i < length; ++i) {
        ASSERT_NEAR(z[i], x[i] + y[i], 1e-5);
      }
    }
  }
}
