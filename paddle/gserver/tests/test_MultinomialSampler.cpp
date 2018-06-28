/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include <random>

#include <gtest/gtest.h>
#include <vector>

#undef PADDLE_DISABLE_TIMER
#include "paddle/utils/Stat.h"

#include "paddle/gserver/layers/MultinomialSampler.h"
#include "paddle/utils/Util.h"

using namespace paddle;  // NOLINT
using namespace std;     // NOLINT

class MultinomialSamplerTester : public MultinomialSampler {
 public:
  MultinomialSamplerTester(real* prob, int size)
      : MultinomialSampler(prob, size) {}

  template <typename Rand1>
  int testGen(Rand1 rand1) {
    return gen1(rand1);
  }
};

TEST(MultinomialSampler, gen) {
  int numGrids = 1024 * 1024;
  int size = 1024 * 4;
  default_random_engine reng;

  for (size_t iter = 0; iter < 256; ++iter) {
    uniform_int_distribution<int> rand(1, numGrids / size * 1.8);
    vector<real> prob;
    int sum = 0;
    for (int i = 0; i < size; ++i) {
      prob.push_back(rand(reng));
      sum += prob.back();
    }

    CHECK_LE(sum, numGrids);
    prob.back() += numGrids - sum;

    vector<int> counts(size);
    MultinomialSamplerTester sampler(&prob[0], size);
    counts.assign(size, 0);
    {
      double s = (double)size / (double)numGrids;
      REGISTER_TIMER("MultinomialSampler");
      for (double i = 0; i < numGrids; ++i) {
        int ret = sampler.testGen([i, s]() { return s * i; });
        if (ret < 0 || ret >= size) {
          EXPECT_GE(ret, 0);
          EXPECT_LT(ret, size);
          break;
        }
        ++counts[ret];
      }
    }
    for (int i = 0; i < size; ++i) {
      if (prob[i] != counts[i]) {
        EXPECT_EQ(prob[i], counts[i]);
        LOG(INFO) << iter;
        break;
      }
    }
  }
}

void benchmarkRandom() {
  int n = 1024 * 1024;

  int sum;
  double sum1;

  sum = 0;
  unsigned int seed = 1;
  {
    REGISTER_TIMER("crand");
    for (int i = 0; i < n; ++i) {
      sum += rand_r(&seed) % 1000;
    }
  }
  LOG(INFO) << "sum=" << sum;

  default_random_engine reng;
  uniform_int_distribution<int> rand(1, 1000);
  sum = 0;
  {
    REGISTER_TIMER("stdrand");
    for (int i = 0; i < n; ++i) {
      sum += rand(reng);
    }
  }
  LOG(INFO) << "sum=" << sum;

  sum = 0;
  {
    REGISTER_TIMER("default_random_engine");
    for (int i = 0; i < n; ++i) {
      sum += reng();
    }
  }
  LOG(INFO) << "sum=" << sum;

  uniform_real_distribution<double> rand1(0, 1);
  sum1 = 0;
  {
    REGISTER_TIMER("stdrand1");
    for (int i = 0; i < n; ++i) {
      sum1 += rand1(reng);
    }
  }
  LOG(INFO) << "sum1=" << sum1;

  sum1 = 0;
  {
    real a = 1.0f / (real)RAND_MAX;
    REGISTER_TIMER("crand1");
    for (int i = 0; i < n; ++i) {
      sum1 += a * rand_r(&seed);
    }
  }
  LOG(INFO) << "sum1=" << sum1;
}

int main(int argc, char** argv) {
  initMain(argc, argv);
  testing::InitGoogleTest(&argc, argv);
  benchmarkRandom();
  int ret = RUN_ALL_TESTS();
  globalStat.printSegTimerStatus();
  return ret;
}
