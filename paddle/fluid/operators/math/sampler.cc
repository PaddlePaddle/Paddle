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

#include "paddle/fluid/operators/math/sampler.h"
#include <iostream>
#include <queue>
#include <utility>
#include <vector>

namespace paddle {
namespace operators {
namespace math {

Sampler::~Sampler() {}

UniformSampler::UniformSampler(int64_t range, unsigned int seed)
    : Sampler(range, seed), inv_range_(1.0 / (range + 1)) {
  random_engine_ = std::make_shared<std::mt19937_64>(seed_);
  dist_ = std::make_shared<std::uniform_int_distribution<>>(0, range);
}

int64_t UniformSampler::Sample() const { return (*dist_)(*random_engine_); }

float UniformSampler::Probability(int64_t value) const { return inv_range_; }

LogUniformSampler::LogUniformSampler(int64_t range, unsigned int seed)
    : Sampler(range, seed), log_range_(log(range + 1)) {
  random_engine_ = std::make_shared<std::mt19937_64>(seed_);
  dist_ = std::make_shared<std::uniform_real_distribution<>>(0, 1);
}

int64_t LogUniformSampler::Sample() const {
  // Got Log Uniform distribution from uniform distribution by
  // inverse_transform_sampling method
  // More details:
  // https://wanghaoshuang.github.io/2017/11/Log-uniform-distribution-sampler/
  const int64_t value =
      static_cast<int64_t>(exp((*dist_)(*random_engine_) * log_range_)) - 1;
  // Mathematically, value should be <= range_, but might not be due to some
  // floating point roundoff, so we mod by range_.
  return value % range_;
}

float LogUniformSampler::Probability(int64_t value) const {
  // Given f(x) = 1/[(x+1) * log_range_]
  // The value's  probability  is integral of f(x) from value to (value + 1)
  // More details:
  // https://wanghaoshuang.github.io/2017/11/Log-uniform-distribution-sampler
  return (log((value + 2.0) / (value + 1.0))) / log_range_;
}

CustomSampler::CustomSampler(int64_t range, const float* probabilities,
                             unsigned int seed)
    : Sampler(range, seed) {
  random_engine_ = std::make_shared<std::mt19937_64>(seed_);
  real_dist_ = std::make_shared<std::uniform_real_distribution<>>(0, 1);
  int_dist_ = std::make_shared<std::uniform_int_distribution<>>(0, range);
  alias_probs_ = std::make_shared<std::vector<float>>(range + 1);
  alias_ = std::make_shared<std::vector<int64_t>>(range + 1);
  probs_ = std::make_shared<std::vector<float>>(range + 1);

  std::queue<std::pair<int64_t, float>> bigs;
  std::queue<std::pair<int64_t, float>> littles;
  for (int64_t i = 0; i <= range; ++i) {
    (*probs_)[i] = probabilities[i];
    float normal_prob = probabilities[i] * (range + 1);
    if (normal_prob - 1.0 > 1e-4) {
      bigs.emplace(i, normal_prob);
    } else if (1.0 - normal_prob > 1e-4) {
      littles.emplace(i, normal_prob);
    } else {
      (*alias_probs_)[i] = normal_prob;
      (*alias_)[i] = -1;
    }
  }

  while ((!littles.empty()) && (!bigs.empty())) {
    auto big = bigs.front();
    auto little = littles.front();
    bigs.pop();
    littles.pop();
    (*alias_probs_)[little.first] = little.second;
    (*alias_)[little.first] = big.first;
    auto big_left = big.second - (1 - little.second);
    if (big_left - 1.0 > 1e-4) {
      bigs.emplace(big.first, big_left);
    } else if (1.0 - big_left > 1e-4) {
      littles.emplace(big.first, big_left);
    } else {
      (*alias_probs_)[big.first] = big_left;
      (*alias_)[big.first] = -1;
    }
  }

  if (!littles.empty()) {  // littles.second is close to 1.0
    auto little = littles.front();
    (*alias_probs_)[little.first] = 1.0;
    (*alias_)[little.first] = -1;
  }

  if (!bigs.empty()) {  // bigs.second is close to 1.0
    auto big = bigs.front();
    (*alias_probs_)[big.first] = 1.0;
    (*alias_)[big.first] = -1;
  }
}

int64_t CustomSampler::Sample() const {
  auto index = (*int_dist_)(*random_engine_);
  auto p = (*real_dist_)(*random_engine_);
  if (p > (*alias_probs_)[index]) {
    return (*alias_)[index];
  } else {
    return index;
  }
}

float CustomSampler::Probability(int64_t value) const {
  return (*probs_)[value];
}

}  // namespace math
}  // namespace operators
}  // namespace paddle
