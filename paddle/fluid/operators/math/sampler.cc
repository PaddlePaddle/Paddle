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

#include "paddle/fluid/operators/math/sampler.h"

namespace paddle {
namespace random {

Sampler::~Sampler() {}

UniformSampler::UniformSampler(int64 range)
    : Sampler(range), inv_range_(1.0 / range) {
  random_engine_ = std::make_shared<std::mt19937>(seed_);
  dist_ = std::make_shared<std::uniform_int_distribution<>>(0, range);
}

UniformSampler::UniformSampler(int64 range, unsigned int seed)
    : Sampler(range, seed), inv_range_(1.0 / range) {
  random_engine_ = std::make_shared<std::mt19937>(seed_);
  dist_ = std::make_shared<std::uniform_int_distribution<>>(0, range);
}

int64 UniformSampler::Sample() const { return (*dist_)(*random_engine_); }

float UniformSampler::Probability(int64 value) const { return inv_range_; }

LogUniformSampler::LogUniformSampler(int64 range)
    : Sampler(range), log_range_(log(range + 1)) {
  random_engine_ = std::make_shared<std::mt19937>(seed_);
  dist_ = std::make_shared<std::uniform_real_distribution<>>(0, 1);
}

LogUniformSampler::LogUniformSampler(int64 range, unsigned int seed)
    : Sampler(range, seed), log_range_(log(range + 1)) {
  random_engine_ = std::make_shared<std::mt19937>(seed_);
  dist_ = std::make_shared<std::uniform_real_distribution<>>(0, 1);
}
int64 LogUniformSampler::Sample() const {
  // Got Log Uniform distribution from uniform distribution by
  // inverse_transform_sampling method
  // More details:
  // https://wanghaoshuang.github.io/2017/11/Log-uniform-distribution-sampler/
  const int64 value =
      static_cast<int64>(exp((*dist_)(*random_engine_) * log_range_)) - 1;
  // Mathematically, value should be <= range_, but might not be due to some
  // floating point roundoff, so we mod by range_.
  return value % range_;
}

float LogUniformSampler::Probability(int64 value) const {
  // Given f(x) = 1/[(x+1) * log_range_]
  // The value's  probability  is integral of f(x) from value to (value + 1)
  // More details:
  // https://wanghaoshuang.github.io/2017/11/Log-uniform-distribution-sampler
  return (log((value + 2.0) / (value + 1.0))) / log_range_;
}

}  // namespace random
}  // namespace paddle
