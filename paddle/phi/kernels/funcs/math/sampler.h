// Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include <cstdint>
#include <memory>
#include <random>
#include <vector>

#include "paddle/phi/core/enforce.h"

namespace phi {
namespace math {

// TODO(wanghaoshuang): Support for GPU

/**
 * Sample integers from [0, range).
 */
class Sampler {
 public:
  explicit Sampler(int64_t range, unsigned int seed = 0UL) : range_(range) {
    PADDLE_ENFORCE_GT(
        range,
        0,
        common::errors::InvalidArgument(
            "Range should be greater than 0, but received %d.", range));
    if (seed == 0) {
      std::random_device r;
      seed_ = r();
    } else {
      seed_ = seed;
    }
  }

  virtual ~Sampler();

  // Sample a single value
  virtual int64_t Sample() const = 0;

  // The probability that a single call to Sample() returns the given value.
  virtual float Probability(int64_t value) const = 0;

  int64_t range() { return range_; }

 protected:
  const int64_t range_;
  unsigned int seed_;
};

/**
 * Sample integers from [0, range).
 * And the distribution function is:
 * P(x) = 1 / range
 */
class UniformSampler : public Sampler {
 public:
  explicit UniformSampler(int64_t range, unsigned int seed = 0UL);

  ~UniformSampler() override {}

  int64_t Sample() const override;

  float Probability(int64_t value) const override;

 private:
  const float inv_range_;
  std::shared_ptr<std::mt19937_64> random_engine_;
  std::shared_ptr<std::uniform_int_distribution<>> dist_;
};

/**
 * Sample integers from [0, range).
 * And the distribution function is:
 * P(x) = (1/ln(range+1)) * ln(1 + 1/(x + 1))
 */
class LogUniformSampler : public Sampler {
 public:
  explicit LogUniformSampler(int64_t range, unsigned int seed = 0UL);

  ~LogUniformSampler() override {}

  int64_t Sample() const override;

  float Probability(int64_t value) const override;

 private:
  const float log_range_;
  std::shared_ptr<std::mt19937_64> random_engine_;
  std::shared_ptr<std::uniform_real_distribution<>> dist_;
};

/**
 * Sample integers from [0, range) from custom distribution.
 */
class CustomSampler : public Sampler {
 public:
  explicit CustomSampler(int64_t range,
                         const float* probabilities,
                         const int* alias,
                         const float* alias_probabilities,
                         unsigned int seed = 0UL);

  ~CustomSampler() override {}

  int64_t Sample() const override;

  float Probability(int64_t value) const override;

 private:
  const float* alias_probs_;
  const int* alias_;
  const float* probs_;
  const int exceptional_val = -1;
  std::shared_ptr<std::mt19937_64> random_engine_;
  std::shared_ptr<std::uniform_real_distribution<>> real_dist_;
  std::shared_ptr<std::uniform_int_distribution<>> int_dist_;
};

}  // namespace math
}  // namespace phi
