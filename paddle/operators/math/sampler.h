/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserve.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#pragma once
#include <memory>
#include <random>
typedef long int64;
namespace paddle {
namespace operators {
namespace math {

// TODO: Support for GPU

/**
* Sample integers from [0, range).
*/
class Sampler {
 public:
  explicit Sampler(int64 range) : range_(range) { /* check range > 0*/
  }
  virtual ~Sampler();
  // Sample a single value
  virtual int64 Sample() const = 0;
  // The probability that a single call to Sample() returns the given value.
  virtual float Probability(int64 value) const = 0;

  int64 range() { return range_; };

 protected:
  const int64 range_;
};

/**
 * Sample integers from [0, range).
 * And the distribution function is:
 * P(x) = 1 / range
 */
class UniformSampler : public Sampler {
 public:
  explicit UniformSampler(int64 range);

  ~UniformSampler() override {}

  int64 Sample() const override;

  float Probability(int64 value) const override;

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
  explicit LogUniformSampler(int64 range);

  ~LogUniformSampler() override {}

  int64 Sample() const override;

  float Probability(int64 value) const override;

 private:
  const float log_range_;
  std::shared_ptr<std::mt19937_64> random_engine_;
  std::shared_ptr<std::uniform_real_distribution<>> dist_;
};

}  // math
}  // namespace operators
}  // namespace paddle
