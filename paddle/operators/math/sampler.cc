#include "sampler.h"

namespace paddle {
namespace random {

Sampler::~Sampler() {}

UniformSampler::UniformSampler(int64 range)
    : Sampler(range), inv_range_(1.0 / range) {
  std::random_device r;
  random_engine_ = std::make_shared<std::mt19937>(r());
  dist_ = std::make_shared<std::uniform_int_distribution<>>(0, range);
}

int64 UniformSampler::Sample() const { return (*dist_)(*random_engine_); }

float UniformSampler::Probability(int64 value) const { return inv_range_; }

LogUniformSampler::LogUniformSampler(int64 range)
    : Sampler(range), log_range_(log(range + 1)) {
  std::random_device r;
  random_engine_ = std::make_shared<std::mt19937>(r());
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
