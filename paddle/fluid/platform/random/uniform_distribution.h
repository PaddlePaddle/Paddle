// Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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
#include "paddle/fluid/platform/random/philox_engine.h"

namespace paddle {
namespace platform {
namespace random {

template <typename T>
class UniformRealDistribution;

template <>
class UniformRealDistribution<float> {
 public:
  using ResultType = float;
  static constexpr size_t N = 4;

  inline HOSTDEVICE UniformRealDistribution(float min, float max)
      : min_(min), max_(max), pos_(N) {}

  inline HOSTDEVICE float operator()(Philox32x4& eng) {
    if (pos_ == result_.size) {
      // regenerate from engine
      result_ = eng();
      pos_ = 0;
    }
    uint32_t result = result_[pos_++];
    // cast result to float first, since result could be UINT32_MAX
    float result_fp = (static_cast<float>(result) + 1) / UINT32_MAX;
    return (result_fp * (max_ - min_)) + min_;
  }

 private:
  float min_;
  float max_;
  LargeInt<4> result_;
  size_t pos_;
};

template <>
class UniformRealDistribution<double> {
 public:
  using ResultType = double;
  static constexpr size_t N = 2;

  inline HOSTDEVICE UniformRealDistribution(double min, double max)
      : min_(min), max_(max), pos_(result_.size) {}

  inline HOSTDEVICE double operator()(Philox32x4& eng) {
    if (pos_ == result_.size) {
      // regenerate from engine
      result_ = eng();
      pos_ = 0;
    }
    uint64_t result = result_[pos_++];
    result <<= 32;
    result |= result_[pos_++];
    // cast result to double first, since result could be UINT64_MAX
    double result_fp = (static_cast<double>(result) + 1) / UINT64_MAX;
    return (result_fp * (max_ - min_)) + min_;
  }

 private:
  float min_;
  float max_;
  LargeInt<4> result_;
  size_t pos_;
};

}  // namespace random
}  // namespace platform
}  // namespace paddle
