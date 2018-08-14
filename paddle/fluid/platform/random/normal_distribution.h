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

#include <cmath>
#include "paddle/fluid/platform/random/philox_engine.h"
#include "paddle/fluid/platform/random/uniform_distribution.h"
namespace paddle {
namespace platform {
namespace random {

template <typename T>
class NormalDistribution {
 public:
  constexpr static size_t N = UniformRealDistribution<T>::N;
  using ResultType = T;
  inline HOSTDEVICE NormalDistribution(T mean, T std)
      : mean_(mean), std_(std), uniform_(0, 1), is_valid_{false} {}

  inline HOSTDEVICE T operator()(Philox32x4& eng) {
    // shameless borrow from
    // thrust::random::detail::normal_distribution_portable
    // using Marsaglia's "polar method"
    if (!is_valid_) {
      r1_ = uniform_(eng);
      r2_ = uniform_(eng);
      cached_rho_ =
          std::sqrt(-static_cast<T>(2) * std::log(static_cast<T>(1) - r2_));
      is_valid_ = true;
    } else {
      is_valid_ = false;
    }
    T result = cached_rho_ *
               (is_valid_ ? std::cos(TwoPi * r1_) : std::sin(TwoPi * r2_));
    return mean_ + std_ * result;
  }

 private:
  T mean_;
  T std_;
  UniformRealDistribution<T> uniform_;
  T r1_;
  T r2_;
  T cached_rho_;
  T is_valid_;
  constexpr static T TwoPi = 3.14159265358979323846 * 2;
};

}  // namespace random
}  // namespace platform
}  // namespace paddle
