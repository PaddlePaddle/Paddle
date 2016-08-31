/* Copyright (c) 2016 Baidu, Inc. All Rights Reserve.

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

#include <random>

#include "paddle/utils/TypeDefs.h"

namespace paddle {

/**
 * @brief Given the probability of N objects, the sampler random select
 * one of the object.
 * @note: prob does not have to be unnormalized.
 *
 * The space requirement is O(N)=O(N * sizeof(Interval)).
 * The computational complexity of generate one sample is O(1).
 */
class MultinomialSampler {
public:
  MultinomialSampler(const real* prob, int size);

  /**
   * @brief Generate a random sample.
   * @param g is a random number engine. See <random>.
   * @return Random integer.
   */
  template <typename URNG>
  int gen(URNG& g) {
    return gen1([&g, this]() { return rand_(g); });
  }

protected:
  /**
   * @brief Generation
   * @param[in] rand rand is a real random number distribution
   * for the range [0, size).
   * @return random int number or intervals_[random_int_number].otherId.
   */
  template <typename Rand>
  int gen1(Rand rand) {
    double r = rand();  // NOLINT
    int i = (int)r;
    r -= i;
    return r < intervals_[i].thresh ? i : intervals_[i].otherId;
  }

  struct Interval {
    int otherId;
    real thresh;
  };

  /// The probability of each interval will be 1./size
  std::vector<Interval> intervals_;
  std::uniform_real_distribution<double> rand_;
};

}  // namespace paddle
