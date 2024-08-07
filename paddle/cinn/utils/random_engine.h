// Copyright (c) 2023 CINN Authors. All Rights Reserved.
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

#include <glog/logging.h>
#include <stdint.h>

#include <random>
#include "paddle/common/enforce.h"

namespace cinn {
namespace utils {

/**
 * LinearRandomEngine is a random number engine using linear congruence
 * algorithm. The transition function of state is: x(i + 1) = (multiplier * x(i)
 * + increment) mod modulus. Its interface and members are roughly the same as
 * std::linear_congruential_engine, which can be used for std::xxx_distribution.
 * The difference from std::linear_congruential_engine is that the
 * LinearRandomEngine does not own the random seed, but holds the pointer of the
 * random seed and transfers the state for other objects.
 */
class LinearRandomEngine {
 public:
  using StateType = int64_t;
  // the type name "result_type" is needed by std::xxx_distribution
  using result_type = uint32_t;

  // The minimum possible value of random state
  static constexpr result_type min() { return 0; }
  // The maximum possible value of random state
  static constexpr result_type max() { return modulus - 1; }
  // The multiplier
  static constexpr StateType multiplier = 48271;
  // The increment
  static constexpr StateType increment = 0;
  // The modulus
  static constexpr StateType modulus = 2147483647;

  // Construct a linear random engine with a random state pointer
  explicit LinearRandomEngine(StateType* state) : state_(state) {}

  // operator() is needed by std::xxx_distribution
  result_type operator()() { return Next(); }

  // Get a device random state
  static StateType GetDeviceRandomValue() {
    return (std::random_device()()) % modulus;
  }

  // Normalize the random seed to the range of [1, modulus - 1]
  static StateType NormalizeState(StateType state) {
    if (state == -1) {
      state = GetDeviceRandomValue();
    } else {
      state %= modulus;
    }
    if (state == 0) {
      state = 1;
    }
    PADDLE_ENFORCE_GE(state,
                      0,
                      ::common::errors::PreconditionNotMet(
                          "Random seed must be greater than 0"));

    return state;
  }

  // Fork a new state for another Random Generator from current state
  StateType ForkState() { return (Next() * 32767) % 1999999973; }

 private:
  // Move the state to the next and return the new state
  result_type Next() {
    *state_ = (increment + (*state_) * multiplier) % modulus;
    return *state_;
  }

 private:
  StateType* state_;
};

// Fork a new random state for another Random Generator, the original seed will
// be changed to next state.
inline LinearRandomEngine::StateType ForkRandomState(
    LinearRandomEngine::StateType* rand_seed) {
  return LinearRandomEngine(rand_seed).ForkState();
}

// Sample Integers from uniform distribution [min, max)
int SampleUniformInt(int min,
                     int max,
                     LinearRandomEngine::StateType* rand_seed);

// Sample Real Numbers from uniform distribution [min, max)
double SampleUniformDouble(double min,
                           double max,
                           LinearRandomEngine::StateType* rand_seed);

// Sample Integers from distribution of input weights
template <typename T>
int SampleDiscreteFromDistribution(const std::vector<T>& weights,
                                   LinearRandomEngine::StateType* rand_seed) {
  PADDLE_ENFORCE_GT(
      weights.size(),
      0,
      ::common::errors::PreconditionNotMet("Size of target weights is empty."));
  LinearRandomEngine engine(rand_seed);
  std::discrete_distribution<int> dist(weights.begin(), weights.end());
  return dist(engine);
}

}  // namespace utils
}  // namespace cinn
